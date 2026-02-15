"""
YAQIZ Telegram Notification Service
Sends violation alerts during live monitoring and video analysis summaries.
"""

import asyncio
import logging
import time
from typing import Dict, Optional
from urllib.parse import quote
import httpx

from app.core.config import settings

logger = logging.getLogger("yaqiz.telegram")

# Rate limiting: don't spam Telegram (max 1 alert every 10 seconds)
_last_alert_time: float = 0
ALERT_COOLDOWN_SECONDS = 10


class TelegramService:
    """Sends notifications to Telegram via Bot API"""

    _instance: Optional['TelegramService'] = None

    @classmethod
    def get_instance(cls) -> 'TelegramService':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.enabled = bool(settings.TELEGRAM_ENABLED and settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID)
        self.bot_token = settings.TELEGRAM_BOT_TOKEN
        self.chat_id = settings.TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

        if self.enabled:
            logger.info("✅ Telegram notifications enabled")
        else:
            logger.info("ℹ️  Telegram notifications disabled (set TELEGRAM_BOT_TOKEN & TELEGRAM_CHAT_ID in .env)")

    async def _send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message via Telegram Bot API"""
        if not self.enabled:
            return False
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    f"{self.base_url}/sendMessage",
                    json={
                        "chat_id": self.chat_id,
                        "text": text,
                        "parse_mode": parse_mode,
                    },
                )
                if resp.status_code == 200:
                    return True
                else:
                    logger.warning(f"Telegram API error {resp.status_code}: {resp.text}")
                    return False
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    async def _send_photo(self, photo_path: str, caption: str = "") -> bool:
        """Send a photo via Telegram Bot API"""
        if not self.enabled:
            return False
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                with open(photo_path, "rb") as f:
                    resp = await client.post(
                        f"{self.base_url}/sendPhoto",
                        data={
                            "chat_id": self.chat_id,
                            "caption": caption,
                            "parse_mode": "HTML",
                        },
                        files={"photo": ("alert.jpg", f, "image/jpeg")},
                    )
                if resp.status_code == 200:
                    return True
                else:
                    logger.warning(f"Telegram photo error {resp.status_code}: {resp.text}")
                    return False
        except Exception as e:
            logger.error(f"Telegram photo send failed: {e}")
            return False

    # ── Live Monitoring Alert ────────────────────────────────

    async def send_violation_alert(self, result: Dict, frame_number: int = 0) -> bool:
        """
        Send a live violation alert to Telegram.
        Rate-limited to avoid spamming.
        """
        global _last_alert_time
        now = time.time()

        # Rate limiting
        if now - _last_alert_time < ALERT_COOLDOWN_SECONDS:
            return False
        _last_alert_time = now

        violations = result.get('violations', [])
        if not violations:
            return False

        # Build violation list
        violation_lines = []
        for v in violations[:5]:  # Max 5 per alert
            name = v['class_name'].replace('NO-', '❌ Missing ')
            conf = v['confidence'] * 100
            violation_lines.append(f"  • {name} ({conf:.0f}%)")

        violations_text = "\n".join(violation_lines)
        workers = result.get('workers_count', 0)
        helmet = result.get('helmet_compliance', 100)
        vest = result.get('vest_compliance', 100)
        mask = result.get('mask_compliance', 100)

        text = (
            f"🚨 <b>YAQIZ — PPE تحذير مخالفة</b>\n\n"
            f"⚠️ تم رصد <b>{len(violations)}</b> مخالفة!\n\n"
            f"{violations_text}\n\n"
            f"👷 عدد العمال: {workers}\n"
            f"🪖 نسبة الخوذة: {helmet:.0f}%\n"
            f"🦺 نسبة السترة: {vest:.0f}%\n"
            f"😷 نسبة الكمامة: {mask:.0f}%\n\n"
            f"🕐 Frame #{frame_number}"
        )

        return await self._send_message(text)

    # ── Video Analysis Summary ───────────────────────────────

    async def send_video_summary(self, summary: Dict, session_id: int = 0) -> bool:
        """Send a video analysis completion summary to Telegram"""

        total_frames = summary.get('total_frames', 0)
        total_detections = summary.get('total_detections', 0)
        violations = summary.get('violations_count', 0)
        compliance = summary.get('compliance_rate', 100)
        helmet = summary.get('helmet_compliance', 100)
        vest = summary.get('vest_compliance', 100)
        workers = summary.get('workers_detected', 0)
        alerts_count = summary.get('alerts_generated', 0)

        # Status emoji
        if compliance >= 90:
            status = "✅ ممتاز"
        elif compliance >= 70:
            status = "⚠️ متوسط"
        else:
            status = "🔴 خطير"

        text = (
            f"📊 <b>YAQIZ — ملخص تحليل الفيديو</b>\n\n"
            f"📋 Session #{session_id}\n"
            f"🎬 إجمالي الفريمات: {total_frames}\n"
            f"🔍 إجمالي الاكتشافات: {total_detections}\n\n"
            f"{'═' * 25}\n"
            f"📈 <b>نتائج الالتزام</b>\n"
            f"{'═' * 25}\n\n"
            f"🎯 نسبة الالتزام: <b>{compliance:.1f}%</b> {status}\n"
            f"❌ المخالفات: {violations}\n"
            f"🪖 الخوذة: {helmet:.1f}%\n"
            f"🦺 السترة: {vest:.1f}%\n"
            f"👷 عدد العمال: {workers}\n"
            f"🔔 التنبيهات: {alerts_count}\n\n"
            f"{'═' * 25}\n"
            f"✅ <i>اكتمل التحليل بنجاح</i>"
        )

        return await self._send_message(text)

    # ── Workstation Alert (Additive — does NOT alter existing methods) ──

    async def send_workstation_alert(self, alert: Dict) -> bool:
        """
        Send a workstation fatigue/attention alert to Telegram.

        Uses the SAME ``_send_message`` transport as PPE alerts.
        Rate-limited via the module-level cooldown so workstation and PPE
        alerts share a single throttle — no duplicate spam.

        Parameters
        ----------
        alert : dict
            Must contain ``alert_type``, ``severity``, ``message``.
            Optional: ``confidence``, ``source``, ``timestamp``.
        """
        global _last_alert_time
        now = time.time()

        if now - _last_alert_time < ALERT_COOLDOWN_SECONDS:
            return False
        _last_alert_time = now

        alert_type = alert.get("alert_type", "unknown")
        severity = alert.get("severity", "info")
        message = alert.get("message", "")
        confidence = alert.get("confidence", 0)
        source = alert.get("source", "workstation")

        # Severity emoji
        sev_emoji = {"critical": "🔴", "warning": "⚠️", "info": "ℹ️"}.get(severity, "🔔")

        # Human-readable alert type
        readable_type = alert_type.replace("workstation_", "").replace("_", " ").title()

        text = (
            f"{sev_emoji} <b>YAQIZ — تنبيه محطة العمل</b>\n\n"
            f"📌 النوع: <b>{readable_type}</b>\n"
            f"⚡ الشدة: <b>{severity.upper()}</b>\n"
            f"💬 الرسالة: {message}\n"
            f"🎯 الثقة: {confidence * 100:.0f}%\n"
            f"📍 المصدر: {source}\n\n"
            f"🕐 {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        return await self._send_message(text)


# Singleton accessor
def get_telegram_service() -> TelegramService:
    return TelegramService.get_instance()
