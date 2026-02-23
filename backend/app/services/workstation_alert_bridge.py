"""
YAQIZ Workstation → Alert Bridge Service
=========================================
Thin adapter that funnels workstation fatigue/attention alerts into the
EXISTING unified Alert table (``alerts``) and broadcasts them on the
EXISTING ``"alerts"`` WebSocket channel.

Design principles:
  • **Additive only** — calls existing ORM models and ``ws_manager``;
    never modifies their signatures.
  • **Single alert source** — all workstation alerts become ``Alert`` rows
    with ``session_id`` pointing to a bridge ``DetectionSession`` so the
    existing Dashboard / Alerts Center queries pick them up automatically.
  • ``WorkstationAlert`` is deliberately **NOT** used.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from sqlalchemy.orm import Session as SASession

from app.core.database import SessionLocal
from app.models.detection import Alert, DetectionSession
from app.models.workstation import WorkstationSession
from app.services.websocket_manager import ws_manager
from app.services.telegram_service import get_telegram_service

logger = logging.getLogger("yaqiz.workstation.bridge")


# ─────────────────────────────────────────────────────────
# Session lifecycle
# ─────────────────────────────────────────────────────────

def create_bridge_session(user_id: Optional[int]) -> Optional[int]:
    """
    Create a bridge ``DetectionSession(session_type='workstation')`` row.

    Returns the bridge session ID so that every ``Alert`` written for this
    monitoring run can be linked back to a session that is visible to the
    existing dashboard queries (``WHERE DetectionSession.user_id = …``).
    """
    db: SASession = SessionLocal()
    try:
        bridge = DetectionSession(
            session_type="workstation",
            status="processing",
            user_id=user_id,
        )
        db.add(bridge)
        db.commit()
        db.refresh(bridge)
        logger.info(
            "Bridge DetectionSession #%s created (user_id=%s)",
            bridge.id, user_id,
        )
        return bridge.id
    except Exception as exc:
        logger.error("Failed to create bridge DetectionSession: %s", exc)
        db.rollback()
        return None
    finally:
        db.close()


def create_workstation_session(user_id: Optional[int]) -> Optional[int]:
    """
    Create a ``WorkstationSession`` row for detailed fatigue tracking.
    """
    db: SASession = SessionLocal()
    try:
        ws = WorkstationSession(status="active", user_id=user_id)
        db.add(ws)
        db.commit()
        db.refresh(ws)
        logger.info("WorkstationSession #%s created (user_id=%s)", ws.id, user_id)
        return ws.id
    except Exception as exc:
        logger.error("Failed to create WorkstationSession: %s", exc)
        db.rollback()
        return None
    finally:
        db.close()


def finalise_sessions(
    *,
    ws_session_id: Optional[int],
    bridge_session_id: Optional[int],
    frame_count: int = 0,
    fatigue_sum: float = 0.0,
    attention_sum: float = 0.0,
    peak_fatigue: float = 0.0,
    max_yawns: int = 0,
    max_blinks: int = 0,
    max_eye_rubs: int = 0,
    alert_count: int = 0,
) -> None:
    """
    Finalise **both** the ``WorkstationSession`` and the bridge
    ``DetectionSession`` when the WebSocket disconnects.
    """
    now = datetime.utcnow()
    avg_fatigue = fatigue_sum / max(frame_count, 1)
    avg_attention = attention_sum / max(frame_count, 1)

    db: SASession = SessionLocal()
    try:
        # ── WorkstationSession ──
        if ws_session_id is not None:
            ws_sess = db.query(WorkstationSession).get(ws_session_id)
            if ws_sess:
                ws_sess.status = "ended"
                ws_sess.total_frames = frame_count
                ws_sess.avg_fatigue_score = avg_fatigue
                ws_sess.avg_attention_score = avg_attention
                ws_sess.peak_fatigue_score = peak_fatigue
                ws_sess.total_yawns = max_yawns
                ws_sess.total_blinks = max_blinks
                ws_sess.total_eye_rubs = max_eye_rubs
                ws_sess.total_alerts = alert_count
                ws_sess.ended_at = now
                logger.info(
                    "WorkstationSession #%s finalised: "
                    "%d frames, peak_fatigue=%.0f, alerts=%d",
                    ws_session_id, frame_count, peak_fatigue, alert_count,
                )

        # ── Bridge DetectionSession ──
        if bridge_session_id is not None:
            bridge = db.query(DetectionSession).get(bridge_session_id)
            if bridge:
                bridge.status = "completed"
                bridge.total_frames = frame_count
                bridge.total_detections = frame_count
                bridge.violations_count = alert_count
                bridge.compliance_rate = max(100.0 - avg_fatigue, 0.0)
                bridge.completed_at = now
                bridge.summary = {
                    "type": "workstation",
                    "avg_fatigue_score": round(avg_fatigue, 1),
                    "avg_attention_score": round(avg_attention, 1),
                    "peak_fatigue_score": round(peak_fatigue, 1),
                    "total_yawns": max_yawns,
                    "total_blinks": max_blinks,
                    "total_eye_rubs": max_eye_rubs,
                    "total_alerts": alert_count,
                }
                logger.info(
                    "Bridge DetectionSession #%s finalised "
                    "(compliance_rate=%.1f%%)",
                    bridge_session_id, bridge.compliance_rate,
                )

        db.commit()
    except Exception as exc:
        logger.error("Failed to finalise workstation sessions: %s", exc)
        db.rollback()
    finally:
        db.close()


# ─────────────────────────────────────────────────────────
# Alert persistence + broadcast
# ─────────────────────────────────────────────────────────

def persist_alerts(
    alerts: List[Dict[str, Any]],
    bridge_session_id: Optional[int],
) -> int:
    """
    Write workstation alerts to the **existing** ``alerts`` table.

    Each ``Alert`` row gets ``session_id = bridge_session_id``, which links
    it to the bridge ``DetectionSession`` so that the Dashboard and Alerts
    Center queries (``WHERE session_id IN (user's sessions)``) automatically
    include workstation alerts — **zero changes to existing queries**.

    Returns the number of alerts successfully persisted.
    """
    if not alerts:
        return 0

    db: SASession = SessionLocal()
    persisted = 0
    try:
        for a in alerts:
            row = Alert(
                alert_type=a["alert_type"],
                severity=a["severity"],
                message=a["message"],
                confidence=a.get("confidence", 0.0),
                session_id=bridge_session_id,
            )
            db.add(row)
            persisted += 1
        db.commit()
        logger.debug(
            "Persisted %d workstation alert(s) (bridge_session=%s)",
            persisted, bridge_session_id,
        )
    except Exception as exc:
        logger.error("Failed to persist workstation alerts: %s", exc)
        db.rollback()
        persisted = 0
    finally:
        db.close()
    return persisted


async def broadcast_alerts(alerts: List[Dict[str, Any]]) -> None:
    """
    Broadcast each alert on the **existing** ``"alerts"`` WebSocket channel
    via ``ws_manager.send_alert()``.

    Clients already subscribed to ``/ws/alerts`` (e.g. the Alerts Center)
    receive workstation alerts in real-time with no frontend changes.
    """
    for alert_data in alerts:
        try:
            await ws_manager.send_alert(alert_data)
            logger.debug(
                "Broadcast workstation alert: %s [%s]",
                alert_data.get("alert_type"), alert_data.get("severity"),
            )
        except Exception as exc:
            logger.warning("Failed to broadcast workstation alert: %s", exc)


async def notify_telegram(alerts: List[Dict[str, Any]]) -> None:
    """
    Send workstation alerts to Telegram via the **existing**
    ``TelegramService``.  Uses ``send_workstation_alert()`` which
    shares the same rate-limiter as PPE alerts — no spam, no
    duplicate notification paths.

    Fails gracefully when Telegram is not configured.
    """
    telegram = get_telegram_service()
    if not telegram.enabled:
        return

    for alert_data in alerts:
        try:
            await telegram.send_workstation_alert(alert_data)
        except Exception as exc:
            logger.debug("Telegram workstation alert skipped: %s", exc)


async def persist_and_broadcast(
    alerts: List[Dict[str, Any]],
    bridge_session_id: Optional[int],
) -> int:
    """
    Convenience wrapper: persist to DB → broadcast on WebSocket → notify
    via Telegram.  Returns the number of alerts persisted.
    """
    count = persist_alerts(alerts, bridge_session_id)
    await broadcast_alerts(alerts)
    await notify_telegram(alerts)
    return count
