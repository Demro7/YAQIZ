"""
YAQIZ Workstation Metrics Transformation Layer
Converts DetectionStats from fatigue_detector_core into YAQIZ-standard metrics and alerts.
"""

import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("yaqiz.workstation.metrics")


# ── Severity thresholds ──────────────────────────────────

FATIGUE_SCORE_THRESHOLDS = {
    "critical": 80,
    "warning": 50,
    "info": 20,
}

ATTENTION_SCORE_THRESHOLDS = {
    "good": 70,
    "moderate": 40,
    "poor": 0,
}


@dataclass
class WorkstationMetrics:
    """Standardized YAQIZ workstation metrics"""
    timestamp: float = 0.0

    # Core scores (0-100)
    attention_score: float = 100.0
    fatigue_score: float = 0.0

    # Face metrics
    face_detected: bool = False
    ear: float = 0.0          # Eye Aspect Ratio
    mar: float = 0.0          # Mouth Aspect Ratio
    blink_rate: int = 0       # blinks per 60s
    yawn_count: int = 0

    # Head pose
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    head_roll: float = 0.0
    head_pose_status: str = "unknown"  # "forward", "left", "right", "up", "down", "unknown"

    # Activity
    typing_cpm: int = 0
    scroll_spm: int = 0
    user_idle: bool = False
    idle_time: float = 0.0
    active_window: str = "Unknown"

    # Presence
    presence: bool = False
    absence_duration: float = 0.0

    # Hands
    eye_rub_count: int = 0
    hand_on_head: bool = False

    # Alerts generated this frame
    alerts: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "attention_score": round(self.attention_score, 1),
            "fatigue_score": round(self.fatigue_score, 1),
            "face_detected": self.face_detected,
            "ear": round(self.ear, 4),
            "mar": round(self.mar, 4),
            "blink_rate": self.blink_rate,
            "yawn_count": self.yawn_count,
            "head_yaw": round(self.head_yaw, 1),
            "head_pitch": round(self.head_pitch, 1),
            "head_roll": round(self.head_roll, 1),
            "head_pose_status": self.head_pose_status,
            "typing_cpm": self.typing_cpm,
            "scroll_spm": self.scroll_spm,
            "user_idle": self.user_idle,
            "idle_time": round(self.idle_time, 1),
            "active_window": self.active_window,
            "presence": self.presence,
            "absence_duration": round(self.absence_duration, 1),
            "eye_rub_count": self.eye_rub_count,
            "hand_on_head": self.hand_on_head,
            "alerts": self.alerts,
        }


class MetricsTransformer:
    """
    Transforms raw DetectionStats from the fatigue detector into
    standardized YAQIZ WorkstationMetrics, including computed scores
    and generated alerts in the unified YAQIZ alert format.
    """

    def __init__(self):
        self._session_start = time.time()
        self._last_face_time: float = time.time()
        self._absence_start: Optional[float] = None
        self._idle_start: Optional[float] = None
        self._alert_cooldowns: Dict[str, float] = {}
        self._alert_cooldown_seconds = 15.0  # rate-limit per alert type

    def transform(self, stats_dict: Dict[str, Any]) -> WorkstationMetrics:
        """
        Convert a DetectionStats.to_dict() output into WorkstationMetrics.
        Also computes attention_score, fatigue_score, and generates unified alerts.
        """
        now = stats_dict.get("timestamp", time.time())
        metrics = WorkstationMetrics(timestamp=now)

        # ── Face & presence ──
        metrics.face_detected = stats_dict.get("face_detected", False)
        metrics.presence = metrics.face_detected

        if metrics.face_detected:
            self._last_face_time = now
            self._absence_start = None
            metrics.absence_duration = 0.0
        else:
            if self._absence_start is None:
                self._absence_start = now
            metrics.absence_duration = now - self._absence_start

        # ── Eye / mouth ──
        metrics.ear = stats_dict.get("ear", 0.0)
        metrics.mar = stats_dict.get("mar", 0.0)
        metrics.blink_rate = stats_dict.get("blink_count", 0)
        metrics.yawn_count = stats_dict.get("yawn_count", 0)

        # ── Head pose ──
        angles = stats_dict.get("angles")
        if angles:
            metrics.head_yaw = angles.get("yaw", 0.0)
            metrics.head_pitch = angles.get("pitch", 0.0)
            metrics.head_roll = angles.get("roll", 0.0)
            metrics.head_pose_status = self._classify_head_pose(
                metrics.head_yaw, metrics.head_pitch
            )
        else:
            metrics.head_pose_status = "unknown"

        # ── Activity ──
        metrics.typing_cpm = stats_dict.get("typing_cpm", 0)
        metrics.scroll_spm = stats_dict.get("scroll_spm", 0)
        metrics.user_idle = stats_dict.get("user_idle", False)
        metrics.active_window = stats_dict.get("active_window", "Unknown")

        if metrics.user_idle:
            if self._idle_start is None:
                self._idle_start = now
            metrics.idle_time = now - self._idle_start
        else:
            self._idle_start = None
            metrics.idle_time = 0.0

        # ── Hands ──
        metrics.eye_rub_count = stats_dict.get("eye_rub_count", 0)
        metrics.hand_on_head = stats_dict.get("hand_on_head", False)

        # ── Compute scores ──
        metrics.fatigue_score = self._compute_fatigue_score(metrics)
        metrics.attention_score = self._compute_attention_score(metrics)

        # ── Generate YAQIZ alerts from raw alerts ──
        raw_alerts = stats_dict.get("alerts", [])
        metrics.alerts = self._transform_alerts(raw_alerts, metrics, now)

        return metrics

    # ──────────────────────────────────────────────────────
    # Score computation
    # ──────────────────────────────────────────────────────

    @staticmethod
    def _compute_fatigue_score(m: WorkstationMetrics) -> float:
        """
        0 = fully alert, 100 = extreme fatigue.
        Weighted composite of multiple fatigue signals.
        """
        score = 0.0

        # Low EAR → eyes closing → fatigue
        if m.face_detected and m.ear > 0:
            if m.ear < 0.18:
                score += 35
            elif m.ear < 0.22:
                score += 15

        # High yawn count
        if m.yawn_count >= 5:
            score += 30
        elif m.yawn_count >= 3:
            score += 20
        elif m.yawn_count >= 1:
            score += 8

        # Eye rubbing
        if m.eye_rub_count >= 3:
            score += 15
        elif m.eye_rub_count >= 1:
            score += 5

        # Hand on head
        if m.hand_on_head:
            score += 10

        # Head dropped
        if m.head_pitch < -15:
            score += 15

        # Absence penalizes
        if not m.face_detected and m.absence_duration > 10:
            score += 10

        return min(score, 100.0)

    @staticmethod
    def _compute_attention_score(m: WorkstationMetrics) -> float:
        """
        100 = fully attentive, 0 = not attentive at all.
        """
        score = 100.0

        # Not present
        if not m.face_detected:
            return 0.0

        # Not looking forward
        if m.head_pose_status not in ("forward", "unknown"):
            score -= 25

        # Eyes mostly closed
        if m.ear < 0.20:
            score -= 20

        # Way too much yawning
        if m.yawn_count >= 3:
            score -= 15

        # User idle
        if m.user_idle:
            score -= 15

        # Eye rubbing
        if m.eye_rub_count >= 2:
            score -= 10

        # Hand on head
        if m.hand_on_head:
            score -= 10

        return max(score, 0.0)

    @staticmethod
    def _classify_head_pose(yaw: float, pitch: float) -> str:
        if abs(yaw) < 15 and abs(pitch) < 15:
            return "forward"
        if yaw > 15:
            return "right"
        if yaw < -15:
            return "left"
        if pitch > 15:
            return "up"
        if pitch < -15:
            return "down"
        return "forward"

    # ──────────────────────────────────────────────────────
    # Alert generation — uses SAME schema as PPE alerts
    # ──────────────────────────────────────────────────────

    def _transform_alerts(
        self,
        raw_alerts: List[Dict],
        metrics: WorkstationMetrics,
        now: float,
    ) -> List[Dict[str, Any]]:
        """
        Convert fatigue-module alerts into YAQIZ unified alert format:
        {alert_type, severity, message, confidence, source}
        """
        yaqiz_alerts: List[Dict[str, Any]] = []

        # Map from fatigue AlertType → YAQIZ alert_type + severity
        ALERT_MAP = {
            "YAWN": ("workstation_yawn", "warning"),
            "EYE_CLOSURE": ("workstation_eye_closure", "critical"),
            "HEAD_POSE": ("workstation_head_pose", "warning"),
            "HEAD_DROP": ("workstation_head_drop", "critical"),
            "EYE_RUB": ("workstation_eye_rub", "warning"),
            "HAND_HEAD": ("workstation_hand_head", "warning"),
            "ABSENCE": ("workstation_absence", "critical"),
            "IDLE": ("workstation_idle", "info"),
        }

        for raw in raw_alerts:
            raw_type = raw.get("type", "")
            mapped = ALERT_MAP.get(raw_type)
            if not mapped:
                continue

            alert_type, severity = mapped

            # Rate-limit per type
            last = self._alert_cooldowns.get(alert_type, 0)
            if now - last < self._alert_cooldown_seconds:
                continue
            self._alert_cooldowns[alert_type] = now

            # Derive confidence from fatigue score
            confidence = min(metrics.fatigue_score / 100.0, 1.0)
            if severity == "critical":
                confidence = max(confidence, 0.8)
            elif severity == "warning":
                confidence = max(confidence, 0.5)

            yaqiz_alerts.append({
                "alert_type": alert_type,
                "severity": severity,
                "message": raw.get("message", f"Workstation alert: {raw_type}"),
                "confidence": round(confidence, 2),
                "source": "workstation",
            })

        # Generate synthetic high-fatigue alert if score crosses threshold
        if metrics.fatigue_score >= 80:
            if self._should_fire("workstation_high_fatigue", now):
                yaqiz_alerts.append({
                    "alert_type": "workstation_high_fatigue",
                    "severity": "critical",
                    "message": f"High fatigue detected — score {metrics.fatigue_score:.0f}/100",
                    "confidence": 0.95,
                    "source": "workstation",
                })

        return yaqiz_alerts

    def _should_fire(self, alert_type: str, now: float) -> bool:
        last = self._alert_cooldowns.get(alert_type, 0)
        if now - last < self._alert_cooldown_seconds:
            return False
        self._alert_cooldowns[alert_type] = now
        return True
