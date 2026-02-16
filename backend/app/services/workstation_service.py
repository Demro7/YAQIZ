"""
YAQIZ Workstation Service
Wraps the existing detection_core.FatigueDetectorCore into a clean async service
for the YAQIZ platform.  Initialises the detector ONCE and reuses it across frames.

This service does NOT modify detection_core — it simply consumes its API.
"""

import asyncio
import logging
import time
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger("yaqiz.workstation.service")

# Make detection_model importable
# In local dev:  backend/app/services/ → 4 parents up → repo root
# In Docker:     /app/app/services/    → 3 parents up → /app (where detection_model is mounted)
_SERVICE_DIR = Path(__file__).resolve().parent
for _depth in (3, 4):  # Try Docker path first, then local dev path
    _candidate = _SERVICE_DIR
    for _ in range(_depth):
        _candidate = _candidate.parent
    if (_candidate / "detection_model").is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
        break


class WorkstationService:
    """
    Singleton service that wraps FatigueDetectorCore for YAQIZ integration.
    
    - Initialises MediaPipe models exactly once.
    - Provides an async process_frame() that runs the CPU-bound detection
      in a thread-pool executor so it doesn't block the event loop.
    - Returns raw DetectionStats dicts ready for the metrics transformer.
    """

    _instance: Optional["WorkstationService"] = None
    _detector = None   # FatigueDetectorCore instance
    _ready: bool = False

    @classmethod
    def get_instance(cls) -> "WorkstationService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if WorkstationService._detector is not None:
            # Already initialised via another instance path
            self._ready = True
            return
        self._initialise_detector()

    # ──────────────────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────────────────

    def _initialise_detector(self):
        """
        Lazy-load FatigueDetectorCore so that import-time errors don't
        crash the rest of the backend.
        """
        try:
            from detection_model.detection_core import FatigueDetectorCore
            WorkstationService._detector = FatigueDetectorCore()
            WorkstationService._ready = True
            logger.info("Workstation detector initialised (MediaPipe FaceMesh + Hands)")
        except Exception as e:
            logger.error(f"Failed to initialise workstation detector: {e}")
            WorkstationService._ready = False

    @property
    def is_ready(self) -> bool:
        return WorkstationService._ready and WorkstationService._detector is not None

    # ──────────────────────────────────────────────────────
    # Frame processing
    # ──────────────────────────────────────────────────────

    def process_frame_sync(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Synchronous wrapper — runs FatigueDetectorCore.process_frame().
        Returns the DetectionStats as a dict.
        """
        if not self.is_ready:
            return self._empty_stats()

        try:
            stats = WorkstationService._detector.process_frame(frame)
            return stats.to_dict()
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return self._empty_stats()

    async def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Async wrapper — offloads CPU-heavy MediaPipe work to a thread.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process_frame_sync, frame)

    # ──────────────────────────────────────────────────────
    # Activity tracker control
    # ──────────────────────────────────────────────────────

    def start_activity_tracker(self):
        """Start system activity tracking (keyboard/mouse monitoring)."""
        if self.is_ready:
            try:
                WorkstationService._detector.start_activity_tracker()
                logger.info("Activity tracker started")
            except Exception as e:
                logger.warning(f"Activity tracker start failed: {e}")

    def stop_activity_tracker(self):
        """Stop system activity tracking."""
        if self.is_ready:
            try:
                WorkstationService._detector.stop_activity_tracker()
                logger.info("Activity tracker stopped")
            except Exception as e:
                logger.warning(f"Activity tracker stop failed: {e}")

    # ──────────────────────────────────────────────────────
    # Recent alerts
    # ──────────────────────────────────────────────────────

    def get_recent_alerts(self, limit: int = 100) -> list:
        if not self.is_ready:
            return []
        return WorkstationService._detector.get_recent_alerts(limit)

    # ──────────────────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────────────────

    def cleanup(self):
        if WorkstationService._detector is not None:
            try:
                WorkstationService._detector.cleanup()
            except Exception:
                pass
            WorkstationService._detector = None
            WorkstationService._ready = False
            logger.info("Workstation detector cleaned up")

    # ──────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────

    @staticmethod
    def _empty_stats() -> Dict[str, Any]:
        """Return a safe empty stats dict when the detector isn't ready."""
        return {
            "timestamp": time.time(),
            "face_detected": False,
            "ear": 0.0,
            "mar": 0.0,
            "blink_count": 0,
            "yawn_count": 0,
            "angles": None,
            "eye_rub_count": 0,
            "hand_on_head": False,
            "typing_cpm": 0,
            "scroll_spm": 0,
            "user_idle": False,
            "active_window": "Unknown",
            "alerts": [],
        }


# ── Singleton accessor ───────────────────────────────────

def get_workstation_service() -> WorkstationService:
    return WorkstationService.get_instance()
