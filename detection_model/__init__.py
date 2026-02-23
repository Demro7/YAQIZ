"""
ErgoAI Detection Model Package
Fatigue detection system using MediaPipe + OpenCV.

Usage (Standalone with OpenCV window):
    from detection_model.fatigue_detector import FatigueDetectionSystem, FatigueConfig
    system = FatigueDetectionSystem(FatigueConfig())
    system.run()

Usage (Headless API - returns stats per frame):
    from detection_model.detection_core import FatigueDetectorCore
    import cv2
    
    detector = FatigueDetectorCore()
    detector.start_activity_tracker()
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    stats = detector.process_frame(frame)
    print(stats.to_dict())
    detector.cleanup()
"""

from .detection_core import FatigueDetectorCore, DetectionStats, AlertType, Alert
from .fatigue_detector import FatigueDetectionSystem, FatigueConfig
from .activity_tracker import SystemActivityTracker

__all__ = [
    "FatigueDetectorCore",
    "DetectionStats",
    "AlertType",
    "Alert",
    "FatigueDetectionSystem",
    "FatigueConfig",
    "SystemActivityTracker",
]
