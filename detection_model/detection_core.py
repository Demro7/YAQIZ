"""
Fatigue Detection Core - Headless/API Version
Returns DetectionStats per frame without rendering UI.
Use this for web APIs, background services, or integration into other projects.
"""

import cv2
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import mediapipe as mp
import numpy as np
import math
import os
import urllib.request

from .activity_tracker import SystemActivityTracker


# ============================================================================
# MEDIAPIPE COMPATIBILITY LAYER
# mediapipe >= 0.10.30 removed mp.solutions; use mp.tasks API instead.
# ============================================================================

_USE_TASKS_API = not hasattr(mp, 'solutions')

if _USE_TASKS_API:
    _MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.models')
    _FACE_MODEL_PATH = os.path.join(_MODELS_DIR, 'face_landmarker.task')
    _HAND_MODEL_PATH = os.path.join(_MODELS_DIR, 'hand_landmarker.task')
    _FACE_MODEL_URL = (
        'https://storage.googleapis.com/mediapipe-models/'
        'face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
    )
    _HAND_MODEL_URL = (
        'https://storage.googleapis.com/mediapipe-models/'
        'hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'
    )

    def _ensure_models_downloaded():
        """Download MediaPipe .task model files if not already cached."""
        os.makedirs(_MODELS_DIR, exist_ok=True)
        for url, path in [
            (_FACE_MODEL_URL, _FACE_MODEL_PATH),
            (_HAND_MODEL_URL, _HAND_MODEL_PATH),
        ]:
            if not os.path.exists(path):
                print(f'[MediaPipe] Downloading {os.path.basename(path)} ...')
                urllib.request.urlretrieve(url, path)
                print(f'[MediaPipe] Saved {os.path.basename(path)}')

    class _LandmarkListWrapper:
        """Wraps new-API landmark list to provide old-API .landmark[idx] access."""
        __slots__ = ('landmark',)
        def __init__(self, landmarks):
            self.landmark = landmarks

    class _FaceResultWrapper:
        """Wraps FaceLandmarkerResult -> old-style .multi_face_landmarks."""
        __slots__ = ('multi_face_landmarks',)
        def __init__(self, result):
            if result.face_landmarks:
                self.multi_face_landmarks = [
                    _LandmarkListWrapper(fl) for fl in result.face_landmarks
                ]
            else:
                self.multi_face_landmarks = None

    class _HandResultWrapper:
        """Wraps HandLandmarkerResult -> old-style .multi_hand_landmarks."""
        __slots__ = ('multi_hand_landmarks',)
        def __init__(self, result):
            if result.hand_landmarks:
                self.multi_hand_landmarks = [
                    _LandmarkListWrapper(hl) for hl in result.hand_landmarks
                ]
            else:
                self.multi_hand_landmarks = None


class AlertType(str, Enum):
    """Alert types matching the existing system"""
    YAWN = "YAWN"
    EYE_CLOSURE = "EYE_CLOSURE"
    HEAD_POSE = "HEAD_POSE"
    HEAD_DROP = "HEAD_DROP"
    EYE_RUB = "EYE_RUB"
    HAND_HEAD = "HAND_HEAD"
    ABSENCE = "ABSENCE"
    IDLE = "IDLE"


@dataclass
class Alert:
    """Represents a single alert event"""
    type: AlertType
    message: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class DetectionStats:
    """Per-frame detection statistics"""
    timestamp: float
    face_detected: bool
    ear: float = 0.0
    mar: float = 0.0
    blink_count: int = 0
    yawn_count: int = 0
    angles: Optional[Tuple[float, float, float]] = None
    eye_rub_count: int = 0
    hand_on_head: bool = False
    typing_cpm: int = 0
    scroll_spm: int = 0
    user_idle: bool = False
    active_window: str = "Unknown"
    alerts: List[Alert] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "timestamp": self.timestamp,
            "face_detected": self.face_detected,
            "ear": round(self.ear, 4),
            "mar": round(self.mar, 4),
            "blink_count": self.blink_count,
            "yawn_count": self.yawn_count,
            "angles": {
                "yaw": round(self.angles[0], 2),
                "pitch": round(self.angles[1], 2),
                "roll": round(self.angles[2], 2)
            } if self.angles else None,
            "eye_rub_count": self.eye_rub_count,
            "hand_on_head": self.hand_on_head,
            "typing_cpm": self.typing_cpm,
            "scroll_spm": self.scroll_spm,
            "user_idle": self.user_idle,
            "active_window": self.active_window,
            "alerts": [{"type": a.type.value, "message": a.message, "timestamp": a.timestamp} for a in self.alerts]
        }


# ============================================================================
# LANDMARK DEFINITIONS
# ============================================================================

class FaceLandmarks:
    """Centralized landmark indices for MediaPipe FaceMesh (468 landmarks)"""
    UPPER_LIP = 13
    LOWER_LIP = 14
    LEFT_MOUTH = 308
    RIGHT_MOUTH = 78
    
    LEFT_EYE_INDICES = {
        'top': 159, 'bottom': 145,
        'left': 33, 'right': 133,
        'v1': 158, 'v2': 153
    }
    
    RIGHT_EYE_INDICES = {
        'top': 386, 'bottom': 374,
        'left': 362, 'right': 263,
        'v1': 385, 'v2': 380
    }
    
    LEFT_EYE_REGION = [33, 133, 159, 145, 158, 153, 246, 161]
    RIGHT_EYE_REGION = [362, 263, 386, 374, 385, 380, 466, 388]
    HEAD_CENTER = [1, 4, 152, 200]
    
    HEAD_MODEL_3D = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ], dtype=np.float64)
    
    HEAD_MODEL_2D_INDICES = [1, 152, 33, 263, 61, 291]


# ============================================================================
# UTILITY CLASSES
# ============================================================================

class GeometryUtils:
    @staticmethod
    def euclidean_distance_2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    
    @staticmethod
    def euclidean_distance_landmarks(p1, p2) -> float:
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


class EARCalculator:
    @staticmethod
    def calculate(eye_landmarks: Dict[str, int], face_landmarks) -> float:
        top = face_landmarks.landmark[eye_landmarks['top']]
        bottom = face_landmarks.landmark[eye_landmarks['bottom']]
        left = face_landmarks.landmark[eye_landmarks['left']]
        right = face_landmarks.landmark[eye_landmarks['right']]
        v1 = face_landmarks.landmark[eye_landmarks['v1']]
        v2 = face_landmarks.landmark[eye_landmarks['v2']]
        
        vertical_1 = GeometryUtils.euclidean_distance_landmarks(top, bottom)
        vertical_2 = GeometryUtils.euclidean_distance_landmarks(v1, v2)
        horizontal = GeometryUtils.euclidean_distance_landmarks(left, right)
        
        return (vertical_1 + vertical_2) / (2.0 * horizontal) if horizontal > 0 else 0.0


class MARCalculator:
    @staticmethod
    def calculate(face_landmarks) -> float:
        upper = face_landmarks.landmark[FaceLandmarks.UPPER_LIP]
        lower = face_landmarks.landmark[FaceLandmarks.LOWER_LIP]
        left = face_landmarks.landmark[FaceLandmarks.LEFT_MOUTH]
        right = face_landmarks.landmark[FaceLandmarks.RIGHT_MOUTH]
        
        vertical = GeometryUtils.euclidean_distance_landmarks(upper, lower)
        horizontal = GeometryUtils.euclidean_distance_landmarks(left, right)
        
        return vertical / horizontal if horizontal > 0 else 0.0


class HeadPoseEstimator:
    def __init__(self):
        self.model_points = FaceLandmarks.HEAD_MODEL_3D
        self.dist_coeffs = np.zeros((4, 1))
    
    def estimate(self, face_landmarks, frame_width: int, frame_height: int) -> Optional[Tuple[float, float, float]]:
        image_points = np.array([
            (face_landmarks.landmark[idx].x * frame_width,
             face_landmarks.landmark[idx].y * frame_height)
            for idx in FaceLandmarks.HEAD_MODEL_2D_INDICES
        ], dtype=np.float64)
        
        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        success, rotation_vector, _ = cv2.solvePnP(
            self.model_points, image_points, camera_matrix,
            self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None
        
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        
        if sy >= 1e-6:
            pitch = math.atan2(-rotation_matrix[2, 0], sy)
            yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            roll = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        else:
            pitch = math.atan2(-rotation_matrix[2, 0], sy)
            yaw = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            roll = 0
        
        return (math.degrees(yaw), math.degrees(pitch), math.degrees(roll))


# ============================================================================
# STATE TRACKERS
# ============================================================================

class BlinkCounter:
    def __init__(self, ear_threshold: float = 0.21, consecutive_frames: int = 2, window_seconds: float = 60.0):
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.window_seconds = window_seconds
        self.timestamps = deque()
        self.frame_counter = 0
        self.is_blinking = False
    
    def update(self, ear: float, current_time: float) -> int:
        while self.timestamps and (current_time - self.timestamps[0]) > self.window_seconds:
            self.timestamps.popleft()
        
        if ear < self.ear_threshold:
            self.frame_counter += 1
        else:
            if self.frame_counter >= self.consecutive_frames and not self.is_blinking:
                self.timestamps.append(current_time)
                self.is_blinking = True
            self.frame_counter = 0
            self.is_blinking = False
        
        return len(self.timestamps)


class TimedAlertTracker:
    def __init__(self, threshold_time: float):
        self.threshold_time = threshold_time
        self.start_time: Optional[float] = None
        self.alert_active = False
    
    def update(self, condition_active: bool, current_time: float) -> Tuple[bool, float]:
        if condition_active:
            if self.start_time is None:
                self.start_time = current_time
            elapsed = current_time - self.start_time
            return elapsed >= self.threshold_time, elapsed
        else:
            self.start_time = None
            self.alert_active = False
            return False, 0.0
    
    def reset(self):
        self.start_time = None
        self.alert_active = False


class EyeRubCounter:
    def __init__(self, distance_threshold: float = 0.05, limit: int = 3, debounce_time: float = 0.5):
        self.distance_threshold = distance_threshold
        self.limit = limit
        self.debounce_time = debounce_time
        self.count = 0
        self.is_rubbing = False
        self.last_rub_time = 0.0
        self.alert_triggered = False
    
    def update(self, is_rubbing: bool, current_time: float) -> int:
        if is_rubbing:
            if not self.is_rubbing:
                if current_time - self.last_rub_time > self.debounce_time:
                    self.count += 1
                    self.last_rub_time = current_time
                self.is_rubbing = True
        else:
            self.is_rubbing = False
        return self.count
    
    def should_alert(self) -> bool:
        return self.count >= self.limit


# ============================================================================
# FATIGUE DETECTOR CORE - Headless API version
# ============================================================================

class FatigueDetectorCore:
    """
    Core fatigue detection system for API/integration use.
    Processes frames and returns DetectionStats without rendering UI.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._initialize_config()
        self._initialize_mediapipe()
        self._initialize_trackers()
        
        # State
        self.yawn_count = 0
        self.yawn_active = False
        self.smooth_pitch = 0.0
        self.pose_smoothing_factor = 0.2
        
        # Alert buffer (ring buffer for recent alerts)
        self.alerts_buffer: deque = deque(maxlen=500)
        
        # Activity tracker
        self.activity_tracker: Optional[SystemActivityTracker] = None
        
    def _initialize_config(self):
        """Extract config values with defaults"""
        detection = self.config.get("detection", {})
        timing = self.config.get("timing", {})
        general = self.config.get("general", {})
        
        self.YAWN_MAR_THRESHOLD = detection.get("YAWN_MAR_THRESHOLD", 0.6)
        self.YAWN_FATIGUE_LIMIT = detection.get("YAWN_FATIGUE_LIMIT", 3)
        self.EYE_CLOSURE_THRESHOLD = detection.get("EYE_CLOSURE_THRESHOLD", 0.2)
        self.BLINK_EAR_THRESHOLD = detection.get("BLINK_EAR_THRESHOLD", 0.21)
        self.BLINK_CONSECUTIVE_FRAMES = detection.get("BLINK_CONSECUTIVE_FRAMES", 2)
        self.HEAD_YAW_THRESHOLD = detection.get("HEAD_YAW_THRESHOLD", 20.0)
        self.HEAD_PITCH_THRESHOLD = detection.get("HEAD_PITCH_THRESHOLD", 20.0)
        self.HEAD_DROP_PITCH_THRESHOLD = detection.get("HEAD_DROP_PITCH_THRESHOLD", 15.0)
        self.EYE_RUB_DISTANCE_THRESHOLD = detection.get("EYE_RUB_DISTANCE_THRESHOLD", 0.05)
        self.EYE_RUB_LIMIT = detection.get("EYE_RUB_LIMIT", 3)
        self.HAND_HEAD_DISTANCE_THRESHOLD = detection.get("HAND_HEAD_DISTANCE_THRESHOLD", 0.15)
        
        self.EYE_CLOSURE_WARNING_TIME = timing.get("EYE_CLOSURE_WARNING_TIME", 10.0)
        self.HEAD_POSE_WARNING_TIME = timing.get("HEAD_POSE_WARNING_TIME", 20.0)
        self.HEAD_DROP_WARNING_TIME = timing.get("HEAD_DROP_WARNING_TIME", 5.0)
        self.HAND_HEAD_WARNING_TIME = timing.get("HAND_HEAD_WARNING_TIME", 10.0)
        self.ABSENCE_WARNING_TIME = timing.get("ABSENCE_WARNING_TIME", 20.0)
        self.IDLE_WARNING_TIME = timing.get("IDLE_WARNING_TIME", 20.0)
        self.BLINK_WINDOW_SECONDS = timing.get("BLINK_WINDOW_SECONDS", 60.0)
        self.EYE_RUB_DEBOUNCE_TIME = timing.get("EYE_RUB_DEBOUNCE_TIME", 0.5)
        
        self.ALARM_ENABLED = general.get("ALARM_ENABLED", True)
        self.ML_ENABLED = general.get("ML_ENABLED", True)
        self.ML_LOG_INTERVAL = general.get("ML_LOG_INTERVAL", 60.0)
    
    def _initialize_mediapipe(self):
        if _USE_TASKS_API:
            _ensure_models_downloaded()
            face_opts = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=_FACE_MODEL_PATH
                ),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=1,
            )
            self._face_landmarker = (
                mp.tasks.vision.FaceLandmarker.create_from_options(face_opts)
            )
            hand_opts = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=_HAND_MODEL_PATH
                ),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_hands=2,
            )
            self._hand_landmarker = (
                mp.tasks.vision.HandLandmarker.create_from_options(hand_opts)
            )
        else:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        
        self.head_pose_estimator = HeadPoseEstimator()
    
    def _initialize_trackers(self):
        self.blink_counter = BlinkCounter(
            ear_threshold=self.BLINK_EAR_THRESHOLD,
            consecutive_frames=self.BLINK_CONSECUTIVE_FRAMES,
            window_seconds=self.BLINK_WINDOW_SECONDS
        )
        self.eye_closure_tracker = TimedAlertTracker(self.EYE_CLOSURE_WARNING_TIME)
        self.head_pose_tracker = TimedAlertTracker(self.HEAD_POSE_WARNING_TIME)
        self.head_drop_tracker = TimedAlertTracker(self.HEAD_DROP_WARNING_TIME)
        self.hand_head_tracker = TimedAlertTracker(self.HAND_HEAD_WARNING_TIME)
        self.absence_tracker = TimedAlertTracker(self.ABSENCE_WARNING_TIME)
        self.idle_tracker = TimedAlertTracker(self.IDLE_WARNING_TIME)
        self.eye_rub_counter = EyeRubCounter(
            distance_threshold=self.EYE_RUB_DISTANCE_THRESHOLD,
            limit=self.EYE_RUB_LIMIT,
            debounce_time=self.EYE_RUB_DEBOUNCE_TIME
        )
    
    def start_activity_tracker(self):
        if self.activity_tracker is None:
            self.activity_tracker = SystemActivityTracker()
    
    def stop_activity_tracker(self):
        if self.activity_tracker:
            try:
                self.activity_tracker.cleanup()
            except:
                pass
            self.activity_tracker = None
    
    def _add_alert(self, alert_type: AlertType, message: str) -> Alert:
        alert = Alert(type=alert_type, message=message)
        self.alerts_buffer.append(alert)
        return alert
    
    def get_recent_alerts(self, limit: int = 200) -> List[Dict[str, Any]]:
        alerts = list(self.alerts_buffer)[-limit:]
        return [{"type": a.type.value, "message": a.message, "timestamp": a.timestamp} for a in alerts]
    
    def process_frame(self, frame: np.ndarray) -> DetectionStats:
        """
        Process a single frame and return detection statistics.
        This is the main per-frame function.
        """
        current_time = time.time()
        frame_height, frame_width = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if _USE_TASKS_API:
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(frame_rgb),
            )
            face_results = _FaceResultWrapper(
                self._face_landmarker.detect(mp_image)
            )
            hand_results = _HandResultWrapper(
                self._hand_landmarker.detect(mp_image)
            )
        else:
            face_results = self.face_mesh.process(frame_rgb)
            hand_results = self.hands.process(frame_rgb)
        
        stats = DetectionStats(
            timestamp=current_time,
            face_detected=face_results.multi_face_landmarks is not None
        )
        
        if self.activity_tracker:
            activity_stats = self.activity_tracker.get_activity_stats()
            stats.typing_cpm = activity_stats.get("typing_cpm", 0)
            stats.scroll_spm = activity_stats.get("scroll_spm", 0)
            stats.user_idle = activity_stats.get("is_idle", False)
            stats.active_window = activity_stats.get("active_window", "Unknown")
        
        frame_alerts: List[Alert] = []
        
        # Absence check
        should_alert_absence, elapsed_absence = self.absence_tracker.update(
            not stats.face_detected, current_time
        )
        if should_alert_absence and not self.absence_tracker.alert_active:
            alert = self._add_alert(AlertType.ABSENCE, f"User absent for {elapsed_absence:.1f}s")
            frame_alerts.append(alert)
            self.absence_tracker.alert_active = True
        
        # Idle check
        should_alert_idle, elapsed_idle = self.idle_tracker.update(stats.user_idle, current_time)
        if should_alert_idle and not self.idle_tracker.alert_active:
            alert = self._add_alert(AlertType.IDLE, f"User idle for {elapsed_idle:.1f}s")
            frame_alerts.append(alert)
            self.idle_tracker.alert_active = True
        
        if stats.face_detected:
            face_landmarks = face_results.multi_face_landmarks[0]
            
            # EAR
            left_ear = EARCalculator.calculate(FaceLandmarks.LEFT_EYE_INDICES, face_landmarks)
            right_ear = EARCalculator.calculate(FaceLandmarks.RIGHT_EYE_INDICES, face_landmarks)
            stats.ear = (left_ear + right_ear) / 2.0
            
            # MAR
            stats.mar = MARCalculator.calculate(face_landmarks)
            
            # Blinks
            stats.blink_count = self.blink_counter.update(stats.ear, current_time)
            
            # Yawn
            is_yawning = stats.mar > self.YAWN_MAR_THRESHOLD
            if is_yawning and not self.yawn_active:
                self.yawn_count += 1
                self.yawn_active = True
                if self.yawn_count >= self.YAWN_FATIGUE_LIMIT:
                    alert = self._add_alert(AlertType.YAWN, f"Yawn count: {self.yawn_count} - Fatigue detected!")
                    frame_alerts.append(alert)
            elif not is_yawning:
                self.yawn_active = False
            stats.yawn_count = self.yawn_count
            
            # Eye closure
            is_closed = stats.ear < self.EYE_CLOSURE_THRESHOLD
            should_alert_closure, elapsed_closure = self.eye_closure_tracker.update(is_closed, current_time)
            if should_alert_closure and not self.eye_closure_tracker.alert_active:
                alert = self._add_alert(AlertType.EYE_CLOSURE, f"Eyes closed for {elapsed_closure:.1f}s")
                frame_alerts.append(alert)
                self.eye_closure_tracker.alert_active = True
            
            # Head pose
            angles = self.head_pose_estimator.estimate(face_landmarks, frame_width, frame_height)
            if angles:
                yaw, raw_pitch, roll = angles
                self.smooth_pitch = (raw_pitch * self.pose_smoothing_factor) + \
                                   (self.smooth_pitch * (1.0 - self.pose_smoothing_factor))
                stats.angles = (yaw, self.smooth_pitch, roll)
                
                not_looking = abs(yaw) > self.HEAD_YAW_THRESHOLD or abs(raw_pitch) > self.HEAD_PITCH_THRESHOLD
                should_alert_pose, elapsed_pose = self.head_pose_tracker.update(not_looking, current_time)
                if should_alert_pose and not self.head_pose_tracker.alert_active:
                    alert = self._add_alert(AlertType.HEAD_POSE, f"Not looking forward for {elapsed_pose:.1f}s")
                    frame_alerts.append(alert)
                    self.head_pose_tracker.alert_active = True
                
                head_dropped = self.smooth_pitch < -self.HEAD_DROP_PITCH_THRESHOLD
                should_alert_drop, elapsed_drop = self.head_drop_tracker.update(head_dropped, current_time)
                if should_alert_drop and not self.head_drop_tracker.alert_active:
                    alert = self._add_alert(AlertType.HEAD_DROP, f"Head dropped for {elapsed_drop:.1f}s")
                    frame_alerts.append(alert)
                    self.head_drop_tracker.alert_active = True
            
            # Hands
            is_rubbing = False
            hand_on_head = False
            
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    if self._detect_hand_eye_contact(hand_landmarks, face_landmarks):
                        is_rubbing = True
                    if self._detect_hand_head_proximity(hand_landmarks, face_landmarks):
                        hand_on_head = True
            
            stats.hand_on_head = hand_on_head
            
            stats.eye_rub_count = self.eye_rub_counter.update(is_rubbing, current_time)
            if self.eye_rub_counter.should_alert() and not self.eye_rub_counter.alert_triggered:
                alert = self._add_alert(AlertType.EYE_RUB, f"Eye rubbing count: {stats.eye_rub_count}")
                frame_alerts.append(alert)
                self.eye_rub_counter.alert_triggered = True
            
            should_alert_hand, elapsed_hand = self.hand_head_tracker.update(hand_on_head, current_time)
            if should_alert_hand and not self.hand_head_tracker.alert_active:
                alert = self._add_alert(AlertType.HAND_HEAD, f"Hand on head for {elapsed_hand:.1f}s")
                frame_alerts.append(alert)
                self.hand_head_tracker.alert_active = True
        else:
            stats.eye_rub_count = self.eye_rub_counter.count
        
        stats.alerts = frame_alerts
        return stats
    
    def _detect_hand_eye_contact(self, hand_landmarks, face_landmarks) -> bool:
        hand_points = [hand_landmarks.landmark[idx] for idx in [8, 12, 16, 20, 4, 0]]
        
        for hand_point in hand_points:
            hand_pos = (hand_point.x, hand_point.y)
            for eye_idx in FaceLandmarks.LEFT_EYE_REGION + FaceLandmarks.RIGHT_EYE_REGION:
                eye_pos = (face_landmarks.landmark[eye_idx].x, face_landmarks.landmark[eye_idx].y)
                if GeometryUtils.euclidean_distance_2d(hand_pos, eye_pos) < self.EYE_RUB_DISTANCE_THRESHOLD:
                    return True
        return False
    
    def _detect_hand_head_proximity(self, hand_landmarks, face_landmarks) -> bool:
        hand_points = [hand_landmarks.landmark[idx] for idx in [0, 9, 8]]
        
        head_positions = [
            (face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y)
            for idx in FaceLandmarks.HEAD_CENTER
        ]
        head_center = (
            sum(p[0] for p in head_positions) / len(head_positions),
            sum(p[1] for p in head_positions) / len(head_positions)
        )
        
        for hand_point in hand_points:
            hand_pos = (hand_point.x, hand_point.y)
            if GeometryUtils.euclidean_distance_2d(hand_pos, head_center) < self.HAND_HEAD_DISTANCE_THRESHOLD:
                return True
        return False
    
    def cleanup(self):
        """Release resources"""
        self.stop_activity_tracker()
        if _USE_TASKS_API:
            for attr in ('_face_landmarker', '_hand_landmarker'):
                obj = getattr(self, attr, None)
                if obj is not None:
                    try:
                        obj.close()
                    except Exception:
                        pass
        else:
            try:
                if hasattr(self, 'face_mesh') and hasattr(self.face_mesh, 'close'):
                    self.face_mesh.close()
            except Exception:
                pass
            try:
                if hasattr(self, 'hands') and hasattr(self.hands, 'close'):
                    self.hands.close()
            except Exception:
                pass
