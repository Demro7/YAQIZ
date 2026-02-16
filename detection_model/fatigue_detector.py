"""
Fatigue Detection System - Production Grade
Real-time computer vision system for driver/operator fatigue monitoring

Architecture: Clean, modular design with separation of concerns
Patterns: Strategy (detection algorithms), Singleton (resources), Observer (alerts)
Performance: Optimized for real-time processing with O(1) lookups and O(n) minimal computations
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math

# winsound is Windows-only; provide a no-op fallback on other platforms
try:
    import winsound
    _HAS_WINSOUND = True
except ImportError:
    _HAS_WINSOUND = False
from collections import deque
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List, Dict
from abc import ABC, abstractmethod

from .activity_tracker import SystemActivityTracker

# ============================================================================
# CONFIGURATION - Centralized configuration for easy tuning
# ============================================================================

@dataclass(frozen=True)
class FatigueConfig:
    """Immutable configuration for fatigue detection thresholds and parameters"""
    
    # Yawning Detection
    YAWN_MAR_THRESHOLD: float = 0.6
    YAWN_FATIGUE_LIMIT: int = 3
    
    # Eye Closure Detection (EAR = Eye Aspect Ratio)
    EYE_CLOSURE_THRESHOLD: float = 0.2
    EYE_CLOSURE_WARNING_TIME: float = 10.0
    
    # Blink Detection
    BLINK_EAR_THRESHOLD: float = 0.21
    BLINK_CONSECUTIVE_FRAMES: int = 2
    BLINK_WINDOW_SECONDS: float = 60.0
    
    # Head Pose Detection
    HEAD_YAW_THRESHOLD: float = 20.0
    HEAD_PITCH_THRESHOLD: float = 20.0
    HEAD_POSE_WARNING_TIME: float = 20.0
    
    # Head Drop Detection (sleeping/nodding off)
    HEAD_DROP_PITCH_THRESHOLD: float = 15.0
    HEAD_DROP_WARNING_TIME: float = 5.0
    
    # Eye Rubbing Detection
    EYE_RUB_DISTANCE_THRESHOLD: float = 0.05
    EYE_RUB_LIMIT: int = 3
    EYE_RUB_DEBOUNCE_TIME: float = 0.5
    
    # Hand-to-Head Detection (fatigue posture)
    HAND_HEAD_DISTANCE_THRESHOLD: float = 0.15
    HAND_HEAD_WARNING_TIME: float = 10.0
    
    # Absence & Inactivity Detection
    ABSENCE_WARNING_TIME: float = 20.0
    IDLE_WARNING_TIME: float = 20.0
    
    # Alarm Frequencies (Hz)
    ALARM_HEAD_POSE: int = 1000
    ALARM_EYE_CLOSURE: int = 1500
    ALARM_EYE_RUB: int = 2000
    ALARM_HAND_HEAD: int = 1800
    ALARM_HEAD_DROP: int = 2500
    ALARM_ABSENCE: int = 3000
    ALARM_IDLE: int = 1000
    
    # Log file
    LOG_FILE: str = "fatigue_events_log.txt"


# ============================================================================
# ENUMS - Type-safe event definitions
# ============================================================================

class FatigueEvent(Enum):
    """Enumeration of fatigue detection events"""
    YAWN = "YAWN"
    EYE_CLOSURE = "EYE_CLOSURE"
    BLINK = "BLINK"
    BLINK_REPORT = "BLINK_REPORT"
    HEAD_POSE_ALERT = "HEAD_POSE_ALERT"
    HEAD_DROP = "HEAD_DROP"
    EYE_RUB = "EYE_RUB"
    HAND_ON_HEAD = "HAND_ON_HEAD"
    FATIGUE_DETECTED = "FATIGUE_DETECTED"
    USER_ABSENT = "USER_ABSENT"
    USER_IDLE = "USER_IDLE"


# ============================================================================
# LANDMARK DEFINITIONS - MediaPipe FaceMesh indices
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
# LOGGING SERVICE - Singleton pattern for centralized logging
# ============================================================================

class FatigueLogger:
    """Singleton logger for fatigue events with timestamp"""
    
    _instance = None
    
    def __new__(cls, log_file: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, log_file: str = None):
        if self._initialized:
            return
        self.log_file = log_file or FatigueConfig.LOG_FILE
        self._initialized = True
    
    def log(self, event: FatigueEvent, details: str = "") -> None:
        """Log event with timestamp."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {event.value}: {details}\n")
        except Exception as e:
            print(f"Logging error: {e}")


# ============================================================================
# ML DATA COLLECTOR - Structured dataset generation for predictive modeling
# ============================================================================

class MLDataCollector:
    """
    Collects structured physiological and behavioral data for ML training.
    Generates CSV dataset correlating user activity with fatigue indicators.
    """
    
    def __init__(self, csv_file: str = "ergo_ml_dataset.csv", log_interval: float = 60.0):
        self.csv_file = csv_file
        self.log_interval = log_interval
        self.last_log_time = 0.0
        self.session_start_time = time.time()
        self._initialize_csv()
    
    def _initialize_csv(self) -> None:
        """Create CSV file with header if it doesn't exist."""
        import os
        
        file_exists = os.path.exists(self.csv_file)
        
        if not file_exists:
            try:
                with open(self.csv_file, "w", encoding="utf-8") as f:
                    header = (
                        "Timestamp,Session_Time,EAR,MAR,Blink_Count_60s,"
                        "Head_Pitch,Head_Yaw,Head_Roll,"
                        "Typing_CPM,Scroll_SPM,Is_Idle,Fatigue_Label\n"
                    )
                    f.write(header)
                print(f"ML Dataset initialized: {self.csv_file}")
            except Exception as e:
                print(f"Error initializing ML dataset: {e}")
    
    def should_log(self, current_time: float) -> bool:
        return (current_time - self.last_log_time) >= self.log_interval
    
    def log_snapshot(
        self,
        current_time: float,
        ear: float,
        mar: float,
        blink_count: int,
        head_angles: Optional[Tuple[float, float, float]],
        typing_cpm: int,
        scroll_spm: int,
        is_idle: bool,
        fatigue_label: str = ""
    ) -> None:
        """Log a single data snapshot to CSV."""
        try:
            session_time = current_time - self.session_start_time
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if head_angles:
                yaw, pitch, roll = head_angles
            else:
                yaw, pitch, roll = 0.0, 0.0, 0.0
            
            idle_flag = 1 if is_idle else 0
            
            row = (
                f"{timestamp},{session_time:.1f},{ear:.4f},{mar:.4f},{blink_count},"
                f"{pitch:.2f},{yaw:.2f},{roll:.2f},"
                f"{typing_cpm},{scroll_spm},{idle_flag},{fatigue_label}\n"
            )
            
            with open(self.csv_file, "a", encoding="utf-8") as f:
                f.write(row)
            
            self.last_log_time = current_time
            
        except Exception as e:
            print(f"ML logging error: {e}")
    
    def get_session_duration(self) -> float:
        return time.time() - self.session_start_time


# ============================================================================
# ALARM SERVICE - Singleton pattern for audio alerts
# ============================================================================

class AlarmService:
    """Singleton service for playing audio alarms with distinct frequencies"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self.head_down_alarm_active = False
        self.last_head_down_beep_time = 0
        self.head_down_beep_interval = 1.5
        self._initialized = True
    
    def play(self, frequency: int, duration_ms: int) -> None:
        if not _HAS_WINSOUND:
            return
        try:
            winsound.Beep(frequency, duration_ms)
        except Exception as e:
            print(f"Alarm error: {e}")
    
    def start_continuous_head_down_alarm(self, current_time: float) -> None:
        self.head_down_alarm_active = True
        if current_time - self.last_head_down_beep_time >= self.head_down_beep_interval:
            if not _HAS_WINSOUND:
                self.last_head_down_beep_time = current_time
                return
            try:
                winsound.Beep(2800, 400)
                self.last_head_down_beep_time = current_time
            except Exception as e:
                print(f"Head down alarm error: {e}")
    
    def stop_continuous_head_down_alarm(self) -> None:
        self.head_down_alarm_active = False
        self.last_head_down_beep_time = 0


# ============================================================================
# GEOMETRY UTILITIES - Fast distance calculations
# ============================================================================

class GeometryUtils:
    """Static utility class for geometric computations"""
    
    @staticmethod
    def euclidean_distance_2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    
    @staticmethod
    def euclidean_distance_landmarks(p1, p2) -> float:
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


# ============================================================================
# DETECTION ALGORITHMS - Strategy pattern implementation
# ============================================================================

class FatigueDetector(ABC):
    """Abstract base class for fatigue detection strategies"""
    
    @abstractmethod
    def detect(self, *args, **kwargs) -> bool:
        pass


class EARCalculator:
    """Eye Aspect Ratio (EAR) Calculator"""
    
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
    """Mouth Aspect Ratio (MAR) Calculator"""
    
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
    """Head pose estimation using solvePnP algorithm"""
    
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
# STATE MANAGERS - Encapsulated state tracking with clean interfaces
# ============================================================================

class BlinkCounter:
    """Efficient blink counting with rolling time window"""
    
    def __init__(self, config: FatigueConfig):
        self.config = config
        self.timestamps = deque()
        self.frame_counter = 0
        self.is_blinking = False
        self.last_report_time = time.time()
        self.logger = FatigueLogger()
    
    def update(self, ear: float, current_time: float) -> int:
        while self.timestamps and (current_time - self.timestamps[0]) > self.config.BLINK_WINDOW_SECONDS:
            self.timestamps.popleft()
        
        if ear < self.config.BLINK_EAR_THRESHOLD:
            self.frame_counter += 1
        else:
            if self.frame_counter >= self.config.BLINK_CONSECUTIVE_FRAMES and not self.is_blinking:
                self.timestamps.append(current_time)
                self.is_blinking = True
                self.logger.log(FatigueEvent.BLINK, f"Total in window: {len(self.timestamps)}")
            
            self.frame_counter = 0
            self.is_blinking = False
        
        if current_time - self.last_report_time >= self.config.BLINK_WINDOW_SECONDS:
            self.logger.log(FatigueEvent.BLINK_REPORT, f"Blink count in last 60s: {len(self.timestamps)}")
            self.last_report_time = current_time
        
        return len(self.timestamps)


class TimedAlertTracker:
    """Generic timer-based alert tracker for various fatigue indicators"""
    
    def __init__(self, threshold_time: float):
        self.threshold_time = threshold_time
        self.start_time: Optional[float] = None
        self.alert_active = False
    
    def update(self, condition_active: bool, current_time: float) -> Tuple[bool, float]:
        if condition_active:
            if self.start_time is None:
                self.start_time = current_time
            elapsed = current_time - self.start_time
            should_alert = elapsed >= self.threshold_time
            return should_alert, elapsed
        else:
            self.start_time = None
            self.alert_active = False
            return False, 0.0
    
    def reset(self):
        self.start_time = None
        self.alert_active = False
    
    @property
    def is_active(self) -> bool:
        return self.alert_active


class EyeRubCounter:
    """Eye rub counter with debouncing to prevent double-counting"""
    
    def __init__(self, config: FatigueConfig):
        self.config = config
        self.count = 0
        self.is_rubbing = False
        self.last_rub_time = 0.0
        self.alert_triggered = False
    
    def update(self, is_rubbing: bool, current_time: float) -> int:
        if is_rubbing:
            if not self.is_rubbing:
                if current_time - self.last_rub_time > self.config.EYE_RUB_DEBOUNCE_TIME:
                    self.count += 1
                    self.last_rub_time = current_time
                self.is_rubbing = True
        else:
            self.is_rubbing = False
        return self.count
    
    def should_alert(self) -> bool:
        return self.count >= self.config.EYE_RUB_LIMIT


# ============================================================================
# MAIN FATIGUE DETECTION SYSTEM - Orchestrates all components
# ============================================================================

class FatigueDetectionSystem:
    """
    Main system orchestrating all fatigue detection components.
    Standalone version with OpenCV window and audio alarms.
    """
    
    def __init__(self, config: FatigueConfig = None):
        self.config = config or FatigueConfig()
        
        # Initialize MediaPipe
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
        
        # Initialize components
        self.logger = FatigueLogger(self.config.LOG_FILE)
        self.alarm = AlarmService()
        self.head_pose_estimator = HeadPoseEstimator()
        self.blink_counter = BlinkCounter(self.config)
        self.ml_collector = MLDataCollector(csv_file="ergo_ml_dataset.csv", log_interval=60.0)
        
        # Initialize trackers
        self.eye_closure_tracker = TimedAlertTracker(self.config.EYE_CLOSURE_WARNING_TIME)
        self.head_pose_tracker = TimedAlertTracker(self.config.HEAD_POSE_WARNING_TIME)
        self.head_drop_tracker = TimedAlertTracker(self.config.HEAD_DROP_WARNING_TIME)
        self.hand_head_tracker = TimedAlertTracker(self.config.HAND_HEAD_WARNING_TIME)
        self.eye_rub_counter = EyeRubCounter(self.config)
        self.activity_tracker = SystemActivityTracker()
        
        # Absence & Inactivity trackers
        self.absence_tracker = TimedAlertTracker(self.config.ABSENCE_WARNING_TIME)
        self.idle_tracker = TimedAlertTracker(self.config.IDLE_WARNING_TIME)
        
        # Head pose smoothing
        self.smooth_pitch = 0.0
        self.pose_smoothing_factor = 0.2
        
        # Yawn detection state
        self.yawn_count = 0
        self.yawn_active = False
        
        # Camera
        self.cap = cv2.VideoCapture(0)
        
        print("Fatigue Detection System initialized successfully")
    
    def _detect_yawning(self, face_landmarks) -> Tuple[bool, float]:
        mar = MARCalculator.calculate(face_landmarks)
        is_yawning = mar > self.config.YAWN_MAR_THRESHOLD
        
        if is_yawning and not self.yawn_active:
            self.yawn_count += 1
            self.yawn_active = True
            self.logger.log(FatigueEvent.YAWN, f"Count: {self.yawn_count}, MAR: {mar:.3f}")
        elif not is_yawning:
            self.yawn_active = False
        
        return is_yawning, mar
    
    def _detect_eye_closure(self, face_landmarks, current_time: float) -> Tuple[bool, float, int]:
        left_ear = EARCalculator.calculate(FaceLandmarks.LEFT_EYE_INDICES, face_landmarks)
        right_ear = EARCalculator.calculate(FaceLandmarks.RIGHT_EYE_INDICES, face_landmarks)
        avg_ear = (left_ear + right_ear) / 2.0
        
        is_closed = avg_ear < self.config.EYE_CLOSURE_THRESHOLD
        blink_count = self.blink_counter.update(avg_ear, current_time)
        
        should_alert, elapsed = self.eye_closure_tracker.update(is_closed, current_time)
        
        if should_alert and not self.eye_closure_tracker.is_active:
            self.alarm.play(self.config.ALARM_EYE_CLOSURE, 700)
            self.logger.log(FatigueEvent.EYE_CLOSURE, f"Duration: {elapsed:.1f}s, EAR: {avg_ear:.3f}")
            self.eye_closure_tracker.alert_active = True
        
        return is_closed, avg_ear, blink_count
    
    def _detect_head_pose(self, face_landmarks, frame_width: int, frame_height: int,
                          current_time: float) -> Optional[Tuple[float, float, float]]:
        angles = self.head_pose_estimator.estimate(face_landmarks, frame_width, frame_height)
        
        if angles is None:
            return None
        
        yaw, raw_pitch, roll = angles
        
        self.smooth_pitch = (raw_pitch * self.pose_smoothing_factor) + \
                           (self.smooth_pitch * (1.0 - self.pose_smoothing_factor))
        
        not_looking = abs(yaw) > self.config.HEAD_YAW_THRESHOLD or abs(raw_pitch) > self.config.HEAD_PITCH_THRESHOLD
        should_alert, elapsed = self.head_pose_tracker.update(not_looking, current_time)
        
        if should_alert and not self.head_pose_tracker.is_active:
            self.alarm.play(self.config.ALARM_HEAD_POSE, 500)
            self.logger.log(FatigueEvent.HEAD_POSE_ALERT, f"Yaw: {yaw:.1f}, Pitch: {raw_pitch:.1f}, Duration: {elapsed:.1f}s")
            self.head_pose_tracker.alert_active = True
        
        head_dropped = self.smooth_pitch < -self.config.HEAD_DROP_PITCH_THRESHOLD
        should_alert_drop, elapsed_drop = self.head_drop_tracker.update(head_dropped, current_time)
        
        if should_alert_drop:
            if not self.head_drop_tracker.is_active:
                self.logger.log(FatigueEvent.HEAD_DROP, f"Smoothed Pitch: {self.smooth_pitch:.1f}, Duration: {elapsed_drop:.1f}s - CONTINUOUS ALARM STARTED")
                self.head_drop_tracker.alert_active = True
            self.alarm.start_continuous_head_down_alarm(current_time)
        else:
            if self.head_drop_tracker.is_active:
                self.logger.log(FatigueEvent.HEAD_DROP, f"Head raised (Smoothed Pitch: {self.smooth_pitch:.1f}) - Alarm stopped")
                self.alarm.stop_continuous_head_down_alarm()
                self.head_drop_tracker.alert_active = False
        
        return (yaw, self.smooth_pitch, roll)
    
    def _detect_hand_eye_contact(self, hand_landmarks, face_landmarks) -> bool:
        hand_points = [hand_landmarks.landmark[idx] for idx in [8, 12, 16, 20, 4, 0]]
        
        for hand_point in hand_points:
            hand_pos = (hand_point.x, hand_point.y)
            
            for eye_idx in FaceLandmarks.LEFT_EYE_REGION:
                eye_pos = (face_landmarks.landmark[eye_idx].x, face_landmarks.landmark[eye_idx].y)
                if GeometryUtils.euclidean_distance_2d(hand_pos, eye_pos) < self.config.EYE_RUB_DISTANCE_THRESHOLD:
                    return True
            
            for eye_idx in FaceLandmarks.RIGHT_EYE_REGION:
                eye_pos = (face_landmarks.landmark[eye_idx].x, face_landmarks.landmark[eye_idx].y)
                if GeometryUtils.euclidean_distance_2d(hand_pos, eye_pos) < self.config.EYE_RUB_DISTANCE_THRESHOLD:
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
            if GeometryUtils.euclidean_distance_2d(hand_pos, head_center) < self.config.HAND_HEAD_DISTANCE_THRESHOLD:
                return True
        
        return False
    
    def _process_hands(self, hand_results, face_landmarks, current_time: float) -> Tuple[bool, bool, int]:
        is_rubbing = False
        hand_on_head = False
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                if self._detect_hand_eye_contact(hand_landmarks, face_landmarks):
                    is_rubbing = True
                if self._detect_hand_head_proximity(hand_landmarks, face_landmarks):
                    hand_on_head = True
        
        rub_count = self.eye_rub_counter.update(is_rubbing, current_time)
        
        if self.eye_rub_counter.should_alert() and not self.eye_rub_counter.alert_triggered:
            self.alarm.play(self.config.ALARM_EYE_RUB, 600)
            self.logger.log(FatigueEvent.EYE_RUB, f"Count: {rub_count}")
            self.eye_rub_counter.alert_triggered = True
        
        should_alert, elapsed = self.hand_head_tracker.update(hand_on_head, current_time)
        
        if should_alert and not self.hand_head_tracker.is_active:
            self.alarm.play(self.config.ALARM_HAND_HEAD, 700)
            self.logger.log(FatigueEvent.HAND_ON_HEAD, f"Duration: {elapsed:.1f}s")
            self.hand_head_tracker.alert_active = True
        
        return is_rubbing, hand_on_head, rub_count
    
    def _render_ui(self, frame, stats: dict) -> None:
        """Render all UI elements on frame"""
        if stats['is_yawning']:
            cv2.putText(frame, "YAWNING!", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(frame, f"MAR: {stats['mar']:.2f}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Yawn Count: {stats['yawn_count']}", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        if stats['yawn_count'] >= self.config.YAWN_FATIGUE_LIMIT:
            cv2.putText(frame, "FATIGUE DETECTED!", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        if stats['angles']:
            yaw, pitch, roll = stats['angles']
            cv2.putText(frame, f"Yaw: {yaw:.1f}deg", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
            cv2.putText(frame, f"Pitch: {pitch:.1f}deg", (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
            cv2.putText(frame, f"Roll: {roll:.1f}deg", (30, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
            
            if stats['head_pose_alert']:
                cv2.putText(frame, f"Not Looking: {stats['head_pose_time']:.1f}s", (30, 330),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                if stats['head_pose_time'] >= self.config.HEAD_POSE_WARNING_TIME:
                    cv2.putText(frame, "WARNING: Please look forward!", (30, 380),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        eye_state = "CLOSED" if stats['eyes_closed'] else "OPEN"
        eye_color = (0, 0, 255) if stats['eyes_closed'] else (0, 255, 0)
        cv2.putText(frame, f"EAR: {stats['ear']:.3f}", (30, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Eyes: {eye_state}", (30, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2)
        
        if stats['eye_closure_alert']:
            cv2.putText(frame, f"Eyes Closed: {stats['eye_closure_time']:.1f}s", (30, 480),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
            if stats['eye_closure_time'] >= self.config.EYE_CLOSURE_WARNING_TIME:
                cv2.putText(frame, "ALERT: Eyes closed too long!", (30, 520),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        cv2.putText(frame, f"Blinks (60s): {stats['blink_count']}", (30, 550),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 0), 2)
        
        if stats['is_rubbing']:
            cv2.putText(frame, "Rubbing Eyes...", (30, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        cv2.putText(frame, f"Eye Rubs: {stats['rub_count']}", (30, 610), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        if stats['rub_count'] >= self.config.EYE_RUB_LIMIT:
            cv2.putText(frame, "WARNING: Eye rubbing detected!", (30, 650),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        if stats['head_drop_alert']:
            cv2.putText(frame, f"Head Dropped: {stats['head_drop_time']:.1f}s", (30, 690),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
            
            if stats['head_drop_time'] >= self.config.HEAD_DROP_WARNING_TIME:
                alarm_color = (0, 0, 255) if int(time.time() * 2) % 2 == 0 else (0, 255, 255)
                alarm_text = "ALARM: HEAD DOWN!"
                font_scale = 2.0
                thickness = 5
                text_size = cv2.getTextSize(alarm_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                text_width, text_height = text_size
                frame_h, frame_w = frame.shape[:2]
                center_x = (frame_w - text_width) // 2
                center_y = (frame_h + text_height) // 2
                cv2.putText(frame, alarm_text, (center_x, center_y),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, alarm_color, thickness)
        
        if stats['hand_on_head_alert']:
            cv2.putText(frame, f"Hand on Head: {stats['hand_head_time']:.1f}s", (30, 770),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)
            if stats['hand_head_time'] >= self.config.HAND_HEAD_WARNING_TIME:
                cv2.putText(frame, "WARNING: Possible sleep/fatigue detected!", (30, 810),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        frame_h, frame_w = frame.shape[:2]
        right_x = frame_w - 400
        
        window_title = stats.get('activity_window', 'Unknown')
        if len(window_title) > 25: window_title = window_title[:22] + "..."
        cv2.putText(frame, f"Win: {window_title}", (right_x, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        typing_cpm = stats.get('typing_cpm', 0)
        scroll_spm = stats.get('scroll_spm', 0)
        cv2.putText(frame, f"Type: {typing_cpm} CPM", (right_x, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame, f"Scroll: {scroll_spm} SPM", (right_x, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        if stats.get('user_idle', False):
            cv2.putText(frame, "STATUS: IDLE", (right_x, 170),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "STATUS: ACTIVE", (right_x, 170),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        center_x = frame_w // 2
        center_y = frame_h // 2
        
        if stats.get('user_absent', False):
            warning_text = "USER ABSENT!"
            text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)[0]
            text_x = center_x - text_size[0] // 2
            text_y = center_y - 100
            cv2.rectangle(frame, (text_x - 20, text_y - text_size[1] - 20),
                         (text_x + text_size[0] + 20, text_y + 20), (0, 0, 255), -1)
            cv2.putText(frame, warning_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
            cv2.putText(frame, f"Missing for {stats['absence_time']:.1f}s",
                       (text_x, text_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        if stats.get('idle_alert', False):
            warning_text = "WAKE UP!"
            subtext = "ACTIVE BREAK NEEDED"
            text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 5)[0]
            text_x = center_x - text_size[0] // 2
            text_y = center_y + 50
            cv2.rectangle(frame, (text_x - 30, text_y - text_size[1] - 30),
                         (text_x + text_size[0] + 30, text_y + 80), (0, 0, 255), -1)
            cv2.putText(frame, warning_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 5)
            sub_size = cv2.getTextSize(subtext, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            sub_x = center_x - sub_size[0] // 2
            cv2.putText(frame, subtext, (sub_x, text_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(frame, f"Idle for {stats['idle_time']:.1f}s",
                       (sub_x, text_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    def run(self) -> None:
        """Main loop for real-time fatigue detection"""
        print("Starting fatigue detection... Press ESC to exit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (1280, 720))
            current_time = time.time()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_height, frame_width = frame.shape[:2]
            
            face_results = self.face_mesh.process(frame_rgb)
            hand_results = self.hands.process(frame_rgb)
            
            stats = {
                'is_yawning': False, 'mar': 0.0, 'yawn_count': self.yawn_count,
                'eyes_closed': False, 'ear': 0.0, 'blink_count': 0,
                'eye_closure_alert': False, 'eye_closure_time': 0.0,
                'angles': None, 'head_pose_alert': False, 'head_pose_time': 0.0,
                'head_drop_alert': False, 'head_drop_time': 0.0,
                'is_rubbing': False, 'rub_count': 0,
                'hand_on_head_alert': False, 'hand_head_time': 0.0,
                'activity_window': "Unknown", 'typing_cpm': 0,
                'scroll_spm': 0, 'user_idle': True,
                'user_absent': False, 'absence_time': 0.0,
                'idle_alert': False, 'idle_time': 0.0
            }

            activity_stats = self.activity_tracker.get_activity_stats()
            stats['activity_window'] = activity_stats['active_window']
            stats['typing_cpm'] = activity_stats['typing_cpm']
            stats['scroll_spm'] = activity_stats['scroll_spm']
            stats['user_idle'] = activity_stats['is_idle']
            
            face_detected = face_results.multi_face_landmarks is not None
            should_alert_absence, elapsed_absence = self.absence_tracker.update(not face_detected, current_time)
            
            if should_alert_absence:
                if not self.absence_tracker.is_active:
                    self.alarm.play(self.config.ALARM_ABSENCE, 800)
                    self.logger.log(FatigueEvent.USER_ABSENT, f"Duration: {elapsed_absence:.1f}s")
                    self.absence_tracker.alert_active = True
                stats['user_absent'] = True
                stats['absence_time'] = elapsed_absence
            
            should_alert_idle, elapsed_idle = self.idle_tracker.update(stats['user_idle'], current_time)
            
            if should_alert_idle:
                if not self.idle_tracker.is_active:
                    self.alarm.play(self.config.ALARM_IDLE, 600)
                    self.logger.log(FatigueEvent.USER_IDLE, f"Duration: {elapsed_idle:.1f}s")
                    self.idle_tracker.alert_active = True
                stats['idle_alert'] = True
                stats['idle_time'] = elapsed_idle
            
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]
                
                stats['is_yawning'], stats['mar'] = self._detect_yawning(face_landmarks)
                stats['eyes_closed'], stats['ear'], stats['blink_count'] = self._detect_eye_closure(face_landmarks, current_time)
                stats['angles'] = self._detect_head_pose(face_landmarks, frame_width, frame_height, current_time)
                
                stats['eye_closure_alert'] = self.eye_closure_tracker.start_time is not None
                if stats['eye_closure_alert']:
                    stats['eye_closure_time'] = current_time - self.eye_closure_tracker.start_time
                
                stats['head_pose_alert'] = self.head_pose_tracker.start_time is not None
                if stats['head_pose_alert']:
                    stats['head_pose_time'] = current_time - self.head_pose_tracker.start_time
                
                stats['head_drop_alert'] = self.head_drop_tracker.start_time is not None
                if stats['head_drop_alert']:
                    stats['head_drop_time'] = current_time - self.head_drop_tracker.start_time
                
                stats['is_rubbing'], hand_on_head, stats['rub_count'] = self._process_hands(hand_results, face_landmarks, current_time)
                
                stats['hand_on_head_alert'] = self.hand_head_tracker.start_time is not None
                if stats['hand_on_head_alert']:
                    stats['hand_head_time'] = current_time - self.hand_head_tracker.start_time
            
            if self.ml_collector.should_log(current_time):
                self.ml_collector.log_snapshot(
                    current_time=current_time,
                    ear=stats.get('ear', 0.0),
                    mar=stats.get('mar', 0.0),
                    blink_count=stats.get('blink_count', 0),
                    head_angles=stats.get('angles'),
                    typing_cpm=stats.get('typing_cpm', 0),
                    scroll_spm=stats.get('scroll_spm', 0),
                    is_idle=stats.get('user_idle', True),
                    fatigue_label=""
                )
            
            self._render_ui(frame, stats)
            cv2.imshow("Fatigue Detection System", frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        self.cleanup()
    
    def cleanup(self) -> None:
        """Release resources gracefully"""
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception as e:
            print(f"Camera release error: {e}")
        
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Window cleanup error: {e}")
        
        try:
            if hasattr(self.face_mesh, '_graph') and self.face_mesh._graph:
                self.face_mesh.close()
        except Exception as e:
            print(f"Face mesh cleanup error: {e}")
        
        try:
            if hasattr(self.hands, '_graph') and self.hands._graph:
                self.hands.close()
        except Exception as e:
            print(f"Hands cleanup error: {e}")
        
        try:
            self.activity_tracker.cleanup()
        except Exception as e:
            print(f"Activity tracker cleanup error: {e}")
        
        print("System shutdown complete")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    config = FatigueConfig()
    system = FatigueDetectionSystem(config)
    
    try:
        system.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        system.cleanup()


if __name__ == "__main__":
    main()
