"""
YAQIZ Detection Service
Wraps the existing YOLO PPE detection pipeline into a clean service layer.
Reuses the original model weights and class definitions.

Performance optimizations:
- GPU (CUDA) inference with FP16 half-precision when available
- Configurable frame skipping to reduce processing load
- Smaller inference resolution for faster throughput
"""

import cv2
import math
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Generator, Tuple
from ultralytics import YOLO
from app.core.config import settings
import logging

logger = logging.getLogger("yaqiz.detection")


# Original class names from the PPE detection model
CLASS_NAMES = [
    'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
    'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle'
]

# Classification categories
SAFE_CLASSES = {'Hardhat', 'Mask', 'Safety Vest'}
VIOLATION_CLASSES = {'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest'}
NEUTRAL_CLASSES = {'Person', 'Safety Cone', 'machinery', 'vehicle'}

# Color mapping (BGR for OpenCV)
COLORS = {
    'safe': (0, 255, 0),        # Green
    'violation': (0, 0, 255),    # Red
    'equipment': (0, 149, 255),  # Orange
    'person': (255, 45, 85),     # Purple
}

# ── Performance defaults ──
DEFAULT_FRAME_SKIP = 3          # Process every Nth frame (1 = no skip)
DEFAULT_IMGSZ = 640             # Inference resolution (lower = faster)


class DetectionService:
    """Core YOLO PPE detection service"""

    _instance: Optional['DetectionService'] = None
    _model: Optional[YOLO] = None
    _device: str = "cpu"
    _use_half: bool = False

    @classmethod
    def get_instance(cls) -> 'DetectionService':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.confidence_threshold = settings.YOLO_CONFIDENCE
        self._setup_device()
        self._load_model()

    def _setup_device(self):
        """Select GPU if available, enable FP16 half-precision"""
        if torch.cuda.is_available():
            DetectionService._device = "cuda"
            DetectionService._use_half = True
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
            logger.info(f"🚀 GPU detected: {gpu_name} ({vram:.1f} GB VRAM)")
            logger.info("   Using CUDA with FP16 half-precision")
        else:
            DetectionService._device = "cpu"
            DetectionService._use_half = False
            logger.info("⚠️  No GPU detected — running on CPU (slower)")

    def _load_model(self):
        """Load YOLO model (reusing existing weights) and move to GPU"""
        if DetectionService._model is None:
            weights = settings.weights_path
            if not weights.exists():
                # Fallback to other weight locations
                alt_paths = [
                    settings.base_dir / "best.pt",
                    settings.base_dir / "ppe.pt",
                    settings.base_dir / "YOLO-Weights" / "ppe.pt",
                ]
                for alt in alt_paths:
                    if alt.exists():
                        weights = alt
                        break
            logger.info(f"Loading YOLO model from: {weights}")
            DetectionService._model = YOLO(str(weights))
            # Move model to GPU
            DetectionService._model.to(DetectionService._device)
            logger.info(f"Model device: {DetectionService._device}")
        self.model = DetectionService._model

    def detect_frame(self, frame: np.ndarray, confidence: Optional[float] = None,
                     imgsz: int = DEFAULT_IMGSZ) -> Dict:
        """
        Run detection on a single frame (no tracking).
        Returns structured detection results with compliance analysis.
        Uses GPU + FP16 when available for maximum speed.
        """
        conf = confidence or self.confidence_threshold
        results = self.model(
            frame,
            stream=False,
            conf=conf,
            device=self._device,
            half=self._use_half,
            imgsz=imgsz,
            verbose=False,
        )
        return self._parse_results(results)

    def track_frame(self, frame: np.ndarray, confidence: Optional[float] = None,
                    imgsz: int = DEFAULT_IMGSZ, persist: bool = True) -> Dict:
        """
        Run detection + tracking on a single frame.
        Uses YOLO's built-in ByteTrack tracker for persistent IDs.
        Returns same structure as detect_frame but with track_id on each detection.
        """
        conf = confidence or self.confidence_threshold
        results = self.model.track(
            frame,
            stream=False,
            conf=conf,
            device=self._device,
            half=self._use_half,
            imgsz=imgsz,
            verbose=False,
            persist=persist,
            tracker="bytetrack.yaml",
        )
        return self._parse_results(results, with_tracking=True)

    def _parse_results(self, results, with_tracking: bool = False) -> Dict:
        """Parse YOLO results into structured detection data"""

        detections = []
        violations = []
        safe_equipment = []
        workers = 0

        for r in results:
            boxes = r.boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                box_conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = CLASS_NAMES[cls]

                is_violation = class_name in VIOLATION_CLASSES
                is_safe = class_name in SAFE_CLASSES

                detection = {
                    'class_name': class_name,
                    'confidence': float(box_conf),
                    'bbox': [x1, y1, x2, y2],
                    'is_violation': is_violation,
                }

                # Add track ID if tracking is enabled and available
                if with_tracking and box.id is not None:
                    detection['track_id'] = int(box.id[0])

                detections.append(detection)

                if is_violation:
                    violations.append(detection)
                elif is_safe:
                    safe_equipment.append(detection)

                if class_name == 'Person':
                    workers += 1

        # Compliance calculation
        helmet_total = sum(1 for d in detections if d['class_name'] in ('Hardhat', 'NO-Hardhat'))
        helmet_safe = sum(1 for d in detections if d['class_name'] == 'Hardhat')
        vest_total = sum(1 for d in detections if d['class_name'] in ('Safety Vest', 'NO-Safety Vest'))
        vest_safe = sum(1 for d in detections if d['class_name'] == 'Safety Vest')
        mask_total = sum(1 for d in detections if d['class_name'] in ('Mask', 'NO-Mask'))
        mask_safe = sum(1 for d in detections if d['class_name'] == 'Mask')

        return {
            'detections': detections,
            'violations': violations,
            'workers_count': workers,
            'total_detections': len(detections),
            'violations_count': len(violations),
            'helmet_compliance': (helmet_safe / helmet_total * 100) if helmet_total > 0 else 100.0,
            'vest_compliance': (vest_safe / vest_total * 100) if vest_total > 0 else 100.0,
            'mask_compliance': (mask_safe / mask_total * 100) if mask_total > 0 else 100.0,
        }

    def annotate_frame(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Draw bounding boxes, labels and track IDs on frame"""
        annotated = frame.copy()

        for det in result['detections']:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            conf = det['confidence']
            track_id = det.get('track_id')

            # Build label with track ID for persons
            if track_id is not None and class_name == 'Person':
                label = f'Worker #{track_id}'
            else:
                label = f'{class_name} {conf}'

            # Choose color
            if class_name in SAFE_CLASSES:
                color = COLORS['safe']
            elif class_name in VIOLATION_CLASSES:
                color = COLORS['violation']
            elif class_name in ('machinery', 'vehicle'):
                color = COLORS['equipment']
            else:
                color = COLORS['person']

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            # Draw label background
            t_size = cv2.getTextSize(label, 0, fontScale=0.8, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 6
            cv2.rectangle(annotated, (x1, y1), c2, color, -1, cv2.LINE_AA)
            cv2.putText(annotated, label, (x1, y1 - 4), 0, 0.8, [255, 255, 255],
                        thickness=2, lineType=cv2.LINE_AA)

        # Draw status overlay
        self._draw_status_overlay(annotated, result)

        return annotated

    def _draw_status_overlay(self, frame: np.ndarray, result: Dict):
        """Draw compliance status overlay on frame"""
        h, w = frame.shape[:2]

        # Semi-transparent overlay bar at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 45), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Status text
        workers = result['workers_count']
        violations = result['violations_count']
        helmet_comp = result['helmet_compliance']
        vest_comp = result['vest_compliance']

        status_color = (0, 255, 0) if violations == 0 else (0, 0, 255)
        status_text = "COMPLIANT" if violations == 0 else f"VIOLATIONS: {violations}"

        cv2.putText(frame, f"YAQIZ | Workers: {workers} | {status_text}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"Helmet: {helmet_comp:.0f}% | Vest: {vest_comp:.0f}%",
                    (w - 350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def process_video_stream(self, source, confidence: Optional[float] = None,
                             frame_skip: int = DEFAULT_FRAME_SKIP,
                             tracker: 'WorkerTracker' = None) -> Generator:
        """
        Generator that processes video frames with tracking and frame skipping.
        Yields (annotated_frame, detection_result, tracking_data) tuples.
        source: file path string or integer (0 for webcam)
        frame_skip: process every Nth frame (1 = no skip, 3 = skip 2 out of 3)
        tracker: optional WorkerTracker instance for persistent ID tracking
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"Cannot open video source: {source}")
            return

        frame_num = 0
        last_result = None
        last_tracking = None

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                frame_num += 1

                # Only run detection every Nth frame
                if frame_num % frame_skip == 0 or last_result is None:
                    result = self.track_frame(frame, confidence)
                    result['frame_number'] = frame_num
                    annotated = self.annotate_frame(frame, result)
                    last_result = result
                    # Update worker tracker
                    tracking_data = None
                    if tracker is not None:
                        tracking_data = tracker.update(result['detections'], frame_num)
                    last_tracking = tracking_data
                else:
                    # Reuse last detection result, just re-draw on current frame
                    annotated = self.annotate_frame(frame, last_result)
                    result = {**last_result, 'frame_number': frame_num}
                    tracking_data = last_tracking

                yield annotated, result, tracking_data
        finally:
            cap.release()

    def process_image(self, image_path: str, confidence: Optional[float] = None) -> Tuple[np.ndarray, Dict]:
        """Process a single image and return annotated image + results"""
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Cannot read image: {image_path}")

        result = self.detect_frame(frame, confidence)
        annotated = self.annotate_frame(frame, result)
        return annotated, result

    def process_video_file(self, input_path: str, output_path: str,
                           confidence: Optional[float] = None,
                           progress_callback=None,
                           frame_skip: int = DEFAULT_FRAME_SKIP,
                           tracker: 'WorkerTracker' = None) -> Dict:
        """
        Process entire video file with tracking and save annotated output.
        Uses frame skipping to dramatically reduce processing time.
        Returns session summary including worker tracking data.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        all_detections = 0
        all_violations = 0
        total_helmet_comp = 0
        total_vest_comp = 0
        frames_with_workers = 0
        total_workers = 0
        alerts = []
        frame_num = 0
        frames_processed = 0
        last_result = None

        logger.info(f"Processing video: {total_frames} frames, skip={frame_skip}, device={self._device}")

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                frame_num += 1

                # Only run inference every Nth frame
                if frame_num % frame_skip == 0 or last_result is None:
                    result = self.track_frame(frame, confidence)
                    last_result = result
                    frames_processed += 1

                    # Update worker tracker
                    if tracker is not None:
                        tracker.update(result['detections'], frame_num)
                else:
                    result = last_result

                annotated = self.annotate_frame(frame, result)
                out.write(annotated)

                all_detections += result['total_detections']
                all_violations += result['violations_count']
                total_helmet_comp += result['helmet_compliance']
                total_vest_comp += result['vest_compliance']

                if result['workers_count'] > 0:
                    frames_with_workers += 1
                    total_workers += result['workers_count']

                # Generate alerts for violations (only on processed frames)
                if frame_num % frame_skip == 0 or frames_processed <= 1:
                    for v in result['violations']:
                        alerts.append({
                            'alert_type': f"missing_{v['class_name'].replace('NO-', '').lower()}",
                            'severity': 'critical' if v['confidence'] > 0.8 else 'warning',
                            'message': f"{v['class_name']} detected (conf: {v['confidence']:.2f})",
                            'confidence': v['confidence'],
                            'frame_number': frame_num,
                        })

                if progress_callback and frame_num % 30 == 0:
                    progress_callback(frame_num, total_frames)
        finally:
            cap.release()
            out.release()

        compliance = ((all_detections - all_violations) / all_detections * 100) if all_detections > 0 else 100

        logger.info(f"Done: {frame_num} frames, {frames_processed} inferred (skip={frame_skip})")

        summary = {
            'total_frames': frame_num,
            'frames_inferred': frames_processed,
            'frame_skip': frame_skip,
            'total_detections': all_detections,
            'violations_count': all_violations,
            'compliance_rate': round(compliance, 1),
            'helmet_compliance': round(total_helmet_comp / max(frame_num, 1), 1),
            'vest_compliance': round(total_vest_comp / max(frame_num, 1), 1),
            'workers_detected': total_workers,
            'alerts': alerts[:100],  # Limit stored alerts
            'alerts_generated': len(alerts),
            'device': self._device,
        }

        # Add worker tracking summary
        if tracker is not None:
            tracking_summary = tracker.get_summary()
            summary['tracking'] = tracking_summary
            summary['worker_records'] = tracker.get_all_workers()
            logger.info(f"Tracking: {tracking_summary['total_unique_workers']} unique workers, "
                        f"{tracking_summary['violating_workers']} with violations")

        return summary


# Singleton accessor
def get_detection_service() -> DetectionService:
    return DetectionService.get_instance()
