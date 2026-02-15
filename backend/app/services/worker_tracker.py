"""
YAQIZ Worker Tracker Service
Assigns persistent IDs to workers across frames using YOLO's built-in
ByteTrack / BoT-SORT tracker. Maintains a per-worker violation log.
"""

import time
import math
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger("yaqiz.tracker")


@dataclass
class WorkerRecord:
    """Per-worker tracking record"""
    worker_id: int
    first_seen_frame: int
    last_seen_frame: int
    first_seen_time: float
    last_seen_time: float
    total_frames: int = 0
    # Violation counters
    no_hardhat_count: int = 0
    no_vest_count: int = 0
    no_mask_count: int = 0
    hardhat_count: int = 0
    vest_count: int = 0
    mask_count: int = 0
    # Status
    current_violations: List[str] = field(default_factory=list)
    is_compliant: bool = True
    # History log (last N events)
    log: List[Dict] = field(default_factory=list)

    @property
    def violation_count(self) -> int:
        return self.no_hardhat_count + self.no_vest_count + self.no_mask_count

    @property
    def has_violations(self) -> bool:
        return self.violation_count > 0

    @property
    def duration_seconds(self) -> float:
        return self.last_seen_time - self.first_seen_time

    def to_dict(self) -> Dict:
        return {
            "worker_id": self.worker_id,
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
            "total_frames": self.total_frames,
            "duration_seconds": round(self.duration_seconds, 1),
            "is_compliant": self.is_compliant,
            "current_violations": self.current_violations,
            "violations": {
                "no_hardhat": self.no_hardhat_count,
                "no_vest": self.no_vest_count,
                "no_mask": self.no_mask_count,
                "total": self.violation_count,
            },
            "equipment_seen": {
                "hardhat": self.hardhat_count,
                "vest": self.vest_count,
                "mask": self.mask_count,
            },
            "log": self.log[-20:],  # Last 20 events
        }


class WorkerTracker:
    """
    Tracks workers across video frames with persistent IDs.
    Uses spatial IoU matching to associate PPE detections with tracked persons.
    """

    def __init__(self):
        self.workers: Dict[int, WorkerRecord] = {}
        self._frame_count: int = 0
        self._iou_threshold: float = 0.3  # Overlap threshold for PPE-to-person association

    def reset(self):
        """Reset tracker state for a new session"""
        self.workers.clear()
        self._frame_count = 0

    @staticmethod
    def _iou(box_a: List[int], box_b: List[int]) -> float:
        """Calculate Intersection over Union between two bboxes [x1,y1,x2,y2]"""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        return inter / (area_a + area_b - inter)

    @staticmethod
    def _contains(person_box: List[int], item_box: List[int]) -> bool:
        """Check if item_box center is inside person_box (for PPE association)"""
        cx = (item_box[0] + item_box[2]) / 2
        cy = (item_box[1] + item_box[3]) / 2
        return (person_box[0] <= cx <= person_box[2] and
                person_box[1] <= cy <= person_box[3])

    @staticmethod
    def _overlap_ratio(person_box: List[int], item_box: List[int]) -> float:
        """How much of item_box overlaps with expanded person_box"""
        # Expand person box vertically (PPE can be above/below person center)
        px1 = person_box[0] - 30
        py1 = person_box[1] - 50
        px2 = person_box[2] + 30
        py2 = person_box[3] + 20
        x1 = max(px1, item_box[0])
        y1 = max(py1, item_box[1])
        x2 = min(px2, item_box[2])
        y2 = min(py2, item_box[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        item_area = max(1, (item_box[2] - item_box[0]) * (item_box[3] - item_box[1]))
        return inter / item_area

    def update(self, tracked_detections: List[Dict], frame_number: int) -> Dict:
        """
        Update tracker with detections that include track IDs.
        
        tracked_detections: list of dicts with keys:
            class_name, confidence, bbox, is_violation, track_id (optional)
        
        Returns summary with unique_workers, unique_violators, worker_records.
        """
        self._frame_count = frame_number
        now = time.time()

        # Separate persons (tracked) from PPE/violation items
        persons = []
        ppe_items = []
        for det in tracked_detections:
            if det['class_name'] == 'Person' and det.get('track_id') is not None:
                persons.append(det)
            elif det['class_name'] not in ('Safety Cone', 'machinery', 'vehicle'):
                ppe_items.append(det)

        # Update/create worker records for each tracked person
        active_ids = set()
        for person in persons:
            tid = person['track_id']
            active_ids.add(tid)

            if tid not in self.workers:
                self.workers[tid] = WorkerRecord(
                    worker_id=tid,
                    first_seen_frame=frame_number,
                    last_seen_frame=frame_number,
                    first_seen_time=now,
                    last_seen_time=now,
                )

            w = self.workers[tid]
            w.last_seen_frame = frame_number
            w.last_seen_time = now
            w.total_frames += 1

            # Associate PPE items to this person by spatial proximity
            person_violations = []
            person_equipment = []
            for item in ppe_items:
                # Check if PPE item overlaps with this person
                if self._overlap_ratio(person['bbox'], item['bbox']) > self._iou_threshold:
                    if item['is_violation']:
                        person_violations.append(item['class_name'])
                    elif item['class_name'] in ('Hardhat', 'Safety Vest', 'Mask'):
                        person_equipment.append(item['class_name'])

            # Update violation counters
            frame_violations = []
            if 'NO-Hardhat' in person_violations:
                w.no_hardhat_count += 1
                frame_violations.append('NO-Hardhat')
            if 'NO-Safety Vest' in person_violations:
                w.no_vest_count += 1
                frame_violations.append('NO-Safety Vest')
            if 'NO-Mask' in person_violations:
                w.no_mask_count += 1
                frame_violations.append('NO-Mask')

            # Update equipment counters
            if 'Hardhat' in person_equipment:
                w.hardhat_count += 1
            if 'Safety Vest' in person_equipment:
                w.vest_count += 1
            if 'Mask' in person_equipment:
                w.mask_count += 1

            w.current_violations = frame_violations
            w.is_compliant = len(frame_violations) == 0

            # Add log entry if there are violations (deduplicate)
            if frame_violations:
                # Only log if different from last entry or >5 seconds apart
                should_log = True
                if w.log:
                    last = w.log[-1]
                    if (set(last.get('violations', [])) == set(frame_violations)
                            and frame_number - last.get('frame', 0) < 30):
                        should_log = False

                if should_log:
                    w.log.append({
                        "frame": frame_number,
                        "violations": frame_violations,
                        "time": now,
                    })

        # Build summary
        unique_workers = len(self.workers)
        unique_violators = sum(1 for w in self.workers.values() if w.has_violations)
        active_workers = len(active_ids)
        active_violators = sum(
            1 for tid in active_ids
            if tid in self.workers and not self.workers[tid].is_compliant
        )

        return {
            "unique_workers": unique_workers,
            "unique_violators": unique_violators,
            "active_workers": active_workers,
            "active_violators": active_violators,
            "worker_records": {
                tid: self.workers[tid].to_dict()
                for tid in active_ids
                if tid in self.workers
            },
        }

    def get_all_workers(self) -> List[Dict]:
        """Get all tracked worker records"""
        return [w.to_dict() for w in sorted(self.workers.values(), key=lambda x: x.worker_id)]

    def get_worker(self, worker_id: int) -> Optional[Dict]:
        """Get a single worker record"""
        w = self.workers.get(worker_id)
        return w.to_dict() if w else None

    def get_summary(self) -> Dict:
        """Get aggregate tracking summary"""
        total = len(self.workers)
        violators = sum(1 for w in self.workers.values() if w.has_violations)
        compliant = total - violators
        return {
            "total_unique_workers": total,
            "compliant_workers": compliant,
            "violating_workers": violators,
            "compliance_rate": round(compliant / max(total, 1) * 100, 1),
            "total_violations": sum(w.violation_count for w in self.workers.values()),
        }


# Global tracker instances (one per context)
_live_tracker: Optional[WorkerTracker] = None
_video_trackers: Dict[int, WorkerTracker] = {}


def get_live_tracker() -> WorkerTracker:
    """Get or create the live monitoring tracker"""
    global _live_tracker
    if _live_tracker is None:
        _live_tracker = WorkerTracker()
    return _live_tracker


def reset_live_tracker() -> WorkerTracker:
    """Reset the live tracker for a new session"""
    global _live_tracker
    _live_tracker = WorkerTracker()
    return _live_tracker


def get_video_tracker(session_id: int) -> WorkerTracker:
    """Get or create a tracker for a video processing session"""
    if session_id not in _video_trackers:
        _video_trackers[session_id] = WorkerTracker()
    return _video_trackers[session_id]


def remove_video_tracker(session_id: int):
    """Clean up video tracker after processing"""
    _video_trackers.pop(session_id, None)
