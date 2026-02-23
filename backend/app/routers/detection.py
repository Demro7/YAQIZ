"""
Detection Router
Handles video upload, image upload, and detection sessions.
"""

import os
import uuid
import asyncio
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, Query
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from typing import Optional

from app.core.database import get_db
from app.core.config import settings
from app.core.security import get_current_user
from app.models.user import User
from app.models.detection import DetectionSession, Alert
from app.models.schemas import SessionResponse, AlertCreate
from app.services.detection_service import get_detection_service
from app.services.websocket_manager import ws_manager
from app.services.telegram_service import get_telegram_service
from app.services.worker_tracker import get_video_tracker, remove_video_tracker, get_live_tracker
import cv2
import logging

logger = logging.getLogger("yaqiz.detection_router")

router = APIRouter(prefix="/api/detection", tags=["Detection"])


def _save_upload(upload: UploadFile, subdir: str) -> str:
    """Save uploaded file and return path"""
    ext = Path(upload.filename).suffix or ".mp4"
    filename = f"{uuid.uuid4().hex}{ext}"
    dest = settings.upload_path / subdir
    dest.mkdir(parents=True, exist_ok=True)
    filepath = dest / filename
    with open(filepath, "wb") as f:
        content = upload.file.read()
        f.write(content)
    return str(filepath)


async def _process_video_task(session_id: int, input_path: str, output_path: str, confidence: float, frame_skip: int = 3):
    """Background task for video processing with worker tracking"""
    from app.core.database import SessionLocal
    db = SessionLocal()
    try:
        session = db.query(DetectionSession).filter(DetectionSession.id == session_id).first()
        if not session:
            logger.error(f"Session {session_id} not found")
            return
        session.status = "processing"
        db.commit()
        logger.info(f"Video processing started for session {session_id}")

        service = get_detection_service()
        tracker = get_video_tracker(session_id)

        # Run sync processing in thread
        loop = asyncio.get_running_loop()
        summary = await loop.run_in_executor(
            None, service.process_video_file, input_path, output_path, confidence, None, frame_skip, tracker
        )

        # Update session
        session.status = "completed"
        session.total_frames = summary['total_frames']
        session.processed_frames = summary['total_frames']
        session.total_detections = summary['total_detections']
        session.violations_count = summary['violations_count']
        session.compliance_rate = summary['compliance_rate']
        session.summary = summary
        session.result_file = output_path
        session.completed_at = datetime.utcnow()
        db.commit()

        # Store alerts
        for alert_data in summary.get('alerts', [])[:50]:
            alert = Alert(
                alert_type=alert_data['alert_type'],
                severity=alert_data['severity'],
                message=alert_data['message'],
                confidence=alert_data['confidence'],
                frame_number=alert_data.get('frame_number'),
                session_id=session_id,
            )
            db.add(alert)
        db.commit()

        # Notify via WebSocket
        await ws_manager.send_progress(session_id, 100, {"status": "completed", "summary": summary})

        logger.info(f"Video processing completed for session {session_id}: "
                     f"{summary['total_frames']} frames, {summary['violations_count']} violations")

        # Send summary to Telegram
        try:
            telegram = get_telegram_service()
            await telegram.send_video_summary(summary, session_id)
        except Exception as e:
            logger.warning(f"Telegram summary failed: {e}")

    except Exception as e:
        logger.error(f"Video processing error for session {session_id}: {e}", exc_info=True)
        try:
            session = db.query(DetectionSession).filter(DetectionSession.id == session_id).first()
            if session:
                session.status = "failed"
                db.commit()
        except Exception as db_err:
            logger.error(f"Failed to update session status: {db_err}")
    finally:
        remove_video_tracker(session_id)
        db.close()


@router.post("/upload-video", response_model=SessionResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    confidence: float = Query(default=0.5, ge=0.1, le=1.0),
    frame_skip: int = Query(default=3, ge=1, le=30, description="Process every Nth frame (1=no skip, 3=3x faster)"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Upload a video for PPE detection processing"""
    if not file.filename:
        raise HTTPException(400, "No file provided")

    # Save upload
    input_path = _save_upload(file, "videos")

    # Create output path
    output_filename = f"result_{uuid.uuid4().hex}.mp4"
    output_path = str(settings.results_path / output_filename)

    # Create session record
    session = DetectionSession(
        session_type="video",
        source_file=input_path,
        result_file=output_path,
        status="pending",
        user_id=current_user.id,
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    # Start background processing with frame skipping
    background_tasks.add_task(_process_video_task, session.id, input_path, output_path, confidence, frame_skip)

    return SessionResponse.model_validate(session)


@router.post("/upload-image")
async def upload_image(
    file: UploadFile = File(...),
    confidence: float = Query(default=0.5, ge=0.1, le=1.0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Upload an image for PPE detection"""
    if not file.filename:
        raise HTTPException(400, "No file provided")

    input_path = _save_upload(file, "images")
    service = get_detection_service()

    try:
        annotated, result = service.process_image(input_path, confidence)

        # Save result image
        result_filename = f"result_{uuid.uuid4().hex}.jpg"
        result_path = str(settings.results_path / result_filename)
        cv2.imwrite(result_path, annotated)

        # Create session record
        session = DetectionSession(
            session_type="image",
            source_file=input_path,
            result_file=result_path,
            total_detections=result['total_detections'],
            violations_count=result['violations_count'],
            compliance_rate=(100 - (result['violations_count'] / max(result['total_detections'], 1) * 100)),
            status="completed",
            summary=result,
            user_id=current_user.id,
            completed_at=datetime.utcnow(),
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        # Store alerts for violations
        for v in result.get('violations', []):
            alert = Alert(
                alert_type=f"missing_{v['class_name'].replace('NO-', '').lower()}",
                severity='critical' if v['confidence'] > 0.8 else 'warning',
                message=f"{v['class_name']} detected (conf: {v['confidence']:.2f})",
                confidence=v['confidence'],
                session_id=session.id,
            )
            db.add(alert)
        db.commit()

        # Send violation alert to Telegram for image analysis
        if result.get('violations'):
            try:
                telegram = get_telegram_service()
                await telegram.send_violation_alert(result)
            except Exception as e:
                logger.warning(f"Telegram image alert failed: {e}")

        return {
            "session_id": session.id,
            "result_image": f"/api/detection/result/{result_filename}",
            "detections": result['detections'],
            "total_detections": result['total_detections'],
            "violations_count": result['violations_count'],
            "helmet_compliance": result['helmet_compliance'],
            "vest_compliance": result['vest_compliance'],
            "mask_compliance": result['mask_compliance'],
            "workers_count": result['workers_count'],
        }
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        raise HTTPException(500, f"Processing error: {str(e)}")


@router.get("/sessions")
def list_sessions(
    skip: int = 0,
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List detection sessions"""
    sessions = (
        db.query(DetectionSession)
        .filter(DetectionSession.user_id == current_user.id)
        .order_by(DetectionSession.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return [SessionResponse.model_validate(s) for s in sessions]


@router.get("/sessions/{session_id}")
def get_session(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get session details"""
    session = db.query(DetectionSession).filter(
        DetectionSession.id == session_id,
        DetectionSession.user_id == current_user.id,
    ).first()
    if not session:
        raise HTTPException(404, "Session not found")
    return {
        "session": SessionResponse.model_validate(session),
        "summary": session.summary,
    }


@router.get("/result/{filename}")
def get_result_file(filename: str):
    """Serve result files (images/videos)"""
    filepath = settings.results_path / filename
    if not filepath.exists():
        raise HTTPException(404, "Result file not found")
    return FileResponse(str(filepath))


@router.get("/live-feed")
def live_camera_feed(
    confidence: float = Query(default=0.5, ge=0.1, le=1.0),
    frame_skip: int = Query(default=2, ge=1, le=15, description="Process every Nth frame"),
):
    """Stream webcam feed with YOLO detection + tracking as MJPEG"""
    service = get_detection_service()
    tracker = get_live_tracker()

    def generate():
        for annotated, result, tracking_data in service.process_video_stream(
            0, confidence, frame_skip=frame_skip, tracker=tracker
        ):
            ret, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return StreamingResponse(
        generate(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )


@router.get("/workers")
def get_tracked_workers():
    """Get all tracked workers from live monitoring session"""
    tracker = get_live_tracker()
    return {
        "summary": tracker.get_summary(),
        "workers": tracker.get_all_workers(),
    }


@router.get("/workers/{worker_id}")
def get_worker_log(worker_id: int):
    """Get detailed log for a specific tracked worker"""
    tracker = get_live_tracker()
    record = tracker.get_worker(worker_id)
    if not record:
        raise HTTPException(404, f"Worker #{worker_id} not found")
    return record
