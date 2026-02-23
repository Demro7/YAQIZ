"""
WebSocket Router
Real-time streaming for live detection and alerts.
"""

import asyncio
import cv2
import json
import logging
import platform
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from app.services.detection_service import get_detection_service
from app.services.websocket_manager import ws_manager
from app.services.telegram_service import get_telegram_service
from app.services.worker_tracker import get_live_tracker, reset_live_tracker

logger = logging.getLogger("yaqiz.ws")

router = APIRouter(tags=["WebSocket"])


def _open_camera(index: int = 0) -> cv2.VideoCapture:
    """Try multiple backends to open camera reliably (especially on Windows)"""
    backends = []
    if platform.system() == "Windows":
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "MSMF"),
            (cv2.CAP_ANY, "Any"),
        ]
    else:
        backends = [
            (cv2.CAP_V4L2, "V4L2"),
            (cv2.CAP_ANY, "Any"),
        ]

    for backend, name in backends:
        logger.info(f"Trying camera {index} with backend {name} ({backend})")
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            # Set reasonable resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            logger.info(f"Camera opened successfully with {name}")
            return cap
        cap.release()
        logger.warning(f"Failed to open camera with {name}")

    # Last fallback: plain VideoCapture(0)
    logger.info("Trying plain VideoCapture(0) as final fallback")
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        logger.info("Camera opened with default backend")
    return cap


@router.websocket("/ws/live")
async def websocket_live_feed(websocket: WebSocket, confidence: float = Query(default=0.5)):
    """WebSocket endpoint for live camera detection with tracking and structured data"""
    await ws_manager.connect(websocket, "live_feed")
    service = get_detection_service()
    tracker = reset_live_tracker()  # Fresh tracker for each session

    try:
        cap = _open_camera(0)
        if not cap.isOpened():
            logger.error("Cannot access camera with any backend")
            await websocket.send_json({"type": "error", "message": "Cannot access camera. Make sure a webcam is connected and not in use by another application."})
            await websocket.close()
            return

        frame_num = 0
        while True:
            success, frame = cap.read()
            if not success:
                await asyncio.sleep(0.01)
                continue

            frame_num += 1

            # Run detection with tracking (persistent worker IDs)
            result = service.track_frame(frame, confidence)
            tracking_data = tracker.update(result['detections'], frame_num)
            annotated = service.annotate_frame(frame, result)

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ret:
                continue

            import base64
            frame_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')

            # Send frame + detection data + tracking info
            await websocket.send_json({
                "type": "frame",
                "frame": frame_b64,
                "data": {
                    "frame_number": frame_num,
                    "total_detections": result['total_detections'],
                    "violations_count": result['violations_count'],
                    "workers_count": result['workers_count'],
                    "helmet_compliance": result['helmet_compliance'],
                    "vest_compliance": result['vest_compliance'],
                    "mask_compliance": result['mask_compliance'],
                    "detections": result['detections'][:20],
                },
                "tracking": {
                    "unique_workers": tracking_data['unique_workers'] if tracking_data else 0,
                    "unique_violators": tracking_data['unique_violators'] if tracking_data else 0,
                    "active_workers": tracking_data['active_workers'] if tracking_data else 0,
                    "active_violators": tracking_data['active_violators'] if tracking_data else 0,
                    "worker_records": tracking_data['worker_records'] if tracking_data else {},
                },
            })

            # Broadcast alerts for violations
            for v in result.get('violations', []):
                await ws_manager.send_alert({
                    "alert_type": f"missing_{v['class_name'].replace('NO-', '').lower()}",
                    "severity": "critical" if v['confidence'] > 0.8 else "warning",
                    "message": f"{v['class_name']} detected (conf: {v['confidence']:.2f})",
                    "confidence": v['confidence'],
                    "frame_number": frame_num,
                })

            # Send violation alert to Telegram (rate-limited)
            if result.get('violations'):
                try:
                    telegram = get_telegram_service()
                    sent = await telegram.send_violation_alert(result, frame_num)
                    if sent:
                        logger.info(f"Telegram violation alert sent (frame #{frame_num})")
                except Exception as e:
                    logger.warning(f"Telegram alert failed: {e}")

            # Check for client messages (e.g., confidence adjustment)
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                data = json.loads(msg)
                if data.get("type") == "set_confidence":
                    confidence = float(data["value"])
            except asyncio.TimeoutError:
                pass

            await asyncio.sleep(0.03)  # ~30fps cap

    except WebSocketDisconnect:
        logger.info("Live feed client disconnected")
    except Exception as e:
        logger.error(f"Live feed error: {e}")
    finally:
        if 'cap' in locals():
            cap.release()
        ws_manager.disconnect(websocket, "live_feed")


@router.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket endpoint for real-time alert streaming"""
    await ws_manager.connect(websocket, "alerts")
    try:
        while True:
            # Keep connection alive, alerts are pushed via broadcast
            data = await websocket.receive_text()
            # Client can send acknowledgments
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, "alerts")


@router.websocket("/ws/processing")
async def websocket_processing(websocket: WebSocket):
    """WebSocket endpoint for video processing progress"""
    await ws_manager.connect(websocket, "processing")
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, "processing")
