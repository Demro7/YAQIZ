"""
YAQIZ Workstation Router
=========================
WebSocket endpoint for real-time workstation fatigue monitoring.
Completely isolated from existing PPE WebSocket endpoints.

All persistence and broadcast logic is delegated to
``workstation_alert_bridge`` — this router is a thin controller.
"""

import base64
import json
import logging
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db, SessionLocal
from app.models.workstation import WorkstationSession
from app.services.workstation_service import get_workstation_service
from app.services.workstation_metrics import MetricsTransformer
from app.services.websocket_manager import ws_manager
from app.services import workstation_alert_bridge as bridge

logger = logging.getLogger("yaqiz.workstation.router")

router = APIRouter(tags=["Workstation"])


# ══════════════════════════════════════════════════════════
# UTILITY — extract user_id from optional JWT (WebSocket)
# ══════════════════════════════════════════════════════════

def _resolve_user_id_from_token(token: Optional[str]) -> Optional[int]:
    """
    Best-effort extraction of user_id from a JWT token supplied as a
    query parameter on the WebSocket URL.
    Returns None when the token is absent or invalid — the session simply
    proceeds without a user binding (unauthenticated guest mode).
    """
    if not token:
        return None
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        username = payload.get("sub")
        if not username:
            return None
        db = SessionLocal()
        try:
            from app.models.user import User
            user = db.query(User).filter(User.username == username).first()
            return user.id if user else None
        finally:
            db.close()
    except (JWTError, Exception):
        return None


# ══════════════════════════════════════════════════════════
# REST endpoints
# ══════════════════════════════════════════════════════════

@router.get("/api/workstation/health")
def workstation_health():
    """Check if workstation detection module is available"""
    service = get_workstation_service()
    return {
        "available": service.is_ready,
        "module": "FatigueDetectorCore (MediaPipe)",
    }


@router.get("/api/workstation/sessions")
def list_workstation_sessions(
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db),
):
    """List workstation monitoring sessions"""
    sessions = (
        db.query(WorkstationSession)
        .order_by(WorkstationSession.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return [
        {
            "id": s.id,
            "status": s.status,
            "total_frames": s.total_frames,
            "avg_fatigue_score": round(s.avg_fatigue_score, 1),
            "avg_attention_score": round(s.avg_attention_score, 1),
            "peak_fatigue_score": round(s.peak_fatigue_score, 1),
            "total_yawns": s.total_yawns,
            "total_alerts": s.total_alerts,
            "created_at": s.created_at.isoformat() if s.created_at else None,
            "ended_at": s.ended_at.isoformat() if s.ended_at else None,
        }
        for s in sessions
    ]


# ══════════════════════════════════════════════════════════
# WebSocket endpoint — production wiring via alert bridge
# ══════════════════════════════════════════════════════════

@router.websocket("/ws/workstation")
async def websocket_workstation(
    websocket: WebSocket,
    token: Optional[str] = Query(default=None),
):
    """
    Real-time workstation monitoring WebSocket.

    Protocol:
    - Client sends base64-encoded JPEG frames as JSON:
      {"type": "frame", "data": "<base64 jpeg>"}
    - Server responds with:
      {"type": "metrics", "data": {...workstation metrics...}}
    - Client can send control messages:
      {"type": "start_activity_tracker"}
      {"type": "stop_activity_tracker"}

    Optional query parameter ?token=<jwt> to bind the session to a user,
    which makes workstation alerts visible on the dashboard & Alerts Center.
    """
    await ws_manager.connect(websocket, "workstation")

    service = get_workstation_service()
    transformer = MetricsTransformer()

    # ── Resolve user identity (best-effort, non-blocking) ──
    user_id = _resolve_user_id_from_token(token)

    # Tracking for session summary
    frame_count = 0
    fatigue_sum = 0.0
    attention_sum = 0.0
    peak_fatigue = 0.0
    max_yawns = 0
    max_blinks = 0
    max_eye_rubs = 0
    alert_count = 0

    # ── Create sessions via bridge ──
    bridge_session_id = bridge.create_bridge_session(user_id)
    ws_session_id = bridge.create_workstation_session(user_id)

    logger.info(
        "Workstation WebSocket connected "
        "(ws_session=%s, bridge_session=%s, user_id=%s)",
        ws_session_id, bridge_session_id, user_id,
    )

    if not service.is_ready:
        await websocket.send_json({
            "type": "error",
            "message": "Workstation detection module not available. "
                       "Ensure mediapipe is installed.",
        })
        ws_manager.disconnect(websocket, "workstation")
        return

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type", "")

            # ── Control messages ──
            if msg_type == "start_activity_tracker":
                service.start_activity_tracker()
                await websocket.send_json({"type": "info", "message": "Activity tracker started"})
                continue

            if msg_type == "stop_activity_tracker":
                service.stop_activity_tracker()
                await websocket.send_json({"type": "info", "message": "Activity tracker stopped"})
                continue

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            # ── Frame processing ──
            if msg_type == "frame":
                frame_b64 = msg.get("data", "")
                if not frame_b64:
                    continue

                # Decode base64 JPEG → numpy array
                try:
                    img_bytes = base64.b64decode(frame_b64)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                except Exception:
                    continue

                # Process frame (offloaded to thread pool)
                stats_dict = await service.process_frame(frame)

                # Transform into YAQIZ metrics
                metrics = transformer.transform(stats_dict)

                frame_count += 1
                fatigue_sum += metrics.fatigue_score
                attention_sum += metrics.attention_score
                peak_fatigue = max(peak_fatigue, metrics.fatigue_score)
                max_yawns = max(max_yawns, metrics.yawn_count)
                max_blinks = max(max_blinks, metrics.blink_rate)
                max_eye_rubs = max(max_eye_rubs, metrics.eye_rub_count)

                # ── Persist + broadcast alerts via bridge ──
                if metrics.alerts:
                    persisted = await bridge.persist_and_broadcast(
                        metrics.alerts, bridge_session_id,
                    )
                    alert_count += persisted

                # Send metrics back to client
                await websocket.send_json({
                    "type": "metrics",
                    "data": metrics.to_dict(),
                    "session_id": ws_session_id,
                    "frame_number": frame_count,
                })

    except WebSocketDisconnect:
        logger.info(
            "Workstation client disconnected "
            "(ws_session=%s, frames=%d)",
            ws_session_id, frame_count,
        )
    except Exception as e:
        logger.error("Workstation WS error: %s", e, exc_info=True)
    finally:
        ws_manager.disconnect(websocket, "workstation")
        # Finalise both session records via bridge
        bridge.finalise_sessions(
            ws_session_id=ws_session_id,
            bridge_session_id=bridge_session_id,
            frame_count=frame_count,
            fatigue_sum=fatigue_sum,
            attention_sum=attention_sum,
            peak_fatigue=peak_fatigue,
            max_yawns=max_yawns,
            max_blinks=max_blinks,
            max_eye_rubs=max_eye_rubs,
            alert_count=alert_count,
        )
