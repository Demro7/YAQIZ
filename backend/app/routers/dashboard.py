"""
Dashboard Router
Provides aggregated stats for the executive dashboard.
"""

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from datetime import datetime, timedelta

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.models.detection import DetectionSession, Alert
from app.models.schemas import DashboardStats, SessionResponse, AlertResponse

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])


@router.get("/stats", response_model=DashboardStats)
def get_dashboard_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get executive dashboard statistics"""
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    total_sessions = db.query(func.count(DetectionSession.id)).filter(
        DetectionSession.user_id == current_user.id
    ).scalar() or 0

    total_detections = db.query(func.sum(DetectionSession.total_detections)).filter(
        DetectionSession.user_id == current_user.id
    ).scalar() or 0

    total_violations = db.query(func.sum(DetectionSession.violations_count)).filter(
        DetectionSession.user_id == current_user.id
    ).scalar() or 0

    avg_compliance = db.query(func.avg(DetectionSession.compliance_rate)).filter(
        DetectionSession.user_id == current_user.id,
        DetectionSession.status == "completed",
    ).scalar() or 100.0

    active_alerts = db.query(func.count(Alert.id)).filter(
        Alert.is_read == False,
        Alert.session_id.in_(
            db.query(DetectionSession.id).filter(DetectionSession.user_id == current_user.id)
        )
    ).scalar() or 0

    sessions_today = db.query(func.count(DetectionSession.id)).filter(
        DetectionSession.user_id == current_user.id,
        DetectionSession.created_at >= today,
    ).scalar() or 0

    recent_sessions = (
        db.query(DetectionSession)
        .filter(DetectionSession.user_id == current_user.id)
        .order_by(desc(DetectionSession.created_at))
        .limit(5)
        .all()
    )

    recent_alerts = (
        db.query(Alert)
        .filter(
            Alert.session_id.in_(
                db.query(DetectionSession.id).filter(DetectionSession.user_id == current_user.id)
            )
        )
        .order_by(desc(Alert.created_at))
        .limit(10)
        .all()
    )

    return DashboardStats(
        total_sessions=total_sessions,
        total_detections=int(total_detections),
        total_violations=int(total_violations),
        overall_compliance=round(float(avg_compliance), 1),
        active_alerts=active_alerts,
        sessions_today=sessions_today,
        recent_sessions=[SessionResponse.model_validate(s) for s in recent_sessions],
        recent_alerts=[AlertResponse.model_validate(a) for a in recent_alerts],
    )


@router.get("/alerts")
def get_alerts(
    skip: int = 0,
    limit: int = 50,
    severity: str = None,
    unread_only: bool = False,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get alerts with filtering"""
    query = db.query(Alert).filter(
        Alert.session_id.in_(
            db.query(DetectionSession.id).filter(DetectionSession.user_id == current_user.id)
        )
    )

    if severity:
        query = query.filter(Alert.severity == severity)
    if unread_only:
        query = query.filter(Alert.is_read == False)

    alerts = query.order_by(desc(Alert.created_at)).offset(skip).limit(limit).all()
    total = query.count()

    return {
        "alerts": [AlertResponse.model_validate(a) for a in alerts],
        "total": total,
    }


@router.put("/alerts/{alert_id}/read")
def mark_alert_read(
    alert_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Mark alert as read"""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(404, "Alert not found")
    alert.is_read = True
    db.commit()
    return {"message": "Alert marked as read"}


@router.put("/alerts/mark-all-read")
def mark_all_read(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Mark all alerts as read"""
    db.query(Alert).filter(
        Alert.is_read == False,
        Alert.session_id.in_(
            db.query(DetectionSession.id).filter(DetectionSession.user_id == current_user.id)
        )
    ).update({Alert.is_read: True}, synchronize_session=False)
    db.commit()
    return {"message": "All alerts marked as read"}
