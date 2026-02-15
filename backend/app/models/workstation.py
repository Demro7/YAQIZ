"""
Workstation Monitoring Models
Additive DB model for workstation sessions — does NOT modify existing detection models.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, ForeignKey
from datetime import datetime
from app.core.database import Base


class WorkstationSession(Base):
    """Tracks a workstation monitoring session (one per WebSocket connection)."""
    __tablename__ = "workstation_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    status = Column(String(20), default="active")  # active, ended
    total_frames = Column(Integer, default=0)
    avg_fatigue_score = Column(Float, default=0.0)
    avg_attention_score = Column(Float, default=100.0)
    peak_fatigue_score = Column(Float, default=0.0)
    total_yawns = Column(Integer, default=0)
    total_blinks = Column(Integer, default=0)
    total_eye_rubs = Column(Integer, default=0)
    total_alerts = Column(Integer, default=0)
    absence_seconds = Column(Float, default=0.0)
    idle_seconds = Column(Float, default=0.0)
    summary = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
