"""
Detection & Alert Models
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, ForeignKey
from datetime import datetime
from app.core.database import Base


class DetectionSession(Base):
    __tablename__ = "detection_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_type = Column(String(20), nullable=False)  # "video", "image", "live"
    source_file = Column(String(255), nullable=True)
    result_file = Column(String(255), nullable=True)
    total_frames = Column(Integer, default=0)
    processed_frames = Column(Integer, default=0)
    total_detections = Column(Integer, default=0)
    violations_count = Column(Integer, default=0)
    compliance_rate = Column(Float, default=0.0)
    status = Column(String(20), default="pending")  # pending, processing, completed, failed
    summary = Column(JSON, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    alert_type = Column(String(50), nullable=False)  # missing_helmet, missing_vest, missing_mask
    severity = Column(String(20), default="warning")  # critical, warning, info
    message = Column(String(500), nullable=False)
    confidence = Column(Float, default=0.0)
    frame_number = Column(Integer, nullable=True)
    session_id = Column(Integer, ForeignKey("detection_sessions.id"), nullable=True)
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
