"""
Pydantic Schemas for API request/response validation
"""

from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime


# ── Auth Schemas ─────────────────────────────────────────
class UserRegister(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str] = None
    role: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# ── Detection Schemas ────────────────────────────────────
class DetectionResult(BaseModel):
    class_name: str
    confidence: float
    bbox: List[int]
    is_violation: bool


class FrameAnalysis(BaseModel):
    frame_number: int
    detections: List[DetectionResult]
    workers_count: int
    helmet_compliance: float
    vest_compliance: float
    mask_compliance: float
    violations: List[str]


class SessionResponse(BaseModel):
    id: int
    session_type: str
    status: str
    total_detections: int
    violations_count: int
    compliance_rate: float
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class SessionSummary(BaseModel):
    total_frames: int
    total_detections: int
    violations_count: int
    compliance_rate: float
    helmet_compliance: float
    vest_compliance: float
    workers_detected: int
    alerts_generated: int


# ── Alert Schemas ────────────────────────────────────────
class AlertResponse(BaseModel):
    id: int
    alert_type: str
    severity: str
    message: str
    confidence: float
    frame_number: Optional[int] = None
    is_read: bool
    created_at: datetime

    class Config:
        from_attributes = True


class AlertCreate(BaseModel):
    alert_type: str
    severity: str
    message: str
    confidence: float
    frame_number: Optional[int] = None
    session_id: Optional[int] = None


# ── Dashboard Schemas ────────────────────────────────────
class DashboardStats(BaseModel):
    total_sessions: int
    total_detections: int
    total_violations: int
    overall_compliance: float
    active_alerts: int
    sessions_today: int
    recent_sessions: List[SessionResponse]
    recent_alerts: List[AlertResponse]
