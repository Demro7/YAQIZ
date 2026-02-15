"""
YAQIZ - FastAPI Application Entry Point
Production-grade PPE Detection AI Platform
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.database import init_db
from app.utils.logger import setup_logging

# Setup logging
setup_logging("DEBUG" if settings.DEBUG else "INFO")
logger = logging.getLogger("yaqiz.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager"""
    logger.info("=" * 60)
    logger.info("  YAQIZ PPE Detection Platform - Starting")
    logger.info("=" * 60)

    # Initialize database
    init_db()
    # Ensure workstation model tables exist
    from app.models.workstation import WorkstationSession  # noqa: F401
    init_db()
    logger.info("Database initialized")

    # Pre-load YOLO model
    try:
        from app.services.detection_service import get_detection_service
        get_detection_service()
        logger.info("YOLO model loaded successfully")
    except Exception as e:
        logger.warning(f"Model pre-load failed (will load on first request): {e}")

    logger.info(f"Environment: {settings.YAQIZ_ENV}")
    logger.info(f"CORS Origins: {settings.cors_origins_list}")
    logger.info("YAQIZ is ready!")
    logger.info("=" * 60)

    yield

    logger.info("YAQIZ shutting down...")


# Create FastAPI app
app = FastAPI(
    title="YAQIZ - PPE Detection AI Platform",
    description="Intelligent PPE Safety Monitoring System powered by YOLOv8",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static result files
settings.results_path.mkdir(parents=True, exist_ok=True)
app.mount("/results", StaticFiles(directory=str(settings.results_path)), name="results")

# Include routers
from app.routers import auth, detection, dashboard, websocket, workstation

app.include_router(auth.router)
app.include_router(detection.router)
app.include_router(dashboard.router)
app.include_router(websocket.router)
app.include_router(workstation.router)


# Health check endpoint
@app.get("/health")
def health_check():
    import torch
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    return {
        "status": "healthy",
        "service": "YAQIZ",
        "version": "1.0.0",
        "device": "cuda" if gpu_available else "cpu",
        "gpu": gpu_name,
    }


@app.get("/api/info")
def api_info():
    return {
        "name": "YAQIZ API",
        "version": "1.0.0",
        "description": "PPE Detection AI Platform",
        "endpoints": {
            "auth": "/api/auth",
            "detection": "/api/detection",
            "dashboard": "/api/dashboard",
            "workstation": "/api/workstation",
            "websocket_live": "/ws/live",
            "websocket_alerts": "/ws/alerts",
            "websocket_workstation": "/ws/workstation",
            "health": "/health",
        }
    }
