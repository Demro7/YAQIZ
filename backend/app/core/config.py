"""
YAQIZ Configuration
Central configuration loaded from environment variables.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # App
    APP_NAME: str = "YAQIZ"
    YAQIZ_ENV: str = "development"
    DEBUG: bool = True

    # Auth
    SECRET_KEY: str = "change-me-in-dotenv"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours

    # YOLO
    YOLO_WEIGHTS_PATH: str = "../YOLO-Weights/ppe.pt"
    YOLO_CONFIDENCE: float = 0.5
    YOLO_FRAME_SKIP: int = 3      # Process every Nth frame (1=no skip)

    # Database
    DATABASE_URL: str = "sqlite:///./yaqiz.db"

    # CORS
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000,http://localhost:8000"

    # Telegram Notifications
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""
    TELEGRAM_ENABLED: bool = False

    # Uploads
    MAX_UPLOAD_SIZE_MB: int = 500
    UPLOAD_DIR: str = "uploads"
    RESULTS_DIR: str = "results"

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",")]

    @property
    def base_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent

    @property
    def weights_path(self) -> Path:
        p = Path(self.YOLO_WEIGHTS_PATH)
        if not p.is_absolute():
            p = self.base_dir / p
        return p

    @property
    def upload_path(self) -> Path:
        p = self.base_dir / self.UPLOAD_DIR
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def results_path(self) -> Path:
        p = self.base_dir / self.RESULTS_DIR
        p.mkdir(parents=True, exist_ok=True)
        return p

    class Config:
        env_file = Path(__file__).resolve().parent.parent.parent / ".env"
        extra = "allow"


settings = Settings()
