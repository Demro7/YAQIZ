"""
YAQIZ WebSocket Manager
Handles real-time streaming of detection results and alerts.
"""

import asyncio
import json
import logging
from typing import Dict, List, Set
from fastapi import WebSocket

logger = logging.getLogger("yaqiz.websocket")


class ConnectionManager:
    """Manages WebSocket connections for real-time streaming"""

    def __init__(self):
        # Channel-based connections
        self.active_connections: Dict[str, Set[WebSocket]] = {
            "live_feed": set(),
            "alerts": set(),
            "processing": set(),
            "workstation": set(),
        }

    async def connect(self, websocket: WebSocket, channel: str = "live_feed"):
        await websocket.accept()
        if channel not in self.active_connections:
            self.active_connections[channel] = set()
        self.active_connections[channel].add(websocket)
        logger.info(f"Client connected to channel: {channel} (total: {len(self.active_connections[channel])})")

    def disconnect(self, websocket: WebSocket, channel: str = "live_feed"):
        if channel in self.active_connections:
            self.active_connections[channel].discard(websocket)
        logger.info(f"Client disconnected from channel: {channel}")

    async def broadcast_to_channel(self, channel: str, message: dict):
        """Send message to all clients on a channel"""
        if channel not in self.active_connections:
            return

        dead = set()
        for ws in self.active_connections[channel]:
            try:
                await ws.send_json(message)
            except Exception:
                dead.add(ws)

        for ws in dead:
            self.active_connections[channel].discard(ws)

    async def send_alert(self, alert: dict):
        """Broadcast alert to alert channel"""
        await self.broadcast_to_channel("alerts", {
            "type": "alert",
            "data": alert
        })

    async def send_detection_update(self, result: dict):
        """Send detection result to live feed channel"""
        await self.broadcast_to_channel("live_feed", {
            "type": "detection",
            "data": result
        })

    async def send_progress(self, session_id: int, progress: float, details: dict = None):
        """Send processing progress update"""
        await self.broadcast_to_channel("processing", {
            "type": "progress",
            "data": {
                "session_id": session_id,
                "progress": progress,
                **(details or {})
            }
        })

    @property
    def total_connections(self) -> int:
        return sum(len(conns) for conns in self.active_connections.values())


# Global instance
ws_manager = ConnectionManager()
