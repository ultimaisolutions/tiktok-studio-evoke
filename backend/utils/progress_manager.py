"""WebSocket connection and progress broadcast manager."""

import asyncio
import uuid
from typing import Dict, Set, Optional, Any
from datetime import datetime
from fastapi import WebSocket
import logging

logger = logging.getLogger(__name__)


class ProgressManager:
    """Manages WebSocket connections and broadcasts progress updates."""

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.job_states: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, job_id: str) -> None:
        """Accept WebSocket connection and register it for a job."""
        await websocket.accept()
        async with self._lock:
            if job_id not in self.active_connections:
                self.active_connections[job_id] = set()
            self.active_connections[job_id].add(websocket)

        # Send current state if exists
        if job_id in self.job_states:
            try:
                await websocket.send_json(self.job_states[job_id])
            except Exception as e:
                logger.warning(f"Failed to send initial state: {e}")

        logger.info(f"WebSocket connected for job {job_id}")

    async def disconnect(self, websocket: WebSocket, job_id: str) -> None:
        """Remove WebSocket connection for a job."""
        async with self._lock:
            if job_id in self.active_connections:
                self.active_connections[job_id].discard(websocket)
                if not self.active_connections[job_id]:
                    del self.active_connections[job_id]
        logger.info(f"WebSocket disconnected for job {job_id}")

    async def broadcast(
        self,
        job_id: str,
        event_type: str,
        data: Dict[str, Any],
        store_state: bool = True
    ) -> None:
        """Broadcast progress update to all connected clients for a job."""
        message = {
            "type": "progress",
            "job_id": job_id,
            "event": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }

        # Store state for new connections
        if store_state:
            self.job_states[job_id] = message

        # Broadcast to all connections
        if job_id in self.active_connections:
            disconnected = set()
            for connection in self.active_connections[job_id].copy():
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.warning(f"Failed to send to WebSocket: {e}")
                    disconnected.add(connection)

            # Clean up disconnected clients
            if disconnected:
                async with self._lock:
                    for conn in disconnected:
                        self.active_connections[job_id].discard(conn)

    def create_job(self, job_type: str) -> str:
        """Create a new job and return its ID."""
        job_id = str(uuid.uuid4())[:8]
        self.job_states[job_id] = {
            "type": "progress",
            "job_id": job_id,
            "event": "job_created",
            "data": {
                "status": "pending",
                "job_type": job_type,
                "progress": 0,
                "total": 0,
                "completed": 0,
                "failed": 0,
                "errors": []
            },
            "timestamp": datetime.now().isoformat()
        }
        logger.info(f"Created job {job_id} of type {job_type}")
        return job_id

    def get_job_state(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get current state of a job."""
        return self.job_states.get(job_id)

    def update_job_state(self, job_id: str, **kwargs) -> None:
        """Update job state without broadcasting."""
        if job_id in self.job_states:
            self.job_states[job_id]["data"].update(kwargs)
            self.job_states[job_id]["timestamp"] = datetime.now().isoformat()

    async def job_started(
        self,
        job_id: str,
        job_type: str,
        total: int
    ) -> None:
        """Broadcast job started event."""
        await self.broadcast(job_id, "job_started", {
            "status": "running",
            "job_type": job_type,
            "total": total,
            "completed": 0,
            "failed": 0,
            "progress": 0
        })

    async def job_progress(
        self,
        job_id: str,
        completed: int,
        total: int,
        current_task: Optional[str] = None,
        failed: int = 0,
        **kwargs
    ) -> None:
        """Broadcast job progress update."""
        progress = completed / total if total > 0 else 0
        data = {
            "status": "running",
            "completed": completed,
            "total": total,
            "failed": failed,
            "progress": progress,
            "current_task": current_task
        }
        # Add any extra fields (workers_active, current_videos, downloaded, analyzed, etc.)
        data.update(kwargs)
        await self.broadcast(job_id, "job_progress", data)

    async def job_completed(
        self,
        job_id: str,
        result: Optional[Dict[str, Any]] = None
    ) -> None:
        """Broadcast job completed event."""
        await self.broadcast(job_id, "job_completed", {
            "status": "completed",
            "progress": 1.0,
            "result": result
        })

    async def job_failed(
        self,
        job_id: str,
        error: str
    ) -> None:
        """Broadcast job failed event."""
        current_state = self.job_states.get(job_id, {}).get("data", {})
        errors = current_state.get("errors", [])
        errors.append(error)
        await self.broadcast(job_id, "job_failed", {
            "status": "failed",
            "error": error,
            "errors": errors
        })

    async def download_start(
        self,
        job_id: str,
        url: str,
        index: int,
        total: int
    ) -> None:
        """Broadcast download start event."""
        await self.broadcast(job_id, "download_start", {
            "url": url,
            "index": index,
            "total": total,
            "status": "downloading"
        }, store_state=False)

    async def download_complete(
        self,
        job_id: str,
        url: str,
        success: bool,
        path: Optional[str] = None,
        message: Optional[str] = None
    ) -> None:
        """Broadcast download complete event."""
        await self.broadcast(job_id, "download_complete", {
            "url": url,
            "success": success,
            "path": path,
            "message": message
        }, store_state=False)

    async def analysis_start(
        self,
        job_id: str,
        video_path: str,
        index: int,
        total: int
    ) -> None:
        """Broadcast analysis start event."""
        await self.broadcast(job_id, "analysis_start", {
            "video_path": video_path,
            "index": index,
            "total": total
        }, store_state=False)

    async def analysis_complete(
        self,
        job_id: str,
        video_path: str,
        success: bool,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Broadcast analysis complete event."""
        await self.broadcast(job_id, "analysis_complete", {
            "video_path": video_path,
            "success": success,
            "metrics": metrics
        }, store_state=False)

    async def studio_login_required(self, job_id: str) -> None:
        """Broadcast that manual Studio login is required."""
        await self.broadcast(job_id, "studio_login_required", {
            "status": "awaiting_login",
            "instructions": "Please log in to TikTok Studio in the browser window that opened."
        })

    async def studio_video_found(
        self,
        job_id: str,
        video_id: str,
        index: int,
        total: Optional[int] = None
    ) -> None:
        """Broadcast that a video was found in Studio."""
        await self.broadcast(job_id, "studio_video_found", {
            "video_id": video_id,
            "index": index,
            "total": total
        }, store_state=False)

    async def studio_screenshot(
        self,
        job_id: str,
        video_id: str,
        tab: str,
        path: str
    ) -> None:
        """Broadcast that a screenshot was captured."""
        await self.broadcast(job_id, "studio_screenshot", {
            "video_id": video_id,
            "tab": tab,
            "path": path
        }, store_state=False)

    def cleanup_job(self, job_id: str) -> None:
        """Remove job state and connections."""
        if job_id in self.job_states:
            del self.job_states[job_id]
        if job_id in self.active_connections:
            del self.active_connections[job_id]
        logger.info(f"Cleaned up job {job_id}")


# Global instance
progress_manager = ProgressManager()
