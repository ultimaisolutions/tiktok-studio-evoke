"""WebSocket routes for real-time progress updates."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import logging

from backend.utils.progress_manager import progress_manager

router = APIRouter(tags=["websocket"])
logger = logging.getLogger(__name__)


@router.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for receiving real-time job progress updates.

    Connect to /ws/{job_id} after starting a download, studio, or analysis job.

    Messages received will have format:
    {
        "type": "progress",
        "job_id": "abc123",
        "event": "download_progress",
        "data": {...},
        "timestamp": "2024-01-01T12:00:00.000Z"
    }

    Event types:
    - job_started: Job has begun processing
    - job_progress: Progress update with completed/total counts
    - job_completed: Job finished successfully
    - job_failed: Job failed with error
    - download_start: Starting to download a URL
    - download_complete: Finished downloading a URL
    - analysis_start: Starting to analyze a video
    - analysis_complete: Finished analyzing a video
    - studio_login_required: Manual login needed
    - studio_video_found: Found a video in Studio
    - studio_screenshot: Captured a screenshot
    """
    await progress_manager.connect(websocket, job_id)

    try:
        while True:
            # Keep connection alive, receive any client messages
            data = await websocket.receive_text()
            # Client can send ping/pong or other messages
            if data == "ping":
                await websocket.send_text("pong")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for job {job_id}")
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
    finally:
        await progress_manager.disconnect(websocket, job_id)
