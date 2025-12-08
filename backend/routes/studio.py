"""Studio API routes for TikTok Studio automation."""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional

from backend.models.schemas import StudioRequest
from backend.services.studio_service import StudioService
from backend.utils.progress_manager import progress_manager

router = APIRouter(prefix="/studio", tags=["studio"])

# Service instance
_studio_service: Optional[StudioService] = None


def get_studio_service() -> StudioService:
    """Dependency to get studio service instance."""
    global _studio_service
    if _studio_service is None:
        _studio_service = StudioService(progress_manager)
    return _studio_service


@router.post("/start", response_model=dict)
async def start_studio_session(
    request: StudioRequest,
    service: StudioService = Depends(get_studio_service)
):
    """
    Start a TikTok Studio scraping session.

    This will launch a browser, attempt login, and begin scraping analytics.
    Returns session_id for tracking progress via WebSocket.
    """
    session_id = await service.start_session(request)
    return {
        "session_id": session_id,
        "status": "started",
        "message": "Studio session started. Watch for login_required event if manual login needed."
    }


@router.get("/status/{session_id}", response_model=dict)
async def get_studio_status(
    session_id: str,
    service: StudioService = Depends(get_studio_service)
):
    """Get status of a Studio scraping session."""
    status = service.get_session_status(session_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return status


@router.post("/continue/{session_id}", response_model=dict)
async def continue_after_login(
    session_id: str,
    service: StudioService = Depends(get_studio_service)
):
    """
    Continue a Studio session after user has completed manual login.

    Call this endpoint after receiving studio_login_required event
    and user has logged in via the browser window.
    """
    success = await service.continue_after_login(session_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Session not found or not awaiting login"
        )
    return {"session_id": session_id, "status": "continuing"}


@router.post("/stop/{session_id}", response_model=dict)
async def stop_studio_session(
    session_id: str,
    service: StudioService = Depends(get_studio_service)
):
    """Stop a running Studio scraping session."""
    success = await service.stop_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return {"session_id": session_id, "status": "stopped"}
