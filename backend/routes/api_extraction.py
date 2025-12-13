"""API Extraction routes for capturing TikTok API patterns."""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional

from backend.models.schemas import ExtractionRequest
from backend.services.api_extraction_service import APIExtractionService
from backend.utils.progress_manager import progress_manager

router = APIRouter(prefix="/extractor", tags=["extractor"])

# Service instance
_extraction_service: Optional[APIExtractionService] = None


def get_extraction_service() -> APIExtractionService:
    """Dependency to get extraction service instance."""
    global _extraction_service
    if _extraction_service is None:
        _extraction_service = APIExtractionService(progress_manager)
    return _extraction_service


@router.post("/start", response_model=dict)
async def start_extraction(
    request: ExtractionRequest,
    service: APIExtractionService = Depends(get_extraction_service)
):
    """
    Start an API pattern extraction session.

    This will launch a browser, navigate to TikTok Studio pages,
    and capture XHR/Fetch network traffic to extract API patterns.
    Returns session_id for tracking progress via WebSocket.
    """
    session_id = await service.start_extraction(request)
    return {
        "session_id": session_id,
        "status": "started",
        "message": "Extraction session started. Watch for login_required event if manual login needed."
    }


@router.get("/status/{session_id}", response_model=dict)
async def get_extraction_status(
    session_id: str,
    service: APIExtractionService = Depends(get_extraction_service)
):
    """Get status of an extraction session."""
    status = service.get_session_status(session_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return status


@router.post("/continue/{session_id}", response_model=dict)
async def continue_after_login(
    session_id: str,
    service: APIExtractionService = Depends(get_extraction_service)
):
    """
    Continue an extraction session after user has completed manual login.

    Call this endpoint after receiving extraction_login_required event
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
async def stop_extraction_session(
    session_id: str,
    service: APIExtractionService = Depends(get_extraction_service)
):
    """Stop a running extraction session."""
    success = await service.stop_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return {"session_id": session_id, "status": "stopped"}


@router.get("/patterns", response_model=dict)
async def get_patterns(
    service: APIExtractionService = Depends(get_extraction_service)
):
    """
    Get saved API patterns.

    Returns the extracted API patterns including video list API,
    analytics API, and other discovered endpoints.
    """
    patterns = service.get_patterns()
    if patterns is None:
        return {
            "found": False,
            "message": "No patterns have been extracted yet. Run an extraction first."
        }
    return {
        "found": True,
        "patterns": patterns
    }


@router.post("/apply", response_model=dict)
async def apply_patterns(
    service: APIExtractionService = Depends(get_extraction_service)
):
    """
    Apply saved patterns to the scraper.

    The studio scraper will use these patterns for more reliable
    API requests that are resilient to UI changes.
    """
    result = service.apply_patterns()
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["message"])
    return result


@router.delete("/patterns", response_model=dict)
async def delete_patterns(
    service: APIExtractionService = Depends(get_extraction_service)
):
    """Delete saved API patterns."""
    success = service.delete_patterns()
    return {
        "success": success,
        "message": "Patterns deleted" if success else "Failed to delete patterns"
    }
