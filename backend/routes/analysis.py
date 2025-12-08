"""Analysis API routes for video analysis."""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional

from backend.models.schemas import AnalysisRequest, AnalysisOptions
from backend.services.analysis_service import AnalysisService
from backend.utils.progress_manager import progress_manager

router = APIRouter(prefix="/analysis", tags=["analysis"])

# Service instance
_analysis_service: Optional[AnalysisService] = None


def get_analysis_service() -> AnalysisService:
    """Dependency to get analysis service instance."""
    global _analysis_service
    if _analysis_service is None:
        _analysis_service = AnalysisService(progress_manager)
    return _analysis_service


@router.post("/start", response_model=dict)
async def start_analysis(
    request: AnalysisRequest,
    service: AnalysisService = Depends(get_analysis_service)
):
    """
    Start batch video analysis.

    If video_paths is not provided, will scan output_dir for videos.
    Returns job_id for tracking progress via WebSocket.
    """
    job_id = await service.start_analysis(request)
    return {
        "job_id": job_id,
        "status": "started",
        "message": "Analysis job started. Connect to WebSocket for progress updates."
    }


@router.get("/status/{job_id}", response_model=dict)
async def get_analysis_status(
    job_id: str,
    service: AnalysisService = Depends(get_analysis_service)
):
    """Get status of an analysis job."""
    status = service.get_job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return status


@router.post("/single", response_model=dict)
async def analyze_single_video(
    video_path: str,
    options: Optional[AnalysisOptions] = None,
    service: AnalysisService = Depends(get_analysis_service)
):
    """
    Analyze a single video synchronously.

    Returns full analysis results immediately (may take time for large videos).
    """
    if options is None:
        options = AnalysisOptions()

    result = await service.analyze_single(video_path, options)
    return result


@router.post("/cancel/{job_id}", response_model=dict)
async def cancel_analysis(
    job_id: str,
    service: AnalysisService = Depends(get_analysis_service)
):
    """Cancel a running analysis job."""
    success = service.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return {"job_id": job_id, "status": "cancelled"}
