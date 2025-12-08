"""Scraper API routes for downloading TikTok videos."""

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from typing import List, Optional

from backend.models.schemas import DownloadRequest, JobStatus, BrowserType
from backend.services.scraper_service import ScraperService
from backend.utils.progress_manager import progress_manager

router = APIRouter(prefix="/scraper", tags=["scraper"])

# Service instance (will be initialized in main.py)
_scraper_service: Optional[ScraperService] = None


def get_scraper_service() -> ScraperService:
    """Dependency to get scraper service instance."""
    global _scraper_service
    if _scraper_service is None:
        _scraper_service = ScraperService(progress_manager)
    return _scraper_service


@router.post("/download", response_model=dict)
async def start_download(
    request: DownloadRequest,
    service: ScraperService = Depends(get_scraper_service)
):
    """
    Start downloading videos from provided URLs.

    Returns job_id for tracking progress via WebSocket.
    """
    job_id = await service.start_download(request)
    return {"job_id": job_id, "status": "started", "total_urls": len(request.urls)}


@router.post("/download-file", response_model=dict)
async def download_from_file(
    file: UploadFile = File(...),
    output_dir: str = "videos",
    browser: BrowserType = BrowserType.chrome,
    no_browser: bool = False,
    analyze: bool = False,
    service: ScraperService = Depends(get_scraper_service)
):
    """
    Upload a text file with URLs and start downloading.

    File should contain one URL per line. Lines starting with # are ignored.
    """
    # Read and parse file
    contents = await file.read()
    text = contents.decode('utf-8')

    urls = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if line and not line.startswith('#'):
            urls.append(line)

    if not urls:
        raise HTTPException(status_code=400, detail="No valid URLs found in file")

    # Create download request
    request = DownloadRequest(
        urls=urls,
        output_dir=output_dir,
        browser=browser,
        no_browser=no_browser,
        analyze=analyze
    )

    job_id = await service.start_download(request)
    return {"job_id": job_id, "status": "started", "total_urls": len(urls)}


@router.get("/status/{job_id}", response_model=dict)
async def get_download_status(
    job_id: str,
    service: ScraperService = Depends(get_scraper_service)
):
    """Get status of a download job."""
    status = service.get_job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return status


@router.post("/cancel/{job_id}", response_model=dict)
async def cancel_download(
    job_id: str,
    service: ScraperService = Depends(get_scraper_service)
):
    """Cancel a running download job."""
    success = service.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return {"job_id": job_id, "status": "cancelled"}
