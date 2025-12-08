"""Service wrapper for TikTokScraper with async job management."""

import asyncio
import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scraper import TikTokScraper
from backend.utils.progress_manager import ProgressManager
from backend.models.schemas import DownloadRequest, BrowserType


class ScraperService:
    """Async wrapper for TikTokScraper with progress broadcasting."""

    def __init__(self, progress_manager: ProgressManager):
        self.progress_manager = progress_manager
        self.logger = logging.getLogger(__name__)
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._active_jobs: Dict[str, Dict[str, Any]] = {}

    async def start_download(self, request: DownloadRequest) -> str:
        """
        Start a download job asynchronously.

        Args:
            request: Download request with URLs and options

        Returns:
            Job ID for tracking progress
        """
        job_id = self.progress_manager.create_job("download")
        self._active_jobs[job_id] = {"status": "pending", "request": request}

        # Start download in background
        asyncio.create_task(self._run_download(job_id, request))

        return job_id

    async def _run_download(self, job_id: str, request: DownloadRequest) -> None:
        """Run download job in background with progress updates."""
        try:
            # Create scraper instance
            scraper = TikTokScraper(request.output_dir, self.logger)

            # Initialize browser if needed
            if not request.no_browser:
                loop = asyncio.get_event_loop()
                success = await loop.run_in_executor(
                    self._executor,
                    scraper.initialize_browser,
                    request.browser.value,
                    False
                )
                if not success:
                    self.logger.warning("Browser initialization failed, continuing without auth")

            # Broadcast job started
            await self.progress_manager.job_started(job_id, "download", len(request.urls))

            # Download each URL
            results = {
                "total": len(request.urls),
                "success": 0,
                "failed": 0,
                "successful_urls": [],
                "failed_urls": [],
            }

            for i, url in enumerate(request.urls):
                # Broadcast download start
                await self.progress_manager.download_start(
                    job_id, url, i + 1, len(request.urls)
                )

                # Run download in executor (blocking)
                loop = asyncio.get_event_loop()
                success, message = await loop.run_in_executor(
                    self._executor,
                    scraper.download_video,
                    url
                )

                # Broadcast result
                if success:
                    results["success"] += 1
                    results["successful_urls"].append({"url": url, "message": message})
                    await self.progress_manager.download_complete(
                        job_id, url, True, path=message
                    )
                else:
                    results["failed"] += 1
                    results["failed_urls"].append({"url": url, "error": message})
                    await self.progress_manager.download_complete(
                        job_id, url, False, message=message
                    )

                # Update progress
                await self.progress_manager.job_progress(
                    job_id,
                    completed=results["success"] + results["failed"],
                    total=len(request.urls),
                    current_task=f"Downloaded {i + 1}/{len(request.urls)}",
                    failed=results["failed"]
                )

            # Job completed
            self._active_jobs[job_id]["status"] = "completed"
            self._active_jobs[job_id]["result"] = results
            await self.progress_manager.job_completed(job_id, result=results)

        except Exception as e:
            self.logger.error(f"Download job {job_id} failed: {e}")
            self._active_jobs[job_id]["status"] = "failed"
            self._active_jobs[job_id]["error"] = str(e)
            await self.progress_manager.job_failed(job_id, str(e))

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a download job."""
        if job_id in self._active_jobs:
            job = self._active_jobs[job_id]
            state = self.progress_manager.get_job_state(job_id)
            return {
                "job_id": job_id,
                "status": job.get("status", "unknown"),
                "result": job.get("result"),
                "error": job.get("error"),
                "progress": state.get("data", {}) if state else {}
            }
        return None

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running download job."""
        if job_id in self._active_jobs:
            self._active_jobs[job_id]["status"] = "cancelled"
            return True
        return False
