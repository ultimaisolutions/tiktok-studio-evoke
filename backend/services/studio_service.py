"""Service wrapper for TikTokStudioScraper with async job management."""

import asyncio
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from studio_scraper import TikTokStudioScraper, run_studio_scraper
from scraper import TikTokScraper
from analyzer import VideoAnalyzer, get_config
from backend.utils.progress_manager import ProgressManager
from backend.models.schemas import StudioRequest, ThoroughnessPreset


class StudioService:
    """Async wrapper for TikTokStudioScraper with progress broadcasting."""

    def __init__(self, progress_manager: ProgressManager):
        self.progress_manager = progress_manager
        self.logger = logging.getLogger(__name__)
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
        self._scrapers: Dict[str, TikTokStudioScraper] = {}

    async def start_session(self, request: StudioRequest) -> str:
        """
        Start a TikTok Studio scraping session.

        Args:
            request: Studio scraping request with options

        Returns:
            Session ID for tracking progress
        """
        session_id = self.progress_manager.create_job("studio")
        self._active_sessions[session_id] = {
            "status": "pending",
            "request": request,
            "awaiting_login": False
        }

        # Start scraping in background
        asyncio.create_task(self._run_studio_session(session_id, request))

        return session_id

    async def _run_studio_session(self, session_id: str, request: StudioRequest) -> None:
        """Run Studio scraping session with progress updates."""
        try:
            self._active_sessions[session_id]["status"] = "initializing"

            # Create scraper instance
            scraper = TikTokStudioScraper(
                output_dir=request.output_dir,
                logger=self.logger,
                browser_type=request.studio_browser.value,
                cdp_port=request.cdp_port,
                username=request.username
            )
            self._scrapers[session_id] = scraper

            # Initialize browser and authenticate
            await self.progress_manager.broadcast(session_id, "studio_initializing", {
                "status": "initializing",
                "message": "Launching browser..."
            })

            success = await scraper.initialize()

            if not success:
                # Check if manual login is required
                self._active_sessions[session_id]["awaiting_login"] = True
                await self.progress_manager.studio_login_required(session_id)
                return

            # Prepare video scraper if download enabled
            video_scraper = None
            if not request.skip_download:
                video_scraper = TikTokScraper(request.output_dir, self.logger)
                if request.cookie_browser:
                    video_scraper.initialize_browser(request.cookie_browser.value, required=False)

            # Prepare analyzer if analysis enabled
            analyzer = None
            if not request.skip_analysis and request.analysis_options:
                config = get_config(
                    preset=request.analysis_options.thoroughness.value,
                    sample_percentage=request.analysis_options.sample_percent,
                    color_clusters=request.analysis_options.color_clusters,
                    motion_resolution=request.analysis_options.motion_res,
                    face_model=request.analysis_options.face_model.value if request.analysis_options.face_model else None,
                    workers=request.analysis_options.workers,
                    enable_audio=not request.analysis_options.skip_audio,
                    scene_detection=request.analysis_options.scene_detection,
                    full_resolution=request.analysis_options.full_resolution,
                )
                analyzer = VideoAnalyzer(config, self.logger)

            # Start scraping with progress callbacks
            await self.progress_manager.job_started(session_id, "studio", 0)

            # Run the scraping
            results = await scraper.scrape_all_videos(
                video_scraper=video_scraper,
                analyzer=analyzer,
                skip_download=request.skip_download,
                skip_analysis=request.skip_analysis,
            )

            # Update final status
            self._active_sessions[session_id]["status"] = "completed"
            self._active_sessions[session_id]["result"] = results
            await self.progress_manager.job_completed(session_id, result=results)

        except Exception as e:
            self.logger.error(f"Studio session {session_id} failed: {e}")
            self._active_sessions[session_id]["status"] = "failed"
            self._active_sessions[session_id]["error"] = str(e)
            await self.progress_manager.job_failed(session_id, str(e))

        finally:
            # Cleanup scraper
            if session_id in self._scrapers:
                try:
                    await self._scrapers[session_id].close()
                except:
                    pass
                del self._scrapers[session_id]

    async def continue_after_login(self, session_id: str) -> bool:
        """
        Continue session after user has logged in manually.

        Args:
            session_id: Session to continue

        Returns:
            True if session was continued
        """
        if session_id not in self._active_sessions:
            return False

        session = self._active_sessions[session_id]
        if not session.get("awaiting_login"):
            return False

        session["awaiting_login"] = False

        # Re-check login and continue
        scraper = self._scrapers.get(session_id)
        if scraper:
            # Re-verify login
            is_logged_in = await scraper._check_logged_in()
            if is_logged_in:
                scraper._is_logged_in = True
                # Restart the scraping process
                request = session["request"]
                asyncio.create_task(self._run_studio_session(session_id, request))
                return True

        return False

    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a Studio session."""
        if session_id in self._active_sessions:
            session = self._active_sessions[session_id]
            state = self.progress_manager.get_job_state(session_id)
            return {
                "session_id": session_id,
                "status": session.get("status", "unknown"),
                "awaiting_login": session.get("awaiting_login", False),
                "result": session.get("result"),
                "error": session.get("error"),
                "progress": state.get("data", {}) if state else {}
            }
        return None

    async def stop_session(self, session_id: str) -> bool:
        """Stop a running Studio session."""
        if session_id in self._active_sessions:
            self._active_sessions[session_id]["status"] = "cancelled"

            # Close browser
            if session_id in self._scrapers:
                try:
                    await self._scrapers[session_id].close()
                except:
                    pass
                del self._scrapers[session_id]

            return True
        return False
