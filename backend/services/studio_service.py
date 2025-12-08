"""Service wrapper for TikTokStudioScraper with async job management."""

import asyncio
import sys
import logging
import traceback
from pathlib import Path
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from studio_scraper import TikTokStudioScraper, run_studio_scraper
from scraper import TikTokScraper
from analyzer import VideoAnalyzer, get_config
from backend.utils.progress_manager import ProgressManager
from backend.models.schemas import StudioRequest, ThoroughnessPreset


def _run_async_in_thread(coro):
    """
    Run an async coroutine in a new thread with its own event loop.

    This is a workaround for Python 3.13 + Windows asyncio subprocess issues.
    Playwright needs to spawn browser processes, which fails in the main event loop
    on Python 3.13 Windows. Running in a separate thread with a fresh event loop works.
    """
    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _run_full_session_sync(scraper, video_scraper, analyzer, skip_download, skip_analysis):
    """
    Run entire scraper session in a SINGLE event loop.

    This is critical - Playwright objects (browser, page) are bound to the event loop
    they were created in. Running init and scrape in separate loops causes errors.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Initialize browser
        success = loop.run_until_complete(scraper.initialize(interactive=False))
        if not success:
            # Login required - return early, browser stays open
            # Note: The loop stays open so we can continue later
            return {"needs_login": True, "loop": loop}

        # Run scraping in the SAME loop
        results = loop.run_until_complete(scraper.scrape_all_videos(
            video_scraper=video_scraper,
            analyzer=analyzer,
            skip_download=skip_download,
            skip_analysis=skip_analysis,
        ))

        # Clean up
        loop.run_until_complete(scraper.close())
        loop.close()

        return {"success": True, "results": results}
    except Exception as e:
        # Try to close scraper on error
        try:
            loop.run_until_complete(scraper.close())
        except:
            pass
        try:
            loop.close()
        except:
            pass
        return {"error": str(e), "traceback": traceback.format_exc()}


def _continue_session_after_login_sync(scraper, loop, video_scraper, analyzer, skip_download, skip_analysis):
    """
    Continue scraper session after manual login, using the SAME event loop.
    """
    asyncio.set_event_loop(loop)
    try:
        # Verify login
        is_logged_in = loop.run_until_complete(scraper._check_logged_in())
        if not is_logged_in:
            return {"error": "Not logged in after manual login"}

        scraper._is_logged_in = True

        # Run scraping in the SAME loop
        results = loop.run_until_complete(scraper.scrape_all_videos(
            video_scraper=video_scraper,
            analyzer=analyzer,
            skip_download=skip_download,
            skip_analysis=skip_analysis,
        ))

        # Clean up
        loop.run_until_complete(scraper.close())
        loop.close()

        return {"success": True, "results": results}
    except Exception as e:
        try:
            loop.run_until_complete(scraper.close())
        except:
            pass
        try:
            loop.close()
        except:
            pass
        return {"error": str(e), "traceback": traceback.format_exc()}


# Thread pool for running Playwright operations
_playwright_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="playwright")


class StudioService:
    """Async wrapper for TikTokStudioScraper with progress broadcasting."""

    def __init__(self, progress_manager: ProgressManager):
        self.progress_manager = progress_manager
        self.logger = logging.getLogger(__name__)
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
        self._scrapers: Dict[str, TikTokStudioScraper] = {}
        self._session_loops: Dict[str, asyncio.AbstractEventLoop] = {}  # Store event loops for login flow

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

    async def _run_studio_session(self, session_id: str, request: StudioRequest, continue_from_login: bool = False) -> None:
        """Run Studio scraping session with progress updates."""
        try:
            if not continue_from_login:
                # Fresh start - create new scraper
                self._active_sessions[session_id]["status"] = "initializing"

                # Initialize browser and authenticate
                await self.progress_manager.broadcast(session_id, "studio_initializing", {
                    "status": "initializing",
                    "message": "Launching browser..."
                })

                try:
                    scraper = TikTokStudioScraper(
                        output_dir=request.output_dir,
                        logger=self.logger,
                        browser_type=request.studio_browser.value,
                        cdp_port=request.cdp_port,
                        username=request.username
                    )
                    self._scrapers[session_id] = scraper
                except Exception as init_error:
                    raise Exception(f"Failed to create scraper: {init_error}")

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

                # Store for login continuation
                self._active_sessions[session_id]["video_scraper"] = video_scraper
                self._active_sessions[session_id]["analyzer"] = analyzer

                await self.progress_manager.job_started(session_id, "studio", 0)

                # Run ENTIRE session (init + scrape) in a SINGLE event loop
                # This is critical - Playwright objects are bound to their creation loop
                main_loop = asyncio.get_event_loop()
                result = await main_loop.run_in_executor(
                    _playwright_executor,
                    lambda: _run_full_session_sync(
                        scraper, video_scraper, analyzer,
                        request.skip_download, request.skip_analysis
                    )
                )

                if result.get("needs_login"):
                    # Manual login required - browser stays open
                    # Store the event loop for continuation
                    self._session_loops[session_id] = result.get("loop")
                    self._active_sessions[session_id]["awaiting_login"] = True
                    await self.progress_manager.studio_login_required(session_id)
                    return
                elif result.get("error"):
                    raise Exception(result["error"])
                else:
                    results = result["results"]

            else:
                # Continuing after manual login - use existing scraper and loop
                self.logger.info(f"Continuing session {session_id} after login")

                scraper = self._scrapers.get(session_id)
                session_loop = self._session_loops.get(session_id)

                if not scraper:
                    raise Exception("No scraper found for session")
                if not session_loop:
                    raise Exception("No event loop found for session")

                video_scraper = self._active_sessions[session_id].get("video_scraper")
                analyzer = self._active_sessions[session_id].get("analyzer")

                # Continue in the SAME event loop
                main_loop = asyncio.get_event_loop()
                result = await main_loop.run_in_executor(
                    _playwright_executor,
                    lambda: _continue_session_after_login_sync(
                        scraper, session_loop, video_scraper, analyzer,
                        request.skip_download, request.skip_analysis
                    )
                )

                if result.get("error"):
                    raise Exception(result["error"])

                results = result["results"]

                # Clean up stored loop
                if session_id in self._session_loops:
                    del self._session_loops[session_id]

            # Update final status
            self._active_sessions[session_id]["status"] = "completed"
            self._active_sessions[session_id]["result"] = results
            await self.progress_manager.job_completed(session_id, result=results)

        except Exception as e:
            # Ensure error message is never empty
            error_str = str(e).strip()
            if not error_str:
                error_str = f"{type(e).__name__}: {repr(e)}"

            self.logger.error(f"Studio session {session_id} failed: {error_str}")
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")

            self._active_sessions[session_id]["status"] = "failed"
            self._active_sessions[session_id]["error"] = error_str
            await self.progress_manager.job_failed(session_id, error_str)

        finally:
            # Cleanup - but NOT if awaiting login (browser needs to stay open)
            session = self._active_sessions.get(session_id, {})
            if not session.get("awaiting_login"):
                if session_id in self._scrapers:
                    del self._scrapers[session_id]
                if session_id in self._session_loops:
                    del self._session_loops[session_id]

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

        # Continue scraping with existing browser
        request = session["request"]
        asyncio.create_task(self._run_studio_session(session_id, request, continue_from_login=True))
        return True

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

            # Close browser (run in thread for Windows compatibility)
            if session_id in self._scrapers:
                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        _playwright_executor,
                        lambda: _run_async_in_thread(self._scrapers[session_id].close())
                    )
                except:
                    pass
                del self._scrapers[session_id]

            return True
        return False
