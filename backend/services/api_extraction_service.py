"""Service wrapper for TikTokAPIExtractor with async job management."""

import asyncio
import sys
import logging
import traceback
from pathlib import Path
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api_extractor import TikTokAPIExtractor, PATTERNS_FILE
from backend.utils.progress_manager import ProgressManager
from backend.models.schemas import ExtractionRequest


def _run_extraction_sync(extractor, sample_video_count, progress_callback=None):
    """
    Run extraction session in a SINGLE event loop.

    Playwright objects (browser, page) are bound to the event loop
    they were created in. This ensures all operations use the same loop.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Initialize browser
        success = loop.run_until_complete(extractor.initialize(interactive=False))
        if not success:
            # Login required - browser stays open
            return {"needs_login": True, "loop": loop}

        # Run extraction in the SAME loop
        async def extract_with_callback():
            return await extractor.extract_patterns(
                sample_video_count=sample_video_count,
                progress_callback=progress_callback
            )

        results = loop.run_until_complete(extract_with_callback())

        # Save patterns
        extractor.save_patterns()

        # Clean up
        loop.run_until_complete(extractor.close())
        loop.close()

        return {"success": True, "results": results}
    except Exception as e:
        try:
            loop.run_until_complete(extractor.close())
        except:
            pass
        try:
            loop.close()
        except:
            pass
        return {"error": str(e), "traceback": traceback.format_exc()}


def _continue_extraction_after_login_sync(extractor, loop, sample_video_count, progress_callback=None):
    """
    Continue extraction session after manual login, using the SAME event loop.
    """
    asyncio.set_event_loop(loop)
    try:
        # Verify login
        is_logged_in = loop.run_until_complete(extractor.continue_after_login())
        if not is_logged_in:
            return {"error": "Not logged in after manual login"}

        # Run extraction in the SAME loop
        async def extract_with_callback():
            return await extractor.extract_patterns(
                sample_video_count=sample_video_count,
                progress_callback=progress_callback
            )

        results = loop.run_until_complete(extract_with_callback())

        # Save patterns
        extractor.save_patterns()

        # Clean up
        loop.run_until_complete(extractor.close())
        loop.close()

        return {"success": True, "results": results}
    except Exception as e:
        try:
            loop.run_until_complete(extractor.close())
        except:
            pass
        try:
            loop.close()
        except:
            pass
        return {"error": str(e), "traceback": traceback.format_exc()}


# Thread pool for running Playwright operations
_playwright_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="extractor")


class APIExtractionService:
    """Async wrapper for TikTokAPIExtractor with progress broadcasting."""

    def __init__(self, progress_manager: ProgressManager):
        self.progress_manager = progress_manager
        self.logger = logging.getLogger(__name__)
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
        self._extractors: Dict[str, TikTokAPIExtractor] = {}
        self._session_loops: Dict[str, asyncio.AbstractEventLoop] = {}

    async def start_extraction(self, request: ExtractionRequest) -> str:
        """
        Start an API pattern extraction session.

        Args:
            request: Extraction request with options

        Returns:
            Session ID for tracking progress
        """
        session_id = self.progress_manager.create_job("extraction")
        self._active_sessions[session_id] = {
            "status": "pending",
            "request": request,
            "awaiting_login": False
        }

        # Start extraction in background
        asyncio.create_task(self._run_extraction_session(session_id, request))

        return session_id

    async def _run_extraction_session(self, session_id: str, request: ExtractionRequest, continue_from_login: bool = False) -> None:
        """Run extraction session with progress updates."""
        try:
            if not continue_from_login:
                # Fresh start - create new extractor
                self._active_sessions[session_id]["status"] = "initializing"

                await self.progress_manager.broadcast(session_id, "extraction_initializing", {
                    "status": "initializing",
                    "message": "Launching browser..."
                })

                try:
                    extractor = TikTokAPIExtractor(
                        logger=self.logger,
                        browser_type=request.studio_browser.value,
                        cdp_port=request.cdp_port,
                    )
                    self._extractors[session_id] = extractor
                except Exception as init_error:
                    raise Exception(f"Failed to create extractor: {init_error}")

                # Store request for login continuation
                self._active_sessions[session_id]["sample_video_count"] = request.sample_video_count

                await self.progress_manager.job_started(session_id, "extraction", 0)

                # Create progress callback
                async def progress_callback(message: str, progress: float):
                    await self.progress_manager.job_progress(
                        session_id,
                        completed=int(progress * 100),
                        total=100,
                        current_task=message
                    )

                # Run extraction in a single event loop
                main_loop = asyncio.get_event_loop()
                result = await main_loop.run_in_executor(
                    _playwright_executor,
                    lambda: _run_extraction_sync(
                        extractor, request.sample_video_count
                    )
                )

                if result.get("needs_login"):
                    # Manual login required - browser stays open
                    self._session_loops[session_id] = result.get("loop")
                    self._active_sessions[session_id]["awaiting_login"] = True
                    await self.progress_manager.broadcast(session_id, "extraction_login_required", {
                        "status": "awaiting_login",
                        "message": "Please log in to TikTok in the browser window"
                    })
                    return
                elif result.get("error"):
                    raise Exception(result["error"])
                else:
                    results = result["results"]

            else:
                # Continuing after manual login
                self.logger.info(f"Continuing extraction {session_id} after login")

                extractor = self._extractors.get(session_id)
                session_loop = self._session_loops.get(session_id)

                if not extractor:
                    raise Exception("No extractor found for session")
                if not session_loop:
                    raise Exception("No event loop found for session")

                sample_video_count = self._active_sessions[session_id].get("sample_video_count", 3)

                # Continue in the SAME event loop
                main_loop = asyncio.get_event_loop()
                result = await main_loop.run_in_executor(
                    _playwright_executor,
                    lambda: _continue_extraction_after_login_sync(
                        extractor, session_loop, sample_video_count
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
            error_str = str(e).strip()
            if not error_str:
                error_str = f"{type(e).__name__}: {repr(e)}"

            self.logger.error(f"Extraction session {session_id} failed: {error_str}")
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")

            self._active_sessions[session_id]["status"] = "failed"
            self._active_sessions[session_id]["error"] = error_str
            await self.progress_manager.job_failed(session_id, error_str)

        finally:
            # Cleanup - but NOT if awaiting login
            session = self._active_sessions.get(session_id, {})
            if not session.get("awaiting_login"):
                if session_id in self._extractors:
                    del self._extractors[session_id]
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

        # Continue extraction with existing browser
        request = session["request"]
        asyncio.create_task(self._run_extraction_session(session_id, request, continue_from_login=True))
        return True

    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an extraction session."""
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
        """Stop a running extraction session."""
        if session_id in self._active_sessions:
            self._active_sessions[session_id]["status"] = "cancelled"

            # Close browser
            if session_id in self._extractors:
                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        _playwright_executor,
                        lambda: asyncio.run(self._extractors[session_id].close())
                    )
                except:
                    pass
                del self._extractors[session_id]

            return True
        return False

    def get_patterns(self) -> Optional[Dict[str, Any]]:
        """Get saved API patterns."""
        return TikTokAPIExtractor.load_patterns()

    def delete_patterns(self) -> bool:
        """Delete saved API patterns."""
        return TikTokAPIExtractor.delete_patterns()

    def apply_patterns(self) -> Dict[str, Any]:
        """
        Apply saved patterns to scraper configuration.

        Returns:
            Dictionary with applied patterns info
        """
        patterns = TikTokAPIExtractor.load_patterns()
        if not patterns:
            return {"success": False, "message": "No patterns found"}

        # The patterns are already saved to file
        # The studio_scraper will load them when needed
        return {
            "success": True,
            "message": "Patterns are available for scraper",
            "video_list_api": patterns.get("video_list_api") is not None,
            "analytics_api": patterns.get("analytics_api") is not None,
            "other_apis_count": len(patterns.get("other_apis", [])),
            "last_updated": patterns.get("last_updated")
        }
