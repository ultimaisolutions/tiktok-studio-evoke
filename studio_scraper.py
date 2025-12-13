"""
TikTok Studio Scraper - Automates extraction of analytics data from TikTok Studio.

Captures screenshots of all 3 analytics tabs for each video, extracts video URLs,
and integrates with the existing download/analysis pipeline.
"""

import asyncio
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from playwright.async_api import async_playwright, Browser, Page, BrowserContext

from utils import ensure_directory, timestamp_to_date
from datetime import datetime


def normalize_username(username: str) -> str:
    """Normalize username: remove @, lowercase, remove dots."""
    if not username:
        return "unknown_user"
    return username.lstrip("@").lower().replace(".", "")


# TikTok Studio URLs
STUDIO_HOME_URL = "https://www.tiktok.com/tiktokstudio"
STUDIO_CONTENT_URL = "https://www.tiktok.com/tiktokstudio/content"
TIKTOK_LOGIN_URL = "https://www.tiktok.com/login"

# API patterns file path
API_PATTERNS_FILE = Path("api_patterns.json")

# Tab names (Hebrew UI based on screenshots)
TAB_NAMES = {
    "overview": "סקירה כללית",
    "viewers": "צופים",
    "engagement": "מעורבות",
}


class TikTokStudioScraper:
    """
    Automates TikTok Studio to capture analytics screenshots and extract video URLs.

    Features:
    - Cookie-first authentication with manual login fallback
    - Captures 3 analytics tabs per video (Overview, Viewers, Engagement)
    - Extracts video URLs for download
    - Incremental processing (skips already processed videos)
    """

    def __init__(self, output_dir: str, logger, browser_type: str = "chromium", cdp_port: int = None, username: str = None):
        """
        Initialize the Studio scraper.

        Args:
            output_dir: Base directory for saving screenshots and videos
            logger: Logger instance for status messages
            browser_type: Playwright browser type (chromium, firefox, webkit)
            cdp_port: Custom CDP port for connecting to existing browser
            username: TikTok username for constructing video URLs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.logger.info(f"Output directory: {self.output_dir.resolve()}")
        self.browser_type = browser_type
        self.cdp_port = cdp_port
        self.account_username = username  # Used for constructing video URLs

        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.playwright = None

        self._is_logged_in = False
        self._processed_videos = set()
        self._connected_to_existing = False

        # Load API patterns if available
        self._api_patterns = self._load_api_patterns()

    async def initialize(self, interactive: bool = True) -> bool:
        """
        Initialize browser and attempt authentication.

        Flow: Try cookies → Connect to existing browser → Manual login in new browser

        Args:
            interactive: If True (CLI mode), wait for user input for manual login.
                        If False (web service mode), return False when login required.

        Returns:
            True if successfully logged in, False if login required (web mode) or failed
        """
        self.logger.info("Initializing TikTok Studio scraper...")

        # Launch Playwright
        self.playwright = await async_playwright().start()

        # Step 1: Try to connect to existing browser with Studio already open
        if await self._try_connect_to_existing_browser():
            self.logger.info("Using existing browser with TikTok Studio")
            return True

        # Step 2: Fallback to launching new browser
        self.logger.info("Launching new browser...")

        # Select browser type
        if self.browser_type == "firefox":
            browser_launcher = self.playwright.firefox
        elif self.browser_type == "webkit":
            browser_launcher = self.playwright.webkit
        else:
            browser_launcher = self.playwright.chromium

        # Launch browser in headful mode so user can see/interact
        try:
            self.browser = await browser_launcher.launch(headless=False)
        except Exception as e:
            error_msg = str(e)
            # Check for common Playwright errors
            if "Executable doesn't exist" in error_msg or "browserType.launch" in error_msg:
                raise Exception(
                    f"Playwright {self.browser_type} browser not installed. "
                    f"Run 'playwright install {self.browser_type}' to install it. "
                    f"Original error: {error_msg}"
                )
            elif "Target page, context or browser has been closed" in error_msg:
                raise Exception(f"Browser was closed unexpectedly: {error_msg}")
            else:
                raise Exception(f"Failed to launch {self.browser_type} browser: {error_msg}")

        # Create context with viewport
        self.context = await self.browser.new_context(
            viewport={"width": 1920, "height": 1080},
            locale="he-IL"  # Hebrew locale based on screenshots
        )

        self.page = await self.context.new_page()

        # Try to load cookies from browser
        cookies_loaded = await self._try_load_cookies()

        # Navigate to TikTok Studio
        self.logger.info("Navigating to TikTok Studio...")
        await self.page.goto(STUDIO_HOME_URL, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(3)  # Allow time for page to fully load

        # Check if we're logged in
        self._is_logged_in = await self._check_logged_in()

        if not self._is_logged_in:
            # Step 3: Manual login required
            self.logger.info("Not logged in. Manual login required.")

            if not interactive:
                # Web service mode: return False to signal login needed
                # Caller should handle showing login UI and calling continue_after_login
                self.logger.info("Non-interactive mode: returning False for manual login flow")
                return False

            # CLI mode: wait for user input
            print("\n" + "=" * 50)
            print("  MANUAL LOGIN REQUIRED")
            print("=" * 50)
            print("  Please log in to TikTok in the browser window.")
            print("  Press Enter here when you're done logging in...")
            print("=" * 50)

            # Wait for user to press Enter
            await asyncio.get_event_loop().run_in_executor(None, input)

            # Re-check login status
            await self.page.goto(STUDIO_HOME_URL, wait_until="domcontentloaded", timeout=60000)
            await asyncio.sleep(3)  # Allow time for page to fully load
            self._is_logged_in = await self._check_logged_in()

            if not self._is_logged_in:
                self.logger.error("Still not logged in. Please try again.")
                return False

        self.logger.info("Successfully logged in to TikTok Studio!")
        return True

    async def _try_load_cookies(self) -> bool:
        """
        Try to load cookies from local browser using browser-cookie3.

        Returns:
            True if cookies were loaded successfully
        """
        try:
            import browser_cookie3

            self.logger.info("Attempting to load cookies from local browser...")

            # Try browsers in order - Safari for macOS, Chrome, Firefox, Edge
            cookie_jar = None
            browsers_to_try = [
                ("Chrome", browser_cookie3.chrome),
                ("Firefox", browser_cookie3.firefox),
                ("Edge", browser_cookie3.edge),
                ("Chromium", browser_cookie3.chromium),
            ]

            # Add Safari on macOS
            if platform.system() == "Darwin":
                browsers_to_try.insert(0, ("Safari", browser_cookie3.safari))

            for browser_name, cookie_func in browsers_to_try:
                try:
                    cookie_jar = cookie_func(domain_name=".tiktok.com")
                    if cookie_jar:
                        self.logger.info(f"Found cookies from {browser_name}")
                        break
                except Exception as e:
                    self.logger.debug(f"Could not get cookies from {browser_name}: {e}")
                    continue

            if not cookie_jar:
                self.logger.info("No cookies found in local browsers")
                return False

            # Convert to Playwright cookie format
            playwright_cookies = []
            for cookie in cookie_jar:
                if ".tiktok.com" in cookie.domain:
                    playwright_cookies.append({
                        "name": cookie.name,
                        "value": cookie.value,
                        "domain": cookie.domain,
                        "path": cookie.path or "/",
                        "secure": cookie.secure,
                        "httpOnly": bool(cookie.has_nonstandard_attr("HttpOnly")),
                    })

            if playwright_cookies:
                await self.context.add_cookies(playwright_cookies)
                self.logger.info(f"Loaded {len(playwright_cookies)} cookies")
                return True

            return False

        except ImportError:
            self.logger.warning("browser-cookie3 not installed, skipping cookie loading")
            return False
        except Exception as e:
            self.logger.warning(f"Error loading cookies: {e}")
            return False

    async def _find_chrome_debugging_port(self) -> Optional[int]:
        """
        Scan Chrome/Edge/Brave processes to find one with remote-debugging-port flag.

        Returns:
            Port number if found, None otherwise
        """
        try:
            import psutil

            for proc in psutil.process_iter(['name', 'cmdline']):
                try:
                    name = proc.info['name'].lower()
                    cmdline = proc.info['cmdline'] or []

                    # Check if it's a Chromium-based browser
                    if not any(x in name for x in ['chrome', 'chromium', 'msedge', 'edge', 'brave']):
                        continue

                    # Look for remote-debugging-port argument
                    for i, arg in enumerate(cmdline):
                        if '--remote-debugging-port=' in arg:
                            port = int(arg.split('=')[1])
                            return port
                        elif arg == '--remote-debugging-port' and i + 1 < len(cmdline):
                            try:
                                port = int(cmdline[i + 1])
                                return port
                            except ValueError:
                                continue
                except (psutil.NoSuchProcess, psutil.AccessDenied, IndexError):
                    continue

            return None

        except ImportError:
            self.logger.warning("psutil not installed, skipping process detection")
            return None
        except Exception as e:
            self.logger.debug(f"Error detecting Chrome process: {e}")
            return None

    async def _verify_cdp_endpoint(self, port: int) -> bool:
        """
        Verify that a CDP endpoint is responding.

        Args:
            port: Port number to check

        Returns:
            True if endpoint is responding
        """
        try:
            import requests

            endpoint = f"http://localhost:{port}/json/version"
            response = requests.get(endpoint, timeout=1)
            return response.status_code == 200

        except Exception:
            return False

    async def _find_cdp_endpoint(self) -> Optional[str]:
        """
        Find an active CDP endpoint for connecting to existing browser.

        Returns:
            Endpoint URL (http://localhost:PORT) or None
        """
        ports_to_check = []

        # If user specified a custom port, only check that
        if self.cdp_port:
            ports_to_check = [self.cdp_port]
        else:
            # First try to detect from running processes
            detected_port = await self._find_chrome_debugging_port()
            if detected_port:
                ports_to_check.append(detected_port)

            # Then scan common ports
            ports_to_check.extend(range(9222, 9230))

        # Try each port
        for port in ports_to_check:
            if await self._verify_cdp_endpoint(port):
                endpoint = f"http://localhost:{port}"
                self.logger.info(f"Found CDP endpoint at {endpoint}")
                return endpoint

        return None

    async def _find_tiktok_studio_pages(self, browser: Browser) -> list:
        """
        Find all pages with TikTok Studio URLs in the connected browser.

        Args:
            browser: Connected browser instance

        Returns:
            List of tuples: (page, url, is_studio)
            is_studio=True means it's on tiktokstudio domain, False means just tiktok.com
        """
        studio_pages = []

        try:
            # Get all contexts (usually just one for CDP connections)
            for context in browser.contexts:
                for page in context.pages:
                    url = page.url

                    # Check if it's a TikTok Studio URL
                    # Example: https://www.tiktok.com/tiktokstudio/analytics/7513280379839712530
                    if 'tiktokstudio' in url.lower():
                        studio_pages.append((page, url, True))
                    elif 'tiktok.com' in url.lower():
                        studio_pages.append((page, url, False))

            # Sort: Studio pages first, then others
            studio_pages.sort(key=lambda x: (not x[2], x[1]))

            return studio_pages

        except Exception as e:
            self.logger.error(f"Error finding TikTok pages: {e}")
            return []

    async def _verify_page_logged_in(self, page: Page) -> bool:
        """
        Verify that a page is logged in to TikTok Studio.

        Args:
            page: Page to check

        Returns:
            True if logged in
        """
        try:
            # Navigate to Studio if not already there
            current_url = page.url
            if 'tiktokstudio' not in current_url.lower():
                await page.goto(STUDIO_HOME_URL, wait_until="networkidle", timeout=30000)

            # Use existing login check logic
            # Wait a moment for page to settle
            await asyncio.sleep(0.5)

            current_url = page.url

            # If on login page, not logged in
            if "login" in current_url.lower():
                return False

            # If on Studio page, check for logged-in indicators
            if "tiktokstudio" in current_url:
                try:
                    await page.wait_for_selector('[class*="sidebar"]', timeout=2000)
                    return True
                except:
                    pass

                # Alternative: check if there's no login button
                login_button = await page.query_selector('button:has-text("Log in")')
                if not login_button:
                    return True

            return False

        except Exception as e:
            self.logger.debug(f"Error verifying login status: {e}")
            return False

    async def _select_page_from_list(self, pages: list) -> Optional[Page]:
        """
        Let user select a TikTok Studio page from multiple options.

        Args:
            pages: List of tuples (page, url, is_studio)

        Returns:
            Selected page or None
        """
        print("\n" + "=" * 70)
        print("  MULTIPLE TIKTOK STUDIO TABS FOUND")
        print("=" * 70)
        print("  Please select which tab to use:\n")

        for i, (page, url, is_studio) in enumerate(pages, 1):
            marker = "[STUDIO]" if is_studio else "[TIKTOK]"
            # Truncate URL if too long
            display_url = url if len(url) < 60 else url[:57] + "..."
            print(f"  {i}. {marker} {display_url}")

        print("\n" + "=" * 70)

        # Get user input
        while True:
            try:
                choice = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: input(f"  Enter number (1-{len(pages)}): ")
                )
                choice_num = int(choice)
                if 1 <= choice_num <= len(pages):
                    selected_page = pages[choice_num - 1][0]
                    self.logger.info(f"User selected page {choice_num}")
                    return selected_page
                else:
                    print(f"  Please enter a number between 1 and {len(pages)}")
            except ValueError:
                print("  Please enter a valid number")
            except KeyboardInterrupt:
                return None

    async def _try_connect_to_existing_browser(self) -> bool:
        """
        Try to connect to an existing Chromium browser with TikTok Studio open.

        Returns:
            True if successfully connected and found a logged-in Studio page
        """
        try:
            self.logger.info("Attempting to connect to existing browser...")

            # Find CDP endpoint
            endpoint = await self._find_cdp_endpoint()

            if not endpoint:
                self.logger.info("No browser with remote debugging found")

                # If CDP port is specified, try to auto-launch Chrome
                if self.cdp_port:
                    self.logger.info(f"Attempting to auto-launch Chrome on port {self.cdp_port}...")
                    if await self._launch_chrome_with_debugging(self.cdp_port):
                        # Retry finding endpoint after launching
                        endpoint = await self._find_cdp_endpoint()
                        if not endpoint:
                            self.logger.warning("Chrome launched but could not connect")
                            self._show_cdp_instructions()
                            return False
                    else:
                        self.logger.warning("Failed to auto-launch Chrome")
                        self._show_cdp_instructions()
                        return False
                else:
                    self._show_cdp_instructions()
                    return False

            # Connect to browser
            self.logger.info(f"Connecting to browser at {endpoint}...")

            try:
                self.browser = await self.playwright.chromium.connect_over_cdp(
                    endpoint,
                    timeout=10000  # 10 second timeout
                )
                self._connected_to_existing = True

            except Exception as e:
                self.logger.warning(f"Failed to connect to CDP endpoint: {e}")
                self._show_cdp_instructions()
                return False

            # Find TikTok Studio pages
            self.logger.info("Searching for TikTok Studio tabs...")
            studio_pages = await self._find_tiktok_studio_pages(self.browser)

            if not studio_pages:
                self.logger.info("No TikTok Studio tabs found in browser")
                # Disconnect and return False to try next fallback
                await self.browser.close()
                self.browser = None
                self._connected_to_existing = False
                return False

            # Filter for logged-in pages
            self.logger.info("Verifying login status...")
            logged_in_pages = []

            for page, url, is_studio in studio_pages:
                self.logger.debug(f"Checking page: {url}")
                if await self._verify_page_logged_in(page):
                    logged_in_pages.append((page, url, is_studio))
                    self.logger.debug(f"  ✓ Logged in")
                else:
                    self.logger.debug(f"  ✗ Not logged in")

            if not logged_in_pages:
                self.logger.info("No logged-in TikTok Studio tabs found")
                await self.browser.close()
                self.browser = None
                self._connected_to_existing = False
                return False

            # Select page (single or user choice)
            if len(logged_in_pages) == 1:
                selected_page = logged_in_pages[0][0]
                self.logger.info(f"Using Studio tab: {logged_in_pages[0][1]}")
            else:
                selected_page = await self._select_page_from_list(logged_in_pages)
                if not selected_page:
                    self.logger.info("No page selected by user")
                    await self.browser.close()
                    self.browser = None
                    self._connected_to_existing = False
                    return False

            # Set up the page and context
            self.page = selected_page
            self.context = self.page.context
            self._is_logged_in = True

            self.logger.info("Successfully connected to existing browser!")
            return True

        except Exception as e:
            self.logger.error(f"Error connecting to existing browser: {e}")
            # Clean up on error
            if self.browser:
                try:
                    await self.browser.close()
                except:
                    pass
                self.browser = None
            self._connected_to_existing = False
            return False

    def _show_cdp_instructions(self):
        """Show instructions for launching Chrome with remote debugging."""
        print("\n" + "=" * 70)
        print("  TIP: Connect to Existing Browser")
        print("=" * 70)
        print("  To use an existing browser, launch Chrome with remote debugging:")
        print()
        print("  Windows:")
        print('    chrome.exe --remote-debugging-port=9222 --user-data-dir="C:\\temp\\chrome_profile"')
        print()
        print("  macOS/Linux:")
        print('    google-chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome_profile')
        print()
        print("  Then navigate to TikTok Studio and log in before running this script.")
        print("=" * 70 + "\n")

    async def _launch_chrome_with_debugging(self, port: int) -> bool:
        """
        Auto-launch Chrome with remote debugging enabled.

        Args:
            port: The debugging port to use

        Returns:
            True if Chrome was successfully launched
        """
        self.logger.info(f"Attempting to launch Chrome with remote debugging on port {port}...")

        # Find Chrome executable based on platform
        chrome_path = None
        system = platform.system()

        if system == "Windows":
            # Check common Windows paths
            possible_paths = [
                os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
                os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
                os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    chrome_path = path
                    break
        elif system == "Darwin":  # macOS
            mac_paths = [
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                os.path.expanduser("~/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            ]
            for path in mac_paths:
                if os.path.exists(path):
                    chrome_path = path
                    break
        else:  # Linux
            chrome_path = shutil.which("google-chrome") or shutil.which("chrome") or shutil.which("chromium")

        if not chrome_path:
            self.logger.warning("Chrome executable not found. Please install Google Chrome.")
            return False

        self.logger.info(f"Found Chrome at: {chrome_path}")

        # Create temp profile directory
        if system == "Windows":
            profile_dir = os.path.expandvars(r"%TEMP%\tiktok_studio_chrome_profile")
        else:
            profile_dir = "/tmp/tiktok_studio_chrome_profile"

        # Launch Chrome with debugging
        try:
            cmd = [
                chrome_path,
                f"--remote-debugging-port={port}",
                f"--user-data-dir={profile_dir}",
                "--no-first-run",
                "--no-default-browser-check",
                STUDIO_HOME_URL
            ]

            self.logger.info(f"Launching Chrome: {' '.join(cmd)}")

            # Use Popen to launch without blocking
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True if system != "Windows" else False,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if system == "Windows" else 0
            )

            # Wait for Chrome to start and enable debugging endpoint
            self.logger.info("Waiting for Chrome to start...")
            for i in range(10):  # Wait up to 5 seconds
                await asyncio.sleep(0.5)
                if await self._verify_cdp_endpoint(port):
                    self.logger.info(f"Chrome is ready on port {port}")
                    return True

            self.logger.warning("Chrome started but debugging endpoint not available")
            return False

        except Exception as e:
            self.logger.error(f"Failed to launch Chrome: {e}")
            return False

    async def _check_logged_in(self) -> bool:
        """
        Check if we're currently logged in to TikTok Studio.

        Returns:
            True if logged in
        """
        try:
            # Wait a moment for page to settle
            await asyncio.sleep(0.5)

            current_url = self.page.url

            # If we're on login page, not logged in
            if "login" in current_url.lower():
                return False

            # If we're on Studio page, likely logged in
            if "tiktokstudio" in current_url:
                # Try to find content that indicates logged in state
                # Look for the sidebar with videos or creator name
                try:
                    await self.page.wait_for_selector('[class*="sidebar"]', timeout=2000)
                    return True
                except:
                    pass

                # Alternative: check if there's no login button visible
                login_button = await self.page.query_selector('button:has-text("Log in")')
                if not login_button:
                    return True

            return False

        except Exception as e:
            self.logger.debug(f"Error checking login status: {e}")
            return False

    async def scrape_all_videos(
        self,
        video_scraper=None,
        analyzer=None,
        skip_download: bool = False,
        skip_analysis: bool = False,
    ) -> dict:
        """
        Scrape all videos from TikTok Studio using batch processing.

        Videos are collected and processed in batches, so files appear
        in real-time as each batch is processed. Optionally downloads
        and analyzes each video immediately after screenshots.

        Args:
            video_scraper: TikTokScraper instance for downloading videos
            analyzer: VideoAnalyzer instance for analyzing videos
            skip_download: If True, don't download videos
            skip_analysis: If True, don't analyze videos

        Returns:
            Dictionary with results summary
        """
        results = {
            "total": 0,
            "processed": 0,
            "downloaded": 0,
            "analyzed": 0,
            "skipped": 0,
            "failed": 0,
            "videos": [],
            "errors": [],
        }

        self.logger.info("Starting to scrape all videos from TikTok Studio...")

        try:
            # If connected to existing browser via CDP, don't navigate
            # User already has the correct analytics page open with sidebar
            if self._connected_to_existing:
                self.logger.info("Using current page from CDP connection (already on analytics page)")
                await asyncio.sleep(1)
            else:
                # Launched new browser - navigate to Studio home (NOT /content)
                # Studio home will show analytics page with sidebar
                await self.page.goto(STUDIO_HOME_URL, wait_until="networkidle", timeout=30000)
                await asyncio.sleep(3)

            # Process videos in batches as they are discovered
            batch_num = 0
            global_idx = 0

            async for video_batch in self._get_video_list_batched(batch_size=50):
                batch_num += 1
                batch_size = len(video_batch)
                results["total"] += batch_size

                print(f"\n{'=' * 60}")
                print(f"  PROCESSING BATCH {batch_num} ({batch_size} videos)")
                print(f"  Total discovered so far: {results['total']}")
                print(f"{'=' * 60}")

                # Process each video in the batch
                for batch_idx, video_info in enumerate(video_batch, 1):
                    global_idx += 1
                    video_id = video_info.get("video_id", f"unknown_{global_idx}")

                    self.logger.info(f"\n[Batch {batch_num}, {batch_idx}/{batch_size}] Processing video: {video_id}")

                    try:
                        # Check if already processed
                        if self._is_video_processed(video_id):
                            self.logger.info(f"  Skipping (already processed): {video_id}")
                            results["skipped"] += 1
                            continue

                        # Process the video (screenshots + optional download)
                        video_result = await self._process_single_video(
                            video_info,
                            video_scraper=video_scraper if not skip_download else None,
                        )

                        if video_result.get("success"):
                            results["processed"] += 1
                            results["videos"].append(video_result)

                            # Track download status
                            if video_result.get("downloaded"):
                                results["downloaded"] += 1

                                # Run analysis immediately after download
                                if analyzer and not skip_analysis and video_result.get("video_path"):
                                    self.logger.info(f"  Analyzing video...")
                                    try:
                                        analysis_result = analyzer.analyze_video(video_result["video_path"])
                                        if analysis_result and not analysis_result.errors:
                                            # Update metadata JSON with analysis
                                            json_path = video_result.get("json_path")
                                            if json_path:
                                                analyzer.update_metadata_file(json_path, analysis_result)
                                            results["analyzed"] += 1
                                            self.logger.info(f"  Analysis complete")
                                        else:
                                            self.logger.warning(f"  Analysis had errors: {analysis_result.errors if analysis_result else 'No result'}")
                                    except Exception as e:
                                        self.logger.error(f"  Analysis failed: {e}")

                            self.logger.info(f"  Successfully processed: {video_id}")
                        else:
                            results["failed"] += 1
                            results["errors"].append({
                                "video_id": video_id,
                                "error": video_result.get("error", "Unknown error")
                            })
                            self.logger.warning(f"  Failed to process: {video_id}")

                    except Exception as e:
                        results["failed"] += 1
                        results["errors"].append({
                            "video_id": video_id,
                            "error": str(e)
                        })
                        self.logger.error(f"  Error processing {video_id}: {e}")

                # Print batch summary
                print(f"\n  Batch {batch_num} complete: {results['processed']} processed, {results['downloaded']} downloaded, {results['analyzed']} analyzed, {results['failed']} failed")

            if results["total"] == 0:
                self.logger.warning("No videos found in TikTok Studio")

            return results

        except Exception as e:
            # Check if it's a browser disconnection error
            error_msg = str(e).lower()
            if any(x in error_msg for x in ['target closed', 'disconnected', 'connection closed', 'browser closed']):
                self.logger.error("Browser was closed by user")
                print("\n" + "=" * 70)
                print("  BROWSER CLOSED")
                print("=" * 70)
                print(f"  Progress saved: {results['processed']} processed, {results['downloaded']} downloaded, {results['analyzed']} analyzed")
                print(f"  Remaining: {results['total'] - results['processed'] - results['skipped']} videos")
                print("  Exiting gracefully...")
                print("=" * 70 + "\n")
                results["error"] = "Browser closed by user"
                results["interrupted"] = True
                return results
            else:
                # Re-raise other exceptions
                raise

    async def _get_cookies_for_api(self) -> dict:
        """
        Extract cookies from the Playwright browser session for API requests.

        Returns:
            Dictionary of cookies suitable for requests library
        """
        cookies = await self.context.cookies()
        return {cookie['name']: cookie['value'] for cookie in cookies}

    def _load_api_patterns(self) -> Optional[dict]:
        """
        Load API patterns from file if available.

        Returns:
            Dictionary with API patterns or None if not found
        """
        try:
            if API_PATTERNS_FILE.exists():
                with open(API_PATTERNS_FILE, 'r', encoding='utf-8') as f:
                    patterns = json.load(f)
                    if patterns.get('video_list_api'):
                        self.logger.info("Loaded API patterns from file")
                        return patterns
        except Exception as e:
            self.logger.warning(f"Failed to load API patterns: {e}")
        return None

    def _get_api_params(self) -> dict:
        """
        Get standard query parameters for TikTok Studio API requests.

        Uses extracted patterns if available, otherwise falls back to defaults.

        Returns:
            Dictionary of query parameters
        """
        # Check if we have extracted patterns with query params
        if self._api_patterns and self._api_patterns.get('video_list_api'):
            pattern = self._api_patterns['video_list_api']
            if pattern.get('query_params'):
                self.logger.info("Using extracted API query parameters")
                # Filter out pagination params - those are handled separately
                params = {k: v for k, v in pattern['query_params'].items()
                          if k not in ('cursor', 'count')}
                return params

        # Default parameters
        import time
        return {
            'aid': '1988',
            'app_language': 'en',
            'app_name': 'tiktok_creator_center',
            'browser_language': 'en-US',
            'browser_name': 'Mozilla',
            'browser_platform': 'Win32',
            'channel': 'tiktok_web',
            'device_platform': 'web_pc',
            'priority_region': '',
            'region': '',
        }

    def _fetch_video_list_page(self, cookies: dict, params: dict, cursor: int) -> dict:
        """
        Synchronous helper to fetch a single page of video list from API.

        Uses extracted endpoint, method, and headers from patterns if available.
        Supports both GET and POST methods based on what was captured during extraction.

        Args:
            cookies: Session cookies
            params: API parameters
            cursor: Pagination cursor

        Returns:
            API response as dictionary
        """
        import requests

        # Default configuration
        base_url = "https://www.tiktok.com/api/creator/item/list/"
        method = "GET"
        headers = {
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Content-Type': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': 'https://www.tiktok.com/tiktokstudio/content',
        }
        request_body = None

        # Use endpoint, method, and headers from patterns if available
        if self._api_patterns and self._api_patterns.get('video_list_api'):
            pattern = self._api_patterns['video_list_api']

            if pattern.get('endpoint'):
                base_url = pattern['endpoint']
                self.logger.info(f"Using extracted endpoint: {base_url}")

            if pattern.get('method'):
                method = pattern['method'].upper()
                self.logger.info(f"Using extracted method: {method}")

            # Merge in headers from extracted patterns
            if pattern.get('headers'):
                headers.update(pattern['headers'])

            # For POST requests, build request body
            if method == "POST":
                request_body = {
                    "cursor": cursor,
                    "size": 50,
                    "query": {
                        "sort_orders": [],
                    },
                    "conditions": [],
                    "is_recent_posts": False,
                }

        # Include query params in URL
        request_params = {**params, 'cursor': cursor, 'count': 50}

        # Make request using appropriate method
        if method == "POST":
            self.logger.debug(f"POST {base_url} with body: {request_body}")
            response = requests.post(
                base_url,
                params=request_params,
                json=request_body,
                cookies=cookies,
                headers=headers,
                timeout=30
            )
        else:
            self.logger.debug(f"GET {base_url}")
            response = requests.get(
                base_url,
                params=request_params,
                cookies=cookies,
                headers=headers,
                timeout=30
            )

        response.raise_for_status()
        return response.json()

    async def _fetch_video_list_via_browser(self, cursor: int = 0) -> dict:
        """
        Fetch video list using browser's fetch API (through Playwright).

        This ensures fresh security tokens are generated by TikTok's JavaScript.
        The browser automatically handles X-Bogus and other security tokens.

        Args:
            cursor: Pagination cursor

        Returns:
            API response as dictionary
        """
        # Default endpoint (may be overridden by patterns)
        endpoint = "https://www.tiktok.com/tiktok/creator/manage/item_list/v1/"
        method = "POST"

        # Use patterns if available
        if self._api_patterns and self._api_patterns.get('video_list_api'):
            pattern = self._api_patterns['video_list_api']
            if pattern.get('endpoint'):
                endpoint = pattern['endpoint']
                self.logger.info(f"Using extracted endpoint: {endpoint}")
            if pattern.get('method'):
                method = pattern['method'].upper()

        # Build request body for POST
        request_body = {
            "cursor": cursor,
            "size": 50,
            "query": {"sort_orders": []},
            "conditions": [],
            "is_recent_posts": False,
        }

        self.logger.info(f"Fetching video list via browser ({method} {endpoint}, cursor={cursor})")

        # Execute fetch in browser context - browser handles all security tokens
        try:
            if method == "POST":
                result = await self.page.evaluate('''
                    async (args) => {
                        try {
                            const response = await fetch(args.endpoint, {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json',
                                    'Accept': 'application/json',
                                },
                                body: JSON.stringify(args.body),
                                credentials: 'include'
                            });
                            if (!response.ok) {
                                return { error: `HTTP ${response.status}: ${response.statusText}` };
                            }
                            return await response.json();
                        } catch (e) {
                            return { error: e.message };
                        }
                    }
                ''', {"endpoint": endpoint, "body": request_body})
            else:
                # GET request with query params
                result = await self.page.evaluate('''
                    async (args) => {
                        try {
                            const url = new URL(args.endpoint);
                            url.searchParams.set('cursor', args.cursor);
                            url.searchParams.set('count', '50');

                            const response = await fetch(url.toString(), {
                                method: 'GET',
                                headers: {
                                    'Accept': 'application/json',
                                },
                                credentials: 'include'
                            });
                            if (!response.ok) {
                                return { error: `HTTP ${response.status}: ${response.statusText}` };
                            }
                            return await response.json();
                        } catch (e) {
                            return { error: e.message };
                        }
                    }
                ''', {"endpoint": endpoint, "cursor": cursor})

            if result.get('error'):
                raise Exception(result['error'])

            return result

        except Exception as e:
            self.logger.warning(f"Browser fetch failed: {e}")
            raise

    async def _get_video_list_via_api(self) -> list:
        """
        Fetch video list using TikTok Studio's internal API.

        This is much faster than UI automation as it fetches all videos in bulk
        with pagination, and includes video metadata and download URLs.

        Returns:
            List of video info dictionaries with metadata and playAddr (download URLs)
        """
        self.logger.info("Attempting to fetch video list via API...")

        try:
            # Get cookies from browser session
            cookies = await self._get_cookies_for_api()
            params = self._get_api_params()

            all_videos = []
            cursor = 0
            has_more = True
            loop = asyncio.get_event_loop()

            while has_more:
                self.logger.info(f"Fetching videos from API (cursor={cursor})...")

                # Use browser-based fetch (handles security tokens automatically)
                try:
                    data = await self._fetch_video_list_via_browser(cursor)
                except Exception as e:
                    self.logger.warning(f"Browser fetch failed, trying requests fallback: {e}")
                    # Fallback to requests-based fetch (may fail due to expired tokens)
                    data = await loop.run_in_executor(
                        None,
                        lambda c=cursor: self._fetch_video_list_page(cookies, params, c)
                    )

                # Check for errors - handle both status formats
                status = data.get('status_code', data.get('statusCode', -1))
                if status != 0 and status != -1:
                    self.logger.warning(f"API error: {data.get('status_msg', data.get('statusMsg', 'Unknown'))}")
                    break

                # Extract video items - handle different response formats
                items = data.get('item_list', data.get('itemList', data.get('items', [])))

                if not items:
                    self.logger.info("No more videos in API response")
                    break

                for item in items:
                    # Handle different field naming conventions (camelCase vs snake_case)
                    video_id = str(item.get('item_id', item.get('itemId', '')))
                    if not video_id:
                        continue

                    video_info = {
                        'video_id': video_id,
                        'analytics_url': f"https://www.tiktok.com/tiktokstudio/analytics/{video_id}/overview",
                        # Include metadata from API
                        'api_metadata': {
                            'desc': item.get('desc', ''),
                            'create_time': item.get('create_time', item.get('createTime', 0)),
                            'duration': item.get('duration', 0),
                            'play_count': item.get('playCount', item.get('play_count', 0)),
                            'like_count': item.get('LikeCount', item.get('like_count', item.get('likeCount', 0))),
                            'comment_count': item.get('comment_count', item.get('commentCount', 0)),
                            'share_count': item.get('share_count', item.get('shareCount', 0)),
                            'favorite_count': item.get('favorite_count', item.get('favoriteCount', 0)),
                            'cover_url': item.get('cover_url', item.get('coverUrl', [])),
                            'play_addr': item.get('playAddr', item.get('play_addr', [])),
                        }
                    }
                    all_videos.append(video_info)

                self.logger.info(f"Fetched {len(items)} videos (total: {len(all_videos)})")

                # Check for more pages
                has_more = data.get('has_more', data.get('hasMore', False))
                cursor = data.get('cursor', cursor + len(items))

                # Small delay between requests
                await asyncio.sleep(0.3)

            self.logger.info(f"API fetch complete: {len(all_videos)} videos found")
            return all_videos

        except Exception as e:
            self.logger.warning(f"API fetch failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return []

    async def _get_video_list_batched(self, batch_size: int = 50):
        """
        Get videos from TikTok Studio sidebar and yield them in batches.

        First attempts to fetch via API (much faster), falls back to UI automation
        if API fails.

        This is an async generator that yields batches of videos as they are
        collected, allowing processing to start before all videos are discovered.

        Args:
            batch_size: Number of videos per batch (default: 50)

        Yields:
            List of video info dictionaries (batch_size at a time)
        """
        # First, try API-based fetching (much faster)
        try:
            api_videos = await self._get_video_list_via_api()
            if api_videos:
                self.logger.info(f"Successfully fetched {len(api_videos)} videos via API!")

                # Yield in batches
                for i in range(0, len(api_videos), batch_size):
                    batch = api_videos[i:i + batch_size]
                    self.logger.info(f"Yielding API batch {i // batch_size + 1} ({len(batch)} videos)")
                    yield batch

                return  # API succeeded, no need for UI automation
        except Exception as e:
            self.logger.warning(f"API fetch failed, falling back to UI automation: {e}")

        # Fallback: UI automation (scrolling + clicking)
        self.logger.info("Using UI automation to fetch video list...")

        videos = []
        seen_video_ids = set()
        total_yielded = 0

        try:
            self.logger.info("Loading video list from sidebar...")

            # Wait for sidebar to load
            try:
                await self.page.wait_for_selector('[class*="sidebar"]', timeout=2000)
                self.logger.info("Sidebar detected")
            except:
                self.logger.warning("Sidebar selector not found, continuing anyway")

            await asyncio.sleep(1)

            # Find the scrollable container (sidebar video list)
            sidebar_selectors = [
                '[data-e2e="components_analytics_VideoSelector_VideoSelectContainer"]',
                '[class*="VideoSelector"]',
                '[class*="sidebar"] [class*="scroll"]',
                '[class*="sidebar"]'
            ]

            sidebar_container = None
            for selector in sidebar_selectors:
                try:
                    sidebar_container = await self.page.query_selector(selector)
                    if sidebar_container:
                        self.logger.info(f"Found sidebar container: {selector}")
                        break
                except:
                    continue

            if not sidebar_container:
                self.logger.warning("Could not find scrollable sidebar container, using page scroll")

            # Handle infinite scroll - scroll until all videos are loaded
            self.logger.info("Loading all videos via infinite scroll...")
            previous_count = 0
            scroll_attempts = 0
            max_scroll_attempts = 50

            while scroll_attempts < max_scroll_attempts:
                video_buttons = await self.page.query_selector_all(
                    'button img[data-tt="components_AnalyticsVideoSelector_Image"]'
                )
                current_count = len(video_buttons)

                self.logger.info(f"Scroll attempt {scroll_attempts + 1}: Found {current_count} videos")

                if current_count == previous_count and scroll_attempts > 0:
                    self.logger.info("No new videos loaded, finished scrolling")
                    break

                previous_count = current_count

                if sidebar_container:
                    await self.page.evaluate('''(element) => {
                        element.scrollTop = element.scrollHeight;
                    }''', sidebar_container)
                else:
                    await self.page.evaluate('window.scrollTo(0, document.body.scrollHeight)')

                await asyncio.sleep(1.5)
                scroll_attempts += 1

            # Get all video button elements after scrolling
            video_images = await self.page.query_selector_all(
                'button img[data-tt="components_AnalyticsVideoSelector_Image"]'
            )

            self.logger.info(f"Found {len(video_images)} total video images after scrolling")

            self.logger.info("Getting clickable button elements...")
            video_buttons = await self.page.query_selector_all('button:has(img[data-tt="components_AnalyticsVideoSelector_Image"])')

            self.logger.info(f"Found {len(video_buttons)} clickable video buttons")

            # Click each button to extract video ID, yield in batches
            for idx, button in enumerate(video_buttons, 1):
                try:
                    # Log progress every 50 videos
                    if idx % 50 == 0:
                        self.logger.info(f"Extracting video ID {idx}/{len(video_buttons)}...")

                    await button.click()
                    await asyncio.sleep(0.3)

                    new_url = self.page.url
                    match = re.search(r'/analytics/(\d+)', new_url)

                    if match:
                        video_id = match.group(1)

                        if video_id in seen_video_ids:
                            continue

                        seen_video_ids.add(video_id)
                        analytics_url = f"https://www.tiktok.com/tiktokstudio/analytics/{video_id}/overview"

                        videos.append({
                            "video_id": video_id,
                            "analytics_url": analytics_url,
                        })

                        self.logger.debug(f"  [{idx}] Found video: {video_id}")

                        # Yield batch when we have enough videos
                        if len(videos) >= batch_size:
                            total_yielded += len(videos)
                            self.logger.info(f"Yielding batch of {len(videos)} videos (total yielded: {total_yielded})")
                            yield videos
                            videos = []

                except Exception as e:
                    self.logger.debug(f"Error processing button {idx}: {e}")
                    continue

            # Yield any remaining videos
            if videos:
                total_yielded += len(videos)
                self.logger.info(f"Yielding final batch of {len(videos)} videos (total yielded: {total_yielded})")
                yield videos

            self.logger.info(f"Successfully extracted {total_yielded} unique video IDs")

        except Exception as e:
            self.logger.error(f"Error getting video list: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Yield any videos we collected before the error
            if videos:
                yield videos

    def _is_video_processed(self, video_id: str) -> bool:
        """
        Check if a video has already been processed (screenshots exist).

        Args:
            video_id: TikTok video ID

        Returns:
            True if all screenshots exist
        """
        # Search for existing screenshots in output directory
        for folder in self.output_dir.rglob("*"):
            if folder.is_dir():
                overview_path = folder / f"{video_id}_overview.png"
                viewers_path = folder / f"{video_id}_viewers.png"
                engagement_path = folder / f"{video_id}_engagement.png"

                if overview_path.exists() and viewers_path.exists() and engagement_path.exists():
                    return True

        return False

    async def _process_single_video(self, video_info: dict, video_scraper=None) -> dict:
        """
        Process a single video: capture all 3 tab screenshots, extract video URL,
        and optionally download the video immediately.

        Args:
            video_info: Dictionary with video_id and analytics_url
            video_scraper: Optional TikTokScraper instance for downloading videos

        Returns:
            Result dictionary with success status and data
        """
        result = {
            "success": False,
            "video_id": video_info["video_id"],
            "video_url": None,
            "screenshots": {},
            "metadata": {},
            "downloaded": False,
            "video_path": None,
            "json_path": None,
        }

        try:
            video_id = video_info["video_id"]
            analytics_url = video_info["analytics_url"]

            # Navigate to video analytics page
            self.logger.info(f"  Navigating to analytics: {analytics_url}")
            await self.page.goto(analytics_url, wait_until="networkidle", timeout=60000)
            await asyncio.sleep(3)

            # Extract metadata from page (username, date, etc.)
            metadata = await self._extract_video_metadata()
            result["metadata"] = metadata

            # Determine output folder
            # Use provided account_username if per-video extraction fails
            username = metadata.get("username")
            if not username or username == "unknown_user":
                username = self.account_username or "unknown_user"
            # Normalize username for consistent folder naming
            username = normalize_username(username)

            create_date = metadata.get("create_date", timestamp_to_date(None))
            # Validate date - use today if invalid (e.g., 7592-00-00)
            try:
                datetime.strptime(create_date, "%Y-%m-%d")
                # Also check for clearly invalid dates
                year = int(create_date.split("-")[0])
                if year < 2016 or year > 2030:  # TikTok didn't exist before 2016
                    raise ValueError("Invalid year")
            except (ValueError, IndexError):
                create_date = datetime.now().strftime("%Y-%m-%d")

            output_folder = self.output_dir / username / create_date
            ensure_directory(output_folder)

            # Verify folder was created
            if not output_folder.exists():
                raise RuntimeError(f"Failed to create output folder: {output_folder}")
            self.logger.info(f"  Output folder: {output_folder}")

            # Capture screenshots of all 3 tabs
            screenshots = {}

            # Tab 1: Overview (סקירה כללית) - should be default
            self.logger.info(f"  Capturing Overview tab...")
            overview_path = output_folder / f"{video_id}_overview.png"
            await self._capture_tab_screenshot("overview", overview_path)
            screenshots["overview"] = str(overview_path)

            # Tab 2: Viewers (צופים)
            self.logger.info(f"  Capturing Viewers tab...")
            await self._click_tab("viewers")
            viewers_path = output_folder / f"{video_id}_viewers.png"
            await self._capture_tab_screenshot("viewers", viewers_path)
            screenshots["viewers"] = str(viewers_path)

            # Tab 3: Engagement (מעורבות)
            self.logger.info(f"  Capturing Engagement tab...")
            await self._click_tab("engagement")
            engagement_path = output_folder / f"{video_id}_engagement.png"
            await self._capture_tab_screenshot("engagement", engagement_path)
            screenshots["engagement"] = str(engagement_path)

            result["screenshots"] = screenshots

            # Extract video URL by clicking thumbnail
            self.logger.info(f"  Extracting video URL...")
            video_url = await self._extract_video_url(video_id, username=username)
            result["video_url"] = video_url

            result["success"] = True
            result["output_folder"] = str(output_folder)

            # Download video immediately if scraper provided
            if video_scraper and video_url:
                self.logger.info(f"  Downloading video...")
                try:
                    download_success, download_message = video_scraper.download_video(video_url)
                    if download_success:
                        self.logger.info(f"  Downloaded: {download_message}")
                        result["downloaded"] = True
                        # Set paths for downloaded files
                        video_path = output_folder / f"{video_id}.mp4"
                        json_path = output_folder / f"{video_id}.json"
                        if video_path.exists():
                            result["video_path"] = str(video_path)
                        if json_path.exists():
                            result["json_path"] = str(json_path)
                    else:
                        self.logger.warning(f"  Download failed: {download_message}")
                        result["download_error"] = download_message
                except Exception as e:
                    self.logger.error(f"  Download error: {e}")
                    result["download_error"] = str(e)

        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"  Error processing video: {e}")

        return result

    async def _extract_video_metadata(self) -> dict:
        """
        Extract video metadata from the analytics page.

        Returns:
            Dictionary with username, date, title, etc.
        """
        metadata = {}

        try:
            # Extract from URL - pattern: /analytics/{video_id}
            current_url = self.page.url

            # Try to extract username from page content
            # Look for username in the page header or sidebar
            username_selectors = [
                '[class*="username"]',
                '[class*="author"]',
                '[class*="creator"]',
                'a[href*="/@"]',
            ]

            for selector in username_selectors:
                try:
                    element = await self.page.query_selector(selector)
                    if element:
                        text = await element.inner_text()
                        if text and text.startswith("@"):
                            metadata["username"] = text[1:]  # Remove @
                            break
                        elif text:
                            metadata["username"] = text
                            break
                except:
                    continue

            # Try to extract date from page
            # Based on screenshots, date appears in format like "7.6.2025" with Hebrew text
            date_pattern = r'(\d{1,2})\.(\d{1,2})\.(\d{4})'
            page_content = await self.page.content()
            date_match = re.search(date_pattern, page_content)
            if date_match:
                day, month, year = date_match.groups()
                metadata["create_date"] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"

            # Try to extract title/description
            title_selectors = [
                '[class*="title"]',
                '[class*="description"]',
                '[class*="caption"]',
            ]

            for selector in title_selectors:
                try:
                    element = await self.page.query_selector(selector)
                    if element:
                        text = await element.inner_text()
                        if text and len(text) > 3:
                            metadata["title"] = text[:200]  # Limit length
                            break
                except:
                    continue

        except Exception as e:
            self.logger.debug(f"Error extracting metadata: {e}")

        return metadata

    async def _click_tab(self, tab_name: str) -> bool:
        """
        Click on a specific analytics tab.

        Args:
            tab_name: Tab identifier (overview, viewers, engagement)

        Returns:
            True if tab was clicked successfully
        """
        try:
            hebrew_name = TAB_NAMES.get(tab_name)
            if not hebrew_name:
                return False

            # Try multiple selector strategies
            selectors = [
                f'button:has-text("{hebrew_name}")',
                f'a:has-text("{hebrew_name}")',
                f'div:has-text("{hebrew_name}")',
                f'[role="tab"]:has-text("{hebrew_name}")',
                f'text="{hebrew_name}"',
            ]

            for selector in selectors:
                try:
                    element = await self.page.query_selector(selector)
                    if element:
                        await element.click()
                        await asyncio.sleep(2)  # Wait for tab content to load
                        return True
                except:
                    continue

            # Fallback: try clicking by position based on URL pattern
            # Tabs change URL: /analytics/{id} -> /analytics/{id}/viewers -> /analytics/{id}/engagement
            tab_url_suffix = {
                "viewers": "/viewers",
                "engagement": "/engagement",
            }

            if tab_name in tab_url_suffix:
                current_url = self.page.url
                base_url = re.sub(r'/(viewers|engagement)$', '', current_url)
                new_url = base_url + tab_url_suffix[tab_name]
                await self.page.goto(new_url, wait_until="networkidle", timeout=30000)
                await asyncio.sleep(2)
                return True

            return False

        except Exception as e:
            self.logger.debug(f"Error clicking tab {tab_name}: {e}")
            return False

    async def _capture_tab_screenshot(self, tab_name: str, output_path: Path) -> bool:
        """
        Capture screenshot of the current tab.

        Args:
            tab_name: Tab identifier for logging
            output_path: Path to save screenshot

        Returns:
            True if screenshot was captured
        """
        try:
            # Wait for content to fully load
            await asyncio.sleep(1.5)

            # Take full page screenshot
            await self.page.screenshot(path=str(output_path), full_page=True)

            # Verify file was actually saved
            if not output_path.exists():
                self.logger.error(f"    Screenshot NOT saved: {output_path}")
                return False

            self.logger.info(f"    Saved: {output_path.name}")
            return True

        except Exception as e:
            self.logger.error(f"    Error capturing screenshot: {e}")
            return False

    async def _extract_video_url(self, video_id: str, username: str = None) -> Optional[str]:
        """
        Extract the TikTok video URL by clicking on the thumbnail.

        Args:
            video_id: Video ID for constructing fallback URL
            username: Username for constructing correct URL format

        Returns:
            Video URL string or None
        """
        try:
            # Try to find and click the video thumbnail
            thumbnail_selectors = [
                '[class*="thumbnail"]',
                '[class*="video-preview"]',
                '[class*="cover"]',
                'video',
                'img[src*="tiktok"]',
            ]

            for selector in thumbnail_selectors:
                try:
                    element = await self.page.query_selector(selector)
                    if element:
                        # Check if clicking opens a new tab
                        async with self.context.expect_page() as new_page_info:
                            await element.click()
                            try:
                                new_page = await asyncio.wait_for(new_page_info.value, timeout=5)
                                video_url = new_page.url
                                await new_page.close()

                                if "tiktok.com" in video_url and "/video/" in video_url:
                                    return video_url
                            except asyncio.TimeoutError:
                                pass
                except:
                    continue

            # Fallback: construct URL from video_id and username
            # TikTok video URL format requires username: https://www.tiktok.com/@username/video/{id}
            if username and username != "unknown_user":
                # Remove @ prefix if already present
                clean_username = username.lstrip("@")
                fallback_url = f"https://www.tiktok.com/@{clean_username}/video/{video_id}"
            else:
                # Last resort: try without username (may not work)
                fallback_url = f"https://www.tiktok.com/video/{video_id}"

            self.logger.info(f"    Using fallback URL: {fallback_url}")
            return fallback_url

        except Exception as e:
            self.logger.debug(f"Error extracting video URL: {e}")
            # Return constructed URL as fallback
            if username and username != "unknown_user":
                clean_username = username.lstrip("@")
                return f"https://www.tiktok.com/@{clean_username}/video/{video_id}"
            return f"https://www.tiktok.com/video/{video_id}"

    async def close(self):
        """Clean up browser resources."""
        try:
            if self.page and not self._connected_to_existing:
                await self.page.close()
            if self.context and not self._connected_to_existing:
                await self.context.close()
            if self.browser:
                if self._connected_to_existing:
                    # For connected browsers, just disconnect (don't close the browser)
                    await self.browser.close()
                    self.logger.info("Disconnected from browser")
                else:
                    # For launched browsers, close completely
                    await self.browser.close()
                    self.logger.info("Browser closed")
            if self.playwright:
                await self.playwright.stop()
        except Exception as e:
            self.logger.debug(f"Error closing browser: {e}")


async def run_studio_scraper(
    output_dir: str,
    logger,
    browser_type: str = "chromium",
    skip_download: bool = False,
    skip_analysis: bool = False,
    cdp_port: int = None,
    video_scraper=None,
    analyzer=None,
    username: str = None,
) -> dict:
    """
    Main entry point for running the TikTok Studio scraper.

    Args:
        output_dir: Output directory for files
        logger: Logger instance
        browser_type: Browser to use (chromium, firefox, webkit)
        skip_download: If True, only capture screenshots
        skip_analysis: If True, skip video analysis
        cdp_port: Custom CDP port for connecting to existing browser
        video_scraper: TikTokScraper instance for downloading videos (optional)
        analyzer: VideoAnalyzer instance for analyzing videos (optional)
        username: TikTok username for constructing video URLs

    Returns:
        Results summary dictionary
    """
    scraper = TikTokStudioScraper(output_dir, logger, browser_type, cdp_port=cdp_port, username=username)

    try:
        # Initialize and login
        if not await scraper.initialize():
            return {"success": False, "error": "Failed to initialize/login"}

        # Scrape all videos, with optional immediate download and analysis
        results = await scraper.scrape_all_videos(
            video_scraper=video_scraper,
            analyzer=analyzer,
            skip_download=skip_download,
            skip_analysis=skip_analysis,
        )
        results["success"] = True

        return results

    except Exception as e:
        logger.error(f"Studio scraper error: {e}")
        return {"success": False, "error": str(e)}

    finally:
        await scraper.close()
