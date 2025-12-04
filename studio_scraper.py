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
from pathlib import Path
from typing import Optional

from playwright.async_api import async_playwright, Browser, Page, BrowserContext

from utils import ensure_directory, timestamp_to_date


# TikTok Studio URLs
STUDIO_HOME_URL = "https://www.tiktok.com/tiktokstudio"
STUDIO_CONTENT_URL = "https://www.tiktok.com/tiktokstudio/content"
TIKTOK_LOGIN_URL = "https://www.tiktok.com/login"

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

    def __init__(self, output_dir: str, logger, browser_type: str = "chromium"):
        """
        Initialize the Studio scraper.

        Args:
            output_dir: Base directory for saving screenshots and videos
            logger: Logger instance for status messages
            browser_type: Playwright browser type (chromium, firefox, webkit)
        """
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.browser_type = browser_type

        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.playwright = None

        self._is_logged_in = False
        self._processed_videos = set()

    async def initialize(self) -> bool:
        """
        Initialize browser and attempt authentication.

        Returns:
            True if successfully logged in, False otherwise
        """
        self.logger.info("Initializing TikTok Studio scraper...")

        # Launch Playwright
        self.playwright = await async_playwright().start()

        # Select browser type
        if self.browser_type == "firefox":
            browser_launcher = self.playwright.firefox
        elif self.browser_type == "webkit":
            browser_launcher = self.playwright.webkit
        else:
            browser_launcher = self.playwright.chromium

        # Launch browser in headful mode so user can see/interact
        # Note: --start-maximized doesn't work on macOS/Linux, so we rely on viewport instead
        self.browser = await browser_launcher.launch(
            headless=False,
        )

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
        await self.page.goto(STUDIO_HOME_URL, wait_until="networkidle", timeout=60000)

        # Check if we're logged in
        self._is_logged_in = await self._check_logged_in()

        if not self._is_logged_in:
            self.logger.info("Not logged in. Please log in manually in the browser window.")
            print("\n" + "=" * 50)
            print("  MANUAL LOGIN REQUIRED")
            print("=" * 50)
            print("  Please log in to TikTok in the browser window.")
            print("  Press Enter here when you're done logging in...")
            print("=" * 50)

            # Wait for user to press Enter
            await asyncio.get_event_loop().run_in_executor(None, input)

            # Re-check login status
            await self.page.goto(STUDIO_HOME_URL, wait_until="networkidle", timeout=60000)
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

    async def _check_logged_in(self) -> bool:
        """
        Check if we're currently logged in to TikTok Studio.

        Returns:
            True if logged in
        """
        try:
            # Wait a moment for page to settle
            await self.page.wait_for_timeout(2000)

            current_url = self.page.url

            # If we're on login page, not logged in
            if "login" in current_url.lower():
                return False

            # If we're on Studio page, likely logged in
            if "tiktokstudio" in current_url:
                # Try to find content that indicates logged in state
                # Look for the sidebar with videos or creator name
                try:
                    await self.page.wait_for_selector('[class*="sidebar"]', timeout=5000)
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

    async def scrape_all_videos(self) -> dict:
        """
        Scrape all videos from TikTok Studio.

        Returns:
            Dictionary with results summary
        """
        results = {
            "total": 0,
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "videos": [],
            "errors": [],
        }

        self.logger.info("Starting to scrape all videos from TikTok Studio...")

        # Navigate to content page to see video list
        await self.page.goto(STUDIO_CONTENT_URL, wait_until="networkidle", timeout=60000)
        await self.page.wait_for_timeout(3000)

        # Get list of videos from the sidebar/content list
        video_elements = await self._get_video_list()

        if not video_elements:
            self.logger.warning("No videos found in TikTok Studio")
            return results

        results["total"] = len(video_elements)
        self.logger.info(f"Found {len(video_elements)} videos to process")

        # Process each video
        for idx, video_info in enumerate(video_elements, 1):
            video_id = video_info.get("video_id", f"unknown_{idx}")

            self.logger.info(f"\n[{idx}/{len(video_elements)}] Processing video: {video_id}")

            try:
                # Check if already processed
                if self._is_video_processed(video_id):
                    self.logger.info(f"  Skipping (already processed): {video_id}")
                    results["skipped"] += 1
                    continue

                # Process the video
                video_result = await self._process_single_video(video_info)

                if video_result.get("success"):
                    results["processed"] += 1
                    results["videos"].append(video_result)
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

        return results

    async def _get_video_list(self) -> list:
        """
        Get list of all videos from TikTok Studio content page.

        Returns:
            List of video info dictionaries
        """
        videos = []

        try:
            # Wait for video list to load
            await self.page.wait_for_timeout(2000)

            # Look for video items in the content list
            # Based on screenshots, videos appear in a sidebar/list with thumbnails
            video_selectors = [
                '[class*="video-item"]',
                '[class*="content-item"]',
                '[class*="post-item"]',
                'a[href*="/analytics/"]',
                '[data-e2e*="video"]',
            ]

            video_elements = []
            for selector in video_selectors:
                elements = await self.page.query_selector_all(selector)
                if elements:
                    video_elements = elements
                    self.logger.info(f"Found videos using selector: {selector}")
                    break

            # If no specific selectors work, try to find links to analytics pages
            if not video_elements:
                # Get all links and filter for analytics URLs
                all_links = await self.page.query_selector_all('a[href*="tiktokstudio/analytics"]')
                video_elements = all_links

            # Extract video IDs from elements
            for element in video_elements:
                try:
                    href = await element.get_attribute("href")
                    if href:
                        # Extract video ID from URL pattern: /analytics/{video_id}
                        match = re.search(r'/analytics/(\d+)', href)
                        if match:
                            video_id = match.group(1)
                            videos.append({
                                "video_id": video_id,
                                "analytics_url": href if href.startswith("http") else f"https://www.tiktok.com{href}",
                                "element": element
                            })
                except Exception as e:
                    self.logger.debug(f"Error extracting video info: {e}")
                    continue

            # Deduplicate by video_id
            seen_ids = set()
            unique_videos = []
            for v in videos:
                if v["video_id"] not in seen_ids:
                    seen_ids.add(v["video_id"])
                    unique_videos.append(v)

            return unique_videos

        except Exception as e:
            self.logger.error(f"Error getting video list: {e}")
            return []

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

    async def _process_single_video(self, video_info: dict) -> dict:
        """
        Process a single video: capture all 3 tab screenshots and extract video URL.

        Args:
            video_info: Dictionary with video_id and analytics_url

        Returns:
            Result dictionary with success status and data
        """
        result = {
            "success": False,
            "video_id": video_info["video_id"],
            "video_url": None,
            "screenshots": {},
            "metadata": {},
        }

        try:
            video_id = video_info["video_id"]
            analytics_url = video_info["analytics_url"]

            # Navigate to video analytics page
            self.logger.info(f"  Navigating to analytics: {analytics_url}")
            await self.page.goto(analytics_url, wait_until="networkidle", timeout=60000)
            await self.page.wait_for_timeout(3000)

            # Extract metadata from page (username, date, etc.)
            metadata = await self._extract_video_metadata()
            result["metadata"] = metadata

            # Determine output folder
            username = metadata.get("username", "unknown_user")
            create_date = metadata.get("create_date", timestamp_to_date(None))
            output_folder = self.output_dir / username / create_date
            ensure_directory(output_folder)

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
            video_url = await self._extract_video_url(video_id)
            result["video_url"] = video_url

            result["success"] = True
            result["output_folder"] = str(output_folder)

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
                        await self.page.wait_for_timeout(2000)  # Wait for tab content to load
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
                await self.page.wait_for_timeout(2000)
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
            await self.page.wait_for_timeout(1500)

            # Take full page screenshot
            await self.page.screenshot(path=str(output_path), full_page=True)

            self.logger.info(f"    Saved: {output_path.name}")
            return True

        except Exception as e:
            self.logger.error(f"    Error capturing screenshot: {e}")
            return False

    async def _extract_video_url(self, video_id: str) -> Optional[str]:
        """
        Extract the TikTok video URL by clicking on the thumbnail.

        Args:
            video_id: Video ID for constructing fallback URL

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

            # Fallback: construct URL from video_id
            # Standard TikTok video URL format
            fallback_url = f"https://www.tiktok.com/video/{video_id}"
            self.logger.info(f"    Using fallback URL: {fallback_url}")
            return fallback_url

        except Exception as e:
            self.logger.debug(f"Error extracting video URL: {e}")
            # Return constructed URL as fallback
            return f"https://www.tiktok.com/video/{video_id}"

    async def close(self):
        """Clean up browser resources."""
        try:
            if self.page:
                await self.page.close()
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            self.logger.info("Browser closed")
        except Exception as e:
            self.logger.debug(f"Error closing browser: {e}")


async def run_studio_scraper(
    output_dir: str,
    logger,
    browser_type: str = "chromium",
    skip_download: bool = False,
    skip_analysis: bool = False,
) -> dict:
    """
    Main entry point for running the TikTok Studio scraper.

    Args:
        output_dir: Output directory for files
        logger: Logger instance
        browser_type: Browser to use (chromium, firefox, webkit)
        skip_download: If True, only capture screenshots
        skip_analysis: If True, skip video analysis

    Returns:
        Results summary dictionary
    """
    scraper = TikTokStudioScraper(output_dir, logger, browser_type)

    try:
        # Initialize and login
        if not await scraper.initialize():
            return {"success": False, "error": "Failed to initialize/login"}

        # Scrape all videos
        results = await scraper.scrape_all_videos()
        results["success"] = True

        return results

    except Exception as e:
        logger.error(f"Studio scraper error: {e}")
        return {"success": False, "error": str(e)}

    finally:
        await scraper.close()
