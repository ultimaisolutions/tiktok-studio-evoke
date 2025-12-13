"""
TikTok API Extractor - Captures XHR/Fetch traffic from TikTok Studio pages.

Uses Playwright network interception to extract API patterns including:
- Endpoint URLs
- Request headers
- Query parameters
- Response schemas

These patterns can be applied to the scraper for resilience against UI changes.
"""

import asyncio
import json
import platform
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
from urllib.parse import urlparse, parse_qs

from playwright.async_api import async_playwright, Browser, Page, BrowserContext, Request, Response


# TikTok Studio URLs
STUDIO_HOME_URL = "https://www.tiktok.com/tiktokstudio"
STUDIO_CONTENT_URL = "https://www.tiktok.com/tiktokstudio/content"

# Keywords for identifying relevant API endpoints
RELEVANT_API_KEYWORDS = [
    'item/list', 'item_list', 'itemlist',
    'analytics', 'creator',
    'video/list', 'video_list', 'videolist',
    'content/list', 'content_list',
    'post/list', 'post_list',
]

# Response fields that indicate video/content data
VIDEO_DATA_INDICATORS = [
    'itemList', 'item_list', 'items',
    'videoList', 'video_list', 'videos',
    'contentList', 'content_list', 'contents',
    'postList', 'post_list', 'posts',
    'data',
]

# Default patterns file path
PATTERNS_FILE = Path("api_patterns.json")


class CapturedRequest:
    """Represents a captured API request."""

    def __init__(self, request: Request):
        self.url = request.url
        self.method = request.method
        self.headers = dict(request.headers)
        self.post_data = request.post_data
        self.resource_type = request.resource_type
        self.timestamp = datetime.now().isoformat()

        # Parse URL components
        parsed = urlparse(self.url)
        self.endpoint = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        self.query_params = parse_qs(parsed.query)
        # Flatten single-value lists
        self.query_params = {k: v[0] if len(v) == 1 else v for k, v in self.query_params.items()}

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "endpoint": self.endpoint,
            "method": self.method,
            "headers": self.headers,
            "query_params": self.query_params,
            "post_data": self.post_data,
            "resource_type": self.resource_type,
            "timestamp": self.timestamp,
        }


class CapturedResponse:
    """Represents a captured API response."""

    def __init__(self, url: str, status: int, headers: dict, body: Any):
        self.url = url
        self.status = status
        self.headers = headers
        self.body = body
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "status": self.status,
            "headers": self.headers,
            "body": self.body,
            "timestamp": self.timestamp,
        }


class APIPattern:
    """Represents an extracted API pattern."""

    def __init__(
        self,
        endpoint: str,
        method: str,
        headers: Dict[str, str],
        query_params: Dict[str, Any],
        request_body_schema: Optional[Dict] = None,
        response_schema: Optional[Dict] = None,
        sample_response: Optional[Dict] = None,
        captured_at: Optional[str] = None,
        category: str = "unknown",
    ):
        self.endpoint = endpoint
        self.method = method
        self.headers = headers
        self.query_params = query_params
        self.request_body_schema = request_body_schema
        self.response_schema = response_schema
        self.sample_response = sample_response
        self.captured_at = captured_at or datetime.now().isoformat()
        self.category = category

    def to_dict(self) -> dict:
        return {
            "endpoint": self.endpoint,
            "method": self.method,
            "headers": self.headers,
            "query_params": self.query_params,
            "request_body_schema": self.request_body_schema,
            "response_schema": self.response_schema,
            "sample_response": self.sample_response,
            "captured_at": self.captured_at,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "APIPattern":
        return cls(**data)


class TikTokAPIExtractor:
    """
    Extracts API patterns from TikTok Studio using Playwright network interception.

    Features:
    - Captures XHR/Fetch requests during page navigation
    - Filters for relevant video/analytics APIs
    - Extracts request headers, params, and response schemas
    - Saves patterns for use by the scraper
    """

    def __init__(self, logger, browser_type: str = "chromium", cdp_port: int = None):
        """
        Initialize the API extractor.

        Args:
            logger: Logger instance for status messages
            browser_type: Playwright browser type (chromium, firefox, webkit)
            cdp_port: Custom CDP port for connecting to existing browser
        """
        self.logger = logger
        self.browser_type = browser_type
        self.cdp_port = cdp_port

        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.playwright = None

        self._is_logged_in = False
        self._connected_to_existing = False

        # Captured network traffic
        self._captured_requests: Dict[str, CapturedRequest] = {}
        self._captured_responses: List[CapturedResponse] = []
        self._relevant_patterns: List[APIPattern] = []

    async def initialize(self, interactive: bool = True) -> bool:
        """
        Initialize browser and attempt authentication.

        Args:
            interactive: If True (CLI mode), wait for user input for manual login.
                        If False (web service mode), return False when login required.

        Returns:
            True if successfully logged in, False if login required or failed
        """
        self.logger.info("Initializing TikTok API Extractor...")

        # Launch Playwright
        self.playwright = await async_playwright().start()

        # Step 1: Try to connect to existing browser
        if await self._try_connect_to_existing_browser():
            self.logger.info("Using existing browser with TikTok Studio")
            await self._setup_network_interception()

            # Force page reload to trigger fresh API calls after listeners are attached
            # This is necessary because the page is already loaded and initial API calls
            # have already completed before listeners were attached
            self.logger.info("Reloading page to capture fresh API requests...")
            await self.page.reload(wait_until="domcontentloaded", timeout=60000)
            await asyncio.sleep(4)  # Allow time for XHR requests to fire

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

        # Launch browser in headful mode
        try:
            self.browser = await browser_launcher.launch(headless=False)
        except Exception as e:
            error_msg = str(e)
            if "Executable doesn't exist" in error_msg:
                raise Exception(
                    f"Playwright {self.browser_type} browser not installed. "
                    f"Run 'playwright install {self.browser_type}' to install it."
                )
            raise Exception(f"Failed to launch {self.browser_type} browser: {error_msg}")

        # Create context with viewport
        self.context = await self.browser.new_context(
            viewport={"width": 1920, "height": 1080},
            locale="he-IL"
        )

        self.page = await self.context.new_page()

        # Set up network interception BEFORE navigation
        await self._setup_network_interception()

        # Try to load cookies
        await self._try_load_cookies()

        # Navigate to TikTok Studio
        self.logger.info("Navigating to TikTok Studio...")
        await self.page.goto(STUDIO_HOME_URL, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(3)  # Allow time for XHR requests to fire

        # Check if we're logged in
        self._is_logged_in = await self._check_logged_in()

        if not self._is_logged_in:
            self.logger.info("Not logged in. Manual login required.")

            if not interactive:
                return False

            # CLI mode: wait for user input
            print("\n" + "=" * 50)
            print("  MANUAL LOGIN REQUIRED")
            print("=" * 50)
            print("  Please log in to TikTok in the browser window.")
            print("  Press Enter here when you're done logging in...")
            print("=" * 50)

            await asyncio.get_event_loop().run_in_executor(None, input)

            # Re-check login status
            await self.page.goto(STUDIO_HOME_URL, wait_until="domcontentloaded", timeout=60000)
            await asyncio.sleep(3)  # Allow time for XHR requests to fire
            self._is_logged_in = await self._check_logged_in()

            if not self._is_logged_in:
                self.logger.error("Still not logged in. Please try again.")
                return False

        self.logger.info("Successfully logged in to TikTok Studio!")
        return True

    async def continue_after_login(self) -> bool:
        """
        Continue extraction after manual login (for web service mode).

        Returns:
            True if successfully logged in
        """
        if not self.page:
            return False

        await self.page.goto(STUDIO_HOME_URL, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(3)  # Allow time for XHR requests to fire
        self._is_logged_in = await self._check_logged_in()

        if self._is_logged_in:
            self.logger.info("Login verified, continuing extraction...")
            return True

        return False

    async def _setup_network_interception(self):
        """Set up network interception to capture XHR/Fetch requests."""
        self.logger.info("Setting up network interception...")

        async def on_request(request: Request):
            # Only capture XHR/Fetch requests
            if request.resource_type not in ['xhr', 'fetch']:
                return

            # Store request by URL for matching with response
            captured = CapturedRequest(request)
            self._captured_requests[request.url] = captured

            # Log if it looks relevant
            if self._is_relevant_api(request.url):
                self.logger.info(f"  [CAPTURE] {request.method} {captured.endpoint}")

        async def on_response(response: Response):
            # Only process XHR/Fetch responses
            if response.request.resource_type not in ['xhr', 'fetch']:
                return

            url = response.url

            # Try to get response body as JSON
            try:
                body = await response.json()
            except:
                # Not JSON, skip
                return

            # Store the response
            captured = CapturedResponse(
                url=url,
                status=response.status,
                headers=dict(response.headers),
                body=body
            )
            self._captured_responses.append(captured)

            # Check if this looks like a relevant API
            if self._is_relevant_api(url) or self._contains_video_data(body):
                self.logger.info(f"  [RESPONSE] {response.status} {url[:80]}...")

                # Extract pattern
                request = self._captured_requests.get(url)
                if request:
                    pattern = self._extract_pattern(request, captured)
                    if pattern and not self._pattern_exists(pattern):
                        self._relevant_patterns.append(pattern)
                        self.logger.info(f"    -> Extracted pattern: {pattern.category}")

        self.page.on("request", on_request)
        self.page.on("response", on_response)

    def _is_relevant_api(self, url: str) -> bool:
        """Check if URL matches relevant API patterns."""
        url_lower = url.lower()
        return any(keyword in url_lower for keyword in RELEVANT_API_KEYWORDS)

    def _contains_video_data(self, body: Any) -> bool:
        """Check if response body contains video/content data."""
        if not isinstance(body, dict):
            return False

        # Check top-level keys
        for indicator in VIDEO_DATA_INDICATORS:
            if indicator in body:
                value = body[indicator]
                if isinstance(value, list) and len(value) > 0:
                    return True

        # Check nested data
        if 'data' in body and isinstance(body['data'], dict):
            for indicator in VIDEO_DATA_INDICATORS:
                if indicator in body['data']:
                    return True

        return False

    def _extract_pattern(self, request: CapturedRequest, response: CapturedResponse) -> Optional[APIPattern]:
        """Extract an API pattern from request/response pair."""
        try:
            # Determine category
            category = self._categorize_api(request.url, response.body)

            # Generate response schema
            response_schema = self._generate_schema(response.body)

            # Clean headers (remove session-specific ones)
            clean_headers = self._clean_headers(request.headers)

            return APIPattern(
                endpoint=request.endpoint,
                method=request.method,
                headers=clean_headers,
                query_params=request.query_params,
                request_body_schema=self._parse_post_data(request.post_data),
                response_schema=response_schema,
                sample_response=self._truncate_sample(response.body),
                category=category,
            )
        except Exception as e:
            self.logger.warning(f"Failed to extract pattern: {e}")
            return None

    def _categorize_api(self, url: str, body: Any) -> str:
        """Categorize the API based on URL and response."""
        url_lower = url.lower()

        if 'item/list' in url_lower or 'item_list' in url_lower:
            return 'video_list'
        elif 'analytics' in url_lower:
            return 'analytics'
        elif 'creator' in url_lower:
            if isinstance(body, dict):
                if any(k in body for k in ['itemList', 'item_list', 'videos']):
                    return 'video_list'
            return 'creator'
        elif 'content' in url_lower:
            return 'content'

        # Check response body for clues
        if isinstance(body, dict):
            if any(k in body for k in ['itemList', 'item_list', 'items']):
                return 'video_list'
            elif 'analytics' in str(body.keys()).lower():
                return 'analytics'

        return 'other'

    def _generate_schema(self, data: Any, max_depth: int = 3, current_depth: int = 0) -> Dict:
        """Generate a JSON schema from data."""
        if current_depth >= max_depth:
            return {"type": "any", "truncated": True}

        if data is None:
            return {"type": "null"}
        elif isinstance(data, bool):
            return {"type": "boolean"}
        elif isinstance(data, int):
            return {"type": "integer"}
        elif isinstance(data, float):
            return {"type": "number"}
        elif isinstance(data, str):
            return {"type": "string"}
        elif isinstance(data, list):
            if len(data) == 0:
                return {"type": "array", "items": {}}
            # Sample first item for schema
            return {
                "type": "array",
                "items": self._generate_schema(data[0], max_depth, current_depth + 1),
                "sample_length": len(data),
            }
        elif isinstance(data, dict):
            properties = {}
            for key, value in list(data.items())[:20]:  # Limit properties
                properties[key] = self._generate_schema(value, max_depth, current_depth + 1)
            return {
                "type": "object",
                "properties": properties,
            }
        else:
            return {"type": str(type(data).__name__)}

    def _clean_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Remove session-specific headers, keep only structural ones."""
        # Headers to keep (important for API structure)
        keep_headers = [
            'accept', 'accept-language', 'content-type',
            'x-requested-with', 'x-csrftoken',
        ]

        # Headers to skip (session-specific)
        skip_prefixes = ['cookie', 'authorization', 'x-tt-', 'x-secsdk']

        cleaned = {}
        for key, value in headers.items():
            key_lower = key.lower()
            if any(key_lower.startswith(prefix) for prefix in skip_prefixes):
                continue
            if any(keep in key_lower for keep in keep_headers):
                cleaned[key] = value

        return cleaned

    def _parse_post_data(self, post_data: Optional[str]) -> Optional[Dict]:
        """Parse POST data into schema."""
        if not post_data:
            return None

        try:
            data = json.loads(post_data)
            return self._generate_schema(data)
        except:
            return {"type": "string", "raw": post_data[:200] if len(post_data) > 200 else post_data}

    def _truncate_sample(self, body: Any, max_items: int = 2) -> Any:
        """Truncate sample response for storage."""
        if isinstance(body, dict):
            result = {}
            for key, value in body.items():
                if isinstance(value, list) and len(value) > max_items:
                    result[key] = value[:max_items]
                    result[f"_{key}_truncated"] = True
                    result[f"_{key}_total"] = len(value)
                else:
                    result[key] = self._truncate_sample(value, max_items)
            return result
        elif isinstance(body, list) and len(body) > max_items:
            return body[:max_items]
        return body

    def _pattern_exists(self, pattern: APIPattern) -> bool:
        """Check if a similar pattern already exists."""
        for existing in self._relevant_patterns:
            if existing.endpoint == pattern.endpoint and existing.method == pattern.method:
                return True
        return False

    async def extract_patterns(self, sample_video_count: int = 3, progress_callback=None) -> Dict:
        """
        Navigate through Studio pages and extract API patterns.

        New flow:
        1. Navigate to /content page → capture video list API
        2. Find video rows and click analytics buttons → capture analytics API
        3. Done

        Args:
            sample_video_count: Number of video analytics pages to sample
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with extraction results
        """
        self.logger.info("Starting API pattern extraction...")

        results = {
            "total_requests_captured": 0,
            "relevant_patterns": 0,
            "video_list_api": None,
            "analytics_apis": [],
            "other_apis": [],
            "errors": [],
        }

        try:
            # Step 1: Navigate to content page to capture video list API
            if progress_callback:
                await progress_callback("Navigating to content page...", 0.1)

            self.logger.info("Step 1: Navigating to content page...")
            await self.page.goto(STUDIO_CONTENT_URL, wait_until="domcontentloaded", timeout=60000)
            await asyncio.sleep(4)  # Allow time for XHR requests to fire

            # Step 2: Click analytics buttons on video rows to capture analytics API
            if progress_callback:
                await progress_callback(f"Clicking analytics on {sample_video_count} videos...", 0.2)

            self.logger.info(f"Step 2: Clicking analytics buttons on {sample_video_count} videos...")
            await self._click_video_analytics_buttons(sample_video_count, progress_callback)

            # Compile results
            results["total_requests_captured"] = len(self._captured_requests)
            results["relevant_patterns"] = len(self._relevant_patterns)

            for pattern in self._relevant_patterns:
                if pattern.category == 'video_list':
                    if results["video_list_api"] is None:
                        results["video_list_api"] = pattern.to_dict()
                elif pattern.category == 'analytics':
                    results["analytics_apis"].append(pattern.to_dict())
                else:
                    results["other_apis"].append(pattern.to_dict())

            if progress_callback:
                await progress_callback("Extraction complete!", 1.0)

            self.logger.info(f"Extraction complete. Found {len(self._relevant_patterns)} relevant patterns.")

        except Exception as e:
            self.logger.error(f"Error during extraction: {e}")
            results["errors"].append(str(e))

        return results

    async def _click_video_analytics_buttons(self, count: int, progress_callback=None):
        """
        Click analytics buttons on video rows in the /content page.

        On the /content page, each video row has a "צפייה בניתוח נתונים" (View Analytics) button.
        This method finds video rows and clicks their analytics buttons to capture analytics API calls.

        Args:
            count: Number of videos to sample
            progress_callback: Optional callback for progress updates
        """
        # Selectors for video rows on /content page (TikTok Studio uses a table layout)
        row_selectors = [
            'table tbody tr',  # Generic table rows
            'tr[class*="TableRow"]',  # Table rows with TableRow class
            '[data-e2e*="content-item"]',
            '[class*="content-item"]',
            '[class*="video-row"]',
            'div[class*="ContentTable"] tr',  # Content table rows
        ]

        # Find video rows
        rows = []
        for selector in row_selectors:
            try:
                elements = await self.page.query_selector_all(selector)
                if elements and len(elements) > 0:
                    # Skip header row if it's the first element
                    if len(elements) > 1:
                        rows = elements[1:count + 1]  # Skip first row (header), take up to count
                    else:
                        rows = elements[:count]
                    self.logger.info(f"Found {len(elements)} rows with selector: {selector}")
                    break
            except Exception as e:
                self.logger.debug(f"Selector {selector} failed: {e}")
                continue

        if not rows:
            self.logger.warning("Could not find video rows on /content page. Trying alternative approach...")
            # Alternative: Try to find analytics buttons directly
            await self._click_analytics_buttons_directly(count, progress_callback)
            return

        self.logger.info(f"Processing {len(rows)} video rows...")

        for i, row in enumerate(rows):
            try:
                if progress_callback:
                    progress = 0.2 + (0.7 * (i + 1) / len(rows))
                    await progress_callback(f"Clicking analytics for video {i + 1}/{len(rows)}...", progress)

                # Find analytics button within this row
                analytics_btn = None

                # Try Hebrew text first
                analytics_btn = await row.query_selector('button:has-text("צפייה בניתוח נתונים")')
                if not analytics_btn:
                    analytics_btn = await row.query_selector('text="צפייה בניתוח נתונים"')
                if not analytics_btn:
                    analytics_btn = await row.query_selector('button:has-text("View analytics")')
                if not analytics_btn:
                    analytics_btn = await row.query_selector('[data-e2e*="analytics"]')
                if not analytics_btn:
                    # Try any button with "analytics" in class
                    analytics_btn = await row.query_selector('button[class*="analytics"]')
                if not analytics_btn:
                    # Last resort: look for any clickable element that might be analytics
                    analytics_btn = await row.query_selector('a[href*="analytics"]')

                if analytics_btn:
                    await analytics_btn.click()
                    self.logger.info(f"Clicked analytics button for video {i + 1}")
                    await asyncio.sleep(4)  # Wait for analytics page to load and API calls

                    # Navigate back to content page for next video
                    await self.page.goto(STUDIO_CONTENT_URL, wait_until="domcontentloaded", timeout=60000)
                    await asyncio.sleep(3)
                else:
                    self.logger.warning(f"Could not find analytics button in video row {i + 1}")

            except Exception as e:
                self.logger.warning(f"Failed to click analytics for video {i + 1}: {e}")
                # Try to navigate back to content page even on error
                try:
                    await self.page.goto(STUDIO_CONTENT_URL, wait_until="domcontentloaded", timeout=60000)
                    await asyncio.sleep(2)
                except:
                    pass

    async def _click_analytics_buttons_directly(self, count: int, progress_callback=None):
        """
        Fallback method: Try to find and click analytics buttons directly on the page.

        This is used when we can't find video rows but might still find analytics buttons.
        """
        self.logger.info("Trying to find analytics buttons directly...")

        # Try to find all analytics buttons/links on the page
        button_selectors = [
            'button:has-text("צפייה בניתוח נתונים")',
            'button:has-text("View analytics")',
            'a[href*="/analytics/"]',
            '[data-e2e*="analytics-button"]',
        ]

        for selector in button_selectors:
            try:
                buttons = await self.page.query_selector_all(selector)
                if buttons and len(buttons) > 0:
                    self.logger.info(f"Found {len(buttons)} analytics buttons with selector: {selector}")

                    for i, btn in enumerate(buttons[:count]):
                        try:
                            if progress_callback:
                                progress = 0.2 + (0.7 * (i + 1) / min(len(buttons), count))
                                await progress_callback(f"Clicking analytics button {i + 1}...", progress)

                            await btn.click()
                            self.logger.info(f"Clicked analytics button {i + 1}")
                            await asyncio.sleep(4)

                            # Navigate back
                            await self.page.goto(STUDIO_CONTENT_URL, wait_until="domcontentloaded", timeout=60000)
                            await asyncio.sleep(3)
                        except Exception as e:
                            self.logger.warning(f"Failed to click button {i + 1}: {e}")

                    return
            except Exception as e:
                self.logger.debug(f"Selector {selector} failed: {e}")
                continue

        self.logger.warning("Could not find any analytics buttons on the page")

    def save_patterns(self, file_path: Path = None) -> bool:
        """Save extracted patterns to JSON file."""
        file_path = file_path or PATTERNS_FILE

        try:
            data = {
                "video_list_api": None,
                "analytics_api": None,
                "other_apis": [],
                "last_updated": datetime.now().isoformat(),
            }

            for pattern in self._relevant_patterns:
                if pattern.category == 'video_list' and data["video_list_api"] is None:
                    data["video_list_api"] = pattern.to_dict()
                elif pattern.category == 'analytics' and data["analytics_api"] is None:
                    data["analytics_api"] = pattern.to_dict()
                else:
                    data["other_apis"].append(pattern.to_dict())

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Saved patterns to {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save patterns: {e}")
            return False

    @staticmethod
    def load_patterns(file_path: Path = None) -> Optional[Dict]:
        """Load patterns from JSON file."""
        file_path = file_path or PATTERNS_FILE

        try:
            if not file_path.exists():
                return None

            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def delete_patterns(file_path: Path = None) -> bool:
        """Delete patterns file."""
        file_path = file_path or PATTERNS_FILE

        try:
            if file_path.exists():
                file_path.unlink()
            return True
        except Exception:
            return False

    async def close(self):
        """Clean up browser resources."""
        try:
            if self.browser and not self._connected_to_existing:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
        except:
            pass

    # ========== Browser Connection Methods (adapted from studio_scraper.py) ==========

    async def _try_connect_to_existing_browser(self) -> bool:
        """Try to connect to an existing browser with TikTok Studio open."""
        try:
            self.logger.info("Attempting to connect to existing browser...")

            endpoint = await self._find_cdp_endpoint()
            if not endpoint:
                self.logger.info("No browser with remote debugging found")
                return False

            self.logger.info(f"Connecting to browser at {endpoint}...")

            try:
                self.browser = await self.playwright.chromium.connect_over_cdp(
                    endpoint,
                    timeout=10000
                )
                self._connected_to_existing = True
            except Exception as e:
                self.logger.warning(f"Failed to connect to CDP endpoint: {e}")
                return False

            # Find TikTok Studio pages
            studio_pages = await self._find_tiktok_studio_pages()

            if not studio_pages:
                self.logger.info("No TikTok Studio tabs found")
                await self.browser.close()
                self.browser = None
                self._connected_to_existing = False
                return False

            # Use the first studio page
            self.page = studio_pages[0][0]
            self.context = self.page.context
            self._is_logged_in = await self._check_logged_in()

            if not self._is_logged_in:
                self.logger.info("Studio page found but not logged in")
                return False

            self.logger.info("Successfully connected to existing browser!")
            return True

        except Exception as e:
            self.logger.error(f"Error connecting to existing browser: {e}")
            if self.browser:
                try:
                    await self.browser.close()
                except:
                    pass
                self.browser = None
            self._connected_to_existing = False
            return False

    async def _find_cdp_endpoint(self) -> Optional[str]:
        """Find an active CDP endpoint."""
        import aiohttp

        ports_to_check = [self.cdp_port] if self.cdp_port else list(range(9222, 9230))

        for port in ports_to_check:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://localhost:{port}/json/version", timeout=aiohttp.ClientTimeout(total=1)) as resp:
                        if resp.status == 200:
                            return f"http://localhost:{port}"
            except:
                continue

        return None

    async def _find_tiktok_studio_pages(self) -> list:
        """Find all pages with TikTok Studio URLs."""
        studio_pages = []

        try:
            for context in self.browser.contexts:
                for page in context.pages:
                    url = page.url
                    if 'tiktokstudio' in url.lower():
                        studio_pages.append((page, url, True))
                    elif 'tiktok.com' in url.lower():
                        studio_pages.append((page, url, False))

            studio_pages.sort(key=lambda x: (not x[2], x[1]))
        except:
            pass

        return studio_pages

    async def _check_logged_in(self) -> bool:
        """Check if logged in to TikTok Studio."""
        try:
            await asyncio.sleep(0.5)
            current_url = self.page.url

            if "login" in current_url.lower():
                return False

            if "tiktokstudio" in current_url:
                try:
                    await self.page.wait_for_selector('[class*="sidebar"]', timeout=2000)
                    return True
                except:
                    pass

                login_button = await self.page.query_selector('button:has-text("Log in")')
                if not login_button:
                    return True

            return False
        except:
            return False

    async def _try_load_cookies(self) -> bool:
        """Try to load cookies from local browser."""
        try:
            import browser_cookie3

            self.logger.info("Loading cookies from local browser...")

            browsers = [
                ("Chrome", browser_cookie3.chrome),
                ("Firefox", browser_cookie3.firefox),
                ("Edge", browser_cookie3.edge),
            ]

            for name, func in browsers:
                try:
                    cookie_jar = func(domain_name=".tiktok.com")
                    cookies = []
                    for cookie in cookie_jar:
                        cookies.append({
                            "name": cookie.name,
                            "value": cookie.value,
                            "domain": cookie.domain,
                            "path": cookie.path,
                        })

                    if cookies:
                        await self.context.add_cookies(cookies)
                        self.logger.info(f"Loaded {len(cookies)} cookies from {name}")
                        return True
                except:
                    continue

            return False
        except ImportError:
            self.logger.warning("browser_cookie3 not installed")
            return False
        except Exception as e:
            self.logger.warning(f"Failed to load cookies: {e}")
            return False
