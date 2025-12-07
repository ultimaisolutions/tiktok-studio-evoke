"""
TikTok Studio Screenshot OCR Extractor

Extracts analytics data from TikTok Studio screenshots using Google Cloud Vision API.
Falls back to Tesseract OCR if Vision API is unavailable.
Supports three tab types: Overview (סקירה כללית), Viewers (צופים), Engagement (מעורבות)
"""

import re
import os
import json
import time
import argparse
from pathlib import Path
from functools import wraps
from typing import Optional
from dataclasses import dataclass, field, asdict

try:
    from google.cloud import vision
    from google.oauth2 import service_account
    from PIL import Image
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install google-cloud-vision pillow")
    exit(1)

# Optional Tesseract fallback
TESSERACT_AVAILABLE = False
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    pass


def retry_on_transient_error(max_retries: int = 3, base_delay: float = 1.0):
    """
    Decorator to retry Vision API calls on transient errors.

    Implements exponential backoff for:
    - Network timeouts
    - 429 (Too Many Requests)
    - 500-503 (Server errors)

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds (doubled on each retry).
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_str = str(e).lower()

                    # Check if error is retryable
                    is_retryable = any(x in error_str for x in [
                        'timeout', 'deadline', '429', '500', '502', '503',
                        'temporarily unavailable', 'connection', 'resource exhausted'
                    ])

                    if not is_retryable or attempt == max_retries - 1:
                        raise

                    delay = base_delay * (2 ** attempt)
                    print(f"  Retrying in {delay}s (attempt {attempt + 2}/{max_retries})...")
                    time.sleep(delay)

            raise last_exception
        return wrapper
    return decorator


@dataclass
class OverviewData:
    """Data from Overview (סקירה כללית) tab"""
    video_views: Optional[int] = None
    likes: Optional[int] = None
    comments: Optional[int] = None
    shares: Optional[int] = None
    saves: Optional[int] = None
    new_followers: Optional[int] = None
    full_video_watched_percent: Optional[float] = None
    average_watch_time_seconds: Optional[float] = None
    total_play_time_minutes: Optional[float] = None
    traffic_source: dict = field(default_factory=lambda: {
        "for_you": None,
        "personal_profile": None,
        "following": None,
        "search": None,
        "other": None,
        "sound": None
    })


@dataclass
class ViewersData:
    """Data from Viewers (צופים) tab"""
    total_viewers: Optional[int] = None
    new_viewers_percent: Optional[float] = None
    returning_viewers_percent: Optional[float] = None
    followers_percent: Optional[float] = None
    non_followers_percent: Optional[float] = None
    gender: dict = field(default_factory=lambda: {
        "female": None,
        "male": None,
        "other": None
    })
    age_groups: dict = field(default_factory=lambda: {
        "13-17": None,
        "18-24": None,
        "25-34": None,
        "35-44": None,
        "45-54": None,
        "55+": None
    })
    top_locations: list = field(default_factory=list)


@dataclass
class EngagementData:
    """Data from Engagement (מעורבות) tab"""
    likes_timestamp: Optional[str] = None
    top_comments_words: list = field(default_factory=list)


class TikTokStudioOCR:
    """Extract analytics data from TikTok Studio screenshots using Google Cloud Vision API"""

    def __init__(self, use_tesseract_fallback: bool = True):
        """
        Initialize OCR extractor with Google Cloud Vision API.

        Args:
            use_tesseract_fallback: If True, falls back to Tesseract when Vision API fails.

        Requires GOOGLE_APPLICATION_CREDENTIALS environment variable to be set
        to the path of a Google Cloud service account JSON key file.
        """
        self._vision_client = None
        self._last_english_text = ""
        self._last_vision_response = None  # Store response for spatial parsing
        self._use_tesseract_fallback = use_tesseract_fallback and TESSERACT_AVAILABLE

        # Initialize Tesseract path for fallback if available
        if self._use_tesseract_fallback:
            win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if Path(win_path).exists():
                pytesseract.pytesseract.tesseract_cmd = win_path

        # Initialize Google Cloud Vision client
        creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if not creds_path:
            raise ValueError(
                "GOOGLE_APPLICATION_CREDENTIALS environment variable not set.\n"
                "Set it to your Google Cloud service account JSON key file path.\n"
                "Example: set GOOGLE_APPLICATION_CREDENTIALS=C:\\path\\to\\credentials.json"
            )

        if not Path(creds_path).exists():
            raise FileNotFoundError(f"Credentials file not found: {creds_path}")

        credentials = service_account.Credentials.from_service_account_file(creds_path)
        self._vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        print(f"  Initialized Google Cloud Vision client")

    @staticmethod
    def parse_time_to_minutes(time_str: str) -> Optional[float]:
        """
        Convert time string to minutes.
        Handles formats like "822h:29m:10s", "1:30", "1.5s"
        """
        if not time_str:
            return None

        time_str = time_str.strip().lower()

        # Format: "822h:29m:10s"
        match = re.match(r'(\d+)h:(\d+)m:(\d+)s', time_str)
        if match:
            hours = int(match.group(1))
            minutes = int(match.group(2))
            seconds = int(match.group(3))
            total_minutes = hours * 60 + minutes + seconds / 60
            return round(total_minutes, 2)

        # Format: "1:30" (minutes:seconds)
        match = re.match(r'(\d+):(\d+)', time_str)
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            return round(minutes + seconds / 60, 2)

        # Format: "1.5s" or "1.5"
        match = re.search(r'([\d.]+)', time_str)
        if match:
            seconds = float(match.group(1))
            return round(seconds / 60, 2)

        return None

    @staticmethod
    def parse_abbreviated_number(text: str) -> Optional[int]:
        """
        Convert abbreviated numbers to full integers.
        Handles: "1.7M" -> 1700000, "2.5K" -> 2500, "1,234" -> 1234
        """
        if not text:
            return None

        text = str(text).strip().upper().replace(',', '').replace(' ', '')

        multiplier = 1
        if text.endswith('M'):
            multiplier = 1000000
            text = text[:-1]
        elif text.endswith('K'):
            multiplier = 1000
            text = text[:-1]

        match = re.search(r'[\d.]+', text)
        if match:
            try:
                return int(float(match.group()) * multiplier)
            except ValueError:
                return None
        return None

    def extract_text(self, image_path: str) -> str:
        """
        Extract text from image using Google Cloud Vision API with Tesseract fallback.

        Args:
            image_path: Path to the screenshot image file.

        Returns:
            Extracted text from the image.
        """
        try:
            return self._extract_text_vision(image_path)
        except Exception as e:
            print(f"  Vision API error: {e}")
            if self._use_tesseract_fallback and TESSERACT_AVAILABLE:
                print("  Falling back to Tesseract OCR...")
                return self._extract_text_tesseract(image_path)
            raise

    @retry_on_transient_error(max_retries=3, base_delay=1.0)
    def _extract_text_vision(self, image_path: str) -> str:
        """
        Extract text using Google Cloud Vision document_text_detection.

        Uses document_text_detection which is optimized for dense text layouts
        like analytics dashboards. Auto-detects languages including Hebrew.
        """
        with open(image_path, 'rb') as f:
            content = f.read()

        image = vision.Image(content=content)

        # Use document_text_detection for dense text (better for analytics dashboards)
        # Auto-detects languages including Hebrew
        response = self._vision_client.document_text_detection(image=image)

        if response.error.message:
            raise Exception(f"Vision API error: {response.error.message}")

        # Store full response for spatial analysis
        self._last_vision_response = response

        full_text = response.full_text_annotation.text if response.full_text_annotation else ""

        # Extract English/numeric tokens for secondary parsing (K/M suffixes)
        self._last_english_text = self._extract_english_tokens(response)

        return full_text

    def _extract_text_tesseract(self, image_path: str) -> str:
        """
        Fallback: Extract text using Tesseract OCR (original implementation).

        Uses dual-pass approach: Hebrew+English for primary text,
        English-only on grayscale for better number recognition.
        """
        img = Image.open(image_path)
        img_gray = img.convert('L')

        # Primary extraction with Hebrew+English
        text_primary = pytesseract.image_to_string(img, lang='heb+eng')

        # Secondary extraction with grayscale + English only (better for K/M suffixes)
        text_english = pytesseract.image_to_string(img_gray, lang='eng')

        # Store English extraction for metrics parsing
        self._last_english_text = text_english

        return text_primary

    def _extract_english_tokens(self, response) -> str:
        """
        Filter Vision API response to English/numeric tokens only.

        This replicates the secondary Tesseract pass that was English-only,
        which performed better at recognizing K/M suffixes on numbers.

        Args:
            response: Vision API document_text_detection response.

        Returns:
            Space-separated string of English/numeric tokens.
        """
        if not response.text_annotations:
            return ""

        english_parts = []

        # Skip first annotation (it's the full text)
        for annotation in response.text_annotations[1:]:
            text = annotation.description.strip()

            # Check if text is primarily English/numeric
            if self._is_english_or_numeric(text):
                english_parts.append(text)

        return ' '.join(english_parts)

    def _is_english_or_numeric(self, text: str) -> bool:
        """
        Check if text is primarily English characters or numbers.

        Returns True if >70% of characters are:
        - ASCII letters (a-z, A-Z)
        - Digits (0-9)
        - Common punctuation (., ,, %, :, -, +)

        Args:
            text: Text to check.

        Returns:
            True if text is primarily English/numeric.
        """
        if not text:
            return False

        english_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,%:-+/ ')
        matching = sum(1 for c in text if c in english_chars)

        return matching / len(text) > 0.7

    def _get_text_with_positions(self) -> list:
        """
        Extract text annotations with their bounding box positions.

        Returns list of dicts: [{'text': str, 'x': int, 'y': int, 'width': int, 'height': int}]
        """
        if not self._last_vision_response or not self._last_vision_response.text_annotations:
            return []

        results = []
        # Skip first annotation (it's the full concatenated text)
        for annotation in self._last_vision_response.text_annotations[1:]:
            vertices = annotation.bounding_poly.vertices
            if len(vertices) >= 4:
                # Get bounding box coordinates
                x_coords = [v.x for v in vertices if hasattr(v, 'x') and v.x is not None]
                y_coords = [v.y for v in vertices if hasattr(v, 'y') and v.y is not None]

                if x_coords and y_coords:
                    x = min(x_coords)
                    y = min(y_coords)
                    x_max = max(x_coords)
                    y_max = max(y_coords)
                    results.append({
                        'text': annotation.description,
                        'x': x,
                        'y': y,
                        'width': x_max - x,
                        'height': y_max - y
                    })
        return results

    def _group_by_rows(self, annotations: list, y_tolerance: int = 20) -> list:
        """
        Group text annotations into rows based on vertical position.

        Args:
            annotations: List from _get_text_with_positions()
            y_tolerance: Pixels tolerance for same-row grouping

        Returns:
            List of rows, each row sorted left-to-right
        """
        if not annotations:
            return []

        # Sort by Y position first
        sorted_by_y = sorted(annotations, key=lambda a: a['y'])

        rows = []
        current_row = [sorted_by_y[0]]
        current_y = sorted_by_y[0]['y']

        for ann in sorted_by_y[1:]:
            if abs(ann['y'] - current_y) <= y_tolerance:
                current_row.append(ann)
            else:
                # Sort current row by X (left-to-right) and save
                rows.append(sorted(current_row, key=lambda a: a['x']))
                current_row = [ann]
                current_y = ann['y']

        if current_row:
            rows.append(sorted(current_row, key=lambda a: a['x']))

        return rows

    def _find_value_below_label(self, annotations: list, label: str, y_max_distance: int = 80, x_tolerance: int = 100) -> Optional[str]:
        """
        Find a numeric value below a label (for metrics where label is above value).

        Args:
            annotations: List from _get_text_with_positions()
            label: Label text to search for
            y_max_distance: Maximum vertical distance to search below label
            x_tolerance: Horizontal tolerance for alignment

        Returns:
            The value text or None
        """
        # Find the label
        label_ann = None
        for ann in annotations:
            if label in ann['text'] or ann['text'] in label:
                label_ann = ann
                break

        if not label_ann:
            return None

        # Find nearest number below the label
        candidates = []
        for ann in annotations:
            if ann == label_ann:
                continue

            # Check if it contains a number
            if not re.search(r'[\d.,%KMkm]+', ann['text']):
                continue

            # Check if below and aligned
            y_dist = ann['y'] - label_ann['y']
            x_dist = abs(ann['x'] - label_ann['x'])

            if 0 < y_dist <= y_max_distance and x_dist <= x_tolerance:
                candidates.append((y_dist + x_dist * 0.5, ann['text']))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]

        return None

    def _find_value_near_label(self, annotations: list, label: str, search_direction: str = 'right', max_distance: int = 200) -> Optional[str]:
        """
        Find a numeric/percentage value near a label.

        Args:
            annotations: List from _get_text_with_positions()
            label: Label text to search for
            search_direction: 'right', 'left', or 'both'
            max_distance: Maximum pixel distance to consider

        Returns:
            The nearest value text or None
        """
        # Find the label
        label_ann = None
        for ann in annotations:
            if label in ann['text'] or ann['text'] in label:
                label_ann = ann
                break

        if not label_ann:
            return None

        # Find nearest number/percentage on same row
        candidates = []
        for ann in annotations:
            if ann == label_ann:
                continue

            # Check if it's a number or percentage
            if not re.search(r'[\d.,%KMkm]+', ann['text']):
                continue

            # Check vertical alignment (same row)
            if abs(ann['y'] - label_ann['y']) > 30:
                continue

            # Check direction
            if search_direction == 'right':
                if ann['x'] > label_ann['x']:
                    distance = ann['x'] - label_ann['x']
                    if distance < max_distance:
                        candidates.append((distance, ann['text']))
            elif search_direction == 'left':
                if ann['x'] < label_ann['x'] + label_ann['width']:
                    distance = label_ann['x'] - ann['x']
                    if distance < max_distance:
                        candidates.append((distance, ann['text']))
            else:  # both
                distance = abs(ann['x'] - label_ann['x'])
                if distance < max_distance:
                    candidates.append((distance, ann['text']))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]

        return None

    def detect_tab_type(self, text: str) -> str:
        """Detect which tab the screenshot is from based on URL or content"""
        text_lower = text.lower()

        # Check URL first (most reliable)
        if '/engagement' in text_lower:
            return "engagement"
        elif '/viewers' in text_lower:
            return "viewers"
        elif '/analytics/' in text_lower:
            # Default analytics page is overview
            if 'סקירה כללית' in text or 'צפיות בסרטונים' in text or 'זמן הפעלה' in text:
                return "overview"

        # Content-based detection
        if 'המילים המובילות בתגובות' in text or 'לייקים' in text and 'סימנו לייק' in text:
            return "engagement"
        elif 'סה"כ צופים' in text or 'מגדר' in text or 'סוגי הצופים' in text:
            return "viewers"
        elif 'מקור הטראפיק' in text or 'שיעור השימור' in text or 'For You' in text:
            return "overview"

        return "unknown"

    def parse_number(self, text: str) -> Optional[int]:
        """Parse a number, handling K/M suffixes"""
        if not text:
            return None

        text = text.strip().replace(',', '').replace(' ', '')

        multiplier = 1
        if text.upper().endswith('M'):
            multiplier = 1000000
            text = text[:-1]
        elif text.upper().endswith('K'):
            multiplier = 1000
            text = text[:-1]

        match = re.search(r'[\d.]+', text)
        if match:
            try:
                return int(float(match.group()) * multiplier)
            except ValueError:
                return None
        return None

    def parse_percentage(self, text: str) -> Optional[float]:
        """Parse a percentage value"""
        if not text:
            return None

        # Handle cases like "<0.1%" or ">1.6%"
        text = text.strip().replace('%', '').replace('<', '').replace('>', '')
        match = re.search(r'[\d.]+', text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
        return None

    def _extract_metrics_from_text(self, text: str) -> dict:
        """
        Extract metrics (saves, shares, comments, likes, views) using spatial analysis.
        Falls back to legacy line-based parsing if Vision API response not available.
        """
        # Use spatial analysis if Vision API response is available
        if self._last_vision_response:
            result = self._extract_metrics_spatial()
            # If spatial analysis found at least some values, use it
            if any(v is not None for v in result.values()):
                return result

        # Fallback to legacy parsing
        return self._extract_metrics_from_text_legacy(text)

    def _extract_metrics_spatial(self) -> dict:
        """
        Extract metrics using spatial analysis of Vision API response.
        Uses multiple strategies: row pattern matching, label-based, and fallbacks.
        """
        result = {
            'saves': None,
            'shares': None,
            'comments': None,
            'likes': None,
            'views': None
        }

        annotations = self._get_text_with_positions()
        if not annotations:
            return result

        rows = self._group_by_rows(annotations)

        # Strategy 1: Find row with 5 numeric values (metrics row pattern)
        # Pattern: saves, shares, comments, likes, views (in that visual order)
        # Example row: ['209', '123', '25', '1,627', '2.1M']
        for row in rows:
            row_values = []
            for ann in row:
                text = ann['text'].strip()
                # Check if it's a number (possibly with comma, decimal, K/M suffix)
                if re.match(r'^[\d,]+\.?\d*[KkMm]?$', text):
                    parsed = self.parse_abbreviated_number(text)
                    if parsed is not None:
                        row_values.append((parsed, ann['x']))

            # Check if we have 5 numbers with one being very large (views)
            if len(row_values) >= 5:
                # Sort by x position (left to right) to get correct order
                row_values.sort(key=lambda x: x[1])
                values = [v[0] for v in row_values[:5]]

                # Verify: one value should be > 100K (views, typically millions)
                max_val = max(values)
                if max_val >= 100000:
                    # In TikTok Studio layout (left to right):
                    # saves, shares, comments, likes, views
                    result['saves'] = values[0]
                    result['shares'] = values[1]
                    result['comments'] = values[2]
                    result['likes'] = values[3]
                    result['views'] = values[4]
                    return result

        # Strategy 2: Find labels and look for values below them (Hebrew labels)
        metric_labels = {
            'saves': ['שמירות', 'שמירה'],
            'shares': ['שיתופים', 'שיתוף'],
            'comments': ['תגובות', 'הערות'],
            'likes': ['לייקים', 'לייק'],
            'views': ['צפיות', 'בסרטונים']
        }

        for metric, labels in metric_labels.items():
            if result[metric] is not None:
                continue
            for label in labels:
                value_text = self._find_value_below_label(annotations, label)
                if value_text:
                    parsed = self.parse_abbreviated_number(value_text)
                    if parsed is not None:
                        result[metric] = parsed
                        break

        # Strategy 3: If views still not found, look for the largest M-suffixed number
        if result['views'] is None:
            for ann in annotations:
                text = ann['text'].strip()
                if re.match(r'^\d+\.?\d*[Mm]$', text):
                    parsed = self.parse_abbreviated_number(text)
                    if parsed and parsed > 100000:
                        result['views'] = parsed
                        break

        return result

    def _extract_metrics_from_text_legacy(self, text: str) -> dict:
        """
        Legacy: Extract metrics from OCR text using line-based parsing.
        Looks for patterns like "209 123 14K 1,627 10.6M" in order.
        Returns dict with extracted values or None for each metric.
        """
        result = {
            'saves': None,
            'shares': None,
            'comments': None,
            'likes': None,
            'views': None
        }

        # Find all number patterns with optional K/M suffixes
        # Match: "209", "123", "14K", "1,627", "10.6M"
        pattern = r'(\d+(?:[,.]\d+)?)\s*([KkMm])?'

        # Look for a line that contains the metrics pattern
        # Expected: 5 numbers in sequence (saves, shares, comments, likes, views)
        for line in text.split('\n'):
            # Skip lines with time patterns (h:m:s) - these are different metrics
            if re.search(r'\d+h:\d+m:\d+s', line):
                continue

            # Remove date patterns from the line to avoid contamination
            line_clean = re.sub(r'\d+\.\d+\.\d{4}', ' ', line)

            tokens = []
            for match in re.finditer(pattern, line_clean):
                num_str = match.group(1).replace(',', '')
                suffix = match.group(2)

                try:
                    value = float(num_str)
                    if suffix:
                        suffix = suffix.upper()
                        if suffix == 'K':
                            value *= 1000
                        elif suffix == 'M':
                            value *= 1000000
                    tokens.append(int(value))
                except ValueError:
                    continue

            # We expect 5 tokens for a complete metrics row
            if len(tokens) >= 5:
                # Check if pattern matches expected metrics signature:
                # - One very large number (views, typically millions)
                # - One medium number (likes, typically 1K-100K)
                # - Three smaller numbers (saves, shares, comments)
                sorted_by_value = sorted(tokens[:5], reverse=True)

                # Verify: largest should be views (>100K), second should be likes or comments
                if sorted_by_value[0] >= 100000:  # Views should be 100K+
                    # Assign first 5 in positional order: saves, shares, comments, likes, views
                    result['saves'] = tokens[0]
                    result['shares'] = tokens[1]
                    result['comments'] = tokens[2]
                    result['likes'] = tokens[3]
                    result['views'] = tokens[4]
                    return result

        return result

    def extract_overview_data(self, text: str) -> OverviewData:
        """Extract data from Overview tab screenshot"""
        data = OverviewData()

        # Clean unicode directional markers for easier parsing
        text_clean = text.replace('\u200e', '').replace('\u200f', '')

        # Also use English-only OCR text if available (better for K/M suffixes)
        english_text = getattr(self, '_last_english_text', '') or ''

        # Split into lines for line-by-line analysis
        lines = text_clean.split('\n')

        # =================================================================
        # STEP 1: Extract from SECOND ROW (most reliable source for views)
        # Pattern: "164 0.7% [optional time] 822h:29m:10s 2.1M" or "164 0.7% ₪ 822h:29m:10s 10.6M"
        # This row contains: new_followers, full_video_%, avg_watch_time, total_play_time, video_views
        # =================================================================

        # Use spatial analysis to find the row with time format (h:m:s pattern)
        annotations = self._get_text_with_positions()
        rows = self._group_by_rows(annotations) if annotations else []

        for row in rows:
            row_texts = [ann['text'] for ann in row]
            row_combined = ' '.join(row_texts)

            # Look for the time pattern like "822h : 29m : 10s"
            if re.search(r'\d+h.*\d+m.*\d+s', row_combined):
                # Found the second metrics row - extract values

                # Find new_followers: first small number (3 digits) in the row
                for ann in row:
                    text = ann['text'].strip()
                    if re.match(r'^\d{1,3}$', text):
                        val = int(text)
                        if 1 <= val <= 999 and data.new_followers is None:
                            data.new_followers = val
                            break

                # Find full_video_watched_percent: decimal number followed by %
                for i, ann in enumerate(row):
                    text = ann['text'].strip()
                    if re.match(r'^\d+\.?\d*$', text):
                        val = float(text)
                        # Check if '%' follows
                        if i + 1 < len(row) and row[i + 1]['text'].strip() == '%':
                            if 0 < val < 100 and data.full_video_watched_percent is None:
                                data.full_video_watched_percent = val
                                break

                # Find average_watch_time: number followed by 's' (like "1.5s")
                for ann in row:
                    text = ann['text'].strip()
                    match = re.match(r'^(\d+\.?\d*)s$', text)
                    if match and data.average_watch_time_seconds is None:
                        data.average_watch_time_seconds = float(match.group(1))

                # Find total_play_time: look for tokens like "822h", "29m", "10s"
                hours = minutes = seconds = None
                for ann in row:
                    text = ann['text'].strip()
                    h_match = re.match(r'^(\d+)h$', text)
                    m_match = re.match(r'^(\d+)m$', text)
                    s_match = re.match(r'^(\d+)s$', text)

                    if h_match:
                        hours = int(h_match.group(1))
                    elif m_match:
                        minutes = int(m_match.group(1))
                    elif s_match and not text.startswith('1.'):  # Exclude "1.5s" (avg watch time)
                        seconds = int(s_match.group(1))

                if hours is not None and data.total_play_time_minutes is None:
                    minutes = minutes or 0
                    seconds = seconds or 0
                    data.total_play_time_minutes = round(hours * 60 + minutes + seconds / 60, 2)

                # Find video_views: M-suffixed number (capital M only - lowercase m is for minutes)
                for ann in row:
                    text = ann['text'].strip()
                    # Only match capital M for millions (e.g., "2.1M"), not lowercase m (e.g., "29m" for minutes)
                    if re.match(r'^\d+\.?\d*M$', text):
                        parsed = self.parse_abbreviated_number(text)
                        if parsed and data.video_views is None:
                            data.video_views = parsed

                break

        # Fallback: line-based regex parsing for older format
        for line in lines:
            time_match = re.search(r'(\d+h:\d+m:\d+s)', line)
            if time_match:
                if data.video_views is None:
                    line_after_time = line[time_match.end():]
                    views_match = re.search(r'(\d+\.?\d*)[Mm]', line_after_time)
                    if views_match:
                        data.video_views = int(float(views_match.group(1)) * 1000000)

                if data.total_play_time_minutes is None:
                    data.total_play_time_minutes = self.parse_time_to_minutes(time_match.group(1))

                if data.new_followers is None or data.full_video_watched_percent is None:
                    followers_match = re.search(r'(\d{1,4})\s+(\d+\.?\d*)%', line)
                    if followers_match:
                        data.new_followers = int(followers_match.group(1))
                        data.full_video_watched_percent = float(followers_match.group(2))

                if data.average_watch_time_seconds is None:
                    avg_time_match = re.search(r'(\d+\.?\d*)s\s+\d+h:', line)
                    if avg_time_match:
                        data.average_watch_time_seconds = float(avg_time_match.group(1))

                break

        # =================================================================
        # STEP 2: Extract metrics row (likes, comments, shares, saves)
        # Use English-only OCR text which handles K/M suffixes better
        # Pattern: "209 123 14K 1,627 10.6M" (saves, shares, comments, likes, views)
        # =================================================================

        # Parse metrics from English text (better K/M recognition)
        metrics_extracted = self._extract_metrics_from_text(english_text)
        if metrics_extracted:
            if metrics_extracted.get('saves') is not None and data.saves is None:
                data.saves = metrics_extracted['saves']
            if metrics_extracted.get('shares') is not None and data.shares is None:
                data.shares = metrics_extracted['shares']
            if metrics_extracted.get('comments') is not None and data.comments is None:
                data.comments = metrics_extracted['comments']
            if metrics_extracted.get('likes') is not None and data.likes is None:
                data.likes = metrics_extracted['likes']
            if metrics_extracted.get('views') is not None and data.video_views is None:
                data.video_views = metrics_extracted['views']

        # Fallback: try Hebrew text if English didn't work
        if data.likes is None:
            for line in lines:
                # Skip lines that are the second row (contain h:m:s time)
                if re.search(r'\d+h:\d+m:\d+s', line):
                    continue

                # Skip lines with dates like "7.6.2025" or "3.12.2025" to avoid year confusion
                if re.search(r'\d+\.\d+\.\d{4}', line):
                    line_no_date = re.sub(r'\d+\.\d+\.\d{4}', '', line)
                else:
                    line_no_date = line

                # Look for pattern with comma-separated number (likes, typically 1,000-99,999)
                likes_match = re.search(r'(\d{1,2},\d{3})', line_no_date)
                if likes_match:
                    data.likes = int(likes_match.group(1).replace(',', ''))
                    break

        # =================================================================
        # STEP 3: Fallback - if we didn't find likes with comma, try other patterns
        # =================================================================
        if data.likes is None:
            for line in lines:
                if re.search(r'\d+h:\d+m:\d+s', line):
                    continue

                # Remove date patterns
                line_clean = re.sub(r'\d+\.\d+\.\d{4}', '', line)

                # Look for sequence of small numbers that could be the metrics
                # Pattern: several 1-4 digit numbers in sequence
                nums_in_line = re.findall(r'\b(\d{1,4})\b', line_clean)
                if len(nums_in_line) >= 4:
                    # Convert to integers and filter reasonable values
                    int_nums = [int(n) for n in nums_in_line if 1 <= int(n) <= 9999]

                    if len(int_nums) >= 4:
                        # Sort: assume largest is likes, rest are saves/shares/comments
                        int_nums.sort(reverse=True)
                        data.likes = int_nums[0]
                        data.saves = int_nums[1]
                        data.shares = int_nums[2]
                        data.comments = int_nums[3]
                        break

        # =================================================================
        # STEP 4: Extract traffic sources - מקור הטראפיק
        # Use spatial analysis first, then fall back to regex patterns
        # =================================================================

        # Try spatial analysis first if Vision API response is available
        annotations = self._get_text_with_positions()
        rows = self._group_by_rows(annotations) if annotations else []

        if rows:
            # Strategy: Find rows that contain both a percentage and a traffic source keyword
            # Pattern observed: '83.5', '%', 'For', 'You' on same row
            # The percentage value comes BEFORE the '%' symbol in the row
            for row in rows:
                row_text = ' '.join(ann['text'] for ann in row).lower()

                # Extract the FIRST valid percentage from this row
                # Look for pattern: number followed by '%' token
                pct_value = None
                for i, ann in enumerate(row):
                    text = ann['text'].strip()

                    # Check if this is a number that could be a percentage
                    if re.match(r'^\d+\.?\d*$', text):
                        candidate = float(text)
                        # Verify '%' appears nearby (within next 2 tokens)
                        has_percent_sign = False
                        for j in range(i + 1, min(i + 3, len(row))):
                            if row[j]['text'].strip() == '%':
                                has_percent_sign = True
                                break

                        if has_percent_sign and 0 < candidate <= 100:
                            pct_value = candidate
                            break  # Take the first valid percentage

                if pct_value is None:
                    continue

                # Check which traffic source this row represents
                # Note: We check traffic sources in order of specificity
                if ('for' in row_text and 'you' in row_text) or 'foryou' in row_text:
                    if data.traffic_source["for_you"] is None:
                        data.traffic_source["for_you"] = pct_value
                elif 'פרופיל' in row_text or 'profile' in row_text:
                    if data.traffic_source["personal_profile"] is None:
                        data.traffic_source["personal_profile"] = pct_value
                elif 'במעקב' in row_text or 'following' in row_text:
                    if data.traffic_source["following"] is None:
                        data.traffic_source["following"] = pct_value
                elif 'חיפוש' in row_text or 'search' in row_text:
                    if data.traffic_source["search"] is None:
                        data.traffic_source["search"] = pct_value
                elif 'אחר' in row_text and pct_value < 5 and data.traffic_source["other"] is None:
                    # Be careful: אחר could appear in other contexts
                    data.traffic_source["other"] = pct_value
                elif ('סאונד' in row_text or 'sound' in row_text) and data.traffic_source["sound"] is None:
                    data.traffic_source["sound"] = pct_value

        # Fallback: regex-based extraction if spatial analysis didn't find values
        # For You - most reliable, always shows as "For You XX.X%"
        if data.traffic_source["for_you"] is None:
            for_you_match = re.search(r'For\s*You[^\d]*(\d+\.?\d*)%', text_clean, re.IGNORECASE)
            if for_you_match:
                data.traffic_source["for_you"] = float(for_you_match.group(1))

        # Personal profile - פרופיל אישי
        if data.traffic_source["personal_profile"] is None:
            profile_patterns = [
                r'פרופי[לו]l?\s*\S*\s*(\d+\.?\d*)%',  # Direct match
                r'(\d+\.?\d*)%[^\n]*פרופי',            # Percentage before פרופיל
                r'פרופ[וי][לו]l?\s*(\d+\.?\d*)%',      # OCR variants
            ]
            for pattern in profile_patterns:
                match = re.search(pattern, text_clean)
                if match:
                    data.traffic_source["personal_profile"] = float(match.group(1))
                    break

        # Following - במעקב
        if data.traffic_source["following"] is None:
            following_patterns = [
                r'במעקב\s*(\d+\.?\d*)%',
                r'(\d+\.?\d*)%[^\n]*במעקב',
                r'[aב]pyna\s*\S*\s*(\d+\.?\d*)%',
            ]
            for pattern in following_patterns:
                match = re.search(pattern, text_clean, re.IGNORECASE)
                if match:
                    val = float(match.group(1))
                    # If value is 20, it's likely 2.0 (OCR missed decimal)
                    if val == 20:
                        val = 2.0
                    data.traffic_source["following"] = val
                    break

        # Search - חיפוש
        if data.traffic_source["search"] is None:
            search_patterns = [
                r'חיפוש\s*[<>]?(\d+)%',
                r'\bwon\s*[<>]?(\d+)%',
            ]
            for pattern in search_patterns:
                match = re.search(pattern, text_clean, re.IGNORECASE)
                if match:
                    val = int(match.group(1))
                    # "01" means 0.1%
                    if val == 1:
                        data.traffic_source["search"] = 0.1
                    else:
                        data.traffic_source["search"] = float(val)
                    break

        # Other - אחר
        if data.traffic_source["other"] is None:
            other_match = re.search(r'אחר\s*[<>]?(\d+)%', text_clean)
            if other_match:
                val = int(other_match.group(1))
                data.traffic_source["other"] = 0.1 if val == 1 else float(val)

        # Sound - סאונד
        if data.traffic_source["sound"] is None:
            sound_match = re.search(r'סאונד\s*[<>]?(\d+)%', text_clean)
            if sound_match:
                val = int(sound_match.group(1))
                data.traffic_source["sound"] = 0.1 if val == 1 else float(val)

        # =================================================================
        # STEP 5: Calculate personal_profile from remainder if not found
        # Traffic sources should add up to ~100%
        # =================================================================
        if data.traffic_source["personal_profile"] is None and data.traffic_source["for_you"]:
            known_total = sum(v for v in data.traffic_source.values() if v is not None)
            if 80 <= known_total <= 99:
                remaining = round(100 - known_total, 1)
                if 5 <= remaining <= 20:
                    data.traffic_source["personal_profile"] = remaining

        return data

    def extract_viewers_data(self, text: str) -> ViewersData:
        """Extract data from Viewers tab screenshot"""
        data = ViewersData()

        # Clean unicode directional markers
        text_clean = text.replace('\u200e', '').replace('\u200f', '')

        # Also use English-only OCR text if available (better for numbers and age groups)
        english_text = getattr(self, '_last_english_text', '') or ''

        # Total viewers - look for number with M/K suffix (e.g., "1.7M")
        total_match = re.search(r'(\d+\.?\d*)[Mm]\s*\n?\s*\d+\+', text_clean)
        if total_match:
            data.total_viewers = self.parse_abbreviated_number(total_match.group(1) + 'M')
        else:
            # Try סה"כ צופים pattern
            total_match = re.search(r'סה"כ צופים[^\d]*(\d+\.?\d*)\s*([MKmk])?', text_clean)
            if total_match:
                num = total_match.group(1)
                suffix = total_match.group(2) or ''
                data.total_viewers = self.parse_abbreviated_number(num + suffix)

        # Viewer types - סוגי הצופים section
        # The layout shows two bars: new (32%) vs returning (68%)
        # And followers (1%) vs non-followers (99%)

        # Look for consecutive percentage pairs
        # Pattern: "XX%\n...\nצופים חדשים" and "YY%\n...\nצופים חוזרים"
        new_match = re.search(r'(\d+)%[^\n]*\n[^\n]*צופים חדשים', text_clean)
        if new_match:
            data.new_viewers_percent = float(new_match.group(1))

        returning_match = re.search(r'(\d+)%[^\n]*\n[^\n]*חוזרים', text_clean)
        if returning_match:
            data.returning_viewers_percent = float(returning_match.group(1))

        # If percentages appear on same line or near each other
        if not data.new_viewers_percent or not data.returning_viewers_percent:
            viewers_section = re.search(r'סוגי הצופים.*?(\d+)%.*?(\d+)%', text_clean, re.DOTALL)
            if viewers_section:
                # First is usually returning (larger), second is new
                pct1 = float(viewers_section.group(1))
                pct2 = float(viewers_section.group(2))
                if pct1 > pct2:
                    data.returning_viewers_percent = pct1
                    data.new_viewers_percent = pct2
                else:
                    data.new_viewers_percent = pct1
                    data.returning_viewers_percent = pct2

        # Followers vs non-followers
        # Pattern in OCR: "99% 1%" on one line, then "לא עוקבים. עוקבים." on next line
        # Look for two percentages followed by עוקבים within a few lines
        followers_section = re.search(r'(\d+)%\s+(\d+)%[^\n]*\n[^\n]*עוקבים', text_clean)
        if followers_section:
            pct1 = float(followers_section.group(1))
            pct2 = float(followers_section.group(2))
            # Non-followers is usually larger (first value), followers is smaller (second value)
            if pct1 > pct2:
                data.non_followers_percent = pct1
                data.followers_percent = pct2
            else:
                data.followers_percent = pct1
                data.non_followers_percent = pct2
        else:
            # Alternative: look for pattern where percentages are on same line
            followers_alt = re.search(r'(\d+)%\s+(\d+)%', text_clean)
            if followers_alt:
                # Check if this is near עוקבים (within 100 chars)
                match_pos = followers_alt.start()
                context = text_clean[match_pos:match_pos+150]
                if 'עוקבים' in context:
                    pct1 = float(followers_alt.group(1))
                    pct2 = float(followers_alt.group(2))
                    if pct1 > pct2:
                        data.non_followers_percent = pct1
                        data.followers_percent = pct2
                    else:
                        data.followers_percent = pct1
                        data.non_followers_percent = pct2

        # Gender - מגדר section
        # The OCR puts the labels and percentages on separate lines:
        # "מגדר ₪"
        # "₪ נקבה"
        # "₪ זכר"
        # "= 5 אחר"
        # Then later: "95%" "5%" "0%"
        # We need to find percentages that appear AFTER מגדר section

        # Look for pattern where percentages appear after מגדר labels
        gender_section = re.search(r'מגדר.*?נקבה.*?זכר.*?אחר.*?(\d+)%\s*(\d+)%\s*(\d+)%', text_clean, re.DOTALL)
        if gender_section:
            data.gender["female"] = float(gender_section.group(1))
            data.gender["male"] = float(gender_section.group(2))
            data.gender["other"] = float(gender_section.group(3))

        # Age groups - גיל section
        # Pattern: "18-24 45%" or "45% 18-24" or "10.6% 25-34"
        # Use English text first (grayscale OCR is more reliable for these patterns)
        # IMPORTANT: Prefer "XX% age" pattern over "age XX%" to avoid grabbing wrong percentage
        for age_group in ["13-17", "18-24", "25-34", "35-44", "45-54"]:
            # Try English text first (more reliable)
            # Pattern 1: "XX% age" - preferred because it's unambiguous
            match = re.search(rf'(\d+\.?\d*)%\s+{age_group}(?:\s|$)', english_text)
            if match:
                data.age_groups[age_group] = float(match.group(1))
                continue

            # Pattern 2: "age XX%" - but ensure XX% is not followed by another age marker (like +55)
            # Use negative lookahead to avoid grabbing percentage that belongs to next age group
            match = re.search(rf'{age_group}\s+(\d+\.?\d*)%(?!\s*[+]?\d)', english_text)
            if match:
                data.age_groups[age_group] = float(match.group(1))
                continue

            # Fallback to Hebrew text with same logic
            match = re.search(rf'(\d+)%\s*{age_group}', text_clean)
            if match:
                data.age_groups[age_group] = float(match.group(1))
            else:
                match = re.search(rf'{age_group}\s*(\d+)%(?!\s*[+]?\d)', text_clean)
                if match:
                    data.age_groups[age_group] = float(match.group(1))

        # 55+ special case (may appear as "+55" or "55+")
        # Try English text first
        # Pattern: "XX% +55" or "XX% 55+"
        match_55 = re.search(r'(\d+\.?\d*)%\s*\+?55\+?(?:\s|$)', english_text)
        if match_55:
            data.age_groups["55+"] = float(match_55.group(1))
        else:
            # Pattern: "+55 XX%" or "55+ XX%"
            match_55 = re.search(r'\+?55\+?\s+(\d+\.?\d*)%', english_text)
            if match_55:
                data.age_groups["55+"] = float(match_55.group(1))
            else:
                # Fallback to Hebrew text
                match_55 = re.search(r'(\d+)%\s*[+]?55[+]?(?:\s|$)', text_clean)
                if match_55:
                    data.age_groups["55+"] = float(match_55.group(1))
                else:
                    match_55 = re.search(r'[+]?55[+]?\s*(\d+)%', text_clean)
                    if match_55:
                        data.age_groups["55+"] = float(match_55.group(1))

        # Top locations - מיקומים section
        # The OCR structure is complex: country names appear first, then percentages at the end
        # BUT the gender percentages (95%, 5%, 0%) also appear in this area before location percentages
        # Location percentages start with decimal values like 74.2%, 19.5%
        locations = []

        # Get section after מיקומים
        loc_section = re.search(r'מיקומים.*?הודו\s*(.*?)$', text_clean, re.DOTALL)
        if loc_section:
            pcts_text = loc_section.group(1)
            # Find all percentages in this section
            pcts = re.findall(r'[<>]?(\d+\.?\d*)%', pcts_text)
            # First 3 are gender (95, 5, 0), rest are locations
            if len(pcts) >= 6:
                # Skip first 3 (gender), then Israel, Palestine, USA, UK, Other, Australia, India
                loc_pcts = pcts[3:]  # Skip gender
                country_names = ["Israel", "Palestinian Territories", "USA", "UK", "Other", "Australia", "India"]
                for i, name in enumerate(country_names):
                    if i < len(loc_pcts):
                        pct_str = loc_pcts[i].replace('<', '').replace('>', '')
                        try:
                            locations.append({"country": name, "percent": float(pct_str)})
                        except ValueError:
                            pass
        else:
            # Fallback: look for decimal percentages (74.2%, 19.5%, etc.)
            decimal_matches = re.findall(r'(\d+\.\d+)%', text_clean)
            if len(decimal_matches) >= 2:
                locations.append({"country": "Israel", "percent": float(decimal_matches[0])})
                locations.append({"country": "Palestinian Territories", "percent": float(decimal_matches[1])})

        data.top_locations = sorted(locations, key=lambda x: x["percent"], reverse=True)

        return data

    def extract_engagement_data(self, text: str) -> EngagementData:
        """Extract data from Engagement tab screenshot"""
        data = EngagementData()

        # Likes timestamp - when most users liked
        likes_match = re.search(r'ב-(\d+:\d+)', text)
        if likes_match:
            data.likes_timestamp = likes_match.group(1)

        # This tab mostly shows a graph and "top words in comments"
        # which requires more data to be meaningful

        return data

    def process_screenshot(self, image_path: str) -> dict:
        """Process a single screenshot and extract data"""
        print(f"Processing: {image_path}")

        text = self.extract_text(image_path)
        print(f"  Extracted {len(text)} characters of text")

        tab_type = self.detect_tab_type(text)
        print(f"  Detected tab type: {tab_type}")

        if tab_type == "overview":
            data = self.extract_overview_data(text)
        elif tab_type == "viewers":
            data = self.extract_viewers_data(text)
        elif tab_type == "engagement":
            data = self.extract_engagement_data(text)
        else:
            return {
                "tab_type": "unknown",
                "raw_text": text,
                "error": "Could not determine tab type"
            }

        return {
            "tab_type": tab_type,
            "data": asdict(data),
            "raw_text": text
        }

    def process_multiple(self, image_paths: list[str], output_path: Optional[str] = None) -> dict:
        """Process multiple screenshots and combine results"""
        results = {
            "overview": None,
            "viewers": None,
            "engagement": None,
            "raw_texts": {}
        }

        for path in image_paths:
            result = self.process_screenshot(path)
            tab_type = result.get("tab_type", "unknown")

            if tab_type != "unknown":
                results[tab_type] = result.get("data")
                results["raw_texts"][tab_type] = result.get("raw_text")
            else:
                filename = Path(path).stem
                results["raw_texts"][filename] = result.get("raw_text")

        if output_path:
            output_data = {k: v for k, v in results.items() if k != "raw_texts"}
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\nSaved results to: {output_path}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract analytics data from TikTok Studio screenshots using Google Cloud Vision API"
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Path(s) to screenshot image(s)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output JSON file path (default: tiktok_analytics.json)",
        default="tiktok_analytics.json"
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable Tesseract fallback when Vision API fails"
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Include raw OCR text in output"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Output spatial analysis debug information"
    )

    args = parser.parse_args()

    ocr = TikTokStudioOCR(use_tesseract_fallback=not args.no_fallback)
    results = ocr.process_multiple(args.images, args.output)

    print("\n" + "=" * 50)
    print("EXTRACTION SUMMARY")
    print("=" * 50)

    for tab in ["overview", "viewers", "engagement"]:
        if results.get(tab):
            print(f"\n{tab.upper()} TAB:")
            data = results[tab]
            for key, value in data.items():
                if value is not None and value != {} and value != []:
                    if isinstance(value, dict):
                        non_null = {k: v for k, v in value.items() if v is not None}
                        if non_null:
                            print(f"  {key}: {non_null}")
                    elif isinstance(value, list) and value:
                        print(f"  {key}: {value}")
                    else:
                        print(f"  {key}: {value}")

    if args.show_raw:
        print("\n" + "=" * 50)
        print("RAW OCR TEXT")
        print("=" * 50)
        for name, text in results.get("raw_texts", {}).items():
            print(f"\n--- {name} ---")
            print(text[:500] + "..." if len(text) > 500 else text)

    if args.debug:
        print("\n" + "=" * 50)
        print("SPATIAL ANALYSIS DEBUG")
        print("=" * 50)

        # Get annotations from last processed image
        annotations = ocr._get_text_with_positions()
        if annotations:
            rows = ocr._group_by_rows(annotations)
            print(f"\nFound {len(annotations)} text annotations, grouped into {len(rows)} rows:\n")
            for i, row in enumerate(rows):
                row_texts = [f"'{ann['text']}' (x={ann['x']}, y={ann['y']})" for ann in row]
                print(f"Row {i:2d}: {', '.join(row_texts)}")
        else:
            print("\nNo spatial annotations available (Vision API response not stored)")


if __name__ == "__main__":
    main()
