"""
Utility functions for TikTok video scraper.
Handles logging setup, directory creation, and URL parsing.
"""

import logging
import os
import re
from datetime import datetime
from pathlib import Path


def setup_logging(log_file: str = "errors.log") -> logging.Logger:
    """
    Configure logging to both file and console.

    Args:
        log_file: Path to the error log file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("tiktok_scraper")
    logger.setLevel(logging.DEBUG)

    # File handler for errors only
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.ERROR)
    file_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)

    # Console handler for all messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def ensure_directory(path: str | Path) -> Path:
    """
    Create directory and all parent directories if they don't exist.

    Args:
        path: Directory path to create

    Returns:
        Path object of the created directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def extract_video_id(url: str) -> str | None:
    """
    Extract video ID from a TikTok URL.

    Supports formats:
    - https://www.tiktok.com/@username/video/1234567890
    - https://vm.tiktok.com/ABC123/

    Args:
        url: TikTok video URL

    Returns:
        Video ID string or None if not found
    """
    # Standard video URL pattern
    match = re.search(r"/video/(\d+)", url)
    if match:
        return match.group(1)

    # Short URL pattern - return the short code
    match = re.search(r"vm\.tiktok\.com/([A-Za-z0-9]+)", url)
    if match:
        return match.group(1)

    return None


def extract_username_from_url(url: str) -> str | None:
    """
    Extract username from a TikTok URL.

    Args:
        url: TikTok video URL

    Returns:
        Username string or None if not found
    """
    match = re.search(r"@([A-Za-z0-9_.]+)", url)
    if match:
        return match.group(1)
    return None


def timestamp_to_date(timestamp: int | str) -> str:
    """
    Convert Unix timestamp to YYYY-MM-DD format.

    Args:
        timestamp: Unix timestamp (seconds since epoch)

    Returns:
        Date string in YYYY-MM-DD format
    """
    try:
        ts = int(timestamp)
        dt = datetime.fromtimestamp(ts)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError, OSError):
        # Return today's date as fallback
        return datetime.now().strftime("%Y-%m-%d")


def format_metadata(raw_data: dict) -> dict:
    """
    Clean and format raw TikTok metadata for JSON export.

    Args:
        raw_data: Raw metadata dictionary from pyktok

    Returns:
        Cleaned metadata dictionary
    """
    if not raw_data:
        return {}

    # Extract relevant fields from the nested structure
    formatted = {
        "video_id": None,
        "username": None,
        "nickname": None,
        "description": None,
        "create_time": None,
        "create_date": None,
        "duration": None,
        "play_count": None,
        "like_count": None,
        "comment_count": None,
        "share_count": None,
        "music_title": None,
        "music_author": None,
        "hashtags": [],
        "url": None,
    }

    try:
        # Navigate the nested JSON structure
        item_info = raw_data.get("itemInfo", {}).get("itemStruct", {})
        if not item_info:
            # Try alternative structure
            item_info = raw_data.get("__DEFAULT_SCOPE__", {}).get("webapp.video-detail", {}).get("itemInfo", {}).get("itemStruct", {})

        if item_info:
            formatted["video_id"] = item_info.get("id")
            formatted["description"] = item_info.get("desc")
            formatted["create_time"] = item_info.get("createTime")
            formatted["duration"] = item_info.get("video", {}).get("duration")

            # Convert timestamp to date
            if formatted["create_time"]:
                formatted["create_date"] = timestamp_to_date(formatted["create_time"])

            # Author info
            author = item_info.get("author", {})
            formatted["username"] = author.get("uniqueId")
            formatted["nickname"] = author.get("nickname")

            # Stats
            stats = item_info.get("stats", {})
            formatted["play_count"] = stats.get("playCount")
            formatted["like_count"] = stats.get("diggCount")
            formatted["comment_count"] = stats.get("commentCount")
            formatted["share_count"] = stats.get("shareCount")

            # Music info
            music = item_info.get("music", {})
            formatted["music_title"] = music.get("title")
            formatted["music_author"] = music.get("authorName")

            # Extract hashtags from challenges
            challenges = item_info.get("challenges", [])
            if challenges:
                formatted["hashtags"] = [c.get("title") for c in challenges if c.get("title")]

    except Exception:
        pass

    return formatted


def read_urls_from_file(file_path: str) -> list[str]:
    """
    Read TikTok URLs from a text file.

    Args:
        file_path: Path to the text file containing URLs

    Returns:
        List of URL strings (empty lines and comments ignored)
    """
    urls = []
    path = Path(file_path)

    if not path.exists():
        return urls

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith("#"):
                urls.append(line)

    return urls
