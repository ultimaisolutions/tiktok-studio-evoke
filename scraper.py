"""
Core scraper logic for downloading TikTok videos and extracting metadata.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Tuple

import pyktok as pyk

from utils import (
    ensure_directory,
    extract_video_id,
    extract_username_from_url,
    format_metadata,
    timestamp_to_date,
)


class TikTokScraper:
    """Handles downloading TikTok videos and extracting metadata."""

    def __init__(self, output_dir: str, logger):
        """
        Initialize the scraper.

        Args:
            output_dir: Base directory for saving videos
            logger: Logger instance for status messages
        """
        self.output_dir = Path(output_dir)
        self.logger = logger
        self._browser_initialized = False

    def initialize_browser(self, browser: str = "chrome", required: bool = False) -> bool:
        """
        Initialize browser cookies for authenticated requests.

        Args:
            browser: Browser name (chrome, firefox, edge, etc.)
            required: If True, return False on failure. If False, warn and continue.

        Returns:
            True if successful or not required, False if required and failed
        """
        try:
            pyk.specify_browser(browser)
            self._browser_initialized = True
            self.logger.info(f"Browser initialized: {browser}")
            return True
        except Exception as e:
            self.logger.warning(f"Could not initialize browser cookies: {e}")
            self.logger.info("Continuing without browser authentication (public videos should still work)")
            self._browser_initialized = False
            return not required  # Return True if not required, False if required

    def download_video(self, url: str) -> Tuple[bool, str]:
        """
        Download a single TikTok video with metadata.

        Args:
            url: TikTok video URL

        Returns:
            Tuple of (success: bool, message: str)
        """
        video_id = extract_video_id(url)
        if not video_id:
            return False, f"Could not extract video ID from URL: {url}"

        try:
            # Get metadata first
            self.logger.info(f"Fetching metadata for: {url}")
            raw_metadata = pyk.alt_get_tiktok_json(url)

            if not raw_metadata:
                return False, f"Could not fetch metadata for: {url}"

            # Format metadata
            metadata = format_metadata(raw_metadata)
            metadata["source_url"] = url
            metadata["raw_video_id"] = video_id

            # Get username and date for folder structure
            username = metadata.get("username")
            if not username:
                username = extract_username_from_url(url)
            if not username:
                username = "unknown_user"

            create_date = metadata.get("create_date")
            if not create_date:
                create_date = timestamp_to_date(None)  # Use today's date

            # Use video_id from metadata if available
            final_video_id = metadata.get("video_id") or video_id

            # Create folder structure: videos/{username}/{YYYY-MM-DD}/
            video_folder = self.output_dir / username / create_date
            ensure_directory(video_folder)

            # Define output paths
            video_path = video_folder / f"{final_video_id}.mp4"
            json_path = video_folder / f"{final_video_id}.json"

            # Download video to temp location first
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_csv = os.path.join(temp_dir, "temp_metadata.csv")

                self.logger.info(f"Downloading video: {final_video_id}")
                pyk.save_tiktok(url, save_video=True, metadata_fn=temp_csv)

                # Find the downloaded video file in current directory
                # pyktok saves with pattern: {video_id}.mp4
                possible_names = [
                    f"{final_video_id}.mp4",
                    f"{video_id}.mp4",
                ]

                video_found = False
                for name in possible_names:
                    if os.path.exists(name):
                        shutil.move(name, str(video_path))
                        video_found = True
                        break

                # Also check for files matching the video ID pattern in CWD
                if not video_found:
                    for f in os.listdir("."):
                        if f.endswith(".mp4") and (final_video_id in f or video_id in f):
                            shutil.move(f, str(video_path))
                            video_found = True
                            break

                if not video_found:
                    # Video might have failed to download but metadata was retrieved
                    self.logger.warning(f"Video file not found for {final_video_id}, saving metadata only")

            # Save metadata JSON
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            if video_path.exists():
                return True, f"Downloaded: {username}/{create_date}/{final_video_id}.mp4"
            else:
                return True, f"Metadata saved (video unavailable): {username}/{create_date}/{final_video_id}.json"

        except Exception as e:
            error_msg = f"Error downloading {url}: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg

    def process_urls(self, urls: list[str]) -> dict:
        """
        Process multiple URLs and track results.

        Args:
            urls: List of TikTok URLs to download

        Returns:
            Dictionary with success/failure counts and details
        """
        results = {
            "total": len(urls),
            "success": 0,
            "failed": 0,
            "successful_urls": [],
            "failed_urls": [],
        }

        for i, url in enumerate(urls, 1):
            self.logger.info(f"\n[{i}/{len(urls)}] Processing: {url}")

            success, message = self.download_video(url)

            if success:
                results["success"] += 1
                results["successful_urls"].append({"url": url, "message": message})
                self.logger.info(f"SUCCESS: {message}")
            else:
                results["failed"] += 1
                results["failed_urls"].append({"url": url, "error": message})
                self.logger.warning(f"FAILED: {message}")

        return results
