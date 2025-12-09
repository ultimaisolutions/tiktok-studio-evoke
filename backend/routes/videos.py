"""Videos API routes for listing and viewing downloaded videos."""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from backend.models.schemas import VideoInfo

router = APIRouter(prefix="/videos", tags=["videos"])

# Default output directory
DEFAULT_OUTPUT_DIR = "videos"


def normalize_username(username: str) -> str:
    """Normalize username: remove @, lowercase, remove dots."""
    if not username:
        return "unknown"
    return username.lstrip("@").lower().replace(".", "")


@router.get("")
async def list_videos(
    output_dir: str = Query(DEFAULT_OUTPUT_DIR, description="Output directory to scan"),
    username: Optional[str] = Query(None, description="Filter by username"),
    limit: int = Query(10000, ge=1, le=10000, description="Maximum videos to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    sort_by: str = Query("publish_date", description="Sort by: publish_date, download_date"),
    sort_order: str = Query("desc", description="Sort order: asc, desc")
):
    """
    List all downloaded videos with their metadata.

    Returns list of videos sorted by date (newest first).
    """
    output_path = Path(output_dir)

    if not output_path.exists():
        return []

    videos = []

    # Scan for video files
    for mp4_file in output_path.rglob("*.mp4"):
        json_file = mp4_file.with_suffix(".json")

        # Extract info from path structure: videos/{username}/{date}/{video_id}.mp4
        parts = mp4_file.relative_to(output_path).parts

        video_username = normalize_username(parts[0]) if len(parts) > 2 else "unknown"
        video_date = parts[1] if len(parts) > 2 else None
        video_id = mp4_file.stem

        # Filter by username if specified
        if username and video_username != username:
            continue

        # Load metadata if exists
        metadata = None
        analysis = None
        if json_file.exists():
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    analysis = data.pop("analysis", None)
                    metadata = data
            except:
                pass

        # Find screenshots
        screenshots = []
        for suffix in ["_overview.png", "_viewers.png", "_engagement.png"]:
            screenshot_path = mp4_file.parent / f"{video_id}{suffix}"
            if screenshot_path.exists():
                screenshots.append(str(screenshot_path.relative_to(output_path)))

        # Get download date from file modification time
        download_timestamp = mp4_file.stat().st_mtime
        download_date = datetime.fromtimestamp(download_timestamp).strftime("%Y-%m-%d %H:%M")

        videos.append({
            "video_id": video_id,
            "username": video_username,
            "date": video_date,
            "download_date": download_date,
            "download_timestamp": download_timestamp,
            "path": str(mp4_file.relative_to(output_path)),
            "absolute_path": str(mp4_file),
            "metadata_path": str(json_file.relative_to(output_path)) if json_file.exists() else None,
            "screenshots": screenshots,
            "has_metadata": metadata is not None,
            "has_analysis": analysis is not None,
            "metadata": metadata,
            "analysis_summary": _summarize_analysis(analysis) if analysis else None
        })

    # Sort based on parameters
    if sort_by == "download_date":
        sort_key = lambda v: v.get("download_timestamp", 0)
    else:  # publish_date (default)
        sort_key = lambda v: v.get("date", "") or ""

    videos.sort(key=sort_key, reverse=(sort_order == "desc"))

    # Apply pagination and return with total count
    return {"videos": videos[offset:offset + limit], "total": len(videos)}


@router.get("/{video_id}", response_model=dict)
async def get_video_details(
    video_id: str,
    output_dir: str = Query(DEFAULT_OUTPUT_DIR)
):
    """Get detailed information about a specific video."""
    output_path = Path(output_dir)

    # Search for video file
    video_file = None
    for mp4_file in output_path.rglob(f"{video_id}.mp4"):
        video_file = mp4_file
        break

    if not video_file:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    json_file = video_file.with_suffix(".json")

    # Extract path info
    parts = video_file.relative_to(output_path).parts
    video_username = parts[0] if len(parts) > 2 else "unknown"
    video_date = parts[1] if len(parts) > 2 else None

    # Load full metadata and analysis
    metadata = None
    analysis = None
    if json_file.exists():
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                analysis = data.pop("analysis", None)
                metadata = data
        except:
            pass

    # Find screenshots
    screenshots = {}
    for tab in ["overview", "viewers", "engagement"]:
        screenshot_path = video_file.parent / f"{video_id}_{tab}.png"
        if screenshot_path.exists():
            screenshots[tab] = str(screenshot_path.relative_to(output_path))

    return {
        "video_id": video_id,
        "username": video_username,
        "date": video_date,
        "path": str(video_file.relative_to(output_path)),
        "absolute_path": str(video_file),
        "metadata_path": str(json_file.relative_to(output_path)) if json_file.exists() else None,
        "screenshots": screenshots,
        "metadata": metadata,
        "analysis": analysis
    }


@router.get("/{video_id}/file")
async def get_video_file(
    video_id: str,
    output_dir: str = Query(DEFAULT_OUTPUT_DIR)
):
    """Download the video file."""
    output_path = Path(output_dir)

    # Search for video file
    for mp4_file in output_path.rglob(f"{video_id}.mp4"):
        return FileResponse(
            mp4_file,
            media_type="video/mp4",
            filename=f"{video_id}.mp4"
        )

    raise HTTPException(status_code=404, detail=f"Video {video_id} not found")


@router.get("/{video_id}/screenshot/{tab}")
async def get_screenshot(
    video_id: str,
    tab: str,
    output_dir: str = Query(DEFAULT_OUTPUT_DIR)
):
    """Get a screenshot image for a video."""
    if tab not in ["overview", "viewers", "engagement"]:
        raise HTTPException(status_code=400, detail="Invalid tab. Use: overview, viewers, engagement")

    output_path = Path(output_dir)

    # Search for screenshot
    for mp4_file in output_path.rglob(f"{video_id}.mp4"):
        screenshot_path = mp4_file.parent / f"{video_id}_{tab}.png"
        if screenshot_path.exists():
            return FileResponse(
                screenshot_path,
                media_type="image/png",
                filename=f"{video_id}_{tab}.png"
            )
        break

    raise HTTPException(status_code=404, detail=f"Screenshot not found")


@router.post("/{video_id}/open-in-explorer")
async def open_in_explorer(
    video_id: str,
    output_dir: str = Query(DEFAULT_OUTPUT_DIR)
):
    """Open File Explorer and select the video file."""
    output_path = Path(output_dir)

    # Search for video file
    video_file = None
    for mp4_file in output_path.rglob(f"{video_id}.mp4"):
        video_file = mp4_file
        break

    if not video_file:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    # Open File Explorer and select the file (Windows)
    subprocess.run(['explorer', '/select,', str(video_file.resolve())])

    return {"success": True, "path": str(video_file.resolve())}


def _summarize_analysis(analysis: dict) -> dict:
    """Create a brief summary of analysis results."""
    if not analysis:
        return None

    return {
        "version": analysis.get("version"),
        "processing_time_ms": analysis.get("processing_time_ms"),
        "video_quality": {
            "resolution": analysis.get("video_quality", {}).get("resolution"),
            "duration": analysis.get("video_quality", {}).get("duration_seconds"),
            "fps": analysis.get("video_quality", {}).get("fps")
        },
        "motion_level": analysis.get("motion_analysis", {}).get("motion_level"),
        "has_face": analysis.get("content_detection", {}).get("face_detected"),
        "has_person": analysis.get("content_detection", {}).get("person_detected"),
        "has_text": analysis.get("content_detection", {}).get("text_overlay_detected"),
        "has_audio": analysis.get("audio_metrics", {}).get("has_audio"),
        "color_temperature": analysis.get("color_analysis", {}).get("color_temperature"),
    }
