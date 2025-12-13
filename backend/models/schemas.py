"""Pydantic models for request/response validation."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


class BrowserType(str, Enum):
    """Browser types for cookie extraction."""
    chrome = "chrome"
    firefox = "firefox"
    edge = "edge"
    opera = "opera"
    brave = "brave"
    chromium = "chromium"


class StudioBrowserType(str, Enum):
    """Browser types for Playwright Studio automation."""
    chromium = "chromium"
    firefox = "firefox"
    webkit = "webkit"


class ThoroughnessPreset(str, Enum):
    """Analysis thoroughness presets."""
    quick = "quick"
    balanced = "balanced"
    thorough = "thorough"
    maximum = "maximum"
    extreme = "extreme"


class FaceModel(str, Enum):
    """Face detection model types."""
    short = "short"
    full = "full"


class JobStatusEnum(str, Enum):
    """Job status states."""
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class AnalysisOptions(BaseModel):
    """Configuration options for video analysis."""
    thoroughness: ThoroughnessPreset = ThoroughnessPreset.balanced
    sample_frames: Optional[int] = Field(None, ge=1, le=300, description="Override frame count")
    sample_percent: Optional[int] = Field(None, ge=1, le=100, description="Percentage of frames to sample")
    color_clusters: Optional[int] = Field(None, ge=3, le=20, description="K-means color clustering")
    motion_res: Optional[int] = Field(None, ge=80, le=1080, description="Motion analysis resolution")
    face_model: Optional[FaceModel] = Field(None, description="MediaPipe face model type")
    workers: Optional[int] = Field(None, ge=1, le=16, description="Parallel workers for analysis")
    skip_audio: bool = Field(False, description="Skip audio analysis")
    scene_detection: bool = Field(False, description="Enable scene/cut detection")
    full_resolution: bool = Field(False, description="Analyze without downsampling")
    enable_cloud_audio: bool = Field(True, description="Enable cloud speech transcription")
    cloud_audio_language: str = Field("en-US", description="Language code for transcription (BCP-47)")
    gcs_bucket: Optional[str] = Field(None, description="GCS bucket for large video files (>20MB)")
    remove_music: bool = Field(False, description="Remove background music before transcription")

class DownloadRequest(BaseModel):
    """Request to download videos from URLs."""
    urls: List[str] = Field(..., min_length=1, description="List of TikTok URLs")
    output_dir: str = Field("videos", description="Output directory")
    browser: BrowserType = Field(BrowserType.chrome, description="Browser for cookie extraction")
    no_browser: bool = Field(False, description="Skip browser initialization")
    analyze: bool = Field(False, description="Analyze videos after download")
    analysis_options: Optional[AnalysisOptions] = Field(None, description="Analysis configuration")


class StudioRequest(BaseModel):
    """Request to start TikTok Studio scraping session."""
    output_dir: str = Field("videos", description="Output directory")
    studio_browser: StudioBrowserType = Field(StudioBrowserType.chromium, description="Playwright browser")
    skip_download: bool = Field(False, description="Only capture screenshots")
    skip_analysis: bool = Field(False, description="Download but skip analysis")
    cdp_port: Optional[int] = Field(None, ge=1, le=65535, description="CDP port for existing browser")
    username: Optional[str] = Field(None, description="TikTok username")
    cookie_browser: BrowserType = Field(BrowserType.chrome, description="Browser for cookie extraction")
    analysis_options: Optional[AnalysisOptions] = Field(None, description="Analysis configuration")
    # Parallelization options
    studio_workers: Optional[int] = Field(2, ge=1, le=4, description="Parallel browser pages for screenshots")
    download_workers: Optional[int] = Field(4, ge=1, le=8, description="Concurrent video downloads")
    request_delay_ms: Optional[int] = Field(1500, ge=500, le=5000, description="Delay between requests (ms)")


class AnalysisRequest(BaseModel):
    """Request to analyze existing videos."""
    output_dir: str = Field("videos", description="Directory containing videos")
    video_paths: Optional[List[str]] = Field(None, description="Specific video paths to analyze")
    analysis_options: AnalysisOptions = Field(default_factory=AnalysisOptions)


class JobStatus(BaseModel):
    """Status of a running or completed job."""
    job_id: str
    status: JobStatusEnum
    job_type: str = Field(..., description="Type: download, studio, analysis")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Progress 0.0 to 1.0")
    current_task: Optional[str] = Field(None, description="Current task description")
    total: int = Field(0, description="Total items to process")
    completed: int = Field(0, description="Items completed")
    failed: int = Field(0, description="Items failed")
    errors: List[str] = Field(default_factory=list, description="Error messages")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    result: Optional[Dict[str, Any]] = Field(None, description="Final result data")


class VideoInfo(BaseModel):
    """Information about a downloaded video."""
    video_id: str
    username: str
    path: str
    metadata_path: str
    thumbnail_path: Optional[str] = None
    screenshots: List[str] = Field(default_factory=list, description="Studio screenshot paths")
    metadata: Optional[Dict[str, Any]] = None
    analysis: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None


class ProgressUpdate(BaseModel):
    """WebSocket progress update message."""
    job_id: str
    event_type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class ConfigResponse(BaseModel):
    """Configuration options response."""
    browsers: List[str] = Field(default_factory=lambda: [b.value for b in BrowserType])
    studio_browsers: List[str] = Field(default_factory=lambda: [b.value for b in StudioBrowserType])
    presets: List[str] = Field(default_factory=lambda: [p.value for p in ThoroughnessPreset])
    face_models: List[str] = Field(default_factory=lambda: [f.value for f in FaceModel])
    preset_details: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: {
        "quick": {"frame_percent": 20, "color_clusters": 4, "motion_res": 360, "yolo": False, "scene_detect": False},
        "balanced": {"frame_percent": 40, "color_clusters": 6, "motion_res": 480, "yolo": False, "scene_detect": False},
        "thorough": {"frame_percent": 60, "color_clusters": 8, "motion_res": 640, "yolo": True, "scene_detect": True},
        "maximum": {"frame_percent": 70, "color_clusters": 12, "motion_res": 720, "yolo": True, "scene_detect": True},
        "extreme": {"frame_percent": 80, "color_clusters": 16, "motion_res": 1080, "yolo": True, "scene_detect": True},
    })


# ========== API Extraction Models ==========

class APIPattern(BaseModel):
    """Represents an extracted API pattern."""
    endpoint: str = Field(..., description="API endpoint URL (without query params)")
    method: str = Field(..., description="HTTP method (GET, POST, etc.)")
    headers: Dict[str, str] = Field(default_factory=dict, description="Request headers structure")
    query_params: Dict[str, Any] = Field(default_factory=dict, description="Query parameters")
    request_body_schema: Optional[Dict[str, Any]] = Field(None, description="Request body JSON schema")
    response_schema: Dict[str, Any] = Field(default_factory=dict, description="Response JSON schema")
    sample_response: Optional[Dict[str, Any]] = Field(None, description="Truncated sample response")
    captured_at: datetime = Field(default_factory=datetime.now, description="When pattern was captured")
    category: str = Field("unknown", description="API category: video_list, analytics, creator, other")


class APIPatternCollection(BaseModel):
    """Collection of extracted API patterns."""
    video_list_api: Optional[APIPattern] = Field(None, description="Video list API pattern")
    analytics_api: Optional[APIPattern] = Field(None, description="Analytics API pattern")
    other_apis: List[APIPattern] = Field(default_factory=list, description="Other discovered APIs")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")


class ExtractionRequest(BaseModel):
    """Request to start API pattern extraction."""
    studio_browser: StudioBrowserType = Field(StudioBrowserType.chromium, description="Playwright browser")
    cdp_port: Optional[int] = Field(None, ge=1, le=65535, description="CDP port for existing browser")
    sample_video_count: int = Field(3, ge=1, le=10, description="Number of videos to sample for analytics APIs")


class ExtractionStatus(BaseModel):
    """Status of an API extraction session."""
    session_id: str
    status: JobStatusEnum
    progress: float = Field(0.0, ge=0.0, le=1.0)
    current_task: Optional[str] = None
    total_requests_captured: int = 0
    relevant_patterns: int = 0
    needs_login: bool = False
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
