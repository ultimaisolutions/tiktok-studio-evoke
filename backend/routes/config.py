"""Configuration API routes for available options."""

from fastapi import APIRouter

from backend.models.schemas import ConfigResponse, BrowserType, StudioBrowserType, ThoroughnessPreset

router = APIRouter(prefix="/config", tags=["config"])


@router.get("/browsers", response_model=dict)
async def get_available_browsers():
    """Get list of available browsers for cookie extraction."""
    return {
        "browsers": [b.value for b in BrowserType],
        "default": BrowserType.chrome.value
    }


@router.get("/studio-browsers", response_model=dict)
async def get_studio_browsers():
    """Get list of available browsers for Studio automation."""
    return {
        "browsers": [b.value for b in StudioBrowserType],
        "default": StudioBrowserType.chromium.value
    }


@router.get("/presets", response_model=dict)
async def get_thoroughness_presets():
    """Get available analysis thoroughness presets with details."""
    return {
        "presets": [p.value for p in ThoroughnessPreset],
        "default": ThoroughnessPreset.balanced.value,
        "details": {
            "quick": {
                "description": "Fast testing - 20% frame sampling",
                "frame_percent": 20,
                "color_clusters": 4,
                "motion_res": 360,
                "yolo": False,
                "scene_detect": False,
                "recommended_for": "Quick previews, testing"
            },
            "balanced": {
                "description": "Default - 40% frame sampling",
                "frame_percent": 40,
                "color_clusters": 6,
                "motion_res": 480,
                "yolo": False,
                "scene_detect": False,
                "recommended_for": "General use"
            },
            "thorough": {
                "description": "Better accuracy - 60% frame sampling",
                "frame_percent": 60,
                "color_clusters": 8,
                "motion_res": 640,
                "yolo": True,
                "scene_detect": True,
                "recommended_for": "Detailed analysis"
            },
            "maximum": {
                "description": "High quality - 70% frame sampling",
                "frame_percent": 70,
                "color_clusters": 12,
                "motion_res": 720,
                "yolo": True,
                "scene_detect": True,
                "recommended_for": "High quality analysis"
            },
            "extreme": {
                "description": "Maximum detail - 80% frame sampling (Studio default)",
                "frame_percent": 80,
                "color_clusters": 16,
                "motion_res": 1080,
                "yolo": True,
                "scene_detect": True,
                "recommended_for": "Maximum detail, GPU recommended"
            }
        }
    }


@router.get("/all", response_model=ConfigResponse)
async def get_all_config():
    """Get all configuration options."""
    return ConfigResponse()
