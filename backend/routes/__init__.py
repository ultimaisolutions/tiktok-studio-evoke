from .scraper import router as scraper_router
from .studio import router as studio_router
from .analysis import router as analysis_router
from .websocket import router as websocket_router
from .videos import router as videos_router
from .config import router as config_router

__all__ = [
    "scraper_router",
    "studio_router",
    "analysis_router",
    "websocket_router",
    "videos_router",
    "config_router",
]
