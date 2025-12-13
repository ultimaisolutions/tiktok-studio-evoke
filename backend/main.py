"""
FastAPI application entry point for TikTok Studio Scraper API.

Run with: uvicorn backend.main:app --reload --port 8000
"""

import asyncio
import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.routes import (
    scraper_router,
    studio_router,
    analysis_router,
    websocket_router,
    videos_router,
    config_router,
    api_extraction_router,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    # Startup
    logger.info("Starting TikTok Studio API server...")
    logger.info("API documentation available at: http://localhost:8000/docs")
    yield
    # Shutdown
    logger.info("Shutting down TikTok Studio API server...")


# Create FastAPI application
app = FastAPI(
    title="TikTok Studio Scraper API",
    description="""
API for downloading TikTok videos, automating TikTok Studio analytics,
and performing video analysis.

## Features

- **Download Videos**: Download TikTok videos from URLs with metadata
- **Studio Automation**: Capture analytics screenshots from TikTok Studio
- **Video Analysis**: Analyze videos for visual metrics, faces, persons, colors, etc.
- **Real-time Progress**: WebSocket support for live progress updates

## WebSocket

Connect to `/ws/{job_id}` after starting a job to receive real-time progress updates.
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3001",  # Express proxy
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(scraper_router, prefix="/api")
app.include_router(studio_router, prefix="/api")
app.include_router(analysis_router, prefix="/api")
app.include_router(videos_router, prefix="/api")
app.include_router(config_router, prefix="/api")
app.include_router(api_extraction_router, prefix="/api")
app.include_router(websocket_router)  # WebSocket at root level


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "TikTok Studio Scraper API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "scraper": "/api/scraper",
            "studio": "/api/studio",
            "analysis": "/api/analysis",
            "videos": "/api/videos",
            "config": "/api/config",
            "extractor": "/api/extractor",
            "websocket": "/ws/{job_id}"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
