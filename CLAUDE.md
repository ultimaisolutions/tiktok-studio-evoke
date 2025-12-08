# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TikTok video scraper and analyzer with **React UI** and **TikTok Studio automation**.

## Quick Start

### UI Mode (Recommended)

```bash
# Install all dependencies (Node.js + Python)
npm run setup

# Start all services
npm run dev
```

Opens at http://localhost:5173

### CLI Mode

```bash
# Setup
python -m venv .venv && .venv\Scripts\activate  # Windows
pip install -r requirements.txt
playwright install

# Run
python main.py --studio          # TikTok Studio mode
python main.py                   # Download from urls.txt
python main.py --analyze-only    # Analyze existing videos
```

## Architecture

```
tiktok-studio-evoke/
├── backend/                 # FastAPI backend
│   ├── main.py              # App entry point (port 8000)
│   ├── models/schemas.py    # Pydantic models
│   ├── routes/              # API endpoints
│   │   ├── scraper.py       # POST /api/scraper/download
│   │   ├── studio.py        # POST /api/studio/start
│   │   ├── analysis.py      # POST /api/analysis/start
│   │   ├── videos.py        # GET /api/videos
│   │   └── websocket.py     # WS /ws/{job_id}
│   ├── services/            # Business logic wrappers
│   └── utils/               # WebSocket progress manager
├── server/                  # Express proxy (port 3001)
│   └── index.js
├── client/                  # React + Vite (port 5173)
│   └── src/
│       ├── components/Views/  # DownloadView, StudioView, etc.
│       └── hooks/           # useWebSocket, useApi
├── main.py                  # CLI entry point
├── studio_scraper.py        # TikTok Studio Playwright automation
├── scraper.py               # Video download via pyktok
├── analyzer.py              # Video analysis with multiprocessing
├── analysis_models.py       # ML models (MediaPipe/YOLO/Haar)
├── tiktok_studio_ocr.py     # OCR via Google Vision API
├── package.json             # Root scripts (npm run dev)
└── requirements.txt         # Python dependencies
```

## Key Commands

### npm Scripts (Root)

```bash
npm run dev          # Start all 3 services (Vite + Express + FastAPI)
npm run setup        # Install all dependencies
npm run dev:client   # Start React only
npm run dev:server   # Start Express only
npm run dev:api      # Start FastAPI only
```

### CLI Commands

```bash
# TikTok Studio
python main.py --studio
python main.py --studio --skip-download    # Screenshots only
python main.py --studio --skip-analysis

# Standard download
python main.py -i urls.txt -o videos/
python main.py -b firefox --analyze

# Analysis only
python main.py --analyze-only --thoroughness extreme
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/scraper/download` | Start download job |
| GET | `/api/scraper/status/{job_id}` | Get job status |
| POST | `/api/studio/start` | Start Studio session |
| POST | `/api/studio/continue/{id}` | Continue after login |
| POST | `/api/analysis/start` | Start batch analysis |
| GET | `/api/videos` | List downloaded videos |
| GET | `/api/videos/{id}` | Get video details |
| WS | `/ws/{job_id}` | Real-time progress |

## Key Classes

**Backend Services:**
- `ScraperService` - Wraps TikTokScraper with async job management
- `StudioService` - Wraps TikTokStudioScraper, handles login flow
- `AnalysisService` - Wraps VideoAnalyzer with progress callbacks
- `ProgressManager` - WebSocket broadcast manager

**Python Core:**
- `TikTokStudioScraper` - Playwright automation for Studio
- `TikTokScraper` - Video download via pyktok
- `VideoAnalyzer` - Batch analysis with multiprocessing
- `AnalysisModels` - Singleton ML model management

## Analysis Presets

| Preset | Frame % | YOLO | Scene Detect |
|--------|---------|------|--------------|
| `quick` | 20% | No | No |
| `balanced` | 40% | No | No |
| `thorough` | 60% | Yes | Yes |
| `maximum` | 70% | Yes | Yes |
| `extreme` | 80% | Yes | Yes |

## Output Structure

```
videos/
├── studio_urls_{timestamp}.txt
└── {username}/{date}/
    ├── {video_id}.mp4
    ├── {video_id}.json
    ├── {video_id}_overview.png
    ├── {video_id}_viewers.png
    └── {video_id}_engagement.png
```

## Dependencies

**Python:** pyktok, playwright, fastapi, uvicorn, opencv-python-headless, moviepy, ultralytics

**Node.js:** react, vite, express, http-proxy-middleware, concurrently

## OCR Configuration (Google Cloud Vision API)

Set the environment variable for OCR:

```bash
# Windows (PowerShell)
$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\path\to\credentials.json"

# Linux/macOS
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

## Known Issues

- Browser cookies: Close browser before extraction, or use `--no-browser`
- MediaPipe: Requires Python <3.13, falls back to Haar cascades
- Studio mode: May require manual login if cookies not found
- WebSocket: Reconnects automatically on disconnect
