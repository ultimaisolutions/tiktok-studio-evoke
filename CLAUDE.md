# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TikTok video scraper and analyzer with **React UI** and **TikTok Studio automation**. Includes API pattern extraction for resilient scraping.

## Quick Start

### UI Mode (Recommended)

```bash
npm run setup    # Install all dependencies
npm run dev      # Start all services
```

Opens at http://localhost:5173

### CLI Mode

```bash
python -m venv .venv && .venv\Scripts\activate  # Windows
pip install -r requirements.txt
playwright install

python main.py --studio          # TikTok Studio mode
python main.py --analyze-only    # Analyze existing videos
```

## Architecture

```
tiktok-studio-evoke/
├── backend/                 # FastAPI backend (port 8000)
│   ├── main.py              # App entry point
│   ├── models/schemas.py    # Pydantic models
│   ├── routes/
│   │   ├── scraper.py       # POST /api/scraper/download
│   │   ├── studio.py        # POST /api/studio/start
│   │   ├── analysis.py      # POST /api/analysis/start
│   │   ├── videos.py        # GET /api/videos
│   │   ├── api_extraction.py # POST /api/extractor/start
│   │   └── websocket.py     # WS /ws/{job_id}
│   └── services/
│       ├── studio_service.py
│       └── api_extraction_service.py
├── client/                  # React + Vite (port 5173)
│   └── src/components/Views/
│       ├── DownloadView.jsx
│       ├── StudioView.jsx
│       ├── AnalysisView.jsx
│       ├── VideosView.jsx
│       └── APIExtractionView.jsx
├── main.py                  # CLI entry point
├── studio_scraper.py        # TikTok Studio Playwright automation
├── api_extractor.py         # API pattern extraction via network interception
├── scraper.py               # Video download via pyktok
├── analyzer.py              # Video analysis with multiprocessing
└── api_patterns.json        # Extracted API patterns (auto-generated)
```

## Key Classes

**Backend Services:**
- `StudioService` - TikTok Studio scraping with browser-based API fetch
- `APIExtractionService` - Network interception to capture API patterns
- `ScraperService` - Video download job management
- `AnalysisService` - Batch video analysis
- `ProgressManager` - WebSocket broadcast manager

**Python Core:**
- `TikTokStudioScraper` - Playwright automation, uses `page.evaluate()` for API calls
- `TikTokAPIExtractor` - Captures XHR/Fetch traffic, extracts endpoint patterns
- `VideoAnalyzer` - Multiprocessing video analysis

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/studio/start` | Start Studio session |
| POST | `/api/studio/continue/{id}` | Continue after login |
| POST | `/api/extractor/start` | Start API extraction |
| GET | `/api/extractor/patterns` | Get saved patterns |
| POST | `/api/extractor/apply` | Apply patterns to scraper |
| POST | `/api/scraper/download` | Start download job |
| POST | `/api/analysis/start` | Start batch analysis |
| GET | `/api/videos` | List downloaded videos |
| WS | `/ws/{job_id}` | Real-time progress |

## Recent Changes (Dec 2024)

### API Extraction Feature
- New "API Extract" tab in UI captures TikTok Studio API patterns
- Network interception via Playwright `page.on("request/response")`
- Extracts endpoints, headers, params, response schemas
- Saved to `api_patterns.json` for scraper to use

### Browser-Based API Fetch
- `_fetch_video_list_via_browser()` uses `page.evaluate()` with fetch
- Solves security token (X-Bogus) expiration issue
- Browser generates fresh tokens automatically
- Falls back to `requests` library if browser fetch fails

### Key Implementation Details

**API calls through browser context:**
```python
result = await self.page.evaluate('''
    async (args) => {
        const response = await fetch(args.endpoint, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(args.body),
            credentials: 'include'
        });
        return await response.json();
    }
''', {"endpoint": endpoint, "body": request_body})
```

**Pattern loading in scraper:**
```python
self._api_patterns = self._load_api_patterns()  # From api_patterns.json
```

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
└── {username}/{date}/
    ├── {video_id}.mp4
    ├── {video_id}.json
    ├── {video_id}_overview.png
    ├── {video_id}_viewers.png
    └── {video_id}_engagement.png
```

## Known Issues

- TikTok Studio never reaches "networkidle" - use `wait_until="domcontentloaded"`
- Security tokens (X-Bogus) expire quickly - use browser-based fetch
- MediaPipe requires Python <3.13, falls back to Haar cascades
- Studio mode may require manual login if cookies not found
