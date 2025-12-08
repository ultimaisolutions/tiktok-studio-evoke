# TikTok Studio Extractor

A tool for automating TikTok Studio analytics extraction, downloading videos, and performing video analysis — with a **React UI** or CLI.

## Quick Start (UI)

```bash
# Install all dependencies (Node.js + Python + Playwright browsers)
npm run setup

# Start the app (React + Express + FastAPI)
npm run dev
```

Open http://localhost:5173 in your browser.

> **Note**: `npm run setup` automatically installs Playwright's Chromium browser for Studio automation.

## Quick Start (CLI)

```bash
# Setup
python -m venv .venv && .venv\Scripts\activate  # Windows
python -m venv .venv && source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
playwright install

# Download videos from urls.txt
python main.py

# TikTok Studio mode (screenshots + download + analyze)
python main.py --studio

# Analyze existing videos
python main.py --analyze-only
```

## Features

| Feature | Description |
|---------|-------------|
| **Web UI** | React interface for all operations with real-time progress |
| **Studio Mode** | Automate TikTok Studio screenshot capture for analytics |
| **Video Download** | Download videos with metadata from URLs |
| **Video Analysis** | Visual metrics, face/person detection, motion, color, audio |
| **OCR Extraction** | Extract analytics data from screenshots via Google Vision API |

## UI Overview

The web UI provides four main views:

- **Download**: Paste URLs, configure browser/analysis options
- **Studio**: TikTok Studio automation with manual login support
- **Analysis**: Batch analyze existing videos with preset options
- **Videos**: Browse downloaded videos with metadata and analysis results

All operations show real-time progress via WebSocket.

## CLI Commands

### TikTok Studio Mode

```bash
python main.py --studio                    # Full pipeline
python main.py --studio --skip-download    # Screenshots only
python main.py --studio --skip-analysis    # Download without analysis
python main.py --studio --studio-browser firefox
```

### Standard Download

```bash
python main.py                             # Download from urls.txt
python main.py -i urls.txt -o videos/      # Custom input/output
python main.py -b firefox                  # Use Firefox cookies
python main.py --no-browser                # Public videos only
```

### Video Analysis

```bash
python main.py --analyze                   # Download + analyze
python main.py --analyze-only              # Analyze existing videos
python main.py --analyze --thoroughness extreme
python main.py --analyze --sample-percent 70
```

### OCR (Analytics Screenshots)

```bash
python tiktok_studio_ocr.py screenshot.png -o results.json
python tiktok_studio_ocr.py overview.png viewers.png engagement.png -o analytics.json
```

## Analysis Presets

| Preset | Frames | YOLO | Scene Detect | Use Case |
|--------|--------|------|--------------|----------|
| `quick` | 20% | No | No | Fast testing |
| `balanced` | 40% | No | No | Default |
| `thorough` | 60% | Yes | Yes | Detailed |
| `maximum` | 70% | Yes | Yes | High quality |
| `extreme` | 80% | Yes | Yes | Maximum detail |

## Output Structure

```
videos/
├── studio_urls_{timestamp}.txt     # Extracted URLs log
└── {username}/{date}/
    ├── {video_id}.mp4              # Video file
    ├── {video_id}.json             # Metadata + analysis
    ├── {video_id}_overview.png     # Studio Overview screenshot
    ├── {video_id}_viewers.png      # Studio Viewers screenshot
    └── {video_id}_engagement.png   # Studio Engagement screenshot
```

## Requirements

- **Python** 3.8+ (fully compatible with 3.13 on Windows)
- **Node.js** 18+ (for UI)
- **Browsers**: Chrome, Firefox, Edge, Opera, Brave, Chromium

> **Note**: MediaPipe requires Python < 3.13. On Python 3.13+, face detection falls back to Haar cascades automatically.

### Google Cloud Vision (for OCR)

1. Create project at [Google Cloud Console](https://console.cloud.google.com)
2. Enable Cloud Vision API
3. Create service account and download JSON key
4. Set environment variable:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
   ```

## Architecture

```
├── backend/          # FastAPI REST API + WebSocket
├── server/           # Express proxy
├── client/           # React + Vite frontend
├── main.py           # CLI entry point
├── studio_scraper.py # TikTok Studio automation
├── scraper.py        # Video downloader
├── analyzer.py       # Video analysis
└── tiktok_studio_ocr.py  # OCR extraction
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Cookie extraction fails | Close browser, try `--no-browser`, or use different browser |
| MediaPipe not available | Python 3.13+ falls back to Haar cascades automatically |
| Vision API errors | Check `GOOGLE_APPLICATION_CREDENTIALS` and API quota |
| Studio login required | Use manual login when prompted |
| Browser not launching | Ensure Chrome is installed; Studio mode auto-launches Chrome with remote debugging |
| WebSocket disconnects | Check that FastAPI is running on port 8000; UI auto-reconnects with backoff |

## License

Apache License 2.0
