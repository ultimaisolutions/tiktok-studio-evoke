# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python-based TikTok video scraper and analyzer with **TikTok Studio automation**.

Features:
- Download videos with metadata, organize by username and date
- Video analysis (visual metrics, face/person detection, motion, color, scenes, audio)
- **TikTok Studio mode**: Automate extraction of analytics screenshots from TikTok Studio

**Project root:** `tiktok-scraper-analyzer-main/`

## Quick Start

```bash
cd tiktok-scraper-analyzer-main

# Setup (Windows)
python -m venv .venv && .venv\Scripts\activate.bat

# Setup (macOS/Linux)
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
playwright install
```

## Commands

### Standard Mode (URL-based)

```bash
# Download from urls.txt to videos/
python main.py

# Custom input/output
python main.py -i my_urls.txt -o downloads/

# Use specific browser for cookies
python main.py -b firefox

# Download and analyze
python main.py --analyze

# Only analyze existing videos
python main.py --analyze-only

# Analysis presets
python main.py --analyze --thoroughness extreme
```

### TikTok Studio Mode

Automates extraction of analytics data directly from TikTok Studio web interface.

```bash
# Full pipeline: screenshots + download + analyze
python main.py --studio

# Only capture screenshots (no download)
python main.py --studio --skip-download

# Download but skip analysis
python main.py --studio --skip-analysis

# Use Firefox for Studio automation
python main.py --studio --studio-browser firefox

# Custom output directory
python main.py --studio -o studio_data/
```

**What Studio Mode Does:**
1. Opens browser, navigates to TikTok Studio
2. Attempts cookie-based login, falls back to manual login if needed
3. For each video in Studio:
   - Captures screenshots of all 3 analytics tabs (Overview, Viewers, Engagement)
   - Extracts video URL from thumbnail
4. Saves URLs to log file: `studio_urls_{timestamp}.txt`
5. Downloads videos using existing scraper
6. Analyzes with `extreme` preset + 50% frame sampling (Studio defaults)

**Supported Platforms:** Windows, macOS, Linux

## Architecture

```
tiktok-scraper-analyzer-main/
├── main.py              # CLI entry point & orchestration
├── studio_scraper.py    # TikTokStudioScraper: Playwright automation for Studio
├── scraper.py           # TikTokScraper: downloads via pyktok, cookie auth
├── analyzer.py          # VideoAnalyzer: batch processing, metrics extraction
├── analysis_models.py   # AnalysisModels: MediaPipe/YOLO/Haar detection
├── utils.py             # Helpers: logging, URL parsing, metadata formatting
├── urls.txt             # Input URLs (one per line)
└── requirements.txt
```

## Key Classes

**TikTokStudioScraper** ([studio_scraper.py](tiktok-scraper-analyzer-main/studio_scraper.py)):
- `initialize()` - Launch browser, attempt cookie login, fallback to manual
- `scrape_all_videos()` - Iterate through Studio videos, capture screenshots
- `_process_single_video()` - Screenshot 3 tabs, extract video URL
- `_capture_tab_screenshot()` - Save full-page screenshot of analytics tab

**TikTokScraper** ([scraper.py](tiktok-scraper-analyzer-main/scraper.py)):
- `initialize_browser(browser, required)` - Setup browser cookies for pyktok
- `download_video(url)` - Download single video + metadata
- `process_urls(urls)` - Batch process with statistics tracking

**VideoAnalyzer** ([analyzer.py](tiktok-scraper-analyzer-main/analyzer.py)):
- `analyze_video(path)` - Analyze single video for all metrics
- `analyze_batch(paths, workers)` - Parallel processing with multiprocessing
- `update_metadata_file(json_path, result)` - Merge analysis into metadata JSON

**AnalysisModels** ([analysis_models.py](tiktok-scraper-analyzer-main/analysis_models.py)):
- Singleton pattern for lazy-loaded ML models
- `detect_faces(frame, model_type)` - Face detection (MediaPipe or Haar fallback)
- `detect_persons(frame, use_yolo)` - Person detection with fallback chain

## Thoroughness Presets

| Preset | Frames | Color K | YOLO | Scene Detect | Use Case |
|--------|--------|---------|------|--------------|----------|
| `quick` | 15 | 4 | No | No | Fast testing |
| `balanced` | 30 | 6 | No | No | Default |
| `thorough` | 50 | 8 | No | No | Better accuracy |
| `maximum` | 80 | 12 | Yes | No | High quality |
| `extreme` | 150 | 16 | Yes | Yes | Max GPU (Studio default) |

## Output Structure

```
videos/
├── studio_urls_{timestamp}.txt              # Log of extracted URLs (Studio mode)
└── {username}/
    └── {YYYY-MM-DD}/
        ├── {video_id}.mp4                   # Video file
        ├── {video_id}.json                  # Metadata + analysis results
        ├── {video_id}_overview.png          # Studio Overview tab screenshot
        ├── {video_id}_viewers.png           # Studio Viewers tab screenshot
        └── {video_id}_engagement.png        # Studio Engagement tab screenshot
```

## Dependencies

**Core:** pyktok, playwright, browser-cookie3, beautifulsoup4, requests, pandas, numpy

**Analysis:** opencv-python-headless (>=4.8.0), moviepy (>=1.0.3), scikit-image (>=0.21.0)

**GPU:** ultralytics (>=8.0.0) - YOLO for maximum/extreme presets

**Optional:** mediapipe (>=0.10.0) - requires Python <3.13, falls back to Haar cascades

## Known Issues

- Browser cookie extraction may fail with "Unable to get key for cookie decryption" - use `--no-browser` for public videos
- Browser must be closed when extracting cookies (for standard mode)
- MediaPipe requires Python <3.13; uses Haar cascades on newer Python versions
- Studio mode requires manual login if cookies aren't found
