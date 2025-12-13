# TikTok Studio Extractor

An educational tool for understanding TikTok's Creator Studio, video analysis techniques, and browser automation patterns. Built with React, FastAPI, and Playwright.

> **Educational Use**: This project demonstrates web scraping techniques, API pattern extraction, video analysis pipelines, and browser automation for learning purposes.

## Features

- **TikTok Studio Automation** - Playwright-based browser automation for Creator Studio
- **API Pattern Extraction** - Network interception to capture and analyze API structures
- **Video Analysis** - Computer vision analysis (faces, objects, motion, colors, audio)
- **React Web UI** - Modern interface with real-time WebSocket progress updates

## Quick Start

### Web UI (Recommended)

```bash
npm run setup    # Install Python + Node.js dependencies + Playwright browsers
npm run dev      # Start all services (React + Express + FastAPI)
```

Open http://localhost:5173

### CLI

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
playwright install

# Run
python main.py --studio       # TikTok Studio mode
python main.py --analyze-only # Analyze existing videos
```

## CLI Reference

### Basic Options

| Option | Default | Description |
|--------|---------|-------------|
| `-i, --input` | `urls.txt` | Input file with TikTok URLs |
| `-o, --output` | `videos` | Output directory |
| `-b, --browser` | `chrome` | Browser for cookies (chrome/firefox/edge/opera/brave/chromium) |
| `-l, --log` | `errors.log` | Error log file |
| `--no-browser` | - | Skip browser cookies (public videos only) |

### TikTok Studio Options

| Option | Default | Description |
|--------|---------|-------------|
| `--studio` | - | Enable Studio mode (screenshots + download + analyze) |
| `--studio-browser` | `chromium` | Browser for automation (chromium/firefox/webkit) |
| `--skip-download` | - | Only capture screenshots |
| `--skip-analysis` | - | Download without analysis |
| `--cdp-port` | auto | CDP port for existing browser (auto-scans 9222-9229) |
| `--username` | - | TikTok username for video URLs |

### Analysis Options

| Option | Default | Description |
|--------|---------|-------------|
| `--analyze` | - | Analyze videos after download |
| `--analyze-only` | - | Only analyze existing videos |
| `--thoroughness` | `balanced` | Preset: quick/balanced/thorough/maximum/extreme |
| `--sample-percent` | - | Percentage of frames to sample (1-100) |
| `--sample-frames` | - | Number of frames to sample (1-300) |
| `--color-clusters` | - | K-means clusters for color analysis (3-20) |
| `--motion-res` | - | Motion analysis resolution width (80-1080) |
| `--face-model` | - | MediaPipe model: short (fast) / full (accurate) |
| `--workers` | CPU-1 | Parallel analysis workers |
| `--skip-audio` | - | Skip audio analysis |
| `--scene-detection` | - | Enable scene/cut detection |
| `--full-resolution` | - | Analyze at full resolution |

### Examples

```bash
# TikTok Studio - full pipeline
python main.py --studio

# Studio - screenshots only, no download
python main.py --studio --skip-download

# Analyze existing videos with extreme quality
python main.py --analyze-only --thoroughness extreme

# Download from custom file with Firefox cookies
python main.py -i my_urls.txt -b firefox

# Fast analysis with 30% frame sampling
python main.py --analyze-only --sample-percent 30 --thoroughness quick
```

## Analysis Presets

| Preset | Frames | YOLO | Scene Detection |
|--------|--------|------|-----------------|
| quick | 20% | No | No |
| balanced | 40% | No | No |
| thorough | 60% | Yes | Yes |
| maximum | 70% | Yes | Yes |
| extreme | 80% | Yes | Yes |

## Requirements

- Python 3.8+ (3.13 compatible)
- Node.js 18+
- Chrome/Chromium (for Studio automation)

## Architecture

```
backend/            FastAPI REST API + WebSocket
client/             React + Vite frontend
main.py             CLI entry point
studio_scraper.py   TikTok Studio automation
api_extractor.py    API pattern extraction
analyzer.py         Video analysis engine
```

## License

Apache License 2.0
