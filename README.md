# TikTok Studio Extractor

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A Python-based tool for **automating TikTok Studio analytics extraction**, downloading TikTok videos with metadata, and performing comprehensive video analysis.

## Features

### TikTok Studio Automation (NEW)
- **Automated screenshots** of all 3 analytics tabs (Overview, Viewers, Engagement)
- **Cookie-based login** with manual fallback
- **Incremental processing** - skips already processed videos
- **URL logging** - saves extracted URLs for reference
- Cross-platform support (Windows, macOS, Linux)

### Video Downloading
- Download TikTok videos from URLs
- Extract and save video metadata (description, likes, comments, shares, etc.)
- Browser cookie authentication for accessing private/restricted content
- Automatic organization by username and date

### Video Analysis
- **Visual Quality Metrics**: Brightness, contrast, sharpness analysis
- **Face Detection**: MediaPipe or Haar cascade detection
- **Person Detection**: MediaPipe pose, YOLO, or Haar cascade fallback
- **Text Overlay Detection**: Detect on-screen text frequency
- **Motion Analysis**: Frame differencing for motion scoring
- **Color Analysis**: K-means clustering for dominant colors and temperature
- **Scene Detection**: Histogram-based cut detection
- **Audio Analysis**: Volume levels and speech detection

### Performance
- Parallel video processing with configurable worker threads
- GPU acceleration with YOLO for maximum/extreme presets
- Configurable analysis thoroughness presets

## Requirements

- **Python**: 3.8 or higher (MediaPipe requires Python < 3.13)
- **Supported Browsers** (for cookie extraction): Chrome, Firefox, Edge, Opera, Brave, Chromium
- **Optional**: NVIDIA GPU with CUDA for YOLO acceleration

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ultimaisolutions/tiktok-scraper-analyzer.git
cd tiktok-scraper-analyzer
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Playwright Browsers

```bash
playwright install
```

### 5. Optional: Install MediaPipe (Python < 3.13 only)

```bash
pip install mediapipe>=0.10.0
```

## Quick Start

### 1. Add URLs to Download

Edit `urls.txt` and add TikTok video URLs (one per line):

```text
https://www.tiktok.com/@username/video/1234567890123456789
https://www.tiktok.com/@another_user/video/9876543210987654321
```

### 2. Run the Scraper

```bash
python main.py
```

Videos will be saved to `videos/{username}/{date}/{video_id}.mp4` with accompanying `.json` metadata files.

## Usage

### TikTok Studio Mode (Recommended)

Automates extraction of analytics data directly from TikTok Studio web interface.

```bash
# Full pipeline: screenshots + download + analyze
python main.py --studio

# Only capture screenshots (no download)
python main.py --studio --skip-download

# Download but skip analysis
python main.py --studio --skip-analysis

# Use Firefox for automation
python main.py --studio --studio-browser firefox

# Custom output directory
python main.py --studio -o my_data/
```

**What Studio Mode Does:**
1. Opens browser and navigates to TikTok Studio
2. Attempts cookie-based login, falls back to manual login if needed
3. For each video in Studio:
   - Captures screenshots of all 3 analytics tabs
   - Extracts video URL from thumbnail
4. Saves URLs to log file: `studio_urls_{timestamp}.txt`
5. Downloads videos using existing scraper
6. Analyzes with `extreme` preset + 50% frame sampling

### Standard Mode (URL-based)

```bash
# Run with defaults (reads urls.txt, outputs to videos/)
python main.py

# Custom input/output
python main.py -i my_urls.txt -o downloads/

# Use specific browser for cookies
python main.py -b firefox

# Skip browser authentication (public videos only)
python main.py --no-browser

# Full help
python main.py --help
```

### Download + Analyze

```bash
# Download and analyze videos
python main.py --analyze

# Only analyze existing videos (skip downloading)
python main.py --analyze-only
```

### Analysis Options

```bash
# Analysis with thoroughness presets
python main.py --analyze --thoroughness quick      # Fast local testing
python main.py --analyze --thoroughness balanced   # Default
python main.py --analyze --thoroughness thorough   # Better accuracy
python main.py --analyze --thoroughness maximum    # High quality with YOLO
python main.py --analyze --thoroughness extreme    # Max GPU usage, all features

# Percentage-based frame sampling
python main.py --analyze-only --sample-percent 70  # Sample 70% of frames

# Custom analysis configuration
python main.py --analyze --sample-frames 100 --color-clusters 12 --workers 8

# Enable specific GPU features
python main.py --analyze --scene-detection --full-resolution
```

### All CLI Options

| Option | Description |
|--------|-------------|
| `-i, --input` | Input file with TikTok URLs (default: `urls.txt`) |
| `-o, --output` | Output directory for videos (default: `videos/`) |
| `-b, --browser` | Browser for cookies: chrome, firefox, edge, opera, brave, chromium |
| `--no-browser` | Skip browser authentication (public videos only) |
| `--analyze` | Analyze videos after downloading |
| `--analyze-only` | Only analyze existing videos (skip downloading) |
| `--thoroughness` | Analysis preset: quick, balanced, thorough, maximum, extreme |
| `--sample-frames` | Number of frames to sample per video |
| `--sample-percent` | Percentage of frames to sample (overrides --sample-frames) |
| `--color-clusters` | Number of color clusters for palette extraction |
| `--workers` | Number of parallel workers for analysis |
| `--scene-detection` | Enable scene/cut detection |
| `--full-resolution` | Analyze at full resolution (no downsampling) |
| `--studio` | Enable TikTok Studio scraping mode |
| `--studio-browser` | Browser for Studio: chromium, firefox, webkit |
| `--skip-download` | Only capture screenshots (Studio mode) |
| `--skip-analysis` | Skip video analysis (Studio mode) |

## Analysis Features

| Feature | Method | Output |
|---------|--------|--------|
| Brightness | Grayscale mean | mean, std, min, max |
| Contrast | Grayscale std dev | mean, std |
| Sharpness | Laplacian variance | mean, std |
| Face detection | MediaPipe/Haar | detected, count, avg |
| Person detection | MediaPipe/YOLO/Haar | detected, count, avg |
| Text overlay | Contour analysis | detected, frequency |
| Motion level | Frame differencing | score (0-100), level |
| Color palette | K-means clustering | dominant colors, temperature |
| Scene detection | Histogram comparison | scene count, cuts/min, durations |
| Audio | moviepy RMS | volume dB, speech detection |

## Thoroughness Presets

Optimized for modern GPUs (RTX 4060Ti or better):

| Preset | Frames | Color K | Motion Res | YOLO | Scene Detect | Full Res | Use Case |
|--------|--------|---------|------------|------|--------------|----------|----------|
| `quick` | 15 | 4 | 120 | No | No | No | Fast testing |
| `balanced` | 30 | 6 | 240 | No | No | No | Default |
| `thorough` | 50 | 8 | 360 | No | No | No | Better accuracy |
| `maximum` | 80 | 12 | 640 | Yes | No | No | High quality |
| `extreme` | 150 | 16 | 720 | Yes | Yes | Yes | Max GPU usage |

**Frame coverage for 3-min video at 30fps (~5400 frames):**
- `quick`: ~0.3% coverage
- `balanced`: ~0.6% coverage
- `thorough`: ~1% coverage
- `maximum`: ~1.5% coverage
- `extreme`: ~2.8% coverage

## Output Structure

```
videos/
├── studio_urls_{timestamp}.txt              # Log of extracted URLs (Studio mode)
└── {username}/
    └── {YYYY-MM-DD}/
        ├── {video_id}.mp4                   # Video file
        ├── {video_id}.json                  # Metadata + analysis
        ├── {video_id}_overview.png          # Studio Overview tab (Studio mode)
        ├── {video_id}_viewers.png           # Studio Viewers tab (Studio mode)
        └── {video_id}_engagement.png        # Studio Engagement tab (Studio mode)
```

### Metadata JSON Schema

```json
{
  "video_id": "1234567890123456789",
  "username": "creator_username",
  "description": "Video description...",
  "create_time": "2024-01-15T10:30:00Z",
  "statistics": {
    "likes": 15000,
    "comments": 500,
    "shares": 200,
    "views": 100000
  },
  "analysis": {
    "version": "1.1.0",
    "settings": {
      "thoroughness": "extreme",
      "sample_frames": 150,
      "scene_detection": true
    },
    "video_quality": {
      "resolution": { "width": 1080, "height": 1920 },
      "fps": 30,
      "duration_seconds": 21,
      "frames_analyzed": 150
    },
    "visual_metrics": {
      "brightness": { "mean": 128.5, "std": 45.2, "min": 20, "max": 240 },
      "contrast": { "mean": 55.3, "std": 12.1 },
      "sharpness": { "mean": 150.2, "std": 80.5 }
    },
    "content_detection": {
      "face_detected": true,
      "face_count_avg": 1.2,
      "person_detected": true,
      "person_count_avg": 1.5,
      "text_overlay_detected": true,
      "text_overlay_frequency": 0.65
    },
    "motion_analysis": {
      "motion_score": 56.5,
      "motion_level": "high"
    },
    "color_analysis": {
      "dominant_colors": ["#FF5733", "#33FF57", "#3357FF"],
      "color_temperature": "warm"
    },
    "scene_analysis": {
      "scene_count": 5,
      "cuts_per_minute": 12.3,
      "avg_scene_duration": 4.2,
      "scene_durations": [3.5, 4.2, 5.1, 3.8, 4.4]
    },
    "audio_metrics": {
      "has_audio": true,
      "avg_volume_db": -16.6,
      "speech_detected": true
    }
  }
}
```

## Architecture

```
main.py (CLI & orchestration)
    │
    ├── TikTokScraper (scraper.py)
    │   ├── Browser cookie initialization via browser-cookie3
    │   ├── Video download via pyktok library
    │   └── Metadata extraction & file organization
    │
    ├── VideoAnalyzer (analyzer.py)
    │   ├── Frame extraction & sampling
    │   ├── Visual analysis (brightness, contrast, sharpness)
    │   ├── Content detection (faces, persons, text overlay)
    │   ├── Motion analysis
    │   └── Audio analysis
    │
    └── AnalysisModels (analysis_models.py)
        ├── MediaPipe face/pose detection (when available)
        ├── OpenCV Haar cascade fallback
        └── Optional YOLO for maximum preset
```

## Troubleshooting

### Cookie Extraction Issues

**Error:** "Unable to get key for cookie decryption"

**Solution:**
- Close the browser completely before running the scraper
- Use `--no-browser` flag for public videos only
- Try a different browser with `-b` option

### MediaPipe Not Available

**Cause:** MediaPipe requires Python < 3.13

**Solution:** The tool automatically falls back to Haar cascades for face/person detection on Python 3.13+

### YOLO Model Download

**Note:** When using `maximum` or `extreme` presets, YOLO model weights (~6MB) are automatically downloaded on first run.

### Permission Errors

**Windows:** Run terminal as Administrator if encountering permission issues with cookie extraction.

## Dependencies

### Core
- `pyktok` - TikTok video downloading
- `playwright` - Browser automation
- `browser-cookie3` - Browser cookie extraction
- `beautifulsoup4` - HTML parsing
- `requests` - HTTP requests
- `pandas` - Data handling
- `numpy` - Numerical operations

### Video Analysis
- `opencv-python-headless` (>=4.8.0) - Video/image processing
- `moviepy` (>=1.0.3) - Audio extraction
- `scikit-image` (>=0.21.0) - Image analysis

### GPU Acceleration
- `ultralytics` (>=8.0.0) - YOLO for maximum/extreme presets

### Optional
- `mediapipe` (>=0.10.0) - Face/pose detection (Python < 3.13)

## Disclaimers

### Educational and Research Use

This tool is provided for **educational and research purposes only**. Users are responsible for ensuring their use of this tool complies with all applicable laws and regulations.

### Terms of Service

Users must comply with TikTok's Terms of Service when using this tool. This tool is not affiliated with, endorsed by, or sponsored by TikTok or ByteDance.

### Copyright and Content Rights

- Downloaded videos remain the intellectual property of their original creators
- Users must respect copyright laws and creators' rights
- Do not redistribute downloaded content without permission from the original creator
- Consider using this tool only for content you have rights to or for legitimate research purposes

### No Warranty

This software is provided "as is", without warranty of any kind, express or implied. The authors are not responsible for any damages or legal issues arising from the use of this tool.

### Rate Limiting

Be respectful of TikTok's servers. Avoid excessive requests that could be considered abuse or result in IP blocking.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
