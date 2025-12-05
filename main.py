"""
TikTok Video Scraper - Main Entry Point

Downloads TikTok videos from URLs in a text file,
extracts metadata, and organizes them by username and date.
Optionally analyzes videos for visual and audio features.

NEW: TikTok Studio mode - automates extraction of analytics data:
- Screenshots of all 3 tabs (Overview, Viewers, Engagement)
- Downloads videos via thumbnail click
- Analyzes with extreme preset and 50% frame sampling

Usage:
    python main.py                          # Uses default urls.txt and videos/ folder
    python main.py -i my_urls.txt           # Custom input file
    python main.py -o downloads/            # Custom output folder
    python main.py -b firefox               # Use Firefox instead of Chrome
    python main.py --no-browser             # Skip browser auth (public videos only)
    python main.py --analyze                # Download and analyze videos
    python main.py --analyze-only           # Only analyze existing videos
    python main.py --thoroughness maximum   # Use maximum analysis quality

    # TikTok Studio mode
    python main.py --studio                 # Full pipeline: screenshots + download + analyze
    python main.py --studio --skip-download # Only capture screenshots
    python main.py --studio --skip-analysis # Download but don't analyze
"""

import argparse
import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path

from scraper import TikTokScraper
from utils import read_urls_from_file, setup_logging


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download TikTok videos from URLs and organize by user/date.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                        Download from urls.txt to videos/
  python main.py -i links.txt           Use custom input file
  python main.py -o downloads/          Use custom output directory
  python main.py -b firefox             Use Firefox browser cookies
  python main.py --analyze              Download and analyze videos
  python main.py --analyze-only         Only analyze existing videos
  python main.py --thoroughness maximum GPU instance, best quality

TikTok Studio Mode:
  python main.py --studio               Screenshots + download + analyze all videos
  python main.py --studio --skip-download  Only capture screenshots
  python main.py --studio --skip-analysis  Download but skip analysis
  python main.py --studio --studio-browser firefox  Use Firefox for Studio
        """
    )

    parser.add_argument(
        "-i", "--input",
        default="urls.txt",
        help="Input file containing TikTok URLs (default: urls.txt)"
    )

    parser.add_argument(
        "-o", "--output",
        default="videos",
        help="Output directory for downloaded videos (default: videos)"
    )

    parser.add_argument(
        "-b", "--browser",
        default="chrome",
        choices=["chrome", "firefox", "edge", "opera", "brave", "chromium"],
        help="Browser to use for cookies (default: chrome)"
    )

    parser.add_argument(
        "-l", "--log",
        default="errors.log",
        help="Error log file path (default: errors.log)"
    )

    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Skip browser cookie initialization (for public videos only)"
    )

    # Analysis options
    analysis_group = parser.add_argument_group("Analysis Options")

    analysis_group.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze downloaded videos after scraping completes"
    )

    analysis_group.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip downloading, only analyze existing videos in output directory"
    )

    analysis_group.add_argument(
        "--thoroughness",
        choices=["quick", "balanced", "thorough", "maximum", "extreme"],
        default="balanced",
        help="Analysis thoroughness preset (default: balanced). 'extreme' uses GPU heavily."
    )

    analysis_group.add_argument(
        "--sample-frames",
        type=int,
        default=None,
        help="Number of frames to sample per video (1-300, overrides preset)"
    )

    analysis_group.add_argument(
        "--sample-percent",
        type=int,
        default=None,
        help="Percentage of frames to sample (1-100, overrides --sample-frames)"
    )

    analysis_group.add_argument(
        "--color-clusters",
        type=int,
        default=None,
        help="Number of color clusters for k-means (3-20, overrides preset)"
    )

    analysis_group.add_argument(
        "--motion-res",
        type=int,
        default=None,
        help="Motion analysis resolution width in pixels (80-1080, overrides preset)"
    )

    analysis_group.add_argument(
        "--face-model",
        choices=["short", "full"],
        default=None,
        help="MediaPipe face detection model (short=fast, full=accurate)"
    )

    analysis_group.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for analysis (default: CPU count - 1)"
    )

    analysis_group.add_argument(
        "--skip-audio",
        action="store_true",
        help="Skip audio analysis (faster but less complete)"
    )

    analysis_group.add_argument(
        "--scene-detection",
        action="store_true",
        help="Enable scene/cut detection (GPU intensive, auto-enabled for extreme preset)"
    )

    analysis_group.add_argument(
        "--full-resolution",
        action="store_true",
        help="Analyze at full resolution without downsampling (more accurate, slower)"
    )

    # TikTok Studio mode options
    studio_group = parser.add_argument_group("TikTok Studio Options")

    studio_group.add_argument(
        "--studio",
        action="store_true",
        help="Enable TikTok Studio scraping mode (screenshots + download + analyze)"
    )

    studio_group.add_argument(
        "--studio-browser",
        choices=["chromium", "firefox", "webkit"],
        default="chromium",
        help="Browser for Studio automation (default: chromium)"
    )

    studio_group.add_argument(
        "--skip-download",
        action="store_true",
        help="Only capture screenshots, don't download videos (Studio mode)"
    )

    studio_group.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Download but don't analyze videos (Studio mode)"
    )

    studio_group.add_argument(
        "--cdp-port",
        type=int,
        default=None,
        help="Custom CDP port for connecting to existing browser (default: auto-scan 9222-9229)"
    )

    return parser.parse_args()


def print_banner():
    """Print application banner."""
    print("\n" + "=" * 50)
    print("  TikTok Video Scraper")
    print("=" * 50)


def print_summary(results: dict):
    """Print download summary."""
    print("\n" + "=" * 50)
    print("  DOWNLOAD SUMMARY")
    print("=" * 50)
    print(f"  Total URLs:    {results['total']}")
    print(f"  Successful:    {results['success']}")
    print(f"  Failed:        {results['failed']}")
    print("=" * 50)

    if results["failed"] > 0:
        print("\nFailed URLs:")
        for item in results["failed_urls"]:
            print(f"  - {item['url']}")
            print(f"    Error: {item['error']}")

    print()


def print_analysis_summary(summary: dict):
    """Print analysis results summary."""
    print("\n" + "=" * 50)
    print("  ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"  Analyzed:      {summary['analyzed']}")
    print(f"  Failed:        {summary['failed']}")
    print(f"  Time:          {summary['elapsed_seconds']}s")
    print(f"  Speed:         {summary['videos_per_second']} videos/sec")
    print("=" * 50 + "\n")


def print_studio_summary(results: dict):
    """Print TikTok Studio scraping summary."""
    print("\n" + "=" * 50)
    print("  TIKTOK STUDIO SUMMARY")
    print("=" * 50)
    print(f"  Total videos:  {results.get('total', 0)}")
    print(f"  Processed:     {results.get('processed', 0)}")
    print(f"  Skipped:       {results.get('skipped', 0)}")
    print(f"  Failed:        {results.get('failed', 0)}")
    print("=" * 50)

    if results.get("errors"):
        print("\nErrors:")
        for error in results["errors"][:10]:  # Show first 10 errors
            print(f"  - {error.get('video_id', 'unknown')}: {error.get('error', 'Unknown error')}")

    print()


def save_urls_log(output_dir: str, videos: list, logger) -> str:
    """
    Save extracted video URLs to a log file.

    Args:
        output_dir: Output directory path
        videos: List of video info dicts with video_url and video_id
        logger: Logger instance

    Returns:
        Path to the saved log file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"studio_urls_{timestamp}.txt"
    log_path = output_path / log_filename

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# TikTok Studio URLs - Extracted {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total: {len(videos)} videos\n\n")

        for video in videos:
            video_url = video.get("video_url", "")
            video_id = video.get("video_id", "unknown")
            if video_url:
                f.write(f"{video_url}  # video_id: {video_id}\n")

    logger.info(f"Saved {len(videos)} URLs to {log_path}")
    print(f"\n  URLs saved to: {log_path}")

    return str(log_path)


def run_studio_mode(args, logger) -> dict:
    """
    Run TikTok Studio scraping mode.

    Args:
        args: Parsed command line arguments
        logger: Logger instance

    Returns:
        Results dictionary
    """
    from studio_scraper import run_studio_scraper

    print("\n" + "=" * 50)
    print("  TIKTOK STUDIO MODE")
    print("=" * 50)
    print(f"  Output dir:    {args.output}")
    print(f"  Browser:       {args.studio_browser}")
    print(f"  Skip download: {args.skip_download}")
    print(f"  Skip analysis: {args.skip_analysis}")
    print("=" * 50)

    # Run the async studio scraper
    results = asyncio.run(
        run_studio_scraper(
            output_dir=args.output,
            logger=logger,
            browser_type=args.studio_browser,
            skip_download=args.skip_download,
            skip_analysis=args.skip_analysis,
            cdp_port=args.cdp_port,
        )
    )

    if not results.get("success"):
        logger.error(f"Studio scraper failed: {results.get('error', 'Unknown error')}")
        return results

    print_studio_summary(results)

    # Save extracted URLs to log file
    if results.get("videos"):
        save_urls_log(args.output, results["videos"], logger)

    # If not skipping download, download the videos
    if not args.skip_download and results.get("videos"):
        print("\n" + "=" * 50)
        print("  DOWNLOADING VIDEOS")
        print("=" * 50)

        # Initialize scraper for downloading
        scraper = TikTokScraper(args.output, logger)

        # Try to initialize browser cookies
        scraper.initialize_browser(args.browser, required=False)

        # Download each video
        download_results = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "successful_urls": [],
            "failed_urls": [],
        }

        for video_info in results["videos"]:
            video_url = video_info.get("video_url")
            if not video_url:
                continue

            download_results["total"] += 1
            logger.info(f"Downloading: {video_url}")

            success, message = scraper.download_video(video_url)
            if success:
                download_results["success"] += 1
                download_results["successful_urls"].append({"url": video_url, "message": message})
                logger.info(f"SUCCESS: {message}")
            else:
                download_results["failed"] += 1
                download_results["failed_urls"].append({"url": video_url, "error": message})
                logger.warning(f"FAILED: {message}")

        print_summary(download_results)
        results["download_results"] = download_results

    # If not skipping analysis, analyze the videos
    if not args.skip_download and not args.skip_analysis:
        # Override defaults for Studio mode: extreme thoroughness, 50% frame sampling
        if args.thoroughness == "balanced":  # Only override if using default
            args.thoroughness = "extreme"
        if args.sample_percent is None:
            args.sample_percent = 50

        print(f"\nRunning analysis with {args.thoroughness} preset and {args.sample_percent}% frame sampling...")
        analysis_summary = run_analysis(args, logger)
        print_analysis_summary(analysis_summary)
        results["analysis_summary"] = analysis_summary

    return results


def run_analysis(args, logger) -> dict:
    """
    Run batch video analysis.

    Args:
        args: Parsed command line arguments
        logger: Logger instance

    Returns:
        Analysis summary dict
    """
    from analyzer import VideoAnalyzer, get_config, find_videos_to_analyze

    # Find videos to analyze
    videos = find_videos_to_analyze(args.output)

    if not videos:
        logger.warning("No videos found to analyze")
        print(f"\nNo videos found in {args.output}")
        return {"analyzed": 0, "failed": 0, "elapsed_seconds": 0, "videos_per_second": 0}

    # Build configuration from args
    config = get_config(
        preset=args.thoroughness,
        sample_frames=args.sample_frames,
        sample_percentage=args.sample_percent,
        color_clusters=args.color_clusters,
        motion_resolution=args.motion_res,
        face_model=args.face_model,
        enable_audio=not args.skip_audio,
        workers=args.workers,
        scene_detection=args.scene_detection if args.scene_detection else None,
        full_resolution=args.full_resolution if args.full_resolution else None,
    )

    print(f"\n{'=' * 50}")
    print("  VIDEO ANALYSIS")
    print("=" * 50)
    print(f"  Videos found:  {len(videos)}")
    print(f"  Thoroughness:  {config.thoroughness}")
    # Show percentage or frame count
    if config.sample_percentage is not None:
        print(f"  Frame sample:  {int(config.sample_percentage * 100)}% of video")
    else:
        print(f"  Frames/video:  {config.sample_frames}")
    print(f"  Color clusters:{config.color_clusters}")
    print(f"  Motion res:    {config.motion_resolution}px")
    print(f"  Workers:       {config.workers or 'auto'}")
    print(f"  YOLO:          {'enabled' if config.use_yolo else 'disabled'}")
    print(f"  Scene detect:  {'enabled' if config.scene_detection else 'disabled'}")
    print(f"  Full res:      {'enabled' if config.full_resolution else 'disabled'}")
    print(f"  Audio:         {'enabled' if config.enable_audio else 'disabled'}")
    print("=" * 50)

    # Create analyzer
    analyzer = VideoAnalyzer(config, logger)

    start_time = time.time()

    # Progress callback
    def on_progress(completed, total):
        pct = completed * 100 // total
        bar_len = 30
        filled = int(bar_len * completed / total)
        bar = "=" * filled + "-" * (bar_len - filled)
        print(f"\r  Progress: [{bar}] {completed}/{total} ({pct}%)", end="", flush=True)

    print("\n  Starting analysis...\n")

    # Run batch analysis
    video_paths = [v[0] for v in videos]
    results = analyzer.analyze_batch(
        video_paths,
        workers=config.workers,
        progress_callback=on_progress
    )

    print()  # Newline after progress bar

    # Update JSON files with results
    success_count = 0
    fail_count = 0

    for video_path, json_path in videos:
        if video_path in results:
            result = results[video_path]
            if not result.errors or len(result.errors) == 0:
                if analyzer.update_metadata_file(json_path, result):
                    success_count += 1
                else:
                    fail_count += 1
            else:
                # Still save results even with errors
                analyzer.update_metadata_file(json_path, result)
                if result.video_quality:  # Has some valid data
                    success_count += 1
                else:
                    fail_count += 1
                    logger.error(f"Analysis errors for {video_path}: {result.errors}")
        else:
            fail_count += 1

    elapsed = time.time() - start_time

    return {
        "analyzed": success_count,
        "failed": fail_count,
        "elapsed_seconds": round(elapsed, 2),
        "videos_per_second": round(len(videos) / elapsed, 2) if elapsed > 0 else 0
    }


def main():
    """Main entry point."""
    args = parse_arguments()

    print_banner()

    # Setup logging
    logger = setup_logging(args.log)

    # Handle --analyze-only mode
    if args.analyze_only:
        print("\nAnalyze-only mode: skipping downloads")
        summary = run_analysis(args, logger)
        print_analysis_summary(summary)
        if summary["failed"] > 0:
            sys.exit(1)
        return

    # Handle --studio mode
    if args.studio:
        results = run_studio_mode(args, logger)
        if not results.get("success"):
            sys.exit(1)
        if results.get("failed", 0) > 0:
            sys.exit(1)
        return

    # Check input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"\nError: Input file not found: {args.input}")
        print("Create the file and add TikTok URLs (one per line).")
        sys.exit(1)

    # Read URLs from file
    urls = read_urls_from_file(args.input)

    if not urls:
        print(f"\nNo URLs found in {args.input}")
        print("Add TikTok URLs to the file (one per line).")
        sys.exit(1)

    print(f"\nInput file:  {args.input}")
    print(f"Output dir:  {args.output}")
    print(f"Browser:     {args.browser}")
    print(f"URLs found:  {len(urls)}")

    # Initialize scraper
    scraper = TikTokScraper(args.output, logger)

    # Initialize browser (optional - will try but continue if it fails)
    if args.no_browser:
        print("\nSkipping browser initialization (--no-browser flag)")
        print("Note: Only public videos will be accessible")
    else:
        print(f"\nAttempting to initialize {args.browser} browser cookies...")
        scraper.initialize_browser(args.browser, required=False)
        if not scraper._browser_initialized:
            print("Warning: Browser cookies unavailable - continuing without authentication")
            print("Public videos should still work. Private videos may fail.")

    # Process URLs
    print("\nStarting downloads...\n")
    results = scraper.process_urls(urls)

    # Print summary
    print_summary(results)

    # Run analysis if requested
    if args.analyze:
        summary = run_analysis(args, logger)
        print_analysis_summary(summary)

    # Exit with error code if any failures
    if results["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
