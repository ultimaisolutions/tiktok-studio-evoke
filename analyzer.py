"""
Video Analysis Module for TikTok Scraper.
Analyzes downloaded videos for visual and audio features.
Optimized for batch processing with multiprocessing.
"""

import json
import logging
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Tuple

import cv2
import numpy as np

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class AnalysisConfig:
    """Configuration for video analysis."""

    # Thoroughness preset name
    thoroughness: str = "balanced"

    # Frame sampling - absolute count
    sample_frames: int = 10

    # Frame sampling - percentage (0.0-1.0, overrides sample_frames when set)
    sample_percentage: Optional[float] = None

    # Color analysis
    color_clusters: int = 5

    # Motion analysis resolution (width)
    motion_resolution: int = 160

    # Face detection model: "short" (fast) or "full" (accurate)
    face_model: str = "short"

    # Use YOLO for person detection (requires ultralytics)
    use_yolo: bool = False

    # Audio analysis
    enable_audio: bool = True

    # Scene detection - find cuts/transitions
    scene_detection: bool = False

    # Full resolution analysis (no downsampling for visual metrics)
    full_resolution: bool = False

    # Parallel processing
    workers: Optional[int] = None  # None = auto (CPU - 1)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "thoroughness": self.thoroughness,
            "sample_frames": self.sample_frames,
            "color_clusters": self.color_clusters,
            "motion_resolution": self.motion_resolution,
            "face_model": self.face_model,
            "scene_detection": self.scene_detection,
            "full_resolution": self.full_resolution,
        }
        if self.sample_percentage is not None:
            result["sample_percentage"] = self.sample_percentage
        return result


# Preset configurations
# Frame counts optimized for modern GPUs (RTX 4060Ti or better)
THOROUGHNESS_PRESETS: Dict[str, AnalysisConfig] = {
    "quick": AnalysisConfig(
        thoroughness="quick",
        sample_frames=15,      # ~0.3% of 3-min video at 30fps
        color_clusters=4,
        motion_resolution=120,
        face_model="short",
        use_yolo=False,
        enable_audio=True,
        scene_detection=False,
        full_resolution=False,
    ),
    "balanced": AnalysisConfig(
        thoroughness="balanced",
        sample_frames=30,      # ~0.6% of 3-min video at 30fps
        color_clusters=6,
        motion_resolution=240,
        face_model="short",
        use_yolo=False,
        enable_audio=True,
        scene_detection=False,
        full_resolution=False,
    ),
    "thorough": AnalysisConfig(
        thoroughness="thorough",
        sample_frames=50,      # ~1% of 3-min video at 30fps
        color_clusters=8,
        motion_resolution=360,
        face_model="full",
        use_yolo=False,
        enable_audio=True,
        scene_detection=False,
        full_resolution=False,
    ),
    "maximum": AnalysisConfig(
        thoroughness="maximum",
        sample_frames=80,      # ~1.5% of 3-min video at 30fps
        color_clusters=12,
        motion_resolution=640,
        face_model="full",
        use_yolo=True,
        enable_audio=True,
        scene_detection=False,
        full_resolution=False,
    ),
    "extreme": AnalysisConfig(
        thoroughness="extreme",
        sample_frames=150,     # ~2.8% of 3-min video - every 1.2s
        color_clusters=16,     # Rich color palette
        motion_resolution=720, # Near full-res motion analysis
        face_model="full",
        use_yolo=True,         # GPU accelerated person detection
        enable_audio=True,
        scene_detection=True,  # Find cuts/transitions
        full_resolution=True,  # No downsampling for metrics
    ),
}


def get_config(preset: str = "balanced", **overrides) -> AnalysisConfig:
    """
    Get analysis configuration from preset with optional overrides.

    Args:
        preset: Preset name (quick, balanced, thorough, maximum, extreme)
        **overrides: Individual parameter overrides

    Returns:
        AnalysisConfig with preset values and overrides applied
    """
    if preset not in THOROUGHNESS_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Choose from: {list(THOROUGHNESS_PRESETS.keys())}")

    # Start with preset
    base = THOROUGHNESS_PRESETS[preset]
    config = AnalysisConfig(
        thoroughness=base.thoroughness,
        sample_frames=base.sample_frames,
        color_clusters=base.color_clusters,
        motion_resolution=base.motion_resolution,
        face_model=base.face_model,
        use_yolo=base.use_yolo,
        enable_audio=base.enable_audio,
        scene_detection=base.scene_detection,
        full_resolution=base.full_resolution,
    )

    # Apply overrides (with extended limits for extreme mode)
    if "sample_frames" in overrides and overrides["sample_frames"] is not None:
        config.sample_frames = max(1, min(300, overrides["sample_frames"]))  # Extended to 300
    if "color_clusters" in overrides and overrides["color_clusters"] is not None:
        config.color_clusters = max(3, min(20, overrides["color_clusters"]))  # Extended to 20
    if "motion_resolution" in overrides and overrides["motion_resolution"] is not None:
        config.motion_resolution = max(80, min(1080, overrides["motion_resolution"]))  # Extended to 1080
    if "face_model" in overrides and overrides["face_model"] is not None:
        config.face_model = overrides["face_model"]
    if "enable_audio" in overrides and overrides["enable_audio"] is not None:
        config.enable_audio = overrides["enable_audio"]
    if "workers" in overrides and overrides["workers"] is not None:
        config.workers = overrides["workers"]
    if "use_yolo" in overrides and overrides["use_yolo"] is not None:
        config.use_yolo = overrides["use_yolo"]
    if "scene_detection" in overrides and overrides["scene_detection"] is not None:
        config.scene_detection = overrides["scene_detection"]
    if "full_resolution" in overrides and overrides["full_resolution"] is not None:
        config.full_resolution = overrides["full_resolution"]
    if "sample_percentage" in overrides and overrides["sample_percentage"] is not None:
        # Clamp to 1-100 range, then convert to 0.0-1.0
        pct = max(1, min(100, overrides["sample_percentage"]))
        config.sample_percentage = pct / 100.0

    return config


# =============================================================================
# Analysis Result Data Classes
# =============================================================================


@dataclass
class VideoAnalysisResult:
    """Complete analysis results for a video."""

    # File info
    video_path: str
    analyzed_at: str
    version: str = "1.1.0"  # Bumped for scene detection support

    # Settings used
    settings: Dict[str, Any] = field(default_factory=dict)

    # Video quality metrics
    video_quality: Dict[str, Any] = field(default_factory=dict)

    # Visual metrics
    visual_metrics: Dict[str, Any] = field(default_factory=dict)

    # Content detection
    content_detection: Dict[str, Any] = field(default_factory=dict)

    # Motion analysis
    motion_analysis: Dict[str, Any] = field(default_factory=dict)

    # Color analysis
    color_analysis: Dict[str, Any] = field(default_factory=dict)

    # Scene analysis (when enabled)
    scene_analysis: Dict[str, Any] = field(default_factory=dict)

    # Audio metrics
    audio_metrics: Dict[str, Any] = field(default_factory=dict)

    # Processing info
    processing_time_ms: float = 0.0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "version": self.version,
            "analyzed_at": self.analyzed_at,
            "processing_time_ms": self.processing_time_ms,
            "settings": self.settings,
            "video_quality": self.video_quality,
            "visual_metrics": self.visual_metrics,
            "content_detection": self.content_detection,
            "motion_analysis": self.motion_analysis,
            "color_analysis": self.color_analysis,
            "audio_metrics": self.audio_metrics,
            "errors": self.errors if self.errors else [],
        }
        # Only include scene_analysis if it has data
        if self.scene_analysis:
            result["scene_analysis"] = self.scene_analysis
        return result


# =============================================================================
# Core Analysis Functions
# =============================================================================


def _extract_frames(
    video_path: str,
    num_frames: int,
    sample_percentage: Optional[float] = None
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Extract evenly-spaced frames from video.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract (used if sample_percentage is None)
        sample_percentage: Percentage of frames to sample (0.0-1.0, overrides num_frames)

    Returns:
        Tuple of (frames list, video info dict)
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    video_info = {
        "resolution": {"width": width, "height": height},
        "fps": round(fps, 2),
        "duration_seconds": round(duration, 2),
        "total_frames": total_frames,
    }

    # Calculate actual frame count (percentage overrides absolute count)
    if sample_percentage is not None:
        num_frames = int(total_frames * sample_percentage)
        # Ensure at least 5 frames, at most total_frames
        num_frames = max(5, min(num_frames, total_frames))

    # Calculate frame indices to sample
    if total_frames <= num_frames:
        indices = list(range(total_frames))
    else:
        # Always include first and last frame
        indices = [0, total_frames - 1]

        # Fill remaining samples evenly
        remaining = num_frames - 2
        if remaining > 0:
            step = (total_frames - 2) / (remaining + 1)
            for i in range(1, remaining + 1):
                idx = int(i * step)
                if idx not in indices:
                    indices.append(idx)

        indices = sorted(set(indices))[:num_frames]

    # Extract frames
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()

    video_info["frames_analyzed"] = len(frames)
    return frames, video_info


def _analyze_brightness_contrast(frames: List[np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    Analyze brightness and contrast across frames.

    Args:
        frames: List of BGR frames

    Returns:
        Dict with brightness and contrast statistics
    """
    brightness_values = []
    contrast_values = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_values.append(float(np.mean(gray)))
        contrast_values.append(float(np.std(gray)))

    return {
        "brightness": {
            "mean": round(np.mean(brightness_values), 2),
            "std": round(np.std(brightness_values), 2),
            "min": round(np.min(brightness_values), 2),
            "max": round(np.max(brightness_values), 2),
        },
        "contrast": {
            "mean": round(np.mean(contrast_values), 2),
            "std": round(np.std(contrast_values), 2),
        },
    }


def _analyze_sharpness(frames: List[np.ndarray]) -> Dict[str, float]:
    """
    Analyze sharpness using Laplacian variance.

    Args:
        frames: List of BGR frames

    Returns:
        Dict with sharpness statistics
    """
    sharpness_values = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_values.append(float(laplacian_var))

    return {
        "mean": round(np.mean(sharpness_values), 2),
        "std": round(np.std(sharpness_values), 2),
    }


def _extract_dominant_colors(frames: List[np.ndarray], k: int = 5) -> List[Dict]:
    """
    Extract dominant colors using k-means clustering.

    Args:
        frames: List of BGR frames
        k: Number of color clusters

    Returns:
        List of dominant colors with RGB and frequency
    """
    # Collect pixels from all frames (downsampled)
    all_pixels = []

    for frame in frames:
        # Downsample for speed (4x reduction)
        small = cv2.resize(frame, (frame.shape[1] // 4, frame.shape[0] // 4))
        pixels = small.reshape(-1, 3)
        # Sample subset if too many pixels
        if len(pixels) > 1000:
            indices = np.random.choice(len(pixels), 1000, replace=False)
            pixels = pixels[indices]
        all_pixels.append(pixels)

    all_pixels = np.vstack(all_pixels).astype(np.float32)

    # K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(all_pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

    # Count frequency of each cluster
    _, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    # Sort by frequency
    sorted_indices = np.argsort(-counts)

    colors = []
    for idx in sorted_indices:
        # BGR to RGB
        bgr = centers[idx].astype(int).tolist()
        rgb = [bgr[2], bgr[1], bgr[0]]
        frequency = counts[idx] / total
        colors.append({
            "rgb": rgb,
            "frequency": round(frequency, 3)
        })

    return colors


def _analyze_color_temperature(dominant_colors: List[Dict]) -> str:
    """
    Determine color temperature (warm/neutral/cool).

    Args:
        dominant_colors: List of dominant colors with frequencies

    Returns:
        Color temperature string
    """
    if not dominant_colors:
        return "neutral"

    # Weight by frequency
    total_r, total_b, total_weight = 0, 0, 0
    for color in dominant_colors:
        r, g, b = color["rgb"]
        freq = color["frequency"]
        total_r += r * freq
        total_b += b * freq
        total_weight += freq

    if total_weight == 0:
        return "neutral"

    avg_r = total_r / total_weight
    avg_b = total_b / total_weight

    warmth = avg_r - avg_b
    if warmth > 20:
        return "warm"
    elif warmth < -20:
        return "cool"
    return "neutral"


def _analyze_saturation(frames: List[np.ndarray]) -> str:
    """
    Analyze overall saturation level.

    Args:
        frames: List of BGR frames

    Returns:
        Saturation level string (low/medium/high)
    """
    saturation_values = []

    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation_values.append(float(np.mean(hsv[:, :, 1])))

    avg_saturation = np.mean(saturation_values)

    if avg_saturation < 50:
        return "low"
    elif avg_saturation < 120:
        return "medium"
    return "high"


def _detect_text_overlay(frames: List[np.ndarray]) -> Tuple[bool, float]:
    """
    Detect text overlay presence.

    Args:
        frames: List of BGR frames

    Returns:
        Tuple of (has_text, frequency)
    """
    text_count = 0

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Count text-like contours
        text_like = 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / max(h, 1)
            area = w * h

            # Heuristic for text characters
            if 10 < area < 5000 and 0.1 < aspect_ratio < 10:
                text_like += 1

        if text_like > 20:
            text_count += 1

    frequency = text_count / len(frames) if frames else 0
    return text_count > 0, round(frequency, 3)


def _calculate_motion_score(frames: List[np.ndarray], resolution: int = 160) -> Tuple[float, str]:
    """
    Calculate motion/activity level between frames.

    Args:
        frames: List of BGR frames
        resolution: Width to resize frames for analysis

    Returns:
        Tuple of (motion_score, motion_level)
    """
    if len(frames) < 2:
        return 0.0, "low"

    motion_scores = []
    aspect_ratio = frames[0].shape[0] / frames[0].shape[1]
    height = int(resolution * aspect_ratio)

    for i in range(1, len(frames)):
        # Convert and resize
        prev_gray = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        prev_small = cv2.resize(prev_gray, (resolution, height))
        curr_small = cv2.resize(curr_gray, (resolution, height))

        # Frame difference
        diff = cv2.absdiff(prev_small, curr_small)
        motion_scores.append(float(np.mean(diff)))

    avg_motion = np.mean(motion_scores)

    # Normalize to 0-100 scale
    motion_score = min(100, avg_motion * 2)

    if motion_score < 15:
        level = "low"
    elif motion_score < 40:
        level = "medium"
    else:
        level = "high"

    return round(motion_score, 2), level


def _detect_scenes(video_path: str, threshold: float = 30.0) -> Dict[str, Any]:
    """
    Detect scene changes/cuts in video using histogram comparison.
    This is GPU-friendly as it processes frames sequentially with minimal memory.

    Args:
        video_path: Path to video file
        threshold: Histogram difference threshold for scene change (lower = more sensitive)

    Returns:
        Scene analysis dict with scene count, timestamps, and avg scene duration
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {"error": "Cannot open video"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    scene_changes = []
    prev_hist = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to HSV and calculate histogram
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        if prev_hist is not None:
            # Compare histograms using correlation
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            # Lower correlation means bigger difference
            if diff < 0.5:  # Scene change detected
                timestamp = frame_idx / fps if fps > 0 else 0
                scene_changes.append({
                    "frame": frame_idx,
                    "timestamp": round(timestamp, 2),
                    "confidence": round(1 - diff, 3)
                })

        prev_hist = hist.copy()
        frame_idx += 1

    cap.release()

    # Calculate scene statistics
    num_scenes = len(scene_changes) + 1  # N changes = N+1 scenes
    avg_scene_duration = duration / num_scenes if num_scenes > 0 else duration

    # Calculate scene durations
    scene_durations = []
    prev_timestamp = 0
    for change in scene_changes:
        scene_durations.append(round(change["timestamp"] - prev_timestamp, 2))
        prev_timestamp = change["timestamp"]
    # Add final scene
    scene_durations.append(round(duration - prev_timestamp, 2))

    return {
        "scene_count": num_scenes,
        "scene_changes": len(scene_changes),
        "avg_scene_duration": round(avg_scene_duration, 2),
        "min_scene_duration": min(scene_durations) if scene_durations else 0,
        "max_scene_duration": max(scene_durations) if scene_durations else duration,
        "cuts_per_minute": round(len(scene_changes) / (duration / 60), 2) if duration > 0 else 0,
        "scene_timestamps": [c["timestamp"] for c in scene_changes[:20]],  # First 20 cuts
    }


def _detect_faces_and_persons(
    frames: List[np.ndarray],
    face_model: str = "short",
    use_yolo: bool = False
) -> Dict[str, Any]:
    """
    Detect faces and persons across frames.

    Args:
        frames: List of BGR frames
        face_model: "short" or "full" for MediaPipe
        use_yolo: Use YOLO for person detection

    Returns:
        Detection results dict
    """
    from analysis_models import get_models

    models = get_models()

    face_counts = []
    person_counts = []

    for frame in frames:
        face_counts.append(models.detect_faces(frame, face_model))
        person_counts.append(models.detect_persons(frame, use_yolo))

    return {
        "face_detected": max(face_counts) > 0,
        "max_face_count": max(face_counts),
        "avg_face_count": round(np.mean(face_counts), 2),
        "person_detected": max(person_counts) > 0,
        "max_person_count": max(person_counts),
        "avg_person_count": round(np.mean(person_counts), 2),
    }


def _analyze_audio(video_path: str) -> Dict[str, Any]:
    """
    Extract and analyze audio from video.

    Args:
        video_path: Path to video file

    Returns:
        Audio metrics dict
    """
    try:
        # Try new moviepy import first, fall back to legacy
        try:
            from moviepy import VideoFileClip
        except ImportError:
            from moviepy.editor import VideoFileClip

        clip = VideoFileClip(video_path)

        if clip.audio is None:
            clip.close()
            return {
                "has_audio": False,
                "audio_duration_seconds": None,
                "avg_volume_db": None,
                "max_volume_db": None,
                "volume_variance": None,
                "has_speech": None,
            }

        # Extract audio as numpy array
        audio_fps = 22050  # Standard for analysis
        audio_array = clip.audio.to_soundarray(fps=audio_fps)

        # Convert to mono if stereo
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)

        # Calculate RMS volume
        rms = np.sqrt(np.mean(audio_array ** 2))
        max_amplitude = np.max(np.abs(audio_array))

        # Convert to dB
        avg_db = 20 * np.log10(max(rms, 1e-10))
        max_db = 20 * np.log10(max(max_amplitude, 1e-10))

        # Volume variance (dynamic range)
        chunk_size = audio_fps // 10  # 100ms chunks
        chunks = [audio_array[i:i + chunk_size] for i in range(0, len(audio_array), chunk_size)]
        chunk_rms = [np.sqrt(np.mean(c ** 2)) for c in chunks if len(c) > 0]
        volume_variance = np.var(chunk_rms) if chunk_rms else 0

        # Simple speech detection heuristic
        has_speech = avg_db > -30 and volume_variance > 0.001

        duration = clip.duration
        clip.close()

        return {
            "has_audio": True,
            "audio_duration_seconds": round(duration, 2),
            "avg_volume_db": round(avg_db, 1),
            "max_volume_db": round(max_db, 1),
            "volume_variance": round(volume_variance, 6),
            "has_speech": has_speech,
        }

    except Exception as e:
        return {
            "has_audio": False,
            "audio_duration_seconds": None,
            "avg_volume_db": None,
            "max_volume_db": None,
            "volume_variance": None,
            "has_speech": None,
            "audio_error": str(e),
        }


# =============================================================================
# Main Analyzer Class
# =============================================================================


class VideoAnalyzer:
    """
    Analyzes TikTok videos for visual and audio features.
    Designed for batch processing with multiprocessing.
    """

    ANALYSIS_VERSION = "1.0.0"

    def __init__(self, config: Optional[AnalysisConfig] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the analyzer.

        Args:
            config: Analysis configuration (default: balanced preset)
            logger: Logger instance
        """
        self.config = config or get_config("balanced")
        self.logger = logger or logging.getLogger(__name__)

    def analyze_video(self, video_path: str) -> VideoAnalysisResult:
        """
        Analyze a single video file.

        Args:
            video_path: Path to the video file

        Returns:
            VideoAnalysisResult with all metrics
        """
        start_time = time.time()
        errors = []

        result = VideoAnalysisResult(
            video_path=video_path,
            analyzed_at=datetime.utcnow().isoformat() + "Z",
            version=self.ANALYSIS_VERSION,
            settings=self.config.to_dict(),
        )

        # Extract frames
        try:
            frames, video_info = _extract_frames(
                video_path,
                self.config.sample_frames,
                self.config.sample_percentage
            )
            result.video_quality = video_info
        except Exception as e:
            errors.append(f"Frame extraction failed: {e}")
            result.errors = errors
            result.processing_time_ms = round((time.time() - start_time) * 1000, 2)
            return result

        if not frames:
            errors.append("No frames extracted")
            result.errors = errors
            result.processing_time_ms = round((time.time() - start_time) * 1000, 2)
            return result

        # Brightness and contrast
        try:
            bc_results = _analyze_brightness_contrast(frames)
            result.visual_metrics["brightness"] = bc_results["brightness"]
            result.visual_metrics["contrast"] = bc_results["contrast"]
        except Exception as e:
            errors.append(f"Brightness/contrast analysis failed: {e}")

        # Sharpness
        try:
            result.visual_metrics["sharpness"] = _analyze_sharpness(frames)
        except Exception as e:
            errors.append(f"Sharpness analysis failed: {e}")

        # Color analysis
        try:
            dominant_colors = _extract_dominant_colors(frames, self.config.color_clusters)
            result.color_analysis["dominant_colors"] = dominant_colors
            result.color_analysis["color_temperature"] = _analyze_color_temperature(dominant_colors)
            result.color_analysis["saturation_level"] = _analyze_saturation(frames)
        except Exception as e:
            errors.append(f"Color analysis failed: {e}")

        # Text overlay detection
        try:
            has_text, text_freq = _detect_text_overlay(frames)
            result.content_detection["text_overlay_detected"] = has_text
            result.content_detection["text_overlay_frequency"] = text_freq
        except Exception as e:
            errors.append(f"Text detection failed: {e}")

        # Motion analysis
        try:
            motion_score, motion_level = _calculate_motion_score(
                frames, self.config.motion_resolution
            )
            result.motion_analysis["motion_score"] = motion_score
            result.motion_analysis["motion_level"] = motion_level
        except Exception as e:
            errors.append(f"Motion analysis failed: {e}")

        # Face and person detection
        try:
            detection_results = _detect_faces_and_persons(
                frames, self.config.face_model, self.config.use_yolo
            )
            result.content_detection.update(detection_results)
        except Exception as e:
            errors.append(f"Face/person detection failed: {e}")

        # Audio analysis
        if self.config.enable_audio:
            try:
                result.audio_metrics = _analyze_audio(video_path)
            except Exception as e:
                errors.append(f"Audio analysis failed: {e}")
        else:
            result.audio_metrics = {"has_audio": None, "skipped": True}

        # Scene detection (when enabled - GPU intensive)
        if self.config.scene_detection:
            try:
                result.scene_analysis = _detect_scenes(video_path)
            except Exception as e:
                errors.append(f"Scene detection failed: {e}")

        # Add frames analyzed count to video quality
        result.video_quality["frames_analyzed"] = len(frames)

        result.errors = errors
        result.processing_time_ms = round((time.time() - start_time) * 1000, 2)

        return result

    def update_metadata_file(self, json_path: str, analysis: VideoAnalysisResult) -> bool:
        """
        Add analysis results to existing metadata JSON file.

        Args:
            json_path: Path to the {video_id}.json file
            analysis: Analysis results to add

        Returns:
            True if successful, False otherwise
        """
        try:
            # Read existing metadata
            with open(json_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            # Add analysis results (convert numpy types to Python types)
            metadata["analysis"] = _convert_numpy_types(analysis.to_dict())

            # Write back
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            self.logger.error(f"Failed to update {json_path}: {e}")
            return False


    def analyze_batch(
        self,
        video_paths: List[str],
        workers: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, VideoAnalysisResult]:
        """
        Analyze multiple videos in parallel using multiprocessing.

        Args:
            video_paths: List of video file paths
            workers: Number of worker processes (default: auto)
            progress_callback: Optional callback(completed, total) for progress

        Returns:
            Dict mapping video_path -> VideoAnalysisResult
        """
        if not video_paths:
            return {}

        # Determine worker count
        if workers is None:
            workers = self.config.workers
        if workers is None:
            workers = max(1, (os.cpu_count() or 4) - 1)

        # Cap workers at video count and reasonable maximum
        workers = min(workers, len(video_paths), 12)

        results = {}
        completed = 0

        # Prepare arguments for workers
        config_dict = {
            "sample_frames": self.config.sample_frames,
            "sample_percentage": self.config.sample_percentage,
            "color_clusters": self.config.color_clusters,
            "motion_resolution": self.config.motion_resolution,
            "face_model": self.config.face_model,
            "use_yolo": self.config.use_yolo,
            "enable_audio": self.config.enable_audio,
            "thoroughness": self.config.thoroughness,
        }

        args_list = [(path, config_dict) for path in video_paths]

        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_initializer,
            initargs=(config_dict,)
        ) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(_analyze_single_video, args): args[0]
                for args in args_list
            }

            # Collect results as they complete
            for future in as_completed(future_to_path):
                video_path = future_to_path[future]
                try:
                    result = future.result()
                    results[video_path] = result
                except Exception as e:
                    # Create error result
                    results[video_path] = VideoAnalysisResult(
                        video_path=video_path,
                        analyzed_at=datetime.utcnow().isoformat() + "Z",
                        errors=[f"Worker error: {e}"],
                    )

                completed += 1
                if progress_callback:
                    progress_callback(completed, len(video_paths))

        return results


# =============================================================================
# Helper Functions
# =============================================================================


def _convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


# =============================================================================
# Worker Functions for Multiprocessing
# =============================================================================

# Process-local config and models
_worker_config: Optional[Dict] = None


def _worker_initializer(config_dict: Dict):
    """
    Called once when worker process starts.
    Pre-loads models for efficiency.
    """
    global _worker_config
    _worker_config = config_dict

    # Pre-load models
    from analysis_models import warmup_models
    warmup_models(
        face_model=config_dict.get("face_model", "short"),
        use_yolo=config_dict.get("use_yolo", False)
    )


def _analyze_single_video(args: Tuple[str, Dict]) -> VideoAnalysisResult:
    """
    Worker function for parallel processing.
    Must be picklable (top-level function).

    Args:
        args: Tuple of (video_path, config_dict)

    Returns:
        VideoAnalysisResult
    """
    video_path, config_dict = args

    # Create config from dict
    config = AnalysisConfig(
        thoroughness=config_dict.get("thoroughness", "balanced"),
        sample_frames=config_dict.get("sample_frames", 10),
        sample_percentage=config_dict.get("sample_percentage"),
        color_clusters=config_dict.get("color_clusters", 5),
        motion_resolution=config_dict.get("motion_resolution", 160),
        face_model=config_dict.get("face_model", "short"),
        use_yolo=config_dict.get("use_yolo", False),
        enable_audio=config_dict.get("enable_audio", True),
    )

    # Create analyzer and process
    analyzer = VideoAnalyzer(config)
    return analyzer.analyze_video(video_path)


# =============================================================================
# Utility Functions
# =============================================================================


def find_videos_to_analyze(output_dir: str) -> List[Tuple[str, str]]:
    """
    Find all video files and their corresponding JSON files.

    Args:
        output_dir: Base output directory

    Returns:
        List of (video_path, json_path) tuples
    """
    video_files = []
    output_path = Path(output_dir)

    for mp4_file in output_path.rglob("*.mp4"):
        json_file = mp4_file.with_suffix(".json")
        if json_file.exists():
            video_files.append((str(mp4_file), str(json_file)))

    return video_files
