"""
Pre-loaded models for video analysis.
Uses singleton pattern to load models once per process for efficiency.

Supports multiple detection backends:
- MediaPipe (preferred, fast)
- OpenCV Haar cascades (fallback, always available)
- YOLO (optional, for maximum preset)
"""

import logging
import threading
from typing import Optional

import numpy as np

# Lazy imports for optional dependencies
_mediapipe = None
_mediapipe_available = None
_cv2 = None
_yolo = None


def _get_cv2():
    """Lazy import OpenCV."""
    global _cv2
    if _cv2 is None:
        import cv2
        _cv2 = cv2
    return _cv2


def _get_mediapipe():
    """Lazy import MediaPipe (optional)."""
    global _mediapipe, _mediapipe_available
    if _mediapipe_available is None:
        try:
            import mediapipe as mp
            _mediapipe = mp
            _mediapipe_available = True
        except ImportError:
            _mediapipe_available = False
    return _mediapipe if _mediapipe_available else None


def _get_yolo():
    """Lazy import YOLO (optional)."""
    global _yolo
    if _yolo is None:
        try:
            from ultralytics import YOLO
            _yolo = YOLO
        except ImportError:
            _yolo = False  # Mark as unavailable
    return _yolo if _yolo else None


def is_mediapipe_available() -> bool:
    """Check if MediaPipe is available."""
    _get_mediapipe()
    return _mediapipe_available


class AnalysisModels:
    """
    Singleton class for managing pre-loaded ML models.
    Models are loaded lazily on first access and reused.

    Supports fallback from MediaPipe to OpenCV Haar cascades.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._face_detector_short = None
        self._face_detector_full = None
        self._pose_detector = None
        self._yolo_model = None
        self._haar_face_cascade = None
        self._haar_body_cascade = None
        self._logger = logging.getLogger(__name__)
        self._initialized = True

    def _ensure_haar_face_cascade(self):
        """Load OpenCV Haar cascade for face detection (fallback)."""
        if self._haar_face_cascade is None:
            cv2 = _get_cv2()
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar_face_cascade = cv2.CascadeClassifier(cascade_path)
        return self._haar_face_cascade

    def _ensure_haar_body_cascade(self):
        """Load OpenCV Haar cascade for body detection (fallback)."""
        if self._haar_body_cascade is None:
            cv2 = _get_cv2()
            cascade_path = cv2.data.haarcascades + "haarcascade_fullbody.xml"
            self._haar_body_cascade = cv2.CascadeClassifier(cascade_path)
        return self._haar_body_cascade

    def _ensure_face_detector(self, model_type: str = "short"):
        """Lazy-load MediaPipe face detection."""
        mp = _get_mediapipe()
        if mp is None:
            return None  # Will use Haar cascade fallback

        if model_type == "full":
            if self._face_detector_full is None:
                self._face_detector_full = mp.solutions.face_detection.FaceDetection(
                    model_selection=1,  # 1 = full-range (more accurate, slower)
                    min_detection_confidence=0.5
                )
            return self._face_detector_full
        else:
            if self._face_detector_short is None:
                self._face_detector_short = mp.solutions.face_detection.FaceDetection(
                    model_selection=0,  # 0 = short-range (faster)
                    min_detection_confidence=0.5
                )
            return self._face_detector_short

    def _ensure_pose_detector(self):
        """Lazy-load MediaPipe pose detection."""
        mp = _get_mediapipe()
        if mp is None:
            return None  # Will use Haar cascade fallback

        if self._pose_detector is None:
            self._pose_detector = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=0,  # 0 = Lite (fastest)
                min_detection_confidence=0.5
            )
        return self._pose_detector

    def _ensure_yolo(self):
        """Lazy-load YOLO model for person detection."""
        if self._yolo_model is None:
            YOLO = _get_yolo()
            if YOLO:
                try:
                    # Use nano model for speed
                    self._yolo_model = YOLO("yolov8n.pt")
                    self._logger.debug("YOLO model loaded successfully")
                except Exception as e:
                    self._logger.warning(f"Failed to load YOLO model: {e}")
                    self._yolo_model = False  # Mark as failed
        return self._yolo_model if self._yolo_model else None

    def detect_faces_haar(self, frame: np.ndarray) -> int:
        """
        Detect faces using OpenCV Haar cascade (fallback method).

        Args:
            frame: BGR image as numpy array

        Returns:
            Number of faces detected
        """
        try:
            cv2 = _get_cv2()
            cascade = self._ensure_haar_face_cascade()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            return len(faces)
        except Exception as e:
            self._logger.debug(f"Haar face detection error: {e}")
            return 0

    def detect_faces(self, frame: np.ndarray, model_type: str = "short") -> int:
        """
        Detect faces in frame.

        Args:
            frame: BGR image as numpy array
            model_type: "short" for fast, "full" for accurate

        Returns:
            Number of faces detected
        """
        # Try MediaPipe first
        mp = _get_mediapipe()
        if mp is not None:
            try:
                cv2 = _get_cv2()
                detector = self._ensure_face_detector(model_type)

                # MediaPipe expects RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detector.process(rgb)

                if results.detections:
                    return len(results.detections)
                return 0
            except Exception as e:
                self._logger.debug(f"MediaPipe face detection error: {e}")

        # Fall back to Haar cascade
        return self.detect_faces_haar(frame)

    def detect_persons_haar(self, frame: np.ndarray) -> int:
        """
        Detect persons using OpenCV Haar cascade (fallback method).
        Uses face detection as proxy for person detection.

        Args:
            frame: BGR image as numpy array

        Returns:
            Number of persons detected (based on faces)
        """
        # Use face count as person proxy (more reliable than body cascade)
        return min(self.detect_faces_haar(frame), 1)

    def detect_persons_mediapipe(self, frame: np.ndarray) -> int:
        """
        Detect persons using MediaPipe Pose.
        Note: MediaPipe Pose detects one person at a time.

        Args:
            frame: BGR image as numpy array

        Returns:
            1 if person detected, 0 otherwise
        """
        mp = _get_mediapipe()
        if mp is None:
            return self.detect_persons_haar(frame)

        try:
            cv2 = _get_cv2()
            detector = self._ensure_pose_detector()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)

            return 1 if results.pose_landmarks else 0
        except Exception as e:
            self._logger.debug(f"Pose detection error: {e}")
            return self.detect_persons_haar(frame)

    def detect_persons_yolo(self, frame: np.ndarray) -> int:
        """
        Detect persons using YOLO (more accurate, multi-person).

        Args:
            frame: BGR image as numpy array

        Returns:
            Number of persons detected
        """
        try:
            model = self._ensure_yolo()
            if not model:
                # Fall back to MediaPipe or Haar
                return self.detect_persons_mediapipe(frame)

            # Run YOLO inference
            results = model(frame, verbose=False, classes=[0])  # class 0 = person

            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None:
                    return len(boxes)
            return 0
        except Exception as e:
            self._logger.debug(f"YOLO detection error: {e}")
            return self.detect_persons_mediapipe(frame)

    def detect_persons(self, frame: np.ndarray, use_yolo: bool = False) -> int:
        """
        Detect persons in frame.

        Args:
            frame: BGR image as numpy array
            use_yolo: Use YOLO for multi-person detection (requires ultralytics)

        Returns:
            Number of persons detected
        """
        if use_yolo:
            return self.detect_persons_yolo(frame)
        return self.detect_persons_mediapipe(frame)

    def cleanup(self):
        """Release model resources."""
        if self._face_detector_short:
            try:
                self._face_detector_short.close()
            except Exception:
                pass
            self._face_detector_short = None
        if self._face_detector_full:
            try:
                self._face_detector_full.close()
            except Exception:
                pass
            self._face_detector_full = None
        if self._pose_detector:
            try:
                self._pose_detector.close()
            except Exception:
                pass
            self._pose_detector = None
        self._yolo_model = None
        self._haar_face_cascade = None
        self._haar_body_cascade = None

    def is_yolo_available(self) -> bool:
        """Check if YOLO is available."""
        YOLO = _get_yolo()
        return YOLO is not None

    def get_backend_info(self) -> dict:
        """Get info about available detection backends."""
        return {
            "mediapipe_available": is_mediapipe_available(),
            "yolo_available": self.is_yolo_available(),
            "haar_available": True,  # Always available with OpenCV
        }


# Process-local models instance
_process_models: Optional[AnalysisModels] = None


def get_models() -> AnalysisModels:
    """Get or create models for current process."""
    global _process_models
    if _process_models is None:
        _process_models = AnalysisModels()
    return _process_models


def warmup_models(face_model: str = "short", use_yolo: bool = False):
    """
    Pre-load models to avoid cold start latency.

    Args:
        face_model: "short" or "full" for MediaPipe face detection
        use_yolo: Whether to load YOLO model
    """
    models = get_models()

    # Create a dummy frame for warmup
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Warm up face detector (MediaPipe or Haar cascade)
    models.detect_faces(dummy_frame, face_model)

    # Warm up person detector
    models.detect_persons(dummy_frame, use_yolo=False)

    # Optionally warm up YOLO
    if use_yolo:
        models.detect_persons_yolo(dummy_frame)


def print_backend_status():
    """Print which detection backends are available."""
    models = get_models()
    info = models.get_backend_info()

    print("Detection backends:")
    print(f"  MediaPipe: {'available' if info['mediapipe_available'] else 'not available (using Haar cascade fallback)'}")
    print(f"  YOLO:      {'available' if info['yolo_available'] else 'not available'}")
    print(f"  Haar:      always available")
