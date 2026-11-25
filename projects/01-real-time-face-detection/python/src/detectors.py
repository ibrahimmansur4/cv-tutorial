"""Face-detector implementations used by the application.

Haar cascades are a classical, hand-engineered detector.  The OpenCV DNN
implementation loads the ResNet-10 SSD Caffe model and filters its learned
predictions by confidence.  Keeping these two algorithms behind the same
small interface lets the capture and rendering code stay detector-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence, Tuple

import numpy as np

try:
    import cv2 as _cv2
except ImportError:  # Allows argument parsing/tests to explain missing DLLs too.
    _cv2 = None


class FaceDetectionError(RuntimeError):
    """Base class for expected, user-facing detector errors."""


class DependencyError(FaceDetectionError):
    """Raised when OpenCV is not installed."""


class DetectorLoadError(FaceDetectionError):
    """Raised when a detector's model cannot be loaded."""


def require_opencv() -> Any:
    """Return cv2 or raise an actionable error rather than an import traceback."""

    if _cv2 is None:
        raise DependencyError(
            "OpenCV is not installed. Run `python -m pip install -r requirements.txt` "
            "from the Python project directory."
        )
    return _cv2


@dataclass(frozen=True)
class Detection:
    """One face bounding box in image coordinates."""

    left: int
    top: int
    right: int
    bottom: int
    confidence: Optional[float] = None

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def clipped(self, frame_width: int, frame_height: int) -> Optional["Detection"]:
        """Clip this box to a frame, returning None if it has no visible area."""

        left = min(max(self.left, 0), frame_width)
        top = min(max(self.top, 0), frame_height)
        right = min(max(self.right, 0), frame_width)
        bottom = min(max(self.bottom, 0), frame_height)
        if right <= left or bottom <= top:
            return None
        return Detection(left, top, right, bottom, self.confidence)


class FaceDetector(Protocol):
    """Common interface implemented by all face detectors."""

    def detect(self, frame: np.ndarray) -> Sequence[Detection]:
        """Find faces in a BGR frame."""


def _frame_dimensions(frame: np.ndarray) -> Tuple[int, int]:
    if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
        raise FaceDetectionError("Received an invalid video frame.")
    height, width = int(frame.shape[0]), int(frame.shape[1])
    if width <= 0 or height <= 0:
        raise FaceDetectionError("Received an empty video frame.")
    return width, height


class HaarFaceDetector:
    """Frontal-face detector based on OpenCV's Haar cascade."""

    DEFAULT_CASCADE = "haarcascade_frontalface_default.xml"

    def __init__(
        self,
        cascade_path: Optional[Path] = None,
        *,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30),
        cv_module: Any = None,
    ) -> None:
        if scale_factor <= 1.0:
            raise ValueError("scale_factor must be greater than 1.0")
        if min_neighbors < 0:
            raise ValueError("min_neighbors cannot be negative")

        self._cv = cv_module if cv_module is not None else require_opencv()
        if cascade_path is None:
            data = getattr(self._cv, "data", None)
            cascade_directory = getattr(data, "haarcascades", None)
            if not cascade_directory:
                raise DetectorLoadError(
                    "This OpenCV installation does not expose its Haar cascade data. "
                    "Pass the XML file explicitly with --cascade."
                )
            cascade_path = Path(cascade_directory) / self.DEFAULT_CASCADE
        else:
            cascade_path = Path(cascade_path).expanduser()

        if not cascade_path.is_file():
            raise DetectorLoadError(f"Haar cascade file not found: {cascade_path}")

        try:
            classifier = self._cv.CascadeClassifier(str(cascade_path))
        except Exception as exc:
            raise DetectorLoadError(
                f"Could not load Haar cascade: {cascade_path} ({exc})"
            ) from exc
        if classifier.empty():
            raise DetectorLoadError(f"OpenCV rejected Haar cascade: {cascade_path}")

        self._classifier = classifier
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._min_size = min_size

    def detect(self, frame: np.ndarray) -> Sequence[Detection]:
        width, height = _frame_dimensions(frame)
        gray = self._cv.cvtColor(frame, self._cv.COLOR_BGR2GRAY)
        # Equalization improves contrast before the cascade evaluates features.
        gray = self._cv.equalizeHist(gray)
        boxes = self._classifier.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=self._min_size,
        )

        detections = []
        for x, y, box_width, box_height in boxes:
            detection = Detection(
                int(x),
                int(y),
                int(x + box_width),
                int(y + box_height),
            ).clipped(width, height)
            if detection is not None:
                detections.append(detection)
        return detections


class DnnFaceDetector:
    """Face detector backed by OpenCV's ResNet-10 SSD Caffe network."""

    INPUT_SIZE = (300, 300)
    INPUT_MEAN = (104.0, 177.0, 123.0)

    def __init__(
        self,
        config_path: Path,
        model_path: Path,
        *,
        confidence_threshold: float = 0.5,
        cv_module: Any = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")

        config_path = Path(config_path).expanduser()
        model_path = Path(model_path).expanduser()
        if not config_path.is_file():
            raise DetectorLoadError(f"DNN config file not found: {config_path}")
        if not model_path.is_file():
            raise DetectorLoadError(f"DNN model file not found: {model_path}")

        self._cv = cv_module if cv_module is not None else require_opencv()
        try:
            network = self._cv.dnn.readNetFromCaffe(
                str(config_path), str(model_path)
            )
        except Exception as exc:
            raise DetectorLoadError(
                "Could not load the OpenCV SSD face detector from "
                f"{config_path} and {model_path}: {exc}"
            ) from exc

        if hasattr(network, "empty") and network.empty():
            raise DetectorLoadError(
                f"OpenCV rejected the DNN model files: {config_path}, {model_path}"
            )
        self._network = network
        self._confidence_threshold = confidence_threshold

    def detect(self, frame: np.ndarray) -> Sequence[Detection]:
        width, height = _frame_dimensions(frame)
        blob = self._cv.dnn.blobFromImage(
            frame,
            scalefactor=1.0,
            size=self.INPUT_SIZE,
            mean=self.INPUT_MEAN,
            swapRB=False,
            crop=False,
        )
        self._network.setInput(blob)
        raw_output = np.asarray(self._network.forward())
        if raw_output.size == 0:
            return []
        if raw_output.ndim < 2 or raw_output.shape[-1] < 7:
            raise FaceDetectionError(
                f"DNN returned an unexpected output shape: {raw_output.shape}"
            )

        detections = []
        scale = np.array([width, height, width, height], dtype=np.float32)
        for row in raw_output.reshape(-1, raw_output.shape[-1]):
            confidence = float(row[2])
            if not np.isfinite(confidence) or confidence < self._confidence_threshold:
                continue

            coordinates = row[3:7].astype(np.float32) * scale
            if not np.all(np.isfinite(coordinates)):
                continue
            left, top, right, bottom = (int(round(value)) for value in coordinates)
            detection = Detection(
                left, top, right, bottom, confidence
            ).clipped(width, height)
            if detection is not None:
                detections.append(detection)
        return detections


def create_detector(
    detector_name: str,
    *,
    confidence_threshold: float,
    cascade_path: Optional[Path] = None,
    dnn_config_path: Optional[Path] = None,
    dnn_model_path: Optional[Path] = None,
    cv_module: Any = None,
) -> FaceDetector:
    """Construct a detector from validated command-line configuration."""

    if detector_name == "haar":
        return HaarFaceDetector(cascade_path, cv_module=cv_module)
    if detector_name == "dnn":
        if dnn_config_path is None or dnn_model_path is None:
            raise DetectorLoadError(
                "The DNN detector requires both --dnn-config and --dnn-model. "
                "Use the deploy.prototxt and ResNet-10 SSD .caffemodel files."
            )
        return DnnFaceDetector(
            dnn_config_path,
            dnn_model_path,
            confidence_threshold=confidence_threshold,
            cv_module=cv_module,
        )
    raise FaceDetectionError(f"Unknown detector: {detector_name}")
