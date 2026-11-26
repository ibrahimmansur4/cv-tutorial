"""Classical computer-vision pipeline for locating license plates.

The detector deliberately avoids a learned model.  It finds strong horizontal
edge groups and ranks rotated rectangular contours using area, aspect ratio,
and rectangularity.  Each accepted rectangle is perspective-corrected before
being enhanced for OCR.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .errors import DependencyError, DetectionError

try:
    import cv2 as _cv2
except ImportError:  # Keep CLI/help and dependency errors usable without cv2.
    _cv2 = None


def require_opencv() -> Any:
    """Return OpenCV or raise an installation-oriented error."""

    if _cv2 is None:
        raise DependencyError(
            "OpenCV is not installed. Run `python -m pip install -r "
            "requirements.txt` from the Python project directory."
        )
    return _cv2


@dataclass(frozen=True)
class Box:
    """Axis-aligned rectangle in pixel coordinates."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("box width and height must be positive")

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    def clipped(self, frame_width: int, frame_height: int) -> Optional["Box"]:
        left = min(max(self.x, 0), frame_width)
        top = min(max(self.y, 0), frame_height)
        right = min(max(self.right, 0), frame_width)
        bottom = min(max(self.bottom, 0), frame_height)
        if right <= left or bottom <= top:
            return None
        return Box(left, top, right - left, bottom - top)

    def iou(self, other: "Box") -> float:
        """Return intersection-over-union with another rectangle."""

        left = max(self.x, other.x)
        top = max(self.y, other.y)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)
        intersection = max(0, right - left) * max(0, bottom - top)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0


@dataclass(frozen=True)
class DetectorConfig:
    """Thresholds controlling contour generation and candidate filtering."""

    min_area_ratio: float = 0.005
    max_area_ratio: float = 0.08
    min_aspect_ratio: float = 2.0
    max_aspect_ratio: float = 5.5
    min_rectangularity: float = 0.45
    min_confidence: float = 0.45
    blur_kernel: int = 5
    canny_low: int = 50
    canny_high: int = 200
    morph_kernel: Tuple[int, int] = (9, 3)
    morph_iterations: int = 1
    max_candidates: int = 5
    enhancement_width: int = 440

    def __post_init__(self) -> None:
        probabilities = {
            "min_area_ratio": self.min_area_ratio,
            "max_area_ratio": self.max_area_ratio,
            "min_rectangularity": self.min_rectangularity,
            "min_confidence": self.min_confidence,
        }
        for name, value in probabilities.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.min_area_ratio >= self.max_area_ratio:
            raise ValueError("min_area_ratio must be smaller than max_area_ratio")
        if self.min_aspect_ratio <= 0:
            raise ValueError("min_aspect_ratio must be positive")
        if self.min_aspect_ratio >= self.max_aspect_ratio:
            raise ValueError("min_aspect_ratio must be smaller than max_aspect_ratio")
        if self.blur_kernel <= 0 or self.blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be a positive odd integer")
        if self.canny_low < 0 or self.canny_high <= self.canny_low:
            raise ValueError("canny_high must be greater than non-negative canny_low")
        if len(self.morph_kernel) != 2 or any(value <= 0 for value in self.morph_kernel):
            raise ValueError("morph_kernel dimensions must be positive")
        if self.morph_iterations <= 0:
            raise ValueError("morph_iterations must be positive")
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        if self.enhancement_width < 32:
            raise ValueError("enhancement_width must be at least 32 pixels")


@dataclass(frozen=True)
class PlateCandidate:
    """A localized plate and its OCR-ready perspective-corrected image."""

    box: Box
    corners: np.ndarray
    plate_image: np.ndarray
    confidence: float
    aspect_ratio: float
    rectangularity: float


def validate_frame(frame: np.ndarray) -> Tuple[int, int]:
    """Validate an image-like ndarray and return ``(width, height)``."""

    if frame is None or not isinstance(frame, np.ndarray):
        raise DetectionError("Received an invalid image frame.")
    if frame.ndim not in (2, 3) or frame.size == 0:
        raise DetectionError("Received an empty or unsupported image frame.")
    if frame.ndim == 3 and frame.shape[2] not in (1, 3, 4):
        raise DetectionError("Image must be grayscale, BGR, or BGRA.")
    height, width = int(frame.shape[0]), int(frame.shape[1])
    if width <= 0 or height <= 0:
        raise DetectionError("Received an empty image frame.")
    return width, height


def to_grayscale(frame: np.ndarray, cv_module: Any = None) -> np.ndarray:
    """Convert grayscale/BGR/BGRA input into one grayscale channel."""

    cv = cv_module if cv_module is not None else require_opencv()
    validate_frame(frame)
    if frame.ndim == 2:
        return frame.copy()
    if frame.shape[2] == 1:
        return frame[:, :, 0].copy()
    conversion = cv.COLOR_BGRA2GRAY if frame.shape[2] == 4 else cv.COLOR_BGR2GRAY
    try:
        return cv.cvtColor(frame, conversion)
    except Exception as exc:
        raise DetectionError(f"Could not convert the frame to grayscale: {exc}") from exc


def order_corners(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""

    coordinates = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if coordinates.shape != (4, 2) or not np.all(np.isfinite(coordinates)):
        raise DetectionError("Perspective correction requires four finite corner points.")
    ordered = np.empty((4, 2), dtype=np.float32)
    coordinate_sums = coordinates.sum(axis=1)
    coordinate_differences = np.diff(coordinates, axis=1).reshape(-1)
    ordered[0] = coordinates[np.argmin(coordinate_sums)]
    ordered[2] = coordinates[np.argmax(coordinate_sums)]
    ordered[1] = coordinates[np.argmin(coordinate_differences)]
    ordered[3] = coordinates[np.argmax(coordinate_differences)]
    if len(np.unique(ordered, axis=0)) != 4:
        raise DetectionError("Plate corners are degenerate and cannot be warped.")
    return ordered


def perspective_correct(
    frame: np.ndarray, corners: np.ndarray, cv_module: Any = None
) -> np.ndarray:
    """Warp a rotated plate quadrilateral into a front-facing rectangle."""

    cv = cv_module if cv_module is not None else require_opencv()
    validate_frame(frame)
    ordered = order_corners(corners)
    top_left, top_right, bottom_right, bottom_left = ordered
    width = int(
        round(
            max(
                np.linalg.norm(bottom_right - bottom_left),
                np.linalg.norm(top_right - top_left),
            )
        )
    )
    height = int(
        round(
            max(
                np.linalg.norm(top_right - bottom_right),
                np.linalg.norm(top_left - bottom_left),
            )
        )
    )
    if width < 2 or height < 2:
        raise DetectionError("Detected plate is too small for perspective correction.")
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    try:
        transform = cv.getPerspectiveTransform(ordered, destination)
        warped = cv.warpPerspective(frame, transform, (width, height))
        if height > width:
            warped = cv.rotate(warped, cv.ROTATE_90_CLOCKWISE)
    except Exception as exc:
        raise DetectionError(f"Could not perspective-correct a plate: {exc}") from exc
    return warped


def enhance_plate(
    plate: np.ndarray, target_width: int = 440, cv_module: Any = None
) -> np.ndarray:
    """Normalize contrast and produce a clean binary image for Tesseract."""

    cv = cv_module if cv_module is not None else require_opencv()
    width, height = validate_frame(plate)
    if target_width < 32:
        raise ValueError("target_width must be at least 32")
    output_height = max(32, int(round(height * target_width / width)))
    try:
        interpolation = cv.INTER_CUBIC if target_width >= width else cv.INTER_AREA
        resized = cv.resize(plate, (target_width, output_height), interpolation=interpolation)
        gray = to_grayscale(resized, cv)
        contrast = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        denoised = cv.GaussianBlur(contrast, (3, 3), 0)
        _, binary = cv.threshold(
            denoised, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU
        )
        # Tesseract is most reliable with dark glyphs on a light background.
        if float(np.mean(binary)) < 127.0:
            binary = cv.bitwise_not(binary)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
        cleaned = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=1)

        # Plate frames, mounting screws, and compression ringing at the outer
        # edge are often mistaken for leading/trailing ``I`` or ``1`` glyphs.
        # A small relative inset removes those artifacts while retaining the
        # text region across different plate sizes.
        margin_x = max(1, int(round(cleaned.shape[1] * 0.04)))
        margin_y = max(1, int(round(cleaned.shape[0] * 0.05)))
        if cleaned.shape[1] - 2 * margin_x >= 32 and cleaned.shape[0] - 2 * margin_y >= 16:
            cleaned = cleaned[margin_y:-margin_y, margin_x:-margin_x]
        return cleaned
    except DetectionError:
        raise
    except Exception as exc:
        raise DetectionError(f"Could not enhance the detected plate: {exc}") from exc


def _contour_result_contours(result: Sequence[Any]) -> Sequence[np.ndarray]:
    """Extract contours from either OpenCV 3 or OpenCV 4 return values."""

    if len(result) == 2:
        return result[0]
    if len(result) == 3:
        return result[1]
    raise DetectionError("OpenCV returned an unexpected contour result.")


class PlateDetector:
    """Locate likely plates using edges, morphology, and contour geometry."""

    def __init__(self, config: DetectorConfig, *, cv_module: Any = None) -> None:
        self.config = config
        self._cv = cv_module if cv_module is not None else require_opencv()

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = to_grayscale(frame, self._cv)
        try:
            blurred = self._cv.GaussianBlur(
                gray, (self.config.blur_kernel, self.config.blur_kernel), 0
            )
            edges = self._cv.Canny(
                blurred, self.config.canny_low, self.config.canny_high
            )
            kernel = self._cv.getStructuringElement(
                self._cv.MORPH_RECT, self.config.morph_kernel
            )
            return self._cv.morphologyEx(
                edges,
                self._cv.MORPH_CLOSE,
                kernel,
                iterations=self.config.morph_iterations,
            )
        except Exception as exc:
            raise DetectionError(f"Plate preprocessing failed: {exc}") from exc

    def _score(
        self, aspect_ratio: float, rectangularity: float, vertex_count: int
    ) -> float:
        midpoint = (self.config.min_aspect_ratio + self.config.max_aspect_ratio) / 2.0
        half_range = (self.config.max_aspect_ratio - self.config.min_aspect_ratio) / 2.0
        aspect_score = max(0.0, 1.0 - abs(aspect_ratio - midpoint) / half_range)
        geometry_score = 1.0 if vertex_count == 4 else 0.65
        return min(1.0, 0.60 * rectangularity + 0.25 * aspect_score + 0.15 * geometry_score)

    def _candidate_from_contour(
        self,
        contour: np.ndarray,
        frame: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> Optional[PlateCandidate]:
        cv = self._cv
        try:
            contour_area = float(cv.contourArea(contour))
            rotated_rectangle = cv.minAreaRect(contour)
            rectangle_width, rectangle_height = (
                float(rotated_rectangle[1][0]),
                float(rotated_rectangle[1][1]),
            )
        except Exception as exc:
            raise DetectionError(f"Could not measure a plate contour: {exc}") from exc

        short_side = min(rectangle_width, rectangle_height)
        long_side = max(rectangle_width, rectangle_height)
        rectangle_area = short_side * long_side
        if short_side <= 1.0 or rectangle_area <= 0.0 or contour_area <= 0.0:
            return None

        area_ratio = rectangle_area / float(frame_width * frame_height)
        aspect_ratio = long_side / short_side
        rectangularity = min(1.0, contour_area / rectangle_area)
        if not self.config.min_area_ratio <= area_ratio <= self.config.max_area_ratio:
            return None
        if not self.config.min_aspect_ratio <= aspect_ratio <= self.config.max_aspect_ratio:
            return None
        if rectangularity < self.config.min_rectangularity:
            return None

        try:
            perimeter = float(cv.arcLength(contour, True))
            approximation = cv.approxPolyDP(contour, 0.03 * perimeter, True)
            confidence = self._score(aspect_ratio, rectangularity, len(approximation))
            if confidence < self.config.min_confidence:
                return None
            corners = np.asarray(cv.boxPoints(rotated_rectangle), dtype=np.float32)
            raw_x, raw_y, raw_width, raw_height = cv.boundingRect(corners)
        except Exception as exc:
            raise DetectionError(f"Could not fit a rectangle to a plate contour: {exc}") from exc

        clipped_box = Box(
            int(raw_x), int(raw_y), int(raw_width), int(raw_height)
        ).clipped(frame_width, frame_height)
        if clipped_box is None:
            return None
        try:
            corrected = perspective_correct(frame, corners, cv)
            enhanced = enhance_plate(
                corrected, target_width=self.config.enhancement_width, cv_module=cv
            )
        except DetectionError:
            # A single numerically degenerate contour should not abort a frame.
            return None
        return PlateCandidate(
            box=clipped_box,
            corners=order_corners(corners),
            plate_image=enhanced,
            confidence=confidence,
            aspect_ratio=aspect_ratio,
            rectangularity=rectangularity,
        )

    @staticmethod
    def _deduplicate(candidates: Sequence[PlateCandidate]) -> List[PlateCandidate]:
        selected: List[PlateCandidate] = []
        for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
            if all(candidate.box.iou(existing.box) < 0.45 for existing in selected):
                selected.append(candidate)
        return selected

    def detect(self, frame: np.ndarray) -> Sequence[PlateCandidate]:
        """Return ranked OCR-ready plate candidates from a BGR frame."""

        frame_width, frame_height = validate_frame(frame)
        processed = self._preprocess(frame)
        try:
            contour_result = self._cv.findContours(
                processed.copy(), self._cv.RETR_LIST, self._cv.CHAIN_APPROX_SIMPLE
            )
            contours = _contour_result_contours(contour_result)
        except DetectionError:
            raise
        except Exception as exc:
            raise DetectionError(f"Could not find plate contours: {exc}") from exc

        candidates: List[PlateCandidate] = []
        for contour in contours:
            candidate = self._candidate_from_contour(
                contour, frame, frame_width, frame_height
            )
            if candidate is not None:
                candidates.append(candidate)
        return self._deduplicate(candidates)[: self.config.max_candidates]
