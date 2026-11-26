"""MediaPipe Hand Landmarker adapter and landmark rendering."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .errors import ConfigurationError, DependencyError, TrackingError
from .geometry import Point

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)


@dataclass(frozen=True, slots=True)
class HandObservation:
    landmarks: tuple[Point, ...]
    handedness: str = "Unknown"
    score: float = 0.0

    def __post_init__(self) -> None:
        if len(self.landmarks) != 21:
            raise ValueError("a hand observation requires exactly 21 landmarks")
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("handedness score must be between 0 and 1")


def _probability(name: str, value: float) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ConfigurationError(f"{name} must be between 0 and 1")
    return float(value)


class MediaPipeHandTracker:
    """Synchronous VIDEO-mode wrapper around MediaPipe's current Tasks API."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        num_hands: int = 1,
        detection_confidence: float = 0.5,
        presence_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        mediapipe_module: Any = None,
        landmarker: Any = None,
        cv2_module: Any = cv2,
    ) -> None:
        if num_hands <= 0:
            raise ConfigurationError("number of hands must be positive")
        detection_confidence = _probability("detection confidence", detection_confidence)
        presence_confidence = _probability("presence confidence", presence_confidence)
        tracking_confidence = _probability("tracking confidence", tracking_confidence)
        self._cv2 = cv2_module
        self._last_timestamp_ms: int | None = None
        self._owns_landmarker = landmarker is None

        if mediapipe_module is None:
            try:
                import mediapipe as mediapipe_module  # type: ignore[no-redef]
            except (ImportError, OSError) as exc:
                raise DependencyError(
                    "MediaPipe is unavailable. Install python/requirements.txt."
                ) from exc
        self._mp = mediapipe_module

        if landmarker is not None:
            self._landmarker = landmarker
            return

        resolved_model = Path(model_path).expanduser().resolve()
        if not resolved_model.is_file():
            raise ConfigurationError(
                f"MediaPipe hand model not found: {resolved_model}. "
                "Run `python scripts/download_model.py`."
            )
        try:
            options = self._mp.tasks.vision.HandLandmarkerOptions(
                base_options=self._mp.tasks.BaseOptions(model_asset_path=str(resolved_model)),
                running_mode=self._mp.tasks.vision.RunningMode.VIDEO,
                num_hands=int(num_hands),
                min_hand_detection_confidence=detection_confidence,
                min_hand_presence_confidence=presence_confidence,
                min_tracking_confidence=tracking_confidence,
            )
            self._landmarker = self._mp.tasks.vision.HandLandmarker.create_from_options(options)
        except ConfigurationError:
            raise
        except Exception as exc:
            raise TrackingError(f"could not initialize MediaPipe Hand Landmarker: {exc}") from exc

    @staticmethod
    def _category(handedness: Any) -> tuple[str, float]:
        if not handedness:
            return "Unknown", 0.0
        category = handedness[0]
        name = (
            getattr(category, "category_name", None)
            or getattr(category, "display_name", None)
            or "Unknown"
        )
        score = float(getattr(category, "score", 0.0) or 0.0)
        return str(name), min(1.0, max(0.0, score))

    def detect(self, frame: np.ndarray, timestamp_ms: int) -> list[HandObservation]:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise TrackingError("cannot track hands in an empty frame")
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            raise TrackingError("hand tracking expects a BGR or BGRA frame")
        if timestamp_ms < 0:
            raise TrackingError("MediaPipe timestamp cannot be negative")
        if self._last_timestamp_ms is not None and timestamp_ms <= self._last_timestamp_ms:
            raise TrackingError("MediaPipe timestamps must increase for every frame")
        self._last_timestamp_ms = int(timestamp_ms)

        try:
            conversion = (
                self._cv2.COLOR_BGRA2RGB if frame.shape[2] == 4 else self._cv2.COLOR_BGR2RGB
            )
            rgb = np.ascontiguousarray(self._cv2.cvtColor(frame, conversion))
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect_for_video(image, int(timestamp_ms))
        except Exception as exc:
            raise TrackingError(f"MediaPipe hand tracking failed: {exc}") from exc

        observations: list[HandObservation] = []
        hand_landmarks = tuple(getattr(result, "hand_landmarks", ()) or ())
        handedness = tuple(getattr(result, "handedness", ()) or ())
        for index, raw_landmarks in enumerate(hand_landmarks):
            if len(raw_landmarks) != 21:
                continue
            points = tuple(
                Point(float(item.x), float(item.y), float(getattr(item, "z", 0.0) or 0.0))
                for item in raw_landmarks
            )
            name, score = self._category(handedness[index] if index < len(handedness) else ())
            observations.append(HandObservation(points, name, score))
        return observations

    def draw(self, frame: np.ndarray, hand: HandObservation) -> None:
        height, width = frame.shape[:2]
        pixels = [
            (
                min(width - 1, max(0, int(round(point.x * width)))),
                min(height - 1, max(0, int(round(point.y * height)))),
            )
            for point in hand.landmarks
        ]
        for start, end in HAND_CONNECTIONS:
            self._cv2.line(frame, pixels[start], pixels[end], (70, 210, 70), 2, self._cv2.LINE_AA)
        for index, pixel in enumerate(pixels):
            radius = 6 if index in (4, 8, 12) else 3
            color = (30, 80, 255) if index in (4, 8, 12) else (255, 190, 40)
            self._cv2.circle(frame, pixel, radius, color, -1, self._cv2.LINE_AA)
        self._cv2.putText(
            frame,
            f"{hand.handedness} {hand.score:.2f}",
            pixels[0],
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            self._cv2.LINE_AA,
        )

    def close(self) -> None:
        if self._owns_landmarker:
            try:
                self._landmarker.close()
            except Exception as exc:
                raise TrackingError(f"could not close MediaPipe Hand Landmarker: {exc}") from exc

    def __enter__(self) -> "MediaPipeHandTracker":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
