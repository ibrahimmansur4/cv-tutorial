"""MediaPipe Face Landmarker adapter and landmark rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .errors import ConfigurationError, DependencyError, LandmarkError
from .geometry import Point
from .metrics import LEFT_EYE, MAX_LANDMARK_INDEX, MOUTH, RIGHT_EYE


@dataclass(frozen=True, slots=True)
class FaceObservation:
    """One detected face as a full set of mesh landmarks."""

    landmarks: tuple[Point, ...]

    def __post_init__(self) -> None:
        if len(self.landmarks) <= MAX_LANDMARK_INDEX:
            raise ValueError(
                f"a face observation needs more than {MAX_LANDMARK_INDEX} landmarks"
            )


def _probability(name: str, value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ConfigurationError(f"{name} must be between 0 and 1")
    return float(value)


class MediaPipeFaceTracker:
    """Synchronous VIDEO-mode wrapper around MediaPipe's Face Landmarker."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        num_faces: int = 1,
        detection_confidence: float = 0.5,
        presence_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        mediapipe_module: Any = None,
        landmarker: Any = None,
        cv2_module: Any = cv2,
    ) -> None:
        if num_faces <= 0:
            raise ConfigurationError("number of faces must be positive")
        detection_confidence = _probability("detection confidence", detection_confidence)
        presence_confidence = _probability("presence confidence", presence_confidence)
        tracking_confidence = _probability("tracking confidence", tracking_confidence)
        self._cv2 = cv2_module
        self._last_timestamp_ms: int | None = None
        self._owns_landmarker = landmarker is None

        if landmarker is not None:
            self._landmarker = landmarker
            return

        if mediapipe_module is None:
            try:
                import mediapipe as mediapipe_module  # type: ignore[no-redef]
            except (ImportError, OSError) as exc:
                raise DependencyError(
                    "MediaPipe is unavailable. Install python/requirements.txt."
                ) from exc
        self._mp = mediapipe_module

        resolved_model = Path(model_path).expanduser().resolve()
        if not resolved_model.is_file():
            raise ConfigurationError(
                f"MediaPipe face model not found: {resolved_model}. "
                "Run `python scripts/download_model.py`."
            )
        try:
            options = self._mp.tasks.vision.FaceLandmarkerOptions(
                base_options=self._mp.tasks.BaseOptions(model_asset_path=str(resolved_model)),
                running_mode=self._mp.tasks.vision.RunningMode.VIDEO,
                num_faces=int(num_faces),
                min_face_detection_confidence=detection_confidence,
                min_face_presence_confidence=presence_confidence,
                min_tracking_confidence=tracking_confidence,
            )
            self._landmarker = self._mp.tasks.vision.FaceLandmarker.create_from_options(options)
        except ConfigurationError:
            raise
        except Exception as exc:
            raise LandmarkError(
                f"could not initialize MediaPipe Face Landmarker: {exc}"
            ) from exc

    def detect(self, frame: np.ndarray, timestamp_ms: int) -> list[FaceObservation]:
        """Locate faces in one BGR frame."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise LandmarkError("cannot track a face in an empty frame")
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            raise LandmarkError("face tracking expects a BGR or BGRA frame")
        if timestamp_ms < 0:
            raise LandmarkError("MediaPipe timestamp cannot be negative")
        if self._last_timestamp_ms is not None and timestamp_ms <= self._last_timestamp_ms:
            # VIDEO mode uses the timestamp to decide whether it may reuse the
            # previous frame's tracking, so it must advance every frame.
            raise LandmarkError("MediaPipe timestamps must increase for every frame")
        self._last_timestamp_ms = int(timestamp_ms)

        try:
            conversion = (
                self._cv2.COLOR_BGRA2RGB if frame.shape[2] == 4 else self._cv2.COLOR_BGR2RGB
            )
            rgb = np.ascontiguousarray(self._cv2.cvtColor(frame, conversion))
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect_for_video(image, int(timestamp_ms))
        except Exception as exc:
            raise LandmarkError(f"MediaPipe face tracking failed: {exc}") from exc

        faces: list[FaceObservation] = []
        for raw_landmarks in tuple(getattr(result, "face_landmarks", ()) or ()):
            if len(raw_landmarks) <= MAX_LANDMARK_INDEX:
                continue
            faces.append(
                FaceObservation(
                    tuple(
                        Point(
                            float(item.x),
                            float(item.y),
                            float(getattr(item, "z", 0.0) or 0.0),
                        )
                        for item in raw_landmarks
                    )
                )
            )
        return faces

    def draw(self, frame: np.ndarray, face: FaceObservation, *, eyes_closed: bool) -> None:
        """Outline the eyes and inner lip, colouring the eyes by their state."""
        height, width = frame.shape[:2]
        eye_color = (60, 60, 255) if eyes_closed else (80, 230, 80)
        for indices, color in ((LEFT_EYE, eye_color), (RIGHT_EYE, eye_color), (MOUTH, (240, 180, 60))):
            pixels = np.array(
                [face.landmarks[index].to_pixels(width, height) for index in indices],
                dtype=np.int32,
            )
            # p1..p6 trace the outline in order, so a closed polyline is the shape.
            self._cv2.polylines(frame, [pixels], True, color, 1, self._cv2.LINE_AA)
            for pixel in pixels:
                self._cv2.circle(frame, tuple(int(value) for value in pixel), 2, color, -1)

    def close(self) -> None:
        if self._owns_landmarker:
            try:
                self._landmarker.close()
            except Exception as exc:
                raise LandmarkError(
                    f"could not close MediaPipe Face Landmarker: {exc}"
                ) from exc

    def __enter__(self) -> "MediaPipeFaceTracker":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
