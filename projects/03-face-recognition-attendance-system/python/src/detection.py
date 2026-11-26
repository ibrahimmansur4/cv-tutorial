"""Classical Haar-cascade face detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class FaceBox:
    """An axis-aligned face bounding box."""

    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    def clipped(self, frame_shape: Iterable[int]) -> "FaceBox":
        shape = tuple(frame_shape)
        if len(shape) < 2:
            raise ValueError("frame_shape must contain height and width")
        frame_height, frame_width = int(shape[0]), int(shape[1])
        x1 = max(0, min(self.x, frame_width))
        y1 = max(0, min(self.y, frame_height))
        x2 = max(x1, min(self.x2, frame_width))
        y2 = max(y1, min(self.y2, frame_height))
        return FaceBox(x1, y1, x2 - x1, y2 - y1)

    def crop(self, frame: np.ndarray) -> np.ndarray:
        box = self.clipped(frame.shape)
        return frame[box.y : box.y2, box.x : box.x2]


class FaceDetector:
    """Detect frontal faces using OpenCV's bundled Haar cascade."""

    def __init__(
        self,
        cascade_path: str | Path | None = None,
        *,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (48, 48),
        cv2_module: Any = cv2,
        classifier: Any | None = None,
    ) -> None:
        if scale_factor <= 1.0:
            raise ConfigurationError("Haar scale_factor must be greater than 1.0")
        if min_neighbors < 0:
            raise ConfigurationError("Haar min_neighbors cannot be negative")
        if min(min_size) <= 0:
            raise ConfigurationError("Haar min_size values must be positive")

        self._cv2 = cv2_module
        self.scale_factor = float(scale_factor)
        self.min_neighbors = int(min_neighbors)
        self.min_size = (int(min_size[0]), int(min_size[1]))

        if classifier is None:
            cascade = (
                Path(cascade_path).expanduser()
                if cascade_path is not None
                else Path(self._cv2.data.haarcascades)
                / "haarcascade_frontalface_default.xml"
            )
            if not cascade.is_file():
                raise ConfigurationError(f"Haar cascade was not found: {cascade}")
            classifier = self._cv2.CascadeClassifier(str(cascade))
            if classifier.empty():
                raise ConfigurationError(f"OpenCV could not load Haar cascade: {cascade}")
            self.cascade_path: Path | None = cascade.resolve()
        else:
            self.cascade_path = (
                Path(cascade_path).expanduser().resolve()
                if cascade_path is not None
                else None
            )
        self._classifier = classifier

    def detect(self, frame: np.ndarray) -> list[FaceBox]:
        """Return detected faces, largest first."""

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame must be a non-empty NumPy image")
        if frame.ndim == 2:
            gray = frame
        elif frame.ndim == 3 and frame.shape[2] in (3, 4):
            conversion = (
                self._cv2.COLOR_BGRA2GRAY
                if frame.shape[2] == 4
                else self._cv2.COLOR_BGR2GRAY
            )
            gray = self._cv2.cvtColor(frame, conversion)
        else:
            raise ValueError("frame must be grayscale, BGR, or BGRA")

        equalized = self._cv2.equalizeHist(gray)
        detected = self._classifier.detectMultiScale(
            equalized,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=getattr(self._cv2, "CASCADE_SCALE_IMAGE", 0),
        )
        boxes = [FaceBox(*(int(value) for value in item)) for item in detected]
        boxes = [box.clipped(frame.shape) for box in boxes]
        return sorted((box for box in boxes if box.area > 0), key=lambda box: box.area, reverse=True)
