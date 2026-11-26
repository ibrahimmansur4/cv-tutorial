"""Small geometry primitives used by gesture and cursor logic."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class Point:
    """A normalized MediaPipe landmark or a pixel/screen point."""

    x: float
    y: float
    z: float = 0.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.x, self.y, self.z)):
            raise ValueError("point coordinates must be finite")

    def distance_to(self, other: "Point", *, include_z: bool = False) -> float:
        dz = self.z - other.z if include_z else 0.0
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + dz**2)


def joint_angle(first: Point, vertex: Point, third: Point) -> float:
    """Return the smaller 2D angle at ``vertex`` in degrees."""

    first_vector = np.asarray((first.x - vertex.x, first.y - vertex.y), dtype=np.float64)
    third_vector = np.asarray((third.x - vertex.x, third.y - vertex.y), dtype=np.float64)
    denominator = float(np.linalg.norm(first_vector) * np.linalg.norm(third_vector))
    if denominator <= 1e-12:
        return 0.0
    cosine = float(np.clip(np.dot(first_vector, third_vector) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


@dataclass(frozen=True, slots=True)
class CursorMapper:
    """Map a normalized camera point into bounded screen coordinates."""

    screen_width: int
    screen_height: int
    margin: float = 0.08

    def __post_init__(self) -> None:
        if self.screen_width <= 0 or self.screen_height <= 0:
            raise ConfigurationError("screen dimensions must be positive")
        if not math.isfinite(self.margin) or not 0.0 <= self.margin < 0.5:
            raise ConfigurationError("active margin must be between 0 and 0.5")

    def map(self, point: Point) -> Point:
        usable = 1.0 - 2.0 * self.margin
        normalized_x = min(1.0, max(0.0, (point.x - self.margin) / usable))
        normalized_y = min(1.0, max(0.0, (point.y - self.margin) / usable))
        return Point(
            normalized_x * max(0, self.screen_width - 1),
            normalized_y * max(0, self.screen_height - 1),
        )

    def frame_rectangle(self, frame_shape: Sequence[int]) -> tuple[int, int, int, int]:
        if len(frame_shape) < 2:
            raise ValueError("frame shape must contain height and width")
        height, width = int(frame_shape[0]), int(frame_shape[1])
        return (
            int(round(width * self.margin)),
            int(round(height * self.margin)),
            int(round(width * (1.0 - self.margin))),
            int(round(height * (1.0 - self.margin))),
        )
