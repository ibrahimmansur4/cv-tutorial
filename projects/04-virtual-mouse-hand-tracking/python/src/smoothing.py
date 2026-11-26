"""Cursor smoothing and small-motion suppression."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import ConfigurationError
from .geometry import Point


@dataclass(frozen=True, slots=True)
class SmoothedPoint:
    point: Point
    suppressed: bool


class ExponentialCursorSmoother:
    """Low-pass cursor positions and ignore tiny output movements."""

    def __init__(self, alpha: float = 0.35, dead_zone: float = 2.0) -> None:
        if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
            raise ConfigurationError("smoothing alpha must be above 0 and at most 1")
        if not math.isfinite(dead_zone) or dead_zone < 0.0:
            raise ConfigurationError("dead zone must be non-negative")
        self.alpha = float(alpha)
        self.dead_zone = float(dead_zone)
        self._current: Point | None = None

    @property
    def current(self) -> Point | None:
        return self._current

    def reset(self) -> None:
        self._current = None

    def update(self, target: Point) -> SmoothedPoint:
        if self._current is None:
            self._current = Point(target.x, target.y)
            return SmoothedPoint(self._current, False)
        candidate = Point(
            self._current.x + self.alpha * (target.x - self._current.x),
            self._current.y + self.alpha * (target.y - self._current.y),
        )
        if candidate.distance_to(self._current) < self.dead_zone:
            return SmoothedPoint(self._current, True)
        self._current = candidate
        return SmoothedPoint(candidate, False)
