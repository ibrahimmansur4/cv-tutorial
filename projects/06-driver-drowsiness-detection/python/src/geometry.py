"""Minimal point type and distance helpers shared by the metric code."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Point:
    """A face-mesh landmark in normalized image coordinates.

    MediaPipe reports x and y in 0..1 relative to the frame, which keeps the
    aspect ratios independent of capture resolution. `z` is a relative depth
    the metrics here do not use.
    """

    x: float
    y: float
    z: float = 0.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.x) and math.isfinite(self.y) and math.isfinite(self.z)):
            raise ValueError("landmark coordinates must be finite")

    def to_pixels(self, width: int, height: int) -> tuple[int, int]:
        """Project onto a frame, clamped so a slightly out-of-frame point still draws."""
        x = min(width - 1, max(0, int(round(self.x * width))))
        y = min(height - 1, max(0, int(round(self.y * height))))
        return x, y


def distance(first: Point, second: Point, frame_aspect: float = 1.0) -> float:
    """Euclidean distance, with x rescaled so the plane is isotropic.

    MediaPipe normalizes x by the frame width and y by the frame height, so on
    any non-square frame one unit of x covers a different number of pixels than
    one unit of y. Measuring directly in that space stretches every horizontal
    distance by `height / width`, which is exactly the denominator of an aspect
    ratio -- the same eye then reads 0.18 on a portrait frame and 0.36 on a 16:9
    webcam. Multiplying x by `width / height` restores pixel proportions, up to
    a uniform scale that cancels in any ratio of two distances.

    Depth is deliberately ignored: MediaPipe's z is weakly constrained for a
    single camera, and including it adds noise to a ratio that only needs the
    projected shape of the eye.
    """
    if not math.isfinite(frame_aspect) or frame_aspect <= 0.0:
        raise ValueError("frame aspect ratio must be a positive, finite number")
    return math.hypot((first.x - second.x) * frame_aspect, first.y - second.y)
