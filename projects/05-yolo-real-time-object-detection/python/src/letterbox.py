"""Aspect-ratio-preserving resize, and the inverse map back to frame pixels.

A YOLO network accepts a fixed square input, but a webcam frame is not square.
Stretching a 16:9 frame into 640x640 distorts every object in it, so people
become narrow and wide objects become tall; the model was trained on
undistorted shapes and detects the distorted ones less reliably.

Letterboxing scales the frame by a single factor that makes it fit, then pads
the leftover strips with a neutral gray. Geometry is preserved, so the only
cost is that the padded strips carry no information.

Because the network reports boxes in padded-input coordinates, the padding and
scale have to be undone before anything is drawn on the original frame. That
inverse is the whole reason this is a class and not one function: the forward
and backward transforms must agree exactly, so they are stored together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .errors import ConfigurationError

# The fill value Ultralytics uses for letterbox padding; matching it keeps
# inference consistent with how these models were validated.
PAD_VALUE = 114


@dataclass(frozen=True, slots=True)
class Letterbox:
    """The exact scale and padding applied to one frame."""

    scale: float
    pad_x: float
    pad_y: float
    source_size: tuple[int, int]

    def to_source(self, boxes: np.ndarray) -> np.ndarray:
        """Map `(x, y, w, h)` boxes from network input space back to frame pixels.

        Returns `(x1, y1, x2, y2)` corners clipped to the frame, because a model
        routinely predicts boxes that run past the edge of a cropped object.
        """
        if boxes.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
        width, height = self.source_size
        x = (boxes[:, 0] - self.pad_x) / self.scale
        y = (boxes[:, 1] - self.pad_y) / self.scale
        box_width = boxes[:, 2] / self.scale
        box_height = boxes[:, 3] / self.scale
        corners = np.stack((x, y, x + box_width, y + box_height), axis=1)
        corners[:, 0::2] = corners[:, 0::2].clip(0.0, width - 1.0)
        corners[:, 1::2] = corners[:, 1::2].clip(0.0, height - 1.0)
        return corners.astype(np.float32, copy=False)


def letterbox(
    frame: np.ndarray, input_size: tuple[int, int], *, cv2_module: Any = cv2
) -> tuple[np.ndarray, Letterbox]:
    """Fit `frame` into `input_size` without distortion, returning the transform."""
    target_width, target_height = input_size
    if target_width <= 0 or target_height <= 0:
        raise ConfigurationError("network input size must be positive")
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ConfigurationError("cannot letterbox an empty frame")
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise ConfigurationError("letterboxing expects a BGR or BGRA frame")

    source_height, source_width = frame.shape[:2]
    scale = min(target_width / source_width, target_height / source_height)
    # Clamp to at least one pixel so a very wide or very tall frame still resizes.
    scaled_width = max(1, min(target_width, int(round(source_width * scale))))
    scaled_height = max(1, min(target_height, int(round(source_height * scale))))

    # INTER_AREA averages source pixels when shrinking, which keeps small
    # objects sharper than the default bilinear sampling; INTER_LINEAR is the
    # better choice in the rarer case of upscaling a low-resolution frame.
    interpolation = (
        cv2_module.INTER_AREA
        if scaled_width < source_width or scaled_height < source_height
        else cv2_module.INTER_LINEAR
    )
    resized = cv2_module.resize(frame, (scaled_width, scaled_height), interpolation=interpolation)

    pad_x = (target_width - scaled_width) / 2.0
    pad_y = (target_height - scaled_height) / 2.0
    top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
    bottom = target_height - scaled_height - top
    right = target_width - scaled_width - left
    padded = cv2_module.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2_module.BORDER_CONSTANT,
        value=(PAD_VALUE, PAD_VALUE, PAD_VALUE),
    )
    # Report the padding actually applied, not the ideal fractional value, so
    # the inverse transform cannot drift by half a pixel.
    return padded, Letterbox(scale, float(left), float(top), (source_width, source_height))
