"""Overlay motion boxes, status, and the recording indicator."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

import cv2
import numpy as np

from .detector import MotionRegion

FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_ALPHA = 0.55
PANEL_COLOR = (24, 24, 24)
TEXT_COLOR = (255, 255, 255)
MOTION_COLOR = (60, 220, 255)
RECORDING_COLOR = (60, 60, 255)
IDLE_COLOR = (120, 220, 120)


def _scale(width: int) -> float:
    return float(np.clip(width / 1400.0, 0.42, 0.85))


def _blend_panel(
    frame: np.ndarray, top_left: tuple[int, int], bottom_right: tuple[int, int], cv2_module: Any
) -> None:
    x1, y1 = top_left
    x2, y2 = bottom_right
    height, width = frame.shape[:2]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return
    region = frame[y1:y2, x1:x2]
    panel = np.full_like(region, PANEL_COLOR, dtype=np.uint8)
    cv2_module.addWeighted(panel, PANEL_ALPHA, region, 1.0 - PANEL_ALPHA, 0.0, dst=region)


def draw_regions(
    frame: np.ndarray, regions: Sequence[MotionRegion], *, cv2_module: Any = cv2
) -> np.ndarray:
    """Box each moving region, in place."""
    scale = _scale(frame.shape[1])
    for region in regions:
        x1, y1, x2, y2 = region.box
        cv2_module.rectangle(frame, (x1, y1), (x2, y2), MOTION_COLOR, 2)
        cv2_module.putText(
            frame,
            f"{int(region.area)}px",
            (x1, max(14, y1 - 6)),
            FONT,
            scale * 0.8,
            MOTION_COLOR,
            1,
            cv2_module.LINE_AA,
        )
    return frame


def draw_status(
    frame: np.ndarray,
    *,
    motion: bool,
    recording: bool,
    ready: bool,
    fps: float,
    events: int,
    changed_fraction: float,
    model_name: str,
    timestamp: datetime | None = None,
    extra_lines: Sequence[str] = (),
    cv2_module: Any = cv2,
    blink_phase: bool = True,
) -> np.ndarray:
    """Draw the status panel, timestamp, and recording dot, in place."""
    height, width = frame.shape[:2]
    scale = _scale(width)
    thickness = 1 if scale < 0.6 else 2
    line_height = int(round(26 * scale / 0.6))

    if not ready:
        state = "LEARNING BACKGROUND"
        color = (180, 180, 180)
    elif motion:
        state = "MOTION"
        color = MOTION_COLOR
    else:
        state = "CLEAR"
        color = IDLE_COLOR

    rows = [
        f"Status: {state}",
        f"Model: {model_name}",
        f"Changed: {changed_fraction * 100:.2f}%",
        f"Events: {events}",
        f"FPS: {fps:.1f}",
    ]
    rows.extend(extra_lines)

    longest = max(cv2_module.getTextSize(row, FONT, scale, thickness)[0][0] for row in rows)
    margin = 10
    _blend_panel(
        frame,
        (margin - 4, margin - 4),
        (margin + longest + 12, margin + line_height * len(rows) + 8),
        cv2_module,
    )
    for index, row in enumerate(rows):
        cv2_module.putText(
            frame,
            row,
            (margin + 4, margin + line_height * (index + 1) - 6),
            FONT,
            scale,
            color if index == 0 else TEXT_COLOR,
            thickness,
            cv2_module.LINE_AA,
        )

    # A wall-clock timestamp burned into the frame is what makes a recording
    # usable as evidence later; the file name alone is easy to lose.
    stamp = (timestamp or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    (stamp_width, stamp_height), baseline = cv2_module.getTextSize(
        stamp, FONT, scale, thickness
    )
    x = width - stamp_width - 14
    y = height - 12
    _blend_panel(
        frame, (x - 8, y - stamp_height - baseline - 6), (width - 4, y + baseline), cv2_module
    )
    cv2_module.putText(
        frame, stamp, (x, y - baseline + 2), FONT, scale, TEXT_COLOR, thickness, cv2_module.LINE_AA
    )

    if recording and blink_phase:
        radius = max(6, int(round(min(width, height) * 0.012)))
        centre = (18 + radius, height - 18 - radius)
        cv2_module.circle(frame, centre, radius, RECORDING_COLOR, -1, cv2_module.LINE_AA)
        cv2_module.putText(
            frame,
            "REC",
            (centre[0] + radius + 8, centre[1] + radius // 2),
            FONT,
            scale,
            RECORDING_COLOR,
            thickness,
            cv2_module.LINE_AA,
        )
    return frame


def stack_mask(frame: np.ndarray, mask: np.ndarray, *, cv2_module: Any = cv2) -> np.ndarray:
    """Put the binary mask beside the frame, for debugging the pipeline."""
    coloured = cv2_module.cvtColor(mask, cv2_module.COLOR_GRAY2BGR)
    if coloured.shape[:2] != frame.shape[:2]:
        coloured = cv2_module.resize(coloured, (frame.shape[1], frame.shape[0]))
    return np.hstack([frame, coloured])
