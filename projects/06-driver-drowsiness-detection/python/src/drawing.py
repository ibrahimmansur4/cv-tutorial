"""Head-up display: metrics, eye state, drowsiness timer, and warning banner."""

from __future__ import annotations

import math
from typing import Any, Sequence

import cv2
import numpy as np

from .state import AlertLevel, MonitorState

FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_ALPHA = 0.55
PANEL_COLOR = (24, 24, 24)
TEXT_COLOR = (255, 255, 255)

LEVEL_COLORS: dict[AlertLevel, tuple[int, int, int]] = {
    AlertLevel.AWAKE: (90, 220, 90),
    AlertLevel.FATIGUE: (60, 200, 255),
    AlertLevel.DROWSY: (60, 60, 255),
    AlertLevel.NO_FACE: (180, 180, 180),
}
LEVEL_TEXT: dict[AlertLevel, str] = {
    AlertLevel.AWAKE: "AWAKE",
    AlertLevel.FATIGUE: "FATIGUE - take a break",
    AlertLevel.DROWSY: "DROWSY - WAKE UP",
    AlertLevel.NO_FACE: "NO FACE DETECTED",
}


def _scale(width: int) -> float:
    return float(np.clip(width / 1400.0, 0.45, 0.9))


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


def _format(value: float, digits: int = 3) -> str:
    return "--" if not math.isfinite(value) else f"{value:.{digits}f}"


def draw_hud(
    frame: np.ndarray,
    state: MonitorState,
    fps: float,
    *,
    ear_threshold: float,
    extra_lines: Sequence[str] = (),
    cv2_module: Any = cv2,
) -> np.ndarray:
    """Draw the metric panel, in place."""
    height, width = frame.shape[:2]
    scale = _scale(width)
    thickness = 1 if scale < 0.6 else 2
    line_height = int(round(26 * scale / 0.6))

    rows = [
        f"EAR: {_format(state.ear)} (closed below {ear_threshold:.3f})",
        f"Eyes: {state.eye_state.value.upper()}",
        f"Closed for: {state.closure_seconds:.2f}s",
        f"PERCLOS: {state.perclos * 100:.1f}%",
        f"MAR: {_format(state.mar)}{'  YAWNING' if state.yawning else ''}",
        f"Blinks: {state.blinks}   Yawns: {state.yawns}",
        f"FPS: {fps:.1f}",
    ]
    if state.face_missing_seconds > 0.0:
        rows.append(f"Face missing: {state.face_missing_seconds:.1f}s")
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
            TEXT_COLOR,
            thickness,
            cv2_module.LINE_AA,
        )
    return frame


def draw_banner(
    frame: np.ndarray, state: MonitorState, *, cv2_module: Any = cv2, blink_phase: bool = True
) -> np.ndarray:
    """Draw the alert banner, and a flashing border while the alarm is active."""
    height, width = frame.shape[:2]
    scale = _scale(width)
    color = LEVEL_COLORS[state.level]
    text = LEVEL_TEXT[state.level]
    thickness = 2 if scale >= 0.6 else 1

    (text_width, text_height), baseline = cv2_module.getTextSize(
        text, FONT, scale * 1.3, thickness + 1
    )
    x = max(8, (width - text_width) // 2)
    y = height - 18
    _blend_panel(
        frame,
        (x - 12, y - text_height - baseline - 8),
        (x + text_width + 12, y + baseline + 4),
        cv2_module,
    )
    cv2_module.putText(
        frame, text, (x, y), FONT, scale * 1.3, color, thickness + 1, cv2_module.LINE_AA
    )

    # A flashing border is far easier to notice in peripheral vision than text.
    if state.should_alarm and blink_phase:
        border = max(4, int(round(min(width, height) * 0.012)))
        cv2_module.rectangle(
            frame, (0, 0), (width - 1, height - 1), LEVEL_COLORS[AlertLevel.DROWSY], border
        )
    return frame
