"""Draw the scanner's heads-up display.

Kept apart from detection so the pipeline can be tested without a window, and
so the drawing can be exercised on a blank array without a camera.

Two choices here are deliberate. Outlines are drawn as **polygons, not
rectangles**: the detector returns four corners of a code seen in perspective,
and a bounding box would throw away the orientation information that tells the
user how the code is tilted. And a decoded code is drawn green while a located
but unread one is drawn amber, because those two states call for different
actions from the user.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .detect import Detection
from .history import Scan

FONT = cv2.FONT_HERSHEY_SIMPLEX
DECODED_COLOR = (90, 220, 120)
PENDING_COLOR = (80, 180, 255)
PANEL_COLOR = (28, 28, 28)
TEXT_COLOR = (245, 245, 245)
MUTED_COLOR = (170, 170, 170)


@dataclass(frozen=True, slots=True)
class HudState:
    """Everything the overlay needs that is not a detection."""

    fps: float = 0.0
    frame_index: int = 0
    total_scans: int = 0
    suppressed: int = 0
    paused: bool = False
    message: str = ""


def _text_size(text: str, scale: float, thickness: int) -> tuple[int, int]:
    (width, height), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    return width, height + baseline


def draw_label(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    color: tuple[int, int, int] = TEXT_COLOR,
    background: tuple[int, int, int] = PANEL_COLOR,
    scale: float = 0.5,
    thickness: int = 1,
    alpha: float = 0.75,
) -> None:
    """Write text on a translucent plate so it stays readable over any frame.

    Text drawn straight onto video is unreadable the moment the background
    happens to match it; a plate behind it costs one blend and removes the
    problem entirely.
    """
    if not text:
        return
    pad = 5
    width, height = _text_size(text, scale, thickness)
    x, y = origin
    left = max(0, x)
    top = max(0, y - height - pad)
    right = min(frame.shape[1], left + width + 2 * pad)
    bottom = min(frame.shape[0], top + height + 2 * pad)
    if right <= left or bottom <= top:
        return
    patch = frame[top:bottom, left:right]
    plate = np.full_like(patch, background, dtype=frame.dtype)
    cv2.addWeighted(plate, alpha, patch, 1.0 - alpha, 0.0, dst=patch)
    cv2.putText(
        frame,
        text,
        (left + pad, bottom - pad - 2),
        FONT,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_detection(frame: np.ndarray, detection: Detection, *, show_label: bool = True) -> None:
    """Outline one code and name what it holds."""
    color = DECODED_COLOR if detection.decoded else PENDING_COLOR
    polygon = detection.corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [polygon], True, color, 2, cv2.LINE_AA)
    # Mark the first corner so the code's rotation is visible at a glance.
    first = tuple(detection.corners[0].astype(int))
    cv2.circle(frame, first, 4, color, -1, cv2.LINE_AA)
    if not show_label:
        return
    left, top, _, _ = detection.bounds
    draw_label(
        frame,
        detection.label,
        (left, max(18, top - 4)),
        color=color,
        scale=0.52,
    )


def draw_hud(
    frame: np.ndarray,
    state: HudState,
    recent: list[Scan] | None = None,
    *,
    show_help: bool = True,
) -> None:
    """Status line, recent scans, and the key hints."""
    status = f"FPS {state.fps:5.1f}   frame {state.frame_index}   scans {state.total_scans}"
    if state.suppressed:
        status += f"   repeats {state.suppressed}"
    if state.paused:
        status += "   [PAUSED]"
    draw_label(frame, status, (8, 26), scale=0.55)

    if state.message:
        draw_label(frame, state.message, (8, 52), color=PENDING_COLOR, scale=0.52)

    if recent:
        y = 80 if state.message else 54
        draw_label(frame, "Recent", (8, y), color=MUTED_COLOR, scale=0.45)
        for index, scan in enumerate(recent):
            line = f"{scan.when.strftime('%H:%M:%S')}  {scan.symbology}  {scan.summary}"
            draw_label(frame, line, (8, y + 22 * (index + 1)), scale=0.45)

    if show_help:
        hint = "Q quit   SPACE pause   S save frame   C clear"
        draw_label(
            frame,
            hint,
            (8, frame.shape[0] - 8),
            color=MUTED_COLOR,
            scale=0.45,
        )


def annotate(
    frame: np.ndarray,
    detections: list[Detection],
    state: HudState,
    recent: list[Scan] | None = None,
    *,
    show_help: bool = True,
    show_labels: bool = True,
) -> np.ndarray:
    """Return a copy of `frame` with outlines and the HUD drawn on it."""
    canvas = frame.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    for detection in detections:
        draw_detection(canvas, detection, show_label=show_labels)
    draw_hud(canvas, state, recent, show_help=show_help)
    return canvas
