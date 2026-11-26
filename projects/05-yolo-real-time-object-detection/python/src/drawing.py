"""Overlay detections, per-class counts, and timing onto a frame."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .detector import Detection
from .labels import color_for

FONT = cv2.FONT_HERSHEY_SIMPLEX
BOX_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)
PANEL_COLOR = (24, 24, 24)
PANEL_ALPHA = 0.55


def _text_scale(frame_width: int) -> float:
    """Keep labels legible at 320 px and not oversized at 1920 px."""
    return float(np.clip(frame_width / 1600.0, 0.40, 0.85))


def _blend_panel(
    frame: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    cv2_module: Any,
) -> None:
    """Darken a region in place so white text stays readable over any scene."""
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


def draw_detections(
    frame: np.ndarray,
    detections: Sequence[Detection],
    *,
    cv2_module: Any = cv2,
) -> np.ndarray:
    """Draw one labelled rectangle per detection, in place."""
    height, width = frame.shape[:2]
    scale = _text_scale(width)
    thickness = 1 if scale < 0.6 else 2
    for detection in detections:
        x1, y1, x2, y2 = detection.as_int_box()
        color = color_for(detection.class_id)
        cv2_module.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        caption = f"{detection.label} {detection.confidence:.2f}"
        (text_width, text_height), baseline = cv2_module.getTextSize(
            caption, FONT, scale, thickness
        )
        # Place the caption above the box, or inside it when the box touches
        # the top edge of the frame.
        label_bottom = y1 if y1 - text_height - baseline >= 0 else min(height, y2)
        label_top = label_bottom - text_height - baseline
        label_right = min(width, x1 + text_width + 6)
        cv2_module.rectangle(
            frame, (x1, label_top), (label_right, label_bottom), color, cv2_module.FILLED
        )
        cv2_module.putText(
            frame,
            caption,
            (x1 + 3, label_bottom - baseline + 1),
            FONT,
            scale,
            TEXT_COLOR,
            thickness,
            cv2_module.LINE_AA,
        )
    return frame


def draw_counts(
    frame: np.ndarray,
    counts: Mapping[str, int],
    fps: float,
    *,
    extra_lines: Sequence[str] = (),
    cv2_module: Any = cv2,
    max_rows: int = 8,
) -> np.ndarray:
    """Draw the per-class count panel and FPS readout, in place.

    Mirrors the roadmap's example output: one `Class: N` row per detected class,
    then a blank line, then FPS.
    """
    height, width = frame.shape[:2]
    scale = _text_scale(width)
    thickness = 1 if scale < 0.6 else 2
    line_height = int(round(26 * scale / 0.6))

    rows = [f"{label}: {count}" for label, count in list(counts.items())[:max_rows]]
    hidden = max(0, len(counts) - max_rows)
    if hidden:
        rows.append(f"(+{hidden} more class(es))")
    if not rows:
        rows.append("no objects above threshold")
    rows.append(f"FPS: {fps:.1f}")
    rows.extend(extra_lines)

    longest = max(
        cv2_module.getTextSize(row, FONT, scale, thickness)[0][0] for row in rows
    )
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
