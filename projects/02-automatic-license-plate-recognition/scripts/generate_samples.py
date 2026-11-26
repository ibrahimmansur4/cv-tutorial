"""Generate deterministic image and video fixtures for the ALPR tutorial.

The fixtures contain a synthetic car with a high-contrast, mildly perspective-
distorted plate. They are intentionally simple enough for the classical OpenCV
pipeline while still exercising contour filtering, rectification, OCR, and
video tracking.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "assets" / "input"
IMAGE_NAME = "synthetic_vehicle.png"
VIDEO_NAME = "synthetic_drive.avi"
PLATE_TEXT = "CV 2026"


def positive_integer(value: str) -> int:
    """Parse a strictly positive integer for argparse."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def positive_float(value: str) -> float:
    """Parse a finite positive floating-point value for argparse."""

    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a finite number above zero")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic image/video inputs for Project 2."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="destination directory (default: project assets/input)",
    )
    parser.add_argument(
        "--frames",
        type=positive_integer,
        default=72,
        help="number of video frames (default: 72)",
    )
    parser.add_argument(
        "--fps",
        type=positive_float,
        default=24.0,
        help="video frame rate (default: 24)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace fixtures that already exist",
    )
    return parser


def _sky_gradient(width: int, height: int) -> np.ndarray:
    top = np.array([184, 128, 76], dtype=np.float32)
    bottom = np.array([235, 210, 172], dtype=np.float32)
    blend = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    rows = ((1.0 - blend) * top + blend * bottom).astype(np.uint8)
    return np.repeat(rows[:, None, :], width, axis=1)


def _draw_background(width: int, height: int) -> np.ndarray:
    horizon = int(height * 0.47)
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:horizon] = _sky_gradient(width, horizon)
    frame[horizon:] = (56, 58, 62)

    # A deterministic skyline adds realistic edges without resembling a plate.
    building_colors = ((92, 96, 100), (112, 110, 106), (78, 85, 91))
    building_specs = (
        (0.03, 0.20, 0.13, 0.47),
        (0.15, 0.27, 0.27, 0.47),
        (0.70, 0.24, 0.82, 0.47),
        (0.84, 0.17, 0.97, 0.47),
    )
    for index, (x1, y1, x2, y2) in enumerate(building_specs):
        cv2.rectangle(
            frame,
            (int(width * x1), int(height * y1)),
            (int(width * x2), int(height * y2)),
            building_colors[index % len(building_colors)],
            -1,
        )

    cv2.line(
        frame,
        (0, horizon),
        (width - 1, horizon),
        (35, 38, 42),
        max(2, height // 180),
    )
    lane_y = int(height * 0.84)
    dash_width = max(24, width // 11)
    gap = max(18, width // 18)
    x = -dash_width // 2
    while x < width:
        cv2.rectangle(
            frame,
            (x, lane_y),
            (x + dash_width, lane_y + max(4, height // 70)),
            (218, 210, 176),
            -1,
        )
        x += dash_width + gap
    return frame


def _make_plate_template() -> np.ndarray:
    width, height = 360, 100
    plate = np.full((height, width, 3), (238, 242, 244), dtype=np.uint8)
    cv2.rectangle(plate, (3, 3), (width - 4, height - 4), (18, 18, 18), 6)
    cv2.circle(plate, (20, height // 2), 4, (155, 155, 155), -1)
    cv2.circle(plate, (width - 21, height // 2), 4, (155, 155, 155), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.75
    thickness = 4
    (text_width, text_height), baseline = cv2.getTextSize(
        PLATE_TEXT, font, font_scale, thickness
    )
    origin = ((width - text_width) // 2, (height + text_height - baseline) // 2)
    cv2.putText(
        plate,
        PLATE_TEXT,
        origin,
        font,
        font_scale,
        (12, 12, 12),
        thickness,
        cv2.LINE_AA,
    )
    return plate


def _place_plate(frame: np.ndarray, plate: np.ndarray, corners: np.ndarray) -> None:
    plate_height, plate_width = plate.shape[:2]
    source = np.float32(
        [
            [0, 0],
            [plate_width - 1, 0],
            [plate_width - 1, plate_height - 1],
            [0, plate_height - 1],
        ]
    )
    transform = cv2.getPerspectiveTransform(source, corners.astype(np.float32))
    output_size = (frame.shape[1], frame.shape[0])
    warped_plate = cv2.warpPerspective(
        plate,
        transform,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    mask = cv2.warpPerspective(
        np.full((plate_height, plate_width), 255, dtype=np.uint8),
        transform,
        output_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    frame[mask > 0] = warped_plate[mask > 0]


def _draw_vehicle(frame: np.ndarray, x: int, y: int, width: int, height: int) -> None:
    body_color = (52, 65, 177)
    dark_red = (34, 43, 116)
    glass_color = (102, 87, 72)
    outline = (24, 25, 31)

    body_top = y + int(height * 0.28)
    body_bottom = y + int(height * 0.88)
    body = np.array(
        [
            [x + int(width * 0.03), body_top + int(height * 0.15)],
            [x + int(width * 0.16), body_top],
            [x + int(width * 0.84), body_top],
            [x + int(width * 0.97), body_top + int(height * 0.15)],
            [x + int(width * 0.94), body_bottom],
            [x + int(width * 0.06), body_bottom],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(frame, body, body_color, lineType=cv2.LINE_AA)
    cv2.polylines(frame, [body], True, outline, max(2, height // 70), cv2.LINE_AA)

    roof = np.array(
        [
            [x + int(width * 0.25), body_top],
            [x + int(width * 0.35), y + int(height * 0.04)],
            [x + int(width * 0.66), y + int(height * 0.04)],
            [x + int(width * 0.78), body_top],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(frame, roof, dark_red, lineType=cv2.LINE_AA)
    cv2.polylines(frame, [roof], True, outline, max(2, height // 80), cv2.LINE_AA)

    rear_window = np.array(
        [
            [x + int(width * 0.31), body_top - 2],
            [x + int(width * 0.39), y + int(height * 0.10)],
            [x + int(width * 0.62), y + int(height * 0.10)],
            [x + int(width * 0.71), body_top - 2],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(frame, rear_window, glass_color, lineType=cv2.LINE_AA)
    cv2.polylines(frame, [rear_window], True, outline, 2, cv2.LINE_AA)

    light_y = y + int(height * 0.43)
    light_w = max(10, int(width * 0.09))
    light_h = max(6, int(height * 0.10))
    cv2.rectangle(
        frame,
        (x + int(width * 0.09), light_y),
        (x + int(width * 0.09) + light_w, light_y + light_h),
        (45, 45, 232),
        -1,
    )
    cv2.rectangle(
        frame,
        (x + int(width * 0.82), light_y),
        (x + int(width * 0.82) + light_w, light_y + light_h),
        (45, 45, 232),
        -1,
    )

    wheel_radius = max(8, int(height * 0.13))
    for wheel_x in (x + int(width * 0.18), x + int(width * 0.82)):
        cv2.circle(frame, (wheel_x, body_bottom), wheel_radius, outline, -1, cv2.LINE_AA)
        cv2.circle(
            frame,
            (wheel_x, body_bottom),
            max(3, wheel_radius // 2),
            (112, 116, 120),
            -1,
            cv2.LINE_AA,
        )

    plate_width = int(width * 0.47)
    plate_height = max(32, int(height * 0.27))
    center_x = x + width // 2
    top = y + int(height * 0.53)
    skew = max(2, plate_height // 9)
    corners = np.array(
        [
            [center_x - plate_width // 2 + skew, top],
            [center_x + plate_width // 2 - skew, top + 1],
            [center_x + plate_width // 2, top + plate_height],
            [center_x - plate_width // 2, top + plate_height - 1],
        ],
        dtype=np.float32,
    )
    _place_plate(frame, _make_plate_template(), corners)


def create_still_image() -> np.ndarray:
    width, height = 960, 540
    frame = _draw_background(width, height)
    car_width, car_height = 520, 205
    _draw_vehicle(frame, (width - car_width) // 2, 226, car_width, car_height)
    return frame


def create_video_frame(frame_index: int, frame_count: int) -> np.ndarray:
    width, height = 640, 360
    frame = _draw_background(width, height)
    progress = frame_index / max(frame_count - 1, 1)
    car_width, car_height = 330, 146
    start_x = 24
    end_x = width - car_width - 24
    x = int(round(start_x + (end_x - start_x) * progress))
    y = 150 + int(round(4.0 * math.sin(progress * math.tau)))
    _draw_vehicle(frame, x, y, car_width, car_height)
    return frame


def _ensure_writable_targets(paths: Sequence[Path], force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"fixture(s) already exist: {names}; pass --force to replace")


def generate_fixtures(
    output_directory: Path, *, frame_count: int, fps: float, force: bool
) -> tuple[Path, Path]:
    """Create the still image and MJPG video, returning their paths."""

    output_directory = output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    image_path = output_directory / IMAGE_NAME
    video_path = output_directory / VIDEO_NAME
    _ensure_writable_targets((image_path, video_path), force)

    if not cv2.imwrite(str(image_path), create_still_image()):
        raise RuntimeError(f"OpenCV could not write image: {image_path}")

    frame_size = (640, 360)
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, frame_size
    )
    if not writer.isOpened():
        raise RuntimeError(
            "OpenCV could not create the MJPG sample video; verify codec support"
        )
    try:
        for frame_index in range(frame_count):
            writer.write(create_video_frame(frame_index, frame_count))
    finally:
        writer.release()

    return image_path, video_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        image_path, video_path = generate_fixtures(
            args.output_dir,
            frame_count=args.frames,
            fps=args.fps,
            force=args.force,
        )
    except (OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Created image: {image_path}")
    print(f"Created video: {video_path} ({args.frames} frames at {args.fps:g} FPS)")
    print(f"Expected plate text: {PLATE_TEXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
