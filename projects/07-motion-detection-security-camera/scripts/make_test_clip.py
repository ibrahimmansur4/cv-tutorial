#!/usr/bin/env python3
"""Generate a synthetic surveillance clip with known motion events.

Testing a motion detector against a webcam means standing in front of it and
hoping, which is neither repeatable nor possible in CI. This renders a static
"scene" with gentle sensor noise and walks a blob across it at scripted times,
then prints the ground truth so a detector run can be checked against it.

It also optionally simulates the two things that break naive frame differencing:
a sudden light change, and a swaying background object.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "assets" / "input" / "surveillance-demo.avi"

# (start_seconds, duration_seconds) for each walk-through.
WALKS: tuple[tuple[float, float], ...] = ((3.0, 2.5), (10.0, 3.0))
LIGHT_CHANGE_AT = 7.0
LIGHT_CHANGE_FOR = 1.0


def build_scene(width: int, height: int) -> np.ndarray:
    """A plausible static backdrop: floor, wall, and a couple of fixtures."""
    scene = np.full((height, width, 3), 70, dtype=np.uint8)
    cv2.rectangle(scene, (0, int(height * 0.62)), (width, height), (96, 96, 102), -1)
    cv2.rectangle(scene, (int(width * 0.05), int(height * 0.18)),
                  (int(width * 0.28), int(height * 0.60)), (130, 125, 115), -1)
    cv2.rectangle(scene, (int(width * 0.70), int(height * 0.26)),
                  (int(width * 0.92), int(height * 0.52)), (60, 80, 120), -1)
    cv2.circle(scene, (int(width * 0.5), int(height * 0.14)), int(height * 0.05), (180, 180, 170), -1)
    return scene


def draw_walker(frame: np.ndarray, progress: float) -> None:
    """A person-sized blob crossing left to right."""
    height, width = frame.shape[:2]
    x = int(width * (0.08 + 0.84 * progress))
    ground = int(height * 0.74)
    body_h = int(height * 0.30)
    body_w = int(body_h * 0.34)
    cv2.rectangle(
        frame, (x - body_w // 2, ground - body_h), (x + body_w // 2, ground), (48, 52, 60), -1
    )
    cv2.circle(frame, (x, ground - body_h - body_w // 3), body_w // 2, (70, 90, 110), -1)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seconds", type=float, default=16.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--size", default="640x480", help="WIDTHxHEIGHT (default: 640x480)")
    parser.add_argument("--noise", type=float, default=3.0, help="sensor noise sigma")
    parser.add_argument(
        "--light-change", action="store_true", help="add a sudden brightness jump"
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    if args.fps <= 0 or args.seconds <= 0:
        print("error: --fps and --seconds must be positive", file=sys.stderr)
        return 2
    try:
        width, height = (int(piece) for piece in args.size.lower().split("x"))
    except ValueError:
        print("error: --size must look like 640x480", file=sys.stderr)
        return 2

    rng = np.random.default_rng(args.seed)
    scene = build_scene(width, height)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"MJPG"), args.fps, (width, height)
    )
    if not writer.isOpened():
        writer.release()
        print(f"error: could not open {output} for writing", file=sys.stderr)
        return 2

    total = int(args.seconds * args.fps)
    try:
        for index in range(total):
            now = index / args.fps
            frame = scene.copy()
            for start, duration in WALKS:
                if start <= now < start + duration:
                    draw_walker(frame, (now - start) / duration)
            if args.light_change and LIGHT_CHANGE_AT <= now < LIGHT_CHANGE_AT + LIGHT_CHANGE_FOR:
                frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=45)
            # Per-frame sensor noise, which is what makes the blur step matter.
            noise = rng.normal(0.0, args.noise, frame.shape)
            writer.write(np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8))
    finally:
        writer.release()

    print(f"Wrote {output}")
    print(f"Ground truth: {args.seconds:.0f}s at {args.fps} FPS, {len(WALKS)} walk-through(s):")
    for order, (start, duration) in enumerate(WALKS, start=1):
        print(f"  walk {order}: {start:.1f}s -> {start + duration:.1f}s ({duration:.1f}s)")
    if args.light_change:
        print(
            f"  light change at {LIGHT_CHANGE_AT:.1f}s for {LIGHT_CHANGE_FOR:.1f}s "
            "(a false positive for naive differencing)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
