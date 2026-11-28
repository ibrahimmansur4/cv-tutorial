#!/usr/bin/env python3
"""Build a synthetic test clip with scripted blinks and one long eye closure.

Testing a drowsiness detector needs footage of someone falling asleep, which is
awkward to produce on demand and impossible in CI. This script fakes it: it
takes one portrait photo and animates the eyelids closing by stretching the
brow band down over the eyes.

The result is **not** a realistic face and is no substitute for testing on real
footage of a real person. What it is good for is checking that the pipeline end
to end -- landmark fitting, EAR, the state machine, the alarm -- reacts to a
closure of a known duration at a known moment, which is exactly what a
regression check needs.

The clip's script is printed on exit so the detector's report can be compared
against the ground truth that produced it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "assets" / "input" / "portrait.jpg"
DEFAULT_OUTPUT = PROJECT_ROOT / "assets" / "input" / "drowsy-demo.avi"

# (eyelid closure 0..1, duration in seconds)
DEFAULT_SCRIPT: tuple[tuple[float, float], ...] = (
    (0.0, 2.0),
    (0.85, 0.15),  # blink
    (0.0, 2.0),
    (0.85, 0.15),  # blink
    (0.0, 1.5),
    (0.85, 2.5),  # sustained closure: this is the one that must alarm
    (0.0, 2.0),
)


def close_eyes(frame: np.ndarray, amount: float, band: tuple[int, int]) -> np.ndarray:
    """Stretch the brow band down over the eyes to mimic a closing lid."""
    if amount <= 0.0:
        return frame.copy()
    top, bottom = band
    height = bottom - top
    if height <= 2 or top < 16 or bottom >= frame.shape[0]:
        raise ValueError("eye band is outside the frame")
    output = frame.copy()
    keep = max(2, int(height * (1.0 - amount)))
    lid = cv2.resize(
        frame[top - 14 : top + 6], (frame.shape[1], height - keep), interpolation=cv2.INTER_LINEAR
    )
    rest = cv2.resize(
        frame[bottom - keep : bottom], (frame.shape[1], keep), interpolation=cv2.INTER_LINEAR
    )
    output[top:bottom] = np.vstack([lid, rest])
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_INPUT, help="portrait to animate")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--size",
        default="640x800",
        help="output frame size as WIDTHxHEIGHT (default: 640x800)",
    )
    parser.add_argument(
        "--eye-band",
        default="118x160",
        help="rows holding the eyes after resizing, as TOPxBOTTOM (default: 118x160)",
    )
    args = parser.parse_args(argv)

    source = args.image.expanduser().resolve()
    frame = cv2.imread(str(source))
    if frame is None:
        print(
            f"error: could not read {source}. Pass --image with a front-facing portrait.",
            file=sys.stderr,
        )
        return 2
    try:
        width, height = (int(piece) for piece in args.size.lower().split("x"))
        top, bottom = (int(piece) for piece in args.eye_band.lower().split("x"))
    except ValueError:
        print("error: --size and --eye-band must look like 640x800", file=sys.stderr)
        return 2
    if args.fps <= 0:
        print("error: --fps must be positive", file=sys.stderr)
        return 2

    frame = cv2.resize(frame, (width, height))
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"MJPG"), args.fps, (width, height)
    )
    if not writer.isOpened():
        writer.release()
        print(f"error: could not open {output} for writing", file=sys.stderr)
        return 2

    blinks = 0
    closures = 0
    try:
        for amount, seconds in DEFAULT_SCRIPT:
            try:
                rendered = close_eyes(frame, amount, (top, bottom))
            except ValueError as exc:
                print(f"error: {exc}; adjust --eye-band", file=sys.stderr)
                return 2
            if amount > 0.0:
                if seconds <= 0.4:
                    blinks += 1
                else:
                    closures += 1
            for _ in range(max(1, int(round(seconds * args.fps)))):
                writer.write(rendered)
    finally:
        writer.release()

    total = sum(seconds for _, seconds in DEFAULT_SCRIPT)
    print(f"Wrote {output}")
    print(
        f"Ground truth: {total:.1f}s at {args.fps} FPS, "
        f"{blinks} blink(s) and {closures} sustained closure(s) of "
        f"{max(seconds for amount, seconds in DEFAULT_SCRIPT if amount > 0):.1f}s."
    )
    print("The synthetic face is crude; use it to check behaviour, not accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
