"""Measure Haar vs DNN detector throughput across input sizes.

Run from projects/01-real-time-face-detection:

    python scripts/benchmark.py

Times only ``detector.detect(frame)`` -- no capture, no drawing, no encoding.
Writes a markdown table to results/benchmark.md and prints it to stdout.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))

import cv2 as cv  # noqa: E402
import numpy as np  # noqa: E402

from src.detectors import DnnFaceDetector, HaarFaceDetector  # noqa: E402

SIZES = [(320, 240), (480, 480), (640, 480), (800, 600), (1280, 720), (1254, 1254)]
WARMUP = 2
REPEATS = 9


def time_detector(detector, frame, repeats=REPEATS, warmup=WARMUP):
    """Return the median wall-clock milliseconds of one detect() call."""

    for _ in range(warmup):
        detector.detect(frame)
    samples = []
    for _ in range(repeats):
        started_at = time.perf_counter()
        detector.detect(frame)
        samples.append((time.perf_counter() - started_at) * 1000.0)
    return statistics.median(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=PROJECT_ROOT / "assets/input/sample-face.png"
    )
    parser.add_argument(
        "--dnn-config", type=Path, default=PROJECT_ROOT / "models/deploy.prototxt"
    )
    parser.add_argument(
        "--dnn-model",
        type=Path,
        default=PROJECT_ROOT
        / "models/res10_300x300_ssd_iter_140000_fp16.caffemodel",
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/benchmark.md")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    args = parser.parse_args()

    source = cv.imread(str(args.input), cv.IMREAD_COLOR)
    if source is None:
        print(f"error: could not read {args.input}", file=sys.stderr)
        return 2

    haar = HaarFaceDetector()
    dnn = DnnFaceDetector(args.dnn_config, args.dnn_model, confidence_threshold=0.5)

    lines = [
        f"# Benchmark: {args.input.name}",
        "",
        f"- CPU: {platform.processor() or 'unknown'}",
        f"- OS: {platform.platform()}",
        f"- Python: {platform.python_version()}",
        f"- OpenCV: {cv.__version__}, NumPy: {np.__version__}",
        f"- Protocol: {WARMUP} warm-up calls, median of {args.repeats} timed detect() calls",
        "",
        "| Input size | Haar (ms) | Haar FPS | DNN (ms) | DNN FPS | Haar faces | DNN faces |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for width, height in SIZES:
        frame = cv.resize(source, (width, height), interpolation=cv.INTER_AREA)
        haar_ms = time_detector(haar, frame, args.repeats)
        dnn_ms = time_detector(dnn, frame, args.repeats)
        lines.append(
            f"| {width}x{height} | {haar_ms:.1f} | {1000.0 / haar_ms:.1f} "
            f"| {dnn_ms:.1f} | {1000.0 / dnn_ms:.1f} "
            f"| {len(haar.detect(frame))} | {len(dnn.detect(frame))} |"
        )
        print(lines[-1])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
