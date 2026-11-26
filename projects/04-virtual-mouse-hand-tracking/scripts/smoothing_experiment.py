#!/usr/bin/env python3
"""Compare raw cursor jitter with several exponential smoothing factors."""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path
from statistics import fmean
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = PROJECT_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from src.geometry import Point  # noqa: E402
from src.smoothing import ExponentialCursorSmoother  # noqa: E402


def parse_alphas(value: str) -> tuple[float, ...]:
    try:
        alphas = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("alphas must be comma-separated numbers") from exc
    if not alphas or any(not math.isfinite(alpha) or not 0.0 < alpha <= 1.0 for alpha in alphas):
        raise argparse.ArgumentTypeError("every alpha must be above 0 and at most 1")
    return alphas


def rms_jitter(points: Sequence[Point]) -> float:
    """Return RMS distance from the observed centroid in pixels."""

    center_x = fmean(point.x for point in points)
    center_y = fmean(point.y for point in points)
    return math.sqrt(
        fmean((point.x - center_x) ** 2 + (point.y - center_y) ** 2 for point in points)
    )


def experiment(
    *,
    frames: int = 600,
    noise_sigma: float = 5.0,
    alphas: Sequence[float] = (0.15, 0.35, 0.65, 1.0),
    seed: int = 7,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    if frames < 100:
        raise ValueError("frames must be at least 100")
    if not math.isfinite(noise_sigma) or noise_sigma <= 0.0:
        raise ValueError("noise sigma must be positive")

    randomizer = random.Random(seed)
    smoothers = {alpha: ExponentialCursorSmoother(alpha, dead_zone=0.0) for alpha in alphas}
    rows: list[dict[str, float]] = []
    for frame in range(frames):
        raw = Point(
            960.0 + randomizer.gauss(0.0, noise_sigma),
            540.0 + randomizer.gauss(0.0, noise_sigma),
        )
        row = {"frame": float(frame), "raw_x": raw.x, "raw_y": raw.y}
        for alpha, smoother in smoothers.items():
            smoothed = smoother.update(raw).point
            label = f"alpha_{alpha:g}"
            row[f"{label}_x"] = smoothed.x
            row[f"{label}_y"] = smoothed.y
        rows.append(row)

    # Ignore startup so the comparison measures steady-state noise instead of filter warm-up.
    steady = rows[50:]
    metrics = {
        "raw": rms_jitter([Point(row["raw_x"], row["raw_y"]) for row in steady])
    }
    for alpha in alphas:
        label = f"alpha_{alpha:g}"
        metrics[label] = rms_jitter(
            [Point(row[f"{label}_x"], row[f"{label}_y"]) for row in steady]
        )
    return rows, metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--noise-sigma", type=float, default=5.0)
    parser.add_argument("--alphas", type=parse_alphas, default=(0.15, 0.35, 0.65, 1.0))
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "smoothing_comparison.csv",
    )
    args = parser.parse_args(argv)
    try:
        rows, metrics = experiment(
            frames=args.frames,
            noise_sigma=args.noise_sigma,
            alphas=args.alphas,
        )
    except ValueError as exc:
        parser.error(str(exc))

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("Steady-state RMS cursor jitter (lower is better)")
    for label, value in metrics.items():
        print(f"  {label:>10}: {value:6.3f} px")
    print(f"CSV: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
