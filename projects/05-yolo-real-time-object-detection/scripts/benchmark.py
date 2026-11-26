#!/usr/bin/env python3
"""Benchmark YOLO model sizes and input resolutions on this machine.

Two sweeps, because the registered exports differ in what they allow:

* **Model size** - yolov5n/s/m at their native 640x640. One architecture at
  three widths, so speed and accuracy change but nothing else does.
* **Input resolution** - yolo11n at several sizes. It is the only registered
  export with dynamic input axes; the YOLOv5 ONNX files carry baked-in 640x640
  reshape dimensions and fail at any other size.

Detection *quality* has no ground truth here, so it is reported indirectly: the
object count and mean confidence at a fixed threshold. A configuration finding
more objects at similar confidence is finding more of what is present.

Results go to `results/benchmark.csv`, with Markdown tables on stdout.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from src.detector import YoloDetector  # noqa: E402
from src.errors import ConfigurationError, DetectionError  # noqa: E402
from src.models import BENCHMARK_LADDER, RESOLUTION_MODEL, get_spec  # noqa: E402

DEFAULT_IMAGE = PROJECT_ROOT / "assets" / "input" / "bus.jpg"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "benchmark.csv"
DEFAULT_SIZES = (320, 480, 640, 960)
CSV_FIELDS = (
    "sweep",
    "model",
    "input_size",
    "frames",
    "fps",
    "mean_total_ms",
    "median_total_ms",
    "preprocess_ms",
    "inference_ms",
    "postprocess_ms",
    "objects",
    "mean_confidence",
    "candidates",
    "cpu_percent",
)


@dataclass(slots=True)
class BenchmarkRow:
    sweep: str
    model: str
    input_size: int
    frames: int
    totals_ms: list[float] = field(default_factory=list)
    preprocess_ms: list[float] = field(default_factory=list)
    inference_ms: list[float] = field(default_factory=list)
    postprocess_ms: list[float] = field(default_factory=list)
    objects: float = 0.0
    confidences: list[float] = field(default_factory=list)
    candidates: float = 0.0
    cpu_percent: float = float("nan")

    @property
    def mean_total_ms(self) -> float:
        return statistics.fmean(self.totals_ms) if self.totals_ms else 0.0

    @property
    def fps(self) -> float:
        mean = self.mean_total_ms
        return 1000.0 / mean if mean > 0.0 else 0.0

    @property
    def mean_confidence(self) -> float:
        return statistics.fmean(self.confidences) if self.confidences else 0.0

    def _mean(self, samples: list[float]) -> float:
        return round(statistics.fmean(samples), 2) if samples else 0.0

    def as_record(self) -> dict[str, object]:
        return {
            "sweep": self.sweep,
            "model": self.model,
            "input_size": self.input_size,
            "frames": self.frames,
            "fps": round(self.fps, 2),
            "mean_total_ms": round(self.mean_total_ms, 2),
            "median_total_ms": (
                round(statistics.median(self.totals_ms), 2) if self.totals_ms else 0.0
            ),
            "preprocess_ms": self._mean(self.preprocess_ms),
            "inference_ms": self._mean(self.inference_ms),
            "postprocess_ms": self._mean(self.postprocess_ms),
            "objects": round(self.objects, 2),
            "mean_confidence": round(self.mean_confidence, 3),
            "candidates": round(self.candidates, 1),
            "cpu_percent": round(self.cpu_percent, 1),
        }


def _cpu_probe() -> tuple[object, int] | None:
    """Return a primed psutil handle and the logical core count, or None.

    CPU utilisation is a nice-to-have, so a missing optional dependency must
    not stop the benchmark from producing timings.
    """
    try:
        import psutil
    except ImportError:
        return None
    try:
        process = psutil.Process()
        process.cpu_percent(None)  # prime the measurement interval
        return process, max(1, psutil.cpu_count() or 1)
    except Exception:
        return None


def benchmark_one(
    sweep: str,
    model_name: str,
    input_size: int,
    frame: np.ndarray,
    *,
    frames: int,
    confidence: float,
    nms: float,
) -> BenchmarkRow:
    spec = get_spec(model_name)
    detector = YoloDetector(
        spec.path,
        spec.family,
        input_size=(input_size, input_size),
        confidence_threshold=confidence,
        nms_threshold=nms,
    )
    # Warm up outside the measured loop: the first pass pays for lazy buffer
    # allocation and would otherwise dominate a short run.
    detector.warm_up()

    row = BenchmarkRow(sweep=sweep, model=model_name, input_size=input_size, frames=frames)
    probe = _cpu_probe()
    cores = probe[1] if probe is not None else 1
    total_objects = 0
    total_candidates = 0
    for _ in range(frames):
        result = detector.detect(frame)
        row.totals_ms.append(result.total_ms)
        row.preprocess_ms.append(result.preprocess_ms)
        row.inference_ms.append(result.inference_ms)
        row.postprocess_ms.append(result.postprocess_ms)
        total_objects += len(result.detections)
        total_candidates += result.candidates
        row.confidences.extend(detection.confidence for detection in result.detections)
    if probe is not None:
        try:
            # psutil sums across cores, so a busy 8-core machine reports ~800%.
            # Divide by the core count to get a share of the whole machine.
            row.cpu_percent = probe[0].cpu_percent(None) / cores
        except Exception:
            row.cpu_percent = float("nan")
    # Report per-frame figures so they do not grow with --frames.
    row.objects = total_objects / frames
    row.candidates = total_candidates / frames
    return row


def markdown_table(rows: Sequence[BenchmarkRow], *, first_column: str) -> str:
    header = (
        f"| {first_column} | FPS | Total ms | Pre | Net | Post | Objects | Mean conf | CPU % |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for row in rows:
        record = row.as_record()
        cpu = record["cpu_percent"]
        cpu_text = "n/a" if isinstance(cpu, float) and cpu != cpu else f"{cpu}"
        leading = row.model if first_column == "Model" else f"{row.input_size}x{row.input_size}"
        lines.append(
            f"| {leading} | {record['fps']} | {record['mean_total_ms']} "
            f"| {record['preprocess_ms']} | {record['inference_ms']} "
            f"| {record['postprocess_ms']} | {record['objects']} "
            f"| {record['mean_confidence']} | {cpu_text} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default=",".join(BENCHMARK_LADDER),
        help=f"model-size sweep, comma separated (default: {','.join(BENCHMARK_LADDER)})",
    )
    parser.add_argument(
        "--resolution-model",
        default=RESOLUTION_MODEL,
        help=f"model used for the resolution sweep (default: {RESOLUTION_MODEL})",
    )
    parser.add_argument(
        "--sizes",
        default=",".join(str(size) for size in DEFAULT_SIZES),
        help=f"resolution sweep sizes (default: {','.join(map(str, DEFAULT_SIZES))})",
    )
    parser.add_argument(
        "--image", type=Path, default=DEFAULT_IMAGE, help="image to run repeatedly"
    )
    parser.add_argument("--frames", type=int, default=20, help="frames per configuration")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms", type=float, default=0.45)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-models", action="store_true", help="skip the model-size sweep")
    parser.add_argument("--skip-sizes", action="store_true", help="skip the resolution sweep")
    args = parser.parse_args(argv)

    if args.frames <= 0:
        print("error: --frames must be positive", file=sys.stderr)
        return 2

    image_path = args.image.expanduser().resolve()
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(
            f"error: could not read benchmark image: {image_path}. "
            "Pass --image with a photo containing a few objects.",
            file=sys.stderr,
        )
        return 2

    try:
        models = [name.strip() for name in args.models.split(",") if name.strip()]
        sizes = [int(size.strip()) for size in args.sizes.split(",") if size.strip()]
    except ValueError:
        print("error: --sizes must be a comma-separated list of integers", file=sys.stderr)
        return 2

    model_rows: list[BenchmarkRow] = []
    size_rows: list[BenchmarkRow] = []
    try:
        if not args.skip_models:
            for model_name in models:
                spec = get_spec(model_name)
                native = spec.input_size[0]
                print(f"model sweep: {model_name} at {native}x{native} ...", file=sys.stderr)
                model_rows.append(
                    benchmark_one(
                        "model",
                        model_name,
                        native,
                        frame,
                        frames=args.frames,
                        confidence=args.confidence,
                        nms=args.nms,
                    )
                )
        if not args.skip_sizes:
            spec = get_spec(args.resolution_model)
            if not spec.dynamic_input:
                print(
                    f"error: {spec.name} has a fixed input size and cannot be swept across "
                    "resolutions. Pass --resolution-model with a dynamic-axis export.",
                    file=sys.stderr,
                )
                return 2
            for size in sizes:
                print(f"size sweep: {spec.name} at {size}x{size} ...", file=sys.stderr)
                size_rows.append(
                    benchmark_one(
                        "resolution",
                        spec.name,
                        size,
                        frame,
                        frames=args.frames,
                        confidence=args.confidence,
                        nms=args.nms,
                    )
                )
    except (DetectionError, ConfigurationError, cv2.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    rows = model_rows + size_rows
    if not rows:
        print("error: both sweeps were skipped; nothing to measure", file=sys.stderr)
        return 2

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_record())

    print(
        f"\nImage: {image_path.name} {frame.shape[1]}x{frame.shape[0]}, "
        f"{args.frames} frames per configuration, confidence {args.confidence}"
    )
    if model_rows:
        print("\n### Model size (native 640x640 input)\n")
        print(markdown_table(model_rows, first_column="Model"))
    if size_rows:
        print(f"\n### Input resolution ({size_rows[0].model})\n")
        print(markdown_table(size_rows, first_column="Input"))
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
