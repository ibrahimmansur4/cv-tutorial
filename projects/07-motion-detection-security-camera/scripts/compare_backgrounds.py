#!/usr/bin/env python3
"""Score the background models against a clip with known motion windows.

`make_test_clip.py` renders motion at times it knows, so a run over that clip
can be graded frame by frame rather than eyeballed. Each frame is a prediction
("was motion reported?") against a label ("was anything actually moving?"), which
gives the usual counts:

* **Recall** - of the frames that really contained motion, how many were caught.
  Low recall means a real intruder goes unrecorded.
* **Precision** - of the frames reported as motion, how many really were.
  Low precision means a disk full of clips of clouds.

The two trade against each other and the right balance depends on the cost of
each mistake, which is why this prints both rather than one score. Warm-up
frames are excluded from the grading, since the models are told not to report
during them.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_test_clip import WALKS  # noqa: E402

from src.background import make_background  # noqa: E402
from src.detector import MotionDetector  # noqa: E402
from src.errors import SecurityCameraError  # noqa: E402

DEFAULT_CLIP = PROJECT_ROOT / "assets" / "input" / "surveillance-demo.avi"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "background_comparison.csv"
CSV_FIELDS = (
    "model",
    "frames",
    "true_positives",
    "false_positives",
    "false_negatives",
    "precision",
    "recall",
    "f1",
    "fps",
    "mean_ms",
)


@dataclass(slots=True)
class Score:
    model: str
    frames: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    durations_ms: list[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.durations_ms is None:
            self.durations_ms = []

    @property
    def precision(self) -> float:
        predicted = self.true_positives + self.false_positives
        return self.true_positives / predicted if predicted else 0.0

    @property
    def recall(self) -> float:
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.durations_ms) if self.durations_ms else 0.0

    @property
    def fps(self) -> float:
        return 1000.0 / self.mean_ms if self.mean_ms > 0.0 else 0.0

    def as_record(self) -> dict[str, object]:
        return {
            "model": self.model,
            "frames": self.frames,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "fps": round(self.fps, 1),
            "mean_ms": round(self.mean_ms, 2),
        }


def parse_truth(text: str) -> tuple[tuple[float, float], ...]:
    """Parse `start:end,start:end` into labelled motion windows."""
    windows: list[tuple[float, float]] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        start, _, end = chunk.partition(":")
        windows.append((float(start), float(end)))
    return tuple(windows)


def score_model(
    name: str,
    clip: Path,
    truth: Sequence[tuple[float, float]],
    *,
    min_area: float,
    blur: int,
    warmup: int,
    tolerance: float,
) -> Score:
    capture = cv2.VideoCapture(str(clip))
    if not capture.isOpened():
        capture.release()
        raise SecurityCameraError(f"could not open clip: {clip}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 20.0
    detector = MotionDetector(
        make_background(name, warmup_frames=warmup), blur_kernel=blur, min_area=min_area
    )
    score = Score(model=name)
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            now = index / fps
            index += 1
            started = time.perf_counter()
            result = detector.detect(frame)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if not result.ready:
                continue
            score.durations_ms.append(elapsed_ms)
            score.frames += 1
            # A window is widened by `tolerance` because a subject entering or
            # leaving the frame is genuinely ambiguous for a frame or two.
            actual = any(
                start - tolerance <= now <= end + tolerance for start, end in truth
            )
            predicted = result.detected
            if predicted and actual:
                score.true_positives += 1
            elif predicted and not actual:
                score.false_positives += 1
            elif actual and not predicted:
                score.false_negatives += 1
    finally:
        capture.release()
    return score


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_CLIP)
    parser.add_argument(
        "--truth",
        default=",".join(f"{start}:{start + duration}" for start, duration in WALKS),
        help="motion windows as start:end,start:end (defaults to make_test_clip's script)",
    )
    parser.add_argument(
        "--models", default="running-average,mog2", help="comma-separated model names"
    )
    parser.add_argument("--min-area", type=float, default=500.0)
    parser.add_argument("--blur", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.2,
        help="seconds of slack at each window edge (default: 0.2)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    clip = args.input.expanduser().resolve()
    if not clip.is_file():
        print(
            f"error: clip not found: {clip}. Run `python scripts/make_test_clip.py` first.",
            file=sys.stderr,
        )
        return 2
    truth = parse_truth(args.truth)
    if not truth:
        print("error: --truth needs at least one start:end window", file=sys.stderr)
        return 2

    scores: list[Score] = []
    for name in (part.strip() for part in args.models.split(",") if part.strip()):
        print(f"scoring {name} ...", file=sys.stderr)
        try:
            scores.append(
                score_model(
                    name,
                    clip,
                    truth,
                    min_area=args.min_area,
                    blur=args.blur,
                    warmup=args.warmup,
                    tolerance=args.tolerance,
                )
            )
        except (SecurityCameraError, cv2.error) as exc:
            print(f"error: {name}: {exc}", file=sys.stderr)
            return 2

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for score in scores:
            writer.writerow(score.as_record())

    windows = ", ".join(f"{start:.1f}-{end:.1f}s" for start, end in truth)
    print(f"\nClip: {clip.name}   labelled motion: {windows}\n")
    print("| Model | Precision | Recall | F1 | TP | FP | FN | ms/frame | FPS |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for score in scores:
        print(
            f"| {score.model} | {score.precision:.3f} | {score.recall:.3f} | {score.f1:.3f} "
            f"| {score.true_positives} | {score.false_positives} | {score.false_negatives} "
            f"| {score.mean_ms:.2f} | {score.fps:.0f} |"
        )
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
