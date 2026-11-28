#!/usr/bin/env python3
"""Grade the whole chain against rendered puzzles with known digits.

Generates puzzle images under a range of photographic conditions -- tilt, blur,
noise, uneven lighting -- and for each one measures where the pipeline actually
fails, which is rarely where you would guess:

* **board found** - did the quadrilateral search locate the grid at all?
* **cell recall** - of the cells that hold a digit, how many were seen as
  occupied? A missed cell is silently treated as blank, which usually makes the
  puzzle ambiguous rather than wrong.
* **digit accuracy** - of the cells seen as occupied, how many were read
  correctly? One wrong digit makes the puzzle unsolvable or, worse, solvable to
  the wrong answer.
* **solved** - did the end-to-end run produce the correct completed grid?

Both digit classifiers are scored on identical images, so the comparison is
like for like.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_puzzle_image import (  # noqa: E402
    DEFAULT_PUZZLE,
    FONTS,
    board_from_string,
    photograph,
    render_board,
)

from src.classify import TemplateClassifier, make_classifier  # noqa: E402
from src.errors import ConfigurationError, SudokuError  # noqa: E402
from src.pipeline import SudokuPipeline  # noqa: E402
from src.solver import solve  # noqa: E402

DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "evaluation.csv"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "mnist-8.onnx"
CSV_FIELDS = (
    "classifier",
    "font",
    "condition",
    "images",
    "boards_found",
    "cell_recall",
    "digit_accuracy",
    "solved",
    "mean_ms",
)

# (name, kwargs for `photograph`)
CONDITIONS: tuple[tuple[str, dict], ...] = (
    ("clean", {}),
    ("tilt", {"tilt": 0.08}),
    ("blur", {"blur": 5}),
    ("noise", {"noise": 8.0}),
    ("lighting", {"lighting": 0.5}),
    ("combined", {"tilt": 0.06, "blur": 3, "noise": 6.0, "lighting": 0.45}),
)


@dataclass(slots=True)
class Score:
    classifier: str
    font: str
    condition: str
    images: int = 0
    boards_found: int = 0
    occupied_hits: int = 0
    occupied_total: int = 0
    digit_hits: int = 0
    digit_total: int = 0
    solved: int = 0
    durations_ms: list[float] = field(default_factory=list)

    @property
    def cell_recall(self) -> float:
        return self.occupied_hits / self.occupied_total if self.occupied_total else 0.0

    @property
    def digit_accuracy(self) -> float:
        return self.digit_hits / self.digit_total if self.digit_total else 0.0

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.durations_ms) if self.durations_ms else 0.0

    def as_record(self) -> dict[str, object]:
        return {
            "classifier": self.classifier,
            "font": self.font,
            "condition": self.condition,
            "images": self.images,
            "boards_found": self.boards_found,
            "cell_recall": round(self.cell_recall, 4),
            "digit_accuracy": round(self.digit_accuracy, 4),
            "solved": self.solved,
            "mean_ms": round(self.mean_ms, 1),
        }


def make_image(board, condition: dict, seed: int, font: int) -> np.ndarray:
    settings = {"tilt": 0.0, "blur": 0, "noise": 0.0, "lighting": 0.0}
    settings.update(condition)
    return photograph(render_board(board, font=font), seed=seed, **settings)


def evaluate(
    classifier_name: str,
    model_path: Path,
    puzzle: str,
    *,
    repeats: int,
    conditions: Sequence[tuple[str, dict]],
    fonts: Sequence[str],
) -> list[Score]:
    board = board_from_string(puzzle)
    expected_solution, _ = solve(board)
    classifier = make_classifier(classifier_name, model_path=model_path)
    pipeline = SudokuPipeline(classifier, check_unique=False)
    scores: list[Score] = []

    for font_name in fonts:
      for name, condition in conditions:
        score = Score(classifier=classifier_name, font=font_name, condition=name)
        for repeat in range(repeats):
            image = make_image(board, condition, 100 + repeat, FONTS[font_name])
            score.images += 1
            started = time.perf_counter()
            try:
                location, square, cells = pipeline.read_board(image)
            except SudokuError:
                score.durations_ms.append((time.perf_counter() - started) * 1000.0)
                continue
            score.boards_found += 1

            for cell in cells:
                truth = board[cell.row][cell.column]
                if truth:
                    score.occupied_total += 1
                    if cell.occupied:
                        score.occupied_hits += 1
                if cell.occupied and truth:
                    score.digit_total += 1
                    if classifier.predict(cell.patch).digit == truth:
                        score.digit_hits += 1

            try:
                outcome = pipeline.process(image)
                if [list(row) for row in outcome.solution] == expected_solution:
                    score.solved += 1
            except SudokuError:
                pass
            score.durations_ms.append((time.perf_counter() - started) * 1000.0)
        scores.append(score)
    return scores


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puzzle", default=DEFAULT_PUZZLE)
    parser.add_argument(
        "--classifiers", default="template", help="comma-separated: template,onnx"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--repeats", type=int, default=3, help="images per condition")
    parser.add_argument(
        "--fonts",
        default="simplex,plain,complex-small,italic",
        help="comma-separated fonts to render the test puzzles with",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    if args.repeats <= 0:
        print("error: --repeats must be positive", file=sys.stderr)
        return 2
    names = [part.strip() for part in args.classifiers.split(",") if part.strip()]
    if not names:
        print("error: --classifiers needs at least one name", file=sys.stderr)
        return 2
    fonts = [part.strip() for part in args.fonts.split(",") if part.strip()]
    unknown = [name for name in fonts if name not in FONTS]
    if unknown:
        print(
            f"error: unknown font(s): {', '.join(unknown)}. "
            f"Choose from {', '.join(sorted(FONTS))}.",
            file=sys.stderr,
        )
        return 2

    all_scores: list[Score] = []
    for name in names:
        print(f"evaluating {name} ...", file=sys.stderr)
        try:
            all_scores.extend(
                evaluate(
                    name,
                    args.model,
                    args.puzzle,
                    repeats=args.repeats,
                    conditions=CONDITIONS,
                    fonts=fonts,
                )
            )
        except (SudokuError, ConfigurationError, cv2.error) as exc:
            print(f"error: {name}: {exc}", file=sys.stderr)
            return 2

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for score in all_scores:
            writer.writerow(score.as_record())

    givens = sum(1 for row in board_from_string(args.puzzle) for value in row if value)
    print(f"\nPuzzle with {givens} givens, {args.repeats} image(s) per condition\n")
    print("| Classifier | Font | Condition | Board found | Cell recall | Digit accuracy | Solved | ms |")
    print("|---|---|---|---:|---:|---:|---:|---:|")
    for score in all_scores:
        print(
            f"| {score.classifier} | {score.font} | {score.condition} "
            f"| {score.boards_found}/{score.images} "
            f"| {score.cell_recall:.1%} | {score.digit_accuracy:.1%} "
            f"| {score.solved}/{score.images} | {score.mean_ms:.0f} |"
        )
    print(f"\nTemplates in the reference set: {TemplateClassifier().template_count}")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
