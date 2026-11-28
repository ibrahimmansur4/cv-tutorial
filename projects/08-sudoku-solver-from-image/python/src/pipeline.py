"""End-to-end orchestration: find, read, solve, project back.

A webcam adds one problem a still image does not have: the same board is seen
dozens of times a second, and a digit misread on one frame makes the whole
puzzle unsolvable on that frame. Rather than re-solving every frame, the
pipeline caches a solution keyed by the recognized givens, so a stable board is
solved once and then merely re-projected. A frame that reads differently is
simply a cache miss, and a frame that reads as unsolvable falls back to the last
good solution instead of flickering the overlay off.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np

from .cells import Cell, split_cells
from .classify import DigitClassifier, classify_cells
from .errors import BoardNotFoundError, SudokuError, UnsolvableError
from .grid import BoardLocation, find_board, preprocess
from .render import overlay_solution
from .solver import SolveStats, count_solutions, find_conflicts, solve


class FpsMeter:
    def __init__(self, window_size: int = 30) -> None:
        if window_size <= 0:
            raise ValueError("FPS window must be positive")
        self._durations: deque[float] = deque(maxlen=window_size)

    def update(self, elapsed: float) -> float:
        if math.isfinite(elapsed) and elapsed > 0.0:
            self._durations.append(elapsed)
        total = sum(self._durations)
        return len(self._durations) / total if total > 0.0 else 0.0


@dataclass(frozen=True, slots=True)
class SolveOutcome:
    """Everything one successful pass produced."""

    location: BoardLocation
    board_square: np.ndarray
    cells: tuple[Cell, ...]
    givens: tuple[tuple[int, ...], ...]
    confidence: tuple[tuple[float, ...], ...]
    solution: tuple[tuple[int, ...], ...]
    stats: SolveStats
    unique: bool
    from_cache: bool

    @property
    def given_count(self) -> int:
        return sum(1 for row in self.givens for value in row if value)

    @property
    def min_confidence(self) -> float:
        scores = [
            self.confidence[row][column]
            for row in range(9)
            for column in range(9)
            if self.givens[row][column]
        ]
        return min(scores) if scores else 0.0


@dataclass(slots=True)
class PipelineStats:
    frames: int = 0
    boards_found: int = 0
    solved: int = 0
    cache_hits: int = 0
    unsolvable: int = 0
    elapsed_seconds: float = 0.0

    @property
    def fps(self) -> float:
        return self.frames / self.elapsed_seconds if self.elapsed_seconds > 0.0 else 0.0

    def summary(self) -> str:
        if not self.frames:
            return "No frames were processed."
        return (
            f"Processed {self.frames} frame(s) at {self.fps:.1f} FPS; "
            f"board found in {self.boards_found}; solved {self.solved} "
            f"({self.cache_hits} from cache); {self.unsolvable} unsolvable read(s)"
        )


class SudokuPipeline:
    """Find a board, read it, solve it, and draw the answer back on."""

    def __init__(
        self,
        classifier: DigitClassifier,
        *,
        board_size: int = 450,
        cell_block_size: int = 19,
        cell_constant: int = 4,
        min_area_ratio: float = 0.04,
        margin_ratio: float = 0.12,
        min_ink_ratio: float = 0.035,
        check_unique: bool = True,
        cache_size: int = 8,
        cv2_module: Any = cv2,
    ) -> None:
        self.classifier = classifier
        self.board_size = int(board_size)
        self.cell_block_size = int(cell_block_size)
        self.cell_constant = int(cell_constant)
        self.min_area_ratio = float(min_area_ratio)
        self.margin_ratio = float(margin_ratio)
        self.min_ink_ratio = float(min_ink_ratio)
        self.check_unique = bool(check_unique)
        self._cv2 = cv2_module
        self._cache: dict[tuple, tuple[tuple[tuple[int, ...], ...], SolveStats, bool]] = {}
        self._cache_order: deque[tuple] = deque(maxlen=max(1, cache_size))
        self._last: SolveOutcome | None = None
        self.fps = FpsMeter()

    @property
    def last_outcome(self) -> SolveOutcome | None:
        return self._last

    def read_board(self, frame: np.ndarray) -> tuple[BoardLocation, np.ndarray, list[Cell]]:
        """Locate the grid and extract its cleaned cells."""
        location = find_board(
            frame,
            size=self.board_size,
            min_area_ratio=self.min_area_ratio,
            cv2_module=self._cv2,
        )
        warped = location.warp(frame, cv2_module=self._cv2)
        square = preprocess(
            warped,
            block_size=self.cell_block_size,
            constant=self.cell_constant,
            cv2_module=self._cv2,
        )
        cells = split_cells(
            square,
            margin_ratio=self.margin_ratio,
            min_ink_ratio=self.min_ink_ratio,
            cv2_module=self._cv2,
        )
        return location, square, cells

    def _solve_cached(
        self, givens: tuple[tuple[int, ...], ...]
    ) -> tuple[tuple[tuple[int, ...], ...], SolveStats, bool, bool]:
        cached = self._cache.get(givens)
        if cached is not None:
            solution, stats, unique = cached
            return solution, stats, unique, True
        solved, stats = solve(givens)
        solution = tuple(tuple(row) for row in solved)
        unique = True
        if self.check_unique:
            unique = count_solutions(givens, cap=2) == 1
        if len(self._cache_order) == self._cache_order.maxlen:
            oldest = self._cache_order[0]
            self._cache.pop(oldest, None)
        self._cache_order.append(givens)
        self._cache[givens] = (solution, stats, unique)
        return solution, stats, unique, False

    def process(self, frame: np.ndarray) -> SolveOutcome:
        """Run the whole chain. Raises on any stage that cannot complete."""
        location, square, cells = self.read_board(frame)
        board, confidence = classify_cells(cells, self.classifier)
        givens = tuple(tuple(row) for row in board)

        conflicts = find_conflicts(givens)
        if conflicts:
            positions = ", ".join(f"r{row + 1}c{column + 1}" for row, column in conflicts[:4])
            raise UnsolvableError(
                f"the digits read conflict at {positions}; a digit was misread. "
                "Try --debug to see what the classifier saw."
            )

        solution, stats, unique, from_cache = self._solve_cached(givens)
        outcome = SolveOutcome(
            location=location,
            board_square=square,
            cells=tuple(cells),
            givens=givens,
            confidence=tuple(tuple(row) for row in confidence),
            solution=solution,
            stats=stats,
            unique=unique,
            from_cache=from_cache,
        )
        self._last = outcome
        return outcome

    def annotate(
        self,
        frame: np.ndarray,
        outcome: SolveOutcome,
        *,
        draw_givens: bool = False,
    ) -> np.ndarray:
        return overlay_solution(
            frame,
            outcome.location,
            outcome.givens,
            outcome.solution,
            draw_givens=draw_givens,
            cv2_module=self._cv2,
        )
