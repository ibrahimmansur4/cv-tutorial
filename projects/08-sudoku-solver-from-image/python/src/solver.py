"""Backtracking Sudoku solver, with bitmask constraints.

Backtracking is depth-first search with early abandonment. Pick an empty cell,
try a digit that does not immediately conflict, recurse, and if the recursion
fails, undo the choice and try the next digit. Exhausting every digit for a cell
means the puzzle is unsolvable from here, so the search returns and the caller
undoes *its* choice. That is the whole algorithm.

Two refinements make it fast enough to run per frame, and both are worth
understanding because the naive version is genuinely too slow for hard puzzles:

**Bitmask constraints.** Rather than scanning a row, a column, and a box to test
each candidate -- 27 comparisons per attempt -- each row, column and box keeps a
9-bit mask of the digits it already holds. Testing a candidate is then three
bitwise ANDs, and placing or removing a digit is three XORs.

**Minimum remaining values.** Always expand the empty cell with the fewest legal
candidates. A cell with one candidate is forced, so taking it first either makes
progress for free or proves the branch dead immediately. Expanding cells in
reading order instead would explore enormous subtrees before discovering a
contradiction that a constrained cell would have revealed at once.

This module is pure Python with no OpenCV import, so it can be reasoned about
and tested entirely on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

from .errors import UnsolvableError

SIZE = 9
BOX = 3
EMPTY = 0
FULL_MASK = 0x1FF  # bits 0..8 set: every digit 1..9 available


Board = list[list[int]]


def box_index(row: int, column: int) -> int:
    """Which 3x3 box a cell belongs to, numbered left to right, top to bottom."""
    return (row // BOX) * BOX + (column // BOX)


def parse_board(values: Sequence[Sequence[int]]) -> Board:
    """Validate and copy a 9x9 grid of ints, 0 meaning empty."""
    rows = list(values)
    if len(rows) != SIZE:
        raise ValueError(f"a board needs {SIZE} rows, got {len(rows)}")
    board: Board = []
    for index, row in enumerate(rows):
        cells = list(row)
        if len(cells) != SIZE:
            raise ValueError(f"row {index} needs {SIZE} cells, got {len(cells)}")
        converted: list[int] = []
        for value in cells:
            number = int(value)
            if not 0 <= number <= 9:
                raise ValueError(f"cell values must be 0..9, got {number}")
            converted.append(number)
        board.append(converted)
    return board


def format_board(board: Sequence[Sequence[int]], *, blank: str = ".") -> str:
    """Render a board as text with box separators."""
    lines: list[str] = []
    for row_index, row in enumerate(board):
        if row_index and row_index % BOX == 0:
            lines.append("------+-------+------")
        cells: list[str] = []
        for column_index, value in enumerate(row):
            if column_index and column_index % BOX == 0:
                cells.append("|")
            cells.append(blank if value == EMPTY else str(value))
        lines.append(" ".join(cells))
    return "\n".join(lines)


@dataclass(slots=True)
class SolveStats:
    """How much work the search did."""

    assignments: int = 0
    backtracks: int = 0

    def summary(self) -> str:
        return f"{self.assignments} assignment(s), {self.backtracks} backtrack(s)"


class _State:
    """Board plus the row/column/box masks kept in step with it."""

    __slots__ = ("grid", "rows", "columns", "boxes", "empties")

    def __init__(self, board: Board) -> None:
        self.grid = [row[:] for row in board]
        self.rows = [0] * SIZE
        self.columns = [0] * SIZE
        self.boxes = [0] * SIZE
        self.empties: list[tuple[int, int]] = []
        for row in range(SIZE):
            for column in range(SIZE):
                value = self.grid[row][column]
                if value == EMPTY:
                    self.empties.append((row, column))
                    continue
                bit = 1 << (value - 1)
                if (
                    self.rows[row] & bit
                    or self.columns[column] & bit
                    or self.boxes[box_index(row, column)] & bit
                ):
                    raise UnsolvableError(
                        f"the given digits conflict: {value} appears twice in the "
                        f"row, column, or box containing r{row + 1}c{column + 1}"
                    )
                self.rows[row] |= bit
                self.columns[column] |= bit
                self.boxes[box_index(row, column)] |= bit

    def candidates(self, row: int, column: int) -> int:
        """Bitmask of digits that do not immediately conflict."""
        used = self.rows[row] | self.columns[column] | self.boxes[box_index(row, column)]
        return FULL_MASK & ~used

    def place(self, row: int, column: int, value: int) -> None:
        bit = 1 << (value - 1)
        self.grid[row][column] = value
        self.rows[row] |= bit
        self.columns[column] |= bit
        self.boxes[box_index(row, column)] |= bit

    def remove(self, row: int, column: int, value: int) -> None:
        bit = 1 << (value - 1)
        self.grid[row][column] = EMPTY
        self.rows[row] ^= bit
        self.columns[column] ^= bit
        self.boxes[box_index(row, column)] ^= bit


def _bits(mask: int) -> Iterator[int]:
    """Yield the digits set in a candidate mask."""
    while mask:
        lowest = mask & -mask
        yield lowest.bit_length()
        mask ^= lowest


def _search(state: _State, stats: SolveStats, limit: int) -> bool:
    if not state.empties:
        return True

    # Minimum remaining values: expand the most constrained cell first.
    best_position = 0
    best_mask = 0
    best_count = 10
    for position, (row, column) in enumerate(state.empties):
        mask = state.candidates(row, column)
        count = mask.bit_count()
        if count == 0:
            # A dead end found without descending: this branch cannot work.
            return False
        if count < best_count:
            best_position, best_mask, best_count = position, mask, count
            if count == 1:
                break

    row, column = state.empties.pop(best_position)
    for value in _bits(best_mask):
        if limit and stats.assignments >= limit:
            raise UnsolvableError(
                f"gave up after {limit} assignments; the puzzle may be misread or malformed"
            )
        stats.assignments += 1
        state.place(row, column, value)
        if _search(state, stats, limit):
            return True
        state.remove(row, column, value)
        stats.backtracks += 1
    # Put the cell back so the caller's board is unchanged on failure.
    state.empties.insert(best_position, (row, column))
    return False


def find_conflicts(board: Sequence[Sequence[int]]) -> list[tuple[int, int]]:
    """Cells whose given digit repeats in its row, column, or box."""
    conflicts: list[tuple[int, int]] = []
    for row in range(SIZE):
        for column in range(SIZE):
            value = board[row][column]
            if value == EMPTY:
                continue
            for other in range(SIZE):
                if other != column and board[row][other] == value:
                    conflicts.append((row, column))
                    break
                if other != row and board[other][column] == value:
                    conflicts.append((row, column))
                    break
            else:
                start_row, start_column = (row // BOX) * BOX, (column // BOX) * BOX
                for r in range(start_row, start_row + BOX):
                    for c in range(start_column, start_column + BOX):
                        if (r, c) != (row, column) and board[r][c] == value:
                            conflicts.append((row, column))
                            break
                    else:
                        continue
                    break
    return conflicts


def solve(
    board: Sequence[Sequence[int]], *, limit: int = 2_000_000
) -> tuple[Board, SolveStats]:
    """Solve a puzzle, returning the completed board and search statistics.

    Raises `UnsolvableError` when the givens conflict or no solution exists --
    both of which usually mean a digit was misread, not that the puzzle is bad.
    """
    state = _State(parse_board(board))
    stats = SolveStats()
    if not _search(state, stats, limit):
        raise UnsolvableError(
            "no solution exists for the recognized digits; "
            "a digit was probably misread"
        )
    return state.grid, stats


def count_solutions(board: Sequence[Sequence[int]], *, cap: int = 2) -> int:
    """Count solutions up to `cap`. A proper puzzle has exactly one.

    Used to warn that a misread grid has become ambiguous: more than one
    solution means the "solution" shown is only *a* solution, not *the* one.
    """
    if cap < 1:
        raise ValueError("cap must be at least 1")
    state = _State(parse_board(board))
    found = 0

    def walk() -> None:
        nonlocal found
        if found >= cap:
            return
        if not state.empties:
            found += 1
            return
        best_position = 0
        best_mask = 0
        best_count = 10
        for position, (row, column) in enumerate(state.empties):
            mask = state.candidates(row, column)
            count = mask.bit_count()
            if count == 0:
                return
            if count < best_count:
                best_position, best_mask, best_count = position, mask, count
                if count == 1:
                    break
        row, column = state.empties.pop(best_position)
        for value in _bits(best_mask):
            state.place(row, column, value)
            walk()
            state.remove(row, column, value)
            if found >= cap:
                break
        state.empties.insert(best_position, (row, column))

    walk()
    return found


def is_complete(board: Sequence[Sequence[int]]) -> bool:
    return all(value != EMPTY for row in board for value in row)
