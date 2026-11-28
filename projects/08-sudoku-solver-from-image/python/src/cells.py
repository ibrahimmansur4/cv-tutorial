"""Split a flattened board into 81 cells and clean each one for recognition.

The hard part is not cutting the grid into squares -- that is arithmetic. It is
deciding whether a square holds a digit at all, and getting the ink into a shape
a classifier can read.

Three things get in the way:

**Grid lines.** Every cell crop includes slivers of the lines around it. Left in,
they look like ink and turn an empty cell into a "1". Cropping a margin off each
cell removes most of it, and removing long straight runs by shape (see
`remove_grid_lines`) takes the rest.

**Centring and scale.** A digit may sit anywhere in its cell at any size. The
classifier should see the same thing whichever. Each digit is therefore cropped
to its own bounding box, scaled to a fixed height preserving aspect, and pasted
into the centre of a fixed canvas -- the same normalisation MNIST applies.

**Deciding "empty".** Thresholded noise is never perfectly black, so "any white
pixel means a digit" classifies every cell as full. The test used here is that
the ink must form a single blob of a plausible size, roughly central, and tall
enough relative to the cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .errors import ConfigurationError

GRID = 9
# The classifier canvas. 28x28 matches MNIST, so an MNIST-trained model can be
# dropped in without a second normalisation path.
CANVAS = 28
DIGIT_HEIGHT = 20


@dataclass(frozen=True, slots=True)
class Cell:
    """One extracted cell."""

    row: int
    column: int
    patch: np.ndarray  # cleaned CANVAS x CANVAS, white ink on black
    ink_ratio: float
    occupied: bool


def _largest_blob(mask: np.ndarray, cv2_module: Any) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Isolate the biggest connected component and its bounding box."""
    count, labels, stats, _ = cv2_module.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return None
    # Label 0 is the background, so search from 1.
    areas = stats[1:, cv2_module.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    isolated = np.where(labels == best, np.uint8(255), np.uint8(0))
    x = int(stats[best, cv2_module.CC_STAT_LEFT])
    y = int(stats[best, cv2_module.CC_STAT_TOP])
    width = int(stats[best, cv2_module.CC_STAT_WIDTH])
    height = int(stats[best, cv2_module.CC_STAT_HEIGHT])
    return isolated, (x, y, width, height)


def normalize_digit(mask: np.ndarray, *, cv2_module: Any = cv2) -> np.ndarray:
    """Centre and scale a digit blob onto a CANVAS x CANVAS canvas."""
    blob = _largest_blob(mask, cv2_module)
    canvas = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
    if blob is None:
        return canvas
    isolated, (x, y, width, height) = blob
    if width <= 0 or height <= 0:
        return canvas
    cropped = isolated[y : y + height, x : x + width]

    # Scale the longer side to DIGIT_HEIGHT so a "1" keeps its thin shape
    # instead of being stretched into an "8"-shaped box.
    scale = DIGIT_HEIGHT / max(width, height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2_module.resize(
        cropped, (new_width, new_height), interpolation=cv2_module.INTER_AREA
    )
    offset_x = (CANVAS - new_width) // 2
    offset_y = (CANVAS - new_height) // 2
    canvas[offset_y : offset_y + new_height, offset_x : offset_x + new_width] = resized
    return canvas


def remove_grid_lines(
    inner: np.ndarray, *, line_ratio: float = 0.7, cv2_module: Any = cv2
) -> np.ndarray:
    """Erase leftover grid lines without erasing a digit that touches one.

    The obvious approach -- flood-fill every white pixel reachable from the
    border -- is wrong, and wrong in a way that is easy to miss. When a digit
    happens to touch a line fragment, the digit is part of that same connected
    component, so the fill swallows the digit too and the cell reads as empty.
    That is not a rare case: thick outer borders and a slightly imperfect warp
    make corner cells touch their lines routinely.

    Instead, lines are removed by *shape*. A grid line is a long, straight run
    spanning most of the cell; no digit stroke is. Opening with a long thin
    kernel keeps only such runs, and subtracting them severs the digit from the
    line while leaving the digit whole. A thin border band is then cleared to
    catch the stubs that remain.
    """
    height, width = inner.shape[:2]
    if height < 3 or width < 3:
        return inner.copy()
    working = inner.copy()

    horizontal_length = max(3, int(round(width * line_ratio)))
    vertical_length = max(3, int(round(height * line_ratio)))
    horizontal = cv2_module.morphologyEx(
        working,
        cv2_module.MORPH_OPEN,
        cv2_module.getStructuringElement(cv2_module.MORPH_RECT, (horizontal_length, 1)),
    )
    vertical = cv2_module.morphologyEx(
        working,
        cv2_module.MORPH_OPEN,
        cv2_module.getStructuringElement(cv2_module.MORPH_RECT, (1, vertical_length)),
    )
    lines = cv2_module.bitwise_or(horizontal, vertical)
    # Dilate slightly so the anti-aliased edge of a line goes with it.
    lines = cv2_module.dilate(
        lines, cv2_module.getStructuringElement(cv2_module.MORPH_RECT, (2, 2))
    )
    working = cv2_module.subtract(working, lines)

    band = max(1, int(round(min(height, width) * 0.06)))
    working[:band, :] = 0
    working[-band:, :] = 0
    working[:, :band] = 0
    working[:, -band:] = 0
    return working


def clean_cell(
    cell: np.ndarray,
    *,
    margin_ratio: float = 0.12,
    min_ink_ratio: float = 0.035,
    max_ink_ratio: float = 0.70,
    cv2_module: Any = cv2,
) -> tuple[np.ndarray, float, bool]:
    """Trim borders, isolate the digit, and decide whether the cell is filled."""
    if cell.ndim != 2:
        raise ConfigurationError("a cell must be a single-channel image")
    height, width = cell.shape[:2]
    margin_y = int(round(height * margin_ratio))
    margin_x = int(round(width * margin_ratio))
    inner = cell[margin_y : height - margin_y, margin_x : width - margin_x]
    if inner.size == 0:
        return np.zeros((CANVAS, CANVAS), dtype=np.uint8), 0.0, False

    working = remove_grid_lines(inner, cv2_module=cv2_module)
    filled_height, filled_width = working.shape[:2]
    ink_ratio = float(cv2_module.countNonZero(working)) / float(working.size)
    blob = _largest_blob(working, cv2_module)
    occupied = False
    if blob is not None and min_ink_ratio <= ink_ratio <= max_ink_ratio:
        _, (x, y, blob_width, blob_height) = blob
        # A digit is tall relative to its cell and roughly central. A stray
        # speck fails the height test; a surviving line fragment fails centring.
        tall_enough = blob_height >= filled_height * 0.35
        centre_x = x + blob_width / 2.0
        centred = abs(centre_x - filled_width / 2.0) <= filled_width * 0.32
        occupied = bool(tall_enough and centred)

    patch = normalize_digit(working, cv2_module=cv2_module) if occupied else np.zeros(
        (CANVAS, CANVAS), dtype=np.uint8
    )
    return patch, ink_ratio, occupied


def split_cells(
    board: np.ndarray,
    *,
    margin_ratio: float = 0.12,
    min_ink_ratio: float = 0.035,
    max_ink_ratio: float = 0.70,
    cv2_module: Any = cv2,
) -> list[Cell]:
    """Cut a thresholded square board into 81 cleaned cells."""
    if board.ndim != 2:
        raise ConfigurationError("the warped board must be a single-channel image")
    height, width = board.shape[:2]
    if height < GRID or width < GRID:
        raise ConfigurationError("the warped board is too small to split into cells")
    cells: list[Cell] = []
    for row in range(GRID):
        # Compute edges from the full extent rather than a fixed step so
        # rounding error cannot accumulate across the board.
        y1 = round(row * height / GRID)
        y2 = round((row + 1) * height / GRID)
        for column in range(GRID):
            x1 = round(column * width / GRID)
            x2 = round((column + 1) * width / GRID)
            patch, ink_ratio, occupied = clean_cell(
                board[y1:y2, x1:x2],
                margin_ratio=margin_ratio,
                min_ink_ratio=min_ink_ratio,
                max_ink_ratio=max_ink_ratio,
                cv2_module=cv2_module,
            )
            cells.append(Cell(row, column, patch, ink_ratio, occupied))
    return cells


def cell_centres(size: int) -> list[tuple[int, int]]:
    """Pixel centre of every cell in a square board of `size` pixels."""
    step = size / GRID
    return [
        (int((column + 0.5) * step), int((row + 0.5) * step))
        for row in range(GRID)
        for column in range(GRID)
    ]
