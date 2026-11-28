"""Draw the solution and project it back onto the original photo.

The answer is rendered onto a blank square the size of the flattened board, then
pushed through the *inverse* homography so it lands on the photographed grid at
the right perspective. That is the satisfying part of the project: the digits
appear to lie on the page, tilting with it, because they are being mapped
through the same transform that flattened it.

Compositing is done with a mask rather than a plain add, so the drawn digits
replace the pixels under them instead of brightening whatever was there.
"""

from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np

from .cells import GRID
from .grid import BoardLocation

FONT = cv2.FONT_HERSHEY_SIMPLEX
SOLVED_COLOR = (70, 200, 90)
GIVEN_COLOR = (230, 160, 60)
LOW_CONFIDENCE_COLOR = (60, 110, 240)


def render_solution_square(
    size: int,
    givens: Sequence[Sequence[int]],
    solution: Sequence[Sequence[int]],
    *,
    draw_givens: bool = False,
    cv2_module: Any = cv2,
) -> np.ndarray:
    """Draw the filled-in digits on a transparent-ish square canvas."""
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    step = size / GRID
    scale = step / 38.0
    thickness = max(1, int(round(step / 18.0)))
    for row in range(GRID):
        for column in range(GRID):
            value = solution[row][column]
            if not value:
                continue
            was_given = bool(givens[row][column])
            if was_given and not draw_givens:
                continue
            text = str(value)
            (text_width, text_height), _ = cv2_module.getTextSize(
                text, FONT, scale, thickness
            )
            x = int(round((column + 0.5) * step - text_width / 2))
            y = int(round((row + 0.5) * step + text_height / 2))
            cv2_module.putText(
                canvas,
                text,
                (x, y),
                FONT,
                scale,
                GIVEN_COLOR if was_given else SOLVED_COLOR,
                thickness,
                cv2_module.LINE_AA,
            )
    return canvas


def overlay_solution(
    frame: np.ndarray,
    location: BoardLocation,
    givens: Sequence[Sequence[int]],
    solution: Sequence[Sequence[int]],
    *,
    draw_givens: bool = False,
    draw_outline: bool = True,
    cv2_module: Any = cv2,
) -> np.ndarray:
    """Composite the solved digits back onto the original photo."""
    square = render_solution_square(
        location.size, givens, solution, draw_givens=draw_givens, cv2_module=cv2_module
    )
    projected = location.unwarp_onto(square, frame.shape, cv2_module=cv2_module)
    output = frame.copy()
    # Build the mask from the projected drawing itself, so only the pixels the
    # digits actually cover are replaced.
    gray = cv2_module.cvtColor(projected, cv2_module.COLOR_BGR2GRAY)
    _, mask = cv2_module.threshold(gray, 12, 255, cv2_module.THRESH_BINARY)
    inverse_mask = cv2_module.bitwise_not(mask)
    background = cv2_module.bitwise_and(output, output, mask=inverse_mask)
    foreground = cv2_module.bitwise_and(projected, projected, mask=mask)
    output = cv2_module.add(background, foreground)

    if draw_outline:
        corners = location.corners.astype(np.int32).reshape(-1, 1, 2)
        cv2_module.polylines(output, [corners], True, SOLVED_COLOR, 2, cv2_module.LINE_AA)
    return output


def draw_recognition_debug(
    board_square: np.ndarray,
    recognized: Sequence[Sequence[int]],
    confidence: Sequence[Sequence[float]],
    *,
    low_confidence: float = 0.15,
    cv2_module: Any = cv2,
) -> np.ndarray:
    """Annotate the flattened board with what was read and how surely."""
    canvas = (
        cv2_module.cvtColor(board_square, cv2_module.COLOR_GRAY2BGR)
        if board_square.ndim == 2
        else board_square.copy()
    )
    size = canvas.shape[0]
    step = size / GRID
    for row in range(GRID):
        for column in range(GRID):
            value = recognized[row][column]
            if not value:
                continue
            score = confidence[row][column]
            colour = LOW_CONFIDENCE_COLOR if score < low_confidence else SOLVED_COLOR
            x = int(round(column * step)) + 4
            y = int(round((row + 1) * step)) - 6
            cv2_module.putText(
                canvas, f"{value}", (x, y), FONT, step / 60.0, colour, 1, cv2_module.LINE_AA
            )
            cv2_module.putText(
                canvas,
                f"{score:.2f}",
                (x + int(step * 0.32), y),
                FONT,
                step / 130.0,
                colour,
                1,
                cv2_module.LINE_AA,
            )
    for index in range(GRID + 1):
        position = int(round(index * step))
        weight = 2 if index % 3 == 0 else 1
        cv2_module.line(canvas, (0, position), (size, position), (90, 90, 90), weight)
        cv2_module.line(canvas, (position, 0), (position, size), (90, 90, 90), weight)
    return canvas


def cell_mosaic(cells: Sequence[Any], *, cell_size: int = 32, cv2_module: Any = cv2) -> np.ndarray:
    """A 9x9 contact sheet of the cleaned cell patches, for debugging."""
    mosaic = np.zeros((GRID * cell_size, GRID * cell_size), dtype=np.uint8)
    for cell in cells:
        patch = cv2_module.resize(
            cell.patch, (cell_size, cell_size), interpolation=cv2_module.INTER_NEAREST
        )
        y = cell.row * cell_size
        x = cell.column * cell_size
        mosaic[y : y + cell_size, x : x + cell_size] = patch
    coloured = cv2_module.cvtColor(mosaic, cv2_module.COLOR_GRAY2BGR)
    for index in range(GRID + 1):
        position = index * cell_size
        cv2_module.line(coloured, (0, position), (GRID * cell_size, position), (60, 60, 60), 1)
        cv2_module.line(coloured, (position, 0), (position, GRID * cell_size), (60, 60, 60), 1)
    return coloured
