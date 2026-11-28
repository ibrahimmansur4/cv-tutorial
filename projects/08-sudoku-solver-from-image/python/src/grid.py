"""Find the Sudoku board in a frame and flatten it.

```
frame -> grayscale -> blur -> adaptive threshold -> contours
      -> largest quadrilateral -> ordered corners -> homography -> square board
```

**Why adaptive thresholding.** A single global threshold assumes the whole image
has one brightness. A photo of a page almost never does: a hand shadows one
corner, a lamp blows out another. Adaptive thresholding computes a threshold per
neighbourhood, so a grid line that is dark grey in shadow and mid grey in light
is picked up in both places.

**Why the largest quadrilateral.** A Sudoku grid is the largest four-sided thing
in a picture of a Sudoku grid. Contours are approximated with
`approxPolyDP`, which collapses a near-rectangle's many boundary points down to
four corners, and the biggest surviving quad wins.

**Why a homography.** The board is photographed at an angle, so it is a
trapezoid in the image. A homography is the 3x3 projective transform that maps
one plane onto another, and four point correspondences determine it exactly --
which is why finding the four corners is the whole job. Applying it gives a
head-on square, and inverting it puts the answer back onto the original photo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .errors import BoardNotFoundError, ConfigurationError


@dataclass(frozen=True, slots=True)
class BoardLocation:
    """Where the board is, and how to move between photo and flattened views."""

    corners: np.ndarray  # (4, 2) float32, ordered TL, TR, BR, BL
    transform: np.ndarray  # photo -> square
    inverse: np.ndarray  # square -> photo
    size: int
    area: float

    def warp(self, frame: np.ndarray, *, cv2_module: Any = cv2) -> np.ndarray:
        return cv2_module.warpPerspective(frame, self.transform, (self.size, self.size))

    def unwarp_onto(
        self, square: np.ndarray, frame_shape: tuple[int, int], *, cv2_module: Any = cv2
    ) -> np.ndarray:
        height, width = frame_shape[:2]
        return cv2_module.warpPerspective(square, self.inverse, (width, height))


def order_corners(points: np.ndarray) -> np.ndarray:
    """Sort four points into top-left, top-right, bottom-right, bottom-left.

    Uses the standard sum/difference trick: in image coordinates the top-left
    corner has the smallest `x + y` and the bottom-right the largest, while the
    top-right has the smallest `y - x` and the bottom-left the largest. It needs
    no angles and is stable for any rotation short of 45 degrees.
    """
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if array.shape[0] != 4:
        raise ConfigurationError("corner ordering needs exactly four points")
    ordered = np.zeros((4, 2), dtype=np.float32)
    total = array.sum(axis=1)
    difference = array[:, 1] - array[:, 0]
    ordered[0] = array[np.argmin(total)]  # top-left
    ordered[2] = array[np.argmax(total)]  # bottom-right
    ordered[1] = array[np.argmin(difference)]  # top-right
    ordered[3] = array[np.argmax(difference)]  # bottom-left
    if len({tuple(point) for point in ordered}) != 4:
        raise BoardNotFoundError("the detected quadrilateral is degenerate")
    return ordered


def _side_lengths(corners: np.ndarray) -> tuple[float, float, float, float]:
    top = float(np.linalg.norm(corners[1] - corners[0]))
    right = float(np.linalg.norm(corners[2] - corners[1]))
    bottom = float(np.linalg.norm(corners[2] - corners[3]))
    left = float(np.linalg.norm(corners[3] - corners[0]))
    return top, right, bottom, left


def is_plausible_board(
    corners: np.ndarray, *, min_side: float = 40.0, max_aspect: float = 1.8
) -> bool:
    """Reject quads too small or too lopsided to be a Sudoku grid.

    A Sudoku board is square, so even at a steep angle its opposite sides stay
    comparable. This filters out things like a book edge or a table top that
    happen to approximate to four points.
    """
    top, right, bottom, left = _side_lengths(corners)
    if min(top, right, bottom, left) < min_side:
        return False
    horizontal = (top + bottom) / 2.0
    vertical = (right + left) / 2.0
    if horizontal <= 0.0 or vertical <= 0.0:
        return False
    aspect = max(horizontal / vertical, vertical / horizontal)
    return aspect <= max_aspect


def preprocess(
    frame: np.ndarray,
    *,
    blur_kernel: int = 5,
    block_size: int = 11,
    constant: int = 2,
    cv2_module: Any = cv2,
) -> np.ndarray:
    """Grayscale, blur, and adaptively threshold into a line mask."""
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ConfigurationError("cannot process an empty frame")
    if blur_kernel < 1 or blur_kernel % 2 == 0:
        raise ConfigurationError("blur kernel must be a positive odd number")
    if block_size < 3 or block_size % 2 == 0:
        raise ConfigurationError("adaptive block size must be an odd number of at least 3")

    if frame.ndim == 3 and frame.shape[2] in (3, 4):
        code = cv2_module.COLOR_BGRA2GRAY if frame.shape[2] == 4 else cv2_module.COLOR_BGR2GRAY
        gray = cv2_module.cvtColor(frame, code)
    elif frame.ndim == 2:
        gray = frame
    else:
        raise ConfigurationError("expected a BGR, BGRA, or single-channel frame")

    blurred = cv2_module.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    # THRESH_BINARY_INV so ink becomes white on black, which is what contour
    # finding and the digit classifier both expect.
    return cv2_module.adaptiveThreshold(
        blurred,
        255,
        cv2_module.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2_module.THRESH_BINARY_INV,
        block_size,
        constant,
    )


def find_board(
    frame: np.ndarray,
    *,
    size: int = 450,
    blur_kernel: int = 5,
    block_size: int = 11,
    constant: int = 2,
    min_area_ratio: float = 0.04,
    max_aspect: float = 1.8,
    cv2_module: Any = cv2,
) -> BoardLocation:
    """Locate the largest plausible Sudoku grid, or raise."""
    if size < 90:
        raise ConfigurationError("warped board size must be at least 90 pixels")
    mask = preprocess(
        frame,
        blur_kernel=blur_kernel,
        block_size=block_size,
        constant=constant,
        cv2_module=cv2_module,
    )
    # Closing joins grid lines broken by thresholding, so the border forms one
    # contour instead of several disconnected segments.
    kernel = cv2_module.getStructuringElement(cv2_module.MORPH_RECT, (3, 3))
    closed = cv2_module.morphologyEx(mask, cv2_module.MORPH_CLOSE, kernel)

    found = cv2_module.findContours(
        closed, cv2_module.RETR_EXTERNAL, cv2_module.CHAIN_APPROX_SIMPLE
    )
    contours = found[-2]
    if not len(contours):
        raise BoardNotFoundError("no contours were found; is the grid visible and in focus?")

    frame_area = float(frame.shape[0] * frame.shape[1])
    minimum_area = frame_area * min_area_ratio
    best: tuple[float, np.ndarray] | None = None
    for contour in sorted(contours, key=cv2_module.contourArea, reverse=True):
        area = float(cv2_module.contourArea(contour))
        if area < minimum_area:
            break  # sorted by area, so everything after this is smaller too
        perimeter = float(cv2_module.arcLength(contour, True))
        if perimeter <= 0.0:
            continue
        # 2% of the perimeter is loose enough to absorb a wobbly hand-held edge
        # but tight enough not to collapse a real corner.
        approximation = cv2_module.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approximation) != 4 or not cv2_module.isContourConvex(approximation):
            continue
        corners = order_corners(approximation.reshape(-1, 2))
        if not is_plausible_board(corners, max_aspect=max_aspect):
            continue
        best = (area, corners)
        break

    if best is None:
        raise BoardNotFoundError(
            "no four-sided grid large enough was found. Fill more of the frame with "
            "the puzzle, improve the lighting, or lower --min-area-ratio."
        )

    area, corners = best
    destination = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32
    )
    transform = cv2_module.getPerspectiveTransform(corners, destination)
    inverse = cv2_module.getPerspectiveTransform(destination, corners)
    return BoardLocation(
        corners=corners, transform=transform, inverse=inverse, size=size, area=area
    )
