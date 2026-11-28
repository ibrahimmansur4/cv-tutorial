#!/usr/bin/env python3
"""Render a Sudoku puzzle as a photograph-like image, with ground truth.

Testing the whole chain needs an image whose digits are known. This renders a
grid, optionally warps it as though photographed at an angle, and adds the
things that make real photos hard: uneven lighting, blur and sensor noise.

The puzzle itself and the camera pose are both printed, so a solver run can be
graded rather than eyeballed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from src.solver import format_board, parse_board  # noqa: E402

DEFAULT_OUTPUT = PROJECT_ROOT / "assets" / "input" / "puzzle.png"

# A standard newspaper-grade puzzle with a unique solution.
DEFAULT_PUZZLE = (
    "530070000"
    "600195000"
    "098000060"
    "800060003"
    "400803001"
    "700020006"
    "060000280"
    "000419005"
    "000080079"
)


def board_from_string(text: str) -> list[list[int]]:
    cleaned = "".join(character for character in text if not character.isspace())
    if len(cleaned) != 81:
        raise ValueError(f"a puzzle needs 81 characters, got {len(cleaned)}")
    digits = [0 if character in ".0" else int(character) for character in cleaned]
    return parse_board([digits[index * 9 : (index + 1) * 9] for index in range(9)])


# Hershey fonts usable for the digits. The ones the template classifier builds
# its references from are marked; rendering with the others tests whether the
# classifier generalises to a glyph shape it has never seen.
FONTS: dict[str, int] = {
    "simplex": cv2.FONT_HERSHEY_SIMPLEX,  # in the template set
    "duplex": cv2.FONT_HERSHEY_DUPLEX,  # in the template set
    "complex": cv2.FONT_HERSHEY_COMPLEX,  # in the template set
    "triplex": cv2.FONT_HERSHEY_TRIPLEX,  # in the template set
    "plain": cv2.FONT_HERSHEY_PLAIN,  # held out
    "complex-small": cv2.FONT_HERSHEY_COMPLEX_SMALL,  # held out
    "italic": cv2.FONT_HERSHEY_COMPLEX | cv2.FONT_ITALIC,  # held out
    "simplex-italic": cv2.FONT_HERSHEY_SIMPLEX | cv2.FONT_ITALIC,  # held out
}


def render_board(
    board: Sequence[Sequence[int]],
    *,
    size: int = 540,
    margin: int = 30,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
) -> np.ndarray:
    """Draw a clean, head-on Sudoku grid on white paper."""
    side = size + margin * 2
    image = np.full((side, side, 3), 252, dtype=np.uint8)
    step = size / 9.0
    for index in range(10):
        position = int(round(margin + index * step))
        thickness = 3 if index % 3 == 0 else 1
        cv2.line(image, (margin, position), (margin + size, position), (30, 30, 30), thickness)
        cv2.line(image, (position, margin), (position, margin + size), (30, 30, 30), thickness)

    scale = step / 42.0
    thickness = max(1, int(round(step / 22.0)))
    for row in range(9):
        for column in range(9):
            value = board[row][column]
            if not value:
                continue
            text = str(value)
            (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
            x = int(round(margin + (column + 0.5) * step - text_width / 2))
            y = int(round(margin + (row + 0.5) * step + text_height / 2))
            cv2.putText(image, text, (x, y), font, scale, (20, 20, 20), thickness, cv2.LINE_AA)
    return image


def photograph(
    image: np.ndarray,
    *,
    tilt: float,
    blur: int,
    noise: float,
    lighting: float,
    seed: int,
) -> np.ndarray:
    """Make a clean render look like a photo of a page."""
    rng = np.random.default_rng(seed)
    height, width = image.shape[:2]
    output = image

    if tilt > 0.0:
        # Pull the corners in by varying amounts: a projective warp, which is
        # what holding a camera off-axis actually does.
        shift = tilt * min(width, height)
        source = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        jitter = rng.uniform(-shift, shift, size=(4, 2)).astype(np.float32)
        destination = source + jitter
        offset = destination.min(axis=0)
        destination -= offset
        new_width = int(np.ceil(destination[:, 0].max())) + 1
        new_height = int(np.ceil(destination[:, 1].max())) + 1
        matrix = cv2.getPerspectiveTransform(source, destination)
        output = cv2.warpPerspective(
            output,
            matrix,
            (new_width, new_height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(240, 238, 235),
        )

    if lighting > 0.0:
        # A soft brightness gradient, as from a lamp to one side. This is what
        # makes a single global threshold fail and adaptive thresholding work.
        height, width = output.shape[:2]
        gradient_x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
        gradient_y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
        field = np.outer(1.0 + 0.6 * gradient_y, 1.0 + gradient_x)
        field = 1.0 - lighting * (field - field.min()) / max(1e-6, float(np.ptp(field)))
        output = np.clip(output.astype(np.float32) * field[:, :, None], 0, 255).astype(np.uint8)

    if blur > 0:
        kernel = blur if blur % 2 == 1 else blur + 1
        output = cv2.GaussianBlur(output, (kernel, kernel), 0)

    if noise > 0.0:
        grain = rng.normal(0.0, noise, output.shape)
        output = np.clip(output.astype(np.float32) + grain, 0, 255).astype(np.uint8)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puzzle", default=DEFAULT_PUZZLE, help="81 characters, . or 0 for blank")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--truth", type=Path, help="also write the givens as JSON")
    parser.add_argument("--size", type=int, default=540, help="grid size in pixels")
    parser.add_argument("--tilt", type=float, default=0.0, help="perspective distortion, 0..0.12")
    parser.add_argument("--blur", type=int, default=0, help="Gaussian blur kernel")
    parser.add_argument("--noise", type=float, default=0.0, help="sensor noise sigma")
    parser.add_argument("--lighting", type=float, default=0.0, help="uneven lighting, 0..0.6")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--font",
        default="simplex",
        choices=sorted(FONTS),
        help="digit font (default: simplex)",
    )
    args = parser.parse_args(argv)

    try:
        board = board_from_string(args.puzzle)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.size < 180:
        print("error: --size must be at least 180", file=sys.stderr)
        return 2
    if not 0.0 <= args.tilt <= 0.3:
        print("error: --tilt must be between 0 and 0.3", file=sys.stderr)
        return 2

    image = photograph(
        render_board(board, size=args.size, font=FONTS[args.font]),
        tilt=args.tilt,
        blur=args.blur,
        noise=args.noise,
        lighting=args.lighting,
        seed=args.seed,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        print(f"error: could not write {output}", file=sys.stderr)
        return 2

    givens = sum(1 for row in board for value in row if value)
    print(f"Wrote {output}  ({image.shape[1]}x{image.shape[0]})")
    print(f"Ground truth: {givens} givens, font {args.font}")
    print(format_board(board))
    if args.truth:
        truth = args.truth.expanduser().resolve()
        truth.parent.mkdir(parents=True, exist_ok=True)
        truth.write_text(json.dumps({"givens": board}, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {truth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
