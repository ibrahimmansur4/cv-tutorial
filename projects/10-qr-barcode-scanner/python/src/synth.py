"""Build scenes containing codes, and degrade them in controlled ways.

Lives beside the scanner rather than in `scripts/` because the test suite, the
image generator and the benchmark all need the same renderings, and because
every function here produces **ground truth** as well as pixels: the payload
that should come back and the corners it should be found at. That is what makes
an accuracy number mean anything.

Each degradation is one physical effect, applied alone, so an experiment can
attribute a failure to a cause:

- `blur` -- the camera is out of focus or moving
- `rotate` -- the code is not upright
- `perspective` -- the code is on a surface at an angle
- `shrink` -- the code is far away, so there are few pixels per module
- `occlude` -- part of the code is covered
- `noise` -- the sensor is working in poor light

The interesting comparison is the last two. A QR code carries Reed-Solomon
parity and can lose a corner and still read; an EAN-13 carries one check digit,
which detects damage without repairing it. Running the same occlusion sweep
over both measures the difference instead of asserting it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .errors import ConfigurationError
from .symbols import RenderedCode

BACKGROUND = 235


@dataclass(frozen=True, slots=True)
class Placement:
    """Where a code ended up in a scene, and what it should decode to.

    `corners` bound the whole rendered image, quiet zone included, because that
    is what a detector returns. `margin` records how much of that is margin, so
    an experiment can address the symbol itself.
    """

    payload: str
    symbology: str
    corners: np.ndarray  # (4, 2) float32, TL, TR, BR, BL in scene coordinates
    margin: tuple[float, float] = (0.0, 0.0)

    @property
    def centre(self) -> tuple[float, float]:
        point = self.corners.mean(axis=0)
        return float(point[0]), float(point[1])

    @property
    def symbol_box(self) -> tuple[float, float, float, float]:
        """The code without its quiet zone, as (left, top, right, bottom)."""
        xs, ys = self.corners[:, 0], self.corners[:, 1]
        dx, dy = self.margin
        return (
            float(xs.min()) + dx,
            float(ys.min()) + dy,
            float(xs.max()) - dx,
            float(ys.max()) - dy,
        )


@dataclass(frozen=True, slots=True)
class Scene:
    """A rendered image and the codes known to be in it."""

    image: np.ndarray
    placements: tuple[Placement, ...] = field(default_factory=tuple)

    @property
    def payloads(self) -> set[str]:
        return {item.payload for item in self.placements}


def _blank(width: int, height: int, *, colour: bool, level: int = BACKGROUND) -> np.ndarray:
    shape: tuple[int, ...] = (height, width, 3) if colour else (height, width)
    return np.full(shape, level, dtype=np.uint8)


def place(
    code: RenderedCode,
    *,
    width: int = 1280,
    height: int = 720,
    centre: tuple[float, float] | None = None,
    scale: float = 1.0,
    colour: bool = True,
) -> Scene:
    """Drop one code into a frame of the given size, unwarped.

    The default frame is webcam-shaped on purpose. A code rendered at its own
    size is an unrealistically easy image, and -- for 1D barcodes -- actually an
    impossible one, because OpenCV's barcode localiser needs the symbol to be a
    modest fraction of the picture.
    """
    if scale <= 0.0:
        raise ConfigurationError("scale must be positive")
    symbol = code.image
    if scale != 1.0:
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_NEAREST
        symbol = cv2.resize(symbol, None, fx=scale, fy=scale, interpolation=interpolation)
    symbol_height, symbol_width = symbol.shape[:2]
    if symbol_width > width or symbol_height > height:
        raise ConfigurationError(
            f"the code is {symbol_width}x{symbol_height}, larger than the "
            f"{width}x{height} frame; lower --scale or raise --frame"
        )
    scene = _blank(width, height, colour=colour)
    centre_x = width / 2.0 if centre is None else centre[0]
    centre_y = height / 2.0 if centre is None else centre[1]
    left = int(round(centre_x - symbol_width / 2.0))
    top = int(round(centre_y - symbol_height / 2.0))
    left = max(0, min(left, width - symbol_width))
    top = max(0, min(top, height - symbol_height))
    patch = symbol if not colour else cv2.cvtColor(symbol, cv2.COLOR_GRAY2BGR)
    scene[top : top + symbol_height, left : left + symbol_width] = patch
    corners = np.array(
        [
            [left, top],
            [left + symbol_width, top],
            [left + symbol_width, top + symbol_height],
            [left, top + symbol_height],
        ],
        dtype=np.float32,
    )
    margin = (code.quiet[0] * scale, code.quiet[1] * scale)
    return Scene(scene, (Placement(code.payload, code.symbology, corners, margin),))


def grid(
    codes: list[RenderedCode],
    *,
    width: int = 1600,
    height: int = 900,
    columns: int = 2,
    colour: bool = True,
) -> Scene:
    """Lay several codes out on one frame, for multi-code detection.

    Each cell is sized from the grid rather than the code, and a code too large
    for its cell is scaled down to fit, so the layout never overlaps whatever
    mix of QR codes and barcodes it is given.
    """
    if not codes:
        raise ConfigurationError("a grid needs at least one code")
    columns = max(1, min(columns, len(codes)))
    rows = (len(codes) + columns - 1) // columns
    cell_width = width // columns
    cell_height = height // rows
    scene = _blank(width, height, colour=colour)
    placements: list[Placement] = []
    for index, code in enumerate(codes):
        row, column = divmod(index, columns)
        symbol = code.image
        symbol_height, symbol_width = symbol.shape[:2]
        # Leave a fifth of the cell as margin so neighbours cannot touch.
        budget_width, budget_height = int(cell_width * 0.8), int(cell_height * 0.8)
        factor = min(budget_width / symbol_width, budget_height / symbol_height, 1.0)
        if factor < 1.0:
            symbol = cv2.resize(
                symbol, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA
            )
            symbol_height, symbol_width = symbol.shape[:2]
        left = column * cell_width + (cell_width - symbol_width) // 2
        top = row * cell_height + (cell_height - symbol_height) // 2
        patch = symbol if not colour else cv2.cvtColor(symbol, cv2.COLOR_GRAY2BGR)
        scene[top : top + symbol_height, left : left + symbol_width] = patch
        placements.append(
            Placement(
                code.payload,
                code.symbology,
                np.array(
                    [
                        [left, top],
                        [left + symbol_width, top],
                        [left + symbol_width, top + symbol_height],
                        [left, top + symbol_height],
                    ],
                    dtype=np.float32,
                ),
                (code.quiet[0] * factor, code.quiet[1] * factor),
            )
        )
    return Scene(scene, tuple(placements))


def blur(scene: Scene, radius: int) -> Scene:
    """Gaussian blur, as an out-of-focus or moving camera would give.

    `radius` is the kernel half-width in pixels; 0 leaves the scene alone. The
    kernel is forced odd because OpenCV requires it.
    """
    if radius <= 0:
        return scene
    size = 2 * int(radius) + 1
    return Scene(cv2.GaussianBlur(scene.image, (size, size), 0), scene.placements)


def noise(scene: Scene, sigma: float, *, seed: int = 0) -> Scene:
    """Additive Gaussian sensor noise, in grey levels."""
    if sigma <= 0.0:
        return scene
    generator = np.random.default_rng(seed)
    grain = generator.normal(0.0, sigma, scene.image.shape)
    noisy = np.clip(scene.image.astype(np.float32) + grain, 0, 255).astype(np.uint8)
    return Scene(noisy, scene.placements)


def _warp(scene: Scene, matrix: np.ndarray, size: tuple[int, int]) -> Scene:
    """Apply a homography to the image and to the ground-truth corners alike.

    Warping the corners with the same matrix is what keeps the ground truth
    true. Re-deriving them from the warped image would just be the detector's
    job again, and could not be used to check it.
    """
    warped = cv2.warpPerspective(
        scene.image,
        matrix,
        size,
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(BACKGROUND, BACKGROUND, BACKGROUND),
    )
    moved = []
    for item in scene.placements:
        points = cv2.perspectiveTransform(
            item.corners.reshape(-1, 1, 2).astype(np.float32), matrix
        )
        moved.append(
            Placement(item.payload, item.symbology, points.reshape(-1, 2), item.margin)
        )
    return Scene(warped, tuple(moved))


def rotate(scene: Scene, degrees: float) -> Scene:
    """Rotate about the frame centre, keeping the frame size.

    A QR code should be indifferent to this -- its finder patterns are found by
    a ratio test that does not care about angle. A 1D barcode is not, because
    its decoder scans along image rows.
    """
    if degrees % 360 == 0:
        return scene
    height, width = scene.image.shape[:2]
    affine = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), degrees, 1.0)
    matrix = np.vstack([affine, [0.0, 0.0, 1.0]]).astype(np.float32)
    return _warp(scene, matrix, (width, height))


def perspective(scene: Scene, strength: float) -> Scene:
    """Tilt the scene as though the code were on an angled surface.

    `strength` is the fraction of the frame width by which the top edge is
    pulled in. It is a genuine projective warp, not a shear, so the code is
    foreshortened the way a real tilt foreshortens it.
    """
    if strength <= 0.0:
        return scene
    if strength >= 0.5:
        raise ConfigurationError("perspective strength must be below 0.5")
    height, width = scene.image.shape[:2]
    offset = strength * width
    source = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )
    target = np.array(
        [[offset, 0], [width - offset, 0], [width, height], [0, height]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, target)
    return _warp(scene, matrix, (width, height))


def shrink(scene: Scene, factor: float) -> Scene:
    """Make the whole scene smaller, as moving the camera back would.

    This is the test that finds the resolution floor: a decoder needs several
    pixels per module, and below that no amount of cleverness recovers the
    code, because the information is no longer in the image.
    """
    if factor >= 1.0:
        return scene
    if factor <= 0.0:
        raise ConfigurationError("shrink factor must be positive")
    resized = cv2.resize(
        scene.image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA
    )
    moved = tuple(
        Placement(
            item.payload,
            item.symbology,
            (item.corners * factor).astype(np.float32),
            (item.margin[0] * factor, item.margin[1] * factor),
        )
        for item in scene.placements
    )
    return Scene(resized, moved)


OCCLUSION_REGIONS = ("tl", "tr", "br", "bl", "centre")


def occlude(scene: Scene, fraction: float, *, corner: str = "centre") -> Scene:
    """Cover `fraction` of each code's area with a blank patch.

    **Where the patch goes changes what is being measured**, and conflating the
    two gives a nonsense answer.

    `centre` covers the middle and leaves the edges alone. For a QR code that
    means the three finder patterns survive, so the code is still *located* and
    the damage falls on data modules -- which is exactly the test of
    Reed-Solomon recovery. For a 1D barcode it cuts every bar in half, which no
    scanline can route around.

    A corner such as `tl` measures something different: it destroys a QR finder
    pattern, so detection fails before decoding is attempted, and no amount of
    parity helps. On a barcode the same patch is nearly harmless, because bars
    run the full height of the symbol and any uncovered row still crosses all
    95 modules. A 1D code is redundant along its bars; the corner test measures
    that redundancy, not error correction.
    """
    if fraction <= 0.0:
        return scene
    if fraction >= 1.0:
        raise ConfigurationError("occlusion fraction must be below 1")
    if corner not in OCCLUSION_REGIONS:
        raise ConfigurationError(f"region must be one of {list(OCCLUSION_REGIONS)}")
    image = scene.image.copy()
    side = float(np.sqrt(fraction))
    height, width = image.shape[:2]
    for item in scene.placements:
        left, top, right, bottom = item.symbol_box
        patch_width = (right - left) * side
        patch_height = (bottom - top) * side
        if corner == "centre":
            x0 = (left + right) / 2.0 - patch_width / 2.0
            y0 = (top + bottom) / 2.0 - patch_height / 2.0
        else:
            x0 = left if corner in {"tl", "bl"} else right - patch_width
            y0 = top if corner in {"tl", "tr"} else bottom - patch_height
        cv2.rectangle(
            image,
            (max(0, int(round(x0))), max(0, int(round(y0)))),
            (min(width, int(round(x0 + patch_width))), min(height, int(round(y0 + patch_height)))),
            (BACKGROUND,) * (3 if image.ndim == 3 else 1),
            thickness=-1,
        )
    return Scene(image, scene.placements)
