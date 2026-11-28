"""Generate QR codes and EAN-13 barcodes as images.

A scanner needs something to scan. Downloading sample images would make the
tests depend on the network and on files nobody can inspect, so the project
renders its own -- which also gives every test a known ground-truth payload to
compare against, and turns the whole pipeline into a round trip: encode a
string, decode it back, check it survived.

QR encoding is delegated to OpenCV's `QRCodeEncoder`. EAN-13 is written out
here, because it is small enough to be worth seeing in full and because seeing
it explains why 1D barcodes are so much more fragile than QR codes.

**How EAN-13 works.** Thirteen digits become 95 black-and-white modules:

```
101 | 6 left digits (7 modules each) | 01010 | 6 right digits | 101
```

The guard patterns at the ends and the middle give a reader a scale reference,
and the first of the thirteen digits is not drawn at all. It is encoded in the
*choice of alphabet* for the six left-hand digits: each can be drawn from an
odd-parity set or an even-parity set, and which of the 2^6 combinations appears
names the missing digit. That is how twelve digit positions carry thirteen
digits.

**And why it is fragile.** There is no error correction -- only a single check
digit, which detects an error without locating or repairing it. One misread
module changes a digit, the checksum fails, and the whole scan is rejected. A
QR code of the same payload carries Reed-Solomon parity and shrugs off a coffee
ring. The measured difference is in the project README.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .errors import ConfigurationError

# The three EAN-13 digit alphabets, as module strings ("1" is a dark module).
# L and G are the two left-hand sets; R is the right-hand set. R is the bitwise
# complement of L, and G is R reversed -- which is what makes the parity trick
# work: no L pattern can be mistaken for a G pattern.
_LEFT_ODD = (
    "0001101", "0011001", "0010011", "0111101", "0100011",
    "0110001", "0101111", "0111011", "0110111", "0001011",
)
_LEFT_EVEN = (
    "0100111", "0110011", "0011011", "0100001", "0011101",
    "0111001", "0000101", "0010001", "0001001", "0010111",
)
_RIGHT = (
    "1110010", "1100110", "1101100", "1000010", "1011100",
    "1001110", "1010000", "1000100", "1001000", "1110100",
)

# Which alphabet each of the six left digits uses, indexed by the first digit.
_PARITY = (
    "LLLLLL", "LLGLGG", "LLGGLG", "LLGGGL", "LGLLGG",
    "LGGLLG", "LGGGLL", "LGLGLG", "LGLGGL", "LGGLGL",
)

_GUARD_EDGE = "101"
_GUARD_CENTRE = "01010"
EAN13_MODULES = 95


def ean13_check_digit(digits: str) -> int:
    """The thirteenth digit, from the first twelve.

    Weights alternate 1, 3 from the left; the check digit is whatever makes the
    weighted sum a multiple of ten. It catches every single-digit error and
    most transpositions, which is all a checksum can do without parity bits.
    """
    body = digits.strip()
    if len(body) != 12 or not body.isdigit():
        raise ConfigurationError("an EAN-13 check digit needs exactly 12 digits")
    total = sum(int(digit) * (3 if index % 2 else 1) for index, digit in enumerate(body))
    return (10 - total % 10) % 10


def normalise_ean13(digits: str) -> str:
    """Accept 12 or 13 digits and return a valid 13-digit code.

    Twelve digits get a check digit appended. Thirteen are verified, because
    silently "fixing" a mistyped code would make the generator produce
    something other than what was asked for.
    """
    body = "".join(character for character in digits if not character.isspace())
    if not body.isdigit():
        raise ConfigurationError(f"an EAN-13 code must be digits only, got {digits!r}")
    if len(body) == 12:
        return body + str(ean13_check_digit(body))
    if len(body) == 13:
        expected = ean13_check_digit(body[:12])
        if int(body[12]) != expected:
            raise ConfigurationError(
                f"{body} has check digit {body[12]}, but {expected} is correct"
            )
        return body
    raise ConfigurationError(f"an EAN-13 code needs 12 or 13 digits, got {len(body)}")


def ean13_modules(digits: str) -> np.ndarray:
    """The 95-module bar pattern for a code, as a 0/1 array."""
    code = normalise_ean13(digits)
    first = int(code[0])
    pattern = _GUARD_EDGE
    for index, digit in enumerate(code[1:7]):
        table = _LEFT_ODD if _PARITY[first][index] == "L" else _LEFT_EVEN
        pattern += table[int(digit)]
    pattern += _GUARD_CENTRE
    for digit in code[7:]:
        pattern += _RIGHT[int(digit)]
    pattern += _GUARD_EDGE
    if len(pattern) != EAN13_MODULES:
        raise ConfigurationError(f"internal error: built {len(pattern)} modules")
    return np.array([int(bit) for bit in pattern], dtype=np.uint8)


def render_ean13(
    digits: str,
    *,
    module_width: int = 3,
    height: int = 120,
    quiet_zone: int = 11,
    guard_extension: int = 10,
) -> np.ndarray:
    """Draw an EAN-13 barcode as a greyscale image.

    The quiet zone is not decoration. A reader finds the first guard bar by
    looking for a dark run after a light run, and with no margin the page edge
    or a neighbouring object supplies that light run instead. Eleven modules is
    the margin the specification asks for on the left.

    The guard bars are drawn longer than the data bars, as on a real label, so
    a scanline that clips the bottom of the symbol still crosses all three
    guards and can establish the module width.
    """
    if module_width < 1:
        raise ConfigurationError("module width must be at least one pixel")
    if height < 20:
        raise ConfigurationError("barcode height must be at least 20 pixels")
    modules = ean13_modules(digits)
    guard_columns = set(range(0, 3)) | set(range(45, 50)) | set(range(92, 95))

    total_modules = EAN13_MODULES + 2 * quiet_zone
    width = total_modules * module_width
    image = np.full((height, width), 255, dtype=np.uint8)
    data_bottom = max(1, height - guard_extension)
    for index, bit in enumerate(modules):
        if not bit:
            continue
        left = (quiet_zone + index) * module_width
        bottom = height if index in guard_columns else data_bottom
        image[0:bottom, left : left + module_width] = 0
    return image


@dataclass(frozen=True, slots=True)
class RenderedCode:
    """A generated symbol and the payload it should decode to.

    `quiet` records how many pixels of blank margin the image carries on each
    side. It is needed downstream: an experiment that covers "10% of the code"
    means 10% of the code, not of the code plus its margin, and the margin is
    a third of a small QR code's width.
    """

    image: np.ndarray
    payload: str
    symbology: str
    quiet: tuple[int, int] = (0, 0)  # (horizontal, vertical) margin in pixels

    @property
    def size(self) -> tuple[int, int]:
        return self.image.shape[1], self.image.shape[0]

    @property
    def inner(self) -> tuple[int, int, int, int]:
        """The symbol itself as (left, top, width, height), margin removed."""
        height, width = self.image.shape[:2]
        dx, dy = self.quiet
        return dx, dy, max(1, width - 2 * dx), max(1, height - 2 * dy)


def render_qr(
    payload: str,
    *,
    module_size: int = 8,
    border: int = 4,
    correction: str = "M",
) -> RenderedCode:
    """Encode `payload` as a QR code image.

    `module_size` is pixels per module. OpenCV returns one pixel per module,
    which no detector can read, so it is enlarged with nearest-neighbour
    sampling -- any smoothing would blur module edges and defeat the point.

    The border is the quiet zone, four modules wide by the standard. Without it
    the detector cannot tell where the finder patterns end.
    """
    if not payload:
        raise ConfigurationError("a QR code needs a non-empty payload")
    if module_size < 1:
        raise ConfigurationError("module size must be at least one pixel")
    levels = {
        "L": cv2.QRCodeEncoder_CORRECT_LEVEL_L,
        "M": cv2.QRCodeEncoder_CORRECT_LEVEL_M,
        "Q": cv2.QRCodeEncoder_CORRECT_LEVEL_Q,
        "H": cv2.QRCodeEncoder_CORRECT_LEVEL_H,
    }
    key = correction.upper()
    if key not in levels:
        raise ConfigurationError("correction level must be one of L, M, Q, H")
    params = cv2.QRCodeEncoder_Params()
    params.correction_level = levels[key]
    encoder = cv2.QRCodeEncoder_create(params)
    try:
        matrix = encoder.encode(payload)
    except cv2.error as exc:
        raise ConfigurationError(
            f"OpenCV could not encode this payload ({len(payload)} bytes): {exc}"
        ) from exc
    # OpenCV's encoder already surrounds the symbol with a quiet zone of its
    # own, so the margin is not just the border added below. Measuring the
    # extent of the dark modules gives the real figure, and keeps it right if a
    # future OpenCV changes how much it adds. Ground truth that is assumed
    # rather than measured is how an experiment ends up quoting the wrong area.
    dark = np.argwhere(matrix == 0)
    built_in = int(dark[:, 0].min()) if dark.size else 0
    if border:
        matrix = cv2.copyMakeBorder(
            matrix, border, border, border, border, cv2.BORDER_CONSTANT, value=255
        )
    image = cv2.resize(
        matrix, None, fx=module_size, fy=module_size, interpolation=cv2.INTER_NEAREST
    )
    margin = (border + built_in) * module_size
    return RenderedCode(
        image=image, payload=payload, symbology="QR", quiet=(margin, margin)
    )


def render_barcode(digits: str, **kwargs: int) -> RenderedCode:
    """Encode `digits` as an EAN-13 barcode image."""
    code = normalise_ean13(digits)
    module_width = int(kwargs.get("module_width", 3))
    quiet_zone = int(kwargs.get("quiet_zone", 11))
    return RenderedCode(
        image=render_ean13(code, **kwargs),
        payload=code,
        symbology="EAN_13",
        # The bars run to the top and bottom edges, so there is no vertical margin.
        quiet=(quiet_zone * module_width, 0),
    )


def on_canvas(
    code: RenderedCode,
    *,
    margin: int = 60,
    background: int = 235,
    colour: bool = True,
) -> np.ndarray:
    """Place a symbol on a larger, slightly grey background.

    A pure-white border the exact size of the code is an easier image than any
    real photograph. Mounting it on a duller surface with room around it is
    closer to a label on a box, and it gives the perspective warp somewhere to
    put the corners.
    """
    height, width = code.image.shape[:2]
    canvas = np.full((height + 2 * margin, width + 2 * margin), background, dtype=np.uint8)
    canvas[margin : margin + height, margin : margin + width] = code.image
    if colour:
        return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    return canvas
