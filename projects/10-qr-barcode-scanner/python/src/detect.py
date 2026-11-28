"""Find and decode QR codes and 1D barcodes in a frame.

```
frame -> grayscale -> [optional upscale] -> QR detector   -> quad + payload
                                        -> barcode reader -> quad + payload
```

**Detection and decoding are separate problems**, and keeping them separate is
what makes the display useful. Locating a QR code is geometry: find the three
square finder patterns, and their centres give the code's orientation and
scale. Decoding is then sampling a grid through the homography those corners
define, and undoing Reed-Solomon error correction over the result.

Either can fail alone. A code seen at a steep angle is usually *found* and not
*read*, so the scanner draws the outline in a different colour and says
"detected, not decoded" -- which tells the user to straighten up rather than to
keep hunting for the code. A scanner that reported nothing in that case would
be hiding the single most useful piece of feedback it has.

**Why the finder pattern works.** The three corner squares are built so that
any scanline through the middle crosses dark and light runs in the ratio
1:1:3:1:1. Ratios survive scaling and rotation, so the same test finds a code
near or far, upright or tilted, without a search over scale or angle. That is
the whole trick, and it is why QR codes are read so much more reliably than a
1D barcode, which has no such landmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol, Sequence

import cv2
import numpy as np

from .content import Content, classify
from .errors import ConfigurationError

QR_SYMBOLOGY = "QR"
UNDECODED = ""

# The retail symbologies are nested rather than distinct, which trips up any
# naive comparison of payloads. UPC-A *is* EAN-13 with a leading zero, so the
# same printed symbol is reported as `EAN_13` "0123456789050" by one decoder
# and `UPC_A` "123456789050" by another -- both correct, and not equal as
# strings. Comparing on the zero-padded form makes them agree.
_RETAIL_SYMBOLOGIES = frozenset({"EAN_13", "EAN_8", "UPC_A", "UPC_E", "BARCODE"})


def retail_key(payload: str, symbology: str) -> str:
    """A comparable form for a product code, so UPC-A and EAN-13 agree."""
    if symbology.upper() not in _RETAIL_SYMBOLOGIES or not payload.isdigit():
        return payload
    return payload.zfill(13)


def same_payload(left: "Detection", right: "Detection") -> bool:
    """Whether two detections carry the same code, across symbologies."""
    return retail_key(left.payload, left.symbology) == retail_key(
        right.payload, right.symbology
    )


@dataclass(frozen=True, slots=True)
class Detection:
    """One code located in a frame, decoded or not."""

    payload: str
    symbology: str
    corners: np.ndarray  # (4, 2) float32, in frame coordinates
    content: Content | None = None

    @property
    def decoded(self) -> bool:
        """True when the payload was actually read, not merely located."""
        return bool(self.payload)

    @property
    def centre(self) -> tuple[float, float]:
        point = self.corners.mean(axis=0)
        return float(point[0]), float(point[1])

    @property
    def area(self) -> float:
        """Polygon area, used to ignore specks and to report code size."""
        x = self.corners[:, 0]
        y = self.corners[:, 1]
        return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Axis-aligned bounding box as (x, y, width, height)."""
        xs, ys = self.corners[:, 0], self.corners[:, 1]
        left, top = int(np.floor(xs.min())), int(np.floor(ys.min()))
        right, bottom = int(np.ceil(xs.max())), int(np.ceil(ys.max()))
        return left, top, right - left, bottom - top

    @property
    def label(self) -> str:
        if not self.decoded:
            return f"{self.symbology}: detected, not decoded"
        if self.content is not None:
            return self.content.summary
        return self.payload

    def with_content(self) -> "Detection":
        """Attach the payload's recognised type."""
        if not self.decoded or self.content is not None:
            return self
        return Detection(
            payload=self.payload,
            symbology=self.symbology,
            corners=self.corners,
            content=classify(self.payload, symbology=self.symbology),
        )

    def scaled(self, factor: float) -> "Detection":
        """Map corners back to the original frame after a detection upscale."""
        if factor == 1.0:
            return self
        return Detection(
            payload=self.payload,
            symbology=self.symbology,
            corners=(self.corners / factor).astype(np.float32),
            content=self.content,
        )


class CodeReader(Protocol):
    """Anything that can find codes in a greyscale or colour image."""

    name: str

    def read(self, image: np.ndarray) -> list[Detection]:
        ...


def _as_quads(points: Any) -> list[np.ndarray]:
    """Normalise the several shapes OpenCV returns for code corners.

    `detectAndDecodeMulti` gives (N, 4, 2); the single-code `detect` gives
    (1, 4, 2); and both give None when nothing was found. Funnelling them
    through one place keeps that quirk out of the rest of the module.
    """
    if points is None:
        return []
    array = np.asarray(points, dtype=np.float32)
    if array.size == 0:
        return []
    array = array.reshape(-1, 4, 2)
    return [quad.astype(np.float32) for quad in array]


class QRReader:
    """QR codes via OpenCV's detector.

    Two detectors ship with OpenCV. The classic one thresholds the image and
    hunts for the 1:1:3:1:1 runs directly. The ArUco-based one reuses the
    marker detector's corner machinery.

    **ArUco is the default, on measurement rather than taste.** The reasonable
    expectation is that the newer detector buys robustness at some cost in
    speed. It does not: on the same frames it is about twice as fast *and*
    reads codes the classic detector cannot, most clearly under perspective,
    where the classic detector loses every tilted code that ArUco still reads.
    Both numbers are in the README. The classic detector stays available
    because it is what the OpenCV samples use, and because it holds on to a
    *located* outline slightly longer under heavy occlusion, which is worth
    something for on-screen feedback even when nothing can be decoded.

    **The single-code fallback.** Reaching for `detectAndDecodeMulti` and
    leaving it there is the obvious implementation, and it quietly loses codes.
    On a frame holding exactly one QR code, the classic detector's *multi* path
    fails on payloads that its *single* path reads without trouble -- same
    image, same detector, different entry point. So when the multi pass comes
    back empty, this retries with `detectAndDecode` before concluding there is
    nothing there. It costs one extra call on frames that found nothing anyway,
    and it is the difference between reading a code and not.

    The ArUco detector does not need the fallback; its multi path reads
    everything its single path does. That is a reason to prefer it, and the
    reason it is offered at all.
    """

    def __init__(self, *, detector: str = "aruco", use_curved: bool = False) -> None:
        if detector not in {"classic", "aruco"}:
            raise ConfigurationError("QR detector must be 'classic' or 'aruco'")
        self.name = f"qr-{detector}"
        self._kind = detector
        self._use_curved = use_curved
        if detector == "aruco":
            self._detector: Any = cv2.QRCodeDetectorAruco()
        else:
            self._detector = cv2.QRCodeDetector()

    def read(self, image: np.ndarray) -> list[Detection]:
        detections = self._read_multi(image)
        if any(item.decoded for item in detections):
            return detections
        single = self._read_single(image)
        if single is not None and single.decoded:
            return [single]
        # Neither path decoded anything; report whatever was located, so the
        # display can still say "detected, not decoded".
        if detections:
            return detections
        return [single] if single is not None else []

    def _read_multi(self, image: np.ndarray) -> list[Detection]:
        try:
            found, payloads, points, _ = self._detector.detectAndDecodeMulti(image)
        except cv2.error:
            # A malformed or tiny frame makes the detector throw rather than
            # return empty. Nothing was found, which is the useful answer.
            return []
        if not found:
            return []
        quads = _as_quads(points)
        texts: Sequence[str] = payloads or ()
        detections: list[Detection] = []
        for index, quad in enumerate(quads):
            payload = texts[index] if index < len(texts) else UNDECODED
            if not payload and self._use_curved:
                payload = self._retry_curved(image, quad)
            detections.append(Detection(payload or UNDECODED, QR_SYMBOLOGY, quad))
        return detections

    def _read_single(self, image: np.ndarray) -> Detection | None:
        """The one-code path, which succeeds on frames the multi path drops."""
        try:
            payload, points, _ = self._detector.detectAndDecode(image)
        except cv2.error:
            return None
        quads = _as_quads(points)
        if not quads:
            return None
        if not payload and self._use_curved:
            payload = self._retry_curved(image, quads[0])
        return Detection(payload or UNDECODED, QR_SYMBOLOGY, quads[0])

    def _retry_curved(self, image: np.ndarray, quad: np.ndarray) -> str:
        """Second attempt for a code on a curved surface.

        `decodeCurved` samples the grid along a fitted curve instead of a
        straight homography, which recovers codes on a bottle or a mug that the
        flat sampler misses. It is slower, so it is only worth running on a
        code that has already been located and failed to decode.
        """
        try:
            payload, _ = self._detector.decodeCurved(image, quad)
        except (cv2.error, AttributeError):
            return UNDECODED
        return payload or UNDECODED


class BarcodeReader:
    """1D barcodes via OpenCV's `barcode` module.

    No third-party decoder is needed: OpenCV ships EAN-13, EAN-8, UPC-A and
    UPC-E readers. The set is deliberately narrow -- these are the retail
    symbologies -- so Code 128 or Code 39 will not be read. That limit is a
    property of OpenCV, not of the pipeline.

    **The zoom-out retry.** OpenCV localises a barcode by looking for a patch
    where the image gradient points consistently the same way, which is what a
    field of parallel bars looks like *once it is small enough that the
    individual bars blur together*. The detector gets there by shrinking the
    frame, so the symbol has to be a modest part of the picture: measured
    against a rendered EAN-13, detection succeeds while the barcode spans up to
    roughly a third of the frame width and fails above that.

    That is the opposite of the intuition everyone brings to a scanner, and it
    bites hardest in the easiest-looking case -- an image that is nothing but a
    barcode, where the bars stay sharply separated at every scale the detector
    tries and no coherent patch is ever found.

    The fix is to give the detector the zoomed-out view it wants: when the first
    pass finds nothing, mount the whole frame on a larger blank canvas and look
    again. Nothing is resampled, so no detail is lost; the symbol is simply
    smaller *relative to the frame*. Corners come back in canvas coordinates
    and are shifted by the known offset.

    **Why a ladder of fractions rather than one.** The obvious implementation
    picks a good fraction and uses it. Measuring says that does not work: the
    detector's response to zoom is not monotonic, and a symbol that fails at
    one fraction can succeed on either side of it. Sweeping a single rendered
    barcode from 0.5 down to 0.1 produces hits and misses interleaved, with no
    value that works for every code. So the retry walks a short ladder and
    stops at the first fraction that finds something, which is both more
    reliable than any single choice and usually no slower, because most frames
    succeed on the first rung.
    """

    name = "barcode"
    ZOOM_LADDER: tuple[float, ...] = (0.33, 0.2, 0.15)

    def __init__(
        self,
        *,
        zoom_out: float | None = None,
        ladder: Sequence[float] | None = None,
        background: int = 235,
    ) -> None:
        if zoom_out is not None and not 0.0 < zoom_out <= 1.0:
            raise ConfigurationError("zoom-out fraction must be above 0 and at most 1")
        self._detector = cv2.barcode.BarcodeDetector()
        if zoom_out is not None:
            # An explicit fraction means "use exactly this"; 1.0 disables the retry.
            self._ladder: tuple[float, ...] = () if zoom_out >= 1.0 else (zoom_out,)
        elif ladder is not None:
            self._ladder = tuple(ladder)
        else:
            self._ladder = self.ZOOM_LADDER
        self._background = background

    @property
    def ladder(self) -> tuple[float, ...]:
        return self._ladder

    def read(self, image: np.ndarray) -> list[Detection]:
        detections = self._detect(image)
        if detections:
            return detections
        for fraction in self._ladder:
            padded, offset = self._zoomed_out(image, fraction)
            if padded is None:
                continue
            found = self._detect(padded)
            if not found:
                continue
            shift = np.asarray(offset, dtype=np.float32)
            return [
                Detection(
                    item.payload,
                    item.symbology,
                    (item.corners - shift).astype(np.float32),
                    item.content,
                )
                for item in found
            ]
        return detections

    def _detect(self, image: np.ndarray) -> list[Detection]:
        try:
            found, payloads, kinds, points = self._detector.detectAndDecodeWithType(image)
        except cv2.error:
            return []
        if not found:
            return []
        quads = _as_quads(points)
        texts: Sequence[str] = payloads or ()
        types: Sequence[str] = kinds or ()
        detections: list[Detection] = []
        for index, quad in enumerate(quads):
            payload = texts[index] if index < len(texts) else UNDECODED
            symbology = types[index] if index < len(types) and types[index] else "BARCODE"
            detections.append(Detection(payload or UNDECODED, symbology, quad))
        return detections

    def _zoomed_out(
        self, image: np.ndarray, fraction: float
    ) -> tuple[np.ndarray | None, tuple[int, int]]:
        """Centre `image` on a blank canvas so it spans `fraction` of the width."""
        height, width = image.shape[:2]
        canvas_width = int(round(width / fraction))
        canvas_height = int(round(height / fraction))
        if canvas_width <= width or canvas_height <= height:
            return None, (0, 0)
        # A generous cap: beyond this the retry costs more than it can win.
        if canvas_width * canvas_height > 40_000_000:
            return None, (0, 0)
        shape: tuple[int, ...] = (canvas_height, canvas_width)
        if image.ndim == 3:
            shape = (canvas_height, canvas_width, image.shape[2])
        canvas = np.full(shape, self._background, dtype=image.dtype)
        left = (canvas_width - width) // 2
        top = (canvas_height - height) // 2
        canvas[top : top + height, left : left + width] = image
        return canvas, (left, top)


@dataclass(frozen=True, slots=True)
class ScanOptions:
    """How hard to look."""

    upscale: float = 1.0
    min_area: float = 0.0
    curved: bool = False
    zoom_out: float | None = None


class MultiReader:
    """Run several readers over one frame and merge the results.

    Detections are deduplicated geometrically: two readers that find the same
    physical code report nearly the same quad, and reporting it twice would
    double-count it in the history. A decoded hit always wins over an
    undecoded one covering the same area.
    """

    def __init__(
        self, readers: Iterable[CodeReader], options: ScanOptions | None = None
    ) -> None:
        self._readers = list(readers)
        if not self._readers:
            raise ConfigurationError("at least one reader is required")
        self._options = options or ScanOptions()

    @property
    def readers(self) -> list[CodeReader]:
        return list(self._readers)

    def read(self, frame: np.ndarray) -> list[Detection]:
        """Detect in `frame` and return deduplicated, classified detections."""
        if frame is None or frame.size == 0:
            return []
        image = self._prepare(frame)
        factor = self._options.upscale
        found: list[Detection] = []
        for reader in self._readers:
            for detection in reader.read(image):
                found.append(detection.scaled(factor))
        merged = _merge(found)
        if self._options.min_area > 0.0:
            merged = [item for item in merged if item.area >= self._options.min_area]
        return [item.with_content() for item in merged]

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        """Greyscale, then optionally enlarge.

        Both detectors work from intensity, so converting once here avoids each
        reader doing it again. Upscaling helps a code that is physically small
        in frame: the decoder needs several pixels per module to tell one
        module from the next, and interpolation cannot invent detail but it does
        give the sub-pixel sampler room to work.
        """
        image = frame
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        factor = self._options.upscale
        if factor != 1.0:
            image = cv2.resize(
                image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC
            )
        return image


def _merge(detections: Sequence[Detection], *, tolerance: float = 0.5) -> list[Detection]:
    """Drop detections whose centres nearly coincide, preferring decoded ones.

    `tolerance` is a fraction of the smaller code's size rather than a pixel
    count, so the same rule works for a code filling the frame and one in the
    corner.
    """
    kept: list[Detection] = []
    for candidate in sorted(detections, key=lambda item: (not item.decoded, -item.area)):
        duplicate = False
        for existing in kept:
            if _same_code(existing, candidate, tolerance):
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def _same_code(left: Detection, right: Detection, tolerance: float) -> bool:
    lx, ly = left.centre
    rx, ry = right.centre
    distance = float(np.hypot(lx - rx, ly - ry))
    extent = min(np.sqrt(max(left.area, 1.0)), np.sqrt(max(right.area, 1.0)))
    if distance > tolerance * extent:
        return False
    # Different payloads at the same place are two stacked codes, not one --
    # but "different" has to allow for a UPC-A and an EAN-13 reading of the
    # same symbol, which are the same code written two ways.
    return not (left.decoded and right.decoded and not same_payload(left, right))


def build_reader(
    *,
    qr: bool = True,
    barcode: bool = True,
    detector: str = "aruco",
    upscale: float = 1.0,
    min_area: float = 0.0,
    curved: bool = False,
    zoom_out: float | None = None,
) -> MultiReader:
    """Assemble the reader set named by the CLI options."""
    readers: list[CodeReader] = []
    if qr:
        readers.append(QRReader(detector=detector, use_curved=curved))
    if barcode:
        readers.append(BarcodeReader(zoom_out=zoom_out))
    if not readers:
        raise ConfigurationError(
            "nothing to scan for: do not pass both --no-qr and --no-barcode"
        )
    return MultiReader(
        readers,
        ScanOptions(
            upscale=upscale, min_area=min_area, curved=curved, zoom_out=zoom_out
        ),
    )
