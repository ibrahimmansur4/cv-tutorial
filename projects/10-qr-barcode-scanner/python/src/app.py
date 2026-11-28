"""CLI and media loops for the QR and barcode scanner."""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .detect import Detection, MultiReader, build_reader
from .errors import ConfigurationError, MediaError, ScannerError
from .history import Scan, ScanHistory, ScanLog
from .overlay import HudState, annotate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = PROJECT_ROOT / "results" / "scans.csv"
DEFAULT_FRAME_DIR = PROJECT_ROOT / "assets" / "output"
WINDOW_NAME = "QR + Barcode Scanner"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

KEY_QUIT = {ord("q"), ord("Q"), 27}
KEY_PAUSE = {ord(" ")}
KEY_SAVE = {ord("s"), ord("S")}
KEY_CLEAR = {ord("c"), ord("C")}


@dataclass(frozen=True, slots=True)
class AppConfig:
    camera: int
    input_path: Path | None
    csv_path: Path | None
    frame_dir: Path | None
    resolution: tuple[int, int]
    cooldown: float
    upscale: float
    min_area: float
    detector: str
    scan_qr: bool
    scan_barcode: bool
    curved: bool
    zoom_out: float | None
    max_frames: int
    display: bool
    mirror: bool
    quiet: bool
    show_help: bool


def parse_size(value: str) -> tuple[int, int]:
    pieces = value.lower().split("x")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("value must use WIDTHxHEIGHT, for example 1280x720")
    try:
        width, height = (int(piece.strip()) for piece in pieces)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("WIDTH and HEIGHT must be integers") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("WIDTH and HEIGHT must be positive")
    return width, height


def nonnegative_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be non-negative") from exc
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return result


def scale_factor(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive number") from exc
    if not math.isfinite(result) or not 0.1 <= result <= 8.0:
        raise argparse.ArgumentTypeError("value must be between 0.1 and 8.0")
    return result


def zoom_fraction(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be between 0 and 1") from exc
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        raise argparse.ArgumentTypeError("value must be above 0 and at most 1")
    return result


def nonnegative_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a non-negative integer") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return result


class FpsMeter:
    """Frame rate over a sliding window.

    An instantaneous 1/dt reading jumps around far too much to read off a
    screen, and a running average over the whole session stops responding to
    what the scanner is doing now. A short window of recent frame times gives
    a number that is both steady and current.
    """

    def __init__(self, window: int = 30, clock: Any = time.perf_counter) -> None:
        self._window = max(2, window)
        self._clock = clock
        self._times: list[float] = []

    def tick(self) -> float:
        self._times.append(self._clock())
        if len(self._times) > self._window:
            del self._times[0]
        return self.value

    @property
    def value(self) -> float:
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        if span <= 0.0:
            return 0.0
        return (len(self._times) - 1) / span


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scanner.py",
        description="Scan QR codes and barcodes from a webcam, a video, or an image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_argument_group("input")
    source.add_argument("--camera", type=nonnegative_int, default=0, help="camera index")
    source.add_argument(
        "--image", dest="image", type=Path, default=None, help="scan a still image instead"
    )
    source.add_argument(
        "--video", dest="video", type=Path, default=None, help="scan a video file instead"
    )
    source.add_argument(
        "--resolution",
        type=parse_size,
        default=(1280, 720),
        metavar="WxH",
        help="requested camera resolution",
    )
    source.add_argument("--mirror", action="store_true", help="mirror the camera preview")

    detect = parser.add_argument_group("detection")
    detect.add_argument(
        "--qr-detector",
        dest="detector",
        choices=("classic", "aruco"),
        default="aruco",
        help="which OpenCV QR detector to use; aruco is faster and reads more",
    )
    detect.add_argument("--no-qr", dest="scan_qr", action="store_false", help="skip QR codes")
    detect.add_argument(
        "--no-barcode", dest="scan_barcode", action="store_false", help="skip 1D barcodes"
    )
    detect.add_argument(
        "--upscale",
        type=scale_factor,
        default=1.0,
        help="enlarge the frame before detecting, to read small codes",
    )
    detect.add_argument(
        "--min-area",
        type=nonnegative_float,
        default=0.0,
        help="ignore codes smaller than this many pixels in area",
    )
    detect.add_argument(
        "--curved",
        action="store_true",
        help="retry undecoded QR codes with the curved-surface sampler",
    )
    detect.add_argument(
        "--zoom-out",
        dest="zoom_out",
        type=zoom_fraction,
        default=None,
        help=(
            "use one fixed zoom-out fraction for the barcode retry instead of "
            "the default ladder; 1.0 disables the retry entirely"
        ),
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--csv",
        dest="csv_path",
        type=Path,
        default=DEFAULT_CSV,
        help="append accepted scans to this CSV",
    )
    output.add_argument(
        "--no-csv", dest="write_csv", action="store_false", help="do not write a scan log"
    )
    output.add_argument(
        "--save-frames",
        dest="frame_dir",
        type=Path,
        default=None,
        help="write an annotated image for every accepted scan",
    )
    output.add_argument(
        "--cooldown",
        type=nonnegative_float,
        default=3.0,
        help="seconds to suppress a repeat of the same payload",
    )
    output.add_argument(
        "--max-frames",
        type=nonnegative_int,
        default=0,
        help="stop after this many frames (0 means run until quit)",
    )
    output.add_argument(
        "--no-display", dest="display", action="store_false", help="run without a window"
    )
    output.add_argument(
        "--no-help", dest="show_help", action="store_false", help="hide the on-screen key hints"
    )
    output.add_argument("--quiet", action="store_true", help="do not print each scan")
    return parser


def resolve_config(args: argparse.Namespace) -> AppConfig:
    if args.image is not None and args.video is not None:
        raise ConfigurationError("pass either --image or --video, not both")
    input_path = args.image if args.image is not None else args.video
    if input_path is not None:
        input_path = Path(input_path).expanduser()
        if not input_path.exists():
            raise ConfigurationError(f"input not found: {input_path}")
    if args.image is not None and input_path is not None:
        if input_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ConfigurationError(
                f"{input_path.name} is not a recognised image; use --video for footage"
            )
    if not args.scan_qr and not args.scan_barcode:
        raise ConfigurationError(
            "nothing to scan for: do not pass both --no-qr and --no-barcode"
        )
    csv_path = Path(args.csv_path).expanduser() if args.write_csv else None
    frame_dir = Path(args.frame_dir).expanduser() if args.frame_dir is not None else None
    return AppConfig(
        camera=args.camera,
        input_path=input_path,
        csv_path=csv_path,
        frame_dir=frame_dir,
        resolution=args.resolution,
        cooldown=args.cooldown,
        upscale=args.upscale,
        min_area=args.min_area,
        detector=args.detector,
        scan_qr=args.scan_qr,
        scan_barcode=args.scan_barcode,
        curved=args.curved,
        zoom_out=args.zoom_out,
        max_frames=args.max_frames,
        display=args.display,
        mirror=args.mirror,
        quiet=args.quiet,
        show_help=args.show_help,
    )


def reader_for(config: AppConfig) -> MultiReader:
    return build_reader(
        qr=config.scan_qr,
        barcode=config.scan_barcode,
        detector=config.detector,
        upscale=config.upscale,
        min_area=config.min_area,
        curved=config.curved,
        zoom_out=config.zoom_out,
    )


def _is_video_path(path: Path) -> bool:
    return path.suffix.lower() not in IMAGE_SUFFIXES


def _timestamped(directory: Path, symbology: str, index: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = "".join(char if char.isalnum() else "-" for char in symbology).strip("-")
    return directory / f"scan-{stamp}-{index:04d}-{safe or 'code'}.png"


def _save_frame(directory: Path, frame: np.ndarray, symbology: str, index: int) -> Path:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path = _timestamped(directory, symbology, index)
        if not cv2.imwrite(str(path), frame):
            raise MediaError(f"OpenCV refused to write {path}")
    except OSError as exc:
        raise MediaError(f"cannot write into {directory}: {exc}") from exc
    return path


def _report(scan: Scan, quiet: bool) -> None:
    if not quiet:
        print(scan.describe(), flush=True)


def _summarise(history: ScanHistory, log: ScanLog | None, frames: int) -> None:
    print(f"\nScanned {frames} frame(s); {len(history)} distinct code(s).")
    if history.suppressed:
        print(f"Suppressed {history.suppressed} repeat detection(s) during the cooldown.")
    by_symbology = history.counts_by_symbology()
    if by_symbology:
        print("\nBy symbology:")
        for name, count in by_symbology.items():
            print(f"  {name:<10} {count}")
    by_kind = history.counts_by_kind()
    if by_kind:
        print("\nBy content:")
        for name, count in by_kind.items():
            print(f"  {name:<10} {count}")
    if log is not None and log.rows_written:
        print(f"\nLog: {log.path} (+{log.rows_written} row(s))")


def run_image(config: AppConfig, reader: MultiReader) -> int:
    """Scan one still image and print what it holds."""
    assert config.input_path is not None
    frame = cv2.imread(str(config.input_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise MediaError(
            f"could not read {config.input_path}; the file may be corrupt "
            "or in an unsupported format"
        )
    # A still has no cooldown to apply: every distinct code in it is one scan.
    history = ScanHistory(cooldown=0.0)
    detections = reader.read(frame)
    accepted = history.offer_all(detections, frame_index=0)
    for scan in accepted:
        _report(scan, config.quiet)

    log: ScanLog | None = None
    if config.csv_path is not None and accepted:
        log = ScanLog(config.csv_path)
        log.write_all(accepted)
        log.close()

    undecoded = [item for item in detections if not item.decoded]
    if not detections:
        print("No codes found. Try --upscale 2 for a small code, or check the focus.")
    elif undecoded and not accepted:
        print(
            f"Found {len(undecoded)} code(s) but decoded none. "
            "The image may be too blurred, too small, or too steeply angled."
        )

    state = HudState(frame_index=0, total_scans=len(history))
    # No recent-scan panel on a still: the console already lists every code, and
    # the panel would sit on top of whatever code is in the top-left corner.
    canvas = annotate(frame, detections, state, None, show_help=False)
    if config.frame_dir is not None:
        path = _save_frame(config.frame_dir, canvas, "image", 0)
        print(f"Annotated image: {path}")
    if config.display:
        cv2.imshow(WINDOW_NAME, canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    _summarise(history, log, frames=1)
    return 0


def open_capture(config: AppConfig) -> cv2.VideoCapture:
    """Open the webcam or a video file, failing with a usable message."""
    if config.input_path is not None:
        capture = cv2.VideoCapture(str(config.input_path))
        if not capture.isOpened():
            raise MediaError(f"could not open the video {config.input_path}")
        return capture
    capture = cv2.VideoCapture(config.camera)
    if not capture.isOpened():
        raise MediaError(
            f"could not open camera {config.camera}. Another application may be "
            "using it, or try a different --camera index."
        )
    width, height = config.resolution
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    return capture


def run_video(config: AppConfig, reader: MultiReader) -> int:
    """Scan a webcam or video stream until the user quits or it ends."""
    capture = open_capture(config)
    history = ScanHistory(cooldown=config.cooldown)
    log = ScanLog(config.csv_path) if config.csv_path is not None else None
    meter = FpsMeter()
    frames = 0
    paused = False
    message = ""
    live = config.input_path is None
    last_frame: np.ndarray | None = None
    last_detections: list[Detection] = []

    try:
        while True:
            if config.max_frames and frames >= config.max_frames:
                break
            if not paused:
                ok, frame = capture.read()
                if not ok or frame is None:
                    if live:
                        raise MediaError("the camera stopped returning frames")
                    break  # A file simply ran out, which is not an error.
                if config.mirror and live:
                    frame = cv2.flip(frame, 1)
                frames += 1
                fps = meter.tick()
                detections = reader.read(frame)
                accepted = history.offer_all(detections, frame_index=frames)
                for scan in accepted:
                    _report(scan, config.quiet)
                    if log is not None:
                        log.write(scan)
                if accepted:
                    message = f"Captured {accepted[-1].symbology}"
                last_frame = frame
                last_detections = detections
            else:
                fps = meter.value

            if not config.display or last_frame is None:
                if paused:
                    break  # Nothing can change a paused, windowless run.
                continue

            state = HudState(
                fps=fps,
                frame_index=frames,
                total_scans=len(history),
                suppressed=history.suppressed,
                paused=paused,
                message=message,
            )
            canvas = annotate(
                last_frame,
                last_detections,
                state,
                history.recent(),
                show_help=config.show_help,
            )
            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in KEY_QUIT:
                break
            if key in KEY_PAUSE:
                paused = not paused
                message = "Paused" if paused else ""
            elif key in KEY_SAVE:
                directory = config.frame_dir or DEFAULT_FRAME_DIR
                path = _save_frame(directory, canvas, "frame", frames)
                message = f"Saved {path.name}"
            elif key in KEY_CLEAR:
                history = ScanHistory(cooldown=config.cooldown)
                message = "History cleared"
    finally:
        capture.release()
        if config.display:
            cv2.destroyAllWindows()
        if log is not None:
            log.close()

    _summarise(history, log, frames)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = resolve_config(args)
        reader = reader_for(config)
        if config.input_path is not None and not _is_video_path(config.input_path):
            return run_image(config, reader)
        return run_video(config, reader)
    except ScannerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
