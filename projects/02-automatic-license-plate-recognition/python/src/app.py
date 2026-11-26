"""Command-line application and media I/O for automatic plate recognition."""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Optional, Sequence, Tuple

import numpy as np

from .errors import ALPRError, ApplicationError
from .ocr import TesseractOcr
from .pipeline import ALPRPipeline, FrameResult
from .tracking import IoUTracker
from .vision import DetectorConfig, PlateDetector, require_opencv


IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
VIDEO_CODECS = {
    ".avi": "MJPG",
    ".mkv": "mp4v",
    ".mov": "mp4v",
    ".mp4": "mp4v",
}


@dataclass(frozen=True)
class AppConfig:
    """Validated runtime settings for one ALPR run."""

    camera: int
    input_path: Optional[Path]
    output_path: Optional[Path]
    resolution: Tuple[int, int]
    no_display: bool
    max_frames: int
    ocr_interval: int
    tesseract_cmd: Optional[Path]
    ocr_min_confidence: float
    tracker_iou: float
    max_missed: int
    detector: DetectorConfig


@dataclass(frozen=True)
class RunStats:
    """Aggregate metrics emitted after processing finishes."""

    frames_processed: int
    plate_observations: int
    recognized_observations: int
    ocr_calls: int
    average_fps: float


class FpsMeter:
    """Measure responsive rolling FPS and average pipeline throughput."""

    def __init__(self, window_size: int = 30) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self._durations: Deque[float] = deque(maxlen=window_size)
        self._total_duration = 0.0
        self._samples = 0

    def update(self, elapsed_seconds: float) -> float:
        if math.isfinite(elapsed_seconds) and elapsed_seconds > 0:
            self._durations.append(elapsed_seconds)
            self._total_duration += elapsed_seconds
            self._samples += 1
        return self.fps

    @property
    def fps(self) -> float:
        duration = sum(self._durations)
        return len(self._durations) / duration if duration > 0 else 0.0

    @property
    def average_fps(self) -> float:
        return self._samples / self._total_duration if self._total_duration > 0 else 0.0


def parse_size(value: str) -> Tuple[int, int]:
    """Parse a positive ``WIDTHxHEIGHT`` command-line value."""

    components = value.lower().split("x")
    if len(components) != 2:
        raise argparse.ArgumentTypeError("value must use WIDTHxHEIGHT, e.g. 1280x720")
    try:
        width, height = (int(component) for component in components)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("WIDTH and HEIGHT must be integers") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("WIDTH and HEIGHT must be positive")
    return width, height


def probability(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a number between 0 and 1") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return result


def positive_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive number") from exc
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("value must be a positive number")
    return result


def positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return result


def nonnegative_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a non-negative integer") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return result


def odd_positive_integer(value: str) -> int:
    result = positive_integer(value)
    if result % 2 == 0:
        raise argparse.ArgumentTypeError("value must be an odd integer")
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect and perspective-correct license plates, then recognize their "
            "text with Tesseract from an image, video, or webcam."
        )
    )
    source = parser.add_argument_group("media")
    source.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    source.add_argument(
        "--input",
        dest="input_path",
        type=Path,
        help="image or video path; overrides --camera",
    )
    source.add_argument(
        "--output", dest="output_path", type=Path, help="annotated image or video path"
    )
    source.add_argument(
        "--resolution",
        type=parse_size,
        default=(1280, 720),
        metavar="WIDTHxHEIGHT",
        help="requested webcam resolution (default: 1280x720)",
    )
    source.add_argument(
        "--no-display", action="store_true", help="disable the OpenCV GUI window"
    )
    source.add_argument(
        "--max-frames",
        type=nonnegative_integer,
        default=0,
        metavar="N",
        help="stop after N video/webcam frames; 0 has no limit (default: 0)",
    )

    ocr = parser.add_argument_group("OCR and tracking")
    ocr.add_argument(
        "--ocr-interval",
        type=positive_integer,
        default=15,
        metavar="N",
        help="run OCR at most once per tracked plate every N frames (default: 15)",
    )
    ocr.add_argument(
        "--ocr-min-confidence",
        type=probability,
        default=0.0,
        metavar="0..1",
        help="discard OCR tokens below this confidence (default: 0.0)",
    )
    ocr.add_argument(
        "--tesseract-cmd",
        type=Path,
        help="path to the Tesseract executable when it is not on PATH",
    )
    ocr.add_argument(
        "--tracker-iou",
        type=probability,
        default=0.30,
        metavar="0..1",
        help="minimum IoU for matching a plate track (default: 0.30)",
    )
    ocr.add_argument(
        "--max-missed",
        type=nonnegative_integer,
        default=8,
        metavar="N",
        help="forget a track after N missed frames (default: 8)",
    )

    detector = parser.add_argument_group("plate detector")
    detector.add_argument(
        "--confidence",
        type=probability,
        default=0.45,
        metavar="0..1",
        help="minimum contour geometry score (default: 0.45)",
    )
    detector.add_argument(
        "--min-area-ratio",
        type=probability,
        default=0.005,
        help="minimum plate area divided by frame area (default: 0.005)",
    )
    detector.add_argument(
        "--max-area-ratio",
        type=probability,
        default=0.08,
        help="maximum plate area divided by frame area (default: 0.08)",
    )
    detector.add_argument(
        "--min-aspect-ratio",
        type=positive_float,
        default=2.0,
        help="minimum long-side/short-side ratio (default: 2.0)",
    )
    detector.add_argument(
        "--max-aspect-ratio",
        type=positive_float,
        default=5.5,
        help="maximum long-side/short-side ratio (default: 5.5)",
    )
    detector.add_argument(
        "--min-rectangularity",
        type=probability,
        default=0.45,
        help="minimum contour area divided by rotated-box area (default: 0.45)",
    )
    detector.add_argument(
        "--blur-kernel",
        type=odd_positive_integer,
        default=5,
        metavar="ODD",
        help="Gaussian blur kernel size (default: 5)",
    )
    detector.add_argument(
        "--canny-low", type=nonnegative_integer, default=50, help="low Canny threshold"
    )
    detector.add_argument(
        "--canny-high", type=positive_integer, default=200, help="high Canny threshold"
    )
    detector.add_argument(
        "--morph-kernel",
        type=parse_size,
        default=(9, 3),
        metavar="WIDTHxHEIGHT",
        help="closing kernel that joins plate edges (default: 9x3)",
    )
    detector.add_argument(
        "--morph-iterations",
        type=positive_integer,
        default=1,
        help="morphological closing passes (default: 1)",
    )
    detector.add_argument(
        "--max-candidates",
        type=positive_integer,
        default=5,
        metavar="N",
        help="maximum plate candidates per frame (default: 5)",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> AppConfig:
    detector_config = DetectorConfig(
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        min_aspect_ratio=args.min_aspect_ratio,
        max_aspect_ratio=args.max_aspect_ratio,
        min_rectangularity=args.min_rectangularity,
        min_confidence=args.confidence,
        blur_kernel=args.blur_kernel,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        morph_kernel=args.morph_kernel,
        morph_iterations=args.morph_iterations,
        max_candidates=args.max_candidates,
    )
    if args.tracker_iou <= 0:
        raise ValueError("tracker_iou must be greater than 0")
    return AppConfig(
        camera=args.camera,
        input_path=args.input_path,
        output_path=args.output_path,
        resolution=args.resolution,
        no_display=args.no_display,
        max_frames=args.max_frames,
        ocr_interval=args.ocr_interval,
        tesseract_cmd=args.tesseract_cmd,
        ocr_min_confidence=args.ocr_min_confidence,
        tracker_iou=args.tracker_iou,
        max_missed=args.max_missed,
        detector=detector_config,
    )


def _same_file(first: Path, second: Path) -> bool:
    first_resolved = first.expanduser().resolve()
    second_resolved = second.expanduser().resolve()
    if first_resolved == second_resolved:
        return True
    if first_resolved.exists() and second_resolved.exists():
        try:
            return first_resolved.samefile(second_resolved)
        except OSError:
            pass
    return False


def _ensure_output_parent(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.parent.is_dir():
        raise ApplicationError(f"Output directory does not exist: {resolved.parent}")
    return resolved


def _save_image(path: Path, frame: np.ndarray, cv: Any) -> None:
    output_path = _ensure_output_parent(path)
    if output_path.suffix.lower() not in IMAGE_EXTENSIONS:
        extensions = ", ".join(sorted(IMAGE_EXTENSIONS))
        raise ApplicationError(f"Image output must use one of these extensions: {extensions}")
    try:
        written = bool(cv.imwrite(str(output_path), frame))
    except Exception as exc:
        raise ApplicationError(f"Could not write output image {output_path}: {exc}") from exc
    if not written:
        raise ApplicationError(f"Could not write output image: {output_path}")


def _make_video_writer(path: Path, frame: np.ndarray, source_fps: float, cv: Any) -> Any:
    output_path = _ensure_output_parent(path)
    suffix = output_path.suffix.lower()
    if suffix not in VIDEO_CODECS:
        extensions = ", ".join(sorted(VIDEO_CODECS))
        raise ApplicationError(f"Video output must use one of these extensions: {extensions}")
    height, width = frame.shape[:2]
    fps = source_fps if math.isfinite(source_fps) and source_fps > 0 else 30.0
    try:
        fourcc = cv.VideoWriter_fourcc(*VIDEO_CODECS[suffix])
        writer = cv.VideoWriter(str(output_path), fourcc, fps, (width, height))
    except Exception as exc:
        raise ApplicationError(f"Could not create output video {output_path}: {exc}") from exc
    if not writer.isOpened():
        writer.release()
        raise ApplicationError(f"Could not open output video for writing: {output_path}")
    return writer


def _draw_results(frame: np.ndarray, result: FrameResult, fps: float, cv: Any) -> None:
    font = cv.FONT_HERSHEY_SIMPLEX
    for observation in result.observations:
        box = observation.box
        color = (40, 210, 40) if observation.text else (0, 190, 255)
        cv.rectangle(frame, (box.x, box.y), (box.right, box.bottom), color, 2)
        identity = "Plate"
        if observation.track_id is not None:
            identity += f" #{observation.track_id}"
        if observation.text and observation.ocr_confidence > 0:
            label = (
                f"{identity}: {observation.text} "
                f"({observation.ocr_confidence * 100:.0f}%)"
            )
        elif observation.text:
            label = f"{identity}: {observation.text}"
        else:
            label = f"{identity} candidate {observation.detection_confidence * 100:.0f}%"
        label_y = box.y - 8 if box.y >= 24 else min(frame.shape[0] - 8, box.bottom + 20)
        cv.putText(frame, label, (box.x, label_y), font, 0.55, color, 2)
    cv.putText(frame, f"FPS: {fps:.1f}", (10, 26), font, 0.7, (255, 255, 255), 2)
    cv.putText(
        frame,
        f"Plates: {len(result.observations)}",
        (10, 53),
        font,
        0.6,
        (255, 255, 255),
        2,
    )


def _show_frame(frame: np.ndarray, *, image_mode: bool, cv: Any) -> bool:
    try:
        cv.imshow("Automatic License Plate Recognition", frame)
        key = cv.waitKey(0 if image_mode else 1) & 0xFF
    except Exception as exc:
        raise ApplicationError(
            "OpenCV could not open a display window. Re-run with --no-display. "
            f"Original error: {exc}"
        ) from exc
    return key in (27, ord("q"))


def _process_frame(
    frame: np.ndarray,
    pipeline: ALPRPipeline,
    frame_index: int,
    *,
    use_tracking: bool,
    meter: FpsMeter,
    cv: Any,
    clock: Callable[[], float],
) -> Tuple[FrameResult, float]:
    started_at = clock()
    try:
        result = pipeline.process(frame, frame_index, use_tracking=use_tracking)
    except ALPRError:
        raise
    except Exception as exc:
        raise ApplicationError(f"Plate recognition failed: {exc}") from exc
    elapsed = max(clock() - started_at, 1e-9)
    fps = meter.update(elapsed)
    _draw_results(frame, result, fps, cv)
    return result, fps


def _run_image(
    image: np.ndarray,
    config: AppConfig,
    pipeline: ALPRPipeline,
    cv: Any,
    clock: Callable[[], float],
) -> RunStats:
    meter = FpsMeter()
    result, _ = _process_frame(
        image,
        pipeline,
        0,
        use_tracking=False,
        meter=meter,
        cv=cv,
        clock=clock,
    )
    if config.output_path is not None:
        _save_image(config.output_path, image, cv)
    if not config.no_display:
        _show_frame(image, image_mode=True, cv=cv)
    return RunStats(
        frames_processed=1,
        plate_observations=len(result.observations),
        recognized_observations=sum(bool(item.text) for item in result.observations),
        ocr_calls=result.ocr_calls,
        average_fps=meter.average_fps,
    )


def _capture_fps(capture: Any, cv: Any) -> float:
    try:
        return float(capture.get(cv.CAP_PROP_FPS))
    except (TypeError, ValueError, AttributeError, OverflowError):
        return 0.0


def _run_capture(
    capture: Any,
    config: AppConfig,
    pipeline: ALPRPipeline,
    cv: Any,
    clock: Callable[[], float],
) -> RunStats:
    meter = FpsMeter()
    writer = None
    frames_processed = 0
    plate_observations = 0
    recognized_observations = 0
    ocr_calls = 0
    source_fps = _capture_fps(capture, cv)
    try:
        while config.max_frames == 0 or frames_processed < config.max_frames:
            try:
                ok, frame = capture.read()
            except Exception as exc:
                raise ApplicationError(f"Could not read the next video frame: {exc}") from exc
            if not ok or frame is None:
                break
            result, _ = _process_frame(
                frame,
                pipeline,
                frames_processed,
                use_tracking=True,
                meter=meter,
                cv=cv,
                clock=clock,
            )
            frames_processed += 1
            plate_observations += len(result.observations)
            recognized_observations += sum(bool(item.text) for item in result.observations)
            ocr_calls += result.ocr_calls

            if config.output_path is not None:
                if writer is None:
                    writer = _make_video_writer(config.output_path, frame, source_fps, cv)
                try:
                    writer.write(frame)
                except Exception as exc:
                    raise ApplicationError(
                        f"Could not write a frame to {config.output_path}: {exc}"
                    ) from exc
            if not config.no_display and _show_frame(frame, image_mode=False, cv=cv):
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if frames_processed == 0:
        source = config.input_path if config.input_path is not None else config.camera
        raise ApplicationError(f"No frames could be read from source: {source}")
    return RunStats(
        frames_processed,
        plate_observations,
        recognized_observations,
        ocr_calls,
        meter.average_fps,
    )


def run(
    config: AppConfig,
    *,
    cv_module: Any = None,
    detector: Any = None,
    ocr_engine: Any = None,
    tracker: Optional[IoUTracker] = None,
    clock: Callable[[], float] = time.perf_counter,
) -> RunStats:
    """Run ALPR on the configured source and return aggregate statistics."""

    cv = cv_module if cv_module is not None else require_opencv()
    if config.input_path is not None:
        input_path = config.input_path.expanduser().resolve()
        if not input_path.is_file():
            raise ApplicationError(f"Input file not found: {input_path}")
        if config.output_path is not None and _same_file(input_path, config.output_path):
            raise ApplicationError(
                "Input and output paths must be different; refusing to overwrite "
                f"the source media: {input_path}"
            )
    else:
        input_path = None

    if detector is None:
        detector = PlateDetector(config.detector, cv_module=cv)
    if ocr_engine is None:
        ocr_engine = TesseractOcr(
            minimum_confidence=config.ocr_min_confidence,
            tesseract_cmd=config.tesseract_cmd,
        )
    validation = getattr(ocr_engine, "validate_installation", None)
    if callable(validation):
        validation()
    if tracker is None:
        tracker = IoUTracker(config.tracker_iou, config.max_missed)
    pipeline = ALPRPipeline(
        detector,
        ocr_engine,
        tracker=tracker,
        ocr_interval=config.ocr_interval,
    )

    capture = None
    try:
        if input_path is not None:
            try:
                image = cv.imread(str(input_path), cv.IMREAD_COLOR)
            except Exception as exc:
                raise ApplicationError(f"Could not inspect input file {input_path}: {exc}") from exc
            if image is not None:
                return _run_image(image, config, pipeline, cv, clock)
            try:
                capture = cv.VideoCapture(str(input_path))
            except Exception as exc:
                raise ApplicationError(f"Could not open video {input_path}: {exc}") from exc
        else:
            try:
                capture = cv.VideoCapture(config.camera)
            except Exception as exc:
                raise ApplicationError(f"Could not open camera {config.camera}: {exc}") from exc
            requested_width, requested_height = config.resolution
            capture.set(cv.CAP_PROP_FRAME_WIDTH, requested_width)
            capture.set(cv.CAP_PROP_FRAME_HEIGHT, requested_height)

        if capture is None or not capture.isOpened():
            source = input_path if input_path is not None else config.camera
            if capture is not None:
                capture.release()
            raise ApplicationError(f"Could not open video source: {source}")
        return _run_capture(capture, config, pipeline, cv, clock)
    finally:
        if capture is not None and capture.isOpened():
            capture.release()
        if not config.no_display:
            try:
                cv.destroyAllWindows()
            except Exception:
                pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        config = config_from_args(args)
        stats = run(config)
    except KeyboardInterrupt:
        print("Stopped by user.", file=sys.stderr)
        return 130
    except (ALPRError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"Processed {stats.frames_processed} frame(s); "
        f"observed {stats.plate_observations} plate candidate(s); "
        f"recognized text in {stats.recognized_observations} observation(s); "
        f"ran OCR {stats.ocr_calls} time(s); "
        f"average processing FPS: {stats.average_fps:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
