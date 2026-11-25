"""Capture, annotate, and output frames for the face-detection demo."""

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

from .detectors import (
    Detection,
    FaceDetectionError,
    FaceDetector,
    create_detector,
    require_opencv,
)


class ApplicationError(FaceDetectionError):
    """Raised for expected input, capture, display, or output failures."""


@dataclass(frozen=True)
class AppConfig:
    detector: str
    camera: int
    input_path: Optional[Path]
    confidence: float
    resolution: Tuple[int, int]
    cascade_path: Optional[Path]
    dnn_config_path: Optional[Path]
    dnn_model_path: Optional[Path]
    no_display: bool
    max_frames: int
    output_path: Optional[Path]


@dataclass(frozen=True)
class RunStats:
    frames_processed: int
    faces_detected: int
    average_fps: float


class FpsMeter:
    """Track responsive rolling FPS and whole-run average throughput."""

    def __init__(self, window_size: int = 30) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self._durations: Deque[float] = deque(maxlen=window_size)
        self._total_duration = 0.0
        self._sample_count = 0

    def update(self, elapsed_seconds: float) -> float:
        if math.isfinite(elapsed_seconds) and elapsed_seconds > 0:
            self._durations.append(elapsed_seconds)
            self._total_duration += elapsed_seconds
            self._sample_count += 1
        return self.fps

    @property
    def fps(self) -> float:
        total = sum(self._durations)
        return len(self._durations) / total if total > 0 else 0.0

    @property
    def average_fps(self) -> float:
        """Return detector throughput averaged over every valid sample."""

        if self._total_duration <= 0:
            return 0.0
        return self._sample_count / self._total_duration


def parse_resolution(value: str) -> Tuple[int, int]:
    """Parse WIDTHxHEIGHT into a positive integer pair for argparse."""

    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("resolution must use WIDTHxHEIGHT, e.g. 640x480")
    try:
        width, height = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "resolution must use integer WIDTHxHEIGHT values"
        ) from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("resolution dimensions must be positive")
    return width, height


def confidence_value(value: str) -> float:
    try:
        confidence = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("confidence must be a number from 0 to 1") from exc
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise argparse.ArgumentTypeError("confidence must be between 0 and 1")
    return confidence


def nonnegative_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a non-negative integer") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect faces from a webcam, image, or video using a Haar cascade "
            "or an OpenCV DNN SSD model."
        )
    )
    parser.add_argument(
        "--detector", choices=("haar", "dnn"), default="haar", help="detector backend"
    )
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    parser.add_argument(
        "--input",
        dest="input_path",
        type=Path,
        help="image or video path; when supplied, it overrides --camera",
    )
    parser.add_argument(
        "--confidence",
        type=confidence_value,
        default=0.5,
        help="DNN confidence threshold from 0 to 1 (default: 0.5)",
    )
    parser.add_argument(
        "--resolution",
        type=parse_resolution,
        default=(640, 480),
        metavar="WIDTHxHEIGHT",
        help="requested webcam resolution (default: 640x480)",
    )
    parser.add_argument(
        "--cascade", dest="cascade_path", type=Path, help="custom Haar cascade XML"
    )
    parser.add_argument(
        "--dnn-config",
        dest="dnn_config_path",
        type=Path,
        help="Caffe deploy.prototxt for the OpenCV SSD face detector",
    )
    parser.add_argument(
        "--dnn-model",
        dest="dnn_model_path",
        type=Path,
        help="ResNet-10 SSD .caffemodel weights",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="do not open a GUI window (useful in CI or on servers)",
    )
    parser.add_argument(
        "--max-frames",
        type=nonnegative_integer,
        default=0,
        metavar="N",
        help="stop after N frames; 0 means no limit (default: 0)",
    )
    parser.add_argument(
        "--output", dest="output_path", type=Path, help="annotated image or video path"
    )
    return parser


def config_from_args(args: argparse.Namespace) -> AppConfig:
    return AppConfig(
        detector=args.detector,
        camera=args.camera,
        input_path=args.input_path,
        confidence=args.confidence,
        resolution=args.resolution,
        cascade_path=args.cascade_path,
        dnn_config_path=args.dnn_config_path,
        dnn_model_path=args.dnn_model_path,
        no_display=args.no_display,
        max_frames=args.max_frames,
        output_path=args.output_path,
    )


def _ensure_output_parent(output_path: Path) -> None:
    parent = output_path.expanduser().resolve().parent
    if not parent.is_dir():
        raise ApplicationError(f"Output directory does not exist: {parent}")


def _draw_detections(
    frame: np.ndarray,
    detections: Sequence[Detection],
    fps: float,
    detector_name: str,
    cv: Any,
) -> None:
    box_color = (40, 220, 40)
    text_color = (255, 255, 255)
    font = cv.FONT_HERSHEY_SIMPLEX
    for detection in detections:
        cv.rectangle(
            frame,
            (detection.left, detection.top),
            (detection.right, detection.bottom),
            box_color,
            2,
        )
        label = "Face"
        if detection.confidence is not None:
            label += f" {detection.confidence * 100:.1f}%"
        text_y = max(18, detection.top - 8)
        cv.putText(frame, label, (detection.left, text_y), font, 0.55, box_color, 2)

    cv.putText(frame, f"FPS: {fps:.1f}", (10, 25), font, 0.7, text_color, 2)
    cv.putText(
        frame,
        f"Detector: {detector_name.upper()}",
        (10, 52),
        font,
        0.6,
        text_color,
        2,
    )


def _save_image(path: Path, frame: np.ndarray, cv: Any) -> None:
    path = path.expanduser()
    _ensure_output_parent(path)
    try:
        saved = bool(cv.imwrite(str(path), frame))
    except Exception as exc:
        raise ApplicationError(f"Could not write output image {path}: {exc}") from exc
    if not saved:
        raise ApplicationError(
            f"Could not write output image {path}; check its extension and permissions."
        )


def _make_video_writer(
    path: Path, frame: np.ndarray, source_fps: float, cv: Any
) -> Any:
    path = path.expanduser()
    _ensure_output_parent(path)
    suffix = path.suffix.lower()
    codecs = {".avi": "MJPG", ".mp4": "mp4v", ".mov": "mp4v", ".mkv": "mp4v"}
    if suffix not in codecs:
        raise ApplicationError(
            "Video output must use one of these extensions: .avi, .mp4, .mov, .mkv"
        )
    height, width = frame.shape[:2]
    fps = source_fps if math.isfinite(source_fps) and source_fps > 0 else 30.0
    try:
        fourcc = cv.VideoWriter_fourcc(*codecs[suffix])
        writer = cv.VideoWriter(str(path), fourcc, fps, (width, height))
    except Exception as exc:
        raise ApplicationError(f"Could not create output video {path}: {exc}") from exc
    if not writer.isOpened():
        writer.release()
        raise ApplicationError(f"Could not open output video for writing: {path}")
    return writer


def _show_frame(frame: np.ndarray, image_mode: bool, cv: Any) -> bool:
    """Display a frame and return True when the user requests exit."""

    try:
        cv.imshow("Face Detection", frame)
        key = cv.waitKey(0 if image_mode else 1) & 0xFF
    except Exception as exc:
        raise ApplicationError(
            "OpenCV could not open a display window. Re-run with --no-display. "
            f"Original error: {exc}"
        ) from exc
    return key in (27, ord("q"))


def _process_frame(
    frame: np.ndarray,
    detector: FaceDetector,
    fps_meter: FpsMeter,
    detector_name: str,
    cv: Any,
    clock: Callable[[], float],
) -> Tuple[Sequence[Detection], float]:
    started_at = clock()
    try:
        detections = detector.detect(frame)
    except FaceDetectionError:
        raise
    except Exception as exc:
        raise ApplicationError(f"Face detection failed: {exc}") from exc
    elapsed = max(clock() - started_at, 1e-9)
    fps = fps_meter.update(elapsed)
    _draw_detections(frame, detections, fps, detector_name, cv)
    return detections, fps


def _run_image(
    image: np.ndarray,
    config: AppConfig,
    detector: FaceDetector,
    cv: Any,
    clock: Callable[[], float],
) -> RunStats:
    meter = FpsMeter()
    detections, fps = _process_frame(
        image, detector, meter, config.detector, cv, clock
    )
    if config.output_path is not None:
        _save_image(config.output_path, image, cv)
    if not config.no_display:
        _show_frame(image, True, cv)
    return RunStats(1, len(detections), meter.average_fps)


def _run_capture(
    capture: Any,
    config: AppConfig,
    detector: FaceDetector,
    cv: Any,
    clock: Callable[[], float],
) -> RunStats:
    writer = None
    meter = FpsMeter()
    frames_processed = 0
    faces_detected = 0
    source_fps = float(capture.get(cv.CAP_PROP_FPS))
    try:
        while config.max_frames == 0 or frames_processed < config.max_frames:
            try:
                ok, frame = capture.read()
            except Exception as exc:
                raise ApplicationError(f"Could not read the next video frame: {exc}") from exc
            if not ok or frame is None:
                break
            detections, _ = _process_frame(
                frame, detector, meter, config.detector, cv, clock
            )
            frames_processed += 1
            faces_detected += len(detections)

            if config.output_path is not None:
                if writer is None:
                    writer = _make_video_writer(config.output_path, frame, source_fps, cv)
                try:
                    writer.write(frame)
                except Exception as exc:
                    raise ApplicationError(
                        f"Could not write a frame to {config.output_path}: {exc}"
                    ) from exc
            if not config.no_display and _show_frame(frame, False, cv):
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if frames_processed == 0:
        source = config.input_path if config.input_path is not None else config.camera
        raise ApplicationError(f"No frames could be read from source: {source}")
    return RunStats(frames_processed, faces_detected, meter.average_fps)


def run(
    config: AppConfig,
    *,
    cv_module: Any = None,
    detector: Optional[FaceDetector] = None,
    clock: Callable[[], float] = time.perf_counter,
) -> RunStats:
    """Run the configured pipeline and return aggregate statistics."""

    cv = cv_module if cv_module is not None else require_opencv()
    if config.input_path is not None and config.output_path is not None:
        input_path = config.input_path.expanduser().resolve()
        output_path = config.output_path.expanduser().resolve()
        paths_match = input_path == output_path
        if not paths_match and input_path.exists() and output_path.exists():
            try:
                paths_match = input_path.samefile(output_path)
            except OSError:
                # Resolution equality above is still a safe baseline when the
                # platform cannot query file identities.
                pass
        if paths_match:
            raise ApplicationError(
                "Input and output paths must be different; refusing to overwrite "
                f"the source media: {input_path}"
            )
    if detector is None:
        detector = create_detector(
            config.detector,
            confidence_threshold=config.confidence,
            cascade_path=config.cascade_path,
            dnn_config_path=config.dnn_config_path,
            dnn_model_path=config.dnn_model_path,
            cv_module=cv,
        )

    capture = None
    try:
        if config.input_path is not None:
            input_path = config.input_path.expanduser()
            if not input_path.is_file():
                raise ApplicationError(f"Input file not found: {input_path}")
            try:
                image = cv.imread(str(input_path), cv.IMREAD_COLOR)
            except Exception as exc:
                raise ApplicationError(f"Could not read input file {input_path}: {exc}") from exc
            if image is not None:
                return _run_image(image, config, detector, cv, clock)
            try:
                capture = cv.VideoCapture(str(input_path))
            except Exception as exc:
                raise ApplicationError(
                    f"Could not open video source {input_path}: {exc}"
                ) from exc
        else:
            try:
                capture = cv.VideoCapture(config.camera)
            except Exception as exc:
                raise ApplicationError(
                    f"Could not open camera {config.camera}: {exc}"
                ) from exc
            requested_width, requested_height = config.resolution
            capture.set(cv.CAP_PROP_FRAME_WIDTH, requested_width)
            capture.set(cv.CAP_PROP_FRAME_HEIGHT, requested_height)

        if capture is None or not capture.isOpened():
            source = config.input_path if config.input_path is not None else config.camera
            if capture is not None:
                capture.release()
            raise ApplicationError(f"Could not open video source: {source}")
        return _run_capture(capture, config, detector, cv, clock)
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
    config = config_from_args(args)
    try:
        stats = run(config)
    except KeyboardInterrupt:
        print("Stopped by user.", file=sys.stderr)
        return 130
    except (FaceDetectionError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"Processed {stats.frames_processed} frame(s); "
        f"detected {stats.faces_detected} face(s); "
        f"average processing FPS: {stats.average_fps:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
