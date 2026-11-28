"""CLI and media loop for the motion-detection security camera."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2

from .background import make_background
from .detector import MotionDetector
from .errors import ConfigurationError, MediaError, SecurityCameraError
from .events import write_csv, write_json
from .pipeline import RunStats, SecurityCameraPipeline
from .recorder import EventRecorder, RecorderConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECORDINGS = PROJECT_ROOT / "recordings"
DEFAULT_RESULTS = PROJECT_ROOT / "results"
WINDOW_NAME = "Motion Detection Security Camera"
BACKGROUND_CHOICES = ("running-average", "mog2")


@dataclass(frozen=True, slots=True)
class AppConfig:
    camera: int
    input_path: Path | None
    recordings_dir: Path
    results_dir: Path
    resolution: tuple[int, int]
    background: str
    alpha: float
    delta_threshold: int
    history: int
    var_threshold: float
    detect_shadows: bool
    blur_kernel: int
    min_area: float
    dilate_iterations: int
    warmup_frames: int
    pre_roll: float
    post_roll: float
    min_event: float
    max_event: float
    record: bool
    snapshots: bool
    max_frames: int
    display: bool
    show_mask: bool
    draw_boxes: bool
    mirror: bool


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


def odd_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an odd positive integer") from exc
    if result < 1 or result % 2 == 0:
        raise argparse.ArgumentTypeError("value must be an odd positive integer")
    return result


def probability(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be between 0 and 1") from exc
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        raise argparse.ArgumentTypeError("value must be above 0 and at most 1")
    return result


def positive_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("value must be a positive number")
    return result


def nonnegative_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be non-negative") from exc
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return result


def nonnegative_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a non-negative integer") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Turn a webcam into a motion-triggered security camera that records "
            "timestamped clips."
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="recorded video instead of the webcam")
    source.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    parser.add_argument(
        "--recordings",
        type=Path,
        default=DEFAULT_RECORDINGS,
        help=f"directory for event clips (default: {DEFAULT_RECORDINGS})",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help=f"directory for the event log (default: {DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "--resolution",
        type=parse_size,
        default=(1280, 720),
        metavar="WIDTHxHEIGHT",
        help="requested webcam resolution (default: 1280x720)",
    )
    parser.add_argument(
        "--max-frames",
        type=nonnegative_integer,
        default=0,
        metavar="N",
        help="stop after N frames; 0 has no limit (default: 0)",
    )
    parser.add_argument("--no-display", action="store_true", help="disable the OpenCV window")
    parser.add_argument("--show-mask", action="store_true", help="show the binary mask alongside")
    parser.add_argument("--no-boxes", action="store_true", help="do not draw motion boxes")
    parser.add_argument("--no-record", action="store_true", help="detect only, write nothing")
    parser.add_argument("--no-snapshots", action="store_true", help="skip the event thumbnail")
    mirror = parser.add_mutually_exclusive_group()
    mirror.add_argument("--mirror", dest="mirror", action="store_true", help="mirror frames")
    mirror.add_argument(
        "--no-mirror", dest="mirror", action="store_false", help="do not mirror frames"
    )
    parser.set_defaults(mirror=False)

    model = parser.add_argument_group("background model")
    model.add_argument(
        "--background",
        choices=BACKGROUND_CHOICES,
        default="running-average",
        help="background model (default: running-average)",
    )
    model.add_argument(
        "--alpha",
        type=probability,
        default=0.04,
        help="running-average learning rate; lower adapts slower (default: 0.04)",
    )
    model.add_argument(
        "--delta-threshold",
        type=int,
        default=12,
        help="running-average per-pixel difference threshold, 1-255 (default: 12)",
    )
    model.add_argument("--history", type=int, default=300, help="MOG2 history (default: 300)")
    model.add_argument(
        "--var-threshold", type=positive_float, default=24.0, help="MOG2 variance (default: 24)"
    )
    model.add_argument(
        "--no-shadows", action="store_true", help="disable MOG2 shadow removal"
    )
    model.add_argument(
        "--warmup-frames",
        type=nonnegative_integer,
        default=20,
        help="frames to learn the scene before detecting (default: 20)",
    )

    shaping = parser.add_argument_group("mask shaping")
    shaping.add_argument(
        "--blur", type=odd_integer, default=21, help="Gaussian blur kernel, odd (default: 21)"
    )
    shaping.add_argument(
        "--min-area",
        type=nonnegative_float,
        default=500.0,
        help="smallest contour area counted as motion (default: 500)",
    )
    shaping.add_argument(
        "--dilate",
        type=nonnegative_integer,
        default=2,
        help="dilation iterations after morphology (default: 2)",
    )

    recording = parser.add_argument_group("recording")
    recording.add_argument(
        "--pre-roll",
        type=nonnegative_float,
        default=1.5,
        help="seconds of footage kept from before the trigger (default: 1.5)",
    )
    recording.add_argument(
        "--post-roll",
        type=nonnegative_float,
        default=3.0,
        help="seconds to keep recording after motion stops (default: 3)",
    )
    recording.add_argument(
        "--min-event",
        type=nonnegative_float,
        default=0.5,
        help="discard events shorter than this (default: 0.5)",
    )
    recording.add_argument(
        "--max-event",
        type=positive_float,
        default=120.0,
        help="split an event after this long (default: 120)",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> AppConfig:
    if args.camera < 0:
        raise ConfigurationError("--camera cannot be negative")
    if not 0 < args.delta_threshold < 256:
        raise ConfigurationError("--delta-threshold must be between 1 and 255")
    if args.history <= 0:
        raise ConfigurationError("--history must be positive")
    if args.min_event > args.max_event:
        raise ConfigurationError("--min-event cannot exceed --max-event")
    return AppConfig(
        camera=args.camera,
        input_path=args.input,
        recordings_dir=args.recordings,
        results_dir=args.results,
        resolution=args.resolution,
        background=args.background,
        alpha=args.alpha,
        delta_threshold=args.delta_threshold,
        history=args.history,
        var_threshold=args.var_threshold,
        detect_shadows=not args.no_shadows,
        blur_kernel=args.blur,
        min_area=args.min_area,
        dilate_iterations=args.dilate,
        warmup_frames=args.warmup_frames,
        pre_roll=args.pre_roll,
        post_roll=args.post_roll,
        min_event=args.min_event,
        max_event=args.max_event,
        record=not args.no_record,
        snapshots=not args.no_snapshots,
        max_frames=args.max_frames,
        display=not args.no_display,
        show_mask=args.show_mask,
        draw_boxes=not args.no_boxes,
        mirror=args.mirror,
    )


def _open_capture(config: AppConfig, cv2_module: Any) -> tuple[Any, int | str]:
    if config.input_path is not None:
        source_path = config.input_path.expanduser().resolve()
        if not source_path.is_file():
            raise MediaError(f"input video not found: {source_path}")
        source: int | str = str(source_path)
    else:
        source = config.camera
    try:
        capture = cv2_module.VideoCapture(source)
    except Exception as exc:
        raise MediaError(f"could not create video capture for {source}: {exc}") from exc
    if not capture.isOpened():
        capture.release()
        raise MediaError(
            f"could not open video source: {source}. "
            "Check the camera index, or that another program is not using the camera."
        )
    if isinstance(source, int):
        capture.set(cv2_module.CAP_PROP_FRAME_WIDTH, config.resolution[0])
        capture.set(cv2_module.CAP_PROP_FRAME_HEIGHT, config.resolution[1])
    return capture, source


def _show(frame: Any, cv2_module: Any) -> bool:
    try:
        cv2_module.imshow(WINDOW_NAME, frame)
        key = int(cv2_module.waitKey(1)) & 0xFF
    except Exception as exc:
        raise MediaError("OpenCV display failed; retry with --no-display") from exc
    return key not in (27, ord("q"), ord("Q"))


def _build_detector(config: AppConfig, cv2_module: Any) -> MotionDetector:
    background = make_background(
        config.background,
        alpha=config.alpha,
        delta_threshold=config.delta_threshold,
        history=config.history,
        var_threshold=config.var_threshold,
        detect_shadows=config.detect_shadows,
        warmup_frames=config.warmup_frames,
        cv2_module=cv2_module,
    )
    return MotionDetector(
        background,
        blur_kernel=config.blur_kernel,
        min_area=config.min_area,
        dilate_iterations=config.dilate_iterations,
        cv2_module=cv2_module,
    )


def run(
    config: AppConfig,
    *,
    cv2_module: Any = cv2,
    detector: MotionDetector | None = None,
    recorder: EventRecorder | None = None,
    clock: Callable[[], float] = time.perf_counter,
    wall_clock: Callable[[], datetime] = datetime.now,
) -> RunStats:
    """Watch a source and record motion events."""
    detector = detector or _build_detector(config, cv2_module)

    capture = None
    stats = RunStats()
    wall_seconds = 0.0
    last_elapsed = 0.0
    source_fps = 0.0
    started = clock()
    started_wall = wall_clock()
    pipeline: SecurityCameraPipeline | None = None
    now_seconds = 0.0
    try:
        capture, source = _open_capture(config, cv2_module)
        try:
            source_fps = float(capture.get(cv2_module.CAP_PROP_FPS))
        except (TypeError, ValueError, AttributeError, OverflowError):
            source_fps = 0.0
        if not math.isfinite(source_fps) or source_fps <= 0.0:
            source_fps = 20.0

        if recorder is None and config.record:
            recorder = EventRecorder(
                RecorderConfig(
                    directory=config.recordings_dir,
                    fps=source_fps,
                    pre_roll_seconds=config.pre_roll,
                    post_roll_seconds=config.post_roll,
                    min_event_seconds=config.min_event,
                    max_event_seconds=config.max_event,
                    write_snapshot=config.snapshots,
                ),
                cv2_module=cv2_module,
                clock=wall_clock,
            )
        pipeline = SecurityCameraPipeline(
            detector,
            recorder,
            show_mask=config.show_mask,
            draw_boxes=config.draw_boxes,
            cv2_module=cv2_module,
        )

        while config.max_frames == 0 or stats.frames < config.max_frames:
            try:
                ok, frame = capture.read()
            except Exception as exc:
                raise MediaError(f"could not read a frame from {source}: {exc}") from exc
            if not ok or frame is None:
                break
            if config.mirror:
                frame = cv2_module.flip(frame, 1)

            # A recorded clip advances on its own timeline so pre/post-roll are
            # measured in footage seconds, not however fast this machine decodes.
            if config.input_path is not None:
                now_seconds = stats.frames / source_fps
                stamp = started_wall + timedelta(seconds=now_seconds)
            else:
                now_seconds = clock() - started
                stamp = wall_clock()

            frame_started = clock()
            result = pipeline.process(frame, now_seconds, last_elapsed, stamp=stamp)
            annotated = pipeline.annotate(
                frame,
                result,
                now_seconds,
                events=len(stats.events) + (1 if result.finished_event else 0),
                timestamp=stamp,
            )
            last_elapsed = max(clock() - frame_started, 1e-9)
            wall_seconds += last_elapsed
            stats.observe(result)

            if config.display and not _show(annotated, cv2_module):
                break
    finally:
        stats.elapsed_seconds = wall_seconds
        if pipeline is not None:
            final = pipeline.close(now_seconds)
            if final is not None:
                stats.events.append(final)
        if capture is not None:
            capture.release()
        if config.display:
            try:
                cv2_module.destroyAllWindows()
            except Exception:
                pass

    if stats.frames == 0:
        raise MediaError("no frames could be read from the selected video source")

    if config.record and stats.events:
        write_csv(stats.events, config.results_dir / "events.csv")
        write_json(stats.events, config.results_dir / "events.json")
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        config = config_from_args(parser.parse_args(argv))
        stats = run(config)
    except KeyboardInterrupt:
        print("Stopped by user.", file=sys.stderr)
        return 130
    except (SecurityCameraError, OSError, ValueError, cv2.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(stats.summary())
    for event in stats.events:
        name = event.video_path.name if event.video_path else "(not written)"
        print(
            f"  event {event.index}: {event.started_at:%H:%M:%S} "
            f"{event.duration:.1f}s  {event.frames} frames -> {name}"
        )
    if stats.events:
        print(f"Event log: {(config.results_dir / 'events.csv').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
