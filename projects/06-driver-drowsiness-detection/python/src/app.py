"""CLI and media loop for webcam driver-drowsiness detection."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2

from .alarm import make_alarm
from .calibration import Calibration, calibrate, summarize
from .drawing import draw_hud
from .errors import ConfigurationError, DrowsinessError, MediaError
from .face_mesh import MediaPipeFaceTracker
from .metrics import DEFAULT_EAR_THRESHOLD, DEFAULT_MAR_THRESHOLD, compute_metrics
from .pipeline import DrowsinessPipeline, RunStats
from .state import DrowsinessMonitor, MonitorConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "face_landmarker.task"
DEFAULT_PROFILE = PROJECT_ROOT / "results" / "calibration.json"
VIDEO_CODECS = {".avi": "MJPG", ".mp4": "mp4v", ".m4v": "mp4v", ".mov": "mp4v"}
WINDOW_NAME = "Driver Drowsiness Detection"


@dataclass(frozen=True, slots=True)
class AppConfig:
    camera: int
    input_path: Path | None
    output_path: Path | None
    model_path: Path
    profile_path: Path
    resolution: tuple[int, int]
    max_frames: int
    display: bool
    mirror: bool
    show_landmarks: bool
    alarm_enabled: bool
    monitor: MonitorConfig
    calibrate_seconds: float
    closed_ratio: float
    use_profile: bool


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
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("value must be a positive number")
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
            "Detect driver drowsiness from a webcam using the Eye Aspect Ratio and "
            "the duration of eye closure."
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="recorded video instead of the webcam")
    source.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    parser.add_argument("--output", type=Path, help="write the annotated video here")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"MediaPipe Face Landmarker model (default: {DEFAULT_MODEL})",
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
    parser.add_argument(
        "--no-landmarks", action="store_true", help="do not draw the eye and mouth outlines"
    )
    parser.add_argument("--no-alarm", action="store_true", help="stay silent when drowsy")
    mirror = parser.add_mutually_exclusive_group()
    mirror.add_argument("--mirror", dest="mirror", action="store_true", help="mirror frames")
    mirror.add_argument(
        "--no-mirror", dest="mirror", action="store_false", help="do not mirror frames"
    )
    parser.set_defaults(mirror=None)

    calibration = parser.add_argument_group("calibration")
    calibration.add_argument(
        "--calibrate",
        nargs="?",
        type=positive_float,
        const=5.0,
        metavar="SECONDS",
        help="measure your open-eye EAR for SECONDS (default 5) and save a profile",
    )
    calibration.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_PROFILE,
        help=f"calibration profile path (default: {DEFAULT_PROFILE})",
    )
    calibration.add_argument(
        "--no-profile", action="store_true", help="ignore a saved profile and use --ear-threshold"
    )
    calibration.add_argument(
        "--closed-ratio",
        type=probability,
        default=0.75,
        help="fraction of the open-eye baseline counted as closed (default: 0.75)",
    )

    thresholds = parser.add_argument_group("detection thresholds")
    thresholds.add_argument(
        "--ear-threshold",
        type=probability,
        default=DEFAULT_EAR_THRESHOLD,
        help=f"EAR below which the eye is closed (default: {DEFAULT_EAR_THRESHOLD})",
    )
    thresholds.add_argument(
        "--mar-threshold",
        type=probability,
        default=DEFAULT_MAR_THRESHOLD,
        help=f"MAR above which the mouth is open (default: {DEFAULT_MAR_THRESHOLD})",
    )
    thresholds.add_argument(
        "--closure-seconds",
        type=positive_float,
        default=1.5,
        help="continuous closure that triggers the alarm (default: 1.5)",
    )
    thresholds.add_argument(
        "--blink-seconds",
        type=positive_float,
        default=0.4,
        help="closures no longer than this count as blinks (default: 0.4)",
    )
    thresholds.add_argument(
        "--yawn-seconds",
        type=positive_float,
        default=0.7,
        help="mouth open this long counts as a yawn (default: 0.7)",
    )
    thresholds.add_argument(
        "--perclos-window",
        type=positive_float,
        default=60.0,
        help="rolling window for PERCLOS, in seconds (default: 60)",
    )
    thresholds.add_argument(
        "--perclos-warmup",
        type=positive_float,
        default=30.0,
        help="footage needed before PERCLOS is acted on, in seconds (default: 30)",
    )
    thresholds.add_argument(
        "--perclos-threshold",
        type=probability,
        default=0.15,
        help="PERCLOS fraction that reports fatigue (default: 0.15)",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> AppConfig:
    if args.camera < 0:
        raise ConfigurationError("--camera cannot be negative")
    if args.calibrate is not None and args.input is not None:
        raise ConfigurationError("calibrate against the webcam, not a recorded video")
    try:
        monitor = MonitorConfig(
            ear_threshold=args.ear_threshold,
            mar_threshold=args.mar_threshold,
            closure_seconds=args.closure_seconds,
            blink_max_seconds=args.blink_seconds,
            yawn_seconds=args.yawn_seconds,
            perclos_window_seconds=args.perclos_window,
            perclos_threshold=args.perclos_threshold,
            perclos_warmup_seconds=args.perclos_warmup,
        )
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    # A recorded clip is not a mirror; a live preview is.
    mirror = args.mirror if args.mirror is not None else args.input is None
    return AppConfig(
        camera=args.camera,
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        profile_path=args.profile,
        resolution=args.resolution,
        max_frames=args.max_frames,
        display=not args.no_display,
        mirror=mirror,
        show_landmarks=not args.no_landmarks,
        alarm_enabled=not args.no_alarm,
        monitor=monitor,
        calibrate_seconds=args.calibrate or 0.0,
        closed_ratio=args.closed_ratio,
        use_profile=not args.no_profile,
    )


def _paths_collide(first: Path, second: Path) -> bool:
    first_resolved = first.expanduser().resolve()
    second_resolved = second.expanduser().resolve()
    if first_resolved == second_resolved:
        return True
    try:
        return (
            first_resolved.exists()
            and second_resolved.exists()
            and os.path.samefile(first_resolved, second_resolved)
        )
    except OSError:
        return False


def _open_capture(config: AppConfig, cv2_module: Any) -> tuple[Any, int | str]:
    if config.input_path is not None:
        source_path = config.input_path.expanduser().resolve()
        if not source_path.is_file():
            raise MediaError(f"input video not found: {source_path}")
        if config.output_path is not None and _paths_collide(source_path, config.output_path):
            raise MediaError("input and output must be different files")
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


def _make_writer(path: Path, frame: Any, fps: float, cv2_module: Any) -> Any:
    output = path.expanduser().resolve()
    codec = VIDEO_CODECS.get(output.suffix.lower())
    if codec is None:
        raise MediaError("video output must use .avi, .mp4, .m4v, or .mov")
    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    writer = cv2_module.VideoWriter(
        str(output),
        cv2_module.VideoWriter_fourcc(*codec),
        fps if math.isfinite(fps) and fps > 0.0 else 30.0,
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        raise MediaError(f"could not open annotated video output: {output}")
    return writer


def _show(frame: Any, cv2_module: Any) -> bool:
    try:
        cv2_module.imshow(WINDOW_NAME, frame)
        key = int(cv2_module.waitKey(1)) & 0xFF
    except Exception as exc:
        raise MediaError("OpenCV display failed; retry with --no-display") from exc
    return key not in (27, ord("q"), ord("Q"))


def _timestamps(source_fps: float, frames: int, started: float, clock: Callable[[], float]):
    """Strictly increasing millisecond stamps for MediaPipe VIDEO mode."""
    if math.isfinite(source_fps) and source_fps > 0.0:
        return int(round(frames * 1000.0 / source_fps))
    return int(round((clock() - started) * 1000.0))


def _build_tracker(config: AppConfig, cv2_module: Any) -> MediaPipeFaceTracker:
    return MediaPipeFaceTracker(config.model_path, num_faces=1, cv2_module=cv2_module)


def _resolve_monitor(config: AppConfig) -> tuple[MonitorConfig, str]:
    """Apply a saved calibration profile when one is available."""
    if not config.use_profile:
        return config.monitor, f"threshold: {config.monitor.ear_threshold:.3f} (fixed)"
    path = config.profile_path.expanduser()
    if not path.is_file():
        return (
            config.monitor,
            f"threshold: {config.monitor.ear_threshold:.3f} (default; run --calibrate)",
        )
    profile = Calibration.load(path)
    monitor = MonitorConfig(
        ear_threshold=profile.ear_threshold,
        ear_hysteresis=config.monitor.ear_hysteresis,
        mar_threshold=config.monitor.mar_threshold,
        closure_seconds=config.monitor.closure_seconds,
        blink_max_seconds=config.monitor.blink_max_seconds,
        yawn_seconds=config.monitor.yawn_seconds,
        perclos_window_seconds=config.monitor.perclos_window_seconds,
        perclos_threshold=config.monitor.perclos_threshold,
        perclos_warmup_seconds=config.monitor.perclos_warmup_seconds,
        no_face_seconds=config.monitor.no_face_seconds,
    )
    return monitor, (
        f"threshold: {profile.ear_threshold:.3f} "
        f"(calibrated, baseline {profile.baseline_ear:.3f})"
    )


def run_calibration(
    config: AppConfig,
    *,
    cv2_module: Any = cv2,
    tracker: Any = None,
    clock: Callable[[], float] = time.perf_counter,
) -> Calibration:
    """Capture open-eye EAR samples and write a calibration profile."""
    tracker = tracker or _build_tracker(config, cv2_module)
    capture = None
    samples: list[float] = []
    started = clock()
    last_timestamp = -1
    try:
        capture, _ = _open_capture(config, cv2_module)
        print(
            f"Calibrating for {config.calibrate_seconds:.0f}s: "
            "look at the camera with your eyes open and blink normally."
        )
        frames = 0
        while clock() - started < config.calibrate_seconds:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if config.mirror:
                frame = cv2_module.flip(frame, 1)
            height, width = frame.shape[:2]
            timestamp = max(last_timestamp + 1, int(round((clock() - started) * 1000.0)))
            last_timestamp = timestamp
            frames += 1
            faces = tracker.detect(frame, timestamp)
            if faces:
                samples.append(compute_metrics(faces[0].landmarks, width / height).ear)
            if config.display:
                remaining = max(0.0, config.calibrate_seconds - (clock() - started))
                preview = frame.copy()
                cv2_module.putText(
                    preview,
                    f"Calibrating... {remaining:.1f}s  samples: {len(samples)}",
                    (16, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (80, 230, 80),
                    2,
                    cv2_module.LINE_AA,
                )
                if not _show(preview, cv2_module):
                    break
    finally:
        if capture is not None:
            capture.release()
        if config.display:
            try:
                cv2_module.destroyAllWindows()
            except Exception:
                pass
        tracker.close()

    result = calibrate(samples, closed_ratio=config.closed_ratio)
    saved = result.save(config.profile_path)
    print(f"Samples: {summarize(samples)}")
    print(
        f"Baseline EAR {result.baseline_ear:.4f} -> threshold "
        f"{result.ear_threshold:.4f} ({result.closed_ratio:.0%} of baseline)"
    )
    print(f"Saved {saved}")
    return result


def run(
    config: AppConfig,
    *,
    cv2_module: Any = cv2,
    tracker: Any = None,
    alarm: Any = None,
    clock: Callable[[], float] = time.perf_counter,
) -> RunStats:
    """Monitor a webcam or recorded video for drowsiness."""
    monitor_config, threshold_note = _resolve_monitor(config)
    tracker = tracker or _build_tracker(config, cv2_module)
    pipeline = DrowsinessPipeline(
        tracker,
        DrowsinessMonitor(monitor_config),
        alarm=alarm if alarm is not None else make_alarm(config.alarm_enabled),
        show_landmarks=config.show_landmarks,
        cv2_module=cv2_module,
    )
    print(threshold_note)

    capture = None
    writer = None
    stats = RunStats()
    wall_seconds = 0.0
    last_elapsed = 0.0
    previous_now = 0.0
    source_fps = 0.0
    started = clock()
    last_timestamp = -1
    try:
        capture, source = _open_capture(config, cv2_module)
        try:
            source_fps = float(capture.get(cv2_module.CAP_PROP_FPS))
        except (TypeError, ValueError, AttributeError, OverflowError):
            source_fps = 0.0

        while config.max_frames == 0 or stats.frames < config.max_frames:
            try:
                ok, frame = capture.read()
            except Exception as exc:
                raise MediaError(f"could not read a frame from {source}: {exc}") from exc
            if not ok or frame is None:
                break
            if config.mirror:
                frame = cv2_module.flip(frame, 1)

            timestamp = max(
                last_timestamp + 1, _timestamps(source_fps, stats.frames, started, clock)
            )
            last_timestamp = timestamp
            # A recorded clip is replayed on its own timeline, so the state
            # machine sees the durations the footage actually contains rather
            # than however long decoding happened to take.
            if config.input_path is not None and source_fps > 0.0:
                now_seconds = stats.frames / source_fps
            else:
                now_seconds = clock() - started

            frame_started = clock()
            result = pipeline.process(frame, timestamp, now_seconds, last_elapsed)
            annotated = pipeline.annotate(frame, result, now_seconds)
            last_elapsed = max(clock() - frame_started, 1e-9)
            wall_seconds += last_elapsed
            # Durations in the stats follow the same timeline the state machine
            # ran on. For a recorded clip that is footage time, so "alarm
            # seconds" reports how long the eyes were actually shut rather than
            # how long this machine took to decode the frames.
            stats.observe(result, max(0.0, now_seconds - previous_now))
            previous_now = now_seconds

            if config.output_path is not None:
                if writer is None:
                    writer = _make_writer(config.output_path, annotated, source_fps, cv2_module)
                try:
                    writer.write(annotated)
                except Exception as exc:
                    raise MediaError(f"could not write annotated video frame: {exc}") from exc
            if config.display and not _show(annotated, cv2_module):
                break
    finally:
        stats.elapsed_seconds = wall_seconds
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        try:
            pipeline.close()
        finally:
            if config.display:
                try:
                    cv2_module.destroyAllWindows()
                except Exception:
                    pass

    if stats.frames == 0:
        raise MediaError("no frames could be read from the selected video source")
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        config = config_from_args(parser.parse_args(argv))
        if config.calibrate_seconds > 0.0:
            run_calibration(config)
            return 0
        stats = run(config)
    except KeyboardInterrupt:
        print("Stopped by user.", file=sys.stderr)
        return 130
    except (DrowsinessError, OSError, ValueError, cv2.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(stats.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
