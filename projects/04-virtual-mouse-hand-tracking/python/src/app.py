"""CLI and media loop for the hand-tracked virtual mouse."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2

from .controller import PreviewMouseController, PyAutoGuiMouseController
from .errors import ConfigurationError, MediaError, VirtualMouseError
from .gestures import GestureClassifier, GestureDebouncer
from .hand_tracking import MediaPipeHandTracker
from .pipeline import VirtualMousePipeline
from .smoothing import ExponentialCursorSmoother

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "hand_landmarker.task"
VIDEO_CODECS = {".avi": "MJPG", ".mp4": "mp4v", ".m4v": "mp4v", ".mov": "mp4v"}
WINDOW_NAME = "Virtual Mouse - MediaPipe Hands"


@dataclass(frozen=True, slots=True)
class AppConfig:
    camera: int
    input_path: Path | None
    output_path: Path | None
    model_path: Path
    resolution: tuple[int, int]
    preview_screen: tuple[int, int]
    max_frames: int
    display: bool
    mirror: bool
    enable_control: bool
    num_hands: int
    detection_confidence: float
    presence_confidence: float
    tracking_confidence: float
    pinch_threshold: float
    stable_frames: int
    click_cooldown: float
    smoothing_alpha: float
    dead_zone: float
    active_margin: float


@dataclass(slots=True)
class RunStats:
    frames: int = 0
    hands: int = 0
    elapsed_seconds: float = 0.0
    actions: Counter[str] = field(default_factory=Counter)
    control_enabled: bool = False

    @property
    def fps(self) -> float:
        return self.frames / self.elapsed_seconds if self.elapsed_seconds > 0.0 else 0.0

    def summary(self) -> str:
        mode = "control" if self.control_enabled else "safe preview"
        action_text = ", ".join(f"{key}={value}" for key, value in sorted(self.actions.items()))
        return (
            f"Processed {self.frames} frame(s); observed {self.hands} hand(s); "
            f"mode: {mode}; actions: {action_text or 'none'}; average processing FPS: {self.fps:.1f}"
        )


class FpsMeter:
    def __init__(self, window_size: int = 30) -> None:
        if window_size <= 0:
            raise ValueError("FPS window must be positive")
        self._durations: deque[float] = deque(maxlen=window_size)

    def update(self, elapsed: float) -> float:
        if math.isfinite(elapsed) and elapsed > 0.0:
            self._durations.append(elapsed)
        duration = sum(self._durations)
        return len(self._durations) / duration if duration > 0.0 else 0.0


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


def nonnegative_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a non-negative number") from exc
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("value must be a non-negative number")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Track a hand with MediaPipe and preview or explicitly enable gesture-based "
            "mouse control."
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="recorded video input (preview only)")
    source.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    parser.add_argument("--output", type=Path, help="optional annotated video destination")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"MediaPipe Hand Landmarker task model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--resolution",
        type=parse_size,
        default=(1280, 720),
        metavar="WIDTHxHEIGHT",
        help="requested webcam resolution (default: 1280x720)",
    )
    parser.add_argument(
        "--screen-size",
        type=parse_size,
        default=(1920, 1080),
        metavar="WIDTHxHEIGHT",
        help="preview-mode virtual screen size (default: 1920x1080)",
    )
    parser.add_argument(
        "--max-frames",
        type=nonnegative_integer,
        default=0,
        metavar="N",
        help="stop after N frames; 0 has no limit (default: 0)",
    )
    parser.add_argument("--no-display", action="store_true", help="disable the OpenCV window")
    mirror = parser.add_mutually_exclusive_group()
    mirror.add_argument("--mirror", dest="mirror", action="store_true", help="mirror frames")
    mirror.add_argument(
        "--no-mirror", dest="mirror", action="store_false", help="do not mirror frames"
    )
    parser.set_defaults(mirror=True)
    parser.add_argument(
        "--enable-control",
        action="store_true",
        help="allow gestures to move/click the real OS pointer; default is safe preview",
    )

    tracking = parser.add_argument_group("MediaPipe tracking")
    tracking.add_argument("--num-hands", type=positive_integer, default=1)
    tracking.add_argument("--detection-confidence", type=probability, default=0.5)
    tracking.add_argument("--presence-confidence", type=probability, default=0.5)
    tracking.add_argument("--tracking-confidence", type=probability, default=0.5)

    gesture = parser.add_argument_group("gesture and cursor tuning")
    gesture.add_argument("--pinch-threshold", type=positive_float, default=0.32)
    gesture.add_argument("--stable-frames", type=positive_integer, default=3)
    gesture.add_argument("--click-cooldown", type=nonnegative_float, default=0.45)
    gesture.add_argument("--smoothing-alpha", type=positive_float, default=0.35)
    gesture.add_argument("--dead-zone", type=nonnegative_float, default=2.0)
    gesture.add_argument("--active-margin", type=probability, default=0.08)
    return parser


def config_from_args(args: argparse.Namespace) -> AppConfig:
    if args.smoothing_alpha > 1.0:
        raise ConfigurationError("--smoothing-alpha must be at most 1")
    if args.active_margin >= 0.5:
        raise ConfigurationError("--active-margin must be below 0.5")
    if args.enable_control and args.input is not None:
        raise ConfigurationError("recorded video cannot control the real mouse")
    if args.camera < 0:
        raise ConfigurationError("--camera cannot be negative")
    return AppConfig(
        camera=args.camera,
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        resolution=args.resolution,
        preview_screen=args.screen_size,
        max_frames=args.max_frames,
        display=not args.no_display,
        mirror=args.mirror,
        enable_control=args.enable_control,
        num_hands=args.num_hands,
        detection_confidence=args.detection_confidence,
        presence_confidence=args.presence_confidence,
        tracking_confidence=args.tracking_confidence,
        pinch_threshold=args.pinch_threshold,
        stable_frames=args.stable_frames,
        click_cooldown=args.click_cooldown,
        smoothing_alpha=args.smoothing_alpha,
        dead_zone=args.dead_zone,
        active_margin=args.active_margin,
    )


def _paths_collide(first: Path, second: Path) -> bool:
    first_resolved = first.expanduser().resolve()
    second_resolved = second.expanduser().resolve()
    if first_resolved == second_resolved:
        return True
    try:
        return first_resolved.exists() and second_resolved.exists() and os.path.samefile(
            first_resolved, second_resolved
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
        raise MediaError(f"could not open video source: {source}")
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


def run(
    config: AppConfig,
    *,
    cv2_module: Any = cv2,
    tracker: Any = None,
    controller: Any = None,
    clock: Callable[[], float] = time.perf_counter,
) -> RunStats:
    if controller is None:
        if config.enable_control:
            controller = PyAutoGuiMouseController()
        else:
            controller = PreviewMouseController(*config.preview_screen)
    if tracker is None:
        try:
            tracker = MediaPipeHandTracker(
                config.model_path,
                num_hands=config.num_hands,
                detection_confidence=config.detection_confidence,
                presence_confidence=config.presence_confidence,
                tracking_confidence=config.tracking_confidence,
                cv2_module=cv2_module,
            )
        except Exception:
            try:
                controller.close()
            except Exception:
                pass
            raise
    try:
        pipeline = VirtualMousePipeline(
            tracker,
            controller,
            classifier=GestureClassifier(config.pinch_threshold),
            debouncer=GestureDebouncer(config.stable_frames, config.click_cooldown),
            smoother=ExponentialCursorSmoother(config.smoothing_alpha, config.dead_zone),
            active_margin=config.active_margin,
            control_enabled=config.enable_control,
            cv2_module=cv2_module,
        )
    except Exception:
        try:
            controller.close()
        finally:
            tracker.close()
        raise

    capture = None
    writer = None
    stats = RunStats(control_enabled=config.enable_control)
    processing_seconds = 0.0
    source_fps = 0.0
    started_at = clock()
    last_timestamp_ms = -1
    fps_meter = FpsMeter()
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

            if math.isfinite(source_fps) and source_fps > 0.0:
                timestamp_ms = int(round(stats.frames * 1000.0 / source_fps))
            else:
                timestamp_ms = int(round((clock() - started_at) * 1000.0))
            timestamp_ms = max(last_timestamp_ms + 1, timestamp_ms)
            last_timestamp_ms = timestamp_ms

            processing_started = clock()
            result = pipeline.process(frame, timestamp_ms, clock())
            elapsed = max(clock() - processing_started, 1e-9)
            processing_seconds += elapsed
            fps = fps_meter.update(elapsed)
            annotated = pipeline.annotate(frame, result, fps)

            stats.frames += 1
            stats.hands += len(result.hands)
            stats.actions.update(result.actions)

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
        stats.elapsed_seconds = processing_seconds
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
        stats = run(config)
    except KeyboardInterrupt:
        print("Stopped by user; mouse buttons were released.", file=sys.stderr)
        return 130
    except (VirtualMouseError, OSError, ValueError, cv2.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(stats.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
