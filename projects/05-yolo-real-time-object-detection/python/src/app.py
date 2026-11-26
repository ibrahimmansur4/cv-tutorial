"""CLI and media loop for real-time YOLO object detection."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2

from .decode import ModelFamily
from .detector import YoloDetector
from .errors import ConfigurationError, DetectionError, MediaError
from .labels import load_labels, resolve_class_filter
from .models import (
    DEFAULT_MODEL,
    MODELS_DIR,
    REGISTRY,
    ModelSpec,
    available_models,
    family_for_path,
    get_spec,
)
from .pipeline import DetectionPipeline, RunStats, format_counts

VIDEO_CODECS = {".avi": "MJPG", ".mp4": "mp4v", ".m4v": "mp4v", ".mov": "mp4v"}
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
WINDOW_NAME = "YOLO Object Detection"


@dataclass(frozen=True, slots=True)
class AppConfig:
    camera: int
    input_path: Path | None
    output_path: Path | None
    model_path: Path
    family: ModelFamily
    model_label: str
    labels_path: Path | None
    input_size: tuple[int, int]
    resolution: tuple[int, int]
    confidence: float
    nms: float
    classes: tuple[str, ...] | None
    max_frames: int
    display: bool
    mirror: bool
    show_timing: bool


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


def parse_square_size(value: str) -> tuple[int, int]:
    """Accept either `640` or `640x640` for the network input."""
    text = value.strip().lower()
    if "x" not in text:
        try:
            side = int(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("value must be an integer or WIDTHxHEIGHT") from exc
        if side <= 0:
            raise argparse.ArgumentTypeError("network input size must be positive")
        return side, side
    return parse_size(text)


def probability(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a number between 0 and 1") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return result


def nonnegative_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a non-negative integer") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return result


def class_list(value: str) -> tuple[str, ...]:
    tokens = tuple(token.strip() for token in value.split(",") if token.strip())
    if not tokens:
        raise argparse.ArgumentTypeError("--classes needs at least one name or index")
    return tokens


def build_parser() -> argparse.ArgumentParser:
    registered = ", ".join(sorted(REGISTRY))
    parser = argparse.ArgumentParser(
        description="Detect objects in a webcam, video, or image with a pretrained YOLO model.",
        epilog=f"Registered models: {registered}. Use --list-models for details.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="video or image file instead of the webcam")
    source.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    parser.add_argument("--output", type=Path, help="write the annotated video or image here")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"registered model name or a path to an .onnx file (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--family",
        choices=[family.value for family in ModelFamily],
        help="output layout of a custom --model; inferred from the file name when omitted",
    )
    parser.add_argument("--labels", type=Path, help="class-names file, one label per line")
    parser.add_argument(
        "--input-size",
        type=parse_square_size,
        metavar="SIZE",
        help="network input size, e.g. 320 or 640x640 (default: the model's own size)",
    )
    parser.add_argument(
        "--resolution",
        type=parse_size,
        default=(1280, 720),
        metavar="WIDTHxHEIGHT",
        help="requested webcam capture resolution (default: 1280x720)",
    )
    parser.add_argument(
        "--confidence",
        type=probability,
        default=0.25,
        help="minimum detection confidence (default: 0.25)",
    )
    parser.add_argument(
        "--nms",
        type=probability,
        default=0.45,
        help="IoU threshold for non-maximum suppression (default: 0.45)",
    )
    parser.add_argument(
        "--classes",
        type=class_list,
        help="keep only these comma-separated class names or indices, e.g. person,car",
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
        "--show-timing", action="store_true", help="overlay the per-stage millisecond breakdown"
    )
    mirror = parser.add_mutually_exclusive_group()
    mirror.add_argument(
        "--mirror", dest="mirror", action="store_true", help="mirror webcam frames"
    )
    mirror.add_argument(
        "--no-mirror", dest="mirror", action="store_false", help="do not mirror frames"
    )
    parser.set_defaults(mirror=None)
    parser.add_argument(
        "--list-models", action="store_true", help="print the registered models and exit"
    )
    return parser


def _resolve_model(args: argparse.Namespace) -> tuple[Path, ModelFamily, str, tuple[int, int]]:
    """Map --model/--family onto a concrete file, layout, and default input size."""
    requested = str(args.model).strip()
    spec: ModelSpec | None = REGISTRY.get(requested.casefold())
    if spec is not None:
        family = ModelFamily(args.family) if args.family else spec.family
        label = f"model: {spec.name} ({family.value})"
        # A statically exported model rejects any other size deep inside the
        # forward pass, so catch it here where the message can be useful.
        if (
            args.input_size is not None
            and not spec.dynamic_input
            and tuple(args.input_size) != tuple(spec.input_size)
        ):
            native = f"{spec.input_size[0]}x{spec.input_size[1]}"
            dynamic = ", ".join(
                other.name for other in available_models() if other.dynamic_input
            )
            raise ConfigurationError(
                f"{spec.name} was exported with a fixed {native} input and cannot run at "
                f"{args.input_size[0]}x{args.input_size[1]}. Drop --input-size, or use a "
                f"model with dynamic input axes ({dynamic or 'none registered'})."
            )
        return spec.path, family, label, spec.input_size

    candidate = Path(requested).expanduser()
    if not candidate.is_file():
        # A bare unknown name is far more likely a typo than a path, so point at
        # the registry instead of reporting a missing file.
        if not candidate.suffix:
            raise ConfigurationError(
                f"unknown model '{requested}'. Registered models: "
                f"{', '.join(sorted(REGISTRY))}, or pass a path to an .onnx file."
            )
        raise ConfigurationError(f"model file not found: {candidate.resolve()}")
    family = ModelFamily(args.family) if args.family else family_for_path(candidate)
    label = f"model: {candidate.stem} ({family.value})"
    return candidate, family, label, (640, 640)


def config_from_args(args: argparse.Namespace) -> AppConfig:
    model_path, family, label, default_input = _resolve_model(args)
    if args.camera < 0:
        raise ConfigurationError("--camera cannot be negative")
    input_size = args.input_size or default_input
    if input_size[0] % 32 or input_size[1] % 32:
        raise ConfigurationError("--input-size must be a multiple of 32")
    # Mirroring a recorded file or a photo is surprising; mirroring a webcam
    # preview is what makes it feel like a mirror. Default per source.
    mirror = args.mirror if args.mirror is not None else args.input is None
    return AppConfig(
        camera=args.camera,
        input_path=args.input,
        output_path=args.output,
        model_path=model_path,
        family=family,
        model_label=label,
        labels_path=args.labels,
        input_size=input_size,
        resolution=args.resolution,
        confidence=args.confidence,
        nms=args.nms,
        classes=args.classes,
        max_frames=args.max_frames,
        display=not args.no_display,
        mirror=mirror,
        show_timing=args.show_timing,
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


def _build_detector(config: AppConfig, cv2_module: Any) -> YoloDetector:
    labels = load_labels(config.labels_path)
    class_filter = resolve_class_filter(config.classes, labels)
    detector = YoloDetector(
        config.model_path,
        config.family,
        input_size=config.input_size,
        confidence_threshold=config.confidence,
        nms_threshold=config.nms,
        labels=labels,
        class_filter=class_filter,
        cv2_module=cv2_module,
    )
    detector.warm_up()
    return detector


def _open_capture(config: AppConfig, cv2_module: Any) -> tuple[Any, int | str]:
    if config.input_path is not None:
        source_path = config.input_path.expanduser().resolve()
        if not source_path.is_file():
            raise MediaError(f"input file not found: {source_path}")
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


def run_image(
    config: AppConfig,
    *,
    cv2_module: Any = cv2,
    detector: YoloDetector | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> RunStats:
    """Detect in a single still image."""
    assert config.input_path is not None
    source = config.input_path.expanduser().resolve()
    if config.output_path is not None and _paths_collide(source, config.output_path):
        raise MediaError("input and output must be different files")
    frame = cv2_module.imread(str(source))
    if frame is None:
        raise MediaError(f"could not read image: {source}")

    detector = detector or _build_detector(config, cv2_module)
    pipeline = DetectionPipeline(
        detector,
        model_label=config.model_label,
        show_timing=config.show_timing,
        cv2_module=cv2_module,
    )
    started = clock()
    result = pipeline.process(frame, 0.0)
    elapsed = max(clock() - started, 1e-9)
    # A still image has no previous frame to time against, so the rate is only
    # known after the fact. Restate it before drawing, or the overlay would
    # report 0.0 while the printed summary reported the real figure.
    result = replace(result, fps=1.0 / elapsed)
    annotated = pipeline.annotate(frame, result)

    if config.output_path is not None:
        output = config.output_path.expanduser().resolve()
        if output.suffix.lower() not in IMAGE_SUFFIXES:
            raise MediaError(
                f"image output needs an image suffix such as .jpg or .png, got '{output.suffix}'"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2_module.imwrite(str(output), annotated):
            raise MediaError(f"could not write annotated image: {output}")

    stats = RunStats()
    stats.observe(result)
    stats.elapsed_seconds = elapsed
    print(format_counts(result.counts, result.fps))
    if config.display:
        try:
            cv2_module.imshow(WINDOW_NAME, annotated)
            cv2_module.waitKey(0)
        except Exception as exc:
            raise MediaError("OpenCV display failed; retry with --no-display") from exc
        finally:
            try:
                cv2_module.destroyAllWindows()
            except Exception:
                pass
    return stats


def run(
    config: AppConfig,
    *,
    cv2_module: Any = cv2,
    detector: YoloDetector | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> RunStats:
    """Detect over a webcam or video stream."""
    detector = detector or _build_detector(config, cv2_module)
    pipeline = DetectionPipeline(
        detector,
        model_label=config.model_label,
        show_timing=config.show_timing,
        cv2_module=cv2_module,
    )

    capture = None
    writer = None
    stats = RunStats()
    wall_seconds = 0.0
    last_elapsed = 0.0
    source_fps = 0.0
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

            started = clock()
            result = pipeline.process(frame, last_elapsed)
            annotated = pipeline.annotate(frame, result)
            last_elapsed = max(clock() - started, 1e-9)
            wall_seconds += last_elapsed
            stats.observe(result)

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
        if config.display:
            try:
                cv2_module.destroyAllWindows()
            except Exception:
                pass

    if stats.frames == 0:
        raise MediaError("no frames could be read from the selected video source")
    return stats


def _print_models() -> None:
    print(f"{'name':10} {'layout':8} {'size':>8}  {'input':9} description")
    for spec in available_models():
        present = "yes" if spec.path.is_file() else "no"
        size = f"{spec.size_mb:.1f} MB"
        resolution = f"{spec.input_size[0]}x{spec.input_size[1]}"
        print(
            f"{spec.name:10} {spec.family.value:8} {size:>8}  {resolution:9} "
            f"{spec.description} (downloaded: {present})"
        )
    print(f"\nModels are stored in {MODELS_DIR}")


def _is_image(path: Path | None) -> bool:
    return path is not None and path.suffix.casefold() in IMAGE_SUFFIXES


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.list_models:
            _print_models()
            return 0
        config = config_from_args(args)
        stats = run_image(config) if _is_image(config.input_path) else run(config)
    except KeyboardInterrupt:
        print("Stopped by user.", file=sys.stderr)
        return 130
    except (DetectionError, OSError, ValueError, cv2.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(stats.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
