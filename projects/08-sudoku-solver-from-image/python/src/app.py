"""CLI and media loop for the image Sudoku solver."""

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
import numpy as np

from .classify import make_classifier
from .errors import (
    BoardNotFoundError,
    ConfigurationError,
    MediaError,
    SudokuError,
    UnsolvableError,
)
from .pipeline import PipelineStats, SudokuPipeline
from .render import cell_mosaic, draw_recognition_debug
from .solver import format_board

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "mnist-8.onnx"
DEFAULT_RESULTS = PROJECT_ROOT / "results"
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
VIDEO_CODECS = {".avi": "MJPG", ".mp4": "mp4v", ".m4v": "mp4v", ".mov": "mp4v"}
WINDOW_NAME = "Sudoku Solver"
FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass(frozen=True, slots=True)
class AppConfig:
    camera: int
    input_path: Path | None
    output_path: Path | None
    classifier: str
    model_path: Path
    results_dir: Path
    resolution: tuple[int, int]
    board_size: int
    block_size: int
    constant: int
    min_area_ratio: float
    margin_ratio: float
    min_ink_ratio: float
    max_frames: int
    display: bool
    mirror: bool
    draw_givens: bool
    check_unique: bool
    debug: bool


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


def unit_interval(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be between 0 and 1") from exc
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1 (exclusive)")
    return result


def odd_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an odd integer of at least 3") from exc
    if result < 3 or result % 2 == 0:
        raise argparse.ArgumentTypeError("value must be an odd integer of at least 3")
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
            "Find a Sudoku grid in an image, video, or webcam frame, read the digits, "
            "solve it, and project the answer back onto the original."
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="image or video file instead of the webcam")
    source.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    parser.add_argument("--output", type=Path, help="write the annotated result here")
    parser.add_argument(
        "--classifier",
        choices=("template", "onnx"),
        default="template",
        help="digit classifier (default: template, which needs no download)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"ONNX digit model, used with --classifier onnx (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
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
    parser.add_argument("--draw-givens", action="store_true", help="redraw the given digits too")
    parser.add_argument(
        "--no-unique-check",
        action="store_true",
        help="skip verifying the puzzle has exactly one solution",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="also write the warped board, cell mosaic, and recognition overlay",
    )
    mirror = parser.add_mutually_exclusive_group()
    mirror.add_argument("--mirror", dest="mirror", action="store_true", help="mirror frames")
    mirror.add_argument(
        "--no-mirror", dest="mirror", action="store_false", help="do not mirror frames"
    )
    parser.set_defaults(mirror=None)

    tuning = parser.add_argument_group("detection tuning")
    tuning.add_argument(
        "--board-size", type=int, default=450, help="flattened board size in pixels (default: 450)"
    )
    tuning.add_argument(
        "--block-size", type=odd_integer, default=19, help="adaptive threshold block (default: 19)"
    )
    tuning.add_argument(
        "--constant", type=int, default=4, help="adaptive threshold constant (default: 4)"
    )
    tuning.add_argument(
        "--min-area-ratio",
        type=unit_interval,
        default=0.04,
        help="smallest grid area as a fraction of the frame (default: 0.04)",
    )
    tuning.add_argument(
        "--margin-ratio",
        type=unit_interval,
        default=0.12,
        help="fraction trimmed from each cell edge (default: 0.12)",
    )
    tuning.add_argument(
        "--min-ink-ratio",
        type=unit_interval,
        default=0.035,
        help="ink fraction below which a cell is empty (default: 0.035)",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> AppConfig:
    if args.camera < 0:
        raise ConfigurationError("--camera cannot be negative")
    if args.board_size < 90:
        raise ConfigurationError("--board-size must be at least 90")
    if args.classifier == "onnx" and not args.model.expanduser().is_file():
        raise ConfigurationError(
            f"ONNX model not found: {args.model}. Run `python scripts/download_model.py`."
        )
    mirror = args.mirror if args.mirror is not None else args.input is None
    return AppConfig(
        camera=args.camera,
        input_path=args.input,
        output_path=args.output,
        classifier=args.classifier,
        model_path=args.model,
        results_dir=args.results,
        resolution=args.resolution,
        board_size=args.board_size,
        block_size=args.block_size,
        constant=args.constant,
        min_area_ratio=args.min_area_ratio,
        margin_ratio=args.margin_ratio,
        min_ink_ratio=args.min_ink_ratio,
        max_frames=args.max_frames,
        display=not args.no_display,
        mirror=mirror,
        draw_givens=args.draw_givens,
        check_unique=not args.no_unique_check,
        debug=args.debug,
    )


def _build_pipeline(config: AppConfig, cv2_module: Any) -> SudokuPipeline:
    classifier = make_classifier(
        config.classifier, model_path=config.model_path, cv2_module=cv2_module
    )
    return SudokuPipeline(
        classifier,
        board_size=config.board_size,
        cell_block_size=config.block_size,
        cell_constant=config.constant,
        min_area_ratio=config.min_area_ratio,
        margin_ratio=config.margin_ratio,
        min_ink_ratio=config.min_ink_ratio,
        check_unique=config.check_unique,
        cv2_module=cv2_module,
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


def _write_debug(
    config: AppConfig, pipeline: SudokuPipeline, outcome: Any, cv2_module: Any
) -> None:
    directory = config.results_dir.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    cv2_module.imwrite(str(directory / "warped_board.png"), outcome.board_square)
    cv2_module.imwrite(
        str(directory / "cell_mosaic.png"),
        cell_mosaic(outcome.cells, cv2_module=cv2_module),
    )
    cv2_module.imwrite(
        str(directory / "recognition.png"),
        draw_recognition_debug(
            outcome.board_square,
            outcome.givens,
            outcome.confidence,
            cv2_module=cv2_module,
        ),
    )
    print(f"Debug images written to {directory}")


def _overlay_message(
    frame: np.ndarray, message: str, *, cv2_module: Any = cv2
) -> np.ndarray:
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    scale = float(np.clip(width / 1400.0, 0.5, 0.9))
    thickness = 1 if scale < 0.6 else 2
    (text_width, text_height), baseline = cv2_module.getTextSize(
        message, FONT, scale, thickness
    )
    x = max(10, (width - text_width) // 2)
    y = height - 20
    overlay = canvas.copy()
    cv2_module.rectangle(
        overlay,
        (x - 12, y - text_height - baseline - 8),
        (min(width, x + text_width + 12), y + baseline + 4),
        (20, 20, 20),
        -1,
    )
    cv2_module.addWeighted(overlay, 0.55, canvas, 0.45, 0.0, dst=canvas)
    cv2_module.putText(
        canvas, message, (x, y), FONT, scale, (235, 235, 235), thickness, cv2_module.LINE_AA
    )
    return canvas


def run_image(
    config: AppConfig,
    *,
    cv2_module: Any = cv2,
    pipeline: SudokuPipeline | None = None,
) -> PipelineStats:
    """Solve a single still image."""
    assert config.input_path is not None
    source = config.input_path.expanduser().resolve()
    if config.output_path is not None and _paths_collide(source, config.output_path):
        raise MediaError("input and output must be different files")
    frame = cv2_module.imread(str(source))
    if frame is None:
        raise MediaError(f"could not read image: {source}")

    pipeline = pipeline or _build_pipeline(config, cv2_module)
    stats = PipelineStats()
    started = time.perf_counter()
    outcome = pipeline.process(frame)
    stats.frames = 1
    stats.boards_found = 1
    stats.solved = 1
    stats.elapsed_seconds = max(time.perf_counter() - started, 1e-9)

    annotated = pipeline.annotate(frame, outcome, draw_givens=config.draw_givens)
    print(f"Read {outcome.given_count} given digit(s):\n")
    print(format_board(outcome.givens))
    print(f"\nSolved ({outcome.stats.summary()}):\n")
    print(format_board(outcome.solution))
    if not outcome.unique:
        print(
            "\nwarning: this grid has more than one solution, which usually means a "
            "digit was misread. The answer shown is only one of several."
        )
    if outcome.min_confidence < 0.1:
        print(
            f"warning: lowest digit confidence was {outcome.min_confidence:.2f}; "
            "run with --debug to see what the classifier saw."
        )
    if config.debug:
        _write_debug(config, pipeline, outcome, cv2_module)

    if config.output_path is not None:
        output = config.output_path.expanduser().resolve()
        if output.suffix.lower() not in IMAGE_SUFFIXES:
            raise MediaError(
                f"image output needs an image suffix such as .png, got '{output.suffix}'"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2_module.imwrite(str(output), annotated):
            raise MediaError(f"could not write annotated image: {output}")
        print(f"\nWrote {output}")

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
    pipeline: SudokuPipeline | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> PipelineStats:
    """Solve continuously from a webcam or video."""
    pipeline = pipeline or _build_pipeline(config, cv2_module)
    capture = None
    writer = None
    stats = PipelineStats()
    wall_seconds = 0.0
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

            frame_started = clock()
            message = "Point the camera at a Sudoku grid"
            annotated = frame
            try:
                outcome = pipeline.process(frame)
                stats.boards_found += 1
                stats.solved += 1
                if outcome.from_cache:
                    stats.cache_hits += 1
                annotated = pipeline.annotate(
                    frame, outcome, draw_givens=config.draw_givens
                )
                message = (
                    f"Solved {outcome.given_count} givens"
                    + ("" if outcome.unique else "  (multiple solutions - check the read)")
                )
            except BoardNotFoundError:
                pass
            except UnsolvableError as exc:
                stats.boards_found += 1
                stats.unsolvable += 1
                # Keep showing the last good answer rather than flickering off
                # because one frame misread a digit.
                last = pipeline.last_outcome
                if last is not None:
                    try:
                        annotated = pipeline.annotate(
                            frame, last, draw_givens=config.draw_givens
                        )
                    except cv2.error:
                        annotated = frame
                message = str(exc).split(";")[0]
            annotated = _overlay_message(annotated, message, cv2_module=cv2_module)
            wall_seconds += max(clock() - frame_started, 1e-9)
            stats.frames += 1

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


def _is_image(path: Path | None) -> bool:
    return path is not None and path.suffix.casefold() in IMAGE_SUFFIXES


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        config = config_from_args(parser.parse_args(argv))
        stats = run_image(config) if _is_image(config.input_path) else run(config)
    except KeyboardInterrupt:
        print("Stopped by user.", file=sys.stderr)
        return 130
    except (SudokuError, OSError, ValueError, cv2.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if config.input_path is None or not _is_image(config.input_path):
        print(stats.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
