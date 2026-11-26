"""Command-line application for face-recognition attendance."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from .attendance_log import AttendanceLogger
from .detection import FaceBox, FaceDetector
from .embeddings import HOGFaceEmbedder, KnownFaceStore
from .errors import AttendanceError, ConfigurationError, FaceDataError, MediaError
from .pipeline import RecognitionPipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KNOWN_FACES = PROJECT_ROOT / "known_faces"
DEFAULT_CACHE = PROJECT_ROOT / "data" / "embeddings.npz"
DEFAULT_ATTENDANCE = PROJECT_ROOT / "data" / "attendance.csv"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}$")
WINDOW_NAME = "Face Recognition Attendance"


@dataclass(slots=True)
class RunStats:
    frames: int = 0
    faces: int = 0
    recognized: int = 0
    attendance_rows: int = 0
    elapsed_seconds: float = 0.0

    @property
    def fps(self) -> float:
        return self.frames / self.elapsed_seconds if self.elapsed_seconds > 0.0 else 0.0

    def summary(self) -> str:
        return (
            f"Processed {self.frames} frame(s); observed {self.faces} face(s); "
            f"recognized {self.recognized}; recorded {self.attendance_rows} new "
            f"attendance row(s); average processing FPS: {self.fps:.1f}"
        )


def parse_resolution(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", value)
    if not match:
        raise argparse.ArgumentTypeError("resolution must use WIDTHxHEIGHT, for example 1280x720")
    width, height = (int(match.group(1)), int(match.group(2)))
    if width < 1 or height < 1:
        raise argparse.ArgumentTypeError("resolution dimensions must be positive")
    return width, height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recognize faces with Haar + HOG, record daily attendance, or enroll a person."
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="image or video file to process")
    source.add_argument("--camera", type=int, help="camera index (default: 0)")
    parser.add_argument(
        "--enroll",
        metavar="NAME",
        help="capture known-face samples from a webcam instead of recognizing",
    )
    parser.add_argument(
        "--known-faces",
        type=Path,
        default=DEFAULT_KNOWN_FACES,
        help=f"known-face image directory (default: {DEFAULT_KNOWN_FACES})",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE,
        help=f"safe NPZ descriptor cache (default: {DEFAULT_CACHE})",
    )
    parser.add_argument(
        "--attendance",
        type=Path,
        default=DEFAULT_ATTENDANCE,
        help=f"attendance CSV (default: {DEFAULT_ATTENDANCE})",
    )
    parser.add_argument("--output", type=Path, help="annotated image/video destination")
    parser.add_argument("--cascade", type=Path, help="custom Haar cascade XML")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="maximum cosine distance for a match (default: 0.35)",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.1,
        help="Haar scale factor (default: 1.1)",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=5,
        help="Haar minimum neighbors (default: 5)",
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=48,
        help="minimum detected face width/height in pixels (default: 48)",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="ignore any current descriptor cache and rebuild it",
    )
    parser.add_argument(
        "--resolution",
        type=parse_resolution,
        default=None,
        help="requested webcam resolution as WIDTHxHEIGHT",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="stop video/webcam processing after this many frames",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=8,
        help="number of webcam face samples to enroll (default: 8)",
    )
    parser.add_argument(
        "--capture-every",
        type=int,
        default=8,
        help="save one enrollment sample every N eligible frames (default: 8)",
    )
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--display", dest="display", action="store_true", help="show annotated frames")
    display.add_argument("--no-display", dest="display", action="store_false", help="run without GUI windows")
    parser.set_defaults(display=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.threshold <= 2.0:
        raise ConfigurationError("--threshold must be between 0 and 2")
    if args.min_face_size < 1:
        raise ConfigurationError("--min-face-size must be positive")
    if args.max_frames is not None and args.max_frames < 1:
        raise ConfigurationError("--max-frames must be positive")
    if args.samples < 2:
        raise ConfigurationError("--samples must be at least 2")
    if args.capture_every < 1:
        raise ConfigurationError("--capture-every must be positive")
    if args.enroll and args.input is not None:
        raise ConfigurationError("--enroll captures from a webcam and cannot be combined with --input")
    if args.enroll and args.output is not None:
        raise ConfigurationError("--output is not used during enrollment")
    return args


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


def _validate_artifact_paths(args: argparse.Namespace) -> None:
    """Prevent generated files from overwriting an input or one another."""

    paths: list[tuple[str, Path]] = [("cache", args.cache)]
    if not args.enroll:
        paths.append(("attendance", args.attendance))
        if args.input is not None:
            paths.append(("input", args.input))
        if args.output is not None:
            paths.append(("output", args.output))
    for index, (first_name, first_path) in enumerate(paths):
        for second_name, second_path in paths[index + 1 :]:
            if _paths_collide(first_path, second_path):
                raise ConfigurationError(
                    f"--{first_name} and --{second_name} must refer to different files"
                )


def _open_capture(
    source: int | Path,
    cv2_module: Any,
    resolution: tuple[int, int] | None = None,
) -> Any:
    target: int | str = source if isinstance(source, int) else str(source)
    capture = cv2_module.VideoCapture(target)
    if not capture.isOpened():
        capture.release()
        kind = "camera" if isinstance(source, int) else "video"
        raise MediaError(f"could not open {kind}: {source}")
    if resolution is not None and isinstance(source, int):
        capture.set(cv2_module.CAP_PROP_FRAME_WIDTH, resolution[0])
        capture.set(cv2_module.CAP_PROP_FRAME_HEIGHT, resolution[1])
    return capture


def _show(frame: np.ndarray, cv2_module: Any, window_name: str = WINDOW_NAME) -> bool:
    """Show a frame and return ``False`` when Escape or Q is pressed."""

    try:
        cv2_module.imshow(window_name, frame)
        key = int(cv2_module.waitKey(1)) & 0xFF
    except cv2.error as exc:
        raise MediaError("GUI display failed; retry with --no-display") from exc
    return key not in (27, ord("q"), ord("Q"))


def _show_still(frame: np.ndarray, cv2_module: Any) -> None:
    """Show a still image until the user presses a key."""

    try:
        cv2_module.imshow(WINDOW_NAME, frame)
        cv2_module.waitKey(0)
    except cv2.error as exc:
        raise MediaError("GUI display failed; retry with --no-display") from exc


def _video_writer(
    output: Path,
    fps: float,
    frame_size: tuple[int, int],
    cv2_module: Any,
) -> Any:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".avi":
        codec = "MJPG"
    elif suffix in {".mp4", ".m4v", ".mov"}:
        codec = "mp4v"
    else:
        raise MediaError("video output must use .avi, .mp4, .m4v, or .mov")
    writer = cv2_module.VideoWriter(
        str(output),
        cv2_module.VideoWriter_fourcc(*codec),
        fps if np.isfinite(fps) and fps > 0.0 else 25.0,
        frame_size,
    )
    if not writer.isOpened():
        writer.release()
        raise MediaError(f"could not create output video: {output}")
    return writer


def _update_stats(stats: RunStats, observations: list[Any]) -> None:
    stats.frames += 1
    stats.faces += len(observations)
    stats.recognized += sum(observation.match.recognized for observation in observations)
    stats.attendance_rows += sum(observation.attendance_recorded for observation in observations)


def _process_image(
    path: Path,
    output: Path | None,
    pipeline: RecognitionPipeline,
    display: bool,
    cv2_module: Any,
    now_provider: Callable[[], datetime],
) -> RunStats:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise MediaError(f"input image was not found: {path}")
    image = cv2_module.imread(str(path), cv2_module.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise MediaError(f"could not decode input image: {path}")
    if output is not None:
        output = output.expanduser().resolve()
        if _paths_collide(path, output):
            raise MediaError("input and output must be different files")
        if output.suffix.lower() not in IMAGE_EXTENSIONS:
            raise MediaError("image output must use a supported image extension")
    started = time.perf_counter()
    observations = pipeline.process(image, timestamp=now_provider())
    annotated = pipeline.annotate(image, observations)
    stats = RunStats()
    _update_stats(stats, observations)
    stats.elapsed_seconds = time.perf_counter() - started

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2_module.imwrite(str(output), annotated):
            raise MediaError(f"could not write output image: {output}")
    if display:
        _show_still(annotated, cv2_module)
    return stats


def _process_stream(
    source: int | Path,
    output: Path | None,
    pipeline: RecognitionPipeline,
    display: bool,
    max_frames: int | None,
    resolution: tuple[int, int] | None,
    cv2_module: Any,
    now_provider: Callable[[], datetime],
) -> RunStats:
    if isinstance(source, Path):
        source = source.expanduser().resolve()
        if not source.is_file():
            raise MediaError(f"input video was not found: {source}")
        if output is not None and _paths_collide(source, output):
            raise MediaError("input and output must be different files")

    capture = _open_capture(source, cv2_module, resolution)
    writer = None
    stats = RunStats()
    started = time.perf_counter()
    try:
        while max_frames is None or stats.frames < max_frames:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            observations = pipeline.process(frame, timestamp=now_provider())
            annotated = pipeline.annotate(frame, observations)
            _update_stats(stats, observations)
            if output is not None:
                if writer is None:
                    fps = float(capture.get(cv2_module.CAP_PROP_FPS))
                    frame_size = (int(annotated.shape[1]), int(annotated.shape[0]))
                    writer = _video_writer(output, fps, frame_size, cv2_module)
                writer.write(annotated)
            if display and not _show(annotated, cv2_module):
                break
    finally:
        stats.elapsed_seconds = time.perf_counter() - started
        capture.release()
        if writer is not None:
            writer.release()
    if stats.frames == 0:
        raise MediaError(f"no frames could be read from {'camera' if isinstance(source, int) else 'video'}: {source}")
    return stats


def _validated_enrollment_name(name: str) -> str:
    clean = " ".join(name.strip().split())
    if not SAFE_NAME.fullmatch(clean) or clean.endswith((".", " ")):
        raise ConfigurationError(
            "enrollment name must begin with a letter/number and contain at most 64 "
            "letters, numbers, spaces, dots, underscores, or hyphens"
        )
    reserved_stem = clean.split(".", maxsplit=1)[0].rstrip(" .").upper()
    if reserved_stem in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{number}" for number in range(1, 10)),
        *(f"LPT{number}" for number in range(1, 10)),
    }:
        raise ConfigurationError("that enrollment name is reserved by Windows")
    return clean


def _expanded_crop(frame: np.ndarray, box: FaceBox, margin: float = 0.18) -> np.ndarray:
    x_margin = int(round(box.width * margin))
    y_margin = int(round(box.height * margin))
    expanded = FaceBox(
        box.x - x_margin,
        box.y - y_margin,
        box.width + 2 * x_margin,
        box.height + 2 * y_margin,
    ).clipped(frame.shape)
    return expanded.crop(frame)


def _enroll(
    name: str,
    camera_index: int,
    known_faces: Path,
    samples: int,
    capture_every: int,
    detector: FaceDetector,
    store: KnownFaceStore,
    display: bool,
    resolution: tuple[int, int] | None,
    max_frames: int | None,
    cv2_module: Any,
) -> int:
    clean_name = _validated_enrollment_name(name)
    known_root = known_faces.expanduser().resolve()
    person_dir = (known_root / clean_name).resolve()
    try:
        person_dir.relative_to(known_root)
    except ValueError as exc:
        raise ConfigurationError("enrollment destination escaped the known-faces directory") from exc
    person_dir.mkdir(parents=True, exist_ok=True)

    existing_numbers: list[int] = []
    for path in person_dir.glob("sample_*.png"):
        try:
            existing_numbers.append(int(path.stem.removeprefix("sample_")))
        except ValueError:
            continue
    next_number = max(existing_numbers, default=0) + 1
    captured = 0
    seen_frames = 0
    frame_limit = max_frames if max_frames is not None else max(samples * capture_every * 25, 300)
    capture = _open_capture(camera_index, cv2_module, resolution)
    try:
        while captured < samples and seen_frames < frame_limit:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            seen_frames += 1
            faces = detector.detect(frame)
            should_capture = len(faces) == 1 and seen_frames % capture_every == 0
            if should_capture:
                destination = person_dir / f"sample_{next_number + captured:03d}.png"
                if not cv2_module.imwrite(str(destination), _expanded_crop(frame, faces[0])):
                    raise MediaError(f"could not save enrollment sample: {destination}")
                captured += 1

            if display:
                preview = frame.copy()
                for index, box in enumerate(faces):
                    color = (60, 220, 60) if index == 0 else (80, 180, 255)
                    cv2_module.rectangle(preview, (box.x, box.y), (box.x2, box.y2), color, 2)
                cv2_module.putText(
                    preview,
                    f"Enrolling {clean_name}: {captured}/{samples} (Q to stop)",
                    (15, 30),
                    cv2_module.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (50, 230, 50),
                    2,
                    cv2_module.LINE_AA,
                )
                if not _show(preview, cv2_module, "Face Enrollment"):
                    break
    finally:
        capture.release()

    if captured < samples:
        raise MediaError(
            f"enrollment stopped after {captured}/{samples} samples; keep one face visible "
            "and try again"
        )
    index = store.load(force_rebuild=True)
    print(f"Enrolled {clean_name!r} with {captured} sample(s); cache now has {len(index)} embedding(s).")
    return captured


def run(
    args: argparse.Namespace,
    *,
    cv2_module: Any = cv2,
    now_provider: Callable[[], datetime] = datetime.now,
) -> RunStats | int:
    _validate_artifact_paths(args)
    detector = FaceDetector(
        args.cascade,
        scale_factor=args.scale_factor,
        min_neighbors=args.min_neighbors,
        min_size=(args.min_face_size, args.min_face_size),
        cv2_module=cv2_module,
    )
    embedder = HOGFaceEmbedder(cv2_module=cv2_module)
    store = KnownFaceStore(
        args.known_faces,
        args.cache,
        detector,
        embedder,
        cv2_module=cv2_module,
    )

    try:
        if args.enroll:
            camera_index = args.camera if args.camera is not None else 0
            return _enroll(
                args.enroll,
                camera_index,
                args.known_faces,
                args.samples,
                args.capture_every,
                detector,
                store,
                args.display,
                args.resolution,
                args.max_frames,
                cv2_module,
            )

        index = store.load(force_rebuild=args.rebuild_cache)
        report = store.last_report
        print(
            f"Known faces: {len(index)} embedding(s) from {report.discovered} image(s) "
            f"({'cache' if report.used_cache else 'rebuilt'})."
        )
        for warning in report.skipped:
            print(f"warning: skipped {warning}", file=sys.stderr)
        if not len(index):
            print(
                f"warning: no known faces loaded from {store.known_faces_dir}; detections "
                "will be labeled Unknown",
                file=sys.stderr,
            )
        logger = AttendanceLogger(args.attendance, now_provider=now_provider)
        pipeline = RecognitionPipeline(
            detector,
            embedder,
            index,
            logger,
            threshold=args.threshold,
            cv2_module=cv2_module,
        )
        if args.input is not None and args.input.suffix.lower() in IMAGE_EXTENSIONS:
            stats = _process_image(
                args.input,
                args.output,
                pipeline,
                args.display,
                cv2_module,
                now_provider,
            )
        else:
            source: int | Path
            source = args.input if args.input is not None else (args.camera or 0)
            stats = _process_stream(
                source,
                args.output,
                pipeline,
                args.display,
                args.max_frames,
                args.resolution,
                cv2_module,
                now_provider,
            )
        print(stats.summary())
        return stats
    finally:
        if args.display:
            try:
                cv2_module.destroyAllWindows()
            except cv2.error:
                pass


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        run(args)
        return 0
    except (AttendanceError, OSError, ValueError, cv2.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
