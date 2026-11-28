#!/usr/bin/env python3
"""Record EAR over a clip and sweep the closure threshold.

Answers the question the detector cannot answer for you: *where should the
threshold sit for this face?*

The script walks a video (or a live webcam capture), and for every frame logs

* the geometric EAR this project computes, and
* MediaPipe's `eyeBlinkLeft`/`eyeBlinkRight` blendshape.

The blendshape is a learned estimate of eye closure from the same model. It is
not ground truth, but it is an *independent* signal: it comes from a trained
head rather than from six landmark positions, so where the two agree the EAR is
probably reading the eye correctly, and where they diverge the landmarks are
likely being fitted badly.

The threshold sweep then replays the recorded EAR series through the real state
machine at a range of thresholds, reporting how many blinks and how many
drowsiness alarms each produces. A threshold that is too high turns every frame
into a closure; one that is too low never fires at all. The usable band is the
plateau between those two failures.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from src.calibration import percentile  # noqa: E402
from src.errors import ConfigurationError, DrowsinessError  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.state import AlertLevel, DrowsinessMonitor, MonitorConfig  # noqa: E402

DEFAULT_MODEL = PROJECT_ROOT / "models" / "face_landmarker.task"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "ear_trace.csv"
DEFAULT_SWEEP = PROJECT_ROOT / "results" / "threshold_sweep.csv"
TRACE_FIELDS = ("frame", "time_s", "ear", "left_ear", "right_ear", "mar", "blink_blendshape")
SWEEP_FIELDS = ("threshold", "blinks", "alarms", "alarm_seconds", "closed_fraction")


@dataclass(frozen=True, slots=True)
class Sample:
    frame: int
    time_s: float
    ear: float
    left_ear: float
    right_ear: float
    mar: float
    blink_blendshape: float


def _blink_score(blendshapes) -> float:
    """Average of the two eye-blink categories, or NaN when unavailable."""
    if not blendshapes:
        return float("nan")
    scores = {
        category.category_name: float(category.score)
        for category in blendshapes[0]
        if category.category_name in ("eyeBlinkLeft", "eyeBlinkRight")
    }
    if len(scores) != 2:
        return float("nan")
    return (scores["eyeBlinkLeft"] + scores["eyeBlinkRight"]) / 2.0


def collect(
    source: int | str,
    model_path: Path,
    *,
    seconds: float,
    max_frames: int,
    mirror: bool,
) -> list[Sample]:
    """Walk the source once, recording metrics per frame."""
    try:
        import mediapipe as mp
    except (ImportError, OSError) as exc:
        raise ConfigurationError(
            "MediaPipe is unavailable. Install python/requirements.txt."
        ) from exc

    resolved = model_path.expanduser().resolve()
    if not resolved.is_file():
        raise ConfigurationError(
            f"face model not found: {resolved}. Run `python scripts/download_model.py`."
        )

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(resolved)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
    )
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        landmarker.close()
        raise ConfigurationError(f"could not open video source: {source}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    live = isinstance(source, int)
    samples: list[Sample] = []
    started = time.perf_counter()
    last_timestamp = -1
    index = 0
    try:
        while True:
            if max_frames and index >= max_frames:
                break
            if live and time.perf_counter() - started >= seconds:
                break
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            if live or source_fps <= 0.0:
                now = time.perf_counter() - started
            else:
                now = index / source_fps
            timestamp = max(last_timestamp + 1, int(round(now * 1000.0)))
            last_timestamp = timestamp

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp
            )
            landmark_sets = getattr(result, "face_landmarks", ()) or ()
            if landmark_sets:
                from src.geometry import Point

                points = tuple(
                    Point(float(item.x), float(item.y), float(getattr(item, "z", 0.0) or 0.0))
                    for item in landmark_sets[0]
                )
                try:
                    metrics = compute_metrics(points, width / height)
                except DrowsinessError:
                    index += 1
                    continue
                samples.append(
                    Sample(
                        frame=index,
                        time_s=now,
                        ear=metrics.ear,
                        left_ear=metrics.left_ear,
                        right_ear=metrics.right_ear,
                        mar=metrics.mar,
                        blink_blendshape=_blink_score(
                            getattr(result, "face_blendshapes", ()) or ()
                        ),
                    )
                )
            index += 1
    finally:
        capture.release()
        landmarker.close()
    return samples


def correlation(first: Sequence[float], second: Sequence[float]) -> float:
    """Pearson correlation, or NaN when either series is constant."""
    pairs = [
        (a, b)
        for a, b in zip(first, second)
        if math.isfinite(a) and math.isfinite(b)
    ]
    if len(pairs) < 3:
        return float("nan")
    xs = [a for a, _ in pairs]
    ys = [b for _, b in pairs]
    try:
        return statistics.correlation(xs, ys)
    except statistics.StatisticsError:
        return float("nan")


def sweep(samples: Sequence[Sample], thresholds: Sequence[float], base: MonitorConfig):
    """Replay the recorded EAR series through the state machine per threshold."""
    rows = []
    for threshold in thresholds:
        try:
            config = MonitorConfig(
                ear_threshold=threshold,
                mar_threshold=base.mar_threshold,
                closure_seconds=base.closure_seconds,
                blink_max_seconds=base.blink_max_seconds,
                yawn_seconds=base.yawn_seconds,
                perclos_window_seconds=base.perclos_window_seconds,
                perclos_threshold=base.perclos_threshold,
                perclos_warmup_seconds=base.perclos_warmup_seconds,
            )
        except ValueError:
            continue
        monitor = DrowsinessMonitor(config)
        alarms = 0
        alarm_seconds = 0.0
        closed_frames = 0
        was_alarming = False
        previous_time = samples[0].time_s if samples else 0.0
        for sample in samples:
            state = monitor.update(sample.ear, sample.mar, sample.time_s)
            elapsed = max(0.0, sample.time_s - previous_time)
            previous_time = sample.time_s
            if state.eye_state.value == "closed":
                closed_frames += 1
            if state.should_alarm:
                alarm_seconds += elapsed
                if not was_alarming:
                    alarms += 1
            was_alarming = state.should_alarm
        rows.append(
            {
                "threshold": round(threshold, 4),
                "blinks": monitor.blinks,
                "alarms": alarms,
                "alarm_seconds": round(alarm_seconds, 2),
                "closed_fraction": round(closed_frames / len(samples), 4) if samples else 0.0,
            }
        )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="recorded video to analyse")
    source.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    parser.add_argument(
        "--seconds", type=float, default=20.0, help="live capture length (default: 20)"
    )
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N frames")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sweep-output", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--no-mirror", action="store_true")
    args = parser.parse_args(argv)

    if args.seconds <= 0.0:
        print("error: --seconds must be positive", file=sys.stderr)
        return 2

    source_value: int | str
    if args.input is not None:
        path = args.input.expanduser().resolve()
        if not path.is_file():
            print(f"error: input video not found: {path}", file=sys.stderr)
            return 2
        source_value = str(path)
    else:
        source_value = args.camera

    try:
        samples = collect(
            source_value,
            args.model,
            seconds=args.seconds,
            max_frames=args.max_frames,
            mirror=not args.no_mirror,
        )
    except (DrowsinessError, cv2.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if len(samples) < 10:
        print(
            f"error: only {len(samples)} frame(s) contained a face. "
            "Point the camera at a face, or pass a clip that contains one.",
            file=sys.stderr,
        )
        return 2

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRACE_FIELDS)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "frame": sample.frame,
                    "time_s": round(sample.time_s, 4),
                    "ear": round(sample.ear, 5),
                    "left_ear": round(sample.left_ear, 5),
                    "right_ear": round(sample.right_ear, 5),
                    "mar": round(sample.mar, 5),
                    "blink_blendshape": round(sample.blink_blendshape, 5)
                    if math.isfinite(sample.blink_blendshape)
                    else "",
                }
            )

    ears = [sample.ear for sample in samples]
    blinks = [sample.blink_blendshape for sample in samples]
    agreement = correlation(ears, blinks)

    print(f"\nFrames with a face: {len(samples)}")
    print(
        f"EAR  min={min(ears):.4f}  p10={percentile(ears, 0.10):.4f}  "
        f"p30={percentile(ears, 0.30):.4f}  median={statistics.median(ears):.4f}  "
        f"max={max(ears):.4f}"
    )
    if math.isfinite(agreement):
        print(
            f"Correlation between EAR and the learned blink blendshape: {agreement:+.3f} "
            "(strongly negative is expected: EAR falls as the blink score rises)"
        )
    else:
        print("Blink blendshape unavailable for this clip.")

    baseline = percentile(ears, 0.30)
    print(f"Suggested threshold at 75% of the p30 baseline: {baseline * 0.75:.4f}")

    thresholds = [round(0.05 + 0.01 * step, 2) for step in range(26)]
    rows = sweep(samples, thresholds, MonitorConfig())
    sweep_output = args.sweep_output.expanduser().resolve()
    sweep_output.parent.mkdir(parents=True, exist_ok=True)
    with sweep_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SWEEP_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print("\n| Threshold | Blinks | Alarms | Alarm s | Closed frames |")
    print("|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['threshold']:.2f} | {row['blinks']} | {row['alarms']} "
            f"| {row['alarm_seconds']:.1f} | {row['closed_fraction']:.1%} |"
        )
    print(f"\nWrote {output}\nWrote {sweep_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
