"""Per-frame detection, counting, and annotation."""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Mapping

import cv2
import numpy as np

from .detector import Detection, InferenceResult, YoloDetector, count_by_label
from .drawing import draw_counts, draw_detections


class FpsMeter:
    """Rolling-average FPS over a short window.

    A rolling window is used rather than an all-time average so the number
    reacts to a change in load, and rather than an instantaneous value so it
    does not flicker on every frame.
    """

    def __init__(self, window_size: int = 30) -> None:
        if window_size <= 0:
            raise ValueError("FPS window must be positive")
        self._durations: deque[float] = deque(maxlen=window_size)

    def update(self, elapsed_seconds: float) -> float:
        if math.isfinite(elapsed_seconds) and elapsed_seconds > 0.0:
            self._durations.append(elapsed_seconds)
        total = sum(self._durations)
        return len(self._durations) / total if total > 0.0 else 0.0


@dataclass(frozen=True, slots=True)
class FrameResult:
    """What one processed frame produced."""

    detections: tuple[Detection, ...]
    counts: dict[str, int]
    inference: InferenceResult
    fps: float


@dataclass(slots=True)
class RunStats:
    """Totals accumulated across a whole run."""

    frames: int = 0
    detections: int = 0
    elapsed_seconds: float = 0.0
    inference_seconds: float = 0.0
    class_totals: Counter[str] = field(default_factory=Counter)
    peak_objects: int = 0

    @property
    def fps(self) -> float:
        return self.frames / self.elapsed_seconds if self.elapsed_seconds > 0.0 else 0.0

    @property
    def mean_inference_ms(self) -> float:
        return (self.inference_seconds * 1000.0 / self.frames) if self.frames else 0.0

    def observe(self, result: FrameResult) -> None:
        self.frames += 1
        self.detections += len(result.detections)
        self.inference_seconds += result.inference.inference_ms / 1000.0
        self.class_totals.update(result.counts)
        self.peak_objects = max(self.peak_objects, len(result.detections))

    def summary(self) -> str:
        if not self.frames:
            return "No frames were processed."
        top = ", ".join(
            f"{label}={count}" for label, count in self.class_totals.most_common(5)
        )
        return (
            f"Processed {self.frames} frame(s) at {self.fps:.1f} FPS "
            f"({self.mean_inference_ms:.1f} ms mean inference); "
            f"{self.detections} detection(s), peak {self.peak_objects} per frame; "
            f"most seen: {top or 'none'}"
        )


class DetectionPipeline:
    """Run a detector over frames and render the result."""

    def __init__(
        self,
        detector: YoloDetector,
        *,
        model_label: str = "",
        show_timing: bool = False,
        cv2_module: Any = cv2,
        fps_window: int = 30,
    ) -> None:
        self.detector = detector
        self.model_label = model_label
        self.show_timing = bool(show_timing)
        self._cv2 = cv2_module
        self._fps = FpsMeter(fps_window)

    def process(self, frame: np.ndarray, elapsed_seconds: float) -> FrameResult:
        """Detect in `frame`; `elapsed_seconds` is the time the last frame took."""
        inference = self.detector.detect(frame)
        return FrameResult(
            detections=inference.detections,
            counts=count_by_label(inference.detections),
            inference=inference,
            fps=self._fps.update(elapsed_seconds),
        )

    def annotate(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """Return a copy of `frame` with boxes and the count panel drawn on it."""
        canvas = frame.copy()
        draw_detections(canvas, result.detections, cv2_module=self._cv2)
        extra = [self.model_label] if self.model_label else []
        if self.show_timing:
            inference = result.inference
            extra.append(
                f"pre {inference.preprocess_ms:.0f} / "
                f"net {inference.inference_ms:.0f} / "
                f"post {inference.postprocess_ms:.0f} ms"
            )
            extra.append(f"candidates: {inference.candidates}")
        draw_counts(canvas, result.counts, result.fps, extra_lines=extra, cv2_module=self._cv2)
        return canvas


def format_counts(counts: Mapping[str, int], fps: float) -> str:
    """Render counts as the plain-text block shown in the project README."""
    if not counts:
        return f"No objects detected\n\nFPS: {fps:.1f}"
    width = max(len(label) for label in counts) + 1
    rows = "\n".join(f"{label + ':':<{width + 1}}{count}" for label, count in counts.items())
    return f"{rows}\n\nFPS: {fps:.1f}"
