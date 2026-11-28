"""Per-frame orchestration: detect, record, annotate."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import cv2
import numpy as np

from .detector import MotionDetector, MotionResult
from .drawing import draw_regions, draw_status, stack_mask
from .recorder import EventRecorder, MotionEvent

# Recording-dot flash rate, in Hz.
FLASH_HZ = 1.5


class FpsMeter:
    """Rolling-average FPS over a short window."""

    def __init__(self, window_size: int = 30) -> None:
        if window_size <= 0:
            raise ValueError("FPS window must be positive")
        self._durations: deque[float] = deque(maxlen=window_size)

    def update(self, elapsed: float) -> float:
        if math.isfinite(elapsed) and elapsed > 0.0:
            self._durations.append(elapsed)
        total = sum(self._durations)
        return len(self._durations) / total if total > 0.0 else 0.0


@dataclass(frozen=True, slots=True)
class FrameResult:
    motion: MotionResult
    recording: bool
    finished_event: MotionEvent | None
    fps: float


@dataclass(slots=True)
class RunStats:
    frames: int = 0
    motion_frames: int = 0
    recorded_frames: int = 0
    elapsed_seconds: float = 0.0
    events: list[MotionEvent] = field(default_factory=list)

    @property
    def fps(self) -> float:
        return self.frames / self.elapsed_seconds if self.elapsed_seconds > 0.0 else 0.0

    @property
    def motion_fraction(self) -> float:
        return self.motion_frames / self.frames if self.frames else 0.0

    def observe(self, result: FrameResult) -> None:
        self.frames += 1
        if result.motion.detected:
            self.motion_frames += 1
        if result.recording:
            self.recorded_frames += 1
        if result.finished_event is not None:
            self.events.append(result.finished_event)

    def summary(self) -> str:
        if not self.frames:
            return "No frames were processed."
        recorded = sum(event.duration for event in self.events)
        return (
            f"Processed {self.frames} frame(s) at {self.fps:.1f} FPS; "
            f"motion in {self.motion_fraction * 100:.1f}% of frames; "
            f"{len(self.events)} event(s) totalling {recorded:.1f}s recorded"
        )


class SecurityCameraPipeline:
    """Detect motion, drive the recorder, and render the preview."""

    def __init__(
        self,
        detector: MotionDetector,
        recorder: EventRecorder | None,
        *,
        show_mask: bool = False,
        draw_boxes: bool = True,
        cv2_module: Any = cv2,
        fps_window: int = 30,
    ) -> None:
        self.detector = detector
        self.recorder = recorder
        self.show_mask = bool(show_mask)
        self.draw_boxes = bool(draw_boxes)
        self._cv2 = cv2_module
        self._fps = FpsMeter(fps_window)

    def process(
        self,
        frame: np.ndarray,
        now_seconds: float,
        elapsed: float,
        *,
        stamp: datetime | None = None,
    ) -> FrameResult:
        """Analyse one frame and feed the recorder."""
        motion = self.detector.detect(frame)
        finished: MotionEvent | None = None
        recording = False
        if self.recorder is not None:
            finished = self.recorder.update(
                frame,
                motion=motion.detected,
                now_seconds=now_seconds,
                regions=len(motion.regions),
                area=motion.total_area,
                stamp=stamp,
            )
            recording = self.recorder.recording
        return FrameResult(
            motion=motion,
            recording=recording,
            finished_event=finished,
            fps=self._fps.update(elapsed),
        )

    def annotate(
        self,
        frame: np.ndarray,
        result: FrameResult,
        now_seconds: float,
        *,
        events: int,
        timestamp: datetime | None = None,
    ) -> np.ndarray:
        """Return an annotated copy of `frame`."""
        canvas = frame.copy()
        if self.draw_boxes:
            draw_regions(canvas, result.motion.regions, cv2_module=self._cv2)
        extra: list[str] = []
        if self.recorder is not None and self.recorder.recording:
            event = self.recorder.current_event
            if event is not None:
                extra.append(f"Recording: {event.video_path.name if event.video_path else ''}")
        draw_status(
            canvas,
            motion=result.motion.detected,
            recording=result.recording,
            ready=result.motion.ready,
            fps=result.fps,
            events=events,
            changed_fraction=result.motion.changed_fraction,
            model_name=self.detector.background.name,
            timestamp=timestamp,
            extra_lines=extra,
            cv2_module=self._cv2,
            blink_phase=int(now_seconds * FLASH_HZ * 2) % 2 == 0,
        )
        if self.show_mask:
            canvas = stack_mask(canvas, result.motion.mask, cv2_module=self._cv2)
        return canvas

    def close(self, now_seconds: float) -> MotionEvent | None:
        if self.recorder is None:
            return None
        return self.recorder.close(now_seconds)
