"""Per-frame orchestration: landmarks, metrics, state machine, alarm, drawing."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .alarm import Alarm, NullAlarm
from .drawing import draw_banner, draw_hud
from .errors import LandmarkError
from .face_mesh import FaceObservation, MediaPipeFaceTracker
from .metrics import FaceMetrics, compute_metrics
from .state import AlertLevel, DrowsinessMonitor, EyeState, MonitorState

# Border flash rate while the alarm sounds, in Hz.
FLASH_HZ = 3.0


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
    """What one processed frame produced."""

    face: FaceObservation | None
    metrics: FaceMetrics | None
    state: MonitorState
    fps: float


@dataclass(slots=True)
class RunStats:
    """Totals accumulated across a whole run."""

    frames: int = 0
    faces: int = 0
    elapsed_seconds: float = 0.0
    alarm_seconds: float = 0.0
    drowsy_events: int = 0
    blinks: int = 0
    yawns: int = 0
    peak_closure: float = 0.0
    peak_perclos: float = 0.0
    ear_samples: list[float] = field(default_factory=list)
    _was_alarming: bool = False

    @property
    def fps(self) -> float:
        return self.frames / self.elapsed_seconds if self.elapsed_seconds > 0.0 else 0.0

    def observe(self, result: FrameResult, elapsed: float) -> None:
        self.frames += 1
        state = result.state
        if result.face is not None:
            self.faces += 1
        if result.metrics is not None and math.isfinite(result.metrics.ear):
            self.ear_samples.append(result.metrics.ear)
        if state.should_alarm:
            self.alarm_seconds += max(0.0, elapsed)
            if not self._was_alarming:
                self.drowsy_events += 1
        self._was_alarming = state.should_alarm
        self.blinks = state.blinks
        self.yawns = state.yawns
        self.peak_closure = max(self.peak_closure, state.closure_seconds)
        self.peak_perclos = max(self.peak_perclos, state.perclos)

    def summary(self) -> str:
        if not self.frames:
            return "No frames were processed."
        face_rate = 100.0 * self.faces / self.frames
        return (
            f"Processed {self.frames} frame(s) at {self.fps:.1f} FPS; "
            f"face visible in {face_rate:.0f}% of frames; "
            f"{self.blinks} blink(s), {self.yawns} yawn(s); "
            f"{self.drowsy_events} drowsiness event(s) totalling {self.alarm_seconds:.1f}s; "
            f"longest closure {self.peak_closure:.2f}s; peak PERCLOS {self.peak_perclos * 100:.1f}%"
        )


class DrowsinessPipeline:
    """Run the tracker and the state machine over frames, and render the result."""

    def __init__(
        self,
        tracker: MediaPipeFaceTracker,
        monitor: DrowsinessMonitor,
        *,
        alarm: Alarm | None = None,
        show_landmarks: bool = True,
        cv2_module: Any = cv2,
        fps_window: int = 30,
    ) -> None:
        self.tracker = tracker
        self.monitor = monitor
        self.alarm = alarm or NullAlarm()
        self.show_landmarks = bool(show_landmarks)
        self._cv2 = cv2_module
        self._fps = FpsMeter(fps_window)
        self._alarm_active = False

    def process(
        self, frame: np.ndarray, timestamp_ms: int, now_seconds: float, elapsed: float
    ) -> FrameResult:
        """Detect a face and advance the state machine by one frame."""
        height, width = frame.shape[:2]
        try:
            faces = self.tracker.detect(frame, timestamp_ms)
        except LandmarkError:
            # A single bad frame should not end the session; treat it as a
            # frame with no face and let the state machine decide.
            faces = []

        face = faces[0] if faces else None
        metrics: FaceMetrics | None = None
        if face is not None:
            try:
                metrics = compute_metrics(face.landmarks, width / height)
            except LandmarkError:
                metrics = None

        if metrics is None:
            state = self.monitor.update_missing(now_seconds)
        else:
            state = self.monitor.update(metrics.ear, metrics.mar, now_seconds)

        self._drive_alarm(state)
        return FrameResult(
            face=face, metrics=metrics, state=state, fps=self._fps.update(elapsed)
        )

    def _drive_alarm(self, state: MonitorState) -> None:
        if state.should_alarm:
            self.alarm.start()
            self._alarm_active = True
        elif self._alarm_active:
            self.alarm.stop()
            self._alarm_active = False

    def annotate(
        self, frame: np.ndarray, result: FrameResult, now_seconds: float
    ) -> np.ndarray:
        """Return a copy of `frame` with landmarks and the HUD drawn on it."""
        canvas = frame.copy()
        if self.show_landmarks and result.face is not None:
            self.tracker.draw(
                canvas,
                result.face,
                eyes_closed=result.state.eye_state is EyeState.CLOSED,
            )
        extra: list[str] = []
        if result.state.level is AlertLevel.FATIGUE:
            extra.append("PERCLOS above threshold")
        draw_hud(
            canvas,
            result.state,
            result.fps,
            ear_threshold=self.monitor.config.ear_threshold,
            extra_lines=extra,
            cv2_module=self._cv2,
        )
        # Square wave at FLASH_HZ, derived from the clock so the flash rate does
        # not depend on how fast frames arrive.
        phase = int(now_seconds * FLASH_HZ * 2) % 2 == 0
        draw_banner(canvas, result.state, cv2_module=self._cv2, blink_phase=phase)
        return canvas

    def close(self) -> None:
        try:
            self.alarm.close()
        finally:
            self.tracker.close()
