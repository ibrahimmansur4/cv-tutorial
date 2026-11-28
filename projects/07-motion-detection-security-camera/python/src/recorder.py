"""Event recording: pre-roll, post-roll, and timestamped clips.

Three details separate a usable security recorder from a naive one.

**Pre-roll.** A recorder that opens its file the instant motion is detected has
already missed the approach. The interesting part of a clip is usually the
second before the trigger, so recent frames are kept in a ring buffer and
flushed into the file when an event starts.

**Post-roll.** Motion detection is intermittent: a person pausing, or briefly
matching the background, drops the signal for a few frames. Stopping
immediately would shatter one event into a dozen clips, so recording continues
for a cooldown period and only ends if motion stays absent.

**A minimum event length.** A single noisy frame should not leave a 1-frame file
behind. Events shorter than a floor are discarded along with their file.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .errors import ConfigurationError, RecorderError

VIDEO_CODECS = {".avi": "MJPG", ".mp4": "mp4v", ".m4v": "mp4v", ".mov": "mp4v"}


@dataclass(slots=True)
class MotionEvent:
    """One recorded motion episode."""

    index: int
    started_at: datetime
    start_seconds: float
    end_seconds: float | None = None
    frames: int = 0
    peak_regions: int = 0
    peak_area: float = 0.0
    video_path: Path | None = None
    snapshot_path: Path | None = None

    @property
    def duration(self) -> float:
        if self.end_seconds is None:
            return 0.0
        return max(0.0, self.end_seconds - self.start_seconds)

    def as_record(self) -> dict[str, object]:
        return {
            "index": self.index,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "duration_seconds": round(self.duration, 2),
            "frames": self.frames,
            "peak_regions": self.peak_regions,
            "peak_area": round(self.peak_area, 1),
            "video": self.video_path.name if self.video_path else "",
            "snapshot": self.snapshot_path.name if self.snapshot_path else "",
        }


@dataclass(slots=True)
class RecorderConfig:
    directory: Path
    fps: float = 20.0
    pre_roll_seconds: float = 1.5
    post_roll_seconds: float = 3.0
    min_event_seconds: float = 0.5
    max_event_seconds: float = 120.0
    suffix: str = ".avi"
    write_snapshot: bool = True

    def __post_init__(self) -> None:
        if self.fps <= 0.0 or not math.isfinite(self.fps):
            raise ConfigurationError("recorder FPS must be positive")
        for name in ("pre_roll_seconds", "post_roll_seconds", "min_event_seconds"):
            if getattr(self, name) < 0.0:
                raise ConfigurationError(f"{name} cannot be negative")
        if self.max_event_seconds <= 0.0:
            raise ConfigurationError("maximum event length must be positive")
        if self.suffix.lower() not in VIDEO_CODECS:
            raise ConfigurationError(
                f"recordings must use one of {', '.join(sorted(VIDEO_CODECS))}"
            )


class EventRecorder:
    """Write one timestamped clip per motion episode."""

    def __init__(
        self,
        config: RecorderConfig,
        *,
        cv2_module: Any = cv2,
        clock: Any = None,
    ) -> None:
        from datetime import datetime as _datetime

        self.config = config
        self._cv2 = cv2_module
        self._wall_clock = clock or _datetime.now
        capacity = max(1, int(round(config.pre_roll_seconds * config.fps)))
        self._pre_roll: deque[np.ndarray] = deque(maxlen=capacity)
        self._writer: Any = None
        self._event: MotionEvent | None = None
        self._last_motion_seconds = 0.0
        self._events: list[MotionEvent] = []
        self._next_index = 1

    @property
    def recording(self) -> bool:
        return self._event is not None

    @property
    def current_event(self) -> MotionEvent | None:
        return self._event

    @property
    def events(self) -> tuple[MotionEvent, ...]:
        return tuple(self._events)

    def _open_writer(self, frame: np.ndarray, stamp: datetime) -> tuple[Any, Path]:
        directory = self.config.directory.expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        name = f"motion_{stamp:%Y%m%d_%H%M%S}_{self._next_index:03d}{self.config.suffix}"
        path = directory / name
        height, width = frame.shape[:2]
        writer = self._cv2.VideoWriter(
            str(path),
            self._cv2.VideoWriter_fourcc(*VIDEO_CODECS[self.config.suffix.lower()]),
            self.config.fps,
            (width, height),
        )
        if not writer.isOpened():
            writer.release()
            raise RecorderError(f"could not open recording for writing: {path}")
        return writer, path

    def _start(
        self, frame: np.ndarray, now_seconds: float, stamp: datetime | None = None
    ) -> MotionEvent:
        stamp = stamp or self._wall_clock()
        writer, path = self._open_writer(frame, stamp)
        event = MotionEvent(
            index=self._next_index,
            started_at=stamp,
            start_seconds=now_seconds,
            video_path=path,
        )
        self._next_index += 1
        self._writer = writer
        self._event = event
        # Flush the buffered approach into the file before the trigger frame.
        for buffered in self._pre_roll:
            writer.write(buffered)
            event.frames += 1
        self._pre_roll.clear()
        if self.config.write_snapshot:
            snapshot = path.with_suffix(".jpg")
            try:
                if self._cv2.imwrite(str(snapshot), frame):
                    event.snapshot_path = snapshot
            except Exception:
                # A failed thumbnail must not abort the recording itself.
                event.snapshot_path = None
        return event

    def _finish(self, now_seconds: float) -> MotionEvent | None:
        event = self._event
        if event is None:
            return None
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        event.end_seconds = now_seconds
        self._event = None
        if event.duration < self.config.min_event_seconds:
            # Too short to be worth keeping; remove the file as well as the row.
            if event.video_path is not None:
                event.video_path.unlink(missing_ok=True)
            if event.snapshot_path is not None:
                event.snapshot_path.unlink(missing_ok=True)
            return None
        self._events.append(event)
        return event

    def update(
        self,
        frame: np.ndarray,
        *,
        motion: bool,
        now_seconds: float,
        regions: int = 0,
        area: float = 0.0,
        stamp: datetime | None = None,
    ) -> MotionEvent | None:
        """Feed one frame. Returns an event when one has just finished.

        `stamp` is the wall-clock time to label the event with. Passing it lets
        a recorded clip be stamped on its own timeline rather than with the
        time the file happened to be processed.
        """
        finished: MotionEvent | None = None

        if motion:
            self._last_motion_seconds = now_seconds
            if self._event is None:
                self._start(frame, now_seconds, stamp)

        if self._event is not None:
            event = self._event
            if self._writer is not None:
                try:
                    self._writer.write(frame)
                except Exception as exc:
                    raise RecorderError(f"could not write a recording frame: {exc}") from exc
                event.frames += 1
            event.peak_regions = max(event.peak_regions, regions)
            event.peak_area = max(event.peak_area, area)

            quiet_for = now_seconds - self._last_motion_seconds
            too_long = now_seconds - event.start_seconds >= self.config.max_event_seconds
            if too_long or quiet_for >= self.config.post_roll_seconds:
                finished = self._finish(now_seconds)
        else:
            # Only buffer while idle; during an event frames go straight out.
            self._pre_roll.append(frame.copy())

        return finished

    def close(self, now_seconds: float) -> MotionEvent | None:
        """Finish any in-progress event, as at shutdown."""
        finished = self._finish(now_seconds) if self._event is not None else None
        self._pre_roll.clear()
        return finished
