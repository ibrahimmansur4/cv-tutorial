"""The temporal state machine that turns per-frame metrics into a verdict.

This is the heart of the project. A single frame cannot tell you whether a
driver is drowsy: a frame with closed eyes is indistinguishable from a frame
caught in the middle of an ordinary blink. What separates the two is *how long*
the eyes stay shut. Everything here exists to measure duration.

Three signals are tracked, deliberately kept separate because they fail in
different ways:

* **Sustained closure** - eyes below the EAR threshold continuously for
  `closure_seconds`. This is the acute, immediate danger signal.
* **PERCLOS** - the fraction of a rolling window spent with the eyes closed.
  This is the measure the automotive literature settled on, because it catches
  the slow creep of fatigue: many short closures that never individually reach
  the alarm duration still push PERCLOS up.
* **Yawning** - the mouth held open past `yawn_seconds`, counted as discrete
  events. Corroborating evidence rather than an alarm on its own.

Durations are measured in **seconds, not frames**. A frame count silently
encodes the frame rate: "20 consecutive frames" is 0.67 s on a 30 FPS webcam
and 2 s on a laptop struggling at 10 FPS, so a threshold tuned on one machine
misbehaves on another. Taking a timestamp per frame removes that dependency.

This module has no OpenCV or MediaPipe import and takes its time from the
caller, so the whole state machine is testable with a synthetic clock.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from .metrics import (
    DEFAULT_EAR_THRESHOLD,
    DEFAULT_MAR_THRESHOLD,
    EAR_HYSTERESIS,
)


class EyeState(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    UNKNOWN = "unknown"


class AlertLevel(str, Enum):
    """Ordered from calm to urgent."""

    AWAKE = "awake"
    FATIGUE = "fatigue"  # PERCLOS is elevated: tired, not yet dangerous
    DROWSY = "drowsy"  # eyes have been shut long enough to be dangerous
    NO_FACE = "no_face"  # nothing to judge; the driver may have looked away

    @property
    def is_alarming(self) -> bool:
        return self is AlertLevel.DROWSY


@dataclass(frozen=True, slots=True)
class MonitorConfig:
    """Thresholds and durations, all in seconds where they are times."""

    ear_threshold: float = DEFAULT_EAR_THRESHOLD
    ear_hysteresis: float = EAR_HYSTERESIS
    mar_threshold: float = DEFAULT_MAR_THRESHOLD
    closure_seconds: float = 1.5
    blink_max_seconds: float = 0.4
    yawn_seconds: float = 0.7
    perclos_window_seconds: float = 60.0
    perclos_threshold: float = 0.15
    # PERCLOS is a ratio over a window, so it is meaningless until the window
    # holds enough footage. Early on it is dominated by whatever just happened:
    # one ordinary blink in the first second reads as 17% closed and would
    # report fatigue the moment the program starts. Hold the verdict until this
    # much has been observed (clamped to the window length).
    perclos_warmup_seconds: float = 30.0
    no_face_seconds: float = 2.0

    def __post_init__(self) -> None:
        if not 0.0 < self.ear_threshold < 1.0:
            raise ValueError("EAR threshold must be between 0 and 1")
        if self.ear_hysteresis < 0.0:
            raise ValueError("EAR hysteresis cannot be negative")
        if self.mar_threshold <= 0.0:
            raise ValueError("MAR threshold must be positive")
        for name in ("closure_seconds", "blink_max_seconds", "yawn_seconds"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.perclos_window_seconds <= 0.0:
            raise ValueError("PERCLOS window must be positive")
        if not 0.0 <= self.perclos_threshold <= 1.0:
            raise ValueError("PERCLOS threshold must be between 0 and 1")
        if self.perclos_warmup_seconds < 0.0:
            raise ValueError("PERCLOS warm-up cannot be negative")
        if self.blink_max_seconds >= self.closure_seconds:
            raise ValueError(
                "blink_max_seconds must be shorter than closure_seconds, "
                "otherwise an ordinary blink would raise the alarm"
            )


@dataclass(frozen=True, slots=True)
class MonitorState:
    """The verdict for one frame."""

    eye_state: EyeState
    level: AlertLevel
    ear: float
    mar: float
    closure_seconds: float
    perclos: float
    blinks: int
    yawns: int
    yawning: bool
    face_missing_seconds: float

    @property
    def should_alarm(self) -> bool:
        return self.level.is_alarming


@dataclass(slots=True)
class _Window:
    """Rolling record of (timestamp, closed, duration) used for PERCLOS."""

    span_seconds: float
    samples: deque[tuple[float, bool, float]] = field(default_factory=deque)
    closed_seconds: float = 0.0
    total_seconds: float = 0.0

    def add(self, now: float, closed: bool, elapsed: float) -> None:
        if elapsed <= 0.0:
            return
        self.samples.append((now, closed, elapsed))
        self.total_seconds += elapsed
        if closed:
            self.closed_seconds += elapsed
        cutoff = now - self.span_seconds
        while self.samples and self.samples[0][0] < cutoff:
            _, was_closed, duration = self.samples.popleft()
            self.total_seconds -= duration
            if was_closed:
                self.closed_seconds -= duration
        # Floating-point subtraction can leave a tiny negative residue.
        self.closed_seconds = max(0.0, self.closed_seconds)
        self.total_seconds = max(0.0, self.total_seconds)

    @property
    def ratio(self) -> float:
        return self.closed_seconds / self.total_seconds if self.total_seconds > 0.0 else 0.0

    def ready(self, warmup_seconds: float) -> bool:
        """True once enough footage has accumulated for the ratio to mean anything."""
        return self.total_seconds >= warmup_seconds

    def clear(self) -> None:
        self.samples.clear()
        self.closed_seconds = 0.0
        self.total_seconds = 0.0


class DrowsinessMonitor:
    """Accumulate per-frame metrics into an alert level.

    Call `update` once per frame with the current metrics and a monotonic
    timestamp in seconds, or `update_missing` when no face was found.
    """

    def __init__(self, config: MonitorConfig | None = None) -> None:
        self.config = config or MonitorConfig()
        self._eye_state = EyeState.UNKNOWN
        self._closed_since: float | None = None
        self._last_closure_seconds = 0.0
        self._blinks = 0
        self._yawns = 0
        self._yawn_since: float | None = None
        self._yawn_counted = False
        self._face_missing_since: float | None = None
        self._last_timestamp: float | None = None
        self._window = _Window(self.config.perclos_window_seconds)
        self._perclos_warmup = min(
            self.config.perclos_warmup_seconds, self.config.perclos_window_seconds
        )

    @property
    def blinks(self) -> int:
        return self._blinks

    @property
    def yawns(self) -> int:
        return self._yawns

    @property
    def perclos(self) -> float:
        return self._window.ratio

    def _elapsed(self, now: float) -> float:
        if self._last_timestamp is None:
            return 0.0
        return max(0.0, now - self._last_timestamp)

    def _classify_eyes(self, ear: float) -> bool:
        """True when the eyes count as closed, applying hysteresis.

        Closing needs EAR below the threshold; re-opening needs it back above
        the threshold plus the hysteresis margin. Without the gap, a noisy EAR
        resting on the threshold flips state every frame.
        """
        if self._eye_state is EyeState.CLOSED:
            return ear < self.config.ear_threshold + self.config.ear_hysteresis
        return ear < self.config.ear_threshold

    def _update_yawn(self, mar: float, now: float) -> bool:
        if mar >= self.config.mar_threshold:
            if self._yawn_since is None:
                self._yawn_since = now
            elif (
                not self._yawn_counted
                and now - self._yawn_since >= self.config.yawn_seconds
            ):
                self._yawns += 1
                self._yawn_counted = True
            return now - self._yawn_since >= self.config.yawn_seconds
        self._yawn_since = None
        self._yawn_counted = False
        return False

    @property
    def perclos_ready(self) -> bool:
        """Whether PERCLOS has seen enough footage to be acted on."""
        return self._window.ready(self._perclos_warmup)

    def _level(self, closure_seconds: float) -> AlertLevel:
        # Sustained closure is judged from the current closure alone, so it is
        # trustworthy from the first frame. PERCLOS needs history.
        if closure_seconds >= self.config.closure_seconds:
            return AlertLevel.DROWSY
        if self.perclos_ready and self.perclos >= self.config.perclos_threshold:
            return AlertLevel.FATIGUE
        return AlertLevel.AWAKE

    def update(self, ear: float, mar: float, now: float) -> MonitorState:
        """Record one frame in which a face was found."""
        elapsed = self._elapsed(now)
        closed = self._classify_eyes(ear)

        if closed:
            if self._closed_since is None:
                self._closed_since = now
            closure_seconds = now - self._closed_since
        else:
            if self._closed_since is not None:
                # The eye just reopened. A closure shorter than the blink
                # ceiling was an ordinary blink; a longer one was already
                # reported as drowsiness and is not also a blink.
                duration = now - self._closed_since
                if duration <= self.config.blink_max_seconds:
                    self._blinks += 1
                self._last_closure_seconds = duration
            self._closed_since = None
            closure_seconds = 0.0

        self._eye_state = EyeState.CLOSED if closed else EyeState.OPEN
        self._window.add(now, closed, elapsed)
        yawning = self._update_yawn(mar, now)
        self._face_missing_since = None
        self._last_timestamp = now

        return MonitorState(
            eye_state=self._eye_state,
            level=self._level(closure_seconds),
            ear=ear,
            mar=mar,
            closure_seconds=closure_seconds,
            perclos=self.perclos,
            blinks=self._blinks,
            yawns=self._yawns,
            yawning=yawning,
            face_missing_seconds=0.0,
        )

    def update_missing(self, now: float) -> MonitorState:
        """Record one frame in which no face was found.

        A missing face is reported as its own state rather than folded into
        either eye state. Counting it as closed would alarm every time the
        driver turned to check a mirror; counting it as open would hide a
        driver who had slumped out of frame.
        """
        if self._face_missing_since is None:
            self._face_missing_since = now
        missing_for = now - self._face_missing_since

        # Any in-progress closure or yawn is abandoned: without landmarks there
        # is no evidence it is still going on.
        self._closed_since = None
        self._yawn_since = None
        self._yawn_counted = False
        self._eye_state = EyeState.UNKNOWN
        self._last_timestamp = now

        level = (
            AlertLevel.NO_FACE
            if missing_for >= self.config.no_face_seconds
            else self._level(0.0)
        )
        return MonitorState(
            eye_state=EyeState.UNKNOWN,
            level=level,
            ear=float("nan"),
            mar=float("nan"),
            closure_seconds=0.0,
            perclos=self.perclos,
            blinks=self._blinks,
            yawns=self._yawns,
            yawning=False,
            face_missing_seconds=missing_for,
        )

    def reset(self) -> None:
        """Forget all history, as when switching to a different driver."""
        self._eye_state = EyeState.UNKNOWN
        self._closed_since = None
        self._last_closure_seconds = 0.0
        self._blinks = 0
        self._yawns = 0
        self._yawn_since = None
        self._yawn_counted = False
        self._face_missing_since = None
        self._last_timestamp = None
        self._window.clear()
