"""Audible alerting, with a silent fallback when the platform cannot beep.

Sound is the one part of this project that cannot be made portable without a
dependency, so it is isolated here behind a tiny interface. Everything else
works identically whether or not a speaker exists, and a missing beep never
stops the detector: a drowsiness monitor that refuses to run because it cannot
make noise is worse than one that runs quietly.
"""

from __future__ import annotations

import sys
from typing import Callable, Protocol


class Alarm(Protocol):
    """Anything that can start and stop an alert."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class NullAlarm:
    """Does nothing; used with --no-alarm and in tests."""

    def __init__(self) -> None:
        self.started = 0

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        return None

    def close(self) -> None:
        return None


class ConsoleAlarm:
    """Writes the terminal bell character and a warning line.

    Works anywhere stderr does, including over SSH. Whether the bell is audible
    depends on the terminal's own settings.
    """

    def __init__(self, stream=None) -> None:
        self._stream = stream if stream is not None else sys.stderr
        self._active = False

    def start(self) -> None:
        if self._active:
            return
        self._active = True
        try:
            self._stream.write("\a*** DROWSINESS ALARM: wake up ***\n")
            self._stream.flush()
        except (OSError, ValueError):
            # A closed or redirected stream must not take down the detector.
            pass

    def stop(self) -> None:
        self._active = False

    def close(self) -> None:
        self.stop()


class WindowsBeepAlarm:
    """Repeating tone via winsound, available only on Windows."""

    FREQUENCY_HZ = 1200
    DURATION_MS = 400

    def __init__(self, beep: Callable[[int, int], None] | None = None) -> None:
        if beep is None:
            try:
                import winsound
            except ImportError as exc:  # pragma: no cover - platform specific
                raise RuntimeError("winsound is only available on Windows") from exc
            beep = winsound.Beep
        self._beep = beep
        self._active = False

    def start(self) -> None:
        self._active = True
        try:
            self._beep(self.FREQUENCY_HZ, self.DURATION_MS)
        except Exception:
            # Some audio backends raise when no output device is present.
            pass

    def stop(self) -> None:
        self._active = False

    def close(self) -> None:
        self.stop()


def make_alarm(enabled: bool = True) -> Alarm:
    """Pick the best alarm this platform supports."""
    if not enabled:
        return NullAlarm()
    if sys.platform.startswith("win"):
        try:
            return WindowsBeepAlarm()
        except RuntimeError:  # pragma: no cover - platform specific
            pass
    return ConsoleAlarm()
