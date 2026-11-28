"""Turn a stream of per-frame detections into a list of distinct scans.

A webcam decodes the same code in every frame it is visible in, so a 30 fps
camera pointed at one QR code produces about 30 identical results per second.
Reporting all of them is useless; the user scanned *one* code.

**A seen-set is the wrong fix.** Remembering every payload forever and skipping
repeats does stop the flood, but it also means a code can never be scanned
twice -- and scanning the same ticket, or the same product at the till, is a
normal thing to want.

What actually matches the behaviour of a real scanner is a **cooldown**: a
payload is accepted, then suppressed for a few seconds, then accepted again.
It collapses the burst and still allows a deliberate re-scan. The window is the
one number that matters here, and it is a trade-off rather than a correct
value: too short and one wave of a code counts twice, too long and a genuine
re-scan is swallowed.

The CSV is written incrementally with a flush per row, because the interesting
failure is the program being closed mid-session -- and a scan log that loses
the last entries every time is not a log.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence, TextIO

from .content import ContentKind
from .detect import Detection
from .errors import HistoryError

CSV_FIELDS: tuple[str, ...] = (
    "timestamp",
    "symbology",
    "kind",
    "payload",
    "frame",
    "centre_x",
    "centre_y",
    "area",
)


@dataclass(frozen=True, slots=True)
class Scan:
    """One accepted scan, ready to display or persist."""

    payload: str
    symbology: str
    kind: ContentKind
    summary: str
    when: datetime
    frame_index: int
    centre: tuple[float, float]
    area: float

    def as_row(self) -> dict[str, str]:
        return {
            "timestamp": self.when.isoformat(timespec="seconds"),
            "symbology": self.symbology,
            "kind": self.kind.value,
            "payload": self.payload,
            "frame": str(self.frame_index),
            "centre_x": f"{self.centre[0]:.1f}",
            "centre_y": f"{self.centre[1]:.1f}",
            "area": f"{self.area:.0f}",
        }

    def describe(self) -> str:
        """The console block a scanner prints on a hit."""
        clock = self.when.strftime("%H:%M:%S")
        return (
            f"\n{self.symbology} detected\n"
            f"\nContent: {self.payload}\n"
            f"Kind:    {self.kind.value}\n"
            f"Time:    {clock}\n"
        )


@dataclass
class ScanHistory:
    """Accepted scans, with a per-payload cooldown.

    `cooldown` is in seconds. Zero accepts every decode, which is occasionally
    what a test wants and never what a user wants.
    """

    cooldown: float = 3.0
    clock: Callable[[], float] = time.monotonic
    now: Callable[[], datetime] = datetime.now
    scans: list[Scan] = field(default_factory=list)
    _last_seen: dict[tuple[str, str], float] = field(default_factory=dict, repr=False)
    suppressed: int = 0

    def __post_init__(self) -> None:
        if self.cooldown < 0.0:
            raise HistoryError("cooldown cannot be negative")

    def __len__(self) -> int:
        return len(self.scans)

    def __iter__(self) -> Iterator[Scan]:
        return iter(self.scans)

    @property
    def payloads(self) -> list[str]:
        return [scan.payload for scan in self.scans]

    def offer(self, detection: Detection, *, frame_index: int = 0) -> Scan | None:
        """Accept a detection unless its payload is still cooling down.

        The key is (payload, symbology), not payload alone: the same digits as
        an EAN-13 and inside a QR code are two different scans.
        """
        if not detection.decoded:
            return None
        key = (detection.payload, detection.symbology)
        timestamp = self.clock()
        previous = self._last_seen.get(key)
        if previous is not None and timestamp - previous < self.cooldown:
            # Refresh the window so a code held in view keeps being suppressed
            # rather than resurfacing every `cooldown` seconds.
            self._last_seen[key] = timestamp
            self.suppressed += 1
            return None
        self._last_seen[key] = timestamp
        content = detection.content
        scan = Scan(
            payload=detection.payload,
            symbology=detection.symbology,
            kind=content.kind if content is not None else ContentKind.TEXT,
            summary=content.summary if content is not None else detection.payload,
            when=self.now(),
            frame_index=frame_index,
            centre=detection.centre,
            area=detection.area,
        )
        self.scans.append(scan)
        return scan

    def offer_all(
        self, detections: Iterable[Detection], *, frame_index: int = 0
    ) -> list[Scan]:
        accepted = []
        for detection in detections:
            scan = self.offer(detection, frame_index=frame_index)
            if scan is not None:
                accepted.append(scan)
        return accepted

    def recent(self, count: int = 5) -> list[Scan]:
        """The newest scans first, for the on-screen list."""
        if count <= 0:
            return []
        return list(reversed(self.scans[-count:]))

    def counts_by_kind(self) -> dict[str, int]:
        tally: dict[str, int] = {}
        for scan in self.scans:
            tally[scan.kind.value] = tally.get(scan.kind.value, 0) + 1
        return dict(sorted(tally.items()))

    def counts_by_symbology(self) -> dict[str, int]:
        tally: dict[str, int] = {}
        for scan in self.scans:
            tally[scan.symbology] = tally.get(scan.symbology, 0) + 1
        return dict(sorted(tally.items()))


class ScanLog:
    """Append scans to a CSV as they happen.

    Opened lazily so that a run which scans nothing does not leave an empty
    file behind, and so a bad path fails at the first scan rather than at
    start-up with nothing yet to lose.
    """

    def __init__(self, path: Path, *, append: bool = True) -> None:
        self.path = Path(path)
        self._append = append
        self._handle: TextIO | None = None
        self._writer: csv.DictWriter | None = None
        self.rows_written = 0

    def __enter__(self) -> "ScanLog":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _open(self) -> csv.DictWriter:
        if self._writer is not None:
            return self._writer
        existing = self.path.exists() and self.path.stat().st_size > 0
        mode = "a" if (self._append and existing) else "w"
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # newline="" is required on Windows or every row gains a blank line.
            self._handle = self.path.open(mode, newline="", encoding="utf-8")
        except OSError as exc:
            raise HistoryError(f"cannot write the scan log {self.path}: {exc}") from exc
        self._writer = csv.DictWriter(self._handle, fieldnames=list(CSV_FIELDS))
        if mode == "w" or not existing:
            self._writer.writeheader()
        return self._writer

    def write(self, scan: Scan) -> None:
        writer = self._open()
        writer.writerow(scan.as_row())
        if self._handle is not None:
            self._handle.flush()
        self.rows_written += 1

    def write_all(self, scans: Sequence[Scan]) -> None:
        for scan in scans:
            self.write(scan)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = None
        self._writer = None
