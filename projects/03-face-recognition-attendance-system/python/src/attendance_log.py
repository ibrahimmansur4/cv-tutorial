"""CSV attendance recording with per-person, per-day de-duplication."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from .errors import FaceDataError

CSV_FIELDS = ("name", "date", "time")


@dataclass(frozen=True, slots=True)
class AttendanceRecord:
    name: str
    date: str
    time: str


class AttendanceLogger:
    """Append attendance once for each case-insensitive name on a local date."""

    def __init__(
        self,
        path: str | Path,
        *,
        now_provider: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self._now_provider = now_provider
        self.last_record: AttendanceRecord | None = None
        self._known_keys: set[tuple[str, str]] | None = None

    @staticmethod
    def _clean_name(name: str) -> str:
        clean = " ".join(str(name).strip().split())
        if not clean:
            raise FaceDataError("attendance name cannot be empty")
        if len(clean) > 128 or any(ord(character) < 32 for character in clean):
            raise FaceDataError("attendance name contains invalid characters")
        if clean[0] in "=+-@":
            raise FaceDataError("attendance name cannot begin with a spreadsheet formula character")
        return clean

    def _read_keys(self) -> set[tuple[str, str]]:
        if not self.path.exists():
            return set()
        if not self.path.is_file():
            raise FaceDataError(f"attendance path is not a file: {self.path}")
        try:
            with self.path.open("r", newline="", encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames is None:
                    return set()
                if tuple(reader.fieldnames) != CSV_FIELDS:
                    raise FaceDataError(
                        f"attendance CSV must have header: {','.join(CSV_FIELDS)}"
                    )
                return {
                    (row["name"].strip().casefold(), row["date"].strip())
                    for row in reader
                    if row.get("name") and row.get("date")
                }
        except UnicodeError as exc:
            raise FaceDataError(f"attendance CSV is not valid UTF-8: {self.path}") from exc
        except OSError as exc:
            raise FaceDataError(f"could not read attendance CSV {self.path}: {exc}") from exc

    def mark(self, name: str, when: datetime | None = None) -> bool:
        """Append an attendance row; return ``False`` when already marked today."""

        clean_name = self._clean_name(name)
        timestamp = when or self._now_provider()
        date_text = timestamp.date().isoformat()
        key = (clean_name.casefold(), date_text)
        if self._known_keys is None:
            self._known_keys = self._read_keys()
        if key in self._known_keys:
            self.last_record = None
            return False

        record = AttendanceRecord(
            clean_name,
            date_text,
            timestamp.time().replace(microsecond=0).isoformat(),
        )
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            needs_header = not self.path.exists() or self.path.stat().st_size == 0
            with self.path.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
                if needs_header:
                    writer.writeheader()
                writer.writerow(
                    {"name": record.name, "date": record.date, "time": record.time}
                )
        except OSError as exc:
            raise FaceDataError(f"could not write attendance CSV {self.path}: {exc}") from exc
        self._known_keys.add(key)
        self.last_record = record
        return True

    def records(self) -> list[AttendanceRecord]:
        """Read valid records in file order."""

        if not self.path.exists() or self.path.stat().st_size == 0:
            return []
        try:
            with self.path.open("r", newline="", encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream)
                if tuple(reader.fieldnames or ()) != CSV_FIELDS:
                    raise FaceDataError(
                        f"attendance CSV must have header: {','.join(CSV_FIELDS)}"
                    )
                return [
                    AttendanceRecord(row["name"], row["date"], row["time"])
                    for row in reader
                ]
        except (OSError, UnicodeError) as exc:
            raise FaceDataError(f"could not read attendance CSV {self.path}: {exc}") from exc
