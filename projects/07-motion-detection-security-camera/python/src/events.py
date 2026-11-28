"""Persist the motion event log as CSV and JSON."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

from .errors import ConfigurationError
from .recorder import MotionEvent

CSV_FIELDS = (
    "index",
    "started_at",
    "duration_seconds",
    "frames",
    "peak_regions",
    "peak_area",
    "video",
    "snapshot",
)


def write_csv(events: Sequence[MotionEvent], path: Path) -> Path:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for event in events:
                writer.writerow(event.as_record())
    except OSError as exc:
        raise ConfigurationError(f"could not write the event log {destination}: {exc}") from exc
    return destination


def write_json(events: Sequence[MotionEvent], path: Path) -> Path:
    """JSON for the dashboard, which reads it directly."""
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"events": [event.as_record() for event in events]}
    try:
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"could not write the event log {destination}: {exc}") from exc
    return destination


def load_json(path: Path) -> list[dict[str, object]]:
    source = path.expanduser().resolve()
    if not source.is_file():
        return []
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"could not read the event log {source}: {exc}") from exc
    events = payload.get("events") if isinstance(payload, dict) else None
    return list(events) if isinstance(events, list) else []
