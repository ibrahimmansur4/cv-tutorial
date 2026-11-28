"""Per-driver EAR calibration.

A fixed EAR threshold does not transfer between people. Eye shape, the angle
the camera sits at, and whether someone wears glasses all shift the open-eye
baseline, and the published 0.25 figure additionally assumes dlib's 68-point
model rather than MediaPipe's mesh. Measured on the sample faces used here, a
wide-open eye reads nearer 0.20.

Calibration sidesteps all of that by measuring *this* driver with their eyes
open and setting the threshold as a fraction of their own baseline. A relative
threshold is the right shape for the problem: what matters is that the eye is
much less open than usual, not that it crossed some universal constant.

The baseline uses a low percentile rather than the mean, because any blink
during the capture drags the mean down; a percentile ignores a minority of
closed frames.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .errors import ConfigurationError

# Fraction of the open-eye baseline at which the eye counts as closed.
DEFAULT_CLOSED_RATIO = 0.75
# Percentile of the captured samples taken as "eyes open".
BASELINE_PERCENTILE = 0.30
MIN_SAMPLES = 15
PROFILE_VERSION = 1


@dataclass(frozen=True, slots=True)
class Calibration:
    """A measured EAR baseline and the threshold derived from it."""

    baseline_ear: float
    ear_threshold: float
    samples: int
    closed_ratio: float = DEFAULT_CLOSED_RATIO
    version: int = PROFILE_VERSION

    def __post_init__(self) -> None:
        if not math.isfinite(self.baseline_ear) or self.baseline_ear <= 0.0:
            raise ConfigurationError("calibrated baseline EAR must be positive")
        if not math.isfinite(self.ear_threshold) or not 0.0 < self.ear_threshold < 1.0:
            raise ConfigurationError("calibrated EAR threshold must be between 0 and 1")
        if self.samples < 0:
            raise ConfigurationError("sample count cannot be negative")

    def save(self, path: Path) -> Path:
        destination = path.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        source = path.expanduser().resolve()
        if not source.is_file():
            raise ConfigurationError(
                f"calibration profile not found: {source}. "
                "Run with --calibrate to create one."
            )
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"could not read calibration profile {source}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ConfigurationError(f"calibration profile is not an object: {source}")
        version = payload.get("version", 0)
        if version != PROFILE_VERSION:
            raise ConfigurationError(
                f"calibration profile {source} has version {version}, "
                f"expected {PROFILE_VERSION}. Re-run --calibrate."
            )
        try:
            return cls(
                baseline_ear=float(payload["baseline_ear"]),
                ear_threshold=float(payload["ear_threshold"]),
                samples=int(payload.get("samples", 0)),
                closed_ratio=float(payload.get("closed_ratio", DEFAULT_CLOSED_RATIO)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"calibration profile {source} is missing or has invalid fields: {exc}"
            ) from exc


def percentile(samples: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile; `fraction` is 0..1."""
    if not samples:
        raise ConfigurationError("cannot take a percentile of an empty sample set")
    if not 0.0 <= fraction <= 1.0:
        raise ConfigurationError("percentile fraction must be between 0 and 1")
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def calibrate(
    samples: Sequence[float],
    *,
    closed_ratio: float = DEFAULT_CLOSED_RATIO,
    minimum_samples: int = MIN_SAMPLES,
) -> Calibration:
    """Derive a threshold from EAR samples captured with the eyes open."""
    if not 0.0 < closed_ratio < 1.0:
        raise ConfigurationError("closed ratio must be between 0 and 1")
    usable = [value for value in samples if math.isfinite(value) and value > 0.0]
    if len(usable) < minimum_samples:
        raise ConfigurationError(
            f"calibration needs at least {minimum_samples} frames with a visible face; "
            f"got {len(usable)}. Sit facing the camera with your eyes open and retry."
        )
    baseline = percentile(usable, BASELINE_PERCENTILE)
    threshold = baseline * closed_ratio
    if not 0.0 < threshold < 1.0:
        raise ConfigurationError(
            f"derived an unusable threshold ({threshold:.4f}) from baseline {baseline:.4f}"
        )
    return Calibration(
        baseline_ear=baseline,
        ear_threshold=threshold,
        samples=len(usable),
        closed_ratio=closed_ratio,
    )


def summarize(samples: Sequence[float]) -> str:
    """One line describing a capture, for the calibration report."""
    usable = [value for value in samples if math.isfinite(value)]
    if not usable:
        return "no usable samples"
    return (
        f"n={len(usable)} min={min(usable):.3f} "
        f"p30={percentile(usable, 0.30):.3f} "
        f"median={statistics.median(usable):.3f} max={max(usable):.3f}"
    )
