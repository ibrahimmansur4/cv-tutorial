"""COCO class names, deterministic box colors, and class-filter parsing."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from .errors import ConfigurationError

# The 80 COCO classes in the exact order every YOLO model in this project emits
# them. Keeping the list in code means the project runs without a side-car
# names file, while `load_labels` still accepts one for custom models.
COCO_CLASSES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


def load_labels(path: Path | None = None) -> tuple[str, ...]:
    """Return COCO names, or one label per non-empty line of `path`."""
    if path is None:
        return COCO_CLASSES
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ConfigurationError(f"class-names file not found: {resolved}")
    try:
        lines = resolved.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfigurationError(f"could not read class names from {resolved}: {exc}") from exc
    labels = tuple(line.strip() for line in lines if line.strip())
    if not labels:
        raise ConfigurationError(f"class-names file is empty: {resolved}")
    return labels


def class_name(labels: Sequence[str], class_id: int) -> str:
    """Name a class id, tolerating models that emit more classes than we know."""
    if 0 <= class_id < len(labels):
        return labels[class_id]
    return f"class_{class_id}"


def resolve_class_filter(
    selectors: Iterable[str] | None, labels: Sequence[str]
) -> frozenset[int] | None:
    """Turn `--classes person,2,dog` into class ids; None means keep everything.

    Selectors are matched case-insensitively by name first, then as an integer
    index, so both `--classes person` and `--classes 0` select people.
    """
    if selectors is None:
        return None
    lookup = {name.casefold(): index for index, name in enumerate(labels)}
    wanted: set[int] = set()
    unknown: list[str] = []
    for raw in selectors:
        token = raw.strip()
        if not token:
            continue
        index = lookup.get(token.casefold())
        if index is not None:
            wanted.add(index)
            continue
        try:
            numeric = int(token)
        except ValueError:
            unknown.append(token)
            continue
        if 0 <= numeric < len(labels):
            wanted.add(numeric)
        else:
            unknown.append(token)
    if unknown:
        raise ConfigurationError(
            "unknown class selector(s): "
            + ", ".join(sorted(unknown))
            + f". Expected a COCO class name or an index between 0 and {len(labels) - 1}."
        )
    if not wanted:
        raise ConfigurationError("--classes was given but selected no classes")
    return frozenset(wanted)


def color_for(class_id: int) -> tuple[int, int, int]:
    """Stable, well-separated BGR color per class id.

    A golden-ratio hue step keeps neighbouring class ids visually distinct
    without shipping a hand-made palette.
    """
    hue = (class_id * 0.618033988749895) % 1.0
    sector, offset = divmod(hue * 6.0, 1.0)
    saturation, value = 0.85, 1.0
    primary = int(round(value * 255))
    rising = int(round(value * (1.0 - saturation * (1.0 - offset)) * 255))
    falling = int(round(value * (1.0 - saturation * offset) * 255))
    low = int(round(value * (1.0 - saturation) * 255))
    red, green, blue = (
        (primary, rising, low),
        (falling, primary, low),
        (low, primary, rising),
        (low, falling, primary),
        (rising, low, primary),
        (primary, low, falling),
    )[int(sector) % 6]
    return blue, green, red
