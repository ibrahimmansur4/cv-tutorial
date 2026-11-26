"""Turn raw YOLO network output into boxes, class ids, and confidences.

Every YOLO model here is a single-stage detector: one forward pass produces a
fixed grid of candidate boxes, and the differences between generations live
almost entirely in how that grid is laid out in memory.

| Family              | Raw shape        | Row layout                       |
|---------------------|------------------|----------------------------------|
| `yolov5`            | (1, 25200, 85)   | cx cy w h obj c0..c79            |
| `yolov8` (v8..v12)  | (1, 84, 8400)    | cx cy w h c0..c79, channel-major |

Two structural differences matter. First, YOLOv8 and later are anchor-free and
dropped the objectness column: the class score *is* the confidence, so a box
carries 84 values instead of 85. Second, that output is transposed —
attributes on the first axis, boxes on the second — so it must be transposed
before the rows can be read like YOLOv5's.

Both decoders emit the same `RawDetections`, so no other module branches on
model family. Box coordinates are already in network-input pixels for these
exported models; `letterbox.py` maps them back to the original frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

from .errors import DecodeError

NUM_BOX_COORDS = 4


class ModelFamily(str, Enum):
    """Output layout of a YOLO network, not its file format."""

    YOLOV5 = "yolov5"
    YOLOV8 = "yolov8"

    @property
    def transposed_output(self) -> bool:
        """True when the raw tensor is (attributes, boxes) instead of (boxes, attributes)."""
        return self is ModelFamily.YOLOV8

    @property
    def has_objectness(self) -> bool:
        """True when a separate "is there an object here" column precedes the class scores."""
        return self is ModelFamily.YOLOV5

    @property
    def attributes_per_box(self) -> int:
        """Attribute count for an 80-class COCO export, used in error messages."""
        return NUM_BOX_COORDS + (1 if self.has_objectness else 0) + 80


@dataclass(frozen=True, slots=True)
class RawDetections:
    """Pre-NMS candidates in letterboxed network-input pixel space.

    Boxes are `(x, y, w, h)` with a top-left origin, which is the form
    `cv2.dnn.NMSBoxes` expects.
    """

    boxes: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray

    def __post_init__(self) -> None:
        if self.boxes.ndim != 2 or (self.boxes.size and self.boxes.shape[1] != NUM_BOX_COORDS):
            raise DecodeError("decoded boxes must have shape (N, 4)")
        if self.scores.ndim != 1 or self.class_ids.ndim != 1:
            raise DecodeError("decoded scores and class ids must be one-dimensional")
        if not len(self.boxes) == len(self.scores) == len(self.class_ids):
            raise DecodeError("decoded boxes, scores, and class ids must have equal length")

    def __len__(self) -> int:
        return len(self.scores)


def empty_detections() -> RawDetections:
    """A zero-length result, used whenever nothing clears the threshold."""
    return RawDetections(
        np.zeros((0, NUM_BOX_COORDS), dtype=np.float32),
        np.zeros(0, dtype=np.float32),
        np.zeros(0, dtype=np.int32),
    )


def _as_candidate_rows(outputs: Sequence[np.ndarray], family: ModelFamily) -> np.ndarray:
    """Flatten whatever the network returned into one (boxes, attributes) matrix."""
    if not outputs:
        raise DecodeError("the network produced no output tensors")
    blocks: list[np.ndarray] = []
    for output in outputs:
        array = np.asarray(output, dtype=np.float32)
        if array.size == 0:
            continue
        # Drop leading batch axes of size 1 until a 2-D matrix remains.
        while array.ndim > 2 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 1:
            array = array[np.newaxis, :]
        if array.ndim != 2:
            raise DecodeError(
                f"expected a 2-D output after removing batch axes, received shape {array.shape}"
            )
        if family.transposed_output:
            array = array.T
        blocks.append(array)
    if not blocks:
        return np.zeros((0, 0), dtype=np.float32)
    widths = {int(block.shape[1]) for block in blocks}
    if len(widths) != 1:
        raise DecodeError(f"output tensors disagree on attribute count: {sorted(widths)}")
    return np.concatenate(blocks, axis=0) if len(blocks) > 1 else blocks[0]


def _to_corner_boxes(rows: np.ndarray) -> np.ndarray:
    """Centre-form `(cx, cy, w, h)` to top-left `(x, y, w, h)`."""
    centre_x, centre_y, width, height = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    return np.stack(
        (centre_x - width / 2.0, centre_y - height / 2.0, width, height), axis=1
    ).astype(np.float32, copy=False)


def decode(
    outputs: Sequence[np.ndarray],
    family: ModelFamily,
    *,
    confidence_threshold: float = 0.25,
    num_classes: int | None = None,
) -> RawDetections:
    """Decode one forward pass into candidate detections above the threshold."""
    if not 0.0 <= confidence_threshold <= 1.0:
        raise DecodeError("confidence threshold must be between 0 and 1")

    rows = _as_candidate_rows(outputs, family)
    if rows.size == 0:
        return empty_detections()

    minimum_width = NUM_BOX_COORDS + (1 if family.has_objectness else 0) + 1
    if rows.shape[1] < minimum_width:
        raise DecodeError(
            f"a {family.value} output needs at least {minimum_width} attributes per box, "
            f"received {rows.shape[1]}. Is --family set to the right model layout?"
        )

    class_start = NUM_BOX_COORDS + (1 if family.has_objectness else 0)
    class_scores = rows[:, class_start:]
    if num_classes is not None and 0 < num_classes <= class_scores.shape[1]:
        # Ignore padded columns on models exported with a larger class head.
        class_scores = class_scores[:, :num_classes]

    class_ids = np.argmax(class_scores, axis=1)
    best_class_score = class_scores[np.arange(class_scores.shape[0]), class_ids]
    if family.has_objectness:
        # v5 confidence is objectness times the best class probability, so a box
        # the model doubts *contains* anything stays suppressed even when one
        # class clearly dominates the rest.
        scores = rows[:, NUM_BOX_COORDS] * best_class_score
    else:
        scores = best_class_score

    keep = scores >= confidence_threshold
    if not np.any(keep):
        return empty_detections()

    return RawDetections(
        _to_corner_boxes(rows[keep]),
        scores[keep].astype(np.float32, copy=False),
        class_ids[keep].astype(np.int32, copy=False),
    )
