"""Load a YOLO ONNX network with cv2.dnn and run it over single frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Collection, Sequence

import cv2
import numpy as np

from .decode import ModelFamily, RawDetections, decode
from .errors import ConfigurationError, ModelError
from .labels import COCO_CLASSES, class_name
from .letterbox import Letterbox, letterbox

# These exports expect 0..1 RGB input, so the blob scale factor is 1/255.
BLOB_SCALE = 1.0 / 255.0
# YOLO backbones downsample by 32, so the network input must be a multiple of it.
STRIDE = 32


@dataclass(frozen=True, slots=True)
class Detection:
    """One surviving detection, in original-frame pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    label: str

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_int_box(self) -> tuple[int, int, int, int]:
        return int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2))


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Detections for one frame, plus the cost of producing them."""

    detections: tuple[Detection, ...]
    candidates: int
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms


class YoloDetector:
    """A pretrained YOLO ONNX model behind OpenCV's DNN module.

    Only `detect` is on the hot path. It runs the four stages every single-stage
    detector needs: letterbox the frame, run one forward pass, decode the
    candidate grid, and suppress overlapping duplicates.
    """

    def __init__(
        self,
        model_path: str | Path,
        family: ModelFamily,
        *,
        input_size: tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        labels: Sequence[str] = COCO_CLASSES,
        class_filter: Collection[int] | None = None,
        net: Any = None,
        cv2_module: Any = cv2,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ConfigurationError("confidence threshold must be between 0 and 1")
        if not 0.0 < nms_threshold <= 1.0:
            raise ConfigurationError("NMS threshold must be above 0 and at most 1")
        if input_size[0] <= 0 or input_size[1] <= 0:
            raise ConfigurationError("network input size must be positive")
        if input_size[0] % STRIDE or input_size[1] % STRIDE:
            raise ConfigurationError(
                f"network input size must be a multiple of {STRIDE}; "
                "YOLO downsamples by that factor"
            )
        if not labels:
            raise ConfigurationError("at least one class label is required")

        self.family = family
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        self.labels = tuple(labels)
        self.class_filter = frozenset(class_filter) if class_filter is not None else None
        self.model_path = Path(model_path)
        self._cv2 = cv2_module
        self._clock = clock
        self.net = net if net is not None else self._load(self.model_path, cv2_module)
        self._output_names = self._resolve_output_names()

    @staticmethod
    def _load(model_path: Path, cv2_module: Any) -> Any:
        resolved = model_path.expanduser().resolve()
        if not resolved.is_file():
            raise ConfigurationError(
                f"model not found: {resolved}. Run `python scripts/download_model.py`."
            )
        if resolved.suffix.casefold() != ".onnx":
            raise ConfigurationError(
                f"expected an .onnx model but received a {resolved.suffix} file. OpenCV 5 "
                "removed the Darknet importer, so .weights/.cfg models cannot be loaded."
            )
        try:
            net = cv2_module.dnn.readNetFromONNX(str(resolved))
        except Exception as exc:
            raise ModelError(f"could not load ONNX model {resolved}: {exc}") from exc
        if net is None or net.empty():
            raise ModelError(f"loaded an empty network from {resolved}")
        # OpenCV 5's graph engine ignores setPreferableBackend/Target and warns
        # when they are called, so inference simply runs on the CPU here.
        return net

    def _resolve_output_names(self) -> list[str]:
        try:
            names = list(self.net.getUnconnectedOutLayersNames())
        except Exception as exc:
            raise ModelError(f"could not inspect the network outputs: {exc}") from exc
        if not names:
            raise ModelError("the network reports no output layers")
        return names

    def _forward(self, blob: np.ndarray) -> list[np.ndarray]:
        try:
            self.net.setInput(blob)
            outputs = self.net.forward(self._output_names)
        except Exception as exc:
            raise ModelError(f"YOLO forward pass failed: {exc}") from exc
        if isinstance(outputs, np.ndarray):
            return [outputs]
        return list(outputs)

    def _suppress(self, raw: RawDetections, transform: Letterbox) -> tuple[Detection, ...]:
        """Apply the class filter and per-class NMS, then map back to frame pixels."""
        boxes, scores, class_ids = raw.boxes, raw.scores, raw.class_ids

        if self.class_filter is not None:
            wanted = np.fromiter(sorted(self.class_filter), dtype=np.int32)
            keep = np.isin(class_ids, wanted)
            if not np.any(keep):
                return ()
            boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]

        # Per-class NMS: two genuinely overlapping objects of different classes
        # (a person and the tie they wear) must not suppress each other, so a
        # box only competes against boxes carrying the same class id.
        try:
            indices = self._cv2.dnn.NMSBoxesBatched(
                [tuple(float(value) for value in box) for box in boxes],
                [float(value) for value in scores],
                [int(value) for value in class_ids],
                self.confidence_threshold,
                self.nms_threshold,
            )
        except Exception as exc:
            raise ModelError(f"non-maximum suppression failed: {exc}") from exc

        survivors = np.asarray(indices, dtype=np.int32).reshape(-1)
        if survivors.size == 0:
            return ()
        # Draw and report the most confident objects first.
        order = survivors[np.argsort(-scores[survivors], kind="stable")]
        corners = transform.to_source(boxes[order])
        return tuple(
            Detection(
                float(corner[0]),
                float(corner[1]),
                float(corner[2]),
                float(corner[3]),
                float(scores[index]),
                int(class_ids[index]),
                class_name(self.labels, int(class_ids[index])),
            )
            for corner, index in zip(corners, order)
        )

    def detect(self, frame: np.ndarray) -> InferenceResult:
        """Detect objects in one BGR frame."""
        started = self._clock()
        padded, transform = letterbox(frame, self.input_size, cv2_module=self._cv2)
        try:
            blob = self._cv2.dnn.blobFromImage(
                padded, BLOB_SCALE, self.input_size, swapRB=True, crop=False
            )
        except Exception as exc:
            raise ModelError(f"could not build the input blob: {exc}") from exc
        preprocessed = self._clock()

        outputs = self._forward(blob)
        inferred = self._clock()

        raw = decode(
            outputs,
            self.family,
            confidence_threshold=self.confidence_threshold,
            num_classes=len(self.labels),
        )
        detections = self._suppress(raw, transform) if len(raw) else ()
        finished = self._clock()

        return InferenceResult(
            detections=detections,
            candidates=len(raw),
            preprocess_ms=(preprocessed - started) * 1000.0,
            inference_ms=(inferred - preprocessed) * 1000.0,
            postprocess_ms=(finished - inferred) * 1000.0,
        )

    def warm_up(self) -> None:
        """Run one throwaway pass so the first measured frame is not an outlier.

        OpenCV allocates layer buffers lazily, so an un-warmed first inference
        can take several times longer than steady state. Without this, both the
        on-screen FPS and the benchmark would start from a misleading number.
        """
        width, height = self.input_size
        self.detect(np.zeros((height, width, 3), dtype=np.uint8))


def count_by_label(detections: Sequence[Detection]) -> dict[str, int]:
    """Objects per class, ordered by descending count then name."""
    counts: dict[str, int] = {}
    for detection in detections:
        counts[detection.label] = counts.get(detection.label, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
