"""Face detection, embedding, recognition, and attendance orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import cv2
import numpy as np

from .attendance_log import AttendanceLogger
from .detection import FaceBox, FaceDetector
from .embeddings import FaceIndex, HOGFaceEmbedder, Match


@dataclass(frozen=True, slots=True)
class FaceObservation:
    box: FaceBox
    match: Match
    attendance_recorded: bool = False

    @property
    def name(self) -> str:
        return self.match.name or "Unknown"


class RecognitionPipeline:
    """Testable frame-level recognition pipeline."""

    def __init__(
        self,
        detector: FaceDetector,
        embedder: HOGFaceEmbedder,
        index: FaceIndex,
        attendance: AttendanceLogger | None = None,
        *,
        threshold: float = 0.35,
        cv2_module: Any = cv2,
    ) -> None:
        if threshold < 0.0 or threshold > 2.0:
            raise ValueError("threshold must be between 0 and 2")
        self.detector = detector
        self.embedder = embedder
        self.index = index
        self.attendance = attendance
        self.threshold = float(threshold)
        self._cv2 = cv2_module

    def process(
        self,
        frame: np.ndarray,
        *,
        timestamp: datetime | None = None,
    ) -> list[FaceObservation]:
        observations: list[FaceObservation] = []
        for box in self.detector.detect(frame):
            descriptor = self.embedder.embed(box.crop(frame))
            match = self.index.match(descriptor, self.threshold)
            marked = False
            if match.name is not None and self.attendance is not None:
                marked = self.attendance.mark(match.name, when=timestamp)
            observations.append(FaceObservation(box, match, marked))
        return observations

    def annotate(
        self,
        frame: np.ndarray,
        observations: list[FaceObservation],
    ) -> np.ndarray:
        """Draw readable labels on a copy of the source frame."""

        canvas = frame.copy()
        for observation in observations:
            box = observation.box
            recognized = observation.match.recognized
            color = (60, 200, 60) if recognized else (40, 150, 255)
            self._cv2.rectangle(canvas, (box.x, box.y), (box.x2, box.y2), color, 2)
            if recognized:
                label = f"{observation.name}  d={observation.match.distance:.3f}"
                if observation.attendance_recorded:
                    label += "  MARKED"
            else:
                label = "Unknown"
            (text_width, text_height), baseline = self._cv2.getTextSize(
                label,
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                1,
            )
            label_top = max(0, box.y - text_height - baseline - 8)
            self._cv2.rectangle(
                canvas,
                (box.x, label_top),
                (min(canvas.shape[1] - 1, box.x + text_width + 8), box.y),
                color,
                -1,
            )
            self._cv2.putText(
                canvas,
                label,
                (box.x + 4, max(text_height + 1, box.y - baseline - 4)),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (20, 20, 20),
                1,
                self._cv2.LINE_AA,
            )
        return canvas
