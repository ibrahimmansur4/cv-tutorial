"""Composition layer joining localization, tracking, and OCR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from .tracking import IoUTracker
from .vision import Box


@dataclass(frozen=True)
class PlateObservation:
    """One plate result ready for annotation or reporting."""

    box: Box
    detection_confidence: float
    text: str
    ocr_confidence: float
    track_id: Optional[int]
    ocr_performed: bool


@dataclass(frozen=True)
class FrameResult:
    """All ALPR results generated for one input frame."""

    observations: Tuple[PlateObservation, ...]
    ocr_calls: int


class ALPRPipeline:
    """Run localization and rate-limited OCR, with optional video tracking."""

    def __init__(
        self,
        detector: Any,
        ocr_engine: Any,
        *,
        tracker: Optional[IoUTracker] = None,
        ocr_interval: int = 15,
    ) -> None:
        if ocr_interval <= 0:
            raise ValueError("ocr_interval must be positive")
        self.detector = detector
        self.ocr_engine = ocr_engine
        self.tracker = tracker if tracker is not None else IoUTracker()
        self.ocr_interval = ocr_interval

    def process(
        self, frame: np.ndarray, frame_index: int, *, use_tracking: bool
    ) -> FrameResult:
        candidates = tuple(self.detector.detect(frame))
        observations = []
        ocr_calls = 0

        if not use_tracking:
            for candidate in candidates:
                result = self.ocr_engine.recognize(candidate.plate_image)
                ocr_calls += 1
                observations.append(
                    PlateObservation(
                        box=candidate.box,
                        detection_confidence=candidate.confidence,
                        text=result.text,
                        ocr_confidence=result.confidence,
                        track_id=None,
                        ocr_performed=True,
                    )
                )
            return FrameResult(tuple(observations), ocr_calls)

        tracks = self.tracker.update(candidates, frame_index)
        for track in tracks:
            performed = False
            if track.needs_ocr(frame_index, self.ocr_interval):
                result = self.ocr_engine.recognize(track.plate_image)
                track.cache_ocr(result, frame_index)
                ocr_calls += 1
                performed = True
            observations.append(
                PlateObservation(
                    box=track.box,
                    detection_confidence=track.detection_confidence,
                    text=track.text,
                    ocr_confidence=track.ocr_confidence,
                    track_id=track.track_id,
                    ocr_performed=performed,
                )
            )
        return FrameResult(tuple(observations), ocr_calls)
