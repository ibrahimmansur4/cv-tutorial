"""Lightweight IoU association for plate detections in adjacent frames."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .ocr import OcrResult
from .vision import Box, PlateCandidate


@dataclass
class TrackedPlate:
    """Mutable state retained for one plate while it remains visible."""

    track_id: int
    box: Box
    plate_image: np.ndarray
    detection_confidence: float
    last_seen_frame: int
    missed_frames: int = 0
    text: str = ""
    ocr_confidence: float = 0.0
    last_ocr_frame: Optional[int] = None

    def update(self, candidate: PlateCandidate, frame_index: int) -> None:
        self.box = candidate.box
        self.plate_image = candidate.plate_image
        self.detection_confidence = candidate.confidence
        self.last_seen_frame = frame_index
        self.missed_frames = 0

    def needs_ocr(self, frame_index: int, interval: int) -> bool:
        if interval <= 0:
            raise ValueError("OCR interval must be positive")
        return self.last_ocr_frame is None or frame_index - self.last_ocr_frame >= interval

    def cache_ocr(self, result: OcrResult, frame_index: int) -> None:
        # Keep the last readable value when one noisy frame produces no text.
        if result.text or not self.text:
            self.text = result.text
            self.ocr_confidence = result.confidence
        self.last_ocr_frame = frame_index


class IoUTracker:
    """Associate current candidates with prior tracks using greedy IoU matching."""

    def __init__(self, iou_threshold: float = 0.30, max_missed_frames: int = 8) -> None:
        if not math.isfinite(iou_threshold) or not 0.0 < iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be greater than 0 and at most 1")
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames cannot be negative")
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self._tracks: Dict[int, TrackedPlate] = {}
        self._next_track_id = 1
        self._last_frame_index: Optional[int] = None

    @property
    def tracks(self) -> Tuple[TrackedPlate, ...]:
        return tuple(self._tracks[key] for key in sorted(self._tracks))

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1
        self._last_frame_index = None

    def _new_track(self, candidate: PlateCandidate, frame_index: int) -> TrackedPlate:
        track = TrackedPlate(
            track_id=self._next_track_id,
            box=candidate.box,
            plate_image=candidate.plate_image,
            detection_confidence=candidate.confidence,
            last_seen_frame=frame_index,
        )
        self._next_track_id += 1
        self._tracks[track.track_id] = track
        return track

    def update(
        self, candidates: Sequence[PlateCandidate], frame_index: int
    ) -> Sequence[TrackedPlate]:
        """Update state and return only tracks observed in this frame."""

        if frame_index < 0:
            raise ValueError("frame_index cannot be negative")
        if self._last_frame_index is not None and frame_index <= self._last_frame_index:
            raise ValueError("frame_index must increase for every tracker update")
        self._last_frame_index = frame_index

        candidate_list = list(candidates)
        possible_matches = []
        for track_id, track in self._tracks.items():
            for candidate_index, candidate in enumerate(candidate_list):
                overlap = track.box.iou(candidate.box)
                if overlap >= self.iou_threshold:
                    possible_matches.append((overlap, track_id, candidate_index))
        possible_matches.sort(reverse=True)

        matched_tracks = set()
        matched_candidates = set()
        observed: Dict[int, TrackedPlate] = {}
        for _, track_id, candidate_index in possible_matches:
            if track_id in matched_tracks or candidate_index in matched_candidates:
                continue
            track = self._tracks[track_id]
            track.update(candidate_list[candidate_index], frame_index)
            matched_tracks.add(track_id)
            matched_candidates.add(candidate_index)
            observed[candidate_index] = track

        for track_id, track in tuple(self._tracks.items()):
            if track_id not in matched_tracks:
                track.missed_frames += 1
                if track.missed_frames > self.max_missed_frames:
                    del self._tracks[track_id]

        for candidate_index, candidate in enumerate(candidate_list):
            if candidate_index not in matched_candidates:
                observed[candidate_index] = self._new_track(candidate, frame_index)

        return tuple(observed[index] for index in sorted(observed))
