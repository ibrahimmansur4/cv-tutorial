"""Landmark-based gesture classification and temporal de-bouncing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .errors import ConfigurationError
from .geometry import Point, joint_angle

WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20


class Gesture(str, Enum):
    NONE = "none"
    MOVE = "move"
    LEFT_CLICK = "left_click"
    RIGHT_CLICK = "right_click"
    DRAG = "drag"


@dataclass(frozen=True, slots=True)
class FingerState:
    index: bool
    middle: bool
    ring: bool
    pinky: bool


@dataclass(frozen=True, slots=True)
class GestureResult:
    gesture: Gesture
    fingers: FingerState
    pinch_ratio: float


class GestureClassifier:
    """Classify four tutorial gestures from 21 normalized hand landmarks."""

    def __init__(self, pinch_threshold: float = 0.32, extension_angle: float = 155.0) -> None:
        if not math.isfinite(pinch_threshold) or pinch_threshold <= 0.0:
            raise ConfigurationError("pinch threshold must be positive")
        if not math.isfinite(extension_angle) or not 90.0 <= extension_angle <= 180.0:
            raise ConfigurationError("extension angle must be between 90 and 180 degrees")
        self.pinch_threshold = float(pinch_threshold)
        self.extension_angle = float(extension_angle)

    def _extended(self, landmarks: Sequence[Point], mcp: int, pip: int, tip: int) -> bool:
        return joint_angle(landmarks[mcp], landmarks[pip], landmarks[tip]) >= self.extension_angle

    def classify(self, landmarks: Sequence[Point]) -> GestureResult:
        if len(landmarks) != 21:
            raise ValueError("gesture classification requires exactly 21 landmarks")
        fingers = FingerState(
            self._extended(landmarks, INDEX_MCP, INDEX_PIP, INDEX_TIP),
            self._extended(landmarks, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
            self._extended(landmarks, RING_MCP, RING_PIP, RING_TIP),
            self._extended(landmarks, PINKY_MCP, PINKY_PIP, PINKY_TIP),
        )
        palm_scale = max(
            landmarks[WRIST].distance_to(landmarks[MIDDLE_MCP]),
            landmarks[INDEX_MCP].distance_to(landmarks[PINKY_MCP]),
            1e-6,
        )
        pinch_ratio = landmarks[THUMB_TIP].distance_to(landmarks[INDEX_TIP]) / palm_scale

        if fingers.index and pinch_ratio <= self.pinch_threshold:
            gesture = Gesture.LEFT_CLICK
        elif fingers.index and fingers.middle and not fingers.ring and not fingers.pinky:
            gesture = Gesture.RIGHT_CLICK
        elif not any((fingers.index, fingers.middle, fingers.ring, fingers.pinky)):
            gesture = Gesture.DRAG
        elif fingers.index and not any((fingers.middle, fingers.ring, fingers.pinky)):
            gesture = Gesture.MOVE
        else:
            gesture = Gesture.NONE
        return GestureResult(gesture, fingers, pinch_ratio)


@dataclass(frozen=True, slots=True)
class GestureDecision:
    raw: Gesture
    stable: Gesture
    triggered: bool


class GestureDebouncer:
    """Require stable frames and emit click triggers only on gesture entry."""

    DISCRETE = frozenset({Gesture.LEFT_CLICK, Gesture.RIGHT_CLICK})

    def __init__(self, stable_frames: int = 3, cooldown_seconds: float = 0.45) -> None:
        if stable_frames <= 0:
            raise ConfigurationError("stable frame count must be positive")
        if not math.isfinite(cooldown_seconds) or cooldown_seconds < 0.0:
            raise ConfigurationError("click cooldown must be non-negative")
        self.stable_frames = int(stable_frames)
        self.cooldown_seconds = float(cooldown_seconds)
        self._candidate = Gesture.NONE
        self._candidate_frames = 0
        self._stable = Gesture.NONE
        self._last_triggered_at = float("-inf")

    @property
    def stable(self) -> Gesture:
        return self._stable

    def reset(self) -> None:
        self._candidate = Gesture.NONE
        self._candidate_frames = 0
        self._stable = Gesture.NONE

    def update(self, raw: Gesture, now_seconds: float) -> GestureDecision:
        if not math.isfinite(now_seconds):
            raise ValueError("gesture timestamp must be finite")
        if raw == self._candidate:
            self._candidate_frames += 1
        else:
            self._candidate = raw
            self._candidate_frames = 1

        triggered = False
        if self._candidate_frames >= self.stable_frames and raw != self._stable:
            self._stable = raw
            if raw in self.DISCRETE and now_seconds - self._last_triggered_at >= self.cooldown_seconds:
                triggered = True
                self._last_triggered_at = now_seconds
        return GestureDecision(raw, self._stable, triggered)
