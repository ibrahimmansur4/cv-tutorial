"""Frame-level hand tracking, gesture gating, and cursor actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .controller import MouseController
from .geometry import CursorMapper, Point
from .gestures import (
    INDEX_TIP,
    THUMB_TIP,
    Gesture,
    GestureClassifier,
    GestureDecision,
    GestureDebouncer,
    GestureResult,
)
from .hand_tracking import HandObservation, MediaPipeHandTracker
from .smoothing import ExponentialCursorSmoother


@dataclass(frozen=True, slots=True)
class FrameResult:
    hands: tuple[HandObservation, ...]
    classification: GestureResult | None
    decision: GestureDecision
    raw_cursor: Point | None
    cursor: Point | None
    jitter_suppressed: bool
    actions: tuple[str, ...]


class VirtualMousePipeline:
    """Convert MediaPipe landmarks into de-bounced mouse operations."""

    def __init__(
        self,
        tracker: MediaPipeHandTracker,
        controller: MouseController,
        *,
        classifier: GestureClassifier | None = None,
        debouncer: GestureDebouncer | None = None,
        smoother: ExponentialCursorSmoother | None = None,
        active_margin: float = 0.08,
        control_enabled: bool = False,
        cv2_module: Any = cv2,
    ) -> None:
        self.tracker = tracker
        self.controller = controller
        self.classifier = classifier or GestureClassifier()
        self.debouncer = debouncer or GestureDebouncer()
        self.smoother = smoother or ExponentialCursorSmoother()
        screen_width, screen_height = controller.screen_size
        self.mapper = CursorMapper(screen_width, screen_height, active_margin)
        self.control_enabled = bool(control_enabled)
        self._cv2 = cv2_module
        self._dragging = False

    @property
    def dragging(self) -> bool:
        return self._dragging

    def _release_drag(self, actions: list[str]) -> None:
        if self._dragging:
            self.controller.mouse_up("left")
            self._dragging = False
            actions.append("drag_end")

    def process(self, frame: np.ndarray, timestamp_ms: int, now_seconds: float) -> FrameResult:
        hands = tuple(self.tracker.detect(frame, timestamp_ms))
        actions: list[str] = []
        if not hands:
            self._release_drag(actions)
            self.debouncer.reset()
            self.smoother.reset()
            return FrameResult(
                hands,
                None,
                GestureDecision(Gesture.NONE, Gesture.NONE, False),
                None,
                None,
                False,
                tuple(actions),
            )

        hand = hands[0]
        classification = self.classifier.classify(hand.landmarks)
        decision = self.debouncer.update(classification.gesture, now_seconds)
        raw_cursor = self.mapper.map(hand.landmarks[INDEX_TIP])
        smoothed = self.smoother.update(raw_cursor)
        cursor = smoothed.point

        if self._dragging and decision.stable != Gesture.DRAG:
            self._release_drag(actions)

        if decision.stable in {
            Gesture.MOVE,
            Gesture.LEFT_CLICK,
            Gesture.RIGHT_CLICK,
            Gesture.DRAG,
        }:
            self.controller.move_to(round(cursor.x), round(cursor.y))
            actions.append("move")

        if decision.stable == Gesture.DRAG:
            if not self._dragging:
                self.controller.mouse_down("left")
                self._dragging = True
                actions.append("drag_start")
            else:
                actions.append("drag_move")
        elif decision.triggered and decision.stable == Gesture.LEFT_CLICK:
            self.controller.click("left")
            actions.append("left_click")
        elif decision.triggered and decision.stable == Gesture.RIGHT_CLICK:
            self.controller.click("right")
            actions.append("right_click")

        return FrameResult(
            hands,
            classification,
            decision,
            raw_cursor,
            cursor,
            smoothed.suppressed,
            tuple(actions),
        )

    def annotate(self, frame: np.ndarray, result: FrameResult, fps: float) -> np.ndarray:
        canvas = frame.copy()
        x1, y1, x2, y2 = self.mapper.frame_rectangle(canvas.shape)
        self._cv2.rectangle(canvas, (x1, y1), (x2, y2), (150, 150, 150), 1)
        for hand in result.hands:
            self.tracker.draw(canvas, hand)

        if result.hands:
            hand = result.hands[0]
            height, width = canvas.shape[:2]
            index_tip = hand.landmarks[INDEX_TIP]
            thumb_tip = hand.landmarks[THUMB_TIP]
            index_pixel = (int(index_tip.x * width), int(index_tip.y * height))
            thumb_pixel = (int(thumb_tip.x * width), int(thumb_tip.y * height))
            self._cv2.line(canvas, thumb_pixel, index_pixel, (255, 80, 220), 2)
            self._cv2.circle(canvas, index_pixel, 9, (0, 255, 255), 2)

        raw_name = result.decision.raw.value.replace("_", " ").upper()
        stable_name = result.decision.stable.value.replace("_", " ").upper()
        mode = "CONTROL ENABLED" if self.control_enabled else "SAFE PREVIEW"
        color = (40, 70, 255) if self.control_enabled else (60, 220, 60)
        lines = [
            f"Mode: {mode}",
            f"Gesture: {stable_name} (raw {raw_name})",
            f"FPS: {fps:.1f}",
        ]
        if result.classification is not None:
            lines.append(f"Pinch ratio: {result.classification.pinch_ratio:.3f}")
        if result.actions:
            lines.append(f"Actions: {', '.join(result.actions)}")
        if result.jitter_suppressed:
            lines.append("Jitter: suppressed")
        for index, text in enumerate(lines):
            self._cv2.putText(
                canvas,
                text,
                (12, 28 + 25 * index),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color if index == 0 else (255, 255, 255),
                2,
                self._cv2.LINE_AA,
            )
        return canvas

    def close(self) -> None:
        actions: list[str] = []
        try:
            self._release_drag(actions)
        finally:
            try:
                self.controller.close()
            finally:
                self.tracker.close()
