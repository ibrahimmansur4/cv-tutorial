from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from src.app import AppConfig, build_parser, config_from_args, parse_size, run
from src.controller import PreviewMouseController, PyAutoGuiMouseController
from src.errors import ConfigurationError, MediaError, TrackingError
from src.geometry import CursorMapper, Point, joint_angle
from src.gestures import (
    FingerState,
    Gesture,
    GestureClassifier,
    GestureDebouncer,
    GestureResult,
)
from src.hand_tracking import HandObservation, MediaPipeHandTracker
from src.pipeline import VirtualMousePipeline
from src.smoothing import ExponentialCursorSmoother


def landmarks_for(gesture: Gesture) -> tuple[Point, ...]:
    points = [Point(0.5, 0.5) for _ in range(21)]
    points[0] = Point(0.5, 0.9)
    finger_joints = ((5, 6, 8, 0.38), (9, 10, 12, 0.46), (13, 14, 16, 0.54), (17, 18, 20, 0.62))

    extended = {
        Gesture.MOVE: (True, False, False, False),
        Gesture.LEFT_CLICK: (True, False, False, False),
        Gesture.RIGHT_CLICK: (True, True, False, False),
        Gesture.DRAG: (False, False, False, False),
        Gesture.NONE: (True, True, True, True),
    }[gesture]
    for (mcp, pip, tip, x), is_extended in zip(finger_joints, extended):
        points[mcp] = Point(x, 0.68)
        points[pip] = Point(x, 0.53)
        points[tip] = Point(x, 0.33) if is_extended else Point(x + 0.09, 0.53)

    points[4] = Point(0.18, 0.55)
    if gesture is Gesture.LEFT_CLICK:
        points[4] = Point(points[8].x + 0.01, points[8].y + 0.01)
    return tuple(points)


def observation(gesture: Gesture = Gesture.MOVE) -> HandObservation:
    return HandObservation(landmarks_for(gesture), "Right", 0.95)


class GeometryAndGestureTests(unittest.TestCase):
    def test_joint_angle_and_cursor_mapping(self) -> None:
        self.assertAlmostEqual(joint_angle(Point(0, 0), Point(1, 0), Point(2, 0)), 180.0)
        mapper = CursorMapper(1920, 1080, margin=0.1)
        self.assertEqual(mapper.map(Point(0.1, 0.1)), Point(0.0, 0.0))
        self.assertEqual(mapper.map(Point(0.9, 0.9)), Point(1919.0, 1079.0))
        self.assertEqual(mapper.frame_rectangle((720, 1280, 3)), (128, 72, 1152, 648))

    def test_classifier_recognizes_required_gestures(self) -> None:
        classifier = GestureClassifier()
        for expected in Gesture:
            with self.subTest(expected=expected):
                result = classifier.classify(landmarks_for(expected))
                self.assertEqual(result.gesture, expected)
        self.assertLess(classifier.classify(landmarks_for(Gesture.LEFT_CLICK)).pinch_ratio, 0.32)

    def test_classifier_rejects_wrong_landmark_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "21"):
            GestureClassifier().classify((Point(0.0, 0.0),))

    def test_debouncer_stability_entry_and_cooldown(self) -> None:
        debouncer = GestureDebouncer(stable_frames=2, cooldown_seconds=1.0)
        self.assertEqual(debouncer.update(Gesture.LEFT_CLICK, 0.0).stable, Gesture.NONE)
        first = debouncer.update(Gesture.LEFT_CLICK, 0.1)
        self.assertTrue(first.triggered)
        self.assertFalse(debouncer.update(Gesture.LEFT_CLICK, 0.2).triggered)
        debouncer.update(Gesture.MOVE, 0.3)
        debouncer.update(Gesture.MOVE, 0.4)
        debouncer.update(Gesture.LEFT_CLICK, 0.5)
        self.assertFalse(debouncer.update(Gesture.LEFT_CLICK, 0.6).triggered)
        debouncer.update(Gesture.MOVE, 1.2)
        debouncer.update(Gesture.MOVE, 1.3)
        debouncer.update(Gesture.LEFT_CLICK, 1.4)
        self.assertTrue(debouncer.update(Gesture.LEFT_CLICK, 1.5).triggered)

    def test_smoothing_and_dead_zone(self) -> None:
        smoother = ExponentialCursorSmoother(alpha=0.5, dead_zone=2.0)
        self.assertEqual(smoother.update(Point(10, 10)).point, Point(10, 10))
        suppressed = smoother.update(Point(12, 10))
        self.assertTrue(suppressed.suppressed)
        self.assertEqual(suppressed.point, Point(10, 10))
        moved = smoother.update(Point(20, 10))
        self.assertFalse(moved.suppressed)
        self.assertEqual(moved.point, Point(15, 10))
        smoother.reset()
        self.assertIsNone(smoother.current)


class ControllerTests(unittest.TestCase):
    def test_preview_controller_records_and_releases_drag(self) -> None:
        controller = PreviewMouseController(800, 600)
        controller.move_to(1, 2)
        controller.click("right")
        controller.mouse_down()
        controller.close()
        self.assertEqual(
            controller.actions,
            [("move", 1, 2), ("click", "right", None), ("down", "left", None), ("up", "left", None)],
        )

    def test_pyautogui_adapter_uses_fail_safe_and_releases_drag(self) -> None:
        class FakePyAutoGui:
            FAILSAFE = False
            PAUSE = 1.0

            def __init__(self) -> None:
                self.calls: list[tuple[str, tuple, dict]] = []

            def size(self):
                return 1024, 768

            def __getattr__(self, name):
                def call(*args, **kwargs):
                    self.calls.append((name, args, kwargs))

                return call

        fake = FakePyAutoGui()
        controller = PyAutoGuiMouseController(fake)
        controller.move_to(10, 20)
        controller.click("left")
        controller.mouse_down()
        controller.close()
        self.assertTrue(fake.FAILSAFE)
        self.assertEqual(fake.PAUSE, 0.0)
        self.assertEqual([call[0] for call in fake.calls], ["moveTo", "click", "mouseDown", "mouseUp"])


class HandTrackerTests(unittest.TestCase):
    class FakeMp:
        class ImageFormat:
            SRGB = "srgb"

        class Image:
            def __init__(self, *, image_format, data):
                self.image_format = image_format
                self.data = data

    class FakeLandmarker:
        def __init__(self, result=None) -> None:
            self.result = result or SimpleNamespace(hand_landmarks=[], handedness=[])
            self.timestamps: list[int] = []

        def detect_for_video(self, image, timestamp):
            self.timestamps.append(timestamp)
            return self.result

    def test_adapter_converts_result_and_enforces_monotonic_time(self) -> None:
        raw = [SimpleNamespace(x=index / 100, y=index / 100, z=-index / 1000) for index in range(21)]
        category = SimpleNamespace(category_name="Left", score=0.91)
        landmarker = self.FakeLandmarker(SimpleNamespace(hand_landmarks=[raw], handedness=[[category]]))
        tracker = MediaPipeHandTracker("unused", mediapipe_module=self.FakeMp, landmarker=landmarker)
        result = tracker.detect(np.zeros((20, 30, 3), dtype=np.uint8), 5)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].handedness, "Left")
        self.assertAlmostEqual(result[0].landmarks[20].x, 0.2)
        with self.assertRaisesRegex(TrackingError, "increase"):
            tracker.detect(np.zeros((20, 30, 3), dtype=np.uint8), 5)

    def test_adapter_validates_inputs_even_with_injected_landmarker(self) -> None:
        with self.assertRaises(ConfigurationError):
            MediaPipeHandTracker(
                "unused",
                detection_confidence=1.1,
                mediapipe_module=self.FakeMp,
                landmarker=self.FakeLandmarker(),
            )
        tracker = MediaPipeHandTracker(
            "unused", mediapipe_module=self.FakeMp, landmarker=self.FakeLandmarker()
        )
        with self.assertRaisesRegex(TrackingError, "empty"):
            tracker.detect(np.asarray([]), 0)


class FakeTracker:
    def __init__(self, hands_per_frame) -> None:
        self.hands_per_frame = iter(hands_per_frame)
        self.closed = False
        self.timestamps: list[int] = []

    def detect(self, frame, timestamp_ms):
        self.timestamps.append(timestamp_ms)
        return next(self.hands_per_frame)

    def draw(self, frame, hand) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class FakeClassifier:
    def __init__(self, gestures) -> None:
        self.gestures = iter(gestures)

    def classify(self, landmarks) -> GestureResult:
        return GestureResult(next(self.gestures), FingerState(False, False, False, False), 1.0)


class PipelineTests(unittest.TestCase):
    def test_move_click_right_click_drag_and_no_hand_release(self) -> None:
        gestures = [Gesture.MOVE, Gesture.LEFT_CLICK, Gesture.LEFT_CLICK, Gesture.RIGHT_CLICK, Gesture.DRAG, Gesture.DRAG]
        tracker = FakeTracker([[observation()] for _ in gestures] + [[]])
        controller = PreviewMouseController(100, 100)
        pipeline = VirtualMousePipeline(
            tracker,
            controller,
            classifier=FakeClassifier(gestures),
            debouncer=GestureDebouncer(stable_frames=1, cooldown_seconds=0.0),
            smoother=ExponentialCursorSmoother(alpha=1.0, dead_zone=0.0),
        )
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        results = [pipeline.process(frame, index, float(index)) for index in range(7)]
        actions = [action for result in results for action in result.actions]
        self.assertEqual(actions.count("left_click"), 1)
        self.assertEqual(actions.count("right_click"), 1)
        self.assertIn("drag_start", actions)
        self.assertIn("drag_move", actions)
        self.assertEqual(actions[-1], "drag_end")
        self.assertFalse(pipeline.dragging)
        annotated = pipeline.annotate(frame, results[0], 30.0)
        self.assertEqual(annotated.shape, frame.shape)
        self.assertGreater(int(annotated.sum()), 0)
        pipeline.close()
        self.assertTrue(tracker.closed)


def app_config(input_path: Path, output_path: Path | None = None) -> AppConfig:
    return AppConfig(
        camera=0,
        input_path=input_path,
        output_path=output_path,
        model_path=Path("unused"),
        resolution=(320, 240),
        preview_screen=(800, 600),
        max_frames=0,
        display=False,
        mirror=False,
        enable_control=False,
        num_hands=1,
        detection_confidence=0.5,
        presence_confidence=0.5,
        tracking_confidence=0.5,
        pinch_threshold=0.32,
        stable_frames=1,
        click_cooldown=0.0,
        smoothing_alpha=0.35,
        dead_zone=2.0,
        active_margin=0.08,
    )


class AppTests(unittest.TestCase):
    def test_cli_validation_and_safe_default(self) -> None:
        self.assertEqual(parse_size("640x480"), (640, 480))
        args = build_parser().parse_args(["--no-display"])
        self.assertFalse(config_from_args(args).enable_control)
        args = build_parser().parse_args(["--input", "clip.avi", "--enable-control"])
        with self.assertRaisesRegex(ConfigurationError, "recorded video"):
            config_from_args(args)

    def test_recorded_video_loop_and_annotated_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "input.avi"
            output_path = Path(temporary) / "output.avi"
            writer = cv2.VideoWriter(
                str(input_path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48)
            )
            self.assertTrue(writer.isOpened())
            for level in (0, 80, 160):
                writer.write(np.full((48, 64, 3), level, dtype=np.uint8))
            writer.release()

            tracker = FakeTracker([[], [], []])
            controller = PreviewMouseController(800, 600)
            stats = run(app_config(input_path, output_path), tracker=tracker, controller=controller)
            self.assertEqual(stats.frames, 3)
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)
            self.assertTrue(tracker.closed)

    def test_empty_video_is_reported_and_resources_are_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "empty.avi"
            input_path.touch()
            tracker = FakeTracker([])
            controller = PreviewMouseController()
            with self.assertRaises(MediaError):
                run(app_config(input_path), tracker=tracker, controller=controller)
            self.assertTrue(tracker.closed)

    def test_controller_is_closed_when_tracker_creation_fails(self) -> None:
        controller = PreviewMouseController()
        controller.mouse_down()
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "unused.avi"
            with patch("src.app.MediaPipeHandTracker", side_effect=RuntimeError("failed")):
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    run(app_config(input_path), tracker=None, controller=controller)
        # close() is idempotent and leaves the preview controller out of drag state.
        self.assertFalse(controller._left_down)
        self.assertEqual(controller.actions[-1], ("up", "left", None))


if __name__ == "__main__":
    unittest.main()
