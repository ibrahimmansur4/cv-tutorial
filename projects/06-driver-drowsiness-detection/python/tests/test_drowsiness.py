from __future__ import annotations

import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.alarm import ConsoleAlarm, NullAlarm, WindowsBeepAlarm, make_alarm
from src.app import (
    AppConfig,
    build_parser,
    config_from_args,
    parse_size,
    run,
    run_calibration,
)
from src.calibration import (
    DEFAULT_CLOSED_RATIO,
    Calibration,
    calibrate,
    percentile,
    summarize,
)
from src.drawing import draw_banner, draw_hud
from src.errors import ConfigurationError, LandmarkError, MediaError
from src.face_mesh import FaceObservation
from src.geometry import Point, distance
from src.metrics import (
    LEFT_EYE,
    MAX_LANDMARK_INDEX,
    MOUTH,
    RIGHT_EYE,
    aspect_ratio,
    compute_metrics,
)
from src.pipeline import DrowsinessPipeline, FpsMeter, RunStats
from src.state import (
    AlertLevel,
    DrowsinessMonitor,
    EyeState,
    MonitorConfig,
    MonitorState,
)


@contextlib.contextmanager
def quiet():
    """Swallow argparse usage text and CLI prints that tests trigger on purpose."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        yield buffer


def eye_points(opening: float, width: float = 0.08, x: float = 0.3, y: float = 0.5):
    """Six points in p1..p6 order for an eye of a given vertical opening.

    Both vertical pairs are `opening` apart and the corners are `width` apart,
    so the resulting aspect ratio is exactly `opening / width`.
    """
    half = width / 2.0
    return [
        Point(x - half, y),  # p1 outer corner
        Point(x - half / 2, y - opening / 2),  # p2 upper
        Point(x + half / 2, y - opening / 2),  # p3 upper
        Point(x + half, y),  # p4 inner corner
        Point(x + half / 2, y + opening / 2),  # p5 lower
        Point(x - half / 2, y + opening / 2),  # p6 lower
    ]


def face_landmarks(eye_opening: float = 0.03, mouth_opening: float = 0.02):
    """A full landmark list with the indices this project reads filled in."""
    points = [Point(0.5, 0.5) for _ in range(MAX_LANDMARK_INDEX + 1)]
    for indices, opening, centre in (
        (LEFT_EYE, eye_opening, 0.65),
        (RIGHT_EYE, eye_opening, 0.35),
        (MOUTH, mouth_opening, 0.5),
    ):
        for index, point in zip(indices, eye_points(opening, x=centre)):
            points[index] = point
    return tuple(points)


class GeometryTests(unittest.TestCase):
    def test_distance_rescales_x_by_the_frame_aspect(self) -> None:
        a, b = Point(0.0, 0.0), Point(0.1, 0.0)
        self.assertAlmostEqual(distance(a, b), 0.1)
        self.assertAlmostEqual(distance(a, b, 2.0), 0.2)

    def test_vertical_distance_is_untouched_by_the_aspect(self) -> None:
        a, b = Point(0.0, 0.0), Point(0.0, 0.1)
        self.assertAlmostEqual(distance(a, b, 1.78), 0.1)

    def test_invalid_aspect_is_rejected(self) -> None:
        for bad in (0.0, -1.0, float("nan")):
            with self.assertRaises(ValueError):
                distance(Point(0, 0), Point(1, 1), bad)

    def test_point_rejects_non_finite_coordinates(self) -> None:
        with self.assertRaises(ValueError):
            Point(float("nan"), 0.0)

    def test_to_pixels_clamps_into_the_frame(self) -> None:
        self.assertEqual(Point(0.5, 0.5).to_pixels(100, 200), (50, 100))
        self.assertEqual(Point(-1.0, 2.0).to_pixels(100, 200), (0, 199))


class MetricTests(unittest.TestCase):
    def test_aspect_ratio_matches_the_hand_computed_value(self) -> None:
        # (0.02 + 0.02) / (2 * 0.08) = 0.25
        self.assertAlmostEqual(aspect_ratio(eye_points(0.02, width=0.08)), 0.25)
        self.assertAlmostEqual(aspect_ratio(eye_points(0.04, width=0.08)), 0.50)

    def test_ratio_is_invariant_to_how_far_away_the_face_is(self) -> None:
        near = aspect_ratio(eye_points(0.04, width=0.08))
        far = aspect_ratio(eye_points(0.02, width=0.04))  # same shape, half the size
        self.assertAlmostEqual(near, far)

    def test_ratio_is_invariant_to_the_frame_aspect(self) -> None:
        """The regression test for normalized-coordinate distortion.

        MediaPipe normalizes x by width and y by height, so without correction
        the same eye reads differently on a portrait frame and a 16:9 one.
        """
        points = eye_points(0.04, width=0.08)
        square = aspect_ratio(points, 1.0)
        for frame_aspect in (16 / 9, 4 / 3, 820 / 1024):
            scaled = [Point(point.x / frame_aspect, point.y) for point in points]
            self.assertAlmostEqual(aspect_ratio(scaled, frame_aspect), square, places=9)

    def test_closing_the_eye_lowers_the_ratio(self) -> None:
        self.assertLess(aspect_ratio(eye_points(0.005)), aspect_ratio(eye_points(0.04)))

    def test_degenerate_and_malformed_inputs_are_rejected(self) -> None:
        coincident = [Point(0.5, 0.5) for _ in range(6)]
        with self.assertRaises(LandmarkError):
            aspect_ratio(coincident)
        with self.assertRaises(LandmarkError):
            aspect_ratio(eye_points(0.04)[:5])

    def test_compute_metrics_reads_both_eyes_and_the_mouth(self) -> None:
        metrics = compute_metrics(face_landmarks(eye_opening=0.02, mouth_opening=0.04))
        self.assertAlmostEqual(metrics.left_ear, 0.25)
        self.assertAlmostEqual(metrics.right_ear, 0.25)
        self.assertAlmostEqual(metrics.ear, 0.25)
        self.assertAlmostEqual(metrics.asymmetry, 0.0)
        self.assertAlmostEqual(metrics.mar, 0.5)

    def test_short_landmark_lists_are_rejected(self) -> None:
        with self.assertRaises(LandmarkError):
            compute_metrics([Point(0.5, 0.5)] * 10)

    def test_face_observation_requires_a_full_mesh(self) -> None:
        with self.assertRaises(ValueError):
            FaceObservation(tuple(Point(0.5, 0.5) for _ in range(10)))


class MonitorConfigTests(unittest.TestCase):
    def test_a_blink_ceiling_above_the_alarm_duration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MonitorConfig(blink_max_seconds=2.0, closure_seconds=1.5)

    def test_out_of_range_values_are_rejected(self) -> None:
        for kwargs in (
            {"ear_threshold": 0.0},
            {"ear_threshold": 1.5},
            {"ear_hysteresis": -0.1},
            {"mar_threshold": 0.0},
            {"closure_seconds": 0.0},
            {"yawn_seconds": -1.0},
            {"perclos_window_seconds": 0.0},
            {"perclos_threshold": 1.5},
        ):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                MonitorConfig(**kwargs)


class StateMachineTests(unittest.TestCase):
    OPEN = 0.30
    SHUT = 0.05

    def setUp(self) -> None:
        self.config = MonitorConfig(
            ear_threshold=0.15,
            ear_hysteresis=0.02,
            closure_seconds=1.5,
            blink_max_seconds=0.4,
            yawn_seconds=0.7,
            mar_threshold=0.6,
        )
        # These tests run a few seconds of synthetic footage, far less than the
        # default PERCLOS warm-up, so PERCLOS stays out of the verdict unless a
        # test opts in by configuring its own monitor.
        self.monitor = DrowsinessMonitor(self.config)
        self.now = 0.0

    def step(self, ear: float, *, mar: float = 0.1, dt: float = 0.05) -> MonitorState:
        self.now += dt
        return self.monitor.update(ear, mar, self.now)

    def hold(self, ear: float, seconds: float, *, mar: float = 0.1) -> MonitorState:
        state = self.monitor.update(ear, mar, self.now)
        target = self.now + seconds
        while self.now < target:
            state = self.step(ear, mar=mar)
        return state

    def test_open_eyes_stay_awake(self) -> None:
        state = self.hold(self.OPEN, 1.0)
        self.assertIs(state.eye_state, EyeState.OPEN)
        self.assertIs(state.level, AlertLevel.AWAKE)
        self.assertFalse(state.should_alarm)

    def test_a_short_closure_counts_as_a_blink_and_does_not_alarm(self) -> None:
        self.hold(self.OPEN, 1.0)
        state = self.hold(self.SHUT, 0.2)
        self.assertIs(state.level, AlertLevel.AWAKE)
        state = self.hold(self.OPEN, 0.3)
        self.assertEqual(state.blinks, 1)
        self.assertEqual(self.monitor.yawns, 0)

    def test_a_sustained_closure_raises_the_alarm(self) -> None:
        self.hold(self.OPEN, 1.0)
        state = self.hold(self.SHUT, 2.0)
        self.assertIs(state.eye_state, EyeState.CLOSED)
        self.assertIs(state.level, AlertLevel.DROWSY)
        self.assertTrue(state.should_alarm)
        self.assertGreaterEqual(state.closure_seconds, self.config.closure_seconds)

    def test_a_long_closure_is_not_also_counted_as_a_blink(self) -> None:
        self.hold(self.OPEN, 0.5)
        self.hold(self.SHUT, 2.0)
        state = self.hold(self.OPEN, 0.5)
        self.assertEqual(state.blinks, 0)

    def test_the_alarm_clears_once_the_eyes_reopen(self) -> None:
        self.hold(self.OPEN, 0.5)
        self.assertTrue(self.hold(self.SHUT, 2.0).should_alarm)
        self.assertFalse(self.hold(self.OPEN, 0.5).should_alarm)

    def test_hysteresis_prevents_chattering_at_the_threshold(self) -> None:
        """An EAR sitting between the threshold and the release point holds state."""
        self.hold(self.OPEN, 0.5)
        self.step(0.14)  # below threshold: closes
        self.assertIs(self.monitor._eye_state, EyeState.CLOSED)
        # 0.16 is above the threshold but inside the hysteresis band, so the
        # eye stays closed rather than flickering open.
        self.step(0.16)
        self.assertIs(self.monitor._eye_state, EyeState.CLOSED)
        self.step(0.18)  # clears threshold + hysteresis: opens
        self.assertIs(self.monitor._eye_state, EyeState.OPEN)

    def test_perclos_tracks_the_closed_fraction_of_the_window(self) -> None:
        self.hold(self.OPEN, 1.0)
        self.assertAlmostEqual(self.monitor.perclos, 0.0, places=2)
        self.hold(self.SHUT, 1.0)
        self.assertGreater(self.monitor.perclos, 0.4)
        self.assertLess(self.monitor.perclos, 0.6)

    def test_perclos_is_ignored_until_the_window_has_warmed_up(self) -> None:
        """One ordinary blink must not report fatigue seconds into a session.

        PERCLOS is a ratio over a window; while the window is nearly empty a
        single 0.2 s blink is a large fraction of everything seen so far.
        """
        self.hold(self.OPEN, 1.0)
        state = self.hold(self.SHUT, 0.2)
        self.assertGreater(self.monitor.perclos, self.config.perclos_threshold)
        self.assertFalse(self.monitor.perclos_ready)
        self.assertIs(state.level, AlertLevel.AWAKE)

    def test_warmup_is_clamped_to_the_window_length(self) -> None:
        monitor = DrowsinessMonitor(
            MonitorConfig(perclos_window_seconds=2.0, perclos_warmup_seconds=60.0)
        )
        now = 0.0
        for _ in range(60):
            now += 0.05
            monitor.update(0.30, 0.1, now)
        self.assertTrue(monitor.perclos_ready)

    def test_perclos_alone_reports_fatigue_without_alarming(self) -> None:
        monitor = DrowsinessMonitor(
            MonitorConfig(
                ear_threshold=0.15,
                closure_seconds=1.5,
                blink_max_seconds=0.4,
                perclos_window_seconds=10.0,
                perclos_threshold=0.15,
                perclos_warmup_seconds=2.0,
            )
        )
        now = 0.0
        # Repeated 0.3 s closures: each is a blink, none reaches the alarm
        # duration, but together they push PERCLOS over the threshold.
        for _ in range(12):
            for _ in range(6):
                now += 0.05
                state = monitor.update(self.SHUT, 0.1, now)
            for _ in range(8):
                now += 0.05
                state = monitor.update(self.OPEN, 0.1, now)
        self.assertGreater(monitor.perclos, 0.15)
        self.assertIs(state.level, AlertLevel.FATIGUE)
        self.assertFalse(state.should_alarm)

    def test_perclos_forgets_samples_older_than_the_window(self) -> None:
        monitor = DrowsinessMonitor(
            MonitorConfig(ear_threshold=0.15, perclos_window_seconds=1.0)
        )
        now = 0.0
        for _ in range(20):
            now += 0.05
            monitor.update(self.SHUT, 0.1, now)
        self.assertGreater(monitor.perclos, 0.9)
        for _ in range(40):
            now += 0.05
            monitor.update(self.OPEN, 0.1, now)
        self.assertAlmostEqual(monitor.perclos, 0.0, places=2)

    def test_a_held_open_mouth_counts_one_yawn(self) -> None:
        self.hold(self.OPEN, 0.5)
        state = self.hold(self.OPEN, 1.0, mar=0.8)
        self.assertTrue(state.yawning)
        self.assertEqual(state.yawns, 1)
        # Still open: it must not count again.
        state = self.hold(self.OPEN, 1.0, mar=0.8)
        self.assertEqual(state.yawns, 1)
        # Close, then yawn again.
        self.hold(self.OPEN, 0.5, mar=0.1)
        state = self.hold(self.OPEN, 1.0, mar=0.8)
        self.assertEqual(state.yawns, 2)

    def test_briefly_opening_the_mouth_is_not_a_yawn(self) -> None:
        self.hold(self.OPEN, 0.5)
        state = self.hold(self.OPEN, 0.3, mar=0.8)
        self.assertEqual(state.yawns, 0)
        self.assertFalse(state.yawning)

    def test_a_missing_face_is_its_own_state_not_a_closure(self) -> None:
        self.hold(self.OPEN, 1.0)
        self.now += 0.5
        state = self.monitor.update_missing(self.now)
        self.assertIs(state.eye_state, EyeState.UNKNOWN)
        self.assertFalse(state.should_alarm)
        self.assertTrue(math.isnan(state.ear))
        self.assertAlmostEqual(state.face_missing_seconds, 0.0)

    def test_a_prolonged_missing_face_is_reported(self) -> None:
        self.hold(self.OPEN, 1.0)
        for _ in range(60):
            self.now += 0.05
            state = self.monitor.update_missing(self.now)
        self.assertIs(state.level, AlertLevel.NO_FACE)
        self.assertGreaterEqual(state.face_missing_seconds, 2.0)

    def test_losing_the_face_abandons_an_in_progress_closure(self) -> None:
        self.hold(self.OPEN, 0.5)
        self.hold(self.SHUT, 1.0)
        self.now += 0.1
        self.monitor.update_missing(self.now)
        self.now += 0.1
        state = self.monitor.update(self.SHUT, 0.1, self.now)
        # The closure timer restarted rather than resuming at 1.0 s.
        self.assertLess(state.closure_seconds, 0.5)

    def test_reset_clears_all_history(self) -> None:
        self.hold(self.OPEN, 0.5)
        self.hold(self.SHUT, 0.2)
        self.hold(self.OPEN, 0.5)
        self.assertEqual(self.monitor.blinks, 1)
        self.monitor.reset()
        self.assertEqual(self.monitor.blinks, 0)
        self.assertEqual(self.monitor.perclos, 0.0)


class CalibrationTests(unittest.TestCase):
    def test_percentile_interpolates(self) -> None:
        self.assertAlmostEqual(percentile([0.0, 1.0], 0.5), 0.5)
        self.assertAlmostEqual(percentile([5.0], 0.9), 5.0)
        self.assertAlmostEqual(percentile([0.0, 1.0, 2.0, 3.0], 0.0), 0.0)
        with self.assertRaises(ConfigurationError):
            percentile([], 0.5)
        with self.assertRaises(ConfigurationError):
            percentile([1.0], 1.5)

    def test_threshold_is_a_fraction_of_the_measured_baseline(self) -> None:
        result = calibrate([0.30] * 40, closed_ratio=0.75)
        self.assertAlmostEqual(result.baseline_ear, 0.30, places=6)
        self.assertAlmostEqual(result.ear_threshold, 0.225, places=6)
        self.assertEqual(result.samples, 40)

    def test_a_few_blinks_during_capture_do_not_drag_the_baseline_down(self) -> None:
        samples = [0.30] * 36 + [0.05] * 4  # 10% of frames blinking
        result = calibrate(samples)
        self.assertAlmostEqual(result.baseline_ear, 0.30, places=2)

    def test_too_few_samples_is_an_actionable_error(self) -> None:
        with self.assertRaises(ConfigurationError) as context:
            calibrate([0.3] * 5)
        self.assertIn("at least", str(context.exception))

    def test_non_finite_samples_are_discarded(self) -> None:
        with self.assertRaises(ConfigurationError):
            calibrate([float("nan")] * 40)

    def test_invalid_closed_ratio_is_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            calibrate([0.3] * 40, closed_ratio=1.5)

    def test_profile_round_trips_through_disk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            original = calibrate([0.28] * 30)
            original.save(path)
            loaded = Calibration.load(path)
            self.assertAlmostEqual(loaded.ear_threshold, original.ear_threshold)
            self.assertAlmostEqual(loaded.baseline_ear, original.baseline_ear)
            self.assertEqual(loaded.closed_ratio, DEFAULT_CLOSED_RATIO)

    def test_missing_corrupt_and_stale_profiles_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "none.json"
            with self.assertRaises(ConfigurationError):
                Calibration.load(missing)

            corrupt = Path(directory) / "corrupt.json"
            corrupt.write_text("{not json", encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                Calibration.load(corrupt)

            stale = Path(directory) / "stale.json"
            stale.write_text(json.dumps({"version": 0, "baseline_ear": 0.3}), encoding="utf-8")
            with self.assertRaises(ConfigurationError) as context:
                Calibration.load(stale)
            self.assertIn("Re-run --calibrate", str(context.exception))

            incomplete = Path(directory) / "incomplete.json"
            incomplete.write_text(json.dumps({"version": 1}), encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                Calibration.load(incomplete)

    def test_summarize_describes_a_capture(self) -> None:
        self.assertIn("n=3", summarize([0.1, 0.2, 0.3]))
        self.assertEqual(summarize([]), "no usable samples")


class AlarmTests(unittest.TestCase):
    def test_null_alarm_records_without_sounding(self) -> None:
        alarm = NullAlarm()
        alarm.start()
        alarm.stop()
        alarm.close()
        self.assertEqual(alarm.started, 1)

    def test_console_alarm_writes_once_per_episode(self) -> None:
        stream = io.StringIO()
        alarm = ConsoleAlarm(stream)
        alarm.start()
        alarm.start()
        self.assertEqual(stream.getvalue().count("DROWSINESS"), 1)
        alarm.stop()
        alarm.start()
        self.assertEqual(stream.getvalue().count("DROWSINESS"), 2)

    def test_console_alarm_survives_a_broken_stream(self) -> None:
        stream = io.StringIO()
        stream.close()
        ConsoleAlarm(stream).start()  # must not raise

    def test_beep_alarm_swallows_backend_failures(self) -> None:
        def broken(frequency: int, duration: int) -> None:
            raise OSError("no audio device")

        WindowsBeepAlarm(beep=broken).start()  # must not raise

    def test_beep_alarm_calls_the_backend(self) -> None:
        calls: list[tuple[int, int]] = []
        WindowsBeepAlarm(beep=lambda f, d: calls.append((f, d))).start()
        self.assertEqual(len(calls), 1)

    def test_make_alarm_honours_the_disable_flag(self) -> None:
        self.assertIsInstance(make_alarm(False), NullAlarm)
        self.assertIsNotNone(make_alarm(True))


class FakeTracker:
    """Returns a scripted face (or none) and records draw calls."""

    def __init__(self, landmarks=None, *, raises: bool = False) -> None:
        self._landmarks = landmarks
        self._raises = raises
        self.closed = False
        self.drawn = 0

    def detect(self, frame, timestamp_ms):
        if self._raises:
            raise LandmarkError("synthetic tracking failure")
        return [FaceObservation(self._landmarks)] if self._landmarks else []

    def draw(self, frame, face, *, eyes_closed: bool) -> None:
        self.drawn += 1

    def close(self) -> None:
        self.closed = True


class PipelineTests(unittest.TestCase):
    def frame(self) -> np.ndarray:
        return np.full((480, 640, 3), 40, dtype=np.uint8)

    def test_an_open_eyed_face_reads_awake_and_annotates(self) -> None:
        tracker = FakeTracker(face_landmarks(eye_opening=0.04))
        pipeline = DrowsinessPipeline(
            tracker, DrowsinessMonitor(MonitorConfig(ear_threshold=0.15)), alarm=NullAlarm()
        )
        frame = self.frame()
        result = pipeline.process(frame, 100, 1.0, 0.03)
        self.assertIsNotNone(result.face)
        self.assertIs(result.state.eye_state, EyeState.OPEN)
        annotated = pipeline.annotate(frame, result, 1.0)
        self.assertEqual(annotated.shape, frame.shape)
        self.assertEqual(int(frame.max()), 40)  # source untouched
        self.assertEqual(tracker.drawn, 1)

    def test_a_sustained_closure_drives_the_alarm(self) -> None:
        tracker = FakeTracker(face_landmarks(eye_opening=0.002))
        alarm = NullAlarm()
        pipeline = DrowsinessPipeline(
            tracker,
            DrowsinessMonitor(MonitorConfig(ear_threshold=0.15, closure_seconds=1.0)),
            alarm=alarm,
        )
        frame = self.frame()
        now = 0.0
        for index in range(50):
            now += 0.05
            result = pipeline.process(frame, index + 1, now, 0.05)
        self.assertTrue(result.state.should_alarm)
        self.assertGreaterEqual(alarm.started, 1)

    def test_a_tracking_failure_is_treated_as_a_missing_face(self) -> None:
        pipeline = DrowsinessPipeline(
            FakeTracker(raises=True), DrowsinessMonitor(), alarm=NullAlarm()
        )
        result = pipeline.process(self.frame(), 1, 0.1, 0.03)
        self.assertIsNone(result.face)
        self.assertIs(result.state.eye_state, EyeState.UNKNOWN)

    def test_no_face_annotates_without_landmarks(self) -> None:
        tracker = FakeTracker(None)
        pipeline = DrowsinessPipeline(tracker, DrowsinessMonitor(), alarm=NullAlarm())
        frame = self.frame()
        result = pipeline.process(frame, 1, 0.1, 0.03)
        pipeline.annotate(frame, result, 0.1)
        self.assertEqual(tracker.drawn, 0)

    def test_close_releases_the_tracker(self) -> None:
        tracker = FakeTracker(None)
        DrowsinessPipeline(tracker, DrowsinessMonitor(), alarm=NullAlarm()).close()
        self.assertTrue(tracker.closed)

    def test_fps_meter_averages_over_its_window(self) -> None:
        meter = FpsMeter(window_size=2)
        self.assertEqual(meter.update(0.0), 0.0)
        meter.update(0.1)
        self.assertAlmostEqual(meter.update(0.1), 10.0, places=5)
        with self.assertRaises(ValueError):
            FpsMeter(0)

    def test_run_stats_counts_distinct_drowsiness_episodes(self) -> None:
        stats = RunStats()
        self.assertIn("No frames", stats.summary())
        awake = MonitorState(
            EyeState.OPEN, AlertLevel.AWAKE, 0.3, 0.1, 0.0, 0.0, 0, 0, False, 0.0
        )
        drowsy = MonitorState(
            EyeState.CLOSED, AlertLevel.DROWSY, 0.05, 0.1, 2.0, 0.5, 1, 0, False, 0.0
        )
        from src.pipeline import FrameResult

        for state in (awake, drowsy, drowsy, awake, drowsy):
            stats.observe(FrameResult(None, None, state, 20.0), 0.05)
        stats.elapsed_seconds = 0.25
        self.assertEqual(stats.frames, 5)
        self.assertEqual(stats.drowsy_events, 2)
        self.assertAlmostEqual(stats.alarm_seconds, 0.15, places=6)
        self.assertIn("drowsiness event", stats.summary())


class DrawingTests(unittest.TestCase):
    def state(self, level: AlertLevel = AlertLevel.AWAKE, ear: float = 0.25) -> MonitorState:
        return MonitorState(
            EyeState.OPEN, level, ear, 0.1, 0.0, 0.05, 3, 1, False, 0.0
        )

    def test_hud_and_banner_draw_for_every_level(self) -> None:
        for level in AlertLevel:
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            draw_hud(frame, self.state(level), 25.0, ear_threshold=0.15)
            draw_banner(frame, self.state(level), blink_phase=True)
            self.assertGreater(int(frame.max()), 0)

    def test_missing_metrics_render_as_placeholders(self) -> None:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        state = MonitorState(
            EyeState.UNKNOWN,
            AlertLevel.NO_FACE,
            float("nan"),
            float("nan"),
            0.0,
            0.0,
            0,
            0,
            False,
            3.0,
        )
        draw_hud(frame, state, 0.0, ear_threshold=0.15)
        self.assertGreater(int(frame.max()), 0)

    def test_a_small_frame_still_renders(self) -> None:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        draw_hud(frame, self.state(), 10.0, ear_threshold=0.15, extra_lines=["note"])
        draw_banner(frame, self.state(AlertLevel.DROWSY), blink_phase=True)
        self.assertGreater(int(frame.max()), 0)


def config_for(**overrides) -> AppConfig:
    base = dict(
        camera=0,
        input_path=None,
        output_path=None,
        model_path=Path("unused.task"),
        profile_path=Path("unused.json"),
        resolution=(640, 480),
        max_frames=0,
        display=False,
        mirror=False,
        show_landmarks=True,
        alarm_enabled=False,
        monitor=MonitorConfig(ear_threshold=0.15),
        calibrate_seconds=0.0,
        closed_ratio=0.75,
        use_profile=False,
    )
    base.update(overrides)
    return AppConfig(**base)


class FakeCapture:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = list(frames)
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        return 30.0

    def set(self, prop: int, value: float) -> bool:
        return True

    def read(self):
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def release(self) -> None:
        self.released = True


class FakeCv2:
    """Wraps the real cv2 but replaces capture and display."""

    def __init__(self, frames: list[np.ndarray]) -> None:
        self.capture = FakeCapture(frames)
        self.shown = 0

    def __getattr__(self, name: str):
        return getattr(cv2, name)

    def VideoCapture(self, source):
        return self.capture

    def imshow(self, *args) -> None:
        self.shown += 1

    def waitKey(self, delay: int) -> int:
        return -1

    def destroyAllWindows(self) -> None:
        return None


class CommandLineTests(unittest.TestCase):
    def parse(self, argv: list[str]) -> AppConfig:
        return config_from_args(build_parser().parse_args(argv))

    def test_defaults_are_sensible(self) -> None:
        config = self.parse([])
        self.assertEqual(config.camera, 0)
        self.assertTrue(config.mirror)
        self.assertTrue(config.display)
        self.assertTrue(config.alarm_enabled)
        self.assertEqual(config.calibrate_seconds, 0.0)

    def test_calibrate_takes_an_optional_duration(self) -> None:
        self.assertEqual(self.parse(["--calibrate"]).calibrate_seconds, 5.0)
        self.assertEqual(self.parse(["--calibrate", "8"]).calibrate_seconds, 8.0)

    def test_calibrating_against_a_recorded_video_is_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            self.parse(["--calibrate", "--input", "clip.mp4"])

    def test_recorded_input_defaults_to_no_mirroring(self) -> None:
        self.assertFalse(self.parse(["--input", "clip.mp4"]).mirror)
        self.assertTrue(self.parse(["--input", "clip.mp4", "--mirror"]).mirror)

    def test_size_parsing(self) -> None:
        import argparse

        self.assertEqual(parse_size("1280x720"), (1280, 720))
        for bad in ("1280", "x", "0x10", "axb"):
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_size(bad)

    def test_contradictory_durations_are_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            self.parse(["--blink-seconds", "3", "--closure-seconds", "1.5"])

    def test_invalid_values_are_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            self.parse(["--camera", "-1"])
        for bad in (
            ["--ear-threshold", "2"],
            ["--closure-seconds", "0"],
            ["--max-frames", "-1"],
            ["--perclos-threshold", "-1"],
        ):
            with self.assertRaises(SystemExit), quiet():
                build_parser().parse_args(bad)


class MediaLoopTests(unittest.TestCase):
    def frames(self, count: int) -> list[np.ndarray]:
        return [np.full((480, 640, 3), 30, dtype=np.uint8) for _ in range(count)]

    def test_the_loop_processes_every_frame(self) -> None:
        fake = FakeCv2(self.frames(5))
        with quiet():
            stats = run(
                config_for(),
                cv2_module=fake,
                tracker=FakeTracker(face_landmarks(eye_opening=0.04)),
                alarm=NullAlarm(),
            )
        self.assertEqual(stats.frames, 5)
        self.assertEqual(stats.faces, 5)
        self.assertTrue(fake.capture.released)

    def test_max_frames_stops_the_loop_early(self) -> None:
        fake = FakeCv2(self.frames(10))
        with quiet():
            stats = run(
                config_for(max_frames=3),
                cv2_module=fake,
                tracker=FakeTracker(None),
                alarm=NullAlarm(),
            )
        self.assertEqual(stats.frames, 3)

    def test_an_empty_source_is_an_error(self) -> None:
        with self.assertRaises(MediaError), quiet():
            run(
                config_for(),
                cv2_module=FakeCv2([]),
                tracker=FakeTracker(None),
                alarm=NullAlarm(),
            )

    def test_an_unopenable_source_is_reported(self) -> None:
        class ClosedCapture(FakeCapture):
            def isOpened(self) -> bool:
                return False

        fake = FakeCv2([])
        fake.capture = ClosedCapture([])
        with self.assertRaises(MediaError), quiet():
            run(config_for(), cv2_module=fake, tracker=FakeTracker(None), alarm=NullAlarm())

    def test_a_missing_input_file_is_reported(self) -> None:
        with self.assertRaises(MediaError), quiet():
            run(
                config_for(input_path=Path("no-such-clip.mp4")),
                cv2_module=FakeCv2(self.frames(1)),
                tracker=FakeTracker(None),
                alarm=NullAlarm(),
            )

    def test_an_annotated_video_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out.avi"
            with quiet():
                stats = run(
                    config_for(output_path=output),
                    cv2_module=FakeCv2(self.frames(3)),
                    tracker=FakeTracker(face_landmarks(eye_opening=0.04)),
                    alarm=NullAlarm(),
                )
            self.assertEqual(stats.frames, 3)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

    def test_an_unsupported_output_suffix_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(MediaError), quiet():
                run(
                    config_for(output_path=Path(directory) / "out.gif"),
                    cv2_module=FakeCv2(self.frames(1)),
                    tracker=FakeTracker(None),
                    alarm=NullAlarm(),
                )

    def test_a_saved_profile_overrides_the_default_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "profile.json"
            calibrate([0.40] * 30).save(profile)
            config = config_for(use_profile=True, profile_path=profile)
            with quiet() as output:
                run(
                    config,
                    cv2_module=FakeCv2(self.frames(2)),
                    tracker=FakeTracker(face_landmarks(eye_opening=0.04)),
                    alarm=NullAlarm(),
                )
            self.assertIn("calibrated", output.getvalue())

    def test_calibration_captures_samples_and_writes_a_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "profile.json"
            config = config_for(calibrate_seconds=60.0, profile_path=profile)
            # The capture ends when frames run out rather than on the clock.
            fake = FakeCv2(self.frames(40))
            with quiet():
                result = run_calibration(
                    config,
                    cv2_module=fake,
                    tracker=FakeTracker(face_landmarks(eye_opening=0.04)),
                )
            self.assertTrue(profile.is_file())
            # The fake frames are 640x480, so the 4:3 aspect scales the eye
            # width: 0.04 / (0.08 * 4 / 3) = 0.375.
            self.assertAlmostEqual(result.baseline_ear, 0.375, places=3)
            self.assertAlmostEqual(result.ear_threshold, 0.375 * 0.75, places=3)

    def test_calibration_without_a_face_fails_with_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = config_for(
                calibrate_seconds=60.0, profile_path=Path(directory) / "p.json"
            )
            with self.assertRaises(ConfigurationError) as context, quiet():
                run_calibration(
                    config, cv2_module=FakeCv2(self.frames(30)), tracker=FakeTracker(None)
                )
            self.assertIn("visible face", str(context.exception))


if __name__ == "__main__":
    unittest.main()
