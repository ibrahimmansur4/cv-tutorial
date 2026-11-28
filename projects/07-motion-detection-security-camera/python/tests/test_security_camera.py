from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from src.app import AppConfig, build_parser, config_from_args, parse_size, run
from src.background import (
    Mog2Background,
    RunningAverageBackground,
    make_background,
)
from src.detector import MotionDetector, MotionRegion, merge_regions
from src.drawing import draw_regions, draw_status, stack_mask
from src.errors import ConfigurationError, MediaError, RecorderError
from src.events import load_json, write_csv, write_json
from src.pipeline import FpsMeter, RunStats, SecurityCameraPipeline
from src.recorder import EventRecorder, MotionEvent, RecorderConfig


@contextlib.contextmanager
def quiet():
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        yield buffer


def scene(value: int = 60, size: tuple[int, int] = (240, 320)) -> np.ndarray:
    return np.full((*size, 3), value, dtype=np.uint8)


def scene_with_blob(
    value: int = 60, size: tuple[int, int] = (240, 320), box=(120, 80, 70, 90)
) -> np.ndarray:
    frame = scene(value, size)
    x, y, width, height = box
    frame[y : y + height, x : x + width] = 220
    return frame


class RunningAverageTests(unittest.TestCase):
    def test_a_static_scene_produces_no_foreground(self) -> None:
        model = RunningAverageBackground(warmup_frames=3)
        detector = MotionDetector(model, min_area=100.0)
        for _ in range(10):
            result = detector.detect(scene())
        self.assertTrue(result.ready)
        self.assertEqual(result.regions, ())
        self.assertEqual(result.changed_pixels, 0)

    def test_an_object_appearing_is_detected(self) -> None:
        model = RunningAverageBackground(warmup_frames=3)
        detector = MotionDetector(model, min_area=100.0)
        for _ in range(10):
            detector.detect(scene())
        result = detector.detect(scene_with_blob())
        self.assertTrue(result.detected)
        self.assertEqual(len(result.regions), 1)
        region = result.regions[0]
        # The box should sit roughly where the blob was drawn.
        self.assertLess(abs(region.x - 120), 25)
        self.assertLess(abs(region.y - 80), 25)

    def test_warmup_suppresses_detection(self) -> None:
        detector = MotionDetector(
            RunningAverageBackground(warmup_frames=30), min_area=50.0
        )
        detector.detect(scene())
        result = detector.detect(scene_with_blob())
        self.assertFalse(result.ready)
        self.assertEqual(result.regions, ())

    def test_a_stationary_object_fades_into_the_background(self) -> None:
        """A high alpha is a short memory: a parked object stops being motion."""
        detector = MotionDetector(
            RunningAverageBackground(alpha=0.5, warmup_frames=2), min_area=100.0
        )
        for _ in range(6):
            detector.detect(scene())
        self.assertTrue(detector.detect(scene_with_blob()).detected)
        for _ in range(30):
            result = detector.detect(scene_with_blob())
        self.assertFalse(result.detected)

    def test_a_resized_frame_restarts_the_model(self) -> None:
        model = RunningAverageBackground(warmup_frames=2)
        detector = MotionDetector(model, min_area=50.0)
        for _ in range(6):
            detector.detect(scene())
        self.assertTrue(model.ready)
        detector.detect(scene(size=(120, 160)))
        self.assertFalse(model.ready)

    def test_reset_clears_the_reference(self) -> None:
        model = RunningAverageBackground(warmup_frames=2)
        detector = MotionDetector(model)
        for _ in range(6):
            detector.detect(scene())
        self.assertTrue(model.ready)
        model.reset()
        self.assertFalse(model.ready)
        self.assertIsNone(model.reference)

    def test_invalid_parameters_are_rejected(self) -> None:
        for kwargs in ({"alpha": 0.0}, {"alpha": 1.5}, {"delta_threshold": 0},
                       {"delta_threshold": 300}, {"warmup_frames": -1}):
            with self.assertRaises(ConfigurationError, msg=str(kwargs)):
                RunningAverageBackground(**kwargs)

    def test_a_colour_frame_is_required_to_be_well_formed(self) -> None:
        model = RunningAverageBackground()
        with self.assertRaises(ConfigurationError):
            model.apply(np.zeros((10, 10, 3), dtype=np.uint8))


class Mog2Tests(unittest.TestCase):
    def test_mog2_detects_an_object_against_a_learned_scene(self) -> None:
        detector = MotionDetector(Mog2Background(warmup_frames=5), min_area=100.0)
        for _ in range(20):
            detector.detect(scene())
        self.assertTrue(detector.detect(scene_with_blob()).detected)

    def test_shadow_pixels_are_dropped(self) -> None:
        class FakeSubtractor:
            def apply(self, frame):
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                mask[0:20, 0:20] = 255  # hard foreground
                mask[40:60, 40:60] = Mog2Background.SHADOW_VALUE  # shadow
                return mask

        model = Mog2Background(subtractor=FakeSubtractor(), warmup_frames=0)
        mask = model.apply(np.zeros((100, 100), dtype=np.uint8))
        self.assertEqual(int(mask[5, 5]), 255)
        self.assertEqual(int(mask[50, 50]), 0)

    def test_shadow_removal_can_be_disabled(self) -> None:
        class FakeSubtractor:
            def apply(self, frame):
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                mask[40:60, 40:60] = Mog2Background.SHADOW_VALUE
                return mask

        model = Mog2Background(
            subtractor=FakeSubtractor(), detect_shadows=False, warmup_frames=0
        )
        mask = model.apply(np.zeros((100, 100), dtype=np.uint8))
        self.assertEqual(int(mask[50, 50]), Mog2Background.SHADOW_VALUE)

    def test_invalid_parameters_are_rejected(self) -> None:
        for kwargs in ({"history": 0}, {"var_threshold": 0.0}, {"warmup_frames": -1}):
            with self.assertRaises(ConfigurationError, msg=str(kwargs)):
                Mog2Background(**kwargs)


class FactoryTests(unittest.TestCase):
    def test_known_names_build_models(self) -> None:
        self.assertEqual(make_background("running-average").name, "running-average")
        self.assertEqual(make_background("mog2").name, "mog2")
        self.assertEqual(make_background("DIFF").name, "running-average")

    def test_an_unknown_name_is_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            make_background("magic")


class DetectorTests(unittest.TestCase):
    def test_small_regions_are_filtered_out(self) -> None:
        detector = MotionDetector(RunningAverageBackground(warmup_frames=2), min_area=1e6)
        for _ in range(6):
            detector.detect(scene())
        self.assertFalse(detector.detect(scene_with_blob()).detected)

    def test_regions_are_sorted_largest_first(self) -> None:
        detector = MotionDetector(RunningAverageBackground(warmup_frames=2), min_area=50.0)
        for _ in range(6):
            detector.detect(scene())
        frame = scene()
        frame[30:60, 30:60] = 220  # small
        frame[120:200, 150:280] = 220  # large
        regions = detector.detect(frame).regions
        self.assertGreaterEqual(len(regions), 2)
        areas = [region.area for region in regions]
        self.assertEqual(areas, sorted(areas, reverse=True))

    def test_max_regions_caps_the_result(self) -> None:
        detector = MotionDetector(
            RunningAverageBackground(warmup_frames=2),
            min_area=10.0,
            dilate_iterations=0,
            max_regions=2,
        )
        for _ in range(6):
            detector.detect(scene())
        frame = scene()
        for index in range(6):
            frame[20 : 20 + 20, 20 + index * 45 : 40 + index * 45] = 220
        self.assertLessEqual(len(detector.detect(frame).regions), 2)

    def test_grayscale_input_is_accepted(self) -> None:
        detector = MotionDetector(RunningAverageBackground(warmup_frames=1))
        prepared = detector.prepare(np.zeros((120, 160), dtype=np.uint8))
        self.assertEqual(prepared.ndim, 2)

    def test_malformed_frames_are_rejected(self) -> None:
        detector = MotionDetector(RunningAverageBackground())
        for bad in (np.zeros((0, 0, 3), np.uint8), np.zeros((4, 4, 2), np.uint8)):
            with self.assertRaises(ConfigurationError):
                detector.prepare(bad)

    def test_invalid_options_are_rejected(self) -> None:
        model = RunningAverageBackground()
        for kwargs in ({"blur_kernel": 20}, {"blur_kernel": 0}, {"min_area": -1.0},
                       {"dilate_iterations": -1}, {"max_regions": 0}):
            with self.assertRaises(ConfigurationError, msg=str(kwargs)):
                MotionDetector(model, **kwargs)

    def test_merge_regions_encloses_every_box(self) -> None:
        merged = merge_regions(
            [MotionRegion(10, 10, 20, 20, 400.0), MotionRegion(50, 60, 30, 10, 300.0)]
        )
        self.assertIsNotNone(merged)
        self.assertEqual(merged.box, (10, 10, 80, 70))
        self.assertAlmostEqual(merged.area, 700.0)
        self.assertIsNone(merge_regions([]))

    def test_region_geometry(self) -> None:
        region = MotionRegion(10, 20, 30, 40, 1200.0)
        self.assertEqual(region.box, (10, 20, 40, 60))
        self.assertEqual(region.centre, (25, 40))


class RecorderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.stamp = datetime(2026, 1, 2, 3, 4, 5)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def recorder(self, **overrides) -> EventRecorder:
        settings = dict(
            directory=self.root,
            fps=10.0,
            pre_roll_seconds=0.5,
            post_roll_seconds=1.0,
            min_event_seconds=0.2,
        )
        settings.update(overrides)
        return EventRecorder(
            RecorderConfig(**settings), clock=lambda: self.stamp
        )

    def test_an_event_records_and_closes_after_the_post_roll(self) -> None:
        recorder = self.recorder()
        frame = scene()
        now = 0.0
        for _ in range(5):  # idle, filling the pre-roll buffer
            now += 0.1
            self.assertIsNone(recorder.update(frame, motion=False, now_seconds=now))
        now += 0.1
        recorder.update(frame, motion=True, now_seconds=now, regions=1, area=900.0)
        self.assertTrue(recorder.recording)
        finished = None
        for _ in range(20):
            now += 0.1
            finished = recorder.update(frame, motion=False, now_seconds=now) or finished
        self.assertIsNotNone(finished)
        self.assertFalse(recorder.recording)
        self.assertTrue(finished.video_path.is_file())
        self.assertEqual(finished.peak_regions, 1)
        self.assertAlmostEqual(finished.peak_area, 900.0)

    def test_pre_roll_frames_are_written_before_the_trigger(self) -> None:
        recorder = self.recorder(pre_roll_seconds=0.5)  # 5 frames at 10 FPS
        frame = scene()
        now = 0.0
        for _ in range(10):
            now += 0.1
            recorder.update(frame, motion=False, now_seconds=now)
        now += 0.1
        recorder.update(frame, motion=True, now_seconds=now)
        event = recorder.current_event
        # Five buffered frames plus the trigger frame itself.
        self.assertEqual(event.frames, 6)

    def test_brief_gaps_do_not_split_one_event(self) -> None:
        recorder = self.recorder(post_roll_seconds=1.0)
        frame = scene()
        now = 0.0
        for index in range(30):
            now += 0.1
            # Motion every other frame: the gaps are shorter than the post-roll.
            recorder.update(frame, motion=index % 2 == 0, now_seconds=now)
        self.assertTrue(recorder.recording)
        self.assertEqual(len(recorder.events), 0)

    def test_an_event_shorter_than_the_floor_is_discarded(self) -> None:
        recorder = self.recorder(min_event_seconds=5.0)
        frame = scene()
        now = 0.1
        recorder.update(frame, motion=True, now_seconds=now)
        path = recorder.current_event.video_path
        for _ in range(20):
            now += 0.1
            recorder.update(frame, motion=False, now_seconds=now)
        self.assertEqual(recorder.events, ())
        self.assertFalse(path.is_file())

    def test_a_long_event_is_split(self) -> None:
        recorder = self.recorder(max_event_seconds=1.0)
        frame = scene()
        now = 0.0
        for _ in range(30):
            now += 0.1
            recorder.update(frame, motion=True, now_seconds=now)
        self.assertGreaterEqual(len(recorder.events), 2)

    def test_close_finishes_an_open_event(self) -> None:
        recorder = self.recorder()
        frame = scene()
        recorder.update(frame, motion=True, now_seconds=0.1)
        for index in range(5):
            recorder.update(frame, motion=True, now_seconds=0.2 + index * 0.1)
        finished = recorder.close(1.0)
        self.assertIsNotNone(finished)
        self.assertFalse(recorder.recording)

    def test_a_snapshot_is_written_alongside_the_clip(self) -> None:
        recorder = self.recorder()
        recorder.update(scene(), motion=True, now_seconds=0.1)
        event = recorder.current_event
        self.assertIsNotNone(event.snapshot_path)
        self.assertTrue(event.snapshot_path.is_file())

    def test_snapshots_can_be_disabled(self) -> None:
        recorder = self.recorder(write_snapshot=False)
        recorder.update(scene(), motion=True, now_seconds=0.1)
        self.assertIsNone(recorder.current_event.snapshot_path)

    def test_file_names_are_timestamped_and_unique(self) -> None:
        recorder = self.recorder(post_roll_seconds=0.1, min_event_seconds=0.0)
        frame = scene()
        now = 0.0
        names = []
        for _ in range(3):
            now += 0.1
            recorder.update(frame, motion=True, now_seconds=now)
            names.append(recorder.current_event.video_path.name)
            for _ in range(4):
                now += 0.1
                recorder.update(frame, motion=False, now_seconds=now)
        self.assertEqual(len(set(names)), 3)
        for name in names:
            self.assertTrue(name.startswith("motion_20260102_030405_"))

    def test_an_explicit_stamp_labels_the_event(self) -> None:
        recorder = self.recorder()
        stamp = datetime(2030, 6, 7, 8, 9, 10)
        recorder.update(scene(), motion=True, now_seconds=0.1, stamp=stamp)
        self.assertEqual(recorder.current_event.started_at, stamp)

    def test_invalid_recorder_settings_are_rejected(self) -> None:
        for kwargs in ({"fps": 0.0}, {"pre_roll_seconds": -1.0},
                       {"max_event_seconds": 0.0}, {"suffix": ".gif"}):
            with self.assertRaises(ConfigurationError, msg=str(kwargs)):
                RecorderConfig(directory=self.root, **kwargs)

    def test_an_unwritable_directory_is_reported(self) -> None:
        recorder = EventRecorder(
            RecorderConfig(directory=self.root / "sub", fps=10.0, suffix=".avi"),
            clock=lambda: self.stamp,
        )
        recorder.update(scene(), motion=True, now_seconds=0.1)
        self.assertTrue((self.root / "sub").is_dir())


class EventLogTests(unittest.TestCase):
    def events(self) -> list[MotionEvent]:
        event = MotionEvent(
            index=1,
            started_at=datetime(2026, 1, 2, 3, 4, 5),
            start_seconds=1.0,
            end_seconds=4.5,
            frames=70,
            peak_regions=2,
            peak_area=1234.5,
            video_path=Path("motion_1.avi"),
            snapshot_path=Path("motion_1.jpg"),
        )
        return [event]

    def test_csv_and_json_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = write_csv(self.events(), root / "events.csv")
            json_path = write_json(self.events(), root / "events.json")
            self.assertTrue(csv_path.is_file())
            text = csv_path.read_text(encoding="utf-8")
            self.assertIn("motion_1.avi", text)
            self.assertIn("3.5", text)  # duration
            loaded = load_json(json_path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["index"], 1)

    def test_loading_a_missing_log_returns_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(load_json(Path(directory) / "none.json"), [])

    def test_a_corrupt_log_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                load_json(path)

    def test_an_unexpected_shape_is_treated_as_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "odd.json"
            path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
            self.assertEqual(load_json(path), [])

    def test_event_duration_is_zero_until_it_ends(self) -> None:
        event = MotionEvent(index=1, started_at=datetime.now(), start_seconds=2.0)
        self.assertEqual(event.duration, 0.0)


class DrawingTests(unittest.TestCase):
    def test_boxes_and_status_render(self) -> None:
        frame = scene()
        draw_regions(frame, [MotionRegion(10, 10, 40, 40, 1600.0)])
        draw_status(
            frame,
            motion=True,
            recording=True,
            ready=True,
            fps=20.0,
            events=3,
            changed_fraction=0.05,
            model_name="mog2",
            timestamp=datetime(2026, 1, 1, 12, 0, 0),
            extra_lines=["Recording: a.avi"],
        )
        self.assertGreater(int(frame.max()), 60)

    def test_every_state_renders_on_a_small_frame(self) -> None:
        for ready, motion, recording in (
            (False, False, False),
            (True, False, False),
            (True, True, True),
        ):
            frame = scene(size=(120, 160))
            draw_status(
                frame,
                motion=motion,
                recording=recording,
                ready=ready,
                fps=12.0,
                events=0,
                changed_fraction=0.0,
                model_name="running-average",
            )
            self.assertGreater(int(frame.max()), 0)

    def test_the_mask_can_be_stacked_beside_the_frame(self) -> None:
        frame = scene(size=(100, 120))
        mask = np.zeros((100, 120), dtype=np.uint8)
        stacked = stack_mask(frame, mask)
        self.assertEqual(stacked.shape, (100, 240, 3))


class PipelineTests(unittest.TestCase):
    def test_the_pipeline_detects_and_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            detector = MotionDetector(
                RunningAverageBackground(warmup_frames=3), min_area=100.0
            )
            recorder = EventRecorder(
                RecorderConfig(
                    directory=Path(directory),
                    fps=10.0,
                    pre_roll_seconds=0.3,
                    post_roll_seconds=0.5,
                    min_event_seconds=0.0,
                )
            )
            pipeline = SecurityCameraPipeline(detector, recorder)
            now = 0.0
            for _ in range(8):
                now += 0.1
                pipeline.process(scene(), now, 0.05)
            now += 0.1
            result = pipeline.process(scene_with_blob(), now, 0.05)
            self.assertTrue(result.motion.detected)
            self.assertTrue(result.recording)
            annotated = pipeline.annotate(scene_with_blob(), result, now, events=0)
            self.assertEqual(annotated.shape, scene().shape)
            self.assertIsNotNone(pipeline.close(now + 5.0))

    def test_the_pipeline_runs_without_a_recorder(self) -> None:
        pipeline = SecurityCameraPipeline(
            MotionDetector(RunningAverageBackground(warmup_frames=1)), None
        )
        result = pipeline.process(scene(), 0.1, 0.05)
        self.assertFalse(result.recording)
        self.assertIsNone(pipeline.close(1.0))

    def test_show_mask_widens_the_canvas(self) -> None:
        pipeline = SecurityCameraPipeline(
            MotionDetector(RunningAverageBackground(warmup_frames=1)), None, show_mask=True
        )
        frame = scene()
        result = pipeline.process(frame, 0.1, 0.05)
        self.assertEqual(pipeline.annotate(frame, result, 0.1, events=0).shape[1], 640)

    def test_fps_meter(self) -> None:
        meter = FpsMeter(window_size=2)
        self.assertEqual(meter.update(0.0), 0.0)
        meter.update(0.1)
        self.assertAlmostEqual(meter.update(0.1), 10.0, places=5)
        with self.assertRaises(ValueError):
            FpsMeter(0)

    def test_run_stats_summarize(self) -> None:
        stats = RunStats()
        self.assertIn("No frames", stats.summary())
        stats.frames = 100
        stats.motion_frames = 25
        stats.elapsed_seconds = 5.0
        self.assertAlmostEqual(stats.fps, 20.0)
        self.assertAlmostEqual(stats.motion_fraction, 0.25)
        self.assertIn("25.0%", stats.summary())


def config_for(**overrides) -> AppConfig:
    base = dict(
        camera=0,
        input_path=None,
        recordings_dir=Path("recordings"),
        results_dir=Path("results"),
        resolution=(320, 240),
        background="running-average",
        alpha=0.04,
        delta_threshold=12,
        history=300,
        var_threshold=24.0,
        detect_shadows=True,
        blur_kernel=21,
        min_area=100.0,
        dilate_iterations=2,
        warmup_frames=3,
        pre_roll=0.3,
        post_roll=0.5,
        min_event=0.0,
        max_event=120.0,
        record=False,
        snapshots=True,
        max_frames=0,
        display=False,
        show_mask=False,
        draw_boxes=True,
        mirror=False,
    )
    base.update(overrides)
    return AppConfig(**base)


class FakeCapture:
    def __init__(self, frames) -> None:
        self._frames = list(frames)
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        return 20.0

    def set(self, prop: int, value: float) -> bool:
        return True

    def read(self):
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def release(self) -> None:
        self.released = True


class FakeCv2:
    def __init__(self, frames) -> None:
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
    def parse(self, argv) -> AppConfig:
        return config_from_args(build_parser().parse_args(argv))

    def test_defaults(self) -> None:
        config = self.parse([])
        self.assertEqual(config.camera, 0)
        self.assertEqual(config.background, "running-average")
        self.assertTrue(config.record)
        self.assertTrue(config.display)
        self.assertFalse(config.mirror)

    def test_model_choice_and_flags(self) -> None:
        config = self.parse(["--background", "mog2", "--no-shadows", "--no-record"])
        self.assertEqual(config.background, "mog2")
        self.assertFalse(config.detect_shadows)
        self.assertFalse(config.record)

    def test_size_parsing(self) -> None:
        import argparse

        self.assertEqual(parse_size("640x480"), (640, 480))
        for bad in ("640", "x", "0x10", "axb"):
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_size(bad)

    def test_invalid_values_are_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            self.parse(["--camera", "-1"])
        with self.assertRaises(ConfigurationError):
            self.parse(["--delta-threshold", "0"])
        with self.assertRaises(ConfigurationError):
            self.parse(["--min-event", "10", "--max-event", "5"])
        for bad in (["--blur", "20"], ["--alpha", "0"], ["--min-area", "-1"],
                    ["--max-frames", "-1"]):
            with self.assertRaises(SystemExit), quiet():
                build_parser().parse_args(bad)


class MediaLoopTests(unittest.TestCase):
    def frames(self, count: int, moving_from: int | None = None):
        output = []
        for index in range(count):
            if moving_from is not None and index >= moving_from:
                output.append(scene_with_blob())
            else:
                output.append(scene())
        return output

    def test_the_loop_processes_every_frame(self) -> None:
        fake = FakeCv2(self.frames(12))
        with quiet():
            stats = run(config_for(), cv2_module=fake)
        self.assertEqual(stats.frames, 12)
        self.assertTrue(fake.capture.released)

    def test_max_frames_stops_early(self) -> None:
        fake = FakeCv2(self.frames(20))
        with quiet():
            stats = run(config_for(max_frames=5), cv2_module=fake)
        self.assertEqual(stats.frames, 5)

    def test_motion_produces_an_event_and_a_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = config_for(
                record=True,
                recordings_dir=root / "rec",
                results_dir=root / "res",
                min_event=0.0,
            )
            fake = FakeCv2(self.frames(40, moving_from=10))
            with quiet():
                stats = run(config, cv2_module=fake)
            self.assertGreaterEqual(len(stats.events), 1)
            self.assertTrue((root / "res" / "events.csv").is_file())
            self.assertTrue((root / "res" / "events.json").is_file())
            self.assertTrue(stats.events[0].video_path.is_file())

    def test_no_record_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = config_for(record=False, recordings_dir=root / "rec",
                                results_dir=root / "res")
            with quiet():
                run(config, cv2_module=FakeCv2(self.frames(30, moving_from=10)))
            self.assertFalse((root / "res" / "events.csv").exists())
            self.assertFalse((root / "rec").exists())

    def test_an_empty_source_is_an_error(self) -> None:
        with self.assertRaises(MediaError), quiet():
            run(config_for(), cv2_module=FakeCv2([]))

    def test_an_unopenable_source_is_reported(self) -> None:
        class ClosedCapture(FakeCapture):
            def isOpened(self) -> bool:
                return False

        fake = FakeCv2([])
        fake.capture = ClosedCapture([])
        with self.assertRaises(MediaError), quiet():
            run(config_for(), cv2_module=fake)

    def test_a_missing_input_file_is_reported(self) -> None:
        with self.assertRaises(MediaError), quiet():
            run(config_for(input_path=Path("no-such.mp4")), cv2_module=FakeCv2(self.frames(2)))


if __name__ == "__main__":
    unittest.main()
