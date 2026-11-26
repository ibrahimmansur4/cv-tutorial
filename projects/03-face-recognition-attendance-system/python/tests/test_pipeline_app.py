from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PYTHON_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from src import app
from src.attendance_log import AttendanceLogger
from src.detection import FaceBox, FaceDetector
from src.embeddings import FaceIndex, HOGFaceEmbedder, KnownFaceStore, Match
from src.errors import ConfigurationError, MediaError
from src.pipeline import FaceObservation, RecognitionPipeline


class _Detector:
    def __init__(self, boxes: list[FaceBox]) -> None:
        self.boxes = boxes
        self.calls = 0

    def detect(self, frame: np.ndarray) -> list[FaceBox]:
        self.calls += 1
        return list(self.boxes)


class _MeanEmbedder:
    def embed(self, face: np.ndarray) -> np.ndarray:
        if float(np.mean(face)) < 128.0:
            return np.asarray([1.0, 0.0], dtype=np.float32)
        return np.asarray([0.0, 1.0], dtype=np.float32)


class RecognitionPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_known_and_unknown_faces_and_daily_attendance(self) -> None:
        frame = np.zeros((20, 40, 3), dtype=np.uint8)
        frame[:, 20:] = 255
        detector = _Detector([FaceBox(0, 0, 20, 20), FaceBox(20, 0, 20, 20)])
        index = FaceIndex(["Alice"], np.asarray([[1.0, 0.0]], dtype=np.float32))
        logger = AttendanceLogger(Path(self.temporary.name) / "attendance.csv")
        pipeline = RecognitionPipeline(
            detector,  # type: ignore[arg-type]
            _MeanEmbedder(),  # type: ignore[arg-type]
            index,
            logger,
            threshold=0.2,
        )
        timestamp = datetime(2026, 10, 5, 9, 30)

        first = pipeline.process(frame, timestamp=timestamp)
        self.assertEqual([item.name for item in first], ["Alice", "Unknown"])
        self.assertEqual([item.match.recognized for item in first], [True, False])
        self.assertEqual([item.attendance_recorded for item in first], [True, False])
        self.assertEqual(len(logger.records()), 1)

        second = pipeline.process(frame, timestamp=timestamp)
        self.assertEqual([item.attendance_recorded for item in second], [False, False])
        self.assertEqual(len(logger.records()), 1)

    def test_annotation_copies_source_and_draws_known_and_unknown_labels(self) -> None:
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        source = frame.copy()
        pipeline = RecognitionPipeline(
            _Detector([]),  # type: ignore[arg-type]
            _MeanEmbedder(),  # type: ignore[arg-type]
            FaceIndex([], np.empty((0, 2), dtype=np.float32)),
        )
        observations = [
            FaceObservation(FaceBox(10, 30, 30, 30), Match("Alice", 0.1, 0.9), True),
            FaceObservation(FaceBox(70, 30, 25, 25), Match(None, 0.7, 0.3), False),
        ]

        annotated = pipeline.annotate(frame, observations)
        np.testing.assert_array_equal(frame, source)
        self.assertFalse(np.array_equal(annotated, source))

    def test_pipeline_rejects_invalid_threshold(self) -> None:
        for threshold in (-0.01, 2.01):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    RecognitionPipeline(
                        _Detector([]),  # type: ignore[arg-type]
                        _MeanEmbedder(),  # type: ignore[arg-type]
                        FaceIndex([], np.empty((0, 2), dtype=np.float32)),
                        threshold=threshold,
                    )


class _FakePipeline:
    def __init__(self, observations: list[FaceObservation] | None = None) -> None:
        self.observations = observations or []
        self.processed: list[tuple[np.ndarray, datetime | None]] = []
        self.annotated = 0

    def process(self, frame: np.ndarray, *, timestamp: datetime | None = None) -> list[FaceObservation]:
        self.processed.append((frame.copy(), timestamp))
        return list(self.observations)

    def annotate(self, frame: np.ndarray, observations: list[FaceObservation]) -> np.ndarray:
        self.annotated += 1
        return np.clip(frame.astype(np.int16) + 1, 0, 255).astype(np.uint8)


class _FakeCapture:
    def __init__(self, frames: list[np.ndarray], *, opened: bool = True) -> None:
        self.frames = [frame.copy() for frame in frames]
        self.opened = opened
        self.released = False
        self.set_calls: list[tuple[int, float]] = []

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def get(self, prop: int) -> float:
        return 18.0

    def set(self, prop: int, value: float) -> bool:
        self.set_calls.append((prop, value))
        return True

    def release(self) -> None:
        self.released = True


class _FakeWriter:
    def __init__(self, *, opened: bool = True) -> None:
        self.opened = opened
        self.frames: list[np.ndarray] = []
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        self.released = True


class _StreamCV:
    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4

    def __init__(self, capture: _FakeCapture, writer: _FakeWriter | None = None) -> None:
        self.capture = capture
        self.writer = writer or _FakeWriter()
        self.capture_targets: list[int | str] = []
        self.writer_args: tuple[object, ...] | None = None

    def VideoCapture(self, target: int | str) -> _FakeCapture:
        self.capture_targets.append(target)
        return self.capture

    @staticmethod
    def VideoWriter_fourcc(*codec: str) -> int:
        return 1234

    def VideoWriter(self, *args: object) -> _FakeWriter:
        self.writer_args = args
        return self.writer


class AppProcessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.when = datetime(2026, 10, 5, 12, 0, 0)
        self.observation = FaceObservation(
            FaceBox(1, 1, 4, 4), Match("Alice", 0.1, 0.9), True
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_image_processing_is_headless_and_writes_annotation(self) -> None:
        source = self.root / "input.png"
        output = self.root / "nested" / "output.png"
        self.assertTrue(cv2.imwrite(str(source), np.full((12, 16, 3), 10, dtype=np.uint8)))
        pipeline = _FakePipeline([self.observation])

        stats = app._process_image(
            source,
            output,
            pipeline,  # type: ignore[arg-type]
            False,
            cv2,
            lambda: self.when,
        )

        self.assertEqual((stats.frames, stats.faces, stats.recognized, stats.attendance_rows), (1, 1, 1, 1))
        self.assertEqual(pipeline.processed[0][1], self.when)
        written = cv2.imread(str(output))
        self.assertIsNotNone(written)
        self.assertEqual(int(written[0, 0, 0]), 11)

    def test_image_rejects_path_collision_and_bad_output_extension(self) -> None:
        source = self.root / "input.png"
        cv2.imwrite(str(source), np.zeros((8, 8, 3), dtype=np.uint8))
        pipeline = _FakePipeline()
        with self.assertRaisesRegex(MediaError, "different files"):
            app._process_image(source, source, pipeline, False, cv2, lambda: self.when)  # type: ignore[arg-type]
        with self.assertRaisesRegex(MediaError, "supported image extension"):
            app._process_image(source, self.root / "out.txt", pipeline, False, cv2, lambda: self.when)  # type: ignore[arg-type]

    def test_stream_processes_video_writes_each_frame_and_releases_resources(self) -> None:
        source = self.root / "input.avi"
        source.write_bytes(b"fixture")
        frames = [np.full((10, 20, 3), value, dtype=np.uint8) for value in (1, 2, 3)]
        capture = _FakeCapture(frames)
        writer = _FakeWriter()
        fake_cv = _StreamCV(capture, writer)
        pipeline = _FakePipeline([self.observation])

        stats = app._process_stream(
            source,
            self.root / "output.avi",
            pipeline,  # type: ignore[arg-type]
            False,
            None,
            None,
            fake_cv,
            lambda: self.when,
        )

        self.assertEqual(stats.frames, 3)
        self.assertEqual(stats.recognized, 3)
        self.assertEqual(len(writer.frames), 3)
        self.assertEqual(fake_cv.writer_args[-1], (20, 10))  # type: ignore[index]
        self.assertTrue(capture.released)
        self.assertTrue(writer.released)
        self.assertEqual(fake_cv.capture_targets, [str(source.resolve())])

    def test_camera_applies_resolution_honors_frame_limit_and_releases(self) -> None:
        capture = _FakeCapture([np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(3)])
        fake_cv = _StreamCV(capture)
        pipeline = _FakePipeline()

        stats = app._process_stream(
            2,
            None,
            pipeline,  # type: ignore[arg-type]
            False,
            1,
            (640, 480),
            fake_cv,
            lambda: self.when,
        )

        self.assertEqual(stats.frames, 1)
        self.assertEqual(fake_cv.capture_targets, [2])
        self.assertEqual(capture.set_calls, [(fake_cv.CAP_PROP_FRAME_WIDTH, 640), (fake_cv.CAP_PROP_FRAME_HEIGHT, 480)])
        self.assertTrue(capture.released)

    def test_unopenable_or_empty_capture_reports_error_and_releases(self) -> None:
        unopened = _FakeCapture([], opened=False)
        with self.assertRaisesRegex(MediaError, "could not open camera"):
            app._open_capture(0, _StreamCV(unopened))
        self.assertTrue(unopened.released)

        empty = _FakeCapture([])
        with self.assertRaisesRegex(MediaError, "no frames"):
            app._process_stream(
                0,
                None,
                _FakePipeline(),  # type: ignore[arg-type]
                False,
                1,
                None,
                _StreamCV(empty),
                lambda: self.when,
            )
        self.assertTrue(empty.released)

    def test_paths_collide_for_same_path_and_hard_links(self) -> None:
        first = self.root / "first.dat"
        second = self.root / "second.dat"
        first.write_bytes(b"same inode")
        self.assertTrue(app._paths_collide(first, first))
        try:
            os.link(first, second)
        except OSError as exc:
            self.skipTest(f"hard links unavailable: {exc}")
        self.assertTrue(app._paths_collide(first, second))


class _EnrollmentCV(_StreamCV):
    def __init__(self, capture: _FakeCapture, *, write_ok: bool = True) -> None:
        super().__init__(capture)
        self.write_ok = write_ok
        self.saved: list[tuple[Path, tuple[int, ...]]] = []

    def imwrite(self, path: str, image: np.ndarray) -> bool:
        destination = Path(path)
        self.saved.append((destination, image.shape))
        if not self.write_ok:
            return False
        destination.write_bytes(b"sample")
        return True


class _ScriptedDetector:
    def __init__(self, results: list[list[FaceBox]]) -> None:
        self.results = list(results)
        self.calls = 0

    def detect(self, frame: np.ndarray) -> list[FaceBox]:
        result = self.results[self.calls] if self.calls < len(self.results) else []
        self.calls += 1
        return result


class _Store:
    def __init__(self) -> None:
        self.force_values: list[bool] = []

    def load(self, *, force_rebuild: bool = False) -> FaceIndex:
        self.force_values.append(force_rebuild)
        return FaceIndex([], np.empty((0, 2), dtype=np.float32))


class EnrollmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "known"
        self.frame = np.zeros((30, 30, 3), dtype=np.uint8)
        self.face = FaceBox(8, 8, 10, 10)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_enrollment_captures_only_exactly_one_face_and_rebuilds_cache(self) -> None:
        person = self.root / "Alice Smith"
        person.mkdir(parents=True)
        (person / "sample_003.png").write_bytes(b"existing")
        capture = _FakeCapture([self.frame] * 4)
        fake_cv = _EnrollmentCV(capture)
        detector = _ScriptedDetector([[], [self.face, self.face], [self.face], [self.face]])
        store = _Store()

        with contextlib.redirect_stdout(io.StringIO()):
            captured = app._enroll(
                "  Alice   Smith ",
                0,
                self.root,
                2,
                1,
                detector,  # type: ignore[arg-type]
                store,  # type: ignore[arg-type]
                False,
                None,
                4,
                fake_cv,
            )

        self.assertEqual(captured, 2)
        self.assertEqual(detector.calls, 4)
        self.assertEqual([item[0].name for item in fake_cv.saved], ["sample_004.png", "sample_005.png"])
        self.assertTrue(all(shape == (14, 14, 3) for _, shape in fake_cv.saved))
        self.assertEqual(store.force_values, [True])
        self.assertTrue(capture.released)

    def test_incomplete_enrollment_releases_capture_and_does_not_rebuild(self) -> None:
        capture = _FakeCapture([self.frame] * 2)
        fake_cv = _EnrollmentCV(capture)
        store = _Store()
        with self.assertRaisesRegex(MediaError, "0/2 samples"):
            app._enroll(
                "Alice",
                0,
                self.root,
                2,
                1,
                _ScriptedDetector([[], [self.face, self.face]]),  # type: ignore[arg-type]
                store,  # type: ignore[arg-type]
                False,
                None,
                2,
                fake_cv,
            )
        self.assertTrue(capture.released)
        self.assertEqual(store.force_values, [])

    def test_failed_sample_write_releases_capture(self) -> None:
        capture = _FakeCapture([self.frame])
        fake_cv = _EnrollmentCV(capture, write_ok=False)
        with self.assertRaisesRegex(MediaError, "could not save"):
            app._enroll(
                "Alice",
                0,
                self.root,
                2,
                1,
                _ScriptedDetector([[self.face]]),  # type: ignore[arg-type]
                _Store(),  # type: ignore[arg-type]
                False,
                None,
                1,
                fake_cv,
            )
        self.assertTrue(capture.released)

    def test_enrollment_name_validation(self) -> None:
        self.assertEqual(app._validated_enrollment_name("  Alice   Smith  "), "Alice Smith")
        invalid = ["", ".Alice", "../Alice", "Alice/Other", "Alice.", "CON", "com1.txt", "x" * 65]
        for name in invalid:
            with self.subTest(name=name):
                with self.assertRaises(ConfigurationError):
                    app._validated_enrollment_name(name)


class CliTests(unittest.TestCase):
    def test_resolution_and_valid_arguments(self) -> None:
        self.assertEqual(app.parse_resolution(" 640X480 "), (640, 480))
        args = app.parse_args(
            [
                "--camera",
                "2",
                "--no-display",
                "--resolution",
                "320x240",
                "--max-frames",
                "3",
                "--threshold",
                "0.4",
            ]
        )
        self.assertEqual(args.camera, 2)
        self.assertEqual(args.resolution, (320, 240))
        self.assertFalse(args.display)

    def test_invalid_resolution_uses_argparse_error(self) -> None:
        for value in ("640", "0x480", "640x0", "axb"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    app.parse_resolution(value)

    def test_configuration_validation(self) -> None:
        invalid_argv = [
            ["--threshold", "-0.1"],
            ["--threshold", "2.1"],
            ["--min-face-size", "0"],
            ["--max-frames", "0"],
            ["--samples", "1"],
            ["--capture-every", "0"],
            ["--enroll", "Alice", "--input", "x.png"],
            ["--enroll", "Alice", "--output", "x.png"],
        ]
        for argv in invalid_argv:
            with self.subTest(argv=argv):
                with self.assertRaises(ConfigurationError):
                    app.parse_args(argv)

    def test_input_and_camera_are_mutually_exclusive(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                app.parse_args(["--input", "x.png", "--camera", "0"])

    def test_main_converts_expected_configuration_error_to_exit_two(self) -> None:
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            result = app.main(["--threshold", "-1"])
        self.assertEqual(result, 2)
        self.assertIn("error:", errors.getvalue())


class TrackedFixtureIntegrationTests(unittest.TestCase):
    def test_sample_person_builds_index_and_recognizes_tracked_fixture(self) -> None:
        known_faces = PROJECT_ROOT / "known_faces"
        sample = PROJECT_ROOT / "assets" / "input" / "sample-face.png"
        self.assertTrue(sample.is_file())
        image = cv2.imread(str(sample), cv2.IMREAD_COLOR)
        self.assertIsNotNone(image)

        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            detector = FaceDetector()
            embedder = HOGFaceEmbedder()
            store = KnownFaceStore(
                known_faces,
                temporary_path / "embeddings.npz",
                detector,
                embedder,
            )
            index = store.load()
            self.assertEqual(index.labels, ("Sample Person",))
            logger = AttendanceLogger(temporary_path / "attendance.csv")
            pipeline = RecognitionPipeline(detector, embedder, index, logger, threshold=0.35)

            observations = pipeline.process(image, timestamp=datetime(2026, 10, 5, 10, 0))

            self.assertEqual(len(observations), 1)
            self.assertEqual(observations[0].name, "Sample Person")
            self.assertLessEqual(observations[0].match.distance, 1e-6)
            self.assertTrue(observations[0].attendance_recorded)
            self.assertEqual(len(logger.records()), 1)


if __name__ == "__main__":
    unittest.main()
