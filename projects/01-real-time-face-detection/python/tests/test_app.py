"""Tests for CLI validation and reproducible image/video execution."""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

PYTHON_PROJECT = Path(__file__).resolve().parents[1]
if str(PYTHON_PROJECT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PROJECT))

from src.app import (  # noqa: E402
    AppConfig,
    ApplicationError,
    FpsMeter,
    build_argument_parser,
    config_from_args,
    main,
    parse_resolution,
    run,
)
from src.detectors import DependencyError, Detection  # noqa: E402


class _FixedDetector:
    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        height, width = frame.shape[:2]
        return [Detection(1, 2, width - 1, height - 2, 0.75)]


class _Clock:
    def __init__(self):
        self.value = 10.0

    def __call__(self):
        result = self.value
        self.value += 0.02
        return result


class _FakeCapture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.released = False
        self.settings = []

    def isOpened(self):
        return not self.released

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0).copy()

    def set(self, key, value):
        self.settings.append((key, value))
        return True

    @staticmethod
    def get(key):
        del key
        return 24.0

    def release(self):
        self.released = True


class _FakeWriter:
    def __init__(self):
        self.frames = []
        self.released = False

    @staticmethod
    def isOpened():
        return True

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        self.released = True


class _FakeCv:
    IMREAD_COLOR = 1
    FONT_HERSHEY_SIMPLEX = 0
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5

    def __init__(self, *, image=None, frames=()):
        self.image = image
        self.capture = _FakeCapture(frames)
        self.writer = _FakeWriter()
        self.rectangles = []
        self.labels = []
        self.saved_images = []

    def imread(self, path, mode):
        del path, mode
        return None if self.image is None else self.image.copy()

    def VideoCapture(self, source):
        del source
        return self.capture

    def rectangle(self, frame, start, end, color, thickness):
        del frame, color, thickness
        self.rectangles.append((start, end))

    def putText(self, frame, text, origin, font, scale, color, thickness):
        del frame, origin, font, scale, color, thickness
        self.labels.append(text)

    def imwrite(self, path, frame):
        self.saved_images.append((path, frame.copy()))
        return True

    @staticmethod
    def VideoWriter_fourcc(*characters):
        return "".join(characters)

    def VideoWriter(self, path, fourcc, fps, size):
        del path, fourcc, fps, size
        return self.writer

    @staticmethod
    def destroyAllWindows():
        return None


def _config(**overrides):
    values = {
        "detector": "haar",
        "camera": 0,
        "input_path": None,
        "confidence": 0.5,
        "resolution": (640, 480),
        "cascade_path": None,
        "dnn_config_path": None,
        "dnn_model_path": None,
        "no_display": True,
        "max_frames": 0,
        "output_path": None,
    }
    values.update(overrides)
    return AppConfig(**values)


class ArgumentTests(unittest.TestCase):
    def test_resolution_accepts_upper_or_lower_x(self):
        self.assertEqual(parse_resolution("1280x720"), (1280, 720))
        self.assertEqual(parse_resolution("320X240"), (320, 240))

    def test_resolution_rejects_invalid_values(self):
        for value in ("640", "640*480", "0x480", "axb"):
            with self.subTest(value=value), self.assertRaises(
                argparse.ArgumentTypeError
            ):
                parse_resolution(value)

    def test_cli_defaults_and_input_override_are_represented(self):
        parser = build_argument_parser()
        args = parser.parse_args(["--input", "clip.mp4", "--no-display"])
        config = config_from_args(args)
        self.assertEqual(config.detector, "haar")
        self.assertEqual(config.camera, 0)
        self.assertEqual(config.resolution, (640, 480))
        self.assertEqual(config.input_path, Path("clip.mp4"))
        self.assertTrue(config.no_display)


class FpsMeterTests(unittest.TestCase):
    def test_moving_average_uses_frame_durations(self):
        meter = FpsMeter(window_size=2)
        self.assertAlmostEqual(meter.update(0.1), 10.0)
        self.assertAlmostEqual(meter.update(0.3), 5.0)
        self.assertAlmostEqual(meter.update(0.1), 5.0)
        self.assertAlmostEqual(meter.average_fps, 6.0)

    def test_whole_run_average_is_not_limited_to_rolling_window(self):
        meter = FpsMeter(window_size=2)
        meter.update(0.1)
        meter.update(0.1)
        rolling_fps = meter.update(1.0)

        self.assertAlmostEqual(rolling_fps, 2.0 / 1.1)
        self.assertAlmostEqual(meter.average_fps, 3.0 / 1.2)


class PipelineTests(unittest.TestCase):
    def test_same_input_and_output_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "source.png"
            input_path.write_bytes(b"source must remain untouched")
            fake_cv = _FakeCv(image=np.zeros((20, 20, 3), dtype=np.uint8))

            with self.assertRaisesRegex(ApplicationError, "must be different"):
                run(
                    _config(input_path=input_path, output_path=input_path),
                    cv_module=fake_cv,
                    detector=_FixedDetector(),
                    clock=_Clock(),
                )

            self.assertEqual(input_path.read_bytes(), b"source must remain untouched")

    def test_hard_link_to_input_is_rejected_as_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "source.png"
            output_path = Path(temporary_directory) / "alias.png"
            input_path.write_bytes(b"source must remain untouched")
            try:
                os.link(input_path, output_path)
            except OSError as exc:
                self.skipTest(f"hard links are unavailable: {exc}")

            with self.assertRaisesRegex(ApplicationError, "must be different"):
                run(
                    _config(input_path=input_path, output_path=output_path),
                    cv_module=_FakeCv(image=np.zeros((20, 20, 3), dtype=np.uint8)),
                    detector=_FixedDetector(),
                    clock=_Clock(),
                )

            self.assertEqual(input_path.read_bytes(), b"source must remain untouched")

    def test_headless_image_can_be_annotated_and_saved(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "input.png"
            output_path = Path(temporary_directory) / "output.png"
            input_path.write_bytes(b"fake input; decoded by fake cv")
            fake_cv = _FakeCv(image=np.zeros((40, 60, 3), dtype=np.uint8))
            detector = _FixedDetector()

            stats = run(
                _config(input_path=input_path, output_path=output_path),
                cv_module=fake_cv,
                detector=detector,
                clock=_Clock(),
            )

            self.assertEqual(stats.frames_processed, 1)
            self.assertEqual(stats.faces_detected, 1)
            self.assertAlmostEqual(stats.average_fps, 50.0)
            self.assertEqual(len(fake_cv.saved_images), 1)
            self.assertIn("Face 75.0%", fake_cv.labels)
            self.assertTrue(any(label.startswith("FPS:") for label in fake_cv.labels))

    def test_output_path_expands_user_home(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            home = Path(temporary_directory)
            input_path = home / "input.png"
            input_path.write_bytes(b"fake input; decoded by fake cv")
            fake_cv = _FakeCv(image=np.zeros((20, 20, 3), dtype=np.uint8))

            with mock.patch.dict(os.environ, {"USERPROFILE": str(home)}):
                run(
                    _config(
                        input_path=input_path,
                        output_path=Path("~/output.png"),
                    ),
                    cv_module=fake_cv,
                    detector=_FixedDetector(),
                    clock=_Clock(),
                )

            self.assertEqual(Path(fake_cv.saved_images[0][0]), home / "output.png")

    def test_video_honors_max_frames_and_writes_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "input.mp4"
            output_path = Path(temporary_directory) / "output.avi"
            input_path.write_bytes(b"fake input; decoded by fake cv")
            frames = [np.zeros((24, 32, 3), dtype=np.uint8) for _ in range(4)]
            fake_cv = _FakeCv(frames=frames)
            detector = _FixedDetector()

            stats = run(
                _config(
                    input_path=input_path,
                    output_path=output_path,
                    max_frames=2,
                ),
                cv_module=fake_cv,
                detector=detector,
                clock=_Clock(),
            )

            self.assertEqual(stats.frames_processed, 2)
            self.assertEqual(stats.faces_detected, 2)
            self.assertEqual(len(fake_cv.writer.frames), 2)
            self.assertTrue(fake_cv.writer.released)
            self.assertTrue(fake_cv.capture.released)

    def test_camera_resolution_is_requested(self):
        fake_cv = _FakeCv(frames=[np.zeros((20, 20, 3), dtype=np.uint8)])
        run(
            _config(resolution=(800, 600), max_frames=1),
            cv_module=fake_cv,
            detector=_FixedDetector(),
            clock=_Clock(),
        )
        self.assertEqual(
            fake_cv.capture.settings,
            [(fake_cv.CAP_PROP_FRAME_WIDTH, 800), (fake_cv.CAP_PROP_FRAME_HEIGHT, 600)],
        )

    def test_main_reports_missing_opencv_without_traceback(self):
        error_output = io.StringIO()
        with mock.patch("src.app.require_opencv", side_effect=DependencyError("install it")):
            with contextlib.redirect_stderr(error_output):
                exit_code = main(["--no-display", "--max-frames", "1"])
        self.assertEqual(exit_code, 2)
        self.assertEqual(error_output.getvalue().strip(), "error: install it")


if __name__ == "__main__":
    unittest.main()
