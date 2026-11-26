"""Tests for CLI validation and application media orchestration."""

from __future__ import annotations

import argparse
import contextlib
import io
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
    FpsMeter,
    build_argument_parser,
    config_from_args,
    main,
    parse_size,
    run,
)
from src.errors import ApplicationError, OCRError  # noqa: E402
from src.ocr import OcrResult  # noqa: E402
from src.vision import Box, DetectorConfig, PlateCandidate  # noqa: E402


class _Clock:
    def __init__(self):
        self.value = 1.0

    def __call__(self):
        result = self.value
        self.value += 0.02
        return result


class _Detector:
    def detect(self, frame):
        height, width = frame.shape[:2]
        box = Box(2, 2, max(2, width - 4), max(2, height - 4))
        return [
            PlateCandidate(
                box=box,
                corners=np.array(
                    [[2, 2], [box.right, 2], [box.right, box.bottom], [2, box.bottom]],
                    dtype=np.float32,
                ),
                plate_image=np.full((20, 80), 255, dtype=np.uint8),
                confidence=0.9,
                aspect_ratio=4.0,
                rectangularity=0.95,
            )
        ]


class _Ocr:
    def __init__(self, validation_error=None):
        self.calls = 0
        self.validation_error = validation_error

    def validate_installation(self):
        if self.validation_error is not None:
            raise self.validation_error

    def recognize(self, image):
        del image
        self.calls += 1
        return OcrResult("CV2026", 0.91)


class _Capture:
    def __init__(self, frames):
        self.frames = [frame.copy() for frame in frames]
        self.released = False
        self.settings = []

    def isOpened(self):
        return not self.released

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def get(self, key):
        del key
        return 24.0

    def set(self, key, value):
        self.settings.append((key, value))
        return True

    def release(self):
        self.released = True


class _Writer:
    def __init__(self):
        self.frames = []
        self.released = False

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        self.released = True


class _Cv:
    IMREAD_COLOR = 1
    FONT_HERSHEY_SIMPLEX = 0
    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4

    def __init__(self, *, image=None, frames=()):
        self.image = image
        self.capture = _Capture(frames)
        self.writer = _Writer()
        self.saved = []
        self.labels = []

    def imread(self, path, mode):
        del path, mode
        return None if self.image is None else self.image.copy()

    def VideoCapture(self, source):
        del source
        return self.capture

    def rectangle(self, frame, start, end, color, thickness):
        del frame, start, end, color, thickness

    def putText(self, frame, text, origin, font, scale, color, thickness):
        del frame, origin, font, scale, color, thickness
        self.labels.append(text)

    def imwrite(self, path, frame):
        self.saved.append((path, frame.copy()))
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
        "camera": 0,
        "input_path": None,
        "output_path": None,
        "resolution": (640, 480),
        "no_display": True,
        "max_frames": 0,
        "ocr_interval": 2,
        "tesseract_cmd": None,
        "ocr_min_confidence": 0.0,
        "tracker_iou": 0.3,
        "max_missed": 2,
        "detector": DetectorConfig(),
    }
    values.update(overrides)
    return AppConfig(**values)


class ArgumentTests(unittest.TestCase):
    def test_size_parser_and_cli_defaults(self):
        self.assertEqual(parse_size("1280X720"), (1280, 720))
        args = build_argument_parser().parse_args(["--input", "plate.jpg", "--no-display"])
        config = config_from_args(args)
        self.assertEqual(config.input_path, Path("plate.jpg"))
        self.assertEqual(config.ocr_interval, 15)
        self.assertEqual(config.detector.max_aspect_ratio, 5.5)

    def test_size_parser_rejects_invalid_values(self):
        for value in ("640", "640*480", "0x480", "axb"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                parse_size(value)

    def test_zero_tracker_iou_is_rejected(self):
        args = build_argument_parser().parse_args(["--tracker-iou", "0"])
        with self.assertRaisesRegex(ValueError, "greater than 0"):
            config_from_args(args)


class MeterTests(unittest.TestCase):
    def test_rolling_and_whole_run_fps_are_distinct(self):
        meter = FpsMeter(window_size=2)
        meter.update(0.1)
        meter.update(0.1)
        rolling = meter.update(1.0)
        self.assertAlmostEqual(rolling, 2 / 1.1)
        self.assertAlmostEqual(meter.average_fps, 3 / 1.2)


class ApplicationTests(unittest.TestCase):
    def test_headless_image_is_recognized_and_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.png"
            output_path = Path(directory) / "output.png"
            input_path.write_bytes(b"decoded by fake OpenCV")
            cv = _Cv(image=np.zeros((40, 100, 3), dtype=np.uint8))
            stats = run(
                _config(input_path=input_path, output_path=output_path),
                cv_module=cv,
                detector=_Detector(),
                ocr_engine=_Ocr(),
                clock=_Clock(),
            )
            self.assertEqual(stats.frames_processed, 1)
            self.assertEqual(stats.plate_observations, 1)
            self.assertEqual(stats.recognized_observations, 1)
            self.assertEqual(stats.ocr_calls, 1)
            self.assertEqual(len(cv.saved), 1)
            self.assertTrue(any("CV2026" in label for label in cv.labels))

    def test_video_honors_frame_limit_and_throttles_ocr(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.avi"
            output_path = Path(directory) / "output.avi"
            input_path.write_bytes(b"decoded by fake OpenCV")
            frames = [np.zeros((40, 100, 3), dtype=np.uint8) for _ in range(5)]
            cv = _Cv(frames=frames)
            ocr = _Ocr()
            stats = run(
                _config(
                    input_path=input_path,
                    output_path=output_path,
                    max_frames=3,
                    ocr_interval=2,
                ),
                cv_module=cv,
                detector=_Detector(),
                ocr_engine=ocr,
                clock=_Clock(),
            )
            self.assertEqual(stats.frames_processed, 3)
            self.assertEqual(stats.plate_observations, 3)
            self.assertEqual(stats.ocr_calls, 2)
            self.assertEqual(ocr.calls, 2)
            self.assertEqual(len(cv.writer.frames), 3)
            self.assertTrue(cv.writer.released)
            self.assertTrue(cv.capture.released)

    def test_camera_resolution_is_requested(self):
        cv = _Cv(frames=[np.zeros((30, 80, 3), dtype=np.uint8)])
        run(
            _config(max_frames=1, resolution=(800, 600)),
            cv_module=cv,
            detector=_Detector(),
            ocr_engine=_Ocr(),
            clock=_Clock(),
        )
        self.assertEqual(
            cv.capture.settings,
            [(cv.CAP_PROP_FRAME_WIDTH, 800), (cv.CAP_PROP_FRAME_HEIGHT, 600)],
        )

    def test_same_input_and_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.png"
            path.write_bytes(b"must survive")
            with self.assertRaisesRegex(ApplicationError, "must be different"):
                run(
                    _config(input_path=path, output_path=path),
                    cv_module=_Cv(image=np.zeros((20, 30, 3), dtype=np.uint8)),
                    detector=_Detector(),
                    ocr_engine=_Ocr(),
                )
            self.assertEqual(path.read_bytes(), b"must survive")

    def test_main_reports_expected_error_without_traceback(self):
        error = io.StringIO()
        with mock.patch("src.app.run", side_effect=OCRError("install Tesseract")):
            with contextlib.redirect_stderr(error):
                exit_code = main(["--no-display", "--max-frames", "1"])
        self.assertEqual(exit_code, 2)
        self.assertEqual(error.getvalue().strip(), "error: install Tesseract")


if __name__ == "__main__":
    unittest.main()

