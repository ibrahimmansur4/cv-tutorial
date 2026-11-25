"""Unit tests for the two detector backends (no model downloads required)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PYTHON_PROJECT = Path(__file__).resolve().parents[1]
if str(PYTHON_PROJECT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PROJECT))

from src.detectors import (  # noqa: E402
    Detection,
    DetectorLoadError,
    DnnFaceDetector,
    FaceDetectionError,
    HaarFaceDetector,
)


class _FakeCascade:
    def __init__(self, boxes=(), empty=False):
        self.boxes = boxes
        self._empty = empty
        self.arguments = None

    def empty(self):
        return self._empty

    def detectMultiScale(self, gray, **kwargs):
        self.arguments = (gray, kwargs)
        return self.boxes


class _FakeHaarCv:
    COLOR_BGR2GRAY = 6

    def __init__(self, cascade):
        self.cascade = cascade
        self.loaded_path = None

    def CascadeClassifier(self, path):
        self.loaded_path = path
        return self.cascade

    @staticmethod
    def cvtColor(frame, conversion):
        del conversion
        return frame[:, :, 0]

    @staticmethod
    def equalizeHist(gray):
        return gray


class _FakeNetwork:
    def __init__(self, output):
        self.output = output
        self.input_blob = None

    @staticmethod
    def empty():
        return False

    def setInput(self, blob):
        self.input_blob = blob

    def forward(self):
        return self.output


class _FakeDnnModule:
    def __init__(self, network):
        self.network = network
        self.loaded_paths = None
        self.blob_arguments = None

    def readNetFromCaffe(self, config, model):
        self.loaded_paths = (config, model)
        return self.network

    def blobFromImage(self, frame, **kwargs):
        self.blob_arguments = (frame, kwargs)
        return "blob"


class _FakeDnnCv:
    def __init__(self, output):
        self.network = _FakeNetwork(output)
        self.dnn = _FakeDnnModule(self.network)


class DetectionTests(unittest.TestCase):
    def test_clipped_box_stays_inside_frame(self):
        detection = Detection(-5, 3, 110, 90, 0.8).clipped(100, 80)
        self.assertEqual(detection, Detection(0, 3, 100, 80, 0.8))

    def test_clipped_box_discards_non_visible_area(self):
        self.assertIsNone(Detection(110, 5, 130, 20).clipped(100, 80))


class HaarFaceDetectorTests(unittest.TestCase):
    def test_detect_converts_and_clips_boxes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cascade_path = Path(temporary_directory) / "cascade.xml"
            cascade_path.write_text("fixture", encoding="utf-8")
            cascade = _FakeCascade(boxes=[(-2, 10, 20, 30), (80, 80, 0, 5)])
            fake_cv = _FakeHaarCv(cascade)
            detector = HaarFaceDetector(cascade_path, cv_module=fake_cv)

            detections = detector.detect(np.zeros((50, 100, 3), dtype=np.uint8))

            self.assertEqual(detections, [Detection(0, 10, 18, 40)])
            self.assertEqual(cascade.arguments[1]["minNeighbors"], 5)
            self.assertEqual(Path(fake_cv.loaded_path), cascade_path)

    def test_missing_cascade_has_actionable_error(self):
        fake_cv = _FakeHaarCv(_FakeCascade())
        with self.assertRaisesRegex(DetectorLoadError, "not found"):
            HaarFaceDetector(Path("does-not-exist.xml"), cv_module=fake_cv)


class DnnFaceDetectorTests(unittest.TestCase):
    def _model_files(self, directory):
        config = Path(directory) / "deploy.prototxt"
        model = Path(directory) / "weights.caffemodel"
        config.write_text("fixture", encoding="utf-8")
        model.write_bytes(b"fixture")
        return config, model

    def test_detect_filters_by_confidence_and_clips_boxes(self):
        output = np.array(
            [
                [
                    [
                        [0.0, 1.0, 0.90, -0.10, 0.10, 1.20, 0.90],
                        [0.0, 1.0, 0.49, 0.10, 0.10, 0.30, 0.30],
                        [0.0, 1.0, 0.80, 0.80, 0.80, 0.70, 0.90],
                    ]
                ]
            ],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            config, model = self._model_files(temporary_directory)
            fake_cv = _FakeDnnCv(output)
            detector = DnnFaceDetector(
                config, model, confidence_threshold=0.5, cv_module=fake_cv
            )

            detections = detector.detect(np.zeros((100, 200, 3), dtype=np.uint8))

            self.assertEqual(len(detections), 1)
            self.assertEqual(detections[0].left, 0)
            self.assertEqual(detections[0].top, 10)
            self.assertEqual(detections[0].right, 200)
            self.assertEqual(detections[0].bottom, 90)
            self.assertAlmostEqual(detections[0].confidence, 0.9, places=5)
            self.assertEqual(fake_cv.network.input_blob, "blob")
            self.assertEqual(fake_cv.dnn.blob_arguments[1]["size"], (300, 300))

    def test_unexpected_network_output_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config, model = self._model_files(temporary_directory)
            fake_cv = _FakeDnnCv(np.zeros((1, 5), dtype=np.float32))
            detector = DnnFaceDetector(config, model, cv_module=fake_cv)
            with self.assertRaisesRegex(FaceDetectionError, "unexpected output shape"):
                detector.detect(np.zeros((20, 20, 3), dtype=np.uint8))

    def test_missing_model_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = Path(temporary_directory) / "deploy.prototxt"
            config.write_text("fixture", encoding="utf-8")
            with self.assertRaisesRegex(DetectorLoadError, "model file not found"):
                DnnFaceDetector(
                    config,
                    Path(temporary_directory) / "missing.caffemodel",
                    cv_module=_FakeDnnCv(np.empty((0,))),
                )


if __name__ == "__main__":
    unittest.main()
