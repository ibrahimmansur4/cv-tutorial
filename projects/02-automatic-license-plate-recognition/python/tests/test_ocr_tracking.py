"""Tests for OCR normalization, IoU tracking, and OCR throttling."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PYTHON_PROJECT = Path(__file__).resolve().parents[1]
if str(PYTHON_PROJECT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PROJECT))

from src.errors import OCRError  # noqa: E402
from src.ocr import OcrResult, TesseractOcr, sanitize_plate_text  # noqa: E402
from src.pipeline import ALPRPipeline  # noqa: E402
from src.tracking import IoUTracker  # noqa: E402
from src.vision import Box, PlateCandidate  # noqa: E402


class _Output:
    DICT = "dict"


class _CommandHolder:
    tesseract_cmd = "tesseract"


class _FakePytesseract:
    Output = _Output
    pytesseract = _CommandHolder()

    def __init__(self, data=None, version_error=None):
        self.data = data or {"text": [], "conf": []}
        self.version_error = version_error
        self.config = None

    def get_tesseract_version(self):
        if self.version_error is not None:
            raise self.version_error
        return "5.5"

    def image_to_data(self, image, *, config, output_type):
        self.config = config
        self.output_type = output_type
        self.image_shape = image.shape
        return self.data


def _candidate(x: int = 10, y: int = 10) -> PlateCandidate:
    box = Box(x, y, 120, 30)
    return PlateCandidate(
        box=box,
        corners=np.array(
            [[x, y], [x + 120, y], [x + 120, y + 30], [x, y + 30]],
            dtype=np.float32,
        ),
        plate_image=np.full((36, 160), 255, dtype=np.uint8),
        confidence=0.9,
        aspect_ratio=4.0,
        rectangularity=0.95,
    )


class OcrTests(unittest.TestCase):
    def test_text_is_normalized_to_ascii_plate_characters(self):
        self.assertEqual(sanitize_plate_text(" ab-12 / c! "), "AB12C")
        self.assertEqual(sanitize_plate_text("abcdefghijkl", 5), "ABCDE")

    def test_recognition_combines_tokens_and_weights_confidence(self):
        fake = _FakePytesseract(
            {"text": ["", "cv-20", "26"], "conf": [-1, 80, 60]}
        )
        engine = TesseractOcr(pytesseract_module=fake)

        result = engine.recognize(np.full((20, 100), 255, dtype=np.uint8))

        self.assertEqual(result.text, "CV2026")
        self.assertAlmostEqual(result.confidence, (4 * 0.8 + 2 * 0.6) / 6)
        self.assertIn("abcdefghijklmnopqrstuvwxyz", fake.config)

    def test_minimum_confidence_filters_tokens(self):
        fake = _FakePytesseract({"text": ["LOW", "OK"], "conf": [40, 90]})
        result = TesseractOcr(
            minimum_confidence=0.5, pytesseract_module=fake
        ).recognize(np.ones((10, 20), dtype=np.uint8))
        self.assertEqual(result, OcrResult("OK", 0.9))

    def test_missing_explicit_executable_is_actionable(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "tesseract.exe"
            with self.assertRaisesRegex(OCRError, "not found"):
                TesseractOcr(tesseract_cmd=missing, pytesseract_module=_FakePytesseract())

    def test_invalid_confidence_is_rejected(self):
        with self.assertRaises(ValueError):
            OcrResult("ABC", 1.1)


class TrackingTests(unittest.TestCase):
    def test_overlapping_candidate_keeps_track_id(self):
        tracker = IoUTracker(iou_threshold=0.3, max_missed_frames=1)
        first = tracker.update([_candidate(10, 10)], 0)[0]
        second = tracker.update([_candidate(14, 11)], 1)[0]
        self.assertEqual(first.track_id, second.track_id)
        self.assertEqual(second.box, Box(14, 11, 120, 30))

    def test_track_expires_after_missed_frame_budget(self):
        tracker = IoUTracker(iou_threshold=0.3, max_missed_frames=1)
        tracker.update([_candidate()], 0)
        tracker.update([], 1)
        self.assertEqual(len(tracker.tracks), 1)
        tracker.update([], 2)
        self.assertEqual(tracker.tracks, ())

    def test_non_increasing_frame_index_is_rejected(self):
        tracker = IoUTracker()
        tracker.update([_candidate()], 4)
        with self.assertRaises(ValueError):
            tracker.update([_candidate()], 4)


class _Detector:
    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        del frame
        self.calls += 1
        return [_candidate()]


class _Ocr:
    def __init__(self):
        self.calls = 0

    def recognize(self, image):
        del image
        self.calls += 1
        return OcrResult("CV2026", 0.88)


class PipelineTests(unittest.TestCase):
    def test_tracking_reuses_text_between_ocr_intervals(self):
        detector = _Detector()
        ocr = _Ocr()
        pipeline = ALPRPipeline(
            detector, ocr, tracker=IoUTracker(), ocr_interval=3
        )

        results = [
            pipeline.process(np.zeros((20, 20, 3), dtype=np.uint8), index, use_tracking=True)
            for index in range(7)
        ]

        self.assertEqual(ocr.calls, 3)  # frames 0, 3, and 6 rather than every frame
        self.assertEqual([result.ocr_calls for result in results], [1, 0, 0, 1, 0, 0, 1])
        self.assertTrue(all(result.observations[0].text == "CV2026" for result in results))
        self.assertEqual({result.observations[0].track_id for result in results}, {1})

    def test_still_image_runs_ocr_for_every_candidate(self):
        ocr = _Ocr()
        result = ALPRPipeline(_Detector(), ocr).process(
            np.zeros((10, 10, 3), dtype=np.uint8), 0, use_tracking=False
        )
        self.assertEqual(result.ocr_calls, 1)
        self.assertIsNone(result.observations[0].track_id)


if __name__ == "__main__":
    unittest.main()

