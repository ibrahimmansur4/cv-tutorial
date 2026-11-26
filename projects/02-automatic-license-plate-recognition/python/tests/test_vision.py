"""Tests for plate geometry, enhancement, and the real synthetic fixture."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

PYTHON_PROJECT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PYTHON_PROJECT.parent
if str(PYTHON_PROJECT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PROJECT))

from src.errors import DetectionError  # noqa: E402
from src.vision import (  # noqa: E402
    Box,
    DetectorConfig,
    PlateDetector,
    enhance_plate,
    order_corners,
    perspective_correct,
    validate_frame,
)


class BoxTests(unittest.TestCase):
    def test_properties_clipping_and_iou(self):
        box = Box(-10, 5, 30, 20)
        self.assertEqual(box.right, 20)
        self.assertEqual(box.bottom, 25)
        self.assertEqual(box.clipped(100, 100), Box(0, 5, 20, 20))
        self.assertAlmostEqual(Box(0, 0, 20, 20).iou(Box(10, 0, 20, 20)), 1 / 3)

    def test_invalid_or_invisible_box_is_rejected(self):
        with self.assertRaises(ValueError):
            Box(0, 0, 0, 5)
        self.assertIsNone(Box(20, 20, 5, 5).clipped(10, 10))


class ConfigurationTests(unittest.TestCase):
    def test_defaults_match_the_tuned_fixture_range(self):
        config = DetectorConfig()
        self.assertEqual(config.morph_kernel, (9, 3))
        self.assertEqual(config.morph_iterations, 1)
        self.assertEqual(config.max_aspect_ratio, 5.5)

    def test_invalid_ranges_are_rejected(self):
        invalid = (
            {"min_area_ratio": 0.1, "max_area_ratio": 0.05},
            {"min_aspect_ratio": 4.0, "max_aspect_ratio": 3.0},
            {"blur_kernel": 4},
            {"canny_low": 200, "canny_high": 100},
            {"morph_kernel": (0, 3)},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                DetectorConfig(**values)


class GeometryTests(unittest.TestCase):
    def test_order_corners_is_stable_for_shuffled_input(self):
        points = np.array([[90, 80], [10, 10], [15, 85], [95, 15]], dtype=np.float32)
        ordered = order_corners(points)
        np.testing.assert_array_equal(
            ordered,
            np.array([[10, 10], [95, 15], [90, 80], [15, 85]], dtype=np.float32),
        )

    def test_degenerate_corners_are_rejected(self):
        with self.assertRaises(DetectionError):
            order_corners(np.array([[0, 0], [0, 0], [4, 4], [0, 4]]))

    def test_perspective_correction_and_enhancement(self):
        frame = np.full((100, 180, 3), 230, dtype=np.uint8)
        cv2.putText(
            frame,
            "AB12",
            (38, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (10, 10, 10),
            3,
            cv2.LINE_AA,
        )
        corners = np.array([[20, 15], [165, 22], [155, 82], [15, 75]], dtype=np.float32)
        warped = perspective_correct(frame, corners)
        enhanced = enhance_plate(warped, target_width=240)

        self.assertGreater(warped.shape[1], warped.shape[0])
        self.assertEqual(enhanced.ndim, 2)
        self.assertLess(enhanced.shape[1], 240)  # OCR artifact margin is removed.
        self.assertTrue(set(np.unique(enhanced)).issubset({0, 255}))

    def test_frame_validation_rejects_unsupported_channels(self):
        with self.assertRaises(DetectionError):
            validate_frame(np.zeros((10, 10, 2), dtype=np.uint8))


class FixtureIntegrationTests(unittest.TestCase):
    def test_default_detector_finds_only_the_fixture_plate(self):
        fixture = PROJECT_ROOT / "assets" / "input" / "synthetic_vehicle.png"
        image = cv2.imread(str(fixture), cv2.IMREAD_COLOR)
        self.assertIsNotNone(image, f"missing generated fixture: {fixture}")

        candidates = PlateDetector(DetectorConfig()).detect(image)

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertGreater(candidate.confidence, 0.85)
        self.assertGreater(candidate.box.width, 200)
        self.assertGreater(candidate.box.height, 40)
        self.assertGreater(candidate.aspect_ratio, 3.5)
        self.assertLess(candidate.aspect_ratio, 5.0)


if __name__ == "__main__":
    unittest.main()

