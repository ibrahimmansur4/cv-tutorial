from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from src.detection import FaceBox, FaceDetector
from src.embeddings import FaceIndex, HOGFaceEmbedder
from src.errors import ConfigurationError, FaceDataError


class _Classifier:
    def __init__(self, detections: list[tuple[int, int, int, int]]) -> None:
        self.detections = detections
        self.calls: list[dict[str, object]] = []

    def detectMultiScale(self, image: np.ndarray, **kwargs: object) -> list[tuple[int, int, int, int]]:
        self.calls.append({"image": image, **kwargs})
        return self.detections


class _ZeroDescriptor:
    def compute(self, image: np.ndarray) -> np.ndarray:
        return np.zeros((4, 1), dtype=np.float32)

    def getDescriptorSize(self) -> int:
        return 4


class FaceBoxTests(unittest.TestCase):
    def test_geometry_clipping_and_crop(self) -> None:
        box = FaceBox(-2, 1, 7, 6)
        self.assertEqual((box.x2, box.y2, box.area), (5, 7, 42))
        self.assertEqual(box.clipped((5, 4, 3)), FaceBox(0, 1, 4, 4))

        frame = np.arange(5 * 4, dtype=np.uint8).reshape(5, 4)
        crop = box.crop(frame)
        np.testing.assert_array_equal(crop, frame[1:5, 0:4])

    def test_fully_outside_box_becomes_empty(self) -> None:
        self.assertEqual(FaceBox(20, 30, 4, 5).clipped((10, 10)), FaceBox(10, 10, 0, 0))

    def test_clipping_requires_height_and_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "height and width"):
            FaceBox(0, 0, 1, 1).clipped((10,))


class FaceDetectorTests(unittest.TestCase):
    def test_detects_clips_filters_and_sorts_faces(self) -> None:
        classifier = _Classifier(
            [
                (4, 4, 3, 3),
                (-2, -2, 6, 6),
                (20, 20, 2, 2),
                (1, 1, 1, 1),
            ]
        )
        detector = FaceDetector(
            classifier=classifier,
            scale_factor=1.2,
            min_neighbors=3,
            min_size=(12, 14),
        )

        boxes = detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))

        self.assertEqual(boxes, [FaceBox(0, 0, 4, 4), FaceBox(4, 4, 3, 3), FaceBox(1, 1, 1, 1)])
        call = classifier.calls[0]
        self.assertEqual(call["scaleFactor"], 1.2)
        self.assertEqual(call["minNeighbors"], 3)
        self.assertEqual(call["minSize"], (12, 14))
        self.assertEqual(np.asarray(call["image"]).shape, (8, 8))

    def test_accepts_grayscale_and_bgra(self) -> None:
        detector = FaceDetector(classifier=_Classifier([]))
        self.assertEqual(detector.detect(np.zeros((8, 8), dtype=np.uint8)), [])
        self.assertEqual(detector.detect(np.zeros((8, 8, 4), dtype=np.uint8)), [])

    def test_rejects_invalid_frames(self) -> None:
        detector = FaceDetector(classifier=_Classifier([]))
        for frame in (None, np.array([], dtype=np.uint8), np.zeros((4,), dtype=np.uint8), np.zeros((4, 4, 2), dtype=np.uint8)):
            with self.subTest(shape=getattr(frame, "shape", None)):
                with self.assertRaises(ValueError):
                    detector.detect(frame)  # type: ignore[arg-type]

    def test_rejects_invalid_configuration_and_missing_cascade(self) -> None:
        with self.assertRaises(ConfigurationError):
            FaceDetector(classifier=_Classifier([]), scale_factor=1.0)
        with self.assertRaises(ConfigurationError):
            FaceDetector(classifier=_Classifier([]), min_neighbors=-1)
        with self.assertRaises(ConfigurationError):
            FaceDetector(classifier=_Classifier([]), min_size=(0, 10))
        with self.assertRaisesRegex(ConfigurationError, "not found"):
            FaceDetector(Path("definitely-missing-cascade.xml"))

    def test_bundled_haar_cascade_loads(self) -> None:
        detector = FaceDetector()
        self.assertIsNotNone(detector.cascade_path)
        self.assertTrue(detector.cascade_path.is_file())  # type: ignore[union-attr]


class HOGFaceEmbedderTests(unittest.TestCase):
    def test_embedding_is_deterministic_fixed_size_and_normalized(self) -> None:
        rng = np.random.default_rng(1234)
        gray = rng.integers(0, 256, size=(83, 71), dtype=np.uint8)
        embedder = HOGFaceEmbedder()

        first = embedder.embed(gray)
        second = embedder.embed(gray)
        bgr = embedder.embed(np.repeat(gray[:, :, None], 3, axis=2))

        self.assertEqual(first.shape, (8100,))
        self.assertEqual(first.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(first)), 1.0, places=6)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first, bgr, atol=1e-7)

    def test_rejects_bad_image_size_and_face_inputs(self) -> None:
        for image_size in ((16, 32), (33, 32), (32, 47)):
            with self.subTest(image_size=image_size):
                with self.assertRaises(ConfigurationError):
                    HOGFaceEmbedder(image_size)

        embedder = HOGFaceEmbedder()
        for face in (None, np.array([]), np.zeros((4,)), np.zeros((4, 4, 2))):
            with self.subTest(shape=getattr(face, "shape", None)):
                with self.assertRaises(FaceDataError):
                    embedder.embed(face)  # type: ignore[arg-type]

    def test_rejects_zero_descriptor(self) -> None:
        embedder = HOGFaceEmbedder(descriptor=_ZeroDescriptor())
        with self.assertRaisesRegex(FaceDataError, "no usable information"):
            embedder.embed(np.full((40, 40), 127, dtype=np.uint8))


class FaceIndexTests(unittest.TestCase):
    def test_nearest_cosine_match_and_unknown_threshold(self) -> None:
        index = FaceIndex(
            ["Alice", "Bob"],
            np.asarray([[2.0, 0.0], [0.0, 5.0]], dtype=np.float32),
        )
        match = index.match(np.asarray([0.8, 0.6], dtype=np.float32), threshold=0.25)
        self.assertEqual(match.name, "Alice")
        self.assertTrue(match.recognized)
        self.assertAlmostEqual(match.similarity, 0.8, places=6)
        self.assertAlmostEqual(match.distance, 0.2, places=6)

        rejected = index.match(
            np.asarray([0.8, 0.6], dtype=np.float32),
            threshold=match.distance - 1e-6,
        )
        self.assertIsNone(rejected.name)
        self.assertFalse(rejected.recognized)

        boundary = index.match(
            np.asarray([0.8, 0.6], dtype=np.float32),
            threshold=match.distance,
        )
        self.assertEqual(boundary.name, "Alice")

    def test_tie_is_stable_and_uses_first_row(self) -> None:
        index = FaceIndex(["First", "Second"], np.asarray([[1.0, 0.0], [1.0, 0.0]]))
        self.assertEqual(index.match(np.asarray([1.0, 0.0]), threshold=0.0).name, "First")

    def test_empty_index_returns_unknown(self) -> None:
        index = FaceIndex([], np.empty((0, 3), dtype=np.float32))
        match = index.match(np.zeros(3, dtype=np.float32))
        self.assertIsNone(match.name)
        self.assertEqual(match.similarity, 0.0)
        self.assertEqual(match.distance, float("inf"))

    def test_rejects_invalid_index_and_query_data(self) -> None:
        bad_cases = (
            (["Alice"], np.ones(3)),
            (["Alice", "Bob"], np.ones((1, 3))),
            ([""], np.ones((1, 3))),
            (["Alice"], np.asarray([[np.nan, 0.0]])),
            (["Alice"], np.zeros((1, 3))),
        )
        for labels, embeddings in bad_cases:
            with self.subTest(labels=labels, shape=embeddings.shape):
                with self.assertRaises(FaceDataError):
                    FaceIndex(labels, embeddings)

        index = FaceIndex(["Alice"], np.asarray([[1.0, 0.0]]))
        with self.assertRaises(FaceDataError):
            index.match(np.asarray([1.0, 0.0, 0.0]))
        with self.assertRaises(FaceDataError):
            index.match(np.zeros(2))
        with self.assertRaises(ConfigurationError):
            index.match(np.asarray([1.0, 0.0]), threshold=-0.01)
        with self.assertRaises(ConfigurationError):
            index.match(np.asarray([1.0, 0.0]), threshold=2.01)


if __name__ == "__main__":
    unittest.main()
