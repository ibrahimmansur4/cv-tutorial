from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from src.attendance_log import AttendanceLogger, AttendanceRecord, CSV_FIELDS
from src.detection import FaceBox
from src.embeddings import CACHE_SCHEMA_VERSION, KnownFaceStore
from src.errors import FaceDataError


class _DescriptorShape:
    def getDescriptorSize(self) -> int:
        return 3


class _FakeEmbedder:
    image_size = (128, 128)

    def __init__(self) -> None:
        self._descriptor = _DescriptorShape()
        self.calls: list[np.ndarray] = []

    def embed(self, face: np.ndarray) -> np.ndarray:
        self.calls.append(face.copy())
        value = float(face.reshape(-1)[0])
        return np.asarray([value + 1.0, 1.0, 0.5], dtype=np.float32)


class _FakeDetector:
    def __init__(self, *, faces: list[FaceBox] | None = None) -> None:
        self.faces = [FaceBox(1, 1, 4, 4)] if faces is None else faces
        self.calls = 0
        self.cascade_path = Path("fake-haar.xml")
        self.scale_factor = 1.1
        self.min_neighbors = 5
        self.min_size = (48, 48)

    def detect(self, image: np.ndarray) -> list[FaceBox]:
        self.calls += 1
        return list(self.faces)


class _FakeCV:
    IMREAD_COLOR = 1

    @staticmethod
    def imread(path: str, flags: int) -> np.ndarray | None:
        source = Path(path)
        if "unreadable" in source.name:
            return None
        payload = source.read_bytes()
        value = payload[0] if payload else 1
        return np.full((8, 8, 3), value, dtype=np.uint8)


class KnownFaceStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.known = self.root / "known"
        self.cache = self.root / "cache" / "faces.npz"
        (self.known / "Alice").mkdir(parents=True)
        (self.known / "Alice" / "portrait.JPG").write_bytes(b"\x0a-image")
        (self.known / "bob_smith.png").write_bytes(b"\x14-image")
        (self.known / ".hidden").mkdir()
        (self.known / ".hidden" / "ignored.png").write_bytes(b"\x1e-image")
        self.detector = _FakeDetector()
        self.embedder = _FakeEmbedder()
        self.store = KnownFaceStore(
            self.known,
            self.cache,
            self.detector,  # type: ignore[arg-type]
            self.embedder,  # type: ignore[arg-type]
            cv2_module=_FakeCV,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_rebuild_writes_safe_cache_and_second_load_reuses_it(self) -> None:
        first = self.store.load()

        self.assertEqual(first.labels, ("Alice", "bob smith"))
        self.assertEqual(len(first), 2)
        self.assertEqual(self.detector.calls, 2)
        self.assertEqual(len(self.embedder.calls), 2)
        self.assertFalse(self.store.last_report.used_cache)
        self.assertEqual(self.store.last_report.discovered, 2)
        self.assertTrue(self.cache.is_file())

        with np.load(self.cache, allow_pickle=False) as archive:
            self.assertEqual(int(archive["schema"][0]), CACHE_SCHEMA_VERSION)
            self.assertIn(archive["labels"].dtype.kind, {"U", "S"})
            self.assertEqual(archive["embeddings"].dtype, np.float32)
            self.assertEqual(archive["embeddings"].shape, (2, 3))

        second = self.store.load()
        self.assertEqual(second.labels, first.labels)
        np.testing.assert_allclose(second.embeddings, first.embeddings)
        self.assertTrue(self.store.last_report.used_cache)
        self.assertEqual(self.detector.calls, 2)
        self.assertEqual(len(self.embedder.calls), 2)

    def test_added_and_removed_images_invalidate_cache(self) -> None:
        self.store.load()
        initial_calls = self.embedder.calls.__len__()

        new_image = self.known / "Charlie.webp"
        new_image.write_bytes(b"\x28-new")
        added = self.store.load()
        self.assertEqual(len(added), 3)
        self.assertFalse(self.store.last_report.used_cache)
        self.assertEqual(len(self.embedder.calls), initial_calls + 3)

        new_image.unlink()
        removed = self.store.load()
        self.assertEqual(len(removed), 2)
        self.assertFalse(self.store.last_report.used_cache)

    def test_modified_image_and_force_rebuild_invalidate_cache(self) -> None:
        self.store.load()
        calls = len(self.embedder.calls)
        target = self.known / "bob_smith.png"
        target.write_bytes(b"\x32-image-with-different-size")

        self.store.load()
        self.assertFalse(self.store.last_report.used_cache)
        self.assertEqual(len(self.embedder.calls), calls + 2)

        calls = len(self.embedder.calls)
        self.store.load(force_rebuild=True)
        self.assertFalse(self.store.last_report.used_cache)
        self.assertEqual(len(self.embedder.calls), calls + 2)

    def test_corrupt_or_object_dtype_cache_is_discarded_and_rebuilt(self) -> None:
        self.cache.parent.mkdir(parents=True)
        self.cache.write_bytes(b"not an npz archive")
        rebuilt = self.store.load()
        self.assertEqual(len(rebuilt), 2)
        self.assertFalse(self.store.last_report.used_cache)

        images = self.store._images()
        signature = self.store._signature(images)
        np.savez(
            self.cache,
            schema=np.asarray([CACHE_SCHEMA_VERSION], dtype=np.int32),
            signature=np.asarray([signature]),
            labels=np.asarray(["unsafe"], dtype=object),
            embeddings=np.ones((1, 3), dtype=np.float32),
        )
        calls = len(self.embedder.calls)
        safe = self.store.load()
        self.assertEqual(safe.labels, ("Alice", "bob smith"))
        self.assertFalse(self.store.last_report.used_cache)
        self.assertEqual(len(self.embedder.calls), calls + 2)

    def test_missing_face_is_skipped_unless_full_image_fallback_is_enabled(self) -> None:
        detector = _FakeDetector(faces=[])
        embedder = _FakeEmbedder()
        store = KnownFaceStore(
            self.known,
            self.root / "strict.npz",
            detector,  # type: ignore[arg-type]
            embedder,  # type: ignore[arg-type]
            cv2_module=_FakeCV,
        )
        strict = store.load()
        self.assertEqual(len(strict), 0)
        self.assertEqual(strict.dimension, 3)
        self.assertEqual(len(store.last_report.skipped), 2)
        self.assertTrue(all("no face detected" in reason for reason in store.last_report.skipped))

        fallback_store = KnownFaceStore(
            self.known,
            self.root / "fallback.npz",
            detector,  # type: ignore[arg-type]
            embedder,  # type: ignore[arg-type]
            cv2_module=_FakeCV,
            allow_full_image_fallback=True,
        )
        fallback = fallback_store.load()
        self.assertEqual(len(fallback), 2)

    def test_multiple_faces_are_skipped_as_ambiguous(self) -> None:
        detector = _FakeDetector(faces=[FaceBox(0, 0, 3, 3), FaceBox(4, 4, 3, 3)])
        store = KnownFaceStore(
            self.known,
            self.root / "ambiguous.npz",
            detector,  # type: ignore[arg-type]
            _FakeEmbedder(),  # type: ignore[arg-type]
            cv2_module=_FakeCV,
            allow_full_image_fallback=True,
        )

        index = store.load()

        self.assertEqual(len(index), 0)
        self.assertEqual(len(store.last_report.skipped), 2)
        self.assertTrue(
            all("multiple faces detected" in reason for reason in store.last_report.skipped)
        )

    def test_unreadable_image_is_reported_and_skipped(self) -> None:
        (self.known / "unreadable.png").write_bytes(b"broken")
        index = self.store.load()
        self.assertEqual(len(index), 2)
        self.assertEqual(self.store.last_report.discovered, 3)
        self.assertTrue(any("unreadable image" in item for item in self.store.last_report.skipped))

    def test_known_faces_path_must_be_directory(self) -> None:
        regular_file = self.root / "not-a-directory"
        regular_file.write_text("x", encoding="utf-8")
        store = KnownFaceStore(
            regular_file,
            self.cache,
            self.detector,  # type: ignore[arg-type]
            self.embedder,  # type: ignore[arg-type]
            cv2_module=_FakeCV,
        )
        with self.assertRaisesRegex(FaceDataError, "not a directory"):
            store.load()


class AttendanceLoggerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "data" / "attendance.csv"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_exact_header_row_and_records(self) -> None:
        when = datetime(2026, 10, 5, 9, 8, 7, 654321)
        logger = AttendanceLogger(self.path)

        self.assertTrue(logger.mark("  Alice   Example  ", when=when))
        self.assertEqual(self.path.read_text(encoding="utf-8").splitlines()[0], "name,date,time")
        self.assertEqual(
            logger.records(),
            [AttendanceRecord("Alice Example", "2026-10-05", "09:08:07")],
        )
        self.assertEqual(logger.last_record, AttendanceRecord("Alice Example", "2026-10-05", "09:08:07"))

    def test_same_day_is_case_insensitive_and_next_day_is_allowed(self) -> None:
        logger = AttendanceLogger(self.path)
        self.assertTrue(logger.mark("Alice", when=datetime(2026, 10, 5, 8, 0)))
        self.assertFalse(logger.mark(" alice ", when=datetime(2026, 10, 5, 17, 0)))
        self.assertIsNone(logger.last_record)
        self.assertTrue(logger.mark("ALICE", when=datetime(2026, 10, 6, 8, 0)))
        self.assertEqual(len(logger.records()), 2)

        reloaded = AttendanceLogger(self.path)
        self.assertFalse(reloaded.mark("aLiCe", when=datetime(2026, 10, 6, 12, 0)))
        self.assertEqual(len(reloaded.records()), 2)

    def test_csv_quoting_and_unicode_round_trip(self) -> None:
        logger = AttendanceLogger(self.path)
        self.assertTrue(logger.mark("Zoë, Ahmed", when=datetime(2026, 1, 2, 3, 4, 5)))
        with self.path.open("r", newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]["name"], "Zoë, Ahmed")
        self.assertEqual(tuple(rows[0]), CSV_FIELDS)

    def test_formula_control_empty_and_long_names_are_rejected(self) -> None:
        invalid_names = ["", "   ", "=SUM(A1:A2)", "+cmd", "-2", "@name", "line\x00break", "x" * 129]
        logger = AttendanceLogger(self.path)
        for name in invalid_names:
            with self.subTest(name=name[:20]):
                with self.assertRaises(FaceDataError):
                    logger.mark(name, when=datetime(2026, 1, 1))
        self.assertFalse(self.path.exists())

    def test_wrong_header_is_rejected_for_mark_and_records(self) -> None:
        self.path.parent.mkdir(parents=True)
        self.path.write_text("Name,Date,Time\nAlice,2026-01-01,10:00:00\n", encoding="utf-8")
        logger = AttendanceLogger(self.path)
        with self.assertRaisesRegex(FaceDataError, "must have header"):
            logger.mark("Bob", when=datetime(2026, 1, 1))
        with self.assertRaisesRegex(FaceDataError, "must have header"):
            logger.records()

    def test_empty_existing_file_receives_header(self) -> None:
        self.path.parent.mkdir(parents=True)
        self.path.touch()
        logger = AttendanceLogger(self.path)
        self.assertTrue(logger.mark("Alice", when=datetime(2026, 2, 3, 4, 5, 6)))
        self.assertEqual(self.path.read_text(encoding="utf-8").splitlines()[0], "name,date,time")

    def test_attendance_path_cannot_be_directory(self) -> None:
        self.path.mkdir(parents=True)
        logger = AttendanceLogger(self.path)
        with self.assertRaisesRegex(FaceDataError, "not a file"):
            logger.mark("Alice", when=datetime(2026, 1, 1))


if __name__ == "__main__":
    unittest.main()
