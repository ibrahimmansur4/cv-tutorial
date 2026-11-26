from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from src.app import (
    AppConfig,
    build_parser,
    config_from_args,
    parse_size,
    parse_square_size,
    run,
    run_image,
)
from src.decode import ModelFamily, RawDetections, decode
from src.detector import Detection, YoloDetector, count_by_label
from src.drawing import draw_counts, draw_detections
from src.errors import ConfigurationError, DecodeError, MediaError, ModelError
from src.labels import COCO_CLASSES, class_name, color_for, load_labels, resolve_class_filter
from src.letterbox import Letterbox, letterbox
from src.models import REGISTRY, family_for_path, get_spec
from src.pipeline import DetectionPipeline, FpsMeter, RunStats, format_counts

NUM_CLASSES = len(COCO_CLASSES)


@contextlib.contextmanager
def quiet():
    """Swallow the argparse usage text and CLI prints that tests trigger on purpose."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        yield buffer


def v5_rows(boxes: list[tuple[float, float, float, float, float, int]]) -> np.ndarray:
    """Build a (1, N, 85) YOLOv5-style tensor from (cx, cy, w, h, obj, class)."""
    rows = np.zeros((len(boxes), 5 + NUM_CLASSES), dtype=np.float32)
    for index, (cx, cy, width, height, objectness, class_id) in enumerate(boxes):
        rows[index, :4] = (cx, cy, width, height)
        rows[index, 4] = objectness
        rows[index, 5 + class_id] = 1.0
    return rows[np.newaxis, :, :]


def v8_rows(boxes: list[tuple[float, float, float, float, float, int]]) -> np.ndarray:
    """Build a (1, 84, N) YOLOv8-style tensor from (cx, cy, w, h, score, class)."""
    rows = np.zeros((len(boxes), 4 + NUM_CLASSES), dtype=np.float32)
    for index, (cx, cy, width, height, score, class_id) in enumerate(boxes):
        rows[index, :4] = (cx, cy, width, height)
        rows[index, 4 + class_id] = score
    return rows.T[np.newaxis, :, :]


class DecodeTests(unittest.TestCase):
    def test_yolov5_layout_multiplies_objectness_by_class_score(self) -> None:
        outputs = [v5_rows([(100.0, 100.0, 40.0, 60.0, 0.9, 0), (10.0, 10.0, 4.0, 4.0, 0.2, 2)])]
        raw = decode(outputs, ModelFamily.YOLOV5, confidence_threshold=0.5)
        self.assertEqual(len(raw), 1)
        # Top-left corner is centre minus half the extent.
        np.testing.assert_allclose(raw.boxes[0], [80.0, 70.0, 40.0, 60.0])
        self.assertAlmostEqual(float(raw.scores[0]), 0.9, places=5)
        self.assertEqual(int(raw.class_ids[0]), 0)

    def test_yolov8_layout_is_transposed_and_has_no_objectness(self) -> None:
        outputs = [v8_rows([(100.0, 100.0, 40.0, 60.0, 0.77, 5)])]
        raw = decode(outputs, ModelFamily.YOLOV8, confidence_threshold=0.5)
        self.assertEqual(len(raw), 1)
        np.testing.assert_allclose(raw.boxes[0], [80.0, 70.0, 40.0, 60.0])
        self.assertAlmostEqual(float(raw.scores[0]), 0.77, places=5)
        self.assertEqual(int(raw.class_ids[0]), 5)

    def test_both_layouts_agree_on_the_same_boxes(self) -> None:
        boxes = [(50.0, 60.0, 20.0, 30.0, 1.0, 1), (200.0, 180.0, 60.0, 40.0, 1.0, 7)]
        from_v5 = decode([v5_rows(boxes)], ModelFamily.YOLOV5, confidence_threshold=0.5)
        from_v8 = decode([v8_rows(boxes)], ModelFamily.YOLOV8, confidence_threshold=0.5)
        np.testing.assert_allclose(from_v5.boxes, from_v8.boxes)
        np.testing.assert_array_equal(from_v5.class_ids, from_v8.class_ids)

    def test_threshold_filters_and_empty_output_is_handled(self) -> None:
        outputs = [v5_rows([(10.0, 10.0, 4.0, 4.0, 0.1, 0)])]
        self.assertEqual(len(decode(outputs, ModelFamily.YOLOV5, confidence_threshold=0.5)), 0)
        empty = decode([np.zeros((1, 0, 85), dtype=np.float32)], ModelFamily.YOLOV5)
        self.assertEqual(len(empty), 0)

    def test_multiple_output_tensors_are_concatenated(self) -> None:
        first = v5_rows([(100.0, 100.0, 40.0, 60.0, 0.9, 0)])
        second = v5_rows([(300.0, 300.0, 20.0, 20.0, 0.8, 1)])
        raw = decode([first, second], ModelFamily.YOLOV5, confidence_threshold=0.5)
        self.assertEqual(len(raw), 2)

    def test_wrong_family_on_a_narrow_output_is_rejected(self) -> None:
        narrow = np.zeros((1, 3, 4), dtype=np.float32)
        with self.assertRaises(DecodeError):
            decode([narrow], ModelFamily.YOLOV5)
        with self.assertRaises(DecodeError):
            decode([], ModelFamily.YOLOV5)
        with self.assertRaises(DecodeError):
            decode([v5_rows([])], ModelFamily.YOLOV5, confidence_threshold=1.5)

    def test_disagreeing_output_widths_are_rejected(self) -> None:
        with self.assertRaises(DecodeError):
            decode(
                [np.zeros((1, 2, 85), np.float32), np.zeros((1, 2, 84), np.float32)],
                ModelFamily.YOLOV5,
            )

    def test_raw_detections_validates_its_own_shapes(self) -> None:
        with self.assertRaises(DecodeError):
            RawDetections(
                np.zeros((2, 4), np.float32), np.zeros(1, np.float32), np.zeros(1, np.int32)
            )

    def test_family_properties_describe_the_layout(self) -> None:
        self.assertTrue(ModelFamily.YOLOV5.has_objectness)
        self.assertFalse(ModelFamily.YOLOV5.transposed_output)
        self.assertFalse(ModelFamily.YOLOV8.has_objectness)
        self.assertTrue(ModelFamily.YOLOV8.transposed_output)


class LetterboxTests(unittest.TestCase):
    def test_aspect_ratio_is_preserved_and_padding_is_centred(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        padded, transform = letterbox(frame, (640, 640))
        self.assertEqual(padded.shape[:2], (640, 640))
        self.assertAlmostEqual(transform.scale, 1.0)
        # A 4:3 frame in a square input pads 80 rows top and bottom.
        self.assertAlmostEqual(transform.pad_y, 80.0)
        self.assertAlmostEqual(transform.pad_x, 0.0)

    def test_inverse_transform_recovers_the_original_box(self) -> None:
        frame = np.zeros((360, 1280, 3), dtype=np.uint8)
        _, transform = letterbox(frame, (640, 640))
        # A box covering the whole frame, expressed in network input space.
        width = 1280 * transform.scale
        height = 360 * transform.scale
        boxes = np.array([[transform.pad_x, transform.pad_y, width, height]], dtype=np.float32)
        corners = transform.to_source(boxes)
        np.testing.assert_allclose(corners[0], [0.0, 0.0, 1279.0, 359.0], atol=1.0)

    def test_boxes_are_clipped_to_the_frame(self) -> None:
        transform = Letterbox(scale=1.0, pad_x=0.0, pad_y=0.0, source_size=(100, 100))
        corners = transform.to_source(np.array([[-50.0, -50.0, 400.0, 400.0]], dtype=np.float32))
        np.testing.assert_allclose(corners[0], [0.0, 0.0, 99.0, 99.0])
        self.assertEqual(transform.to_source(np.zeros((0, 4), np.float32)).shape, (0, 4))

    def test_padding_uses_the_neutral_fill_value(self) -> None:
        frame = np.zeros((100, 400, 3), dtype=np.uint8)
        padded, _ = letterbox(frame, (640, 640))
        self.assertEqual(int(padded[0, 0, 0]), 114)

    def test_invalid_input_is_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            letterbox(np.zeros((10, 10, 3), np.uint8), (0, 640))
        with self.assertRaises(ConfigurationError):
            letterbox(np.zeros((0, 0, 3), np.uint8), (640, 640))
        with self.assertRaises(ConfigurationError):
            letterbox(np.zeros((10, 10), np.uint8), (640, 640))


class LabelTests(unittest.TestCase):
    def test_coco_list_is_the_expected_80_classes(self) -> None:
        self.assertEqual(len(COCO_CLASSES), 80)
        self.assertEqual(COCO_CLASSES[0], "person")
        self.assertEqual(class_name(COCO_CLASSES, 0), "person")
        self.assertEqual(class_name(COCO_CLASSES, 999), "class_999")

    def test_class_filter_accepts_names_and_indices(self) -> None:
        self.assertEqual(resolve_class_filter(["person"], COCO_CLASSES), frozenset({0}))
        self.assertEqual(resolve_class_filter(["PERSON"], COCO_CLASSES), frozenset({0}))
        self.assertEqual(resolve_class_filter(["0", "2"], COCO_CLASSES), frozenset({0, 2}))
        self.assertIsNone(resolve_class_filter(None, COCO_CLASSES))

    def test_unknown_or_empty_class_filter_is_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            resolve_class_filter(["dragon"], COCO_CLASSES)
        with self.assertRaises(ConfigurationError):
            resolve_class_filter(["999"], COCO_CLASSES)
        with self.assertRaises(ConfigurationError):
            resolve_class_filter([" "], COCO_CLASSES)

    def test_colors_are_stable_and_distinct(self) -> None:
        self.assertEqual(color_for(3), color_for(3))
        self.assertNotEqual(color_for(0), color_for(1))
        for channel in color_for(17):
            self.assertTrue(0 <= channel <= 255)

    def test_labels_file_is_read_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "names.txt"
            path.write_text("cat\n\ndog\n", encoding="utf-8")
            self.assertEqual(load_labels(path), ("cat", "dog"))
            empty = Path(directory) / "empty.txt"
            empty.write_text("\n\n", encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                load_labels(empty)
            with self.assertRaises(ConfigurationError):
                load_labels(Path(directory) / "missing.txt")


class FakeNet:
    """A stand-in for cv2.dnn.Net that replays a fixed output tensor."""

    def __init__(self, outputs: list[np.ndarray]) -> None:
        self._outputs = outputs
        self.blobs: list[np.ndarray] = []

    def empty(self) -> bool:
        return False

    def getUnconnectedOutLayersNames(self) -> list[str]:
        return ["output0"]

    def setInput(self, blob: np.ndarray) -> None:
        self.blobs.append(blob)

    def forward(self, names: list[str]) -> list[np.ndarray]:
        return self._outputs


def detector_with(outputs: list[np.ndarray], **kwargs) -> YoloDetector:
    return YoloDetector(
        Path("unused.onnx"), kwargs.pop("family", ModelFamily.YOLOV5), net=FakeNet(outputs), **kwargs
    )


class DetectorTests(unittest.TestCase):
    def test_detections_are_mapped_back_into_frame_pixels(self) -> None:
        # One box centred in a 640x640 network input, from a 640x480 frame.
        outputs = [v5_rows([(320.0, 320.0, 64.0, 64.0, 0.9, 0)])]
        detector = detector_with(outputs, confidence_threshold=0.5)
        result = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertEqual(len(result.detections), 1)
        detection = result.detections[0]
        self.assertEqual(detection.label, "person")
        # Frame centre is (320, 240); the 64 px box is unscaled at scale 1.0.
        self.assertAlmostEqual(detection.x1, 288.0, delta=1.0)
        self.assertAlmostEqual(detection.y1, 208.0, delta=1.0)
        self.assertAlmostEqual(detection.width, 64.0, delta=1.0)

    def test_overlapping_boxes_of_one_class_are_suppressed(self) -> None:
        outputs = [
            v5_rows(
                [
                    (320.0, 320.0, 100.0, 100.0, 0.9, 0),
                    (322.0, 322.0, 100.0, 100.0, 0.7, 0),
                ]
            )
        ]
        detector = detector_with(outputs, confidence_threshold=0.5)
        result = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))
        self.assertEqual(len(result.detections), 1)
        self.assertAlmostEqual(result.detections[0].confidence, 0.9, places=5)

    def test_overlapping_boxes_of_different_classes_both_survive(self) -> None:
        outputs = [
            v5_rows(
                [
                    (320.0, 320.0, 100.0, 100.0, 0.9, 0),
                    (322.0, 322.0, 100.0, 100.0, 0.8, 27),
                ]
            )
        ]
        detector = detector_with(outputs, confidence_threshold=0.5)
        result = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))
        self.assertEqual(len(result.detections), 2)
        self.assertEqual({d.label for d in result.detections}, {"person", "tie"})

    def test_class_filter_drops_unwanted_classes(self) -> None:
        outputs = [
            v5_rows(
                [
                    (100.0, 100.0, 40.0, 40.0, 0.9, 0),
                    (400.0, 400.0, 40.0, 40.0, 0.9, 2),
                ]
            )
        ]
        detector = detector_with(outputs, confidence_threshold=0.5, class_filter={2})
        result = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))
        self.assertEqual([d.label for d in result.detections], ["car"])

        nothing = detector_with(outputs, confidence_threshold=0.5, class_filter={79})
        self.assertEqual(nothing.detect(np.zeros((640, 640, 3), np.uint8)).detections, ())

    def test_results_are_ordered_by_descending_confidence(self) -> None:
        outputs = [
            v5_rows(
                [
                    (100.0, 100.0, 40.0, 40.0, 0.6, 0),
                    (400.0, 400.0, 40.0, 40.0, 0.95, 2),
                ]
            )
        ]
        detector = detector_with(outputs, confidence_threshold=0.5)
        confidences = [d.confidence for d in detector.detect(np.zeros((640, 640, 3), np.uint8)).detections]
        self.assertEqual(confidences, sorted(confidences, reverse=True))

    def test_blob_is_built_at_the_configured_input_size(self) -> None:
        net = FakeNet([v5_rows([])])
        detector = YoloDetector(
            Path("unused.onnx"), ModelFamily.YOLOV5, net=net, input_size=(320, 320)
        )
        detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertEqual(net.blobs[0].shape, (1, 3, 320, 320))

    def test_timing_fields_are_populated(self) -> None:
        detector = detector_with([v5_rows([(320.0, 320.0, 40.0, 40.0, 0.9, 0)])])
        result = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))
        self.assertGreaterEqual(result.preprocess_ms, 0.0)
        self.assertAlmostEqual(
            result.total_ms,
            result.preprocess_ms + result.inference_ms + result.postprocess_ms,
            places=6,
        )

    def test_invalid_detector_options_are_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            detector_with([], confidence_threshold=1.5)
        with self.assertRaises(ConfigurationError):
            detector_with([], nms_threshold=0.0)
        with self.assertRaises(ConfigurationError):
            detector_with([], input_size=(100, 100))
        with self.assertRaises(ConfigurationError):
            detector_with([], labels=())

    def test_missing_or_non_onnx_model_paths_are_reported_clearly(self) -> None:
        with self.assertRaises(ConfigurationError) as missing:
            YoloDetector(Path("does-not-exist.onnx"), ModelFamily.YOLOV5)
        self.assertIn("download_model.py", str(missing.exception))
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "yolov4.weights"
            weights.write_bytes(b"not a model")
            with self.assertRaises(ConfigurationError) as wrong_format:
                YoloDetector(weights, ModelFamily.YOLOV5)
            self.assertIn("Darknet", str(wrong_format.exception))

    def test_forward_failures_surface_as_model_errors(self) -> None:
        class BrokenNet(FakeNet):
            def forward(self, names: list[str]) -> list[np.ndarray]:
                raise RuntimeError("no kernel")

        detector = YoloDetector(Path("unused.onnx"), ModelFamily.YOLOV5, net=BrokenNet([]))
        with self.assertRaises(ModelError):
            detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))

    def test_counting_groups_by_label(self) -> None:
        detections = [
            Detection(0, 0, 1, 1, 0.9, 0, "person"),
            Detection(0, 0, 1, 1, 0.8, 0, "person"),
            Detection(0, 0, 1, 1, 0.7, 2, "car"),
        ]
        self.assertEqual(count_by_label(detections), {"person": 2, "car": 1})
        self.assertEqual(count_by_label([]), {})


class PipelineTests(unittest.TestCase):
    def test_pipeline_produces_counts_and_an_annotated_copy(self) -> None:
        detector = detector_with([v5_rows([(320.0, 320.0, 80.0, 80.0, 0.9, 0)])])
        pipeline = DetectionPipeline(detector, model_label="model: test", show_timing=True)
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        result = pipeline.process(frame, 0.02)
        self.assertEqual(result.counts, {"person": 1})
        annotated = pipeline.annotate(frame, result)
        self.assertEqual(annotated.shape, frame.shape)
        # The source frame must not be modified in place.
        self.assertEqual(int(frame.max()), 0)
        self.assertGreater(int(annotated.max()), 0)

    def test_fps_meter_averages_over_its_window(self) -> None:
        meter = FpsMeter(window_size=2)
        self.assertEqual(meter.update(0.0), 0.0)
        meter.update(0.1)
        self.assertAlmostEqual(meter.update(0.1), 10.0, places=5)
        with self.assertRaises(ValueError):
            FpsMeter(0)

    def test_run_stats_summarize_a_run(self) -> None:
        stats = RunStats()
        self.assertIn("No frames", stats.summary())
        detector = detector_with([v5_rows([(320.0, 320.0, 80.0, 80.0, 0.9, 0)])])
        pipeline = DetectionPipeline(detector)
        result = pipeline.process(np.zeros((640, 640, 3), dtype=np.uint8), 0.05)
        stats.observe(result)
        stats.elapsed_seconds = 0.5
        self.assertEqual(stats.frames, 1)
        self.assertEqual(stats.peak_objects, 1)
        self.assertAlmostEqual(stats.fps, 2.0)
        self.assertIn("person=1", stats.summary())

    def test_count_block_matches_the_documented_format(self) -> None:
        text = format_counts({"person": 3, "car": 1}, 24.6)
        self.assertEqual(text, "person: 3\ncar:    1\n\nFPS: 24.6")
        self.assertIn("No objects", format_counts({}, 0.0))

    def test_drawing_handles_edge_cases_without_raising(self) -> None:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        # A box touching the top edge forces the caption inside the box.
        draw_detections(frame, [Detection(0.0, 0.0, 50.0, 40.0, 0.9, 0, "person")])
        draw_counts(frame, {f"class{i}": 1 for i in range(12)}, 12.0, extra_lines=["x"])
        draw_counts(frame, {}, 0.0)
        self.assertGreater(int(frame.max()), 0)


class ModelRegistryTests(unittest.TestCase):
    def test_registry_entries_are_consistent(self) -> None:
        for name, spec in REGISTRY.items():
            self.assertEqual(name, spec.name)
            self.assertEqual(len(spec.sha256), 64)
            self.assertTrue(spec.url.startswith("https://"))
            self.assertGreater(spec.size_bytes, 0)
            self.assertEqual(spec.filename, f"{name}.onnx")

    def test_unknown_model_names_are_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            get_spec("yolov99")

    def test_family_is_inferred_from_a_file_name(self) -> None:
        self.assertIs(family_for_path(Path("yolov5s.onnx")), ModelFamily.YOLOV5)
        self.assertIs(family_for_path(Path("yolo11n.onnx")), ModelFamily.YOLOV8)
        self.assertIs(family_for_path(Path("custom.onnx")), ModelFamily.YOLOV8)


class CommandLineTests(unittest.TestCase):
    def parse(self, argv: list[str]) -> AppConfig:
        return config_from_args(build_parser().parse_args(argv))

    def test_defaults_select_the_registered_default_model(self) -> None:
        config = self.parse([])
        self.assertEqual(config.camera, 0)
        self.assertEqual(config.confidence, 0.25)
        self.assertIs(config.family, ModelFamily.YOLOV5)
        self.assertTrue(config.mirror)
        self.assertTrue(config.display)

    def test_size_parsing_accepts_both_forms(self) -> None:
        self.assertEqual(parse_size("1280x720"), (1280, 720))
        self.assertEqual(parse_square_size("320"), (320, 320))
        self.assertEqual(parse_square_size("640x640"), (640, 640))
        import argparse

        for bad in ("1280", "x", "0x10", "axb"):
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_size(bad)
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_square_size("-1")

    def test_recorded_input_defaults_to_no_mirroring(self) -> None:
        self.assertFalse(self.parse(["--input", "clip.mp4"]).mirror)
        self.assertTrue(self.parse(["--input", "clip.mp4", "--mirror"]).mirror)

    def test_family_override_and_unknown_model_are_handled(self) -> None:
        config = self.parse(["--model", "yolov5n", "--family", "yolov8"])
        self.assertIs(config.family, ModelFamily.YOLOV8)
        with self.assertRaises(ConfigurationError) as unknown:
            self.parse(["--model", "nope"])
        self.assertIn("Registered models", str(unknown.exception))
        with self.assertRaises(ConfigurationError):
            self.parse(["--model", "missing.onnx"])

    def test_invalid_options_are_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            self.parse(["--camera", "-1"])
        with self.assertRaises(ConfigurationError):
            self.parse(["--input-size", "100x100"])
        import argparse

        for bad in (["--confidence", "2"], ["--nms", "-1"], ["--max-frames", "-3"]):
            with self.assertRaises(SystemExit), quiet():
                build_parser().parse_args(bad)
        with self.assertRaises(argparse.ArgumentTypeError):
            from src.app import class_list

            class_list(" , ")

    def test_classes_are_split_on_commas(self) -> None:
        self.assertEqual(self.parse(["--classes", "person, car"]).classes, ("person", "car"))


def config_for(**overrides) -> AppConfig:
    base = dict(
        camera=0,
        input_path=None,
        output_path=None,
        model_path=Path("unused.onnx"),
        family=ModelFamily.YOLOV5,
        model_label="model: test",
        labels_path=None,
        input_size=(640, 640),
        resolution=(640, 480),
        confidence=0.5,
        nms=0.45,
        classes=None,
        max_frames=0,
        display=False,
        mirror=False,
        show_timing=False,
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
    """Wraps the real cv2 but replaces capture, display, and writing."""

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


class MediaLoopTests(unittest.TestCase):
    def frames(self, count: int) -> list[np.ndarray]:
        return [np.full((480, 640, 3), 30, dtype=np.uint8) for _ in range(count)]

    def test_video_loop_processes_every_frame(self) -> None:
        fake = FakeCv2(self.frames(4))
        detector = detector_with([v5_rows([(320.0, 320.0, 80.0, 80.0, 0.9, 0)])])
        stats = run(config_for(), cv2_module=fake, detector=detector)
        self.assertEqual(stats.frames, 4)
        self.assertEqual(stats.detections, 4)
        self.assertTrue(fake.capture.released)

    def test_max_frames_stops_the_loop_early(self) -> None:
        fake = FakeCv2(self.frames(10))
        detector = detector_with([v5_rows([])])
        stats = run(config_for(max_frames=3), cv2_module=fake, detector=detector)
        self.assertEqual(stats.frames, 3)

    def test_a_source_with_no_frames_is_an_error(self) -> None:
        fake = FakeCv2([])
        detector = detector_with([v5_rows([])])
        with self.assertRaises(MediaError):
            run(config_for(), cv2_module=fake, detector=detector)

    def test_unopenable_source_is_reported(self) -> None:
        class ClosedCapture(FakeCapture):
            def isOpened(self) -> bool:
                return False

        fake = FakeCv2([])
        fake.capture = ClosedCapture([])
        with self.assertRaises(MediaError):
            run(config_for(), cv2_module=fake, detector=detector_with([v5_rows([])]))

    def test_missing_input_file_is_reported(self) -> None:
        fake = FakeCv2(self.frames(1))
        config = config_for(input_path=Path("no-such-clip.mp4"))
        with self.assertRaises(MediaError):
            run(config, cv2_module=fake, detector=detector_with([v5_rows([])]))

    def test_annotated_video_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out.avi"
            fake = FakeCv2(self.frames(3))
            detector = detector_with([v5_rows([(320.0, 320.0, 80.0, 80.0, 0.9, 0)])])
            stats = run(config_for(output_path=output), cv2_module=fake, detector=detector)
            self.assertEqual(stats.frames, 3)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

    def test_unsupported_video_suffix_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake = FakeCv2(self.frames(1))
            config = config_for(output_path=Path(directory) / "out.gif")
            with self.assertRaises(MediaError):
                run(config, cv2_module=fake, detector=detector_with([v5_rows([])]))

    def test_image_mode_reads_writes_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "frame.png"
            output = Path(directory) / "annotated.png"
            cv2.imwrite(str(source), np.full((480, 640, 3), 40, dtype=np.uint8))
            detector = detector_with([v5_rows([(320.0, 320.0, 80.0, 80.0, 0.9, 0)])])
            config = config_for(input_path=source, output_path=output)
            with quiet():
                stats = run_image(config, detector=detector)
            self.assertEqual(stats.frames, 1)
            self.assertEqual(stats.detections, 1)
            self.assertTrue(output.is_file())

    def test_image_mode_rejects_a_video_output_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "frame.png"
            cv2.imwrite(str(source), np.full((64, 64, 3), 40, dtype=np.uint8))
            config = config_for(input_path=source, output_path=Path(directory) / "out.avi")
            with self.assertRaises(MediaError):
                run_image(config, detector=detector_with([v5_rows([])]))

    def test_unreadable_image_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            broken = Path(directory) / "broken.png"
            broken.write_bytes(b"not an image")
            with self.assertRaises(MediaError):
                run_image(
                    config_for(input_path=broken), detector=detector_with([v5_rows([])])
                )

    def test_input_and_output_must_differ(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "frame.png"
            cv2.imwrite(str(source), np.full((64, 64, 3), 40, dtype=np.uint8))
            config = config_for(input_path=source, output_path=source)
            with self.assertRaises(MediaError):
                run_image(config, detector=detector_with([v5_rows([])]))


if __name__ == "__main__":
    unittest.main()
