from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.app import AppConfig, build_parser, config_from_args, parse_size, run, run_image
from src.cells import CANVAS, GRID, clean_cell, normalize_digit, remove_grid_lines, split_cells
from src.classify import (
    Prediction,
    TemplateClassifier,
    build_templates,
    classify_cells,
    make_classifier,
)
from src.errors import (
    BoardNotFoundError,
    ConfigurationError,
    MediaError,
    RecognitionError,
    UnsolvableError,
)
from src.grid import find_board, is_plausible_board, order_corners, preprocess
from src.pipeline import SudokuPipeline
from src.render import cell_mosaic, draw_recognition_debug, overlay_solution
from src.solver import (
    count_solutions,
    find_conflicts,
    format_board,
    is_complete,
    parse_board,
    solve,
)

EASY = (
    "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
)
HARD = (
    "100007090030020008009600500005300900010080002600004000300000010040000007007000300"
)


def board_from(text: str) -> list[list[int]]:
    digits = [0 if character in ".0" else int(character) for character in text]
    return [digits[index * 9 : (index + 1) * 9] for index in range(9)]


@contextlib.contextmanager
def quiet():
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        yield buffer


def render_puzzle(
    board, *, size: int = 540, margin: int = 30, font: int = cv2.FONT_HERSHEY_SIMPLEX
) -> np.ndarray:
    """A clean head-on render, mirroring scripts/make_puzzle_image.py."""
    side = size + margin * 2
    image = np.full((side, side, 3), 252, dtype=np.uint8)
    step = size / GRID
    for index in range(GRID + 1):
        position = int(round(margin + index * step))
        thickness = 3 if index % 3 == 0 else 1
        cv2.line(image, (margin, position), (margin + size, position), (30, 30, 30), thickness)
        cv2.line(image, (position, margin), (position, margin + size), (30, 30, 30), thickness)
    scale = step / 42.0
    thickness = max(1, int(round(step / 22.0)))
    for row in range(GRID):
        for column in range(GRID):
            value = board[row][column]
            if not value:
                continue
            text = str(value)
            (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
            x = int(round(margin + (column + 0.5) * step - text_width / 2))
            y = int(round(margin + (row + 0.5) * step + text_height / 2))
            cv2.putText(image, text, (x, y), font, scale, (20, 20, 20), thickness, cv2.LINE_AA)
    return image


class SolverTests(unittest.TestCase):
    def assert_valid(self, board) -> None:
        for row in board:
            self.assertEqual(sorted(row), list(range(1, 10)))
        for column in zip(*board):
            self.assertEqual(sorted(column), list(range(1, 10)))
        for row in (0, 3, 6):
            for column in (0, 3, 6):
                box = [board[row + i][column + j] for i in range(3) for j in range(3)]
                self.assertEqual(sorted(box), list(range(1, 10)))

    def test_an_easy_puzzle_solves_and_keeps_its_givens(self) -> None:
        givens = board_from(EASY)
        solution, stats = solve(givens)
        self.assert_valid(solution)
        self.assertGreater(stats.assignments, 0)
        for row in range(9):
            for column in range(9):
                if givens[row][column]:
                    self.assertEqual(solution[row][column], givens[row][column])

    def test_a_hard_puzzle_still_solves(self) -> None:
        self.assert_valid(solve(board_from(HARD))[0])

    def test_an_empty_grid_solves(self) -> None:
        self.assert_valid(solve([[0] * 9 for _ in range(9)])[0])

    def test_a_solved_grid_is_returned_unchanged(self) -> None:
        solution, stats = solve(board_from(EASY))
        again, again_stats = solve(solution)
        self.assertEqual(again, solution)
        self.assertEqual(again_stats.assignments, 0)

    def test_conflicting_givens_are_rejected(self) -> None:
        board = board_from(EASY)
        board[0][2] = 5  # duplicates the 5 already in row 0
        with self.assertRaises(UnsolvableError):
            solve(board)

    def test_an_unsolvable_puzzle_is_reported(self) -> None:
        # Valid givens, but no completion exists.
        board = [[0] * 9 for _ in range(9)]
        board[0][:8] = [1, 2, 3, 4, 5, 6, 7, 8]
        board[1][8] = 9
        with self.assertRaises(UnsolvableError):
            solve(board)

    def test_the_assignment_limit_is_enforced(self) -> None:
        with self.assertRaises(UnsolvableError):
            solve([[0] * 9 for _ in range(9)], limit=1)

    def test_uniqueness_counting(self) -> None:
        self.assertEqual(count_solutions(board_from(EASY)), 1)
        self.assertEqual(count_solutions([[0] * 9 for _ in range(9)], cap=2), 2)
        with self.assertRaises(ValueError):
            count_solutions(board_from(EASY), cap=0)

    def test_removing_a_given_can_make_a_puzzle_ambiguous(self) -> None:
        """Why a misread digit matters: it can silently create a second answer."""
        sparse = [[0] * 9 for _ in range(9)]
        sparse[0][0] = 1
        self.assertGreaterEqual(count_solutions(sparse, cap=2), 2)

    def test_conflict_detection_locates_duplicates(self) -> None:
        board = [[0] * 9 for _ in range(9)]
        board[0][0] = board[0][5] = 7
        self.assertIn((0, 0), find_conflicts(board))
        board = [[0] * 9 for _ in range(9)]
        board[0][0] = board[2][2] = 4  # same box
        self.assertIn((0, 0), find_conflicts(board))
        self.assertEqual(find_conflicts(board_from(EASY)), [])

    def test_board_parsing_validates_shape_and_values(self) -> None:
        with self.assertRaises(ValueError):
            parse_board([[0] * 9] * 8)
        with self.assertRaises(ValueError):
            parse_board([[0] * 8] + [[0] * 9] * 8)
        with self.assertRaises(ValueError):
            parse_board([[10] + [0] * 8] + [[0] * 9] * 8)

    def test_formatting_and_completeness(self) -> None:
        text = format_board(board_from(EASY))
        self.assertIn("------+-------+------", text)
        self.assertIn(".", text)
        self.assertFalse(is_complete(board_from(EASY)))
        self.assertTrue(is_complete(solve(board_from(EASY))[0]))


class GeometryTests(unittest.TestCase):
    def test_corners_are_ordered_clockwise_from_the_top_left(self) -> None:
        scrambled = np.array([[10, 90], [90, 90], [90, 10], [10, 10]], dtype=np.float32)
        ordered = order_corners(scrambled)
        np.testing.assert_allclose(ordered[0], [10, 10])
        np.testing.assert_allclose(ordered[1], [90, 10])
        np.testing.assert_allclose(ordered[2], [90, 90])
        np.testing.assert_allclose(ordered[3], [10, 90])

    def test_ordering_survives_a_realistic_rotation(self) -> None:
        # A square rotated about 20 degrees, which is more tilt than anyone
        # photographs a puzzle with.
        rotated = np.array(
            [[67.7, 88.1], [11.9, 67.7], [32.3, 11.9], [88.1, 32.3]], dtype=np.float32
        )
        ordered = order_corners(rotated)
        self.assertEqual(len({tuple(point) for point in ordered}), 4)
        np.testing.assert_allclose(ordered[0], [32.3, 11.9], rtol=1e-3)  # top-left
        np.testing.assert_allclose(ordered[2], [67.7, 88.1], rtol=1e-3)  # bottom-right

    def test_a_45_degree_diamond_is_rejected_rather_than_mis_ordered(self) -> None:
        """The documented limit of the sum/difference method.

        At exactly 45 degrees the sums and differences tie, so top-left and
        top-right resolve to the same point. Detecting the collision and
        refusing is right: silently returning a collapsed quad would produce a
        homography that warps the board into nonsense.
        """
        diamond = np.array([[50, 8], [92, 50], [50, 92], [8, 50]], dtype=np.float32)
        with self.assertRaises(BoardNotFoundError):
            order_corners(diamond)

    def test_ordering_rejects_the_wrong_number_of_points(self) -> None:
        with self.assertRaises(ConfigurationError):
            order_corners(np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32))

    def test_degenerate_quads_are_rejected(self) -> None:
        with self.assertRaises(BoardNotFoundError):
            order_corners(np.zeros((4, 2), dtype=np.float32))

    def test_plausibility_rejects_tiny_and_lopsided_quads(self) -> None:
        square = np.array([[0, 0], [200, 0], [200, 200], [0, 200]], dtype=np.float32)
        self.assertTrue(is_plausible_board(square))
        tiny = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=np.float32)
        self.assertFalse(is_plausible_board(tiny))
        wide = np.array([[0, 0], [400, 0], [400, 60], [0, 60]], dtype=np.float32)
        self.assertFalse(is_plausible_board(wide))


class BoardDetectionTests(unittest.TestCase):
    def test_a_rendered_board_is_found_and_flattened(self) -> None:
        image = render_puzzle(board_from(EASY))
        location = find_board(image)
        self.assertEqual(location.size, 450)
        warped = location.warp(image)
        self.assertEqual(warped.shape[:2], (450, 450))
        self.assertGreater(location.area, 100_000)

    def test_a_tilted_board_is_found(self) -> None:
        image = render_puzzle(board_from(EASY))
        height, width = image.shape[:2]
        source = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        destination = source + np.array(
            [[25, 12], [-18, 20], [-10, -22], [16, -14]], dtype=np.float32
        )
        warped = cv2.warpPerspective(
            image,
            cv2.getPerspectiveTransform(source, destination),
            (width, height),
            borderValue=(240, 238, 235),
        )
        self.assertIsNotNone(find_board(warped))

    def test_a_frame_with_no_grid_is_reported(self) -> None:
        with self.assertRaises(BoardNotFoundError):
            find_board(np.full((400, 400, 3), 250, dtype=np.uint8))

    def test_the_inverse_transform_round_trips(self) -> None:
        image = render_puzzle(board_from(EASY))
        location = find_board(image)
        marker = np.zeros((location.size, location.size, 3), dtype=np.uint8)
        cv2.rectangle(marker, (10, 10), (120, 120), (0, 255, 0), -1)
        projected = location.unwarp_onto(marker, image.shape)
        self.assertEqual(projected.shape, image.shape)
        self.assertGreater(int(projected.max()), 0)

    def test_preprocess_validates_its_input(self) -> None:
        for kwargs in ({"blur_kernel": 4}, {"block_size": 2}):
            with self.assertRaises(ConfigurationError):
                preprocess(np.zeros((50, 50, 3), np.uint8), **kwargs)
        with self.assertRaises(ConfigurationError):
            preprocess(np.zeros((0, 0, 3), np.uint8))
        with self.assertRaises(ConfigurationError):
            preprocess(np.zeros((4, 4, 2), np.uint8))

    def test_a_too_small_board_size_is_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            find_board(render_puzzle(board_from(EASY)), size=50)


class CellTests(unittest.TestCase):
    def test_a_board_splits_into_81_cells(self) -> None:
        image = render_puzzle(board_from(EASY))
        location = find_board(image)
        square = preprocess(location.warp(image), block_size=19, constant=4)
        cells = split_cells(square)
        self.assertEqual(len(cells), 81)
        self.assertEqual({(cell.row, cell.column) for cell in cells}, {
            (row, column) for row in range(9) for column in range(9)
        })

    def test_occupancy_matches_the_rendered_givens(self) -> None:
        givens = board_from(EASY)
        image = render_puzzle(givens)
        location = find_board(image)
        square = preprocess(location.warp(image), block_size=19, constant=4)
        for cell in split_cells(square):
            self.assertEqual(
                cell.occupied,
                bool(givens[cell.row][cell.column]),
                msg=f"cell r{cell.row}c{cell.column}",
            )

    def test_a_digit_touching_a_grid_line_is_not_erased(self) -> None:
        """The regression test for the flood-fill bug.

        Filling from the border removes the whole connected component, so a
        digit touching a leftover line vanished and the cell read as empty.
        """
        cell = np.zeros((50, 50), dtype=np.uint8)
        cell[:, 0:3] = 255  # a leftover vertical grid line
        cv2.putText(cell, "5", (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 255, 2)
        cell[20:24, 0:16] = 255  # the digit now touches the line
        _, ink_ratio, occupied = clean_cell(cell)
        self.assertTrue(occupied)
        self.assertGreater(ink_ratio, 0.0)

    def test_grid_line_removal_keeps_a_digit(self) -> None:
        inner = np.zeros((40, 40), dtype=np.uint8)
        inner[:, 0:2] = 255
        inner[0:2, :] = 255
        cv2.putText(inner, "7", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
        before = int(cv2.countNonZero(inner))
        cleaned = remove_grid_lines(inner)
        after = int(cv2.countNonZero(cleaned))
        self.assertGreater(after, 0)
        self.assertLess(after, before)

    def test_an_empty_cell_reads_as_empty(self) -> None:
        _, _, occupied = clean_cell(np.zeros((50, 50), dtype=np.uint8))
        self.assertFalse(occupied)

    def test_a_cell_of_pure_noise_is_not_a_digit(self) -> None:
        speck = np.zeros((50, 50), dtype=np.uint8)
        speck[24:26, 24:26] = 255  # too small to be a digit
        self.assertFalse(clean_cell(speck)[2])

    def test_normalisation_centres_and_scales(self) -> None:
        mask = np.zeros((60, 60), dtype=np.uint8)
        cv2.rectangle(mask, (4, 4), (12, 40), 255, -1)  # tall, off-centre
        patch = normalize_digit(mask)
        self.assertEqual(patch.shape, (CANVAS, CANVAS))
        points = cv2.findNonZero(patch)
        x, y, width, height = cv2.boundingRect(points)
        self.assertLessEqual(max(width, height), 21)
        self.assertLess(abs((x + width / 2) - CANVAS / 2), 3)

    def test_normalising_nothing_gives_a_blank_canvas(self) -> None:
        self.assertEqual(int(normalize_digit(np.zeros((40, 40), np.uint8)).max()), 0)

    def test_malformed_input_is_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            clean_cell(np.zeros((10, 10, 3), np.uint8))
        with self.assertRaises(ConfigurationError):
            split_cells(np.zeros((10, 10, 3), np.uint8))
        with self.assertRaises(ConfigurationError):
            split_cells(np.zeros((4, 4), np.uint8))


class ClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = TemplateClassifier()

    def test_templates_cover_every_digit(self) -> None:
        vectors, labels = build_templates()
        self.assertEqual(vectors.shape[0], labels.shape[0])
        self.assertEqual(set(labels.tolist()), set(range(1, 10)))

    def test_every_digit_is_recognised_in_several_fonts(self) -> None:
        for font in (
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_PLAIN,
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
        ):
            for digit in range(1, 10):
                canvas = np.zeros((64, 64), dtype=np.uint8)
                text = str(digit)
                scale = 1.4 if font != cv2.FONT_HERSHEY_PLAIN else 3.0
                (width, height), _ = cv2.getTextSize(text, font, scale, 2)
                cv2.putText(
                    canvas,
                    text,
                    ((64 - width) // 2, (64 + height) // 2),
                    font,
                    scale,
                    255,
                    2,
                    cv2.LINE_AA,
                )
                patch = normalize_digit(canvas)
                self.assertEqual(
                    self.classifier.predict(patch).digit,
                    digit,
                    msg=f"font {font}, digit {digit}",
                )

    def test_a_blank_patch_predicts_nothing(self) -> None:
        prediction = self.classifier.predict(np.zeros((CANVAS, CANVAS), np.uint8))
        self.assertTrue(prediction.is_blank)
        self.assertEqual(prediction.confidence, 0.0)

    def test_a_wrong_sized_patch_is_rejected(self) -> None:
        with self.assertRaises(RecognitionError):
            self.classifier.predict(np.zeros((10, 10), np.uint8))

    def test_the_factory_builds_known_classifiers(self) -> None:
        self.assertEqual(make_classifier("template").name, "template")
        with self.assertRaises(ConfigurationError):
            make_classifier("nonsense")
        with self.assertRaises(ConfigurationError):
            make_classifier("onnx")

    def test_onnx_never_predicts_zero(self) -> None:
        """Sudoku has no zero, so index 0 must be excluded from the argmax."""

        class FakeNet:
            def empty(self):
                return False

            def setInput(self, blob):
                self.blob = blob

            def forward(self):
                scores = np.full(10, -5.0, dtype=np.float32)
                scores[0] = 50.0  # the model is sure it is a zero
                scores[7] = 10.0
                return scores.reshape(1, 10)

        from src.classify import OnnxDigitClassifier

        classifier = OnnxDigitClassifier(Path("unused.onnx"), net=FakeNet())
        patch = np.zeros((CANVAS, CANVAS), np.uint8)
        patch[10:18, 10:18] = 255
        self.assertEqual(classifier.predict(patch).digit, 7)

    def test_classify_cells_builds_a_board(self) -> None:
        givens = board_from(EASY)
        image = render_puzzle(givens)
        location = find_board(image)
        square = preprocess(location.warp(image), block_size=19, constant=4)
        board, confidence = classify_cells(split_cells(square), self.classifier)
        self.assertEqual(board, givens)
        self.assertEqual(len(confidence), 9)


class PipelineTests(unittest.TestCase):
    def test_a_rendered_puzzle_is_read_and_solved(self) -> None:
        givens = board_from(EASY)
        image = render_puzzle(givens)
        pipeline = SudokuPipeline(TemplateClassifier())
        outcome = pipeline.process(image)
        self.assertEqual([list(row) for row in outcome.givens], givens)
        self.assertTrue(outcome.unique)
        self.assertEqual(outcome.given_count, 30)
        self.assertFalse(outcome.from_cache)
        for row in outcome.solution:
            self.assertEqual(sorted(row), list(range(1, 10)))

    def test_the_second_pass_is_served_from_cache(self) -> None:
        image = render_puzzle(board_from(EASY))
        pipeline = SudokuPipeline(TemplateClassifier())
        pipeline.process(image)
        self.assertTrue(pipeline.process(image).from_cache)

    def test_the_overlay_lands_on_the_original_frame(self) -> None:
        image = render_puzzle(board_from(EASY))
        pipeline = SudokuPipeline(TemplateClassifier())
        outcome = pipeline.process(image)
        annotated = pipeline.annotate(image, outcome)
        self.assertEqual(annotated.shape, image.shape)
        self.assertFalse(np.array_equal(annotated, image))

    def test_a_frame_with_no_board_raises(self) -> None:
        pipeline = SudokuPipeline(TemplateClassifier())
        with self.assertRaises(BoardNotFoundError):
            pipeline.process(np.full((400, 400, 3), 250, np.uint8))

    def test_debug_renderers_produce_images(self) -> None:
        givens = board_from(EASY)
        image = render_puzzle(givens)
        pipeline = SudokuPipeline(TemplateClassifier())
        outcome = pipeline.process(image)
        mosaic = cell_mosaic(outcome.cells)
        self.assertEqual(mosaic.shape[:2], (9 * 32, 9 * 32))
        debug = draw_recognition_debug(
            outcome.board_square, outcome.givens, outcome.confidence
        )
        self.assertEqual(debug.shape[:2], outcome.board_square.shape[:2])

    def test_overlay_can_redraw_the_givens(self) -> None:
        image = render_puzzle(board_from(EASY))
        pipeline = SudokuPipeline(TemplateClassifier())
        outcome = pipeline.process(image)
        with_givens = overlay_solution(
            image, outcome.location, outcome.givens, outcome.solution, draw_givens=True
        )
        without = overlay_solution(
            image, outcome.location, outcome.givens, outcome.solution, draw_givens=False
        )
        self.assertFalse(np.array_equal(with_givens, without))


def config_for(**overrides) -> AppConfig:
    base = dict(
        camera=0,
        input_path=None,
        output_path=None,
        classifier="template",
        model_path=Path("unused.onnx"),
        results_dir=Path("results"),
        resolution=(640, 480),
        board_size=450,
        block_size=19,
        constant=4,
        min_area_ratio=0.04,
        margin_ratio=0.12,
        min_ink_ratio=0.035,
        max_frames=0,
        display=False,
        mirror=False,
        draw_givens=False,
        check_unique=True,
        debug=False,
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
        self.assertEqual(config.classifier, "template")
        self.assertEqual(config.camera, 0)
        self.assertTrue(config.check_unique)
        self.assertTrue(config.mirror)

    def test_an_image_input_defaults_to_no_mirroring(self) -> None:
        self.assertFalse(self.parse(["--input", "puzzle.png"]).mirror)

    def test_onnx_without_a_model_is_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            self.parse(["--classifier", "onnx", "--model", "missing.onnx"])

    def test_size_parsing(self) -> None:
        import argparse

        self.assertEqual(parse_size("1280x720"), (1280, 720))
        for bad in ("1280", "x", "0x10"):
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_size(bad)

    def test_invalid_values_are_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            self.parse(["--camera", "-1"])
        with self.assertRaises(ConfigurationError):
            self.parse(["--board-size", "50"])
        for bad in (["--block-size", "4"], ["--min-area-ratio", "1.5"],
                    ["--max-frames", "-1"], ["--margin-ratio", "0"]):
            with self.assertRaises(SystemExit), quiet():
                build_parser().parse_args(bad)


class MediaTests(unittest.TestCase):
    def test_an_image_is_solved_and_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "puzzle.png"
            output = root / "solved.png"
            cv2.imwrite(str(source), render_puzzle(board_from(EASY)))
            config = config_for(input_path=source, output_path=output, results_dir=root)
            with quiet() as text:
                stats = run_image(config)
            self.assertEqual(stats.solved, 1)
            self.assertTrue(output.is_file())
            self.assertIn("Solved", text.getvalue())

    def test_debug_images_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "puzzle.png"
            cv2.imwrite(str(source), render_puzzle(board_from(EASY)))
            with quiet():
                run_image(config_for(input_path=source, results_dir=root, debug=True))
            for name in ("warped_board.png", "cell_mosaic.png", "recognition.png"):
                self.assertTrue((root / name).is_file(), msg=name)

    def test_an_unreadable_image_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            broken = Path(directory) / "broken.png"
            broken.write_bytes(b"not an image")
            with self.assertRaises(MediaError), quiet():
                run_image(config_for(input_path=broken))

    def test_an_image_without_a_board_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            blank = Path(directory) / "blank.png"
            cv2.imwrite(str(blank), np.full((400, 400, 3), 250, np.uint8))
            with self.assertRaises(BoardNotFoundError), quiet():
                run_image(config_for(input_path=blank))

    def test_input_and_output_must_differ(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "puzzle.png"
            cv2.imwrite(str(source), render_puzzle(board_from(EASY)))
            with self.assertRaises(MediaError), quiet():
                run_image(config_for(input_path=source, output_path=source))

    def test_a_wrong_output_suffix_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "puzzle.png"
            cv2.imwrite(str(source), render_puzzle(board_from(EASY)))
            with self.assertRaises(MediaError), quiet():
                run_image(config_for(input_path=source, output_path=root / "out.avi"))

    def test_the_video_loop_keeps_running_without_a_board(self) -> None:
        frames = [np.full((300, 300, 3), 250, np.uint8) for _ in range(4)]
        fake = FakeCv2(frames)
        with quiet():
            stats = run(config_for(), cv2_module=fake)
        self.assertEqual(stats.frames, 4)
        self.assertEqual(stats.boards_found, 0)

    def test_the_video_loop_solves_a_board(self) -> None:
        image = render_puzzle(board_from(EASY))
        fake = FakeCv2([image.copy() for _ in range(3)])
        with quiet():
            stats = run(config_for(max_frames=3), cv2_module=fake)
        self.assertEqual(stats.frames, 3)
        self.assertEqual(stats.solved, 3)
        self.assertGreaterEqual(stats.cache_hits, 2)

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


if __name__ == "__main__":
    unittest.main()
