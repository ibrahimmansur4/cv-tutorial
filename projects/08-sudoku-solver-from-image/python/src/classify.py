"""Digit classifiers for normalised 28x28 cell patches.

Two are provided, and which is better is a question this project answers with
measurements rather than assertion (see `scripts/evaluate.py`).

**Template matching** is the default because it needs no download and no
training framework. Sudoku digits are *printed*, and printed glyphs of the same
digit resemble each other far more than handwriting does, so comparing against
rendered references is a reasonable fit for the problem. Templates are rendered
at import time from OpenCV's built-in Hershey fonts, across several fonts,
weights and small rotations, then normalised exactly like the input. A patch is
scored against every template and the best match wins.

**ONNX MNIST** runs a small pretrained convolutional network through
`cv2.dnn`. It was trained on handwriting, which is a harder problem than print,
and it brings a 26 KB download.

Both return a digit and a confidence, so the pipeline can flag cells it is
unsure about instead of silently guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, Sequence

import cv2
import numpy as np

from .cells import CANVAS, normalize_digit
from .errors import ConfigurationError, RecognitionError

DIGITS = tuple(range(1, 10))  # Sudoku has no zero
# Fonts and scales to render references from. Variety matters more than count:
# the goal is to cover the shapes a digit can take, not to make many copies of
# one shape.
_TEMPLATE_FONTS = (
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
)
_TEMPLATE_SCALES = (1.6, 2.0)
_TEMPLATE_THICKNESS = (2, 3)
_TEMPLATE_ANGLES = (-8.0, 0.0, 8.0)


@dataclass(frozen=True, slots=True)
class Prediction:
    """One classified cell."""

    digit: int
    confidence: float

    @property
    def is_blank(self) -> bool:
        return self.digit == 0


class DigitClassifier(Protocol):
    @property
    def name(self) -> str: ...

    def predict(self, patch: np.ndarray) -> Prediction: ...


def _render_digit(
    digit: int, font: int, scale: float, thickness: int, angle: float
) -> np.ndarray:
    """Draw one digit and normalise it the same way a real cell is normalised."""
    canvas = np.zeros((96, 96), dtype=np.uint8)
    (text_width, text_height), baseline = cv2.getTextSize(
        str(digit), font, scale, thickness
    )
    origin = ((96 - text_width) // 2, (96 + text_height) // 2)
    cv2.putText(canvas, str(digit), origin, font, scale, 255, thickness, cv2.LINE_AA)
    if angle:
        matrix = cv2.getRotationMatrix2D((48.0, 48.0), angle, 1.0)
        canvas = cv2.warpAffine(canvas, matrix, (96, 96), flags=cv2.INTER_LINEAR)
    _, binary = cv2.threshold(canvas, 64, 255, cv2.THRESH_BINARY)
    return normalize_digit(binary)


@lru_cache(maxsize=1)
def build_templates() -> tuple[np.ndarray, np.ndarray]:
    """Render the reference set once: (vectors, labels).

    Vectors are unit-normalised so matching is a cosine similarity, which
    ignores how thick the ink happens to be.
    """
    vectors: list[np.ndarray] = []
    labels: list[int] = []
    for digit in DIGITS:
        for font in _TEMPLATE_FONTS:
            for scale in _TEMPLATE_SCALES:
                for thickness in _TEMPLATE_THICKNESS:
                    for angle in _TEMPLATE_ANGLES:
                        patch = _render_digit(digit, font, scale, thickness, angle)
                        vector = patch.astype(np.float32).ravel()
                        norm = float(np.linalg.norm(vector))
                        if norm <= 1e-6:
                            continue
                        vectors.append(vector / norm)
                        labels.append(digit)
    if not vectors:
        raise RecognitionError("could not render any digit templates")
    return np.vstack(vectors), np.asarray(labels, dtype=np.int32)


class TemplateClassifier:
    """Nearest-template matching over rendered printed digits."""

    def __init__(self, *, min_confidence: float = 0.0) -> None:
        self.min_confidence = float(min_confidence)
        self._vectors, self._labels = build_templates()

    @property
    def name(self) -> str:
        return "template"

    @property
    def template_count(self) -> int:
        return int(self._vectors.shape[0])

    def predict(self, patch: np.ndarray) -> Prediction:
        if patch is None or patch.shape != (CANVAS, CANVAS):
            raise RecognitionError(f"the classifier expects a {CANVAS}x{CANVAS} patch")
        vector = patch.astype(np.float32).ravel()
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-6:
            return Prediction(0, 0.0)
        similarities = self._vectors @ (vector / norm)
        best = int(np.argmax(similarities))
        score = float(similarities[best])
        digit = int(self._labels[best])

        # Confidence is the margin over the best *different* digit, not the raw
        # similarity. A patch resembling every digit equally is not a confident
        # read even if the top score is high.
        others = similarities[self._labels != digit]
        runner_up = float(np.max(others)) if others.size else 0.0
        confidence = max(0.0, min(1.0, score - runner_up)) if score > 0.0 else 0.0
        if score < self.min_confidence:
            return Prediction(0, confidence)
        return Prediction(digit, confidence)


class OnnxDigitClassifier:
    """A pretrained MNIST network run through cv2.dnn."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        net: Any = None,
        cv2_module: Any = cv2,
    ) -> None:
        self._cv2 = cv2_module
        self.model_path = Path(model_path)
        if net is not None:
            self.net = net
        else:
            resolved = self.model_path.expanduser().resolve()
            if not resolved.is_file():
                raise ConfigurationError(
                    f"digit model not found: {resolved}. "
                    "Run `python scripts/download_model.py`."
                )
            try:
                self.net = cv2_module.dnn.readNetFromONNX(str(resolved))
            except Exception as exc:
                raise RecognitionError(f"could not load {resolved}: {exc}") from exc
            if self.net is None or self.net.empty():
                raise RecognitionError(f"loaded an empty network from {resolved}")

    @property
    def name(self) -> str:
        return "onnx"

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        shifted = scores - float(np.max(scores))
        exponentials = np.exp(shifted)
        total = float(np.sum(exponentials))
        return exponentials / total if total > 0.0 else exponentials

    def predict(self, patch: np.ndarray) -> Prediction:
        if patch is None or patch.shape != (CANVAS, CANVAS):
            raise RecognitionError(f"the classifier expects a {CANVAS}x{CANVAS} patch")
        if int(patch.max()) == 0:
            return Prediction(0, 0.0)
        blob = patch.astype(np.float32).reshape(1, 1, CANVAS, CANVAS) / 255.0
        try:
            self.net.setInput(blob)
            output = np.asarray(self.net.forward()).reshape(-1)
        except Exception as exc:
            raise RecognitionError(f"digit inference failed: {exc}") from exc
        if output.size < 10:
            raise RecognitionError(
                f"expected 10 class scores from the digit model, got {output.size}"
            )
        probabilities = self._softmax(output[:10])
        # Index 0 is the digit zero, which cannot appear in Sudoku, so it is
        # excluded rather than allowed to win and produce an impossible board.
        digit = int(np.argmax(probabilities[1:10])) + 1
        return Prediction(digit, float(probabilities[digit]))


def make_classifier(
    name: str, *, model_path: str | Path | None = None, cv2_module: Any = cv2
) -> DigitClassifier:
    """Build a classifier by name."""
    key = name.strip().casefold()
    if key in ("template", "templates"):
        return TemplateClassifier()
    if key in ("onnx", "mnist"):
        if model_path is None:
            raise ConfigurationError("the ONNX classifier needs --model")
        return OnnxDigitClassifier(model_path, cv2_module=cv2_module)
    raise ConfigurationError(f"unknown classifier '{name}'. Choose 'template' or 'onnx'.")


def classify_cells(
    cells: Sequence[Any], classifier: DigitClassifier
) -> tuple[list[list[int]], list[list[float]]]:
    """Classify every occupied cell into a 9x9 board plus a confidence grid."""
    board = [[0] * 9 for _ in range(9)]
    confidence = [[0.0] * 9 for _ in range(9)]
    for cell in cells:
        if not cell.occupied:
            continue
        prediction = classifier.predict(cell.patch)
        board[cell.row][cell.column] = prediction.digit
        confidence[cell.row][cell.column] = prediction.confidence
    return board, confidence
