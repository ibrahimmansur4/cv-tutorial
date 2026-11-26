"""Tesseract adapter and license-plate text normalization."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .errors import DependencyError, OCRError

try:
    import pytesseract as _pytesseract
except (ImportError, OSError):  # Missing Python package or one of its DLLs.
    _pytesseract = None


_NON_PLATE_CHARACTERS = re.compile(r"[^A-Z0-9]+")


def require_pytesseract() -> Any:
    """Return the pytesseract module or explain how to install it."""

    if _pytesseract is None:
        raise DependencyError(
            "pytesseract is not installed. Run `python -m pip install -r "
            "requirements.txt`, then install the separate Tesseract OCR program."
        )
    return _pytesseract


def sanitize_plate_text(text: object, max_characters: int = 16) -> str:
    """Uppercase OCR output and keep only ASCII letters and digits."""

    if max_characters <= 0:
        raise ValueError("max_characters must be positive")
    normalized = _NON_PLATE_CHARACTERS.sub("", str(text).upper())
    return normalized[:max_characters]


@dataclass(frozen=True)
class OcrResult:
    """Sanitized plate text and mean word confidence on a 0..1 scale."""

    text: str
    confidence: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("OCR confidence must be between 0 and 1")


class TesseractOcr:
    """Small dependency-injectable wrapper around pytesseract."""

    def __init__(
        self,
        *,
        minimum_confidence: float = 0.0,
        tesseract_cmd: Optional[Path] = None,
        page_segmentation_mode: int = 7,
        max_characters: int = 16,
        pytesseract_module: Any = None,
    ) -> None:
        if not math.isfinite(minimum_confidence) or not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1")
        if page_segmentation_mode <= 0:
            raise ValueError("page_segmentation_mode must be positive")
        if max_characters <= 0:
            raise ValueError("max_characters must be positive")

        self._module = (
            pytesseract_module
            if pytesseract_module is not None
            else require_pytesseract()
        )
        self._minimum_confidence = minimum_confidence
        self._max_characters = max_characters
        self._config = (
            f"--psm {page_segmentation_mode} "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        )
        if tesseract_cmd is not None:
            command_path = Path(tesseract_cmd).expanduser().resolve()
            if not command_path.is_file():
                raise OCRError(f"Tesseract executable not found: {command_path}")
            try:
                self._module.pytesseract.tesseract_cmd = str(command_path)
            except Exception as exc:
                raise OCRError(f"Could not configure Tesseract: {exc}") from exc

    @staticmethod
    def _is_missing_engine_error(error: Exception) -> bool:
        return isinstance(error, (FileNotFoundError, OSError)) or (
            error.__class__.__name__ == "TesseractNotFoundError"
        )

    def _raise_engine_error(self, error: Exception) -> None:
        if self._is_missing_engine_error(error):
            raise OCRError(
                "The Tesseract OCR executable was not found. Install Tesseract and "
                "ensure it is on PATH, or pass its executable with --tesseract-cmd."
            ) from error
        raise OCRError(f"Tesseract OCR failed: {error}") from error

    def validate_installation(self) -> None:
        """Check the external OCR executable before opening the media source."""

        try:
            self._module.get_tesseract_version()
        except Exception as exc:
            self._raise_engine_error(exc)

    def recognize(self, image: np.ndarray) -> OcrResult:
        """Recognize an enhanced plate image and combine accepted OCR tokens."""

        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise OCRError("Cannot run OCR on an empty plate image.")
        try:
            data = self._module.image_to_data(
                image,
                config=self._config,
                output_type=self._module.Output.DICT,
            )
        except Exception as exc:
            self._raise_engine_error(exc)

        if not isinstance(data, dict):
            raise OCRError("Tesseract returned an unexpected result format.")
        raw_tokens = data.get("text", ())
        raw_confidences = data.get("conf", ())
        tokens = []
        weighted_confidence = 0.0
        total_characters = 0
        for raw_token, raw_confidence in zip(raw_tokens, raw_confidences):
            token = sanitize_plate_text(raw_token, self._max_characters)
            if not token:
                continue
            try:
                confidence_percent = float(raw_confidence)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(confidence_percent) or confidence_percent < 0:
                continue
            confidence = min(1.0, confidence_percent / 100.0)
            if confidence < self._minimum_confidence:
                continue
            remaining = self._max_characters - total_characters
            if remaining <= 0:
                break
            token = token[:remaining]
            tokens.append(token)
            weighted_confidence += confidence * len(token)
            total_characters += len(token)

        if total_characters == 0:
            return OcrResult("", 0.0)
        return OcrResult(
            "".join(tokens),
            min(1.0, weighted_confidence / total_characters),
        )
