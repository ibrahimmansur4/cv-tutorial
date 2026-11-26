"""Expected, user-facing errors raised by the ALPR application."""


class ALPRError(RuntimeError):
    """Base class for recoverable configuration and processing failures."""


class DependencyError(ALPRError):
    """Raised when an optional runtime dependency is unavailable."""


class DetectionError(ALPRError):
    """Raised when a frame cannot be processed by the plate detector."""


class OCRError(ALPRError):
    """Raised when Tesseract cannot be configured or invoked."""


class ApplicationError(ALPRError):
    """Raised for input, capture, display, and output failures."""
