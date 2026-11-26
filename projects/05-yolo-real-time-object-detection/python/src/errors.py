"""Expected, user-facing object-detection failures."""


class DetectionError(Exception):
    """Base exception for expected application failures."""


class ConfigurationError(DetectionError):
    """Raised for invalid options or missing local resources."""


class ModelError(DetectionError):
    """Raised when a YOLO network cannot be loaded or executed."""


class DecodeError(DetectionError):
    """Raised when a network produces an unrecognized output layout."""


class MediaError(DetectionError):
    """Raised when camera/video input or annotated output fails."""
