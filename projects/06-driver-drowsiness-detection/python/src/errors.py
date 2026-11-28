"""Expected, user-facing drowsiness-detection failures."""


class DrowsinessError(Exception):
    """Base exception for expected application failures."""


class ConfigurationError(DrowsinessError):
    """Raised for invalid options or missing local resources."""


class DependencyError(DrowsinessError):
    """Raised when an optional runtime dependency is unavailable."""


class LandmarkError(DrowsinessError):
    """Raised when MediaPipe cannot produce usable face landmarks."""


class MediaError(DrowsinessError):
    """Raised when camera/video input or annotated output fails."""
