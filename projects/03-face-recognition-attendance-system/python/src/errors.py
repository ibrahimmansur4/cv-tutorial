"""Application-specific exceptions with user-friendly messages."""


class AttendanceError(Exception):
    """Base exception for expected application failures."""


class ConfigurationError(AttendanceError):
    """Raised when command-line options or local resources are invalid."""


class FaceDataError(AttendanceError):
    """Raised when known-face data cannot be prepared safely."""


class MediaError(AttendanceError):
    """Raised when an input or output media stream cannot be used."""

