"""Expected, user-facing security-camera failures."""


class SecurityCameraError(Exception):
    """Base exception for expected application failures."""


class ConfigurationError(SecurityCameraError):
    """Raised for invalid options or missing local resources."""


class MediaError(SecurityCameraError):
    """Raised when camera/video input or recording output fails."""


class RecorderError(SecurityCameraError):
    """Raised when an event clip cannot be written."""
