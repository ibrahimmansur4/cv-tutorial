"""Expected, user-facing virtual-mouse failures."""


class VirtualMouseError(Exception):
    """Base exception for expected application failures."""


class ConfigurationError(VirtualMouseError):
    """Raised for invalid options or missing local resources."""


class DependencyError(VirtualMouseError):
    """Raised when an optional runtime dependency is unavailable."""


class TrackingError(VirtualMouseError):
    """Raised when MediaPipe cannot process a hand frame."""


class ControllerError(VirtualMouseError):
    """Raised when the operating-system pointer cannot be controlled."""


class MediaError(VirtualMouseError):
    """Raised when camera/video input or annotated output fails."""
