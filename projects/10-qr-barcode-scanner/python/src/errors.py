"""Expected, user-facing scanner failures."""


class ScannerError(Exception):
    """Base exception for expected application failures."""


class ConfigurationError(ScannerError):
    """Raised for invalid options or missing local resources."""


class MediaError(ScannerError):
    """Raised when camera/image input or output fails."""


class HistoryError(ScannerError):
    """Raised when the scan history cannot be written."""
