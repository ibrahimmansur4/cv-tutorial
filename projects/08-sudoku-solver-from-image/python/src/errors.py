"""Expected, user-facing Sudoku-solver failures."""


class SudokuError(Exception):
    """Base exception for expected application failures."""


class ConfigurationError(SudokuError):
    """Raised for invalid options or missing local resources."""


class BoardNotFoundError(SudokuError):
    """Raised when no plausible Sudoku grid is visible in the frame."""


class RecognitionError(SudokuError):
    """Raised when the digit classifier cannot be built or run."""


class UnsolvableError(SudokuError):
    """Raised when the recognized puzzle has no solution."""


class MediaError(SudokuError):
    """Raised when camera/image input or output fails."""
