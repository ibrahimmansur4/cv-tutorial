"""Python implementation of the automatic license plate recognition demo."""

from .app import AppConfig, RunStats, main, run
from .errors import ALPRError

__all__ = ["ALPRError", "AppConfig", "RunStats", "main", "run"]
