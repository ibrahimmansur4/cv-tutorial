"""Background models: what the scene looks like when nothing is happening.

Motion detection is subtraction. You need a picture of the empty scene, and
anything that differs from it is "motion". The whole problem is keeping that
reference picture current, because a scene is never truly static: light shifts,
clouds pass, the auto-exposure hunts, a chair gets moved and stays moved.

Two models are offered, and they fail in interestingly different ways.

**Running average** is the classic approach the roadmap describes, and it is
worth implementing by hand because every step is visible. The reference frame is
an exponential moving average of past frames:

```
background = (1 - alpha) * background + alpha * frame
```

A small `alpha` means a long memory: the reference changes slowly, so a person
standing still stays visible as motion for a long time, but a sudden light
change takes a long time to forget. A large `alpha` adapts fast but swallows
slow-moving subjects, which "fade into" the background while you watch.

**MOG2** is OpenCV's adaptive Mixture-of-Gaussians subtractor. Instead of one
average value per pixel it keeps a small mixture of distributions, so a pixel
that legitimately alternates between two values -- foliage against sky, a
flickering screen -- can be modelled as background rather than firing constantly.
It also detects shadows, which is the single biggest source of oversized boxes
in the simple method.

Both expose the same interface, so the rest of the project never branches on
which is in use.
"""

from __future__ import annotations

from typing import Any, Protocol

import cv2
import numpy as np

from .errors import ConfigurationError


class BackgroundModel(Protocol):
    """Anything that turns a prepared frame into a foreground mask."""

    @property
    def name(self) -> str: ...

    @property
    def ready(self) -> bool: ...

    def apply(self, prepared: np.ndarray) -> np.ndarray: ...

    def reset(self) -> None: ...


class RunningAverageBackground:
    """Frame differencing against an exponential moving average.

    `apply` expects an already blurred, single-channel frame and returns a
    binary mask of pixels that differ from the reference by more than
    `delta_threshold`.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.04,
        delta_threshold: int = 12,
        warmup_frames: int = 20,
        cv2_module: Any = cv2,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ConfigurationError("background alpha must be above 0 and at most 1")
        if not 0 < delta_threshold < 256:
            raise ConfigurationError("delta threshold must be between 1 and 255")
        if warmup_frames < 0:
            raise ConfigurationError("warm-up frames cannot be negative")
        self.alpha = float(alpha)
        self.delta_threshold = int(delta_threshold)
        self.warmup_frames = int(warmup_frames)
        self._cv2 = cv2_module
        self._reference: np.ndarray | None = None
        self._frames = 0

    @property
    def name(self) -> str:
        return "running-average"

    @property
    def ready(self) -> bool:
        """False until the reference has settled, so startup does not fire."""
        return self._frames > self.warmup_frames

    @property
    def reference(self) -> np.ndarray | None:
        """The current background estimate, as an 8-bit image."""
        if self._reference is None:
            return None
        return self._reference.astype(np.uint8)

    def apply(self, prepared: np.ndarray) -> np.ndarray:
        if prepared.ndim != 2:
            raise ConfigurationError("the background model expects a single-channel frame")
        # Accumulate in float32: averaging 8-bit values in place would quantise
        # away the small per-frame updates that a low alpha depends on.
        current = prepared.astype(np.float32)
        if self._reference is None or self._reference.shape != current.shape:
            self._reference = current.copy()
            self._frames = 0

        delta = self._cv2.absdiff(prepared, self._reference.astype(np.uint8))
        _, mask = self._cv2.threshold(
            delta, self.delta_threshold, 255, self._cv2.THRESH_BINARY
        )
        self._cv2.accumulateWeighted(current, self._reference, self.alpha)
        self._frames += 1
        return mask

    def reset(self) -> None:
        self._reference = None
        self._frames = 0


class Mog2Background:
    """OpenCV's Mixture-of-Gaussians subtractor, with shadow removal."""

    # MOG2 marks shadows with this value instead of 255, so they can be dropped.
    SHADOW_VALUE = 127

    def __init__(
        self,
        *,
        history: int = 300,
        var_threshold: float = 24.0,
        detect_shadows: bool = True,
        warmup_frames: int = 20,
        subtractor: Any = None,
        cv2_module: Any = cv2,
    ) -> None:
        if history <= 0:
            raise ConfigurationError("MOG2 history must be positive")
        if var_threshold <= 0.0:
            raise ConfigurationError("MOG2 variance threshold must be positive")
        if warmup_frames < 0:
            raise ConfigurationError("warm-up frames cannot be negative")
        self.warmup_frames = int(warmup_frames)
        self.detect_shadows = bool(detect_shadows)
        self._cv2 = cv2_module
        self._frames = 0
        if subtractor is not None:
            self._subtractor = subtractor
            return
        try:
            self._subtractor = cv2_module.createBackgroundSubtractorMOG2(
                history=int(history),
                varThreshold=float(var_threshold),
                detectShadows=bool(detect_shadows),
            )
        except Exception as exc:
            raise ConfigurationError(f"could not create the MOG2 subtractor: {exc}") from exc

    @property
    def name(self) -> str:
        return "mog2"

    @property
    def ready(self) -> bool:
        return self._frames > self.warmup_frames

    def apply(self, prepared: np.ndarray) -> np.ndarray:
        if prepared.ndim != 2:
            raise ConfigurationError("the background model expects a single-channel frame")
        mask = self._subtractor.apply(prepared)
        self._frames += 1
        if self.detect_shadows:
            # Keep only hard foreground. A shadow is a real brightness change
            # but not an object, and including it merges a person with the
            # ground they are standing on into one oversized box.
            _, mask = self._cv2.threshold(
                mask, self.SHADOW_VALUE, 255, self._cv2.THRESH_BINARY
            )
        return mask

    def reset(self) -> None:
        self._frames = 0


def make_background(name: str, **kwargs: Any) -> BackgroundModel:
    """Build a background model by name."""
    key = name.strip().casefold()
    if key in ("running-average", "running_average", "average", "diff"):
        return RunningAverageBackground(
            alpha=kwargs.get("alpha", 0.04),
            delta_threshold=kwargs.get("delta_threshold", 12),
            warmup_frames=kwargs.get("warmup_frames", 20),
            cv2_module=kwargs.get("cv2_module", cv2),
        )
    if key in ("mog2", "mog"):
        return Mog2Background(
            history=kwargs.get("history", 300),
            var_threshold=kwargs.get("var_threshold", 24.0),
            detect_shadows=kwargs.get("detect_shadows", True),
            warmup_frames=kwargs.get("warmup_frames", 20),
            cv2_module=kwargs.get("cv2_module", cv2),
        )
    raise ConfigurationError(
        f"unknown background model '{name}'. Choose 'running-average' or 'mog2'."
    )
