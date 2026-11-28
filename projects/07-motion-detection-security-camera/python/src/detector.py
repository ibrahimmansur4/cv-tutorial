"""The motion-detection pipeline: prepare, subtract, clean up, find regions.

```
frame -> grayscale -> blur -> background subtraction -> threshold
      -> morphology -> contours -> area filter -> motion regions
```

Each stage exists to remove a specific kind of false positive:

* **Grayscale** discards colour, which carries almost no motion information but
  triples the work.
* **Blur** removes sensor noise. Without it, every frame differs from every
  other frame at the single-pixel level and the mask is permanent static.
* **Threshold** turns "how different" into "different enough".
* **Morphology** closes the speckled holes that thresholding leaves inside a
  real object, and erases the isolated specks it leaves outside one.
* **Area filter** drops regions too small to be anything worth recording.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

from .background import BackgroundModel
from .errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class MotionRegion:
    """One moving region, in frame pixels."""

    x: int
    y: int
    width: int
    height: int
    area: float

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.width, self.y + self.height

    @property
    def centre(self) -> tuple[int, int]:
        return self.x + self.width // 2, self.y + self.height // 2


@dataclass(frozen=True, slots=True)
class MotionResult:
    """What one frame's analysis produced."""

    regions: tuple[MotionRegion, ...]
    mask: np.ndarray
    changed_pixels: int
    frame_pixels: int
    ready: bool

    @property
    def detected(self) -> bool:
        return bool(self.regions)

    @property
    def total_area(self) -> float:
        return float(sum(region.area for region in self.regions))

    @property
    def changed_fraction(self) -> float:
        return self.changed_pixels / self.frame_pixels if self.frame_pixels else 0.0


class MotionDetector:
    """Turn frames into motion regions using a background model."""

    def __init__(
        self,
        background: BackgroundModel,
        *,
        blur_kernel: int = 21,
        min_area: float = 500.0,
        dilate_iterations: int = 2,
        max_regions: int = 20,
        cv2_module: Any = cv2,
    ) -> None:
        if blur_kernel < 1 or blur_kernel % 2 == 0:
            raise ConfigurationError("blur kernel must be a positive odd number")
        if min_area < 0.0:
            raise ConfigurationError("minimum area cannot be negative")
        if dilate_iterations < 0:
            raise ConfigurationError("dilate iterations cannot be negative")
        if max_regions <= 0:
            raise ConfigurationError("maximum regions must be positive")
        self.background = background
        self.blur_kernel = int(blur_kernel)
        self.min_area = float(min_area)
        self.dilate_iterations = int(dilate_iterations)
        self.max_regions = int(max_regions)
        self._cv2 = cv2_module
        # An ellipse is a better stand-in for a blob of a person or vehicle than
        # a rectangle, which tends to square off the corners of every region.
        self._kernel = cv2_module.getStructuringElement(cv2_module.MORPH_ELLIPSE, (5, 5))

    def prepare(self, frame: np.ndarray) -> np.ndarray:
        """Grayscale and blur, the two steps that make differencing stable."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ConfigurationError("cannot analyse an empty frame")
        if frame.ndim == 3 and frame.shape[2] in (3, 4):
            code = (
                self._cv2.COLOR_BGRA2GRAY
                if frame.shape[2] == 4
                else self._cv2.COLOR_BGR2GRAY
            )
            gray = self._cv2.cvtColor(frame, code)
        elif frame.ndim == 2:
            gray = frame
        else:
            raise ConfigurationError("expected a BGR, BGRA, or single-channel frame")
        return self._cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)

    def clean(self, mask: np.ndarray) -> np.ndarray:
        """Close holes inside regions and erase isolated specks outside them."""
        # Opening first removes the specks; closing then seals the gaps that
        # thresholding punched in the middle of a genuine object. Doing it in
        # the other order would enlarge the specks before trying to remove them.
        cleaned = self._cv2.morphologyEx(mask, self._cv2.MORPH_OPEN, self._kernel)
        cleaned = self._cv2.morphologyEx(cleaned, self._cv2.MORPH_CLOSE, self._kernel)
        if self.dilate_iterations:
            cleaned = self._cv2.dilate(
                cleaned, self._kernel, iterations=self.dilate_iterations
            )
        return cleaned

    def regions_from(self, mask: np.ndarray) -> tuple[MotionRegion, ...]:
        """Contour the mask and keep the regions large enough to matter."""
        found = self._cv2.findContours(
            mask.copy(), self._cv2.RETR_EXTERNAL, self._cv2.CHAIN_APPROX_SIMPLE
        )
        # findContours returns (contours, hierarchy) on OpenCV 4/5 but returned
        # a 3-tuple on 3.x; taking the second-to-last element works on both.
        contours = found[-2]
        regions: list[MotionRegion] = []
        for contour in contours:
            area = float(self._cv2.contourArea(contour))
            if area < self.min_area:
                continue
            x, y, width, height = self._cv2.boundingRect(contour)
            regions.append(MotionRegion(int(x), int(y), int(width), int(height), area))
        regions.sort(key=lambda region: region.area, reverse=True)
        return tuple(regions[: self.max_regions])

    def detect(self, frame: np.ndarray) -> MotionResult:
        """Run the whole pipeline over one frame."""
        prepared = self.prepare(frame)
        mask = self.background.apply(prepared)
        cleaned = self.clean(mask)
        ready = self.background.ready
        # While the model is still settling, report the mask but no regions:
        # the first frames always differ wildly from an unformed reference.
        regions = self.regions_from(cleaned) if ready else ()
        return MotionResult(
            regions=regions,
            mask=cleaned,
            changed_pixels=int(self._cv2.countNonZero(cleaned)),
            frame_pixels=int(cleaned.shape[0] * cleaned.shape[1]),
            ready=ready,
        )

    def reset(self) -> None:
        self.background.reset()


def merge_regions(regions: Sequence[MotionRegion]) -> MotionRegion | None:
    """One box enclosing every region, used for the event summary."""
    if not regions:
        return None
    x1 = min(region.x for region in regions)
    y1 = min(region.y for region in regions)
    x2 = max(region.x + region.width for region in regions)
    y2 = max(region.y + region.height for region in regions)
    return MotionRegion(
        x1, y1, x2 - x1, y2 - y1, float(sum(region.area for region in regions))
    )
