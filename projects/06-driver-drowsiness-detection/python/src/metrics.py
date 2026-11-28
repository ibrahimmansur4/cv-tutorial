"""Eye and mouth aspect ratios computed from MediaPipe face landmarks.

The Eye Aspect Ratio (EAR) comes from Soukupova and Cech, *Real-Time Eye Blink
Detection using Facial Landmarks* (2016). Six points are taken around the eye:
two at the corners and two pairs above and below the pupil.

```
        p2    p3
    p1              p4        EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|)
        p6    p5
```

The numerator measures how open the lid is and the denominator how wide the eye
is. Dividing one by the other is what makes the measure useful: both distances
scale together when the face moves closer to or further from the camera, so the
ratio stays roughly constant while the eye is open and collapses toward zero
when it closes. A raw pixel gap between the eyelids would not — it would shrink
simply by leaning back.

The original paper used dlib's 68-point model. MediaPipe's face mesh has 478
points, so the indices below are the corresponding positions on that mesh.

Every function here takes `frame_aspect` (frame width / height) because
MediaPipe reports normalized coordinates; see `geometry.distance` for why
measuring without it makes the ratio depend on the capture resolution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .errors import LandmarkError
from .geometry import Point, distance

# MediaPipe face-mesh indices, ordered p1..p6 as in the diagram above.
# "Left"/"right" are from the subject's point of view, which is how MediaPipe
# names them; on a mirrored preview the left eye appears on the right.
LEFT_EYE: tuple[int, ...] = (362, 385, 387, 263, 373, 380)
RIGHT_EYE: tuple[int, ...] = (33, 160, 158, 133, 153, 144)

# Inner-lip points: two corners and two vertical pairs, same shape as the eye.
MOUTH: tuple[int, ...] = (78, 81, 311, 308, 402, 178)

# The highest index the project reads, used to validate a landmark set early.
MAX_LANDMARK_INDEX = max((*LEFT_EYE, *RIGHT_EYE, *MOUTH))

# Published EAR thresholds cluster around 0.25 for dlib's 68-point model, but
# MediaPipe's mesh places the eyelid points differently and reads noticeably
# lower: a wide-open eye measures nearer 0.20 here, so 0.25 would classify
# every open eye as shut. 0.15 sits roughly midway between open and closed on
# the faces this was checked against, but eye shape varies enough between
# people that this is only a fallback -- run `--calibrate` for a real value.
DEFAULT_EAR_THRESHOLD = 0.15
# Re-opening must clear the threshold by this margin before the eye counts as
# open again. Without the gap, EAR noise sitting on the threshold toggles the
# state every frame and inflates the blink count.
EAR_HYSTERESIS = 0.02
# A yawn opens the inner lip far wider than speech does.
DEFAULT_MAR_THRESHOLD = 0.60


@dataclass(frozen=True, slots=True)
class FaceMetrics:
    """Aspect ratios for one face in one frame."""

    left_ear: float
    right_ear: float
    mar: float

    @property
    def ear(self) -> float:
        """Both eyes averaged.

        Averaging rather than taking a single eye keeps a brief tracking error
        on one side from reading as a closed eye, and averaging rather than
        taking the minimum keeps a true blink visible in the signal.
        """
        return (self.left_ear + self.right_ear) / 2.0

    @property
    def asymmetry(self) -> float:
        """How far apart the two eyes read; large values suggest a bad landmark fit."""
        return abs(self.left_ear - self.right_ear)


def aspect_ratio(points: Sequence[Point], frame_aspect: float = 1.0) -> float:
    """Vertical opening over horizontal width for six points in p1..p6 order."""
    if len(points) != 6:
        raise LandmarkError("an aspect ratio needs exactly six points")
    p1, p2, p3, p4, p5, p6 = points
    horizontal = distance(p1, p4, frame_aspect)
    if horizontal <= 1e-9:
        # Degenerate: the corners landed on the same spot, so the ratio is
        # undefined. Reporting 0 would read as a closed eye and raise a false
        # alarm, so this is surfaced instead.
        raise LandmarkError("landmark corners are coincident; cannot form an aspect ratio")
    vertical = distance(p2, p6, frame_aspect) + distance(p3, p5, frame_aspect)
    ratio = vertical / (2.0 * horizontal)
    if not math.isfinite(ratio):
        raise LandmarkError("aspect ratio is not a finite number")
    return ratio


def _select(landmarks: Sequence[Point], indices: Sequence[int]) -> list[Point]:
    return [landmarks[index] for index in indices]


def compute_metrics(landmarks: Sequence[Point], frame_aspect: float = 1.0) -> FaceMetrics:
    """Measure both eyes and the mouth from a full face-mesh landmark list.

    `frame_aspect` is the frame's width divided by its height.
    """
    if len(landmarks) <= MAX_LANDMARK_INDEX:
        raise LandmarkError(
            f"face mesh has {len(landmarks)} landmarks; "
            f"index {MAX_LANDMARK_INDEX} is required"
        )
    return FaceMetrics(
        left_ear=aspect_ratio(_select(landmarks, LEFT_EYE), frame_aspect),
        right_ear=aspect_ratio(_select(landmarks, RIGHT_EYE), frame_aspect),
        mar=aspect_ratio(_select(landmarks, MOUTH), frame_aspect),
    )
