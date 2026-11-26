"""Virtual mouse package built with OpenCV and MediaPipe Hands."""

from .gestures import Gesture, GestureClassifier
from .pipeline import VirtualMousePipeline

__all__ = ["Gesture", "GestureClassifier", "VirtualMousePipeline"]
