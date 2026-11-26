"""Face-recognition attendance application package."""

from .attendance_log import AttendanceLogger, AttendanceRecord
from .detection import FaceBox, FaceDetector
from .embeddings import FaceIndex, HOGFaceEmbedder, KnownFaceStore, Match
from .pipeline import FaceObservation, RecognitionPipeline

__all__ = [
    "AttendanceLogger",
    "AttendanceRecord",
    "FaceBox",
    "FaceDetector",
    "FaceIndex",
    "FaceObservation",
    "HOGFaceEmbedder",
    "KnownFaceStore",
    "Match",
    "RecognitionPipeline",
]
