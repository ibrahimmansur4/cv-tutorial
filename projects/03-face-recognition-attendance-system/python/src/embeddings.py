"""HOG face descriptors, cosine matching, and safe on-disk caching."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .detection import FaceDetector
from .errors import ConfigurationError, FaceDataError

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
CACHE_SCHEMA_VERSION = 1


def _normalized(vector: np.ndarray) -> np.ndarray:
    flattened = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(flattened))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise FaceDataError("face descriptor contained no usable information")
    result = flattened / norm
    if not np.all(np.isfinite(result)):
        raise FaceDataError("face descriptor contained non-finite values")
    return result.astype(np.float32, copy=False)


class HOGFaceEmbedder:
    """Build normalized, fixed-size HOG descriptors for face crops."""

    def __init__(
        self,
        image_size: tuple[int, int] = (128, 128),
        *,
        cv2_module: Any = cv2,
        descriptor: Any | None = None,
    ) -> None:
        width, height = (int(image_size[0]), int(image_size[1]))
        if width < 32 or height < 32 or width % 16 or height % 16:
            raise ConfigurationError(
                "HOG image dimensions must be at least 32 and divisible by 16"
            )
        self.image_size = (width, height)
        self._cv2 = cv2_module
        self._descriptor = descriptor or self._cv2.HOGDescriptor(
            self.image_size,
            (16, 16),
            (8, 8),
            (8, 8),
            9,
        )

    def embed(self, face: np.ndarray) -> np.ndarray:
        if face is None or not isinstance(face, np.ndarray) or face.size == 0:
            raise FaceDataError("cannot embed an empty face crop")
        if face.ndim == 2:
            gray = face
        elif face.ndim == 3 and face.shape[2] in (3, 4):
            code = (
                self._cv2.COLOR_BGRA2GRAY
                if face.shape[2] == 4
                else self._cv2.COLOR_BGR2GRAY
            )
            gray = self._cv2.cvtColor(face, code)
        else:
            raise FaceDataError("face crop must be grayscale, BGR, or BGRA")

        resized = self._cv2.resize(gray, self.image_size, interpolation=self._cv2.INTER_AREA)
        equalized = self._cv2.equalizeHist(resized)
        descriptor = self._descriptor.compute(equalized)
        if descriptor is None:
            raise FaceDataError("OpenCV failed to calculate a HOG descriptor")
        return _normalized(descriptor)


@dataclass(frozen=True, slots=True)
class Match:
    """Nearest known face for an embedding."""

    name: str | None
    distance: float
    similarity: float

    @property
    def recognized(self) -> bool:
        return self.name is not None


class FaceIndex:
    """An immutable matrix of normalized embeddings and their labels."""

    def __init__(self, labels: Sequence[str], embeddings: np.ndarray) -> None:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise FaceDataError("embedding index must be a two-dimensional matrix")
        clean_labels = tuple(str(label).strip() for label in labels)
        if len(clean_labels) != matrix.shape[0]:
            raise FaceDataError("embedding labels and rows have different lengths")
        if any(not label for label in clean_labels):
            raise FaceDataError("known-face labels cannot be empty")
        if matrix.size and not np.all(np.isfinite(matrix)):
            raise FaceDataError("embedding index contains non-finite values")

        if matrix.shape[0]:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            if np.any(norms <= 1e-12):
                raise FaceDataError("embedding index contains an empty descriptor")
            matrix = matrix / norms
        self.labels = clean_labels
        self.embeddings = np.ascontiguousarray(matrix, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def dimension(self) -> int:
        return int(self.embeddings.shape[1]) if self.embeddings.ndim == 2 else 0

    def match(self, embedding: np.ndarray, threshold: float = 0.35) -> Match:
        """Match by cosine distance (``1 - cosine_similarity``)."""

        if threshold < 0.0 or threshold > 2.0:
            raise ConfigurationError("recognition threshold must be between 0 and 2")
        if not len(self):
            return Match(None, float("inf"), 0.0)
        query = _normalized(embedding)
        if query.shape[0] != self.dimension:
            raise FaceDataError(
                f"descriptor dimension {query.shape[0]} does not match index dimension "
                f"{self.dimension}"
            )
        similarities = np.clip(self.embeddings @ query, -1.0, 1.0)
        best_index = int(np.argmax(similarities))
        similarity = float(similarities[best_index])
        distance = 1.0 - similarity
        label = self.labels[best_index] if distance <= threshold else None
        return Match(label, distance, similarity)


@dataclass(frozen=True, slots=True)
class LoadReport:
    discovered: int
    loaded: int
    skipped: tuple[str, ...]
    used_cache: bool


class KnownFaceStore:
    """Load named face images and maintain an invalidation-aware NPZ cache.

    Cache arrays contain only numeric and Unicode dtypes and are always opened with
    ``allow_pickle=False``. This prevents a cache file from executing Python objects.
    """

    def __init__(
        self,
        known_faces_dir: str | Path,
        cache_path: str | Path,
        detector: FaceDetector,
        embedder: HOGFaceEmbedder,
        *,
        cv2_module: Any = cv2,
        allow_full_image_fallback: bool = False,
    ) -> None:
        self.known_faces_dir = Path(known_faces_dir).expanduser().resolve()
        self.cache_path = Path(cache_path).expanduser().resolve()
        self.detector = detector
        self.embedder = embedder
        self._cv2 = cv2_module
        self.allow_full_image_fallback = bool(allow_full_image_fallback)
        self.last_report = LoadReport(0, 0, (), False)

    def _images(self) -> list[Path]:
        if not self.known_faces_dir.exists():
            return []
        if not self.known_faces_dir.is_dir():
            raise FaceDataError(
                f"known-faces path is not a directory: {self.known_faces_dir}"
            )
        return sorted(
            (
                path
                for path in self.known_faces_dir.rglob("*")
                if path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
                and not any(part.startswith(".") for part in path.relative_to(self.known_faces_dir).parts)
            ),
            key=lambda item: item.as_posix().casefold(),
        )

    def _label_for(self, path: Path) -> str:
        relative = path.relative_to(self.known_faces_dir)
        raw = relative.stem if len(relative.parts) == 1 else relative.parts[0]
        return raw.replace("_", " ").strip()

    def _signature(self, images: Sequence[Path]) -> str:
        digest = hashlib.sha256()
        digest.update(f"schema:{CACHE_SCHEMA_VERSION}\n".encode("ascii"))
        digest.update(f"hog:{self.embedder.image_size!r}\n".encode("ascii"))
        digest.update(f"root:{self.known_faces_dir}\n".encode("utf-8"))
        digest.update(f"full-image-fallback:{self.allow_full_image_fallback}\n".encode("ascii"))
        digest.update(
            (
                f"haar:{self.detector.cascade_path}|{self.detector.scale_factor}|"
                f"{self.detector.min_neighbors}|{self.detector.min_size}\n"
            ).encode("utf-8")
        )
        for path in images:
            stat = path.stat()
            relative = path.relative_to(self.known_faces_dir).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("ascii"))
        return digest.hexdigest()

    def load(self, *, force_rebuild: bool = False) -> FaceIndex:
        images = self._images()
        signature = self._signature(images)
        if not force_rebuild and self.cache_path.is_file():
            cached = self._load_cache(signature)
            if cached is not None:
                self.last_report = LoadReport(len(images), len(cached), (), True)
                return cached
        return self._rebuild(images, signature)

    def _load_cache(self, expected_signature: str) -> FaceIndex | None:
        try:
            with np.load(self.cache_path, allow_pickle=False) as archive:
                required = {"schema", "signature", "labels", "embeddings"}
                if not required.issubset(archive.files):
                    return None
                schema = int(np.asarray(archive["schema"]).reshape(-1)[0])
                signature = str(np.asarray(archive["signature"]).reshape(-1)[0])
                if schema != CACHE_SCHEMA_VERSION or signature != expected_signature:
                    return None
                labels = np.asarray(archive["labels"])
                embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
                if labels.dtype.kind not in {"U", "S"}:
                    return None
                return FaceIndex([str(label) for label in labels.tolist()], embeddings)
        except (OSError, ValueError, KeyError, FaceDataError):
            # A damaged/stale cache is disposable; source images remain authoritative.
            return None

    def _rebuild(self, images: Sequence[Path], signature: str) -> FaceIndex:
        labels: list[str] = []
        descriptors: list[np.ndarray] = []
        skipped: list[str] = []
        for path in images:
            label = self._label_for(path)
            if not label:
                skipped.append(f"{path.name}: empty label")
                continue
            image = self._cv2.imread(str(path), self._cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                skipped.append(f"{path.name}: unreadable image")
                continue
            try:
                faces = self.detector.detect(image)
                if len(faces) == 1:
                    face = faces[0].crop(image)
                elif not faces and self.allow_full_image_fallback:
                    face = image
                elif not faces:
                    skipped.append(f"{path.name}: no face detected")
                    continue
                else:
                    skipped.append(f"{path.name}: multiple faces detected")
                    continue
                descriptors.append(self.embedder.embed(face))
                labels.append(label)
            except (FaceDataError, ValueError) as exc:
                skipped.append(f"{path.name}: {exc}")

        if descriptors:
            embeddings = np.vstack(descriptors).astype(np.float32, copy=False)
        else:
            # HOGDescriptor.getDescriptorSize is stable and avoids a fake image pass.
            dimension = int(self.embedder._descriptor.getDescriptorSize())
            embeddings = np.empty((0, dimension), dtype=np.float32)
        index = FaceIndex(labels, embeddings)
        self._write_cache(index, signature)
        self.last_report = LoadReport(len(images), len(index), tuple(skipped), False)
        return index

    def _write_cache(self, index: FaceIndex, signature: str) -> None:
        temporary = self.cache_path.with_name(
            f".{self.cache_path.name}.{os.getpid()}.tmp.npz"
        )
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            max_label_length = max((len(label) for label in index.labels), default=1)
            labels = np.asarray(index.labels, dtype=f"<U{max_label_length}")
            np.savez_compressed(
                temporary,
                schema=np.asarray([CACHE_SCHEMA_VERSION], dtype=np.int32),
                signature=np.asarray([signature], dtype="<U64"),
                labels=labels,
                embeddings=index.embeddings,
            )
            os.replace(temporary, self.cache_path)
        except OSError as exc:
            raise FaceDataError(f"could not write embedding cache {self.cache_path}: {exc}") from exc
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
