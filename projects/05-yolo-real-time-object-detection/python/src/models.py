"""Registry of pretrained YOLO models this project can download and run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .decode import ModelFamily
from .errors import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A downloadable ONNX export, with the layout needed to decode it."""

    name: str
    family: ModelFamily
    url: str
    sha256: str
    size_bytes: int
    input_size: tuple[int, int]
    description: str
    # Ultralytics exported YOLOv5 with static shapes: its Reshape layers carry
    # baked-in 640x640 dimensions, so any other input fails inside the forward
    # pass. The YOLO11 export uses dynamic axes and accepts any multiple of 32.
    dynamic_input: bool

    @property
    def filename(self) -> str:
        return f"{self.name}.onnx"

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.filename

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


# Official Ultralytics ONNX exports. YOLOv5 n/s/m share one architecture at
# three widths/depths, which makes them a fair accuracy-versus-speed ladder;
# yolo11n is included because its anchor-free head uses the other output
# layout, so the decoder is exercised on both.
_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="yolov5n",
        family=ModelFamily.YOLOV5,
        url="https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx",
        sha256="04f0e55c26f58d17145b36045780fe1250d5bd2187543e11568e5141d05b3262",
        size_bytes=3981910,
        input_size=(640, 640),
        description="YOLOv5 nano - fastest, lowest accuracy",
        dynamic_input=False,
    ),
    ModelSpec(
        name="yolov5s",
        family=ModelFamily.YOLOV5,
        url="https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx",
        sha256="8bc25a47abdbe2fd23e30eb948746bf856b031ea3cc815166687f52db9f8c659",
        size_bytes=14698981,
        input_size=(640, 640),
        description="YOLOv5 small - balanced",
        dynamic_input=False,
    ),
    ModelSpec(
        name="yolov5m",
        family=ModelFamily.YOLOV5,
        url="https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.onnx",
        sha256="c46186375067d43cf479d014ce8cd3af4c1489fb13b4ebf4e2470d9def12f7d9",
        size_bytes=42606277,
        input_size=(640, 640),
        description="YOLOv5 medium - most accurate, slowest",
        dynamic_input=False,
    ),
    ModelSpec(
        name="yolo11n",
        family=ModelFamily.YOLOV8,
        url="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx",
        sha256="634279b40c07c6391472c51ad45b81ebc48706a9a1fe72dd3396322acd0c053b",
        size_bytes=10930182,
        input_size=(640, 640),
        description="YOLO11 nano - anchor-free head, resizable input",
        dynamic_input=True,
    ),
)

REGISTRY: dict[str, ModelSpec] = {spec.name: spec for spec in _SPECS}
DEFAULT_MODEL = "yolov5n"
# n/s/m only: one architecture at three scales, so a size comparison is not
# confounded by also changing the detection head.
BENCHMARK_LADDER: tuple[str, ...] = ("yolov5n", "yolov5s", "yolov5m")
# The only registered export with dynamic input axes, so the only one that can
# be swept across resolutions.
RESOLUTION_MODEL = "yolo11n"


def available_models() -> tuple[ModelSpec, ...]:
    return _SPECS


def get_spec(name: str) -> ModelSpec:
    """Look up a registered model by name."""
    spec = REGISTRY.get(name.strip().casefold())
    if spec is None:
        raise ConfigurationError(
            f"unknown model '{name}'. Registered models: {', '.join(sorted(REGISTRY))}."
        )
    return spec


def family_for_path(path: Path) -> ModelFamily:
    """Guess the output layout of a user-supplied `.onnx` file from its name.

    Only a hint: `--family` overrides it, and a wrong guess surfaces as a clear
    decode error rather than silent nonsense.
    """
    stem = path.stem.casefold()
    if any(token in stem for token in ("yolov5", "yolov6", "yolov7")):
        return ModelFamily.YOLOV5
    return ModelFamily.YOLOV8
