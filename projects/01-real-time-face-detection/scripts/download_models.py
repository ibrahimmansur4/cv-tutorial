"""Download the official OpenCV face-detector assets without extra packages."""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path
from typing import Optional, Sequence


DOWNLOADS = (
    (
        "haarcascade_frontalface_default.xml",
        "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml",
        100_000,
    ),
    (
        "deploy.prototxt",
        "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt",
        10_000,
    ),
    (
        "res10_300x300_ssd_iter_140000_fp16.caffemodel",
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel",
        5_000_000,
    ),
)


def download_models(destination: Path, *, force: bool = False) -> None:
    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    for name, url, minimum_bytes in DOWNLOADS:
        target = destination / name
        temporary = destination / f"{name}.download"
        if target.is_file() and not force:
            if target.stat().st_size >= minimum_bytes:
                print(f"Already present: {target}")
                continue
            raise RuntimeError(
                f"Existing file appears incomplete: {target}. "
                "Re-run with --force to replace it."
            )

        try:
            print(f"Downloading {name}")
            request = urllib.request.Request(
                url, headers={"User-Agent": "opencv-face-detection-tutorial"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                with temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)

            downloaded_bytes = temporary.stat().st_size
            if downloaded_bytes < minimum_bytes:
                raise RuntimeError(
                    f"Download for {name} is unexpectedly small "
                    f"({downloaded_bytes} bytes)."
                )
            os.replace(temporary, target)
            print(f"Saved: {target} ({downloaded_bytes} bytes)")
        finally:
            temporary.unlink(missing_ok=True)

    print("Model setup complete.")


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=project_root / "models",
        help="download directory (default: project models directory)",
    )
    parser.add_argument(
        "--force", action="store_true", help="replace existing model files"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        download_models(args.destination, force=args.force)
    except (OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
