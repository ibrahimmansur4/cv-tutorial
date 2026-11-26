#!/usr/bin/env python3
"""Download and verify the official MediaPipe Hand Landmarker model bundle."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(path: Path) -> None:
    actual = sha256(path)
    if actual != MODEL_SHA256:
        raise RuntimeError(
            f"model checksum mismatch for {path}; expected {MODEL_SHA256}, received {actual}"
        )
    if not zipfile.is_zipfile(path):
        raise RuntimeError(f"downloaded model is not a valid MediaPipe task bundle: {path}")


def download(output: Path, *, force: bool = False) -> Path:
    output = output.expanduser().resolve()
    if output.exists() and not force:
        validate(output)
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.download")
    try:
        request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "cv-tutorial/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        validate(temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="replace an existing model")
    args = parser.parse_args(argv)
    try:
        output = download(args.output, force=args.force)
    except (OSError, RuntimeError, urllib.error.URLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"MediaPipe model ready: {output}")
    print(f"SHA-256: {MODEL_SHA256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
