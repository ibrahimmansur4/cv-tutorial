#!/usr/bin/env python3
"""Download and verify the optional pretrained MNIST digit model.

The default `--classifier template` needs nothing downloaded. This model is for
`--classifier onnx`, which `scripts/evaluate.py` compares against it.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "mnist-8.onnx"
MODEL_URL = (
    "https://github.com/onnx/models/raw/main/validated/vision/classification/"
    "mnist/model/mnist-8.onnx"
)
MODEL_SHA256 = "2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf"


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
            f"checksum mismatch for {path}; expected {MODEL_SHA256}, received {actual}"
        )


def download(output: Path, *, force: bool = False) -> Path:
    output = output.expanduser().resolve()
    if output.exists() and not force:
        validate(output)
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.download")
    try:
        request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "cv-tutorial/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary.open("wb") as stream:
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
    print(f"MNIST digit model ready: {output}")
    print(f"SHA-256: {MODEL_SHA256}")
    print("Use it with: python python/sudoku.py --classifier onnx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
