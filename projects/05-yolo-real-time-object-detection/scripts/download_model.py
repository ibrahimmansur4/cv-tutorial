#!/usr/bin/env python3
"""Download and verify the pretrained YOLO ONNX models used by this project."""

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
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from src.models import DEFAULT_MODEL, REGISTRY, ModelSpec, available_models  # noqa: E402

USER_AGENT = "cv-tutorial/1.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(path: Path, spec: ModelSpec) -> None:
    """Fail loudly on a truncated or tampered download rather than at inference."""
    actual = sha256(path)
    if actual != spec.sha256:
        raise RuntimeError(
            f"checksum mismatch for {spec.filename}; "
            f"expected {spec.sha256}, received {actual}"
        )


def download(spec: ModelSpec, *, force: bool = False) -> tuple[Path, bool]:
    """Fetch one model, returning its path and whether bytes were transferred."""
    output = spec.path
    if output.exists() and not force:
        validate(output, spec)
        return output, False
    output.parent.mkdir(parents=True, exist_ok=True)
    # Download beside the target and rename only after the digest matches, so an
    # interrupted run never leaves a half-written model that looks usable.
    temporary = output.with_name(f".{output.name}.{os.getpid()}.download")
    try:
        request = urllib.request.Request(spec.url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
        validate(temporary, spec)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output, True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"model name, or 'all' for every registered model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--force", action="store_true", help="re-download an existing model")
    parser.add_argument("--list", action="store_true", help="print the registry and exit")
    args = parser.parse_args(argv)

    if args.list:
        for spec in available_models():
            state = "present" if spec.path.is_file() else "missing"
            print(f"{spec.name:10} {spec.size_mb:6.1f} MB  {spec.family.value:8} {state}")
        return 0

    requested = args.model.strip().casefold()
    if requested == "all":
        specs = list(available_models())
    else:
        spec = REGISTRY.get(requested)
        if spec is None:
            print(
                f"error: unknown model '{args.model}'. "
                f"Registered models: {', '.join(sorted(REGISTRY))}, or 'all'.",
                file=sys.stderr,
            )
            return 2
        specs = [spec]

    for spec in specs:
        try:
            path, fetched = download(spec, force=args.force)
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            print(f"error: {spec.name}: {exc}", file=sys.stderr)
            return 2
        action = "downloaded" if fetched else "already present"
        print(f"{spec.name}: {action} ({spec.size_mb:.1f} MB) -> {path}")
    print(f"\nSHA-256 verified for {len(specs)} model(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
