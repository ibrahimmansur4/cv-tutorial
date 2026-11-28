#!/usr/bin/env python3
"""Render QR codes and EAN-13 barcodes to scan, with optional degradation.

```powershell
python scripts/make_codes.py --qr https://example.com --out assets/input/qr.png
python scripts/make_codes.py --ean 400638133393 --out assets/input/ean.png
python scripts/make_codes.py --sample --out assets/input/sample.png
python scripts/make_codes.py --qr hello --blur 3 --rotate 20 --out assets/input/hard.png
```

Ground-truth payloads and corners are written alongside as JSON with `--truth`,
so a run can be scored rather than eyeballed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from src import synth  # noqa: E402
from src.errors import ScannerError  # noqa: E402
from src.symbols import RenderedCode, render_barcode, render_qr  # noqa: E402

SAMPLE_QR = (
    "https://example.com/scan",
    "WIFI:T:WPA;S:Workshop;P:opencv2024;;",
    "BEGIN:VCARD\nVERSION:3.0\nFN:Ada Lovelace\nTEL:+441234567890\nEND:VCARD",
    "Plain text, no scheme at all",
)
SAMPLE_EAN = ("400638133393", "978014300723", "590123412345")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="make_codes.py",
        description="Render codes to test the scanner against.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    what = parser.add_argument_group("what to render")
    what.add_argument("--qr", action="append", default=None, metavar="TEXT",
                      help="payload for a QR code; repeat for several")
    what.add_argument("--ean", action="append", default=None, metavar="DIGITS",
                      help="12 or 13 digits for an EAN-13; repeat for several")
    what.add_argument("--sample", action="store_true",
                      help="render a sheet of assorted sample codes")
    what.add_argument("--correction", choices=("L", "M", "Q", "H"), default="M",
                      help="QR error-correction level")
    what.add_argument("--module", type=int, default=8,
                      help="pixels per QR module / per barcode bar")

    layout = parser.add_argument_group("layout")
    layout.add_argument("--frame", default="1280x720", metavar="WxH",
                        help="frame size the codes are placed in")
    layout.add_argument("--columns", type=int, default=2,
                        help="columns to use when rendering several codes")
    layout.add_argument("--scale", type=float, default=1.0,
                        help="scale a single code within the frame")
    layout.add_argument("--gray", action="store_true", help="write a greyscale image")

    harm = parser.add_argument_group("degradation")
    harm.add_argument("--blur", type=int, default=0, help="Gaussian blur radius, px")
    harm.add_argument("--noise", type=float, default=0.0, help="sensor noise sigma")
    harm.add_argument("--rotate", type=float, default=0.0, help="rotation, degrees")
    harm.add_argument("--perspective", type=float, default=0.0,
                      help="tilt strength, 0 to 0.49")
    harm.add_argument("--shrink", type=float, default=1.0,
                      help="scale the finished frame down by this factor")
    harm.add_argument("--occlude", type=float, default=0.0,
                      help="fraction of each code to cover, 0 to 0.99")
    harm.add_argument("--occlude-corner", choices=("tl", "tr", "br", "bl"), default="tl",
                      help="which corner of the code to cover")
    harm.add_argument("--seed", type=int, default=0, help="noise seed")

    parser.add_argument("--out", type=Path, required=True, help="image to write")
    parser.add_argument("--truth", type=Path, default=None,
                        help="also write ground-truth JSON here")
    return parser


def parse_frame(value: str) -> tuple[int, int]:
    pieces = value.lower().split("x")
    if len(pieces) != 2:
        raise ScannerError("--frame must look like 1280x720")
    try:
        width, height = (int(piece) for piece in pieces)
    except ValueError as exc:
        raise ScannerError("--frame must look like 1280x720") from exc
    if width < 64 or height < 64:
        raise ScannerError("--frame must be at least 64x64")
    return width, height


def collect(args: argparse.Namespace) -> list[RenderedCode]:
    codes: list[RenderedCode] = []
    qr_payloads = list(args.qr or ())
    ean_payloads = list(args.ean or ())
    if args.sample:
        qr_payloads.extend(SAMPLE_QR)
        ean_payloads.extend(SAMPLE_EAN)
    for payload in qr_payloads:
        codes.append(
            render_qr(payload, module_size=args.module, correction=args.correction)
        )
    for digits in ean_payloads:
        # A bar one QR-module wide makes a barcode far wider than it is tall,
        # so bars get a narrower module and the standard 69-module height.
        bar_module = max(2, args.module // 2)
        codes.append(
            render_barcode(digits, module_width=bar_module, height=69 * bar_module)
        )
    if not codes:
        raise ScannerError("nothing to render: pass --qr, --ean or --sample")
    return codes


def degrade(scene: synth.Scene, args: argparse.Namespace) -> synth.Scene:
    """Apply the requested effects in the order a camera would impose them."""
    if args.occlude:
        scene = synth.occlude(scene, args.occlude, corner=args.occlude_corner)
    if args.perspective:
        scene = synth.perspective(scene, args.perspective)
    if args.rotate:
        scene = synth.rotate(scene, args.rotate)
    if args.shrink != 1.0:
        scene = synth.shrink(scene, args.shrink)
    # Blur and noise come last: they happen in the lens and the sensor, after
    # the geometry is fixed.
    if args.blur:
        scene = synth.blur(scene, args.blur)
    if args.noise:
        scene = synth.noise(scene, args.noise, seed=args.seed)
    return scene


def write_truth(path: Path, scene: synth.Scene, args: argparse.Namespace) -> None:
    payload = {
        "frame": [int(scene.image.shape[1]), int(scene.image.shape[0])],
        "codes": [
            {
                "payload": item.payload,
                "symbology": item.symbology,
                "corners": [[round(float(x), 2), round(float(y), 2)] for x, y in item.corners],
            }
            for item in scene.placements
        ],
        "degradation": {
            "blur": args.blur,
            "noise": args.noise,
            "rotate": args.rotate,
            "perspective": args.perspective,
            "shrink": args.shrink,
            "occlude": args.occlude,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        width, height = parse_frame(args.frame)
        codes = collect(args)
        colour = not args.gray
        if len(codes) == 1:
            scene = synth.place(
                codes[0], width=width, height=height, scale=args.scale, colour=colour
            )
        else:
            scene = synth.grid(
                codes, width=width, height=height, columns=args.columns, colour=colour
            )
        scene = degrade(scene, args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.out), scene.image):
            raise ScannerError(f"OpenCV refused to write {args.out}")
    except ScannerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote {args.out} ({scene.image.shape[1]}x{scene.image.shape[0]})")
    for item in scene.placements:
        print(f"  {item.symbology:<8} {item.payload[:60]}")
    if args.truth is not None:
        write_truth(args.truth, scene, args)
        print(f"Ground truth: {args.truth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
