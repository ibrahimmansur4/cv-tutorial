#!/usr/bin/env python3
"""Measure what the scanner actually survives, and write the README tables.

```powershell
python scripts/benchmark.py --out results
```

Every number the README quotes comes from here. Each experiment sweeps one
degradation over a set of payloads and reports the decode rate, so a claim like
"QR codes tolerate occlusion" becomes a column of percentages that can be
checked and disagreed with.

The experiments:

1. **zoom-out retry** -- barcode decode rate with the retry on and off, which
   is the one change in this project that turns a total failure into a total
   success.
2. **occlusion** -- QR against EAN-13, the Reed-Solomon versus check-digit
   comparison.
3. **blur, rotation, resolution, noise** -- where each symbology gives up.
4. **detector speed** -- the cost of the classic and ArUco QR detectors, and
   of the barcode reader.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from src import synth  # noqa: E402
from src.detect import BarcodeReader, MultiReader, QRReader, ScanOptions, retail_key  # noqa: E402
from src.symbols import RenderedCode, render_barcode, render_qr  # noqa: E402

QR_PAYLOADS = (
    "https://example.com/scan",
    "https://medium.com/@someone/opencv-qr-scanner",
    "WIFI:T:WPA;S:Workshop;P:opencv2024;;",
    "tel:+441234567890",
    "Plain text payload for the scanner",
    "geo:51.5074,-0.1278",
)
EAN_PAYLOADS = (
    "400638133393",
    "978014300723",
    "590123412345",
    "123456789012",
    "888888888888",
    "500000000000",
)

FRAME = (1280, 720)


@dataclass(frozen=True, slots=True)
class Row:
    """One point of a sweep."""

    setting: str
    qr_rate: float | None = None
    ean_rate: float | None = None
    note: str = ""


def qr_scene(payload: str, *, module: int = 8) -> synth.Scene:
    code = render_qr(payload, module_size=module)
    return synth.place(code, width=FRAME[0], height=FRAME[1])


def ean_scene(digits: str, *, module: int = 4) -> synth.Scene:
    code = render_barcode(digits, module_width=module, height=69 * module)
    return synth.place(code, width=FRAME[0], height=FRAME[1])


def decoded_ok(reader: MultiReader, scene: synth.Scene) -> bool:
    """True when every code planted in the scene came back correctly."""
    found = reader.read(scene.image)
    keys = {retail_key(item.payload, item.symbology) for item in found if item.decoded}
    for placement in scene.placements:
        if retail_key(placement.payload, placement.symbology) not in keys:
            return False
    return True


def rate(reader: MultiReader, scenes: Sequence[synth.Scene]) -> float:
    if not scenes:
        return 0.0
    hits = sum(1 for scene in scenes if decoded_ok(reader, scene))
    return 100.0 * hits / len(scenes)


def qr_reader(**kwargs: object) -> MultiReader:
    detector = str(kwargs.get("detector", "classic"))
    return MultiReader([QRReader(detector=detector)], ScanOptions())


def ean_reader(*, zoom: float | None = None) -> MultiReader:
    return MultiReader([BarcodeReader(zoom_out=zoom)], ScanOptions())


def sweep(
    make_scene: Callable[[str], synth.Scene],
    payloads: Sequence[str],
    transform: Callable[[synth.Scene], synth.Scene],
) -> list[synth.Scene]:
    return [transform(make_scene(payload)) for payload in payloads]


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------


def crop_to_code(scene: synth.Scene, margin: int = 12) -> synth.Scene:
    """Cut the frame down to the code plus a thin margin.

    This is the "image that is nothing but a barcode" case -- what you get by
    saving a barcode from a web page -- and the one the zoom-out retry exists
    for. Ground-truth corners move with the crop.
    """
    placement = scene.placements[0]
    height, width = scene.image.shape[:2]
    xs, ys = placement.corners[:, 0], placement.corners[:, 1]
    left = max(0, int(xs.min()) - margin)
    top = max(0, int(ys.min()) - margin)
    right = min(width, int(xs.max()) + margin)
    bottom = min(height, int(ys.max()) + margin)
    shifted = placement.corners - np.array([left, top], dtype=np.float32)
    return synth.Scene(
        scene.image[top:bottom, left:right],
        (synth.Placement(placement.payload, placement.symbology, shifted),),
    )


def experiment_zoom_retry() -> tuple[str, list[Row]]:
    """The retry that makes barcode scanning work at all."""
    rows: list[Row] = []
    for module in (2, 3, 4, 6, 8):
        tight = [
            crop_to_code(ean_scene(digits, module=module)) for digits in EAN_PAYLOADS
        ]
        off = rate(ean_reader(zoom=1.0), tight)
        on = rate(ean_reader(), tight)
        rows.append(Row(f"{module} px/bar", ean_rate=on, note=f"{off:.0f}% without retry"))
    return "Barcode in a frame of its own size: zoom-out retry", rows


def experiment_occlusion() -> tuple[str, list[Row]]:
    """Reed-Solomon against a single check digit: damage to data modules."""
    rows: list[Row] = []
    for fraction in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
        qr = sweep(qr_scene, QR_PAYLOADS, lambda s: synth.occlude(s, fraction))
        ean = sweep(ean_scene, EAN_PAYLOADS, lambda s: synth.occlude(s, fraction))
        rows.append(
            Row(
                f"{fraction * 100:.0f}%",
                qr_rate=rate(qr_reader(), qr),
                ean_rate=rate(ean_reader(), ean),
            )
        )
    return "Centre of the code covered (data modules, landmarks intact)", rows


def experiment_occlusion_corner() -> tuple[str, list[Row]]:
    """The same patch moved to a corner, which measures something else."""
    rows: list[Row] = []
    for fraction in (0.0, 0.05, 0.10, 0.20, 0.30, 0.40):
        qr = sweep(qr_scene, QR_PAYLOADS, lambda s: synth.occlude(s, fraction, corner="tl"))
        ean = sweep(
            ean_scene, EAN_PAYLOADS, lambda s: synth.occlude(s, fraction, corner="tl")
        )
        rows.append(
            Row(
                f"{fraction * 100:.0f}%",
                qr_rate=rate(qr_reader(), qr),
                ean_rate=rate(ean_reader(), ean),
            )
        )
    return "Top-left corner covered (QR finder pattern destroyed)", rows


def experiment_error_levels() -> tuple[str, list[Row]]:
    """Does the error-correction level actually buy damage tolerance?

    This is the table that tests the claim rather than repeating it. Each row
    is one correction level; the figure is the largest centred patch, as a
    percentage of the symbol's own area, that every payload still survives.
    """
    rows: list[Row] = []
    fractions = (0.0, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20)
    for level in ("L", "M", "Q", "H"):

        def make(payload: str, level: str = level) -> synth.Scene:
            code = render_qr(payload, module_size=8, correction=level)
            return synth.place(code, width=FRAME[0], height=FRAME[1])

        survived = 0.0
        detected = 0.0
        for fraction in fractions:
            scenes = sweep(make, QR_PAYLOADS, lambda s: synth.occlude(s, fraction))
            if rate(qr_reader(), scenes) == 100.0:
                survived = fraction
            located = sum(
                1 for scene in scenes if qr_reader().read(scene.image)
            )
            if located == len(scenes):
                detected = fraction
        rows.append(
            Row(
                f"level {level}",
                note=f"decodes to {survived * 100:.0f}%, still located to {detected * 100:.0f}%",
            )
        )
    return "QR error correction: largest centred patch survived", rows


def experiment_blur() -> tuple[str, list[Row]]:
    rows: list[Row] = []
    for radius in (0, 1, 2, 3, 5, 8, 12):
        qr = sweep(qr_scene, QR_PAYLOADS, lambda s: synth.blur(s, radius))
        ean = sweep(ean_scene, EAN_PAYLOADS, lambda s: synth.blur(s, radius))
        rows.append(
            Row(f"{radius} px", qr_rate=rate(qr_reader(), qr), ean_rate=rate(ean_reader(), ean))
        )
    return "Gaussian blur radius", rows


def experiment_rotation() -> tuple[str, list[Row]]:
    rows: list[Row] = []
    for degrees in (0, 5, 15, 30, 45, 60, 90):
        qr = sweep(qr_scene, QR_PAYLOADS, lambda s: synth.rotate(s, degrees))
        ean = sweep(ean_scene, EAN_PAYLOADS, lambda s: synth.rotate(s, degrees))
        rows.append(
            Row(f"{degrees} deg", qr_rate=rate(qr_reader(), qr), ean_rate=rate(ean_reader(), ean))
        )
    return "Rotation in the image plane", rows


def experiment_resolution() -> tuple[str, list[Row]]:
    """How few pixels per module the decoders tolerate."""
    rows: list[Row] = []
    for factor in (1.0, 0.75, 0.5, 0.35, 0.25, 0.15):
        qr = sweep(qr_scene, QR_PAYLOADS, lambda s: synth.shrink(s, factor))
        ean = sweep(ean_scene, EAN_PAYLOADS, lambda s: synth.shrink(s, factor))
        rows.append(
            Row(
                f"x{factor:g}",
                qr_rate=rate(qr_reader(), qr),
                ean_rate=rate(ean_reader(), ean),
                note=f"{8 * factor:.1f} px/module QR, {4 * factor:.1f} px/bar EAN",
            )
        )
    return "Scene scaled down (camera further away)", rows


def experiment_perspective() -> tuple[str, list[Row]]:
    rows: list[Row] = []
    for strength in (0.0, 0.05, 0.10, 0.20, 0.30, 0.40):
        qr = sweep(qr_scene, QR_PAYLOADS, lambda s: synth.perspective(s, strength))
        ean = sweep(ean_scene, EAN_PAYLOADS, lambda s: synth.perspective(s, strength))
        rows.append(
            Row(
                f"{strength:g}",
                qr_rate=rate(qr_reader(), qr),
                ean_rate=rate(ean_reader(), ean),
            )
        )
    return "Perspective tilt strength", rows


def experiment_noise() -> tuple[str, list[Row]]:
    rows: list[Row] = []
    for sigma in (0.0, 10.0, 20.0, 40.0, 60.0, 90.0):
        qr = sweep(qr_scene, QR_PAYLOADS, lambda s: synth.noise(s, sigma))
        ean = sweep(ean_scene, EAN_PAYLOADS, lambda s: synth.noise(s, sigma))
        rows.append(
            Row(f"sigma {sigma:g}", qr_rate=rate(qr_reader(), qr), ean_rate=rate(ean_reader(), ean))
        )
    return "Additive sensor noise", rows


def experiment_speed(repeats: int = 5) -> tuple[str, list[Row]]:
    """Per-frame cost of each reader on a 1280x720 frame."""
    rows: list[Row] = []
    scenes = [qr_scene(payload) for payload in QR_PAYLOADS]
    ean_scenes = [ean_scene(digits) for digits in EAN_PAYLOADS]
    configs: list[tuple[str, MultiReader, list[synth.Scene]]] = [
        ("QR, classic detector", qr_reader(detector="classic"), scenes),
        ("QR, ArUco detector", qr_reader(detector="aruco"), scenes),
        ("Barcode, no retry needed", ean_reader(), ean_scenes),
        ("Barcode, retry exhausted", ean_reader(), [synth.Scene(s.image, ()) for s in [qr_scene(QR_PAYLOADS[0])]]),
    ]
    for name, reader, targets in configs:
        timings: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            for scene in targets:
                reader.read(scene.image)
            timings.append((time.perf_counter() - start) / max(1, len(targets)))
        median = statistics.median(timings) * 1000.0
        rows.append(Row(name, note=f"{median:.1f} ms/frame ({1000.0 / median:.0f} fps)"))
    return "Detector cost per 1280x720 frame", rows


def experiment_frame_cost(repeats: int = 5) -> tuple[str, list[Row]]:
    """How frame size drives the cost, which is the knob that actually matters.

    Reported as both detectors side by side, because the gap between them
    widens with resolution rather than staying proportional.
    """
    rows: list[Row] = []
    for width, height in ((640, 480), (960, 540), (1280, 720), (1920, 1080)):
        timings: dict[str, float] = {}
        scenes = [
            synth.place(render_qr(payload), width=width, height=height)
            for payload in QR_PAYLOADS[:3]
        ]
        for detector in ("classic", "aruco"):
            reader = qr_reader(detector=detector)
            samples: list[float] = []
            for _ in range(repeats):
                start = time.perf_counter()
                for scene in scenes:
                    reader.read(scene.image)
                samples.append((time.perf_counter() - start) / len(scenes))
            timings[detector] = statistics.median(samples) * 1000.0
        rows.append(
            Row(
                f"{width}x{height}",
                note=(
                    f"classic {timings['classic']:.0f} ms "
                    f"({1000.0 / timings['classic']:.0f} fps), "
                    f"aruco {timings['aruco']:.0f} ms "
                    f"({1000.0 / timings['aruco']:.0f} fps), "
                    f"{timings['classic'] / timings['aruco']:.1f}x"
                ),
            )
        )
    return "QR detection cost against frame size", rows


def experiment_detectors() -> tuple[str, list[Row]]:
    """Classic against ArUco on the degradations where they differ."""
    rows: list[Row] = []
    cases: list[tuple[str, Callable[[synth.Scene], synth.Scene]]] = [
        ("clean", lambda s: s),
        ("blur 3 px", lambda s: synth.blur(s, 3)),
        ("blur 5 px", lambda s: synth.blur(s, 5)),
        ("rotate 45 deg", lambda s: synth.rotate(s, 45)),
        ("tilt 0.3", lambda s: synth.perspective(s, 0.3)),
        ("scaled x0.35", lambda s: synth.shrink(s, 0.35)),
        ("occluded 15%", lambda s: synth.occlude(s, 0.15)),
        ("noise sigma 60", lambda s: synth.noise(s, 60.0)),
    ]
    for name, transform in cases:
        scenes = sweep(qr_scene, QR_PAYLOADS, transform)
        classic = rate(qr_reader(detector="classic"), scenes)
        aruco = rate(qr_reader(detector="aruco"), scenes)
        rows.append(Row(name, qr_rate=classic, ean_rate=aruco))
    return "QR decode rate: classic vs ArUco detector", rows


EXPERIMENTS: tuple[tuple[str, Callable[[], tuple[str, list[Row]]], tuple[str, str]], ...] = (
    ("zoom_retry", experiment_zoom_retry, ("Bar module", "Decoded")),
    ("occlusion", experiment_occlusion, ("Covered", "")),
    ("occlusion_corner", experiment_occlusion_corner, ("Covered", "")),
    ("error_levels", experiment_error_levels, ("Correction", "")),
    ("blur", experiment_blur, ("Blur", "")),
    ("rotation", experiment_rotation, ("Rotation", "")),
    ("resolution", experiment_resolution, ("Scale", "")),
    ("perspective", experiment_perspective, ("Tilt", "")),
    ("noise", experiment_noise, ("Noise", "")),
    ("detectors", experiment_detectors, ("Condition", "classic/aruco")),
    ("speed", experiment_speed, ("Reader", "")),
    ("frame_cost", experiment_frame_cost, ("Frame", "")),
)


def render_table(title: str, rows: list[Row], header: tuple[str, str]) -> str:
    """Format a sweep as a Markdown table, omitting columns with no data."""
    has_qr = any(row.qr_rate is not None for row in rows)
    has_ean = any(row.ean_rate is not None for row in rows)
    has_note = any(row.note for row in rows)
    left = header[0] or "Setting"
    columns = [left]
    if has_qr:
        columns.append("classic" if header[1] == "classic/aruco" else "QR")
    if has_ean:
        columns.append("ArUco" if header[1] == "classic/aruco" else "EAN-13")
    if has_note:
        columns.append("Note")
    lines = [f"**{title}**", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * (len(columns) - 1)) + "|")
    for row in rows:
        cells = [row.setting]
        if has_qr:
            cells.append("-" if row.qr_rate is None else f"{row.qr_rate:.0f}%")
        if has_ean:
            cells.append("-" if row.ean_rate is None else f"{row.ean_rate:.0f}%")
        if has_note:
            cells.append(row.note or "")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benchmark.py", description="Measure scanner robustness."
    )
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "results",
                        help="directory for the Markdown and JSON output")
    parser.add_argument("--only", action="append", default=None,
                        choices=[name for name, _, _ in EXPERIMENTS],
                        help="run only these experiments; repeat for several")
    args = parser.parse_args(argv)

    chosen = set(args.only) if args.only else {name for name, _, _ in EXPERIMENTS}
    blocks: list[str] = []
    data: dict[str, list[dict[str, object]]] = {}
    for name, run, header in EXPERIMENTS:
        if name not in chosen:
            continue
        print(f"running {name} ...", flush=True)
        title, rows = run()
        blocks.append(render_table(title, rows, header))
        data[name] = [
            {
                "setting": row.setting,
                "qr_rate": row.qr_rate,
                "ean_rate": row.ean_rate,
                "note": row.note,
            }
            for row in rows
        ]

    args.out.mkdir(parents=True, exist_ok=True)
    report = args.out / "benchmark.md"
    report.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    (args.out / "benchmark.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("\n" + "\n\n".join(blocks))
    print(f"\nWrote {report} and {args.out / 'benchmark.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
