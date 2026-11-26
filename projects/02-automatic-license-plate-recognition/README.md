# Project 2: Automatic License Plate Recognition

This Python project locates a vehicle plate in an image, video, or webcam feed,
rectifies its perspective, enhances the cropped plate, and reads its characters
with Tesseract OCR. Video and camera modes associate detections over time so OCR
does not have to run on every frame.

The implementation is designed as a transparent classical-computer-vision
tutorial. It is intentionally not a production traffic-enforcement system.

## What this project teaches

The pipeline separates three concerns:

- **Computer vision:** grayscale conversion, blur, edge detection,
  morphological closing, contour filtering, perspective correction, and plate
  enhancement.
- **OCR model:** Tesseract converts a normalized plate crop into text and a
  confidence value. Tesseract is a separate system program; `pytesseract` is
  only its Python adapter.
- **Application logic:** input/output, CLI validation, track association, OCR
  scheduling, annotation, FPS measurement, and cleanup.

```text
image / video / webcam
          |
          v
 grayscale -> Gaussian blur -> Canny edges -> morphology
          |
          v
 contours -> geometry filters -> candidate score
          |
          v
 order corners -> perspective warp -> threshold/enhance
          |
          +------> Tesseract OCR ------> normalized plate text
          |
          v
 track association -> cached OCR -> boxes, IDs, text, FPS
```

## Repository layout

```text
02-automatic-license-plate-recognition/
|-- README.md
|-- scripts/
|   `-- generate_samples.py       # deterministic image/video fixture generator
|-- assets/
|   |-- input/                    # generated or local source media
|   `-- output/                   # annotated media and plate crops
|-- results/                      # benchmark tables and experiment notes
`-- python/
    |-- alpr.py                   # command-line entry point
    |-- requirements.txt
    |-- src/                      # detection, OCR, tracking, and app modules
    `-- tests/
```

The two deterministic fixtures and their README-named annotated outputs are
allowlisted by the project `.gitignore`; unrelated local media remains ignored.
The generator makes the tutorial reproducible without using a real vehicle
registration number.

## Requirements

- Python 3.10 or newer
- Tesseract OCR with English language data
- A webcam only for live mode; the generated image/video work without hardware

From the repository root:

```powershell
cd projects/02-automatic-license-plate-recognition
py -3.12 -m venv .venv
./.venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r python/requirements.txt
```

On macOS or Linux, create the environment with `python3.12 -m venv .venv` and
activate it with `source .venv/bin/activate`.

### Install Tesseract

The Python package does not include the OCR executable. Follow the
[official Tesseract installation guide](https://tesseract-ocr.github.io/tessdoc/Installation.html)
for your platform, then verify both the executable and English data:

```text
tesseract --version
tesseract --list-langs
```

`eng` should appear in the language list.

- **Windows:** use the Windows installer referenced by the official guide. Add
  its installation directory (commonly `C:\Program Files\Tesseract-OCR`) to
  `PATH`, or pass the executable explicitly:

  ```powershell
  python python/alpr.py --input assets/input/synthetic_vehicle.png `
    --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"
  ```

- **Ubuntu/Debian:** `sudo apt install tesseract-ocr tesseract-ocr-eng`
- **macOS with Homebrew:** `brew install tesseract`

If Tesseract is installed but not found, open a new terminal after changing
`PATH` or use `--tesseract-cmd`.

## Generate safe sample inputs

The fixture generator uses only OpenCV and NumPy from the Python requirements.
It draws the same fictional `CV 2026` plate every time, including a mild
perspective skew, and creates a three-second moving-car video:

```powershell
python scripts/generate_samples.py
```

It creates:

```text
assets/input/synthetic_vehicle.png
assets/input/synthetic_drive.avi   # 72 MJPG frames at 24 FPS
```

The command refuses to overwrite existing fixtures. Regenerate them with
`--force`, or customize the duration with `--frames` and `--fps`:

```powershell
python scripts/generate_samples.py --force --frames 120 --fps 30
```

The scene is synthetic and deterministic in geometry, colors, motion, and
plate text. Encoded AVI bytes can still differ between OpenCV/codec builds.

## Run the application

Run the examples from `projects/02-automatic-license-plate-recognition`.
Use `python python/alpr.py --help` for the complete option list.

### Static image

```powershell
python python/alpr.py `
  --input assets/input/synthetic_vehicle.png `
  --output assets/output/synthetic_vehicle-annotated.png `
  --no-display
```

The application writes an annotated image. When a plate is found, its label
contains normalized OCR text and the candidate/OCR confidence that is
available.

### Video

```powershell
python python/alpr.py `
  --input assets/input/synthetic_drive.avi `
  --output assets/output/synthetic_drive-annotated.avi `
  --ocr-interval 8 --no-display
```

`--ocr-interval 8` allows OCR at most once every eight frames for an established
track. Detection and association still run every frame, and the most recent
text is cached on that track.

For a short headless smoke test:

```powershell
python python/alpr.py --input assets/input/synthetic_drive.avi `
  --no-display --max-frames 24 --ocr-interval 6
```

### Webcam

```powershell
python python/alpr.py --camera 0 --resolution 1280x720 --ocr-interval 10
```

Hold a plate-like printed card roughly level with the camera. Press `q` or
`Esc` to stop. Use only fictional text or a plate you are authorized to record.

## CLI guide

Input mode is automatic: `--input` is first attempted as an image and otherwise
opened as video. With no `--input`, the application opens `--camera`.

| Option | Purpose |
|---|---|
| `--input PATH` | Image or video source; takes precedence over the camera. |
| `--camera INDEX` | Webcam index used when no input path is supplied. |
| `--resolution WIDTHxHEIGHT` | Requested webcam capture resolution. |
| `--output PATH` | Annotated image or video destination. |
| `--no-display` | Disable GUI windows for automation or servers. |
| `--max-frames N` | Bound a video/camera run; `0` means no limit. |
| `--tesseract-cmd PATH` | Explicit Tesseract executable location. |
| `--ocr-interval N` | Minimum frames between OCR attempts on a track. |
| `--ocr-min-confidence VALUE` | Reject OCR readings below the selected confidence. |
| `--confidence VALUE` | Reject weak plate candidates. |
| `--min-area-ratio`, `--max-area-ratio` | Bound candidate area relative to the frame. |
| `--min-aspect-ratio`, `--max-aspect-ratio` | Bound width-to-height ratio. |
| `--min-rectangularity` | Reject contours that fill too little of their rectangle. |
| `--blur-kernel` | Odd Gaussian-blur kernel size. |
| `--canny-low`, `--canny-high` | Edge-detector hysteresis thresholds. |
| `--morph-kernel WIDTHxHEIGHT` | Kernel that joins characters/plate edges. |
| `--morph-iterations N` | Number of morphology passes. |
| `--max-candidates N` | Cap expensive rectification/OCR work per frame. |
| `--tracker-iou VALUE` | Minimum overlap for detection-to-track association. |
| `--max-missed N` | Frames a track may survive without a matching detection. |

Image output requires an image extension. Video/camera output accepts `.avi`,
`.mp4`, `.mov`, or `.mkv`; `.avi` with MJPG is usually the most portable. Input
and output must not refer to the same path, because annotated output would
otherwise destroy the source.

## How plate detection and recognition work

### 1. Locate plate-shaped regions

The detector converts a BGR frame to grayscale and applies Gaussian blur to
reduce sensor noise. Canny keeps strong intensity transitions, including plate
borders and character strokes. A rectangular morphological kernel closes short
gaps so those edges form usable contours.

Contours are not automatically plates. Each candidate is filtered by its area
relative to the frame, width-to-height ratio, and rectangularity. The remaining
candidates receive a geometry-based confidence score and are sorted before the
configured candidate cap is applied. These relative filters behave more
consistently across resolutions than fixed pixel sizes.

### 2. Rectify and enhance

The four candidate corners are ordered top-left, top-right, bottom-right, and
bottom-left. A homography maps that quadrilateral to a front-facing rectangle.
The crop is converted to grayscale and thresholded/enhanced so dark characters
stand apart from the light background. Perspective correction is important:
OCR accuracy drops quickly when character rows are slanted or compressed.

### 3. Read and normalize text

Tesseract runs in a single-line mode with a license-plate character whitelist.
The application discards punctuation/noise, normalizes whitespace/case, and
rejects readings below `--ocr-min-confidence`. A high candidate confidence does
not imply correct text; localization confidence and OCR confidence measure
different stages.

### 4. Track before repeating OCR

For video, detections are associated with recent tracks using bounding-box
intersection over union (IoU). A track keeps its identifier and last accepted
text through short missed detections. OCR runs for a new track and then only
when its configured interval elapses; cached text is drawn between attempts.

This is a lightweight association tracker rather than an appearance or motion
model. It prevents redundant OCR work and reduces label flicker, but two plates
that cross or overlap can exchange identities.

## Reproducible experiments

Use a recorded clip so every configuration sees identical frames. Warm up once,
then run at least three measured passes per configuration with rendering and
encoding either enabled for all runs or disabled for all runs.

### Detection and OCR quality

Label a representative frame sample with ground-truth plate boxes and text.
Count a localization as correct at IoU `>= 0.5`, and report exact full-plate
match plus character accuracy separately.

| Configuration | Lighting | Resolution | Precision | Recall | Exact OCR | Character accuracy |
|---|---|---:|---:|---:|---:|---:|
| Default | normal | 1280x720 | | | | |
| Tuned morphology | dim | 1280x720 | | | | |

### OCR throttling and tracking

Compare `--ocr-interval 1`, `5`, `10`, and `20`. Record processing FPS, OCR
calls, correct-text frames, visible label flicker, and track-ID switches.

| OCR interval | Mean FPS | OCR calls | Correct-text frames | ID switches | Notes |
|---:|---:|---:|---:|---:|---|
| 1 | | | | | OCR every frame |
| 5 | | | | | |
| 10 | | | | | |
| 20 | | | | | |

### Robustness

Test bright, dim, backlit, motion-blurred, rotated, near, and small/distant
plates. Change only one detector parameter at a time. Record the OpenCV,
Tesseract, and Python versions; CPU model; operating system; run count; input
file hash; and whether display/output encoding was enabled. Store raw notes or
CSV tables under `results/` and representative annotations under
`assets/output/`.

## Tests

The automated tests do not require a webcam or a real registration plate:

```powershell
python -m unittest discover -s python/tests -v
```

Then run the generated image and bounded-video commands above as integration
smoke tests. Unit tests can validate geometry, text normalization, association,
CLI checks, and cleanup, but only an end-to-end run proves that the local
OpenCV codecs and Tesseract executable are available.

## Error handling and troubleshooting

- **`tesseract` is not recognized / not found:** install the executable, restart
  the terminal after updating `PATH`, or pass `--tesseract-cmd`.
- **`eng` trained data is missing:** install English data and confirm it appears
  in `tesseract --list-langs`; check `TESSDATA_PREFIX` if using a custom layout.
- **Input file cannot be opened:** check the path and codec. Generate the known
  PNG/AVI fixtures to separate an environment problem from an input problem.
- **Camera cannot be opened:** close other camera applications, grant OS camera
  permission, and try another `--camera` index.
- **No plate candidate:** use a larger, sharper, more front-facing plate; inspect
  edge contrast; then tune area/aspect/rectangularity and morphology limits.
- **OCR text is wrong:** inspect the rectified crop first. Improve focus and
  lighting, reduce perspective, and tune thresholding before lowering OCR
  confidence.
- **Output writer cannot open:** make sure the parent directory exists, prefer
  MJPG `.avi`, and verify the destination is writable.
- **GUI errors on a headless machine:** add `--no-display` and usually
  `--output` or `--max-frames`.
- **Unstable IDs or text:** lower the OCR interval only if needed, then tune
  tracker IoU/missed-frame tolerance for the motion in the clip.
- **Low FPS:** disable output encoding/display for compute-only measurement,
  lower the input resolution/candidate cap, or increase `--ocr-interval`.

The program reports expected user/environment failures with a non-zero exit
status. Use the generated fixtures first when diagnosing a custom input.

## Limitations and responsible use

- The contour heuristic assumes a reasonably visible, rectangular,
  high-contrast plate. Decorative frames, unusual aspect ratios, glare, severe
  perspective, blur, occlusion, and low resolution can defeat it.
- Plate layouts, fonts, scripts, and legal formats vary by country. A whitelist
  that helps the fictional Latin fixture may reject valid local characters.
- Tesseract is a general OCR engine, not a plate-specific recognizer. A learned
  plate detector and sequence recognizer will usually be more robust.
- IoU association is deliberately simple and does not estimate velocity or
  appearance. It is vulnerable to fast motion, long occlusion, and crossing
  vehicles.
- The generated fixture validates plumbing, not real-world accuracy. Report
  results from representative, consented data before making performance claims.
- Plates can identify or help track people. Follow local law, obtain permission,
  minimize retention, secure outputs, and avoid covert surveillance. Never use
  this tutorial alone for enforcement or other high-stakes decisions.

Useful future extensions include a trained plate detector, country-aware text
validation, optical-flow/Kalman tracking, OCR consensus across several sharp
frames, automatic blur/glare rejection, and structured CSV export.
