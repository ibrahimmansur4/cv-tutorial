# YOLO + OpenCV Real-Time Object Detection

Detect the 80 COCO object classes in a webcam stream, a video file, or a still
image, using a pretrained YOLO model run entirely through OpenCV's DNN module.

There is no PyTorch and no `ultralytics` runtime here. The models are ONNX
exports, and every stage around them — letterboxing, the forward pass, decoding
the raw output grid, non-maximum suppression, drawing — is OpenCV and NumPy.
That keeps the dependency list to three packages and makes each step of the
detection pipeline something you can read.

## Features

- Webcam, video-file, and single-image input
- Four pretrained models across two different YOLO output layouts
- Per-class object counting with a live FPS readout
- Configurable confidence and NMS thresholds
- Class filtering by name or index (`--classes person,car`)
- Aspect-ratio-preserving letterbox preprocessing
- Per-stage timing breakdown (`--show-timing`)
- Annotated video and image export
- SHA-256-verified model downloader
- A benchmark script that measures model size and input resolution

## Requirements and setup

- Python 3.10 or newer (verified with Python 3.12)
- A webcam for live use
- No GPU required; everything below was measured on a 4-core laptop CPU

From this project directory on Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r python/requirements.txt
python scripts/download_model.py
```

On macOS/Linux, activate with `source .venv/bin/activate`.

The downloader fetches the default model (`yolov5n`, 3.8 MB), verifies its
SHA-256 digest, and stores it in `models/`. Models are ignored by Git.

```powershell
python scripts/download_model.py --model all      # every registered model
python scripts/download_model.py --list           # show the registry
```

## Run

Live webcam detection:

```powershell
python python/detect.py
```

A single image, saving the annotated result:

```powershell
python python/detect.py --input assets/input/bus.jpg --output assets/output/bus-annotated.jpg
```

A video file, with no window:

```powershell
python python/detect.py `
  --input assets/input/vtest.avi `
  --output assets/output/vtest-annotated.avi `
  --no-display --model yolo11n --input-size 480
```

Count only people and vehicles, at a higher threshold:

```powershell
python python/detect.py --classes person,car,truck,bus --confidence 0.5
```

`python python/detect.py --help` lists every option; `--list-models` shows what
is registered and which models are already downloaded. `--max-frames N` is
useful for unattended smoke tests, and `--show-timing` overlays where the
milliseconds are going.

Printed output follows the per-class count format:

```text
person: 3
bus:    1

FPS: 10.2
```

## Models

| Name | Layout | Size | Input | Notes |
|---|---|---:|---|---|
| `yolov5n` | `yolov5` | 3.8 MB | 640x640 fixed | default; fastest |
| `yolov5s` | `yolov5` | 14.0 MB | 640x640 fixed | balanced |
| `yolov5m` | `yolov5` | 40.6 MB | 640x640 fixed | most accurate, slowest |
| `yolo11n` | `yolov8` | 10.4 MB | any multiple of 32 | anchor-free head |

Two things are worth knowing before choosing one.

**The YOLOv5 exports have a fixed input size.** Their ONNX graphs carry baked-in
`640x640` reshape dimensions, so passing `--input-size 320` fails inside the
forward pass. The project checks this up front and tells you so, rather than
surfacing an assertion from deep inside OpenCV. `yolo11n` was exported with
dynamic axes and does accept other sizes, which is why it is the model the
resolution benchmark sweeps.

**`--family` describes the output layout, not the file format.** All four files
are ONNX. What differs is how the numbers inside the output tensor are
arranged, which is the next section.

## How it works

The project separates three layers that are easy to conflate:

1. **Machine-learning model.** A pretrained convolutional network predicts, in a
   single pass, a dense grid of candidate boxes with class scores. Nothing here
   trains or fine-tunes it; it is used exactly as exported.
2. **Computer-vision algorithm.** Letterboxing, blob construction, decoding the
   output grid, non-maximum suppression, and the inverse coordinate mapping.
   This is the part this project implements, and it is all classical geometry
   and array manipulation.
3. **Application logic.** Frame capture, counting, FPS smoothing, the class
   filter, drawing, video writing, and the CLI.

### Letterboxing, and why not just resize

A YOLO network takes a fixed square input, but a webcam frame is 16:9. Stretching
it into a square distorts every object: people become narrow, wide objects become
tall. The model was trained on undistorted shapes, so it detects distorted ones
less reliably.

[`letterbox.py`](python/src/letterbox.py) scales by a single factor that makes the
frame fit, then pads the remainder with neutral gray (value 114, matching how
these models were validated). Geometry is preserved; the only cost is that the
padded strips carry no information.

Because the network then reports boxes in *padded* coordinates, the padding and
scale must be undone before anything is drawn. The forward and inverse transforms
are stored together in one object precisely so they cannot drift apart.

### The two output layouts

This is the part most tutorials gloss over, and the reason the same decoder cannot
be copy-pasted between YOLO versions.

| Family | Raw shape | Row layout |
|---|---|---|
| `yolov5` | `(1, 25200, 85)` | `cx cy w h obj c0..c79` |
| `yolov8` (v8–v12) | `(1, 84, 8400)` | `cx cy w h c0..c79`, channel-major |

Two structural differences follow from that table:

- **YOLOv8 and later dropped the objectness column.** YOLOv5 predicts a separate
  "is there an object here at all" score, and the final confidence is objectness
  multiplied by the best class probability — so a box the model doubts contains
  anything stays suppressed even when one class dominates. Anchor-free heads
  since v8 removed that column, so the class score *is* the confidence, giving
  84 values per box instead of 85.
- **The YOLOv8 output is transposed.** Attributes are on the first axis and boxes
  on the second, so it has to be transposed before its rows can be read like
  YOLOv5's. Feeding it to the wrong decoder does not crash; it silently produces
  nonsense, which is why `--family` exists and why the decoder validates widths.

[`decode.py`](python/src/decode.py) handles both and emits one `RawDetections`
type, so no other module branches on model family.

### Non-maximum suppression

A single-stage detector predicts many overlapping boxes for the same object —
25200 candidates per frame for YOLOv5 at 640. After thresholding, NMS keeps the
highest-scoring box in each cluster and discards anything overlapping it by more
than `--nms` IoU.

This project uses **per-class** NMS (`cv2.dnn.NMSBoxesBatched`). Class-agnostic
NMS would let a person box suppress the box of the tie they are wearing, since
those genuinely overlap. Boxes only compete against boxes of the same class.

## Benchmark

```powershell
python scripts/benchmark.py
python scripts/benchmark.py --frames 30 --image assets/input/bus.jpg
```

Two sweeps are run, because the registered exports differ in what they allow: a
model-size sweep at the native 640x640, and a resolution sweep on `yolo11n`, the
only export with dynamic input axes. Results are written to
`results/benchmark.csv` and printed as Markdown.

Detection *quality* has no ground truth here, so it is reported indirectly, as
the object count and mean confidence at a fixed threshold. A configuration that
finds more objects at similar confidence is finding more of what is present.

Measured on an Intel 4-core/8-thread laptop CPU, Python 3.12, OpenCV 5.0.0,
15 frames per configuration on `bus.jpg` (810x1080, 5 objects), confidence 0.25:

### Model size, at the native 640x640 input

| Model | FPS | Total ms | Pre | Net | Post | Objects | Mean conf | CPU % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| yolov5n | 12.9 | 77.3 | 6.3 | 65.2 | 5.8 | 4.0 | 0.695 | 58.5 |
| yolov5s | 4.4 | 225.2 | 7.9 | 210.0 | 7.3 | 5.0 | 0.748 | 67.0 |
| yolov5m | 1.3 | 773.9 | 11.4 | 751.9 | 10.7 | 5.0 | 0.827 | 64.5 |

### Input resolution, `yolo11n`

| Input | FPS | Total ms | Pre | Net | Post | Objects | Mean conf | CPU % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 320x320 | 14.8 | 67.5 | 12.5 | 53.3 | 1.7 | 5.0 | 0.716 | 35.4 |
| 480x480 | 11.8 | 84.5 | 7.3 | 73.3 | 3.9 | 5.0 | 0.773 | 63.6 |
| 640x640 | 8.0 | 124.5 | 9.2 | 111.3 | 4.1 | 5.0 | 0.78 | 73.7 |
| 960x960 | 3.2 | 312.4 | 18.8 | 283.1 | 10.5 | 5.0 | 0.812 | 65.0 |

What the numbers show:

- **Inference dominates.** Preprocessing and postprocessing together never exceed
  29.27 ms, while the forward pass ranges from 53.28 ms to 751.86 ms. Optimising
  the drawing code would be wasted effort.
- **Model size costs roughly linearly in parameters and sharply in time.**
  `yolov5m` is 10.7x the file size of `yolov5n` and 10.0x slower, buying one
  extra detection and a confidence rise from 0.695 to 0.827. On a CPU, the nano
  model is the only one near interactive speed.
- **Resolution is the cheaper quality dial.** Going 320 to 960 costs about 5x the
  time for a confidence rise of 0.716 to 0.812. On this image all four sizes find
  all five objects, but small or distant objects are exactly what lower
  resolutions start to miss — a 320x320 input leaves a distant pedestrian only a
  handful of pixels.
- **CPU utilisation stays in a similar band** across configurations, because
  OpenCV already parallelises the forward pass across cores. Larger models do not
  use *more* CPU; they use it for longer.

Absolute numbers vary with the machine and with background load; the trends are
the point. Re-run the script to get figures for your own hardware.

## Test

From `python/`:

```powershell
python -m unittest discover -s tests -v
```

55 tests covering: both output layouts and their equivalence on identical boxes,
threshold filtering, malformed-output rejection, letterbox geometry and its
inverse, box clipping, class-filter parsing, per-class and same-class NMS
behaviour, confidence ordering, blob sizing, timing fields, error paths for
missing/non-ONNX models and failed forward passes, counting, FPS averaging, the
drawing edge cases, registry consistency, CLI parsing and validation, and the
video/image loops with recorded input and output.

The tests inject a fake network, so they run in under a second and need no
downloaded model.

For a real end-to-end check after downloading a model:

```powershell
python python/detect.py --input assets/input/bus.jpg --no-display --confidence 0.4
```

`bus.jpg` is the standard Ultralytics sample; the expected result is 3 people and
1 bus.

## Project layout

```text
05-yolo-real-time-object-detection/
├── README.md
├── models/                     # downloaded ONNX models (git-ignored)
├── assets/
│   ├── input/                  # source images and videos
│   └── output/                 # annotated results
├── results/                    # benchmark.csv
├── scripts/
│   ├── download_model.py       # registry-driven, SHA-256 verified
│   └── benchmark.py            # model-size and resolution sweeps
└── python/
    ├── detect.py               # entry point
    ├── requirements.txt
    ├── src/
    │   ├── app.py              # CLI, media loop
    │   ├── decode.py           # raw output grid -> candidates
    │   ├── detector.py         # cv2.dnn wrapper, NMS
    │   ├── drawing.py          # boxes, labels, count panel
    │   ├── errors.py           # expected failures
    │   ├── labels.py           # COCO names, colors, class filter
    │   ├── letterbox.py        # aspect-preserving resize + inverse
    │   ├── models.py           # model registry
    │   └── pipeline.py         # per-frame orchestration, FPS
    └── tests/
        └── test_detection.py
```

## Common problems

**`model not found` —** run `python scripts/download_model.py`.

**`was exported with a fixed 640x640 input` —** that model cannot be resized.
Drop `--input-size`, or switch to `--model yolo11n`.

**`could not open video source: 0` —** another program is using the camera, or
the index is wrong. Try `--camera 1`.

**Detections look like nonsense —** `--family` is probably wrong for a custom
`--model`. A YOLOv5-layout tensor read as YOLOv8 decodes without error but
produces meaningless boxes.

**Everything is slow —** this is CPU inference. Use `yolov5n` or `yolo11n`, and
with `yolo11n` lower `--input-size` to 320 or 480.

**Too many overlapping boxes —** raise `--confidence`, or lower `--nms` to
suppress more aggressively.

**`.weights`/`.cfg` model fails to load —** OpenCV 5 removed the Darknet
importer. Only ONNX models work; this is a change from OpenCV 4.

## Limitations and possible improvements

- **CPU-only, so not truly real time for the larger models.** A CUDA-enabled
  OpenCV build, or an ONNX Runtime backend, would change the picture
  substantially. OpenCV 5's graph engine currently ignores
  `setPreferableBackend`/`setPreferableTarget`.
- **COCO's 80 classes only.** Anything outside that list is invisible. A custom
  domain needs a fine-tuned model, which needs a training framework this project
  deliberately avoids.
- **Detection, not tracking.** Each frame is independent, so objects have no
  identity across frames and counts flicker when a detection drops out for a
  frame. Project 14 covers trackers; pairing one with this detector is the usual
  production arrangement.
- **No ground-truth evaluation.** Mean confidence is a proxy for quality, not
  mAP. A real comparison needs a labelled validation set.
- **Fixed-size YOLOv5 exports** limit the resolution experiment to one model.
  Re-exporting with dynamic axes would need `ultralytics` and PyTorch.
- **Letterbox padding is wasted computation.** A 16:9 frame fits into 640x360 of
  a 640x640 input, so about 44% of the forward pass convolves gray bars.
  Rectangular inference avoids this but requires an export shaped for it.
