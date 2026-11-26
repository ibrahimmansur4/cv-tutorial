# Virtual Mouse Using Hand Tracking

Control or safely preview a mouse pointer using one hand, a webcam, OpenCV, and MediaPipe Hand Landmarker. The default mode never touches the operating-system pointer. Real mouse control requires the explicit `--enable-control` flag.

## Features

- Live webcam capture or recorded-video input
- MediaPipe's 21-point hand landmark model
- Index-finger cursor mapping with a configurable active margin
- Thumb/index pinch for left click
- Index/middle two-finger gesture for right click
- Closed fist for click-and-hold dragging
- Exponential cursor smoothing, a dead zone, stable-frame debounce, and click cooldown
- Landmark, gesture, action, pinch-ratio, active-region, and FPS overlays
- Optional annotated video output
- Safe preview mode, PyAutoGUI fail-safe support, and guaranteed drag release during shutdown
- Deterministic smoothing experiment and a dependency-free `unittest` suite

## Safety first

Start in preview mode and tune the gestures before enabling control:

```powershell
python python/virtual_mouse.py
```

The green `SAFE PREVIEW` overlay shows which actions would be issued without moving or clicking the real pointer. Press `Q` or `Esc` in the OpenCV window to quit. `Ctrl+C` also stops the program. Losing the tracked hand or shutting down releases an active drag.

When `--enable-control` is used, PyAutoGUI's corner fail-safe remains enabled: moving the pointer to a screen corner aborts PyAutoGUI operations. Recorded videos are deliberately forbidden from controlling the real pointer.

## Requirements and setup

- Python 3.10 or newer (the project was verified with Python 3.12)
- A webcam for live use
- Windows, macOS, or Linux desktop access for real pointer control
- No dedicated GPU is required

From this project directory on Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r python/requirements.txt
python scripts/download_model.py
```

On macOS/Linux, activate with `source .venv/bin/activate`. The downloader retrieves the pinned official `hand_landmarker.task` bundle, validates its SHA-256 digest, and stores it under `models/`. The model is intentionally ignored by Git.

If the model has already been downloaded, the command only verifies it. Use `--force` to replace it.

## Run

Safe webcam preview:

```powershell
python python/virtual_mouse.py
```

Process a recorded clip and save an annotated result:

```powershell
python python/virtual_mouse.py `
  --input assets/input/hand-demo.mp4 `
  --output assets/output/hand-demo-annotated.avi `
  --no-display
```

Enable real mouse control only after previewing the gestures:

```powershell
python python/virtual_mouse.py --enable-control
```

Example tuning command:

```powershell
python python/virtual_mouse.py `
  --resolution 1280x720 `
  --smoothing-alpha 0.25 `
  --dead-zone 3 `
  --stable-frames 4 `
  --click-cooldown 0.6 `
  --active-margin 0.10
```

Use `python python/virtual_mouse.py --help` for every option. `--max-frames N` is useful for unattended smoke tests, and `--camera N` selects another camera.

## Gestures

| Hand pose | Action |
|---|---|
| Index extended; middle, ring, and pinky folded | Move |
| Thumb and extended index pinched | Left click once on gesture entry |
| Index and middle extended; ring and pinky folded | Right click once on gesture entry |
| Index, middle, ring, and pinky folded | Start or continue left-button drag |

Finger extension is determined from joint angles, so it does not depend on whether a left or right hand is used. Pinch distance is divided by palm size, making the threshold less sensitive to distance from the camera. A gesture must remain stable for several frames before it becomes active; click gestures must also leave and re-enter before firing again.

## How it works

The project intentionally separates three layers:

1. **Machine-learning model:** MediaPipe Hand Landmarker infers 21 normalized 3D landmarks and handedness from each RGB frame. VIDEO mode reuses tracking between frames.
2. **Computer-vision and geometry logic:** OpenCV captures, mirrors, annotates, and writes frames. Joint angles determine finger extension, normalized thumb/index distance detects a pinch, and an active camera rectangle maps to the complete screen.
3. **Application logic:** A temporal state machine debounces poses, rate-limits clicks, manages drag start/end, smooths pointer coordinates, and routes commands to either a no-side-effect preview controller or PyAutoGUI.

MediaPipe receives strictly increasing millisecond timestamps. The first detected hand drives the cursor; other configured hands are displayed but ignored for control.

## Smoothing experiment

Run the deterministic stationary-pointer simulation:

```powershell
python scripts/smoothing_experiment.py
```

It adds repeatable Gaussian landmark noise, compares raw movement (`alpha=1`) against multiple exponential filters, prints steady-state RMS jitter, and writes per-frame data to `results/smoothing_comparison.csv`.

Results from the default 600-frame run:

| Mode | RMS jitter |
|---|---:|
| Raw / alpha 1.00 | 7.138 px |
| Alpha 0.65 | 4.878 px |
| Alpha 0.35 (application default) | 3.180 px |
| Alpha 0.15 | 1.912 px |

Smaller alpha values reject more noise but add more cursor lag. `alpha=1` is raw mapped motion. The dead zone separately suppresses tiny filtered movements; increase it carefully because a large value makes fine pointer movement difficult.

Camera-coordinate movement jitters because landmark inference varies slightly between frames due to sensor noise, compression, lighting, motion blur, partial occlusion, and small changes in the model's estimated hand pose. Mapping a small active camera region across a much larger display magnifies those sub-pixel/normalized-coordinate variations. Temporal low-pass filtering reduces high-frequency variation but necessarily trades responsiveness for stability.

## Test

From `python/`:

```powershell
python -m unittest discover -s tests -v
```

The tests do not control the real mouse. They cover geometry, every gesture, smoothing, cooldown/debounce, controller adapters, timestamp and landmark conversion, pipeline actions and drag cleanup, command-line safeguards, and recorded-video input/output.

For a model/runtime smoke test from `python/`:

```powershell
python -c "import numpy as np; from src.hand_tracking import MediaPipeHandTracker; t=MediaPipeHandTracker('../models/hand_landmarker.task'); print(t.detect(np.zeros((240,320,3), dtype=np.uint8), 0)); t.close()"
```

An empty list is the expected result for the blank frame.

## Project layout

```text
04-virtual-mouse-hand-tracking/
├── README.md
├── assets/
│   ├── input/
│   └── output/
├── models/                    # downloaded task bundle (Git-ignored)
├── python/
│   ├── requirements.txt
│   ├── virtual_mouse.py
│   ├── src/
│   └── tests/
├── results/                   # generated experiment CSV (Git-ignored)
└── scripts/
    ├── download_model.py
    └── smoothing_experiment.py
```

## Common problems

- **Model not found:** Run `python scripts/download_model.py` from the project root.
- **Camera will not open:** Close other camera applications and try another `--camera` index.
- **Window/display error:** Use `--no-display`; optionally save output with `--output`.
- **Pointer is noisy:** Lower `--smoothing-alpha`, increase `--dead-zone` slightly, improve lighting, and keep the hand inside the gray rectangle.
- **Clicks trigger too easily:** Lower `--pinch-threshold`, increase `--stable-frames`, or increase `--click-cooldown`.
- **Cursor feels delayed:** Raise `--smoothing-alpha` or reduce `--stable-frames`.
- **PyAutoGUI cannot access the desktop:** Grant accessibility/input-control permission to the terminal on macOS or the relevant desktop session on Linux.

## Limitations and improvements

- Gesture thresholds are geometric heuristics and vary with pose, viewpoint, and hand anatomy.
- Only the first detected hand controls the pointer.
- A fist starts dragging anywhere; an explicit clutch or calibration gesture would reduce accidental drags.
- Exponential smoothing adds lag. A One Euro or Kalman filter can adapt better to slow and fast motion.
- A calibration screen, handedness-specific gestures, and user profiles would improve ergonomics.
- Camera inference and desktop control should not be used for safety-critical interaction.

Model/API references: [MediaPipe Hand Landmarker for Python](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python) and [Hand Landmarker model overview](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/index).
