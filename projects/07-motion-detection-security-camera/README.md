# Motion Detection Security Camera

Turn a webcam into a motion-triggered recorder: watch a scene, detect movement
against a learned background, and save a timestamped clip of each event with a
thumbnail, an event log, and a local dashboard to review them.

Everything runs on OpenCV and NumPy. Nothing is uploaded anywhere.

## Features

- Webcam and recorded-video input
- Two background models — hand-rolled running-average differencing, and OpenCV's MOG2
- Full classical pipeline: grayscale → blur → subtract → threshold → morphology → contours → area filter
- **Pre-roll buffer**, so a clip includes the seconds *before* the trigger
- Post-roll cooldown, so one event does not shatter into a dozen files
- Minimum-length filter, so a noise spike leaves no stray file behind
- Timestamp burned into every frame; timestamped clip and snapshot per event
- CSV and JSON event logs
- Self-contained local dashboard with players and thumbnails
- Side-by-side mask view (`--show-mask`) for seeing what the pipeline sees
- Ground-truth scoring of both models against a synthetic clip

## Requirements and setup

- Python 3.10 or newer (verified with Python 3.12)
- A webcam for live use
- No GPU required

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r python/requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate`. There is no model to
download — this project is pure classical computer vision.

## Run

Watch the webcam and record events:

```powershell
python python/security_camera.py
```

See what the pipeline sees, which is the fastest way to tune it:

```powershell
python python/security_camera.py --show-mask
```

Analyse a recorded clip:

```powershell
python python/security_camera.py --input assets/input/surveillance-demo.avi --no-display
```

Use the adaptive model, and tune sensitivity:

```powershell
python python/security_camera.py --background mog2 --min-area 1200 --post-roll 5
```

Detect without writing anything:

```powershell
python python/security_camera.py --no-record
```

`--help` lists every option. Press `q` or `Esc` to stop; any event still in
progress is finished and saved.

Clips land in `recordings/` as `motion_YYYYMMDD_HHMMSS_NNN.avi` with a matching
`.jpg` thumbnail, and the log in `results/events.csv` and `events.json`.

## Dashboard

```powershell
python scripts/dashboard.py --serve
```

Builds `recordings/index.html` from the event log and serves it on
`127.0.0.1:8000`, newest event first, each with a player, thumbnail, timestamp,
duration and frame count. The server binds to loopback only — these clips are of
someone's home or office and have no business being reachable from the network.

Without `--serve` the page is still written, but browsers block video playback
from `file://` in many configurations, so HTTP is the reliable way to view it.

## How it works

```
frame -> grayscale -> blur -> background subtraction -> threshold
      -> morphology -> contours -> area filter -> motion regions -> recorder
```

Each stage removes a specific kind of false positive:

- **Grayscale** discards colour, which carries almost no motion information but triples the work.
- **Blur** removes sensor noise. Without it every frame differs from every other
  frame at the pixel level, and the mask is permanent static. This is the single
  most important step, and the one most often left out.
- **Threshold** turns "how different" into "different enough".
- **Morphology** — opening then closing — erases isolated specks, then seals the
  holes thresholding punched inside a real object. The order matters: closing
  first would enlarge the specks before trying to remove them.
- **Area filter** drops regions too small to be worth recording.

### The two background models

Motion detection is subtraction: you need a picture of the empty scene, and
anything differing from it is motion. The whole problem is keeping that picture
current, because scenes are never static — light shifts, clouds pass,
auto-exposure hunts, a chair gets moved and stays moved.

**Running average** is the classic method, implemented by hand here because
every step is visible:

```
background = (1 - alpha) * background + alpha * frame
```

A small `alpha` is a long memory: slow to be fooled by a moving subject, slow to
forget a light change. A large `alpha` adapts fast but swallows slow subjects,
which visibly fade into the background. The accumulation is done in float32 —
averaging 8-bit values in place would quantise away the small per-frame updates
a low alpha depends on.

**MOG2** keeps a small mixture of Gaussians per pixel instead of one average, so
a pixel that legitimately alternates between two values — foliage against sky, a
flickering screen — can be modelled as background. It also flags shadows
separately, which this project discards: a shadow is a real brightness change
but not an object, and keeping it merges a person with the ground they stand on
into one oversized box.

### Which is better? Not a simple answer

`compare_backgrounds.py` grades both against a clip whose motion times are known,
scoring every frame as a prediction against a label. On an Intel 4-core laptop,
16 s at 20 FPS, two walk-throughs:

**Clean scene:**

| Model | Precision | Recall | F1 | FP | FN | ms/frame |
|---|---:|---:|---:|---:|---:|---:|
| running-average | 1.000 | 0.859 | 0.924 | 0 | 18 | 7.4 |
| mog2 | 1.000 | 0.680 | 0.809 | 0 | 41 | 10.3 |

**Same scene plus a one-second lighting change:**

| Model | Precision | Recall | F1 | FP | FN | ms/frame |
|---|---:|---:|---:|---:|---:|---:|
| running-average | 0.743 | 0.859 | 0.797 | 38 | 18 | 6.3 |
| mog2 | 0.813 | 0.680 | 0.740 | 20 | 41 | 10.7 |

Neither model wins outright, which is the interesting part:

- **On a clean scene both are perfectly precise**, and the simple method has
  clearly better recall. MOG2 adapts faster, so a slowly-moving subject starts
  being absorbed into its background model and frames get missed.
- **Under a lighting change the simple method's precision collapses** from 1.000
  to 0.743 — 38 frames of recording a room where nothing happened. MOG2 degrades
  too but far less, which is exactly what the per-pixel mixture is for.
- **MOG2 costs about 40% more time per frame** on the clean scene, and more than
  that once lighting changes. Both are still far faster than real time at this
  resolution.

So: a stable indoor scene is fine with running-average and you get better
recall. Anywhere light changes — a window, a doorway, outdoors — MOG2 earns its
cost. Recall and precision trade against each other, and which to favour depends
on whether you would rather miss an intruder or fill a disk with clips of
clouds.

Reproduce it with:

```powershell
python scripts/make_test_clip.py
python scripts/compare_backgrounds.py
python scripts/make_test_clip.py --light-change --output assets/input/light-change-demo.avi
python scripts/compare_backgrounds.py --input assets/input/light-change-demo.avi
```

### Recording decisions that matter more than they look

**Pre-roll.** A recorder that opens its file when motion is detected has already
missed the approach. Recent frames are kept in a ring buffer and flushed into
the clip when an event starts, so the interesting second *before* the trigger
survives. This is the difference between a usable clip and a clip of someone
already standing in your kitchen.

**Post-roll.** Detection is intermittent — a person pausing, or briefly matching
the background, drops the signal for a few frames. Recording continues for a
cooldown and only stops if motion stays absent, so one visit is one file.

**A minimum event length.** A single noisy frame should not leave a one-frame
file behind. Events under the floor are discarded along with their file and
thumbnail.

**Warm-up.** The background model is told not to report during its first frames.
Without this, startup is one enormous false event, because every frame differs
wildly from an unformed reference.

**Footage time, not wall time.** For a recorded clip, pre-roll, post-roll, and
event timestamps all run on the video's own timeline, so results do not change
with how fast the machine decodes.

## Test

From `python/`:

```powershell
python -m unittest discover -s tests -v
```

58 tests covering: static-scene silence, object detection and box placement,
warm-up suppression, a stationary object fading into the background, frame-size
changes restarting the model, MOG2 shadow handling, area filtering and region
ordering, pre-roll frame counts, gap bridging, short-event discarding, long-event
splitting, unique timestamped names, snapshots, CSV/JSON round-trip and every
corrupt-log path, drawing at every state, the pipeline with and without a
recorder, CLI parsing and validation, and the media loop with a fake camera.

For an end-to-end check:

```powershell
python scripts/make_test_clip.py
python python/security_camera.py --input assets/input/surveillance-demo.avi --no-display
```

The clip contains two walk-throughs; the detector should report two events.

## Project layout

```text
07-motion-detection-security-camera/
├── README.md
├── assets/input/ , assets/output/
├── recordings/                 # event clips, thumbnails, index.html
├── results/                    # events.csv, events.json, comparison CSVs
├── scripts/
│   ├── make_test_clip.py       # synthetic clip with known motion times
│   ├── compare_backgrounds.py  # precision/recall scoring
│   └── dashboard.py            # local HTML dashboard
└── python/
    ├── security_camera.py      # entry point
    ├── requirements.txt
    ├── src/
    │   ├── app.py              # CLI, media loop
    │   ├── background.py       # running-average and MOG2 models
    │   ├── detector.py         # the motion pipeline
    │   ├── drawing.py          # boxes, status panel, timestamp
    │   ├── errors.py
    │   ├── events.py           # CSV/JSON logs
    │   ├── pipeline.py         # per-frame orchestration
    │   └── recorder.py         # pre-roll, post-roll, timestamped clips
    └── tests/
        └── test_security_camera.py
```

## Common problems

**Everything triggers constantly —** raise `--min-area`, raise
`--delta-threshold`, or increase `--blur`. If the scene has changing light,
switch to `--background mog2`.

**Nothing triggers —** lower `--min-area` and `--delta-threshold`. Check the
`--show-mask` view: if the mask is black during real movement, the threshold is
too high; if it is permanent static, the blur is too small.

**A person standing still stops being detected —** that is the background model
learning them. Lower `--alpha` for a longer memory, at the cost of adapting more
slowly to genuine scene changes.

**Clips are split into many files —** raise `--post-roll`.

**The first seconds are one huge event —** raise `--warmup-frames`.

**`could not open video source: 0` —** another program holds the camera, or the
index is wrong. Try `--camera 1`.

**The dashboard shows cards but no video plays —** use `--serve`; `file://`
pages often cannot play local video.

## Limitations and possible improvements

- **It detects motion, not people.** A cat, a curtain, and a burglar are the
  same event. Feeding the cropped regions to the detector from Project 5 would
  turn "motion" into "a person at 21:14", which is the single biggest upgrade
  available here.
- **No tracking or identity.** Each frame is independent, so the system cannot
  count how many people passed or follow one across the scene.
- **Evaluated only on a synthetic clip.** The precision/recall figures come from
  a rendered scene with known motion times, not real footage. They compare the
  models fairly against each other but are not absolute accuracy numbers.
- **No region of interest.** A busy road through the window triggers constantly.
  Masking to a polygon would fix it and is a small change to the detector.
- **Sudden global light changes still cost precision** even with MOG2. Detecting
  that most of the frame changed at once, and treating that as a light change
  rather than motion, is a standard refinement not implemented here.
- **Storage is unbounded.** A real deployment needs retention: delete clips past
  a certain age or total size.
- **No alerting.** Events are written to disk and the dashboard must be opened
  to see them. There is no notification.
- **MJPG clips are large.** The codec is chosen for portability; H.264 would
  shrink them substantially where the platform supports it.
