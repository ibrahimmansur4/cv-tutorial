# Driver Drowsiness Detection

Watch a face through the webcam, measure how open the eyes are, and raise an
alarm when they stay shut long enough to be dangerous.

The detector reports three separate signals, because fatigue shows up in more
than one way:

- **Sustained closure** — the eyes stay shut past a threshold duration. The
  acute, immediate danger signal.
- **PERCLOS** — the percentage of a rolling window spent with the eyes closed.
  The measure the automotive literature settled on, because it catches the slow
  creep of fatigue that no single closure reveals.
- **Yawning** — the mouth held open, counted as discrete events. Corroborating
  evidence, never an alarm on its own.

> **This is a tutorial project, not a safety device.** It runs on one webcam in
> good light, has no validation against a labelled drowsiness dataset, and can
> be defeated by sunglasses, darkness, or simply looking away. Do not rely on it
> to keep anyone awake at the wheel. See [Limitations](#limitations-and-possible-improvements).

## Features

- Webcam and recorded-video input
- Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) from MediaPipe's 478-point face mesh
- **Per-driver calibration** — measures your own open-eye baseline instead of assuming a constant
- Duration-based state machine with blink/drowsiness separation and hysteresis
- PERCLOS over a rolling window, with a warm-up period
- Live HUD: EAR, eye state, closure timer, PERCLOS, blink and yawn counts, FPS
- Flashing border and audible alarm (silent fallback where no beeper exists)
- Annotated video export
- A threshold-sweep experiment that validates EAR against MediaPipe's learned blink signal
- Synthetic test-clip generator, so the alarm path is reproducible without filming anyone asleep

## Requirements and setup

- Python 3.10 or newer (verified with Python 3.12)
- A webcam
- No GPU required

From this project directory on Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r python/requirements.txt
python scripts/download_model.py
```

On macOS/Linux, activate with `source .venv/bin/activate`. The downloader fetches
the pinned official `face_landmarker.task` bundle (3.6 MB), verifies its SHA-256,
and stores it under `models/`, which is ignored by Git.

## Calibrate first

Do this once, before anything else:

```powershell
python python/drowsiness.py --calibrate
```

Sit facing the camera with your eyes open and blink normally for five seconds.
The profile is written to `results/calibration.json` and picked up automatically
on later runs.

This matters more than any other setting. A fixed EAR threshold does not
transfer between people: eye shape, camera angle, and glasses all shift the
open-eye baseline. Calibration measures *your* baseline and sets the threshold
at 75% of it, so what counts as "closed" means "much less open than this person
usually is" rather than "below some constant from a paper".

```powershell
python python/drowsiness.py --calibrate 10        # longer capture
python python/drowsiness.py --closed-ratio 0.65   # stricter: must close further
python python/drowsiness.py --no-profile          # ignore the profile, use --ear-threshold
```

## Run

Live monitoring:

```powershell
python python/drowsiness.py
```

Analyse a recorded clip and save the annotated result:

```powershell
python python/drowsiness.py `
  --input assets/input/drowsy-demo.avi `
  --output assets/output/drowsy-annotated.avi `
  --no-display
```

Tune the timings:

```powershell
python python/drowsiness.py `
  --closure-seconds 2.0 `
  --blink-seconds 0.35 `
  --perclos-window 90 `
  --perclos-threshold 0.2
```

`--help` lists every option. `--no-alarm` stays silent, `--no-landmarks` hides
the eye outlines, and `--max-frames N` is useful for unattended smoke tests.

Press `q` or `Esc` to stop.

## How it works

Three layers, deliberately kept separate:

1. **Machine-learning model.** MediaPipe Face Landmarker predicts 478 face-mesh
   landmarks per frame. Nothing here trains it.
2. **Computer-vision algorithm.** Six landmarks per eye become an aspect ratio;
   six mouth landmarks become another. This is plain geometry over the points.
3. **Application logic.** The state machine that converts a stream of ratios
   into a verdict, plus capture, drawing, alarm, and the CLI.

### The Eye Aspect Ratio

Six points are taken around the eye — two corners, and two pairs above and below
the pupil:

```
        p2    p3
    p1              p4     EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|)
        p6    p5
```

The numerator measures how open the lid is; the denominator, how wide the eye
is. Dividing one by the other is the whole trick: both distances scale together
as the face moves toward or away from the camera, so the ratio stays roughly
constant while the eye is open and collapses toward zero when it shuts. A raw
pixel gap between the eyelids would not — it would shrink just from leaning
back.

The measure comes from Soukupová and Čech, *Real-Time Eye Blink Detection using
Facial Landmarks* (2016), which used dlib's 68-point model. The indices here are
the equivalent positions on MediaPipe's mesh.

### Two things published EAR thresholds will not tell you

**MediaPipe's mesh reads lower than dlib's.** The widely quoted threshold of
0.25 does not transfer. Measured on this project's mesh, a wide-open eye sits
nearer **0.20**, so 0.25 classifies every open eye as shut. The threshold sweep
below shows exactly that happening. This is the single most common reason a
copied EAR tutorial does not work.

**Normalized coordinates distort the ratio.** MediaPipe reports x normalized by
the frame width and y by the frame height. On any non-square frame those are
different numbers of pixels, so measuring directly in that space stretches every
horizontal distance and the ratio inherits the frame's aspect. The same eye —
frame 0 of the demo clip, held at fixed pixel geometry and renormalized onto
each frame shape — then measures:

| Frame | EAR without correction |
|---|---:|
| pixel space (correct) | 0.211 |
| 820×1024 portrait | 0.173 |
| 640×480 | 0.275 |
| 1280×720 | 0.362 |

A factor of two between two ordinary webcam resolutions, for the same eye. Every
distance here therefore rescales x by `width / height` first
([`geometry.py`](python/src/geometry.py)), which restores pixel proportions up to
a uniform scale that cancels in the ratio.

### Why duration, and why seconds

A single frame cannot distinguish a blink from drowsiness — both are just closed
eyes. Only duration separates them, which is why the state machine
([`state.py`](python/src/state.py)) is the core of the project rather than an
afterthought.

Durations are measured in **seconds, not frames**. Counting frames silently
encodes the frame rate: "20 consecutive frames" is 0.67 s on a 30 FPS webcam and
2 s on a laptop struggling at 10 FPS, so a threshold tuned on one machine
misbehaves on another. Taking a timestamp per frame removes that dependency.

A closure shorter than `--blink-seconds` (0.4 s) is counted as a blink. One
longer than `--closure-seconds` (1.5 s) raises the alarm, and is deliberately
*not* also counted as a blink.

### PERCLOS, and why it needs a warm-up

PERCLOS is the fraction of a rolling window spent with the eyes closed. It
catches what sustained-closure detection misses: many short closures, none long
enough to alarm individually, that together mean someone is fading.

Being a ratio over a window, it is meaningless until the window holds enough
footage. Early on it is dominated by whatever just happened — a single ordinary
0.2 s blink one second into a session reads as **17% closed**, which clears the
15% fatigue threshold immediately. Without a guard, every session reports
fatigue within seconds of starting. The monitor therefore withholds the PERCLOS
verdict until `--perclos-warmup` seconds have been observed (30 s by default,
clamped to the window). Sustained closure is judged from the current closure
alone, so it stays trustworthy from the very first frame.

### Smaller decisions worth noting

- **Hysteresis.** Re-opening must clear the threshold by a margin. Without it, an
  EAR sitting on the threshold toggles state every frame and inflates the blink count.
- **A missing face is its own state.** Counting it as closed would alarm every
  time the driver checked a mirror; counting it as open would hide a driver who
  had slumped out of frame. It is reported separately, and a prolonged absence
  gets its own banner.
- **Both eyes are averaged,** not minimised: a brief tracking error on one side
  should not read as a closed eye.

## Experiment

```powershell
python scripts/ear_experiment.py --seconds 30                     # live capture
python scripts/ear_experiment.py --input my-clip.avi --no-mirror  # recorded clip
```

For every frame this logs the geometric EAR alongside MediaPipe's
`eyeBlinkLeft`/`eyeBlinkRight` blendshape — a *learned* estimate of eye closure
from the same model. It is not ground truth, but it is an independent signal:
where the two agree the EAR is reading the eye correctly, and where they diverge
the landmarks are probably being fitted badly. It then replays the recorded EAR
series through the real state machine across a range of thresholds.

Results on the synthetic demo clip (10.3 s, ground truth: 2 blinks and one 2.5 s
closure):

```text
EAR  min=0.0860  p10=0.0920  p30=0.2093  median=0.2119  max=0.2150
Correlation between EAR and the learned blink blendshape: -1.000
Suggested threshold at 75% of the p30 baseline: 0.1570
```

| Threshold | Blinks | Alarms | Alarm s | Closed frames |
|---:|---:|---:|---:|---:|
| 0.05 – 0.08 | 0 | 0 | 0.0 | 0.0% |
| 0.09 – 0.19 | **2** | **1** | **1.0 – 1.05** | 27.2 – 28.6% |
| 0.20 | 0 | 1 | 6.8 | 80.6% |
| 0.21 | 0 | 1 | 8.75 | 99.5% |
| 0.22 – 0.30 | 0 | 1 | 8.8 | 100.0% |

Three regimes, and the middle one is the answer:

- **Below 0.09** the threshold sits under even a fully closed eye, so nothing is
  ever detected.
- **0.09 to 0.19** is the usable plateau. It recovers the ground truth exactly:
  two blinks, one alarm, about one second of alarm for a 2.5 s closure that
  triggers at 1.5 s.
- **At 0.20 and above** the open-eye cluster starts falling below the boundary —
  80.6% of frames at 0.20, 99.5% at 0.21, and every frame from 0.22 up. The
  blink count drops to zero — not because blinking stopped, but because the eyes
  never "open" again, so there are no transitions to count. **The classic 0.25
  threshold lands squarely in this failure region.**

The calibrated suggestion of 0.157 falls in the middle of the usable plateau,
which is the result the calibration formula is supposed to produce.

Two caveats on this particular run. The synthetic clip only contains two
distinct eye states, so the −1.000 correlation is degenerate; real footage gives
a more informative number. And the plateau is wide here because the synthetic
closure is extreme — a real half-closed eye produces a narrower usable band,
which is exactly why calibrating beats guessing.

### Reproducing the demo clip

```powershell
python scripts/make_demo_clip.py
```

This animates eyelids closing over a single portrait photo and prints the ground
truth it generated. The face it produces is crude and is no substitute for real
footage; its purpose is to make the alarm path reproducible without filming
someone falling asleep. Running the detector on it should report 2 blinks and
1 drowsiness event.

## Test

From `python/`:

```powershell
python -m unittest discover -s tests -v
```

74 tests covering: aspect-corrected distance and the resolution-invariance
regression, EAR/MAR against hand-computed values, scale invariance, degenerate
landmarks, blink versus sustained closure, hysteresis, PERCLOS accumulation,
window expiry and the warm-up guard, yawn counting and de-duplication, missing
faces and abandoned closures, calibration maths, profile round-trip and every
corrupt-profile path, all three alarm backends and the selector, HUD rendering
at every alert level, the pipeline, CLI parsing and validation, and the media
loop with recorded input and output.

The tests inject a fake tracker and a synthetic clock, so the whole suite
finishes in a fraction of a second and needs no camera, no model download, and
no real time.

For an end-to-end check after downloading the model:

```powershell
python scripts/make_demo_clip.py
python python/drowsiness.py --input assets/input/drowsy-demo.avi --no-display --no-alarm
```

## Project layout

```text
06-driver-drowsiness-detection/
├── README.md
├── models/                     # face_landmarker.task (git-ignored)
├── assets/input/ , assets/output/
├── results/                    # calibration.json, experiment CSVs
├── scripts/
│   ├── download_model.py       # SHA-256 verified
│   ├── ear_experiment.py       # trace + threshold sweep
│   └── make_demo_clip.py       # synthetic test footage
└── python/
    ├── drowsiness.py           # entry point
    ├── requirements.txt
    ├── src/
    │   ├── app.py              # CLI, media loop, calibration mode
    │   ├── alarm.py            # audible alert with silent fallback
    │   ├── calibration.py      # per-driver baseline and profile
    │   ├── drawing.py          # HUD and alert banner
    │   ├── errors.py
    │   ├── face_mesh.py        # MediaPipe adapter
    │   ├── geometry.py         # Point, aspect-corrected distance
    │   ├── metrics.py          # EAR/MAR and landmark indices
    │   ├── pipeline.py         # per-frame orchestration
    │   └── state.py            # the temporal state machine
    └── tests/
        └── test_drowsiness.py
```

## Common problems

**`MediaPipe face model not found` —** run `python scripts/download_model.py`.

**The alarm fires while you are plainly awake —** the threshold is too high for
your face. Run `--calibrate`. If you already have, lower `--closed-ratio` to
around 0.65.

**It never fires however long you shut your eyes —** the threshold is too low.
Re-calibrate; make sure you are not squinting during the capture, which would
set the baseline too low.

**`FATIGUE` appears immediately on startup —** PERCLOS should be suppressed for
`--perclos-warmup` seconds. If you shortened that, lengthen it again.

**`could not open video source: 0` —** another program holds the camera, or the
index is wrong. Try `--camera 1`.

**No face is detected —** MediaPipe needs reasonable light and a mostly
front-facing view. Glasses are usually fine; sunglasses are not.

**No sound on Linux/macOS —** the fallback writes the terminal bell, and whether
that is audible depends on terminal settings. Sustained visual alerting (the
flashing border) works everywhere.

## Limitations and possible improvements

- **Not validated against ground truth.** There is no labelled drowsiness
  dataset here, so there are no false-positive or false-negative rates. The
  threshold sweep shows which settings are self-consistent, not which are
  correct. A real evaluation needs something like NTHU-DDD.
- **EAR is a proxy, not a measure of sleepiness.** It tracks eyelid position.
  Someone can be dangerously fatigued with their eyes wide open, and microsleeps
  can be shorter than the alarm duration.
- **One camera, good light, front-facing.** Sunglasses defeat it outright. Harsh
  side lighting, night driving, and extreme head angles all degrade landmark
  fitting, and a bad fit produces a wrong EAR rather than no EAR.
- **Head pose is ignored.** A driver looking down at their lap reads as
  partly-closed eyes. Using the mesh's pose estimate to reject or compensate for
  extreme angles would help, as would nodding detection, which is a genuine
  drowsiness cue this project does not use.
- **No per-eye confidence.** The two eyes are averaged; weighting them by how
  well each was fitted would be more robust when one is occluded.
- **Calibration is a single snapshot.** It does not adapt as someone settles in
  or the light changes. Slowly tracking the open-eye baseline during a session
  would handle drift.
- **The alarm is crude.** A real system escalates, and distinguishes a first
  warning from a repeated one.
