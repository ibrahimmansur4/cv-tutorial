# Project 3: Face Recognition Attendance System

This Python project detects frontal faces, converts each face into a normalized
Histogram of Oriented Gradients (HOG) embedding, compares it with registered
people, and records recognized attendance in CSV format. It accepts a still
image, recorded video, or webcam and includes a webcam enrollment mode.

The implementation is deliberately local and transparent: OpenCV supplies the
Haar detector and HOG descriptor, NumPy performs cosine matching, and Python's
CSV module writes attendance. It is a learning project, not a production
biometric identity or access-control system.

## What this project teaches

- How face **detection** differs from face **recognition**.
- How a fixed-size numeric embedding represents a normalized face crop.
- How cosine distance and a threshold turn nearest-neighbor search into a
  recognized/unknown decision.
- How to cache named embeddings safely and invalidate the cache when source
  images change.
- How to record `name`, `date`, and `time` exactly once per person per local day.
- How to design image, video, webcam, and enrollment paths around the same
  testable pipeline.

```text
known_faces images ----------------------------+
      |                                        |
      v                                        v
Haar face detection -> HOG embeddings -> safe NPZ cache
                                             |
image / video / webcam                        |
      |                                      /
      v                                     v
Haar detections -> face crops -> HOG -> cosine nearest neighbor
                                           |
                              distance <= threshold?
                                  /                 \
                              known                 Unknown
                                |
                                v
                 daily CSV de-duplication -> annotation
```

## Repository layout

```text
03-face-recognition-attendance-system/
|-- README.md
|-- known_faces/
|   |-- README.md
|   `-- Sample Person/             # fictional tutorial identity
|       `-- sample-face.png
|-- assets/
|   |-- input/sample-face.png      # deterministic smoke-test input
|   `-- output/                    # generated annotated media
|-- data/
|   |-- embeddings.npz             # generated descriptor cache (ignored)
|   `-- attendance.csv             # generated attendance log (ignored)
|-- results/                        # experiment notes/tables
`-- python/
    |-- attendance.py              # command-line entry point
    |-- requirements.txt
    |-- src/
    |   |-- detection.py
    |   |-- embeddings.py
    |   |-- attendance_log.py
    |   |-- pipeline.py
    |   `-- app.py
    `-- tests/
```

Locally enrolled profiles, the generated embedding cache, and attendance
records are ignored by Git because they can contain biometric or personal data.
Only the fictional `Sample Person` fixture is allowlisted for this tutorial.

## Requirements

- Python 3.10 or newer
- A webcam for live recognition or enrollment only
- Windows, macOS, or Linux with an OpenCV-supported camera backend

No external face model download is required. OpenCV's Python wheel includes the
frontal-face Haar cascade used by default.

From this project directory:

```powershell
py -3.12 -m venv .venv
./.venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r python/requirements.txt
```

On macOS or Linux, create the environment with `python3.12 -m venv .venv` and
activate with `source .venv/bin/activate`.

## Run the deterministic demo

The included portrait is fictional and is present once as a registered face and
once as an input fixture. Rebuild the cache, recognize the image, annotate it,
and record attendance with:

```powershell
python python/attendance.py `
  --input assets/input/sample-face.png `
  --output assets/output/sample-face-annotated.jpg `
  --rebuild-cache --no-display
```

The first run creates:

```text
data/embeddings.npz
data/attendance.csv
assets/output/sample-face-annotated.jpg
```

Run the same command again on the same day. Recognition still occurs, but the
summary reports zero new attendance rows because the CSV already contains that
person/date pair.

The CSV schema is intentionally small:

```csv
name,date,time
Sample Person,2026-10-05,12:34:56
```

Dates and times use the computer's local clock. The example timestamp above is
illustrative; your generated row will use the actual run time.

## Register known people from images

Create one directory per person under `known_faces/` and add one or more clear,
front-facing images:

```text
known_faces/
|-- Ibrahim/
|   |-- front.jpg
|   `-- office-lighting.jpg
`-- Sara Khan/
    `-- portrait.png
```

Each image should contain exactly one intended face. The first directory name
becomes the identity shown on screen and stored in the CSV. Root-level image
files are also supported; their filename stem becomes the identity.

The cache automatically rebuilds when an image path, size, or modification time
changes. Use `--rebuild-cache` to force a refresh.

## Enroll from a webcam

The roadmap enrollment command is supported directly:

```powershell
python python/attendance.py --enroll Ibrahim
```

Enrollment opens camera `0`, waits for exactly one visible face, periodically
saves eight expanded face crops under `known_faces/Ibrahim/`, and rebuilds the
embedding cache. Keep your head mostly frontal while making small natural pose
and expression changes.

Useful enrollment controls:

```powershell
python python/attendance.py --enroll "Sara Khan" --camera 1 `
  --samples 12 --capture-every 6 --resolution 1280x720
```

- `--samples` controls how many face crops are stored; minimum `2`.
- `--capture-every` avoids saving near-identical consecutive frames.
- `--max-frames` provides a hard attempt limit for headless automation.
- Existing profiles are extended with new numbered samples; they are not
  silently deleted or replaced.
- Press `q` or `Esc` to stop. An incomplete enrollment returns an error while
  leaving any captured samples available for inspection.

Enrollment images are sensitive biometric data. Obtain informed permission,
protect the directory, and delete data when it is no longer needed.

## Recognize with a webcam

```powershell
python python/attendance.py --camera 0 --resolution 1280x720
```

Green boxes show a recognized name and cosine distance. Orange boxes show
`Unknown`. `MARKED` appears when that frame creates a new attendance row. Press
`q` or `Esc` to exit.

For a bounded headless camera smoke test:

```powershell
python python/attendance.py --camera 0 --max-frames 20 --no-display
```

## Process recorded video

```powershell
python python/attendance.py `
  --input path/to/input.avi `
  --output assets/output/attendance-annotated.avi `
  --no-display
```

Video output accepts `.avi`, `.mp4`, `.m4v`, or `.mov`. MJPG `.avi` is usually
the most portable across OpenCV builds. Use `--max-frames N` for a bounded run.

## CLI guide

Run `python python/attendance.py --help` for the authoritative option list.

| Option | Purpose |
|---|---|
| `--input PATH` | Process an image or video instead of a camera. |
| `--camera INDEX` | Select a webcam; defaults to `0` when no input is given. |
| `--output PATH` | Write an annotated image or video. |
| `--display` / `--no-display` | Enable or disable GUI windows. |
| `--resolution WIDTHxHEIGHT` | Request a webcam capture resolution. |
| `--max-frames N` | Bound video, webcam, or enrollment processing. |
| `--known-faces PATH` | Override the registered-image directory. |
| `--cache PATH` | Override the generated NPZ embedding cache. |
| `--attendance PATH` | Override the attendance CSV destination. |
| `--rebuild-cache` | Ignore and regenerate the current embedding cache. |
| `--threshold VALUE` | Maximum cosine distance accepted as a match. |
| `--cascade PATH` | Use a custom Haar cascade XML file. |
| `--scale-factor VALUE` | Tune Haar's image-pyramid step. |
| `--min-neighbors N` | Tune the Haar detection consensus requirement. |
| `--min-face-size N` | Ignore faces smaller than `N x N` pixels. |
| `--enroll NAME` | Capture and register webcam samples for a person. |
| `--samples N` | Number of enrollment samples to collect. |
| `--capture-every N` | Enrollment frame interval between samples. |

Input and output may not refer to the same file, including through a hard link.
Expected configuration/media failures produce a concise error and non-zero exit
status instead of a traceback.

## How recognition works

### 1. Detect a face

The detector converts each frame to grayscale, equalizes its histogram, and
runs OpenCV's bundled frontal-face Haar cascade. Area-sorted bounding boxes are
clipped to the image before cropping.

### 2. Build an embedding

Each crop is resized to `128 x 128`, histogram-equalized, and described with
HOG. HOG groups local edge directions into cells and overlapping blocks. The
resulting vector is L2-normalized, which makes its dot product a cosine
similarity.

This is an educational handcrafted embedding, not a modern learned face model.
It is lightweight and inspectable, but much less invariant to pose, age,
lighting, expression, masks, and camera quality.

### 3. Match by cosine distance

For a live descriptor `q` and registered descriptor `r`:

```text
cosine_similarity = r dot q
cosine_distance   = 1 - cosine_similarity
```

The nearest registered sample is accepted only when its distance is at or below
`--threshold` (`0.35` by default). Lower thresholds are stricter and tend to
reduce false acceptance while increasing false rejection. The mathematical
range is `0` to `2`, but useful values for this descriptor are normally much
narrower and must be calibrated on representative consented data.

### 4. Cache without Python objects

Named embeddings are stored as numeric and Unicode arrays in a compressed NPZ
file. The loader always uses `allow_pickle=False`, so a cache cannot deserialize
arbitrary Python objects. Source images remain authoritative; stale, damaged,
or incompatible caches are rebuilt.

### 5. Record attendance once per day

Only recognized identities reach the attendance logger. Before appending, it
compares the case-insensitive normalized name and ISO date with existing rows.
The same person can be recorded again on a different date, but not twice on the
same date. `Unknown` is never written.

## Tests

The automated suite is offline and headless:

```powershell
python -m unittest discover -s python/tests -v
```

It covers geometry, detection adapters, HOG normalization, cosine thresholds,
unknown handling, safe cache round trips/invalidation, CSV validation and daily
deduplication, recognition orchestration, media cleanup, CLI errors, and
enrollment behavior. The included portrait also provides a real Haar/HOG
integration check.

After the tests, run the deterministic image command and a bounded webcam run to
verify the local OpenCV build and camera backend.

## Suggested experiments

### Threshold calibration

Collect separate enrollment and evaluation images with permission. Sweep the
threshold without changing the evaluation set:

| Threshold | True accepts | False rejects | False accepts | Unknown correct |
|---:|---:|---:|---:|---:|
| 0.20 | | | | |
| 0.30 | | | | |
| 0.35 | | | | |
| 0.45 | | | | |

Report the tested identities, samples per identity, camera, lighting, and split
method. Never tune and report accuracy on the same exact images.

### Enrollment diversity

Compare one, four, eight, and twelve registered samples per person. Hold the
recognition clips and threshold constant. Record recognition rate and nearest
distance under front, mild left/right pose, bright, dim, and backlit conditions.

### Detection settings

Change one of `--scale-factor`, `--min-neighbors`, or `--min-face-size` at a
time. Measure missed faces, false detections, and processing FPS. A setting that
detects more rectangles can still make recognition worse if the crops are poor.

Store raw experiment tables under `results/`; avoid committing private face
images or attendance records.

## Troubleshooting

- **No face detected:** face the camera, improve front lighting, move closer,
  remove severe occlusion, and check `--min-face-size`.
- **Known image skipped:** make sure it decodes correctly and contains exactly
  one clear frontal face; inspect warnings printed during cache rebuild.
- **Everyone is Unknown:** force `--rebuild-cache`, verify profile names/images,
  then calibrate the threshold with separate evaluation frames.
- **Wrong person recognized:** lower the threshold. Add varied clean enrollment
  samples and measure false accepts before trusting the configuration.
- **Camera cannot open:** close other camera applications, grant OS permission,
  and try another `--camera` index.
- **GUI unavailable:** add `--no-display`, plus `--max-frames` for a live source.
- **Video writer fails:** use an existing writable destination and prefer MJPG
  `.avi` if the platform lacks an MP4 encoder.
- **CSV header error:** preserve exactly `name,date,time`, or move the unrelated
  CSV and let the application create a new one.
- **Cache looks stale:** run with `--rebuild-cache`; the cache is disposable.

## Limitations and responsible use

- Haar detection is strongest for visible, mostly frontal faces and can miss
  profiles, small faces, blur, glare, or occlusion.
- HOG is not a state-of-the-art identity embedding. Similar-looking people or
  changed capture conditions can produce unsafe decisions.
- The system has no liveness or anti-spoofing protection; a photo or screen may
  be accepted. Never use it for security, payroll, discipline, access control,
  or another high-stakes decision.
- A single threshold does not establish biometric accuracy or fairness.
  Evaluate false acceptance/rejection across the actual population and capture
  conditions, with consent.
- CSV de-duplication is designed for one application process. It does not offer
  database transactions or cross-machine concurrency.
- Face images, embeddings, and attendance records are sensitive personal data.
  Minimize collection, obtain informed consent, restrict access, define a short
  retention period, support deletion, and follow applicable law and policy.

Useful future extensions include learned face embeddings, landmark alignment,
tracking to avoid recomputing descriptors every frame, liveness checks,
encrypted storage, explicit profile deletion, and a transactional attendance
database with audit controls.
