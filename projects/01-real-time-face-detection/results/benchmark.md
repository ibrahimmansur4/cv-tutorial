# Benchmark: Haar vs DNN detector throughput

Produced by `python scripts/benchmark.py`, run three times back to back.

- CPU: Intel Core i5-8365U, 4 cores
- OS: Windows 10 Pro 19045
- Python: 3.12.10, OpenCV: 4.12.0, NumPy: 2.2.6
- Input: `assets/input/sample-face.png` (1254x1254), resized with `cv.resize(..., INTER_AREA)`
- Protocol: 2 warm-up calls, median of 9 timed `detector.detect()` calls, nothing else running
- Both detectors found exactly 1 face at every size, in every run

## Median of the three runs (the table published in ARTICLE.md)

| Input size | Haar (ms) | Haar FPS | DNN (ms) | DNN FPS | Winner |
|---|---:|---:|---:|---:|---|
| 320x240 | 18.7 | 53.5 | 58.9 | 17.0 | Haar, 3.1x |
| 480x480 | 62.8 | 15.9 | 64.5 | 15.5 | tie (within noise) |
| 640x480 | 95.2 | 10.5 | 53.1 | 18.8 | DNN, 1.8x |
| 800x600 | 104.9 | 9.5 | 49.5 | 20.2 | DNN, 2.1x |
| 1280x720 | 249.6 | 4.0 | 52.6 | 19.0 | DNN, 4.7x |
| 1254x1254 | 497.7 | 2.0 | 76.5 | 13.1 | DNN, 6.5x |

## Run 1

| Input size | Haar (ms) | Haar FPS | DNN (ms) | DNN FPS |
|---|---:|---:|---:|---:|
| 320x240 | 18.7 | 53.4 | 49.9 | 20.0 |
| 480x480 | 59.2 | 16.9 | 49.9 | 20.0 |
| 640x480 | 95.2 | 10.5 | 53.0 | 18.9 |
| 800x600 | 104.9 | 9.5 | 49.5 | 20.2 |
| 1280x720 | 256.2 | 3.9 | 58.9 | 17.0 |
| 1254x1254 | 497.7 | 2.0 | 76.5 | 13.1 |

## Run 2

| Input size | Haar (ms) | Haar FPS | DNN (ms) | DNN FPS |
|---|---:|---:|---:|---:|
| 320x240 | 18.7 | 53.6 | 58.9 | 17.0 |
| 480x480 | 62.8 | 15.9 | 64.5 | 15.5 |
| 640x480 | 88.4 | 11.3 | 53.1 | 18.8 |
| 800x600 | 104.1 | 9.6 | 49.2 | 20.3 |
| 1280x720 | 249.6 | 4.0 | 52.6 | 19.0 |
| 1254x1254 | 525.1 | 1.9 | 91.7 | 10.9 |

## Run 3

| Input size | Haar (ms) | Haar FPS | DNN (ms) | DNN FPS |
|---|---:|---:|---:|---:|
| 320x240 | 19.3 | 51.9 | 84.3 | 11.9 |
| 480x480 | 65.2 | 15.3 | 79.5 | 12.6 |
| 640x480 | 99.6 | 10.0 | 87.7 | 11.4 |
| 800x600 | 158.7 | 6.3 | 82.4 | 12.1 |
| 1280x720 | 249.1 | 4.0 | 52.3 | 19.1 |
| 1254x1254 | 426.6 | 2.3 | 57.2 | 17.5 |

## Notes on noise

The Haar column is reproducible: it rises monotonically with pixel area in every
run, and the spread between runs at a given size is small.

The DNN column is noise-dominated. Across the three runs it stays inside roughly
49-92 ms at *every* input size, with no trend. Splitting the DNN's time in two
shows why: `blobFromImage` costs 1.7-3.7 ms regardless of input size, and
`net.forward()` costs 53-57 ms at 320x240, 640x480, 1280x720 and 1254x1254
alike. The network never sees the original frame, so its cost does not depend on
it. Do not read the small rises in the DNN column as scaling.

## Haar configuration sweep (1254x1254, scaleFactor 1.1, 1 correct detection each)

| Haar setting | ms |
|---|---:|
| minSize=(30,30) — this project's default | 508.2 |
| minSize=(60,60) | 287.4 |
| minSize=(120,120) | 60.5 |
| minSize=(240,240) | 28.1 |
| detect on a 0.25-scale copy, scale the box back up | 29.2 |

At 640x480: `scaleFactor=1.1` takes 120.7 ms, `scaleFactor=1.3` takes 45.2 ms,
and both find the face.

The 0.25-scale run returns (316, 256, 612x612) where the full-resolution run
returns (324, 265, 598x598) — about 14 px of difference.
