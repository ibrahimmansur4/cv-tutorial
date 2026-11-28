# Sudoku Solver from an Image

Point a camera at a Sudoku puzzle, or hand it a photo, and get the solved grid
drawn back onto the original page in the correct perspective.

```
photo -> threshold -> contours -> largest quadrilateral -> homography
      -> 9x9 split -> per-cell cleanup -> digit recognition
      -> backtracking solver -> inverse homography -> answer on the page
```

The whole chain is OpenCV and NumPy. The default digit classifier needs no
download and no training framework.

## Features

- Image, video, and live webcam input
- Perspective correction, so the puzzle need not be photographed head-on
- Two digit classifiers — rendered templates (default) and an optional 26 KB ONNX MNIST model
- Bitmask backtracking solver with a minimum-remaining-values heuristic
- Uniqueness check, which catches a misread digit that quietly made the puzzle ambiguous
- Solution projected back through the inverse homography onto the original photo
- Solution caching, so a steady webcam view solves once and then just re-projects
- `--debug` dumps the flattened board, a cell contact sheet, and a confidence overlay
- Ground-truth evaluation across fonts and photographic conditions

## Requirements and setup

- Python 3.10 or newer (verified with Python 3.12)
- No GPU required

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r python/requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate`. Nothing needs
downloading for the default classifier.

## Run

Solve a photo:

```powershell
python python/sudoku.py --input assets/input/puzzle-photo.png --output assets/output/solved.png
```

Live from the webcam:

```powershell
python python/sudoku.py
```

See exactly what the recognizer saw when something goes wrong:

```powershell
python python/sudoku.py --input assets/input/puzzle-photo.png --debug --no-display
```

Make a test image with known digits:

```powershell
python scripts/make_puzzle_image.py --tilt 0.06 --lighting 0.4 --noise 5
python python/sudoku.py --input assets/input/puzzle.png --no-display
```

`--help` lists every option. Press `q` or `Esc` to stop a live run.

## How it works

### Finding the board

The grid is the largest four-sided thing in a picture of a grid, so: threshold,
contour, and keep the biggest quadrilateral.

**Adaptive thresholding** rather than a single global cut-off. A photo of a page
almost never has one brightness — a hand shadows one corner, a lamp blows out
another — so one threshold loses the grid in shadow or floods it in light.
Adaptive thresholding computes a threshold per neighbourhood and picks up a line
that is dark grey in one place and mid grey in another.

**`approxPolyDP`** collapses a contour's hundreds of boundary points into four
corners, at a tolerance of 2% of the perimeter: loose enough to absorb a wobbly
hand-held edge, tight enough not to shave off a real corner. Candidates are then
filtered for convexity, for a plausible minimum size, and for a sane aspect ratio
— a Sudoku board is square, so even at a steep angle its opposite sides stay
comparable, which rejects things like a book edge.

### The homography

A board photographed at an angle is a trapezoid in the image. A **homography** is
the 3×3 projective transform mapping one plane onto another, and four point
correspondences determine it exactly — which is why finding four corners is the
entire job. `getPerspectiveTransform` gives the matrix; applying it flattens the
board to a head-on square, and applying its inverse puts the answer back on the
tilted page. That is why the drawn digits appear to lie on the paper: they are
pushed through the same transform that flattened it.

Corners are ordered with the standard sum/difference trick — the top-left has the
smallest `x + y`, the top-right the smallest `y − x`. It is simple and robust
up to about 45° of rotation, where the ties collapse; that case is detected and
refused rather than silently producing a broken transform.

### Reading the cells

Cutting a square into 81 pieces is arithmetic. The hard parts are deciding
whether a cell holds a digit, and getting the ink into a shape a classifier can
read.

**Removing grid lines is subtler than it looks.** The obvious approach — flood
fill every white pixel reachable from the cell border — is wrong in a way that is
easy to miss. When a digit happens to touch a leftover line, the digit is part of
that same connected component, so the fill swallows the digit and the cell reads
as empty. This is not rare: thick outer borders and a slightly imperfect warp
make corner cells touch their lines routinely. It cost this project its top-left
digit until it was tracked down.

Lines are therefore removed by **shape**, not connectivity. A grid line is a long
straight run spanning most of the cell; no digit stroke is. Opening with a long
thin kernel keeps only such runs, and subtracting them severs the digit from the
line while leaving the digit whole.

**Normalisation** then crops the digit to its own bounding box, scales the longer
side to a fixed height — preserving aspect, so a "1" keeps its thin shape instead
of being stretched into something "8"-ish — and centres it on a 28×28 canvas, the
same convention MNIST uses.

**Deciding "empty"** cannot be "any white pixel", because thresholded noise is
never perfectly black. The test is that the ink forms a single blob of plausible
size, tall enough relative to the cell, and roughly centred.

### Solving

Backtracking is depth-first search with early abandonment: pick an empty cell,
try a non-conflicting digit, recurse, undo on failure. Two refinements make it
fast enough to run per frame:

- **Bitmask constraints.** Each row, column and box keeps a 9-bit mask of the
  digits it holds, so testing a candidate is three bitwise ANDs instead of
  scanning 27 cells.
- **Minimum remaining values.** Always expand the cell with the fewest legal
  candidates. A cell with one candidate is forced, so taking it first either
  makes free progress or proves the branch dead at once. In reading order
  instead, the search explores enormous subtrees before finding a contradiction
  a constrained cell would have revealed immediately.

Measured on an Intel 4-core laptop: a newspaper puzzle solves in about 1 ms with
51 assignments and no backtracking; *AI Escargot*, built specifically to defeat
naive solvers, takes 4.8 ms and 219 assignments.

The solver also **counts solutions up to two**. A proper Sudoku has exactly one;
more than one almost always means a digit was misread, so the result is flagged
rather than presented as *the* answer.

## Which digit classifier? The simple one wins

`scripts/evaluate.py` renders puzzles with known digits and grades the whole
chain. Crucially it can render with fonts the template classifier was **not**
built from, which tests generalisation rather than memorisation.

Two images per cell, 30 givens each, on an Intel 4-core laptop:

| Classifier | Font | In template set? | Cell recall | Digit accuracy | Solved |
|---|---|---|---:|---:|---:|
| template | simplex | yes | 100% | 100% | 2/2 |
| template | plain | **no** | 100% | 100% | 2/2 |
| template | complex-small | **no** | 100% | 100% | 2/2 |
| template | italic | **no** | 100% | 100% | 2/2 |
| onnx | simplex | — | 100% | 100% | 2/2 |
| onnx | plain | — | 100% | 73.3% | 0/2 |
| onnx | complex-small | — | 100% | 86.7% | 0/2 |
| onnx | italic | — | 100% | 93.3% | 0/2 |

The pretrained neural network loses to template matching, which is not the
result most people would predict. The reason is **domain mismatch**: MNIST is
*handwritten* digits, and Sudoku puzzles are *printed*. The CNN is excellent at
the problem it was trained on and merely adequate at this one, while rendered
templates are a near-exact match for printed glyphs.

The "Solved" column shows how brutally per-digit errors compound. Every given
must be right, so accuracy is raised to the power of the number of givens:

| Per-digit accuracy | Chance all 30 givens are right |
|---:|---:|
| 99.9% | 97.0% |
| 99.0% | 74.0% |
| 93.3% | 12.5% |
| 86.7% | 1.4% |
| 73.3% | 0.01% |

93% per-digit accuracy sounds respectable and produces a working solver roughly
one time in eight. This is why the project reports per-digit confidence and
checks uniqueness rather than trusting the read.

**An honest caveat.** Both the templates and the test images are rendered with
OpenCV's Hershey fonts. Using held-out fonts makes this a real generalisation
test rather than a tautology, but it is still one font engine. A newspaper
Sudoku in a typeface neither has seen is a harder problem, and these numbers
should not be read as real-world accuracy.

Reproduce with:

```powershell
python scripts/download_model.py        # only needed for the onnx row
python scripts/evaluate.py --classifiers template,onnx --repeats 2
```

## Test

From `python/`:

```powershell
python -m unittest discover -s tests -v
```

61 tests covering: solving easy, hard, empty and already-solved grids; conflicting
and unsolvable givens; the assignment limit; uniqueness counting; corner ordering
including the 45° degenerate case; board detection on flat and tilted renders;
cell splitting and occupancy against the rendered givens; **the grid-line
regression test for the flood-fill bug**; digit normalisation; every digit in four
fonts; the zero-exclusion rule for the ONNX head; pipeline caching and overlay;
CLI parsing; and the image and video loops.

The tests render their own puzzles, so they need no fixtures and no model
download.

## Project layout

```text
08-sudoku-solver-from-image/
├── README.md
├── models/                     # optional mnist-8.onnx (git-ignored)
├── assets/input/ , assets/output/
├── results/                    # debug images, evaluation.csv
├── scripts/
│   ├── make_puzzle_image.py    # rendered puzzle with ground truth
│   ├── download_model.py       # optional ONNX digit model
│   └── evaluate.py             # accuracy across fonts and conditions
└── python/
    ├── sudoku.py               # entry point
    ├── requirements.txt
    ├── src/
    │   ├── app.py              # CLI, image and video loops
    │   ├── cells.py            # 9x9 split, line removal, normalisation
    │   ├── classify.py         # template and ONNX classifiers
    │   ├── errors.py
    │   ├── grid.py             # thresholding, contours, homography
    │   ├── pipeline.py         # end-to-end orchestration and caching
    │   ├── render.py           # draw and project the solution back
    │   └── solver.py           # bitmask backtracking
    └── tests/
        └── test_sudoku.py
```

## Common problems

**`no four-sided grid large enough was found` —** fill more of the frame with the
puzzle, flatten the page, improve the lighting, or lower `--min-area-ratio`.

**`the digits read conflict at ...` —** a digit was misread. Run with `--debug`
and look at `results/recognition.png`, which shows each digit and its confidence,
and `results/cell_mosaic.png`, which shows exactly what the classifier was given.

**`more than one solution` warning —** usually a given was *missed* rather than
misread, leaving the puzzle under-constrained. Check the cell mosaic for a digit
that came out blank.

**Digits are missed in cells near the border —** raise `--margin-ratio` to trim
more of each cell edge, or lower `--min-ink-ratio`.

**Empty cells are read as digits —** raise `--min-ink-ratio`.

**It is slow on a webcam —** the solve is cached, so a steady view costs almost
nothing after the first frame. If it is still slow, lower `--resolution`.

## Limitations and possible improvements

- **Printed digits only.** The default classifier is built from rendered glyphs.
  A hand-filled puzzle is a different problem, and the ONNX MNIST model — trained
  on exactly that — would be the better choice there.
- **Accuracy is measured on rendered images.** The evaluation uses held-out fonts,
  which is a real generalisation test, but still one font engine. There is no
  benchmark here against photographs of real newspapers.
- **One board per frame**, and it must be the largest quadrilateral. A puzzle
  beside a larger rectangle loses.
- **No temporal voting.** A webcam sees the same board dozens of times a second,
  and taking a per-cell majority across frames would beat trusting any single
  read. The cache stores whole solutions, not per-digit votes.
- **Severe rotation fails.** Corner ordering breaks down near 45°; it is detected
  and refused rather than mishandled, but not solved.
- **Curved pages are not handled.** A homography assumes the board is planar, so
  a creased or curled page warps unevenly.
- **No confidence-driven retry.** When a digit is read with low confidence the
  system warns but does not, for instance, try the second-best digit and see
  whether that makes the puzzle solvable and unique — which would be a cheap and
  effective fix.
