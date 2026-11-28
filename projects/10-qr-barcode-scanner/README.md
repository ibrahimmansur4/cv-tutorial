# QR Code + Barcode Scanner

Point a webcam at a QR code or a product barcode and get the content back,
classified, deduplicated, and logged to CSV.

```
frame -> grayscale -> QR detector      -> quad + payload -\
                   -> barcode reader   -> quad + payload --> dedup -> classify
                                                                      |
                                                     overlay <--------+--> CSV
```

OpenCV and NumPy only. No `pyzbar`, no `qrcode`, no ZBar install — OpenCV ships
both a QR detector and an EAN/UPC barcode reader, and it ships a QR *encoder*
too, so the project generates its own test material and checks itself by round
trip.

## Features

- Webcam, video file, and still image input
- QR codes via either OpenCV detector, with a **single-code fallback** that
  recovers codes the multi-code path silently drops
- EAN-13, EAN-8, UPC-A and UPC-E barcodes, with a **zoom-out retry** that takes
  a frame-filling barcode from never readable to always readable
- Payload classification: URL, Wi-Fi, vCard, email, phone, SMS, geo, calendar,
  product, plain text
- "Detected, not decoded" shown in a different colour, because that state tells
  the user to straighten up rather than keep hunting
- Cooldown-based deduplication, so one code in view is one scan, not thirty a
  second — and a deliberate re-scan still works
- Scan history with per-symbology and per-content tallies
- CSV log, flushed per row
- FPS, live overlay, pause, save-frame, clear-history keys
- Its own EAN-13 encoder and QR renderer, with ground-truth corners
- A benchmark that produces every number quoted below

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

On macOS/Linux, activate with `source .venv/bin/activate`.

## Run

Live from the webcam:

```powershell
python python/scanner.py --camera 0
```

A still image:

```powershell
python python/scanner.py --image assets/input/sample.png
```

While the window is open: **SPACE** pauses, **S** saves the annotated frame,
**C** clears the history, **Q** or **Esc** quits.

Make something to scan first:

```powershell
python scripts/make_codes.py --sample --frame 1800x1350 --columns 3 --out assets/input/sample.png
python scripts/make_codes.py --qr "https://example.com" --out assets/input/qr.png
python scripts/make_codes.py --ean 400638133393 --out assets/input/ean.png
```

Scan it, logging to CSV:

```powershell
python python/scanner.py --image assets/input/sample.png --no-display --csv results/scans.csv
```

```text
QR detected

Content: WIFI:T:WPA;S:Workshop;P:opencv2024;;
Kind:    wifi
Time:    20:42:31

Scanned 1 frame(s); 7 distinct code(s).

By symbology:
  EAN_13     3
  QR         4

By content:
  contact    1
  product    3
  text       1
  url        1
  wifi       1
```

Reproduce the measurements:

```powershell
python scripts/benchmark.py --out results
```

`--help` lists every option.

## How it works

### Finding a QR code

A QR code carries its own landmarks. The three corner squares — the finder
patterns — are built so that **any** scanline through the middle of one crosses
dark and light runs in the ratio 1:1:3:1:1. Ratios are unchanged by scaling and
by rotation, so one test finds a code near or far, upright or on its side,
without searching over scale or angle.

That is the whole trick, and it is why QR codes are read so much more reliably
than 1D barcodes, which have no comparable landmark. Three located finder
patterns give the code's position, orientation and scale; a homography through
them lets the decoder sample the module grid; Reed-Solomon then repairs what the
sampling got wrong.

**Detection and decoding are separate, and separating them is what makes the
display useful.** Either can fail alone. A code at a steep angle is typically
*found* and not *read*, so the scanner outlines it in amber and says "detected,
not decoded" — telling the user to straighten up. A scanner that reported
nothing would be hiding the most useful feedback it has. The measurements below
show how wide that gap is: a QR code is still located under damage four times
greater than the worst it can still decode.

### The single-code fallback

OpenCV offers `detectAndDecodeMulti` for several codes and `detectAndDecode` for
one. Reaching for the multi version and leaving it there is the obvious
implementation, and it quietly loses codes.

On a frame holding exactly one QR code, the classic detector's *multi* path
fails on payloads that its *single* path reads without trouble — same image,
same detector object, different entry point. `https://example.com/scan` is one
such payload: it decodes alone and comes back empty from the multi call.

So when the multi pass finds nothing, the reader retries with the single-code
call before concluding the frame is empty. It costs one extra call on frames
that found nothing anyway. The ArUco detector does not need the fallback — its
multi path reads everything its single path does, which is one of the reasons it
is the default.

### The zoom-out retry, and why barcodes are backwards

This is the part that surprised me.

OpenCV localises a 1D barcode by looking for a patch of the image where the
gradient points consistently in one direction — which is what a field of
parallel bars looks like *once it is small enough that neighbouring bars blur
together*. The detector gets there by shrinking the frame and analysing the
result.

The consequence is that the symbol has to be a **modest fraction of the
picture**. Sweeping a rendered EAN-13 against frame size, detection succeeds
while the barcode spans up to roughly a third of the frame width and fails above
it. That is the opposite of everyone's intuition about scanners, and it bites
hardest in the case that looks easiest: an image that is nothing but a barcode —
exactly what you get by saving one from a web page — where the bars stay sharply
separated at every scale the detector tries.

The fix is to give the detector the zoomed-out view it wants. When the first
pass finds nothing, the whole frame is mounted on a larger blank canvas and
searched again. Nothing is resampled, so no detail is lost; the symbol is simply
smaller relative to the frame, and the corners that come back are shifted by a
known offset.

**A barcode in a frame of its own size, 12 px of margin:**

| Bar module | Decoded, retry on | Decoded, retry off |
|---|---:|---:|
| 2 px/bar | 100% | 50% |
| 3 px/bar | 100% | 0% |
| 4 px/bar | 100% | 0% |
| 6 px/bar | 100% | 0% |
| 8 px/bar | 100% | 0% |

A ladder of fractions is used rather than one good value, because measuring
says one value does not work. The detector's response to zoom is **not
monotonic**: sweeping a single barcode from 0.5 down to 0.1 gives hits and
misses interleaved, and no single fraction reads every code. The retry walks
`0.33, 0.2, 0.15` and stops at the first rung that finds something — more
reliable than any fixed choice, and usually no slower, since most frames succeed
before the retry is reached at all.

### Why EAN-13 is fragile, and QR is not

Thirteen digits become 95 modules:

```
101 | 6 left digits (7 modules each) | 01010 | 6 right digits | 101
```

The guards at the ends and the middle give a reader its scale reference. The
first of the thirteen digits is **never drawn**: it is encoded in the *choice of
alphabet* for the six left-hand digits, each of which can come from an
odd-parity or an even-parity set, and which of the 2⁶ combinations appears names
the missing digit. Twelve drawn positions, thirteen digits.

Error handling is one check digit — weights alternating 1, 3, summing to a
multiple of ten. It **detects** an error without locating or repairing it. One
misread module changes a digit, the checksum fails, and the scan is rejected
outright. A QR code of the same payload carries Reed-Solomon parity and repairs
the damage instead.

Measured, by covering the centre of each symbol — which for a QR code leaves all
three finder patterns intact, so the damage lands on data modules and the test
is genuinely about recovery:

| Centre covered | QR (level M) | EAN-13 |
|---|---:|---:|
| 0% | 100% | 100% |
| 5% | 100% | 100% |
| 10% | 17% | 100% |
| 15% | 0% | 100% |
| 30% | 0% | 100% |
| 40% | 0% | 67% |

And the level does buy what it claims:

| Correction level | Decodes up to | Still *located* up to |
|---|---:|---:|
| L | 3% | 20% |
| M | 5% | 20% |
| Q | 10% | 20% |
| H | 10% | 20% |

Two things in that table are worth sitting with. The first is that **higher
correction genuinely helps** — the ordering is clean. The second is that the
figures are far below the "level H recovers 30%" you will read elsewhere. Both
are true: the published percentages count *codewords*, and these symbols are
small — version 2 to 3, a couple of dozen codewords in total — so a single
contiguous blob wipes out whole codewords faster than its area suggests. Parity
is also spread across the symbol deliberately, which helps scattered specks far
more than one solid square. A large QR code with scattered damage does much
better than this table; a small one with a sticker over the middle does not.

Now move the same patch to a corner, and the table inverts:

| Corner covered | QR | EAN-13 |
|---|---:|---:|
| 5% | 0% | 100% |
| 20% | 0% | 100% |
| 40% | 0% | 100% |

The QR code dies immediately, because the corner holds a finder pattern and
losing it costs the detector its geometry — there is nothing to decode, so
parity never gets a chance. The barcode barely notices, because bars run the
full height of the symbol: any uncovered row still crosses all 95 modules.

**A 1D barcode is redundant along its bars; a QR code is redundant in its data.**
Which one survives depends entirely on which way the damage runs, and a
comparison that covers only one region will confidently report the wrong answer.
This is why the benchmark reports both.

### Blur, distance, tilt, noise

| Blur radius | QR | EAN-13 |
|---|---:|---:|
| 0–3 px | 100% | 100% |
| 5 px | 100% | 83% |
| 8 px | 83% | 0% |
| 12 px | 0% | 0% |

| Scene scale | QR | EAN-13 | Pixels per module |
|---|---:|---:|---|
| ×1 | 100% | 100% | 8.0 QR, 4.0 bar |
| ×0.5 | 100% | 100% | 4.0 QR, 2.0 bar |
| ×0.35 | 100% | 100% | 2.8 QR, 1.4 bar |
| ×0.25 | 100% | 0% | 2.0 QR, 1.0 bar |
| ×0.15 | 0% | 0% | 1.2 QR, 0.6 bar |

The resolution floor is around **two pixels per module**, which is the sampling
limit rather than a shortcoming of the decoder: below it, adjacent modules land
in the same pixel and the information is no longer in the image. `--upscale 2`
cannot conjure it back; it only helps when the modules are still distinguishable
and the sub-pixel sampler needs room to work.

| Rotation | QR | EAN-13 |
|---|---:|---:|
| 0–60° | 100% | 100% |
| 90° | 83% | 100% |

Rotation is close to free for both, which for QR is the ratio test doing its
job. That 83% is the classic detector, which every unlabelled QR column above
uses; the ArUco default reads all six at 90°. Perspective is where the two
OpenCV detectors part company:

| Tilt strength | QR, classic | QR, ArUco |
|---|---:|---:|
| 0–0.2 | 100% | 100% |
| 0.3 | **0%** | **100%** |
| 0.4 | 0% | 0% |

Noise barely registers for either symbology — 100% at σ=90 — but that is partly
an artefact of synthetic codes, which are pure black on pure white. Real print
has far less contrast to spend.

### Which QR detector, and what it costs

ArUco is the default, on measurement rather than taste. The reasonable
expectation is that the newer detector buys robustness at some cost in speed. It
does not: it is faster *and* reads more.

| Frame | classic | ArUco |
|---|---:|---:|
| 640×480 | 11–17 fps | **31–37 fps** |
| 960×540 | 7–8 fps | 18–19 fps |
| 1280×720 | 3–4 fps | 6–10 fps |
| 1920×1080 | 3 fps | 4 fps |

Absolute figures on this laptop varied by about two-fold between runs depending
on what else was running, so treat them as an order of magnitude. The ordering
was stable across every run: ArUco came out 1.4× to 2.9× faster than classic,
never slower.

The practical conclusion is about **resolution, not the detector**. QR detection
cost grows steeply with frame size, and at 1280×720 a live scanner is sluggish
whichever detector is chosen. Drop to `--resolution 640x480` and it is genuinely
real-time; raise it only when the codes are small or far away. The barcode
reader is much cheaper — around 25–60 fps — except when the retry runs its whole
ladder and still finds nothing, which costs roughly what three extra detections
cost, and only happens on frames with no barcode in them.

The classic detector stays available because it is what the OpenCV samples use,
and because it holds on to a *located* outline slightly longer under heavy
occlusion, which is worth something for on-screen feedback even when nothing can
be decoded.

### Not reporting the same code thirty times a second

A webcam decodes the same code in every frame it is visible in. One QR code held
in front of a 30 fps camera produces about 30 identical results per second.

A seen-set — remember every payload, skip repeats — stops the flood and
introduces a worse bug: a code can never be scanned twice. Scanning the same
ticket again, or the same product at the till, is a normal thing to want.

What matches a real scanner is a **cooldown**: accept a payload, suppress it for
a few seconds, then accept it again. The window refreshes while the code stays
in view, so a code held up for a minute does not resurface every three seconds.
The key is `(payload, symbology)`, not the payload alone — the same thirteen
digits as an EAN-13 and inside a QR code are two different scans.

Two readings of one *printed* symbol are not, though. A leading-zero EAN-13 **is**
a UPC-A: OpenCV reports `0123456789050` as `UPC_A` with the zero stripped,
`123456789050`. Those are the same code written two ways, and comparing the
zero-padded form is what stops one barcode being logged twice.

The window is a trade-off rather than a correct value: too short and one wave of
a code counts twice, too long and a genuine re-scan is swallowed. `--cooldown`
exposes it; 3 seconds is the default.

### Classifying the payload

A scanner that prints the raw string makes the user do the parsing. Payloads
mostly announce their own type through a URI scheme or one of the pseudo-schemes
phone cameras act on: `WIFI:`, `BEGIN:VCARD`, `mailto:`, `tel:`, `geo:`,
`SMSTO:`.

None of this is in the QR standard, which stops at "here are your bytes".
Everything above it is convention, so a payload can be perfectly valid and
unrecognisable — in which case it is plain text, which is the honest answer
rather than an error.

Two details are easy to get wrong. A scheme test of "contains a colon"
classifies `Note: buy milk` as a URL, so an unknown scheme is only treated as a
link when it is followed by `//` or is one of the known opaque schemes like
`bitcoin:`. And Wi-Fi payloads backslash-escape separators, so a password of
`pa\;ss` is legal and must not be split on that semicolon — which is why the
parser is an explicit scan rather than a regex.

## Test

From `python/`:

```powershell
python -m unittest discover -s tests -v
```

152 tests, with no fixture files — every image is rendered by the project.

Covered: EAN-13 check digits against published examples; guard patterns and the
first-digit parity trick; rejection of bad check digits, lengths and geometry;
**round trips of our own encoder through OpenCV's decoders**, QR and EAN-13,
including the leading-zero code that comes back as UPC-A; the quiet zone
measured rather than assumed, landing exactly on a finder module; the zoom-out
retry rescuing a frame-filling barcode, and its rescued corners landing where an
unaided detection of the same symbol puts them; the single-code fallback recovering a payload the multi path drops;
upscaled detection still reporting original coordinates; detection/decoding
geometry, merging, and cross-symbology payload equivalence; every content kind,
`Note:`-style prose not being a URL, and escaped Wi-Fi separators; the cooldown
suppressing, expiring, refreshing while held, and keying on symbology; CSV
header, append, overwrite, per-row flush, missing parent directory and
unwritable path; scene synthesis with ground truth that moves under every
transform; centre occlusion defeating decoding while detection survives; overlay
drawing at frame edges; CLI validation; and end-to-end runs over rendered
images, a multi-code sheet, a blank frame, a corrupt file, and a written video
where twelve frames of one code produce exactly one logged scan.

## Project layout

```text
10-qr-barcode-scanner/
├── README.md
├── assets/input/ , assets/output/
├── results/                     # benchmark.md, benchmark.json, scans.csv
├── scripts/
│   ├── make_codes.py            # render codes, degrade them, write ground truth
│   └── benchmark.py             # every number in this README
└── python/
    ├── scanner.py               # entry point
    ├── requirements.txt
    ├── src/
    │   ├── app.py               # CLI, image and video loops, FPS
    │   ├── detect.py            # QR + barcode readers, fallback, zoom retry, merge
    │   ├── content.py           # payload classification
    │   ├── history.py           # cooldown dedup, scan list, CSV log
    │   ├── overlay.py           # outlines and HUD
    │   ├── symbols.py           # QR renderer and EAN-13 encoder
    │   ├── synth.py             # scenes, degradations, ground truth
    │   └── errors.py
    └── tests/
        └── test_scanner.py
```

## Common problems

**Nothing is detected at all —** check the code is in focus and fills a
reasonable part of the frame. For a small or distant code try `--upscale 2`;
below about two pixels per module nothing will help.

**A barcode image on disk is not read —** it is probably too large a fraction of
the frame. The zoom-out retry handles this automatically; if you disabled it with
`--zoom-out 1.0`, re-enable it.

**The outline shows but nothing decodes —** that is "detected, not decoded".
Straighten the code up, improve the light, or move closer. Under perspective
specifically, make sure you are on the default `--qr-detector aruco`.

**Live scanning feels sluggish —** use `--resolution 640x480`. QR detection cost
grows steeply with frame size and dominates everything else.

**The same code is logged repeatedly —** raise `--cooldown`.

**A code will not scan twice —** lower `--cooldown`.

**A barcode reads as `UPC_A` with twelve digits, not thirteen —** that is
correct. A leading-zero EAN-13 is a UPC-A, and the CSV records what the decoder
reported.

**Code 128 or Code 39 is not read —** OpenCV only ships the retail symbologies:
EAN-13, EAN-8, UPC-A, UPC-E. Those others need a different decoder.

## Limitations and possible improvements

- **Only retail 1D symbologies.** Code 128, Code 39, ITF and PDF417 are all
  outside what OpenCV's barcode module decodes, and nothing here can add them;
  that needs ZBar or ZXing.
- **QR detection is the bottleneck**, at roughly 6–10 fps on 1280×720 for a
  CPU-only laptop. Detecting on a downscaled frame and decoding on the full one
  would recover most of the cost and is the obvious next change.
- **No tracking between frames.** Every frame is detected from scratch, which
  wastes work on a code that has not moved. Following the previous quad and
  re-detecting only when it is lost would cut the cost sharply.
- **The accuracy numbers come from rendered images.** The geometry is honest —
  real homographies, real ground-truth corners — but synthetic codes are pure
  black on pure white, with no paper texture, specular highlight, or soft edge.
  The noise results in particular are optimistic for that reason.
- **Occlusion is measured with one solid patch.** Scattered damage is the case
  Reed-Solomon is designed for and would score considerably better; a single
  blob is the pessimistic end of the range.
- **Barcode outlines are indicative, not precise.** QR corners come back
  accurate to about a pixel, but OpenCV reports a 1D barcode as the region where
  it found coherent gradients, which measured about 27 px wide of the bars and
  20 px short of their full height on a 380x276 symbol. Good enough to point at
  the code, not good enough to crop it.
- **No curved-surface decoding, even with `--curved`.** The flag is wired to
  OpenCV's curved sampler for codes on a bottle or mug, but on the OpenCV build
  pinned in `python/requirements.txt` it recovers nothing on either detector:
  the ArUco default does not expose `decodeCurved` at all, and the classic
  detector's call is rejected by this build, so the retry is a silent no-op.
- **The scan log only grows.** There is no de-duplication across runs, no
  rotation, and no size limit on `results/scans.csv`.
- **Nothing is done with what is scanned.** A URL is recognised but not opened,
  and Wi-Fi credentials are parsed but not joined. Acting on payloads is a
  deliberate omission: a scanner that opens links from untrusted codes is a
  security problem, not a feature.
