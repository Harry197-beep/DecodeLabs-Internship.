# decode-labs-4-OCR

# Project 4: Image or Text Recognition (Basic) — OCR Recognizer
**DecodeLabs Internship — Batch 2026**

## Overview

A basic OCR (Optical Character Recognition) pipeline that extracts text
from an image, reports a confidence score for the extraction, and
produces an annotated image showing exactly what was detected and how
confident the model was. This is the "graduation to machine perception"
project — moving from structured data (Projects 1–3) into unstructured
visual data.

## Path Chosen: OCR (not Object Detection)

The brief offers two paths:

| | Path 1: OCR | Path 2: Object Detection |
|---|---|---|
| Goal | Extract machine-readable text | Identify & locate physical objects |
| Library | pytesseract / easyocr | cv2.dnn + MobileNet-SSD |
| Output | Text strings | Bounding boxes + object labels |

This project uses **OCR**, and specifically **`easyocr`** rather than
`pytesseract`. `pytesseract` is only a Python wrapper — it requires the
separate Tesseract OCR *engine* to be installed system-wide (via
Homebrew/apt), which turned into a long dependency chain on this machine
(Xcode Command Line Tools → Homebrew source builds → Python version
mismatches). `easyocr` is pure Python with no external binary
dependency, so it installs with a single `pip install` once the right
Python version is in place.

## Pipeline (IPO Framework)

### INPUT — Load
```python
image = cv2.imread(path)
```

### PROCESS — Pre-processing + OCR

**Step 1: Grayscale Conversion**
Collapses the 3D RGB matrix (H × W × 3) into a 1D intensity matrix
(H × W). Text detection is a brightness problem, not a color problem —
color data is irrelevant noise for this task.

**Step 2: Gaussian Blur**
Smooths out micro-imperfections and sensor/artifact noise *before*
thresholding, so a few stray noisy pixels don't get misread as text
edges.

**Step 3: Adaptive Thresholding**
Converts the smoothed grayscale image into pure black-and-white, using
a threshold computed *locally* per region rather than one global
cutoff. This matters because real photos rarely have even lighting —
one global threshold would wash out text in a shadowed corner while
overexposing text in a bright one.

```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresholded = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, blockSize=31, C=15,
)
```

**OCR extraction:**
```python
reader = easyocr.Reader(["en"], gpu=False)
results = reader.readtext(processed_image)
```
Returns each detected text region as `(bounding_box, text, confidence)`.

### OUTPUT — Confidence Report + Visual Confirmation

The script never blindly trusts OCR output. Every detection carries a
confidence score (the model's own estimate of how sure it is), and the
script:
- Computes the **average confidence** across all detections
- Flags any individual detection below the **80% threshold**
- Draws a **bounding box** around every detected word — green if it met
  the threshold, red if it didn't — with the confidence score printed
  above it

## Milestone Validation (per the project brief)

| Requirement | How this script satisfies it |
|---|---|
| 1. Library Integration | `easyocr`, wrapped with error handling for missing files |
| 2. Pre-Processing Integrity | Grayscale → Gaussian blur → **adaptive** thresholding (not a fixed global threshold) |
| 3. Accuracy Benchmarking | Reports per-detection and average confidence; explicitly checks against the required 80% minimum |
| 4. Visual Confirmation | Saves `annotated_output.png` with labeled bounding boxes + prints the clean extracted text |

## Result

Tested on `sample_text.png` (a synthetically generated image of the
text "DecodeLabs Project 4" with added pixel noise to simulate a real
photo):

```
=== Recognized Text ===
DecodeLabs Project 4

=== Confidence Report ===
Average confidence: 97.1%
Threshold: 80%
Status: PASSED accuracy benchmark.
```

## How to Run

Requires **Python 3.11 or 3.12** — `easyocr`'s dependency (PyTorch)
does not yet have pre-built wheels for Python 3.13, which will trigger
a very long, unnecessary source-compile if you're on 3.13. Using a
virtual environment on a supported Python version is strongly
recommended.

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install easyocr opencv-python-headless
python3 recognizer.py sample_text.png
```

**If you hit an SSL certificate error** on first run (while it
downloads the OCR model):
```bash
pip install --upgrade certifi
export SSL_CERT_FILE=$(python3 -m certifi)
```
Then re-run the script. Note this environment variable only lasts for
the current terminal session — re-run it if you open a new terminal
later.

**Note:** the first run downloads two small models (detection +
recognition, ~10–30 seconds combined). They're cached afterward, so
subsequent runs are fast.

## Key Concepts Demonstrated

- Treating an image as a numerical matrix, not a picture
- Pre-processing for computer vision (grayscale, blur, adaptive threshold)
- Why *adaptive* thresholding beats a single global threshold
- OCR confidence scores and why they shouldn't be trusted blindly
- Visual verification of model output, not just text/console output

## Possible Extensions

- Try Path 2 (Object Detection with MobileNet-SSD) for a different
  perception task — same IPO structure, different output shape
  (bounding boxes + labels instead of text)
- Add deskewing (Step 3 from the pre-processing slide) for photos taken
  at an angle
