"""
DecodeLabs - Project 4: Image or Text Recognition (Basic)
Batch 2026

Path chosen: Path 1 - OCR (Optical Character Recognition)
Library: easyocr (pure Python - no separate system binary required,
unlike pytesseract which needs the Tesseract engine installed
separately via Homebrew/apt).

Pipeline (IPO):
  INPUT   -> Load a raw image (photo/scan of text)
  PROCESS -> Pre-processing (grayscale -> blur -> adaptive threshold)
             then OCR extraction with per-detection confidence scores
  OUTPUT  -> Cleaned text string + confidence report + annotated image

Why pre-process before OCR at all:
Raw images are cluttered with shadows, uneven lighting, and color noise
that OCR engines aren't built to reason about - they expect clean,
high-contrast input. Skipping this step is the single biggest cause of
garbage OCR output.

Step 1 - Grayscale Conversion:
Collapses the 3D RGB matrix (H x W x 3) into a 1D intensity matrix
(H x W). Removes irrelevant color data that has nothing to do with
where the text is.

Step 2 - Gaussian Blur:
Smooths out micro-imperfections and sensor/artifact noise before
thresholding, so the threshold step doesn't get confused by stray
noisy pixels.

Step 3 - Adaptive Thresholding:
Converts the smoothed grayscale image into pure black-and-white,
using a *locally* computed threshold per region (not one global
cutoff) - this handles uneven lighting across the image far better
than a single fixed brightness cutoff would.

Milestone validation this script satisfies (per the project brief):
  1. Library Integration      -> easyocr, error-handled
  2. Pre-Processing Integrity -> grayscale + adaptive threshold, both shown
  3. Accuracy Benchmarking    -> reports per-detection confidence, flags <80%
  4. Visual Confirmation      -> saves an annotated image + prints clean text
"""

import sys
import cv2
import easyocr

IMAGE_PATH = "sample_text.png"
CONFIDENCE_THRESHOLD = 80  # minimum acceptable confidence, per spec
OUTPUT_IMAGE_PATH = "annotated_output.png"


def load_image(path: str):
    """INPUT: Load the raw image from disk."""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at '{path}'. Check the path.")
    return image


def preprocess_image(image):
    """
    PROCESS - Step 1/2/3: Grayscale -> Gaussian Blur -> Adaptive Threshold.
    Returns the cleaned black-and-white image ready for OCR.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresholded = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=15,
    )
    return thresholded


def run_ocr(processed_image):
    """
    PROCESS: Run easyocr and return a list of
    (bounding_box, text, confidence) tuples.
    """
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    results = reader.readtext(processed_image)
    return results


def summarize_results(ocr_results):
    """
    Step 3 validation: Accuracy Benchmarking.
    Build the clean text string and flag any detection below the
    confidence threshold, so low-confidence output isn't silently
    trusted.
    """
    words, confidences, low_confidence_words = [], [], []

    for bbox, text, conf in ocr_results:
        text = text.strip()
        conf_pct = conf * 100

        if not text:
            continue

        words.append(text)
        confidences.append(conf_pct)

        if conf_pct < CONFIDENCE_THRESHOLD:
            low_confidence_words.append((text, conf_pct))

    clean_text = " ".join(words)
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    return clean_text, avg_confidence, low_confidence_words


def draw_annotations(image, ocr_results, save_path: str):
    """
    Step 4 validation: Visual Confirmation.
    Draw a bounding box + confidence label around every recognized word
    so the result can be verified visually, not just trusted as text.
    """
    annotated = image.copy()

    for bbox, text, conf in ocr_results:
        conf_pct = conf * 100
        if not text.strip():
            continue

        # easyocr returns 4 corner points; use them for the box
        pts = [(int(x), int(y)) for x, y in bbox]
        top_left = pts[0]

        color = (0, 200, 0) if conf_pct >= CONFIDENCE_THRESHOLD else (0, 0, 220)

        for i in range(4):
            cv2.line(annotated, pts[i], pts[(i + 1) % 4], color, 2)

        cv2.putText(
            annotated, f"{conf_pct:.0f}%",
            (top_left[0], max(top_left[1] - 8, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
        )

    cv2.imwrite(save_path, annotated)
    return save_path


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH

    print(f"Loading image: {image_path}")
    image = load_image(image_path)

    print("Pre-processing (grayscale -> blur -> adaptive threshold)...")
    processed = preprocess_image(image)

    print("Running OCR (first run downloads the model, ~10-30s)...")
    ocr_results = run_ocr(processed)

    clean_text, avg_confidence, low_confidence_words = summarize_results(ocr_results)

    print("\n=== Recognized Text ===")
    print(clean_text if clean_text else "(no text detected)")

    print(f"\n=== Confidence Report ===")
    print(f"Average confidence: {avg_confidence:.1f}%")
    print(f"Threshold: {CONFIDENCE_THRESHOLD}%")

    if avg_confidence >= CONFIDENCE_THRESHOLD:
        print("Status: PASSED accuracy benchmark.")
    else:
        print("Status: BELOW threshold - consider a cleaner input image.")

    if low_confidence_words:
        print(f"\nLow-confidence words ({len(low_confidence_words)}):")
        for word, conf in low_confidence_words:
            print(f"  '{word}' — {conf:.1f}%")

    output_path = draw_annotations(image, ocr_results, OUTPUT_IMAGE_PATH)
    print(f"\nAnnotated image saved to: {output_path}")


if __name__ == "__main__":
    main()