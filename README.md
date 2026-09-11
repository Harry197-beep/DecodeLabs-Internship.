# DecodeLabs AI Engineering Internship — Batch 2026

A portfolio of four progressive AI/ML projects completed as part of the
DecodeLabs Industrial Training Kit, moving from deterministic rule-based
logic through supervised learning, recommendation systems, and finally
computer vision (both text and object recognition).

Each project builds directly on the last: **dictionaries → supervised
classification → similarity-based ranking → visual perception.** The
throughline across all four: never trust a model's output blindly —
every project validates its own results (confidence scores, F1 metrics,
similarity percentages) rather than assuming correctness.

---

## Project 1: Rule-Based AI Chatbot

**Goal:** Build a chatbot using pure control flow — no machine learning,
just deterministic if-else logic — to master the fundamentals before
introducing probability into the mix.

**What it does:** Responds to greetings, questions, and exit commands
using a dictionary-based knowledge base instead of a long if-elif
chain, with input sanitization and a loose keyword-matching fallback.

**Key concepts:**
- Dictionary lookup (O(1)) vs. if-elif chains (O(n))
- Deterministic "white box" logic — every output traceable to an exact rule
- Why rule-based systems still matter: they're the guardrail layer sitting in front of real LLMs today

**Files:** `chatbot.py`

---

## Project 2: Data Classification Using AI

**Goal:** Move from hand-written rules into supervised learning — teach
a model to classify new data by learning from labeled examples.

**What it does:** Trains a K-Nearest Neighbors classifier on the Iris
flower dataset, predicting species from petal/sepal measurements.

**Result:** 93% accuracy, 0.93 macro F1-score.

**Key concepts:**
- Feature scaling (StandardScaler) — critical for distance-based algorithms
- Train/test split with stratification, to validate on unseen data
- The "Accuracy Mirage" — why raw accuracy can lie on imbalanced data
- Confusion matrix and F1 score as more honest evaluation metrics

**Files:** `classifier.py`

---

## Project 3: AI Recommendation Logic

**Goal:** Build a personalization engine — rank a large pool of items
by relevance to a specific user, without needing historical user data.

**What it does:** Takes 3+ user skills and matches them against 453
real job postings (scraped from Dice.com), returning the top-3 most
similar postings using content-based filtering.

**Key concepts:**
- Content-based filtering vs. collaborative filtering
- TF-IDF — downweights common terms (e.g. "python"), upweights rare/specific ones (e.g. "kubernetes")
- Cosine similarity vs. Euclidean distance — measuring angle (direction of interest) instead of raw distance (which unfairly penalizes shorter lists)
- Graceful degradation on out-of-vocabulary input (tested live with "LLM integration" / "genai" against a pre-LLM-era dataset)

**Files:** `reccomender.py`, `raw_skills.csv`

---

## Project 4: Image or Text Recognition

This project had two valid paths in the brief — both were built and
tested end-to-end.

### Path 1: OCR (Optical Character Recognition)

**Goal:** Extract machine-readable text from an image.

**What it does:** Pre-processes an image (grayscale → Gaussian blur →
adaptive thresholding) then runs OCR via `easyocr`, reporting a
confidence score per detection and flagging anything below an 80%
threshold.

**Result:** 97.1% average confidence on test input.

**Key concepts:**
- Treating an image as a numerical matrix, not a picture
- Why *adaptive* thresholding beats a single global threshold (handles uneven lighting)
- Confidence scores as probability, never certainty
- Visual verification of output (annotated bounding boxes), not just raw text

**Files:** `recognizer.py`, `sample_text.png`, `annotated_output.png`

### Path 2: Object Detection

**Goal:** Locate and label physical objects in an image or video.

**What it does:** Runs a pre-trained MobileNet-SSD network (PASCAL
VOC, 20 object classes) on both static images and video, drawing
labeled bounding boxes around every detected object above a confidence
threshold.

**Result:**
- Image test: bicycle 99.8%, car 99.4%, dog 96.7%
- Video test: 2260 person detections, 163 car, 82 aeroplane across 795 frames, processed at ~47 fps on CPU

**Key concepts:**
- MobileNet backbone (depthwise separable convolutions — lightweight, real-time capable)
- SSD ("Single Shot Detector") — one forward pass instead of sliding-window scanning
- Blob construction — resizing + normalizing input to match training conditions exactly
- Applying the same per-image pipeline to video, frame by frame
- Benchmarking a model's actual confidence distribution rather than assuming a threshold from a spec sheet

**Files:** `detector.py`, `detector_video.py`, `MobileNetSSD_deploy.prototxt`, `MobileNetSSD_deploy.caffemodel`, `sample_objects.jpg`, `test_video_1.avi`, `detected_output.png`, `detected_video_output.mp4`

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| ML / Data | scikit-learn, pandas, numpy |
| Computer Vision | OpenCV, EasyOCR, MobileNet-SSD (Caffe) |
| Core Techniques | TF-IDF, Cosine Similarity, K-Nearest Neighbors, Adaptive Thresholding, Single Shot Detection |

## Repository Structure

Each project lives in its own repo (linked below), reflecting a
week-by-week build as part of the internship's progressive milestone
system:

- **Project 1:** Rule-Based AI Chatbot
- **Project 2:** Iris Data Classifier
- **Project 3:** Tech Stack Recommender
- **Project 4 (OCR):** Text Recognition
- **Project 4 (Extension):** Object Detection — Image & Video

## What This Demonstrates

This progression intentionally mirrors how real AI systems are built
in practice: start with deterministic, explainable logic; layer in
supervised learning once the fundamentals are solid; add
personalization/ranking; and finally extend into unstructured data
(vision). Every project includes honest evaluation — accuracy isn't
assumed, it's measured and reported, including the cases where a model
predictably struggles (e.g., matching modern AI terminology against a
2017-era job dataset, or an older detection model reporting lower raw
confidence than a spec sheet expects).
