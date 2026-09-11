# decodelabs-iris-classifier
#2 project

# Project 2: Data Classification Using AI
**DecodeLabs Internship — Batch 2026**

## Overview

A supervised learning classifier that predicts iris flower species from
measurements, using the K-Nearest Neighbors (KNN) algorithm. This is the
"predictive phase" project — instead of hand-writing rules like Project 1,
the machine learns patterns from labeled historical data and applies them
to new, unseen examples.

## Dataset

**Iris dataset** — a classic benchmark, loaded directly from
`sklearn.datasets` (no external file needed).

| | |
|---|---|
| Samples | 150 (balanced) |
| Classes | 3 — Setosa, Versicolor, Virginica |
| Features | 4 — sepal length, sepal width, petal length, petal width (all in cm) |

## Pipeline (IPO Framework)

### INPUT — Load & Scale
```python
X, y = iris.data, iris.target
```
Features are standardized with `StandardScaler` (mean = 0, variance = 1)
**after** the train/test split, fitting only on training data. KNN is
distance-based, so unscaled features on different ranges would unfairly
dominate the distance calculation. Fitting the scaler on the full dataset
(including test data) would leak test information into training — a
subtle but common mistake avoided here.

### PROCESS — Split & Train
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
```
- 80/20 train/test split, stratified so each class stays balanced in both sets.
- KNN classifies a new point by majority vote among its `K` closest
  neighbors in feature space — "similar things exist in close proximity."

### OUTPUT — Predict & Validate
```python
predictions = model.predict(X_test)
```
Evaluated with a **confusion matrix**, full **classification report**
(precision/recall/F1 per class), and **macro F1 score** — not just raw
accuracy. On imbalanced data, accuracy alone can be misleading (the
"Accuracy Mirage"); looking at precision/recall per class shows where
the model actually struggles.

## Results

| Metric | Score |
|---|---|
| Accuracy | 0.93 |
| Macro F1 | 0.9327 |

```
Confusion Matrix (rows = actual, columns = predicted)
[[10  0  0]
 [ 0 10  0]
 [ 0  2  8]]
```

Setosa was classified perfectly. The few errors were between Versicolor
and Virginica — the two species with the most overlapping measurements.

## Bonus: Tuning K

The script also sweeps `K` from 1–20 to find the "elbow" — the point
where error rate stabilizes. Very low `K` (e.g. K=1) risks overfitting to
noise; very high `K` risks underfitting into an overly generic boundary.

## How to Run

```bash
pip install scikit-learn
python classifier.py
```

## Key Concepts Demonstrated

- Supervised learning pipeline: load → split → scale → train → predict → evaluate
- Feature scaling and why it matters for distance-based algorithms
- Train/test split and stratification
- K-Nearest Neighbors classification
- Evaluation beyond accuracy (confusion matrix, precision, recall, F1)
- Hyperparameter tuning (choosing K)
