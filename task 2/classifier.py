"""
DecodeLabs - Project 2: Data Classification Using AI
Batch 2026

Pipeline (per the brief's IPO framework):
  INPUT   -> Load Iris dataset, scale features
  PROCESS -> Train/test split, fit KNN classifier
  OUTPUT  -> Predict, evaluate with confusion matrix + F1 score

Why scaling matters: Iris features are all in similar ranges (cm),
but in general KNN is distance-based, so features on different scales
would dominate the distance calculation unfairly. StandardScaler fixes
that (mean=0, variance=1) - the "Gatekeeper Rule" slide.

Why train/test split: we need to validate on data the model has never
seen. Training and testing on the same data gives a fake, inflated
accuracy score - the "Accuracy Mirage" the deck warns about.

Why K-Nearest Neighbors: simple, intuitive supervised algorithm.
Classifies a new point by majority vote among its K closest neighbors
in feature space ("similar things exist in close proximity").
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score


def load_data():
    """Load the Iris dataset: 150 samples, 4 features, 3 balanced classes."""
    iris = load_iris()
    X, y = iris.data, iris.target
    return X, y, iris.target_names


def prepare_data(X, y, test_size=0.2, random_state=42):
    """
    Split into train/test sets, then scale features.

    IMPORTANT: fit the scaler on training data only, then transform
    both train and test with those same parameters. Fitting on the
    full dataset (including test) would leak test information into
    training - a subtle but common mistake.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_model(X_train, y_train, n_neighbors=5):
    """Instantiate and fit a KNN classifier."""
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, class_names):
    """Predict on the test set and report metrics beyond raw accuracy."""
    predictions = model.predict(X_test)

    print("=== Confusion Matrix ===")
    print("(rows = actual class, columns = predicted class)")
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    print()

    print("=== Classification Report ===")
    print(classification_report(y_test, predictions, target_names=class_names))

    f1 = f1_score(y_test, predictions, average="macro")
    print(f"Macro F1 Score: {f1:.4f}")

    return predictions, f1


def find_best_k(X_train, y_train, X_test, y_test, k_range=range(1, 21)):
    """
    Bonus: sweep K values to find the 'elbow' - the point where
    error rate stops improving. Low K overfits to noise, high K
    underfits and becomes too generic.
    """
    print("\n=== Tuning K (accuracy per K) ===")
    best_k, best_acc = 1, 0
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        print(f"K={k:2d}  accuracy={acc:.4f}")
        if acc > best_acc:
            best_k, best_acc = k, acc
    print(f"\nBest K found: {best_k} (accuracy={best_acc:.4f})")
    return best_k


def main():
    X, y, class_names = load_data()
    print(f"Loaded Iris dataset: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(class_names)} classes {list(class_names)}\n")

    X_train, X_test, y_train, y_test = prepare_data(X, y)
    print(f"Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples\n")

    model = train_model(X_train, y_train, n_neighbors=5)
    evaluate_model(model, X_test, y_test, class_names)

    find_best_k(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
