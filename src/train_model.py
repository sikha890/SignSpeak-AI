
"""Train and evaluate the SignSpeak-AI landmark classifier."""

import csv
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "raw" / "landmarks.csv"
MODEL_DIR = ROOT / "models"
MODEL_FILE = MODEL_DIR / "gesture_model.joblib"
RANDOM_STATE = 42


def load_dataset():
    """Load and validate the hand-landmark dataset."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Run src/collect_data.py first."
        )

    with DATA_FILE.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        expected_columns = ["label"] + [
            f"{axis}{i}"
            for i in range(21)
            for axis in ("x", "y", "z")
        ]

        if reader.fieldnames != expected_columns:
            raise ValueError(
                "Unexpected CSV columns. Check the dataset header."
            )

        rows = list(reader)

    if not rows:
        raise ValueError("The dataset contains no samples.")

    labels = []
    features = []

    for row_number, row in enumerate(rows, start=2):
        label = row["label"].strip()

        if not label:
            raise ValueError(f"Missing label on CSV row {row_number}.")

        try:
            values = [
                float(row[column])
                for column in expected_columns[1:]
            ]
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid coordinate on CSV row {row_number}."
            ) from error

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite coordinate on CSV row {row_number}."
            )

        labels.append(label)
        features.append(values)

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)

    if len(set(y)) < 2:
        raise ValueError(
            "At least two different gesture labels are needed."
        )

    return X, y


def main():
    """Train, evaluate, and save the gesture classifier."""
    X, y = load_dataset()

    print("=" * 50)
    print("SignSpeak-AI | Model Training")
    print("=" * 50)
    print("Dataset samples:", len(y))
    print("Feature count:", X.shape[1])
    print("Class counts:", dict(Counter(y)))

    # Create a stratified random holdout for preliminary evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTraining samples:", len(y_train))
    print("Testing samples:", len(y_test))

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\nTraining model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    labels = sorted(set(y))

    print(f"\nBaseline test accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=labels,
            zero_division=0,
            digits=4,
        )
    )

    print("Confusion matrix (labels sorted alphabetically):")
    print("Label order:", labels)
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=labels,
        )
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    print("\nModel saved successfully:")
    print(MODEL_FILE)
    print("\nTraining and evaluation completed.")


if __name__ == "__main__":
    main()
