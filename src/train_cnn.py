
"""Train and evaluate a 1D CNN for SignSpeak-AI hand landmarks."""

import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "landmarks.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_PATH = MODEL_DIR / "gesture_cnn.keras"
CLASSES_PATH = MODEL_DIR / "gesture_cnn_classes.json"
REPORT_PATH = REPORT_DIR / "cnn_evaluation_report.json"
CONFUSION_PATH = REPORT_DIR / "cnn_confusion_matrix.csv"


def load_dataset(path):
    """Load labels and numeric landmarks directly from CSV."""
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        header = next(reader)

        if len(header) != 64:
            raise ValueError(
                f"Expected 64 columns (label + 63 features), got {len(header)}"
            )

        rows = list(reader)

    if not rows:
        raise ValueError("Dataset is empty.")

    labels = np.array([row[0].strip() for row in rows])
    try:
        features = np.array(
            [[float(value) for value in row[1:]] for row in rows],
            dtype=np.float32,
        )
    except ValueError as exc:
        raise ValueError("Dataset contains a missing or non-numeric landmark.") from exc

    if features.shape[1] != 63 or not np.isfinite(features).all():
        raise ValueError("Expected 63 finite numeric landmark features per sample.")

    return header, features, labels


def stratified_split(labels, seed=SEED):
    """Split each class into train, validation, and test partitions."""
    rng = np.random.default_rng(seed)
    train_indices, val_indices, test_indices = [], [], []

    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)

        count = len(indices)
        n_test = max(1, round(count * 0.20))
        n_val = max(1, round(count * 0.10))

        test_indices.extend(indices[:n_test])
        val_indices.extend(indices[n_test:n_test + n_val])
        train_indices.extend(indices[n_test + n_val:])

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    return train_indices, val_indices, test_indices


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    header, features, labels = load_dataset(DATA_PATH)
    classes = sorted(np.unique(labels).tolist())
    class_to_id = {label: index for index, label in enumerate(classes)}
    encoded = np.array([class_to_id[label] for label in labels], dtype=np.int32)

    train_idx, val_idx, test_idx = stratified_split(labels)

    # Normalize each feature using training data only.
    mean = features[train_idx].mean(axis=0)
    std = features[train_idx].std(axis=0)
    std[std < 1e-7] = 1.0

    x_train = (features[train_idx] - mean) / std
    x_val = (features[val_idx] - mean) / std
    x_test = (features[test_idx] - mean) / std

    y_train = encoded[train_idx]
    y_val = encoded[val_idx]
    y_test = encoded[test_idx]

    # Conv1D expects (samples, sequence_length, channels).
    x_train = x_train[..., np.newaxis]
    x_val = x_val[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    print(f"Dataset: {len(features)} samples, {features.shape[1]} features")
    print(f"Classes: {classes}")
    print(
        f"Split: train={len(train_idx)}, "
        f"validation={len(val_idx)}, test={len(test_idx)}"
    )

    model = keras.Sequential([
        keras.Input(shape=(63, 1)),
        layers.Conv1D(32, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.25),
        layers.Conv1D(96, kernel_size=3, padding="same", activation="relu"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.30),
        layers.Dense(len(classes), activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
        ),
    ]

    model.summary()

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    probabilities = model.predict(x_test, verbose=0)
    predictions = probabilities.argmax(axis=1)

    # Build confusion matrix using NumPy only.
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(confusion, (y_test, predictions), 1)

    per_class = {}
    for index, label in enumerate(classes):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        per_class[label] = {
            "support": int(confusion[index, :].sum()),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
        }

    model.save(MODEL_PATH)

    with CLASSES_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "classes": classes,
                "feature_columns": header[1:],
                "mean": mean.tolist(),
                "std": std.tolist(),
            },
            file,
            indent=2,
        )

    with CONFUSION_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Actual/Predicted", *classes])
        for index, label in enumerate(classes):
            writer.writerow([label, *confusion[index].tolist()])

    report = {
        "model": "1D CNN",
        "dataset_samples": int(len(features)),
        "feature_count": int(features.shape[1]),
        "classes": classes,
        "split": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "test_loss": round(float(test_loss), 6),
        "test_accuracy": round(float(test_accuracy), 6),
        "per_class_metrics": per_class,
        "confusion_matrix_labels": classes,
        "confusion_matrix": confusion.tolist(),
        "notes": [
            "Normalization statistics were calculated from training data only.",
            "The split is stratified by label but not grouped by recording session.",
            "Samples from the same recording session may occur in multiple splits.",
            "Results may therefore overestimate performance on new recording sessions.",
        ],
    }

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\n--- CNN Evaluation ---")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.2%}")
    print("\nPer-class results:")
    for label, metrics in per_class.items():
        print(
            f"{label}: precision={metrics['precision']:.3f}, "
            f"recall={metrics['recall']:.3f}, "
            f"F1={metrics['f1_score']:.3f}, "
            f"support={metrics['support']}"
        )

    print(f"\nSaved CNN model: {MODEL_PATH}")
    print(f"Saved class/normalization metadata: {CLASSES_PATH}")
    print(f"Saved evaluation report: {REPORT_PATH}")
    print(f"Saved confusion matrix: {CONFUSION_PATH}")


if __name__ == "__main__":
    main()
