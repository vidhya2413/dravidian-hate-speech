from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "model_input"
MODEL_DIR = BASE_DIR / "results_v3" / "models"
OUTPUT_DIR = BASE_DIR / "results_v3" / "evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

experiments = [
    ("tamil", "cnn_bilstm"),
    ("tamil", "cnn_rnn"),
    ("tamil", "cnn_gru"),
    ("malayalam", "cnn_bilstm"),
    ("malayalam", "cnn_rnn"),
    ("malayalam", "cnn_gru"),
]

results = []

for language, model_name in experiments:
    print(f"\n{'=' * 55}")
    print(f"Evaluating: {language} - {model_name}")

    train_df = pd.read_csv(DATA_DIR / f"{language}_train.csv")
    test_df = pd.read_csv(DATA_DIR / f"{language}_test.csv")

    for df in (train_df, test_df):
        if not {"text", "label"}.issubset(df.columns):
            raise ValueError(
                f"{language}: CSV must contain text and label columns."
            )
        df["text"] = df["text"].fillna("").astype(str)
        df["label"] = df["label"].astype(str)

    encoder = LabelEncoder()
    encoder.fit(train_df["label"])

    unknown_labels = set(test_df["label"]) - set(encoder.classes_)
    if unknown_labels:
        raise ValueError(f"Unknown test labels: {unknown_labels}")

    y_true = encoder.transform(test_df["label"])

    model_path = MODEL_DIR / f"{language}_{model_name}.keras"

    model = tf.keras.models.load_model(
        model_path,
        safe_mode=False,
        compile=False,
    )

    # Temporary workaround for the saved Lambda functions.
    for layer_name in (
        "valid_token_mask",
        "negative_pool_values",
        "exclude_padding",
    ):
        layer = model.get_layer(layer_name)
        layer.function.__globals__["tf"] = tf

    probabilities = model(
        test_df["text"].to_numpy(dtype=str),
        training=False,
    ).numpy()

    if probabilities.shape != (len(test_df), len(encoder.classes_)):
        raise ValueError(
            f"Output/label mismatch for {language}-{model_name}: "
            f"predictions={probabilities.shape}, "
            f"classes={len(encoder.classes_)}"
        )

    if not np.isfinite(probabilities).all():
        raise ValueError("Predictions contain NaN or infinite values.")

    y_pred = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    macro_f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0,
    )

    print("Classes:", list(encoder.classes_))
    print(f"Test rows: {len(test_df)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\nPer-class report:\n", report)

    results.append({
        "language": language,
        "model": model_name,
        "test_rows": len(test_df),
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
    })

    report_path = OUTPUT_DIR / f"{language}_{model_name}_report.txt"
    report_path.write_text(
        f"Classes: {list(encoder.classes_)}\n"
        f"Test rows: {len(test_df)}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Weighted F1: {weighted_f1:.4f}\n"
        f"Macro F1: {macro_f1:.4f}\n\n"
        f"{report}",
        encoding="utf-8",
    )

results_df = pd.DataFrame(results)
results_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False,
)

print("\nFINAL COMPARISON")
print(results_df.to_string(index=False))
print(f"\nResults saved in: {OUTPUT_DIR}")