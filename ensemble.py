
from pathlib import Path
import json
import itertools

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "model_input"
MODEL_DIR = BASE_DIR / "results_v3" / "models"
OUTPUT_DIR = BASE_DIR / "results_ensemble"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGES = ["tamil", "malayalam"]
MODEL_TYPES = ["cnn_bilstm", "cnn_rnn", "cnn_gru"]
WEIGHT_STEP = 0.05


def load_data(language):
    train_df = pd.read_csv(DATA_DIR / f"{language}_train.csv")
    dev_df = pd.read_csv(DATA_DIR / f"{language}_dev.csv")
    test_df = pd.read_csv(DATA_DIR / f"{language}_test.csv")

    for df in [train_df, dev_df, test_df]:
        if not {"text", "label"}.issubset(df.columns):
            raise ValueError(
                f"{language}: missing text or label column"
            )
        df["text"] = df["text"].fillna("").astype(str)
        df["label"] = df["label"].astype(str)

    return train_df, dev_df, test_df


def load_one_model(language, model_type):
    model_path = MODEL_DIR / f"{language}_{model_type}.keras"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading: {model_path.name}")

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        safe_mode=False
    )

    # Supply TensorFlow to known saved Lambda functions.
    for layer_name in [
        "valid_token_mask",
        "negative_pool_values",
        "exclude_padding"
    ]:
        try:
            layer = model.get_layer(layer_name)
            if hasattr(layer, "function"):
                layer.function.__globals__["tf"] = tf
        except ValueError:
            pass

    return model


def get_probabilities(model, texts):
    texts = [str(text) for text in texts]

    text_tensor = tf.constant(texts, dtype=tf.string)

    probabilities = model(
        text_tensor,
        training=False
    )

    probabilities = np.asarray(probabilities)

    if probabilities.ndim != 2:
        raise ValueError(
            "Expected predictions shaped as "
            "(number_of_samples, number_of_classes)."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError(
            "Predictions contain NaN or infinity."
        )

    return probabilities

def tune_weights(dev_probabilities, y_dev):
    best_weights = None
    best_macro_f1 = -1.0
    best_weighted_f1 = -1.0
    steps = round(1 / WEIGHT_STEP)

    for a in range(steps + 1):
        for b in range(steps - a + 1):
            c = steps - a - b
            weights = np.array([a, b, c], dtype=float) / steps

            combined = sum(
                weights[i] * dev_probabilities[i]
                for i in range(3)
            )
            predictions = np.argmax(combined, axis=1)

            macro = f1_score(
                y_dev, predictions,
                average="macro", zero_division=0
            )
            weighted = f1_score(
                y_dev, predictions,
                average="weighted", zero_division=0
            )

            if (
                macro > best_macro_f1
                or (
                    np.isclose(macro, best_macro_f1)
                    and weighted > best_weighted_f1
                )
            ):
                best_macro_f1 = macro
                best_weighted_f1 = weighted
                best_weights = weights.copy()

    return best_weights, best_macro_f1, best_weighted_f1


def evaluate_predictions(
    language, model_name, y_true, probabilities, class_names
):
    predictions = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_true, predictions)
    weighted_f1 = f1_score(
        y_true, predictions,
        average="weighted", zero_division=0
    )
    macro_f1 = f1_score(
        y_true, predictions,
        average="macro", zero_division=0
    )

    report = classification_report(
        y_true,
        predictions,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0,
        digits=4
    )

    report_path = OUTPUT_DIR / f"{language}_{model_name}_report.txt"
    report_path.write_text(report, encoding="utf-8")

    print(f"\n{language.upper()} - {model_name}")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(report)

    return {
        "language": language,
        "model": model_name,
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1
    }


def run_experiment(language):
    print("\n" + "=" * 60)
    print(f"LANGUAGE: {language.upper()}")
    print("=" * 60)

    train_df, dev_df, test_df = load_data(language)

    encoder = LabelEncoder()
    encoder.fit(train_df["label"])
    class_names = list(encoder.classes_)

    for split_name, df in [
        ("development", dev_df),
        ("test", test_df)
    ]:
        unknown = set(df["label"]) - set(class_names)
        if unknown:
            raise ValueError(
                f"Unknown labels in {split_name}: {unknown}"
            )

    y_dev = encoder.transform(dev_df["label"])
    y_test = encoder.transform(test_df["label"])

    # Generate development probabilities.
    dev_probabilities = []

    for model_type in MODEL_TYPES:
        model = load_one_model(language, model_type)
        probs = get_probabilities(model, dev_df["text"].tolist())

        if probs.shape[1] != len(class_names):
            raise ValueError(
                f"{model_type}: output class count does not match labels."
            )

        dev_probabilities.append(probs)
        del model

    weights, dev_macro, dev_weighted = tune_weights(
        dev_probabilities, y_dev
    )

    print("\nSELECTED ENSEMBLE WEIGHTS")
    for name, weight in zip(MODEL_TYPES, weights):
        print(f"{name}: {weight:.2f}")

    print(f"Development Macro F1: {dev_macro:.4f}")
    print(f"Development Weighted F1: {dev_weighted:.4f}")

    weights_path = OUTPUT_DIR / f"{language}_weights.json"
    with open(weights_path, "w", encoding="utf-8") as file:
        json.dump({
            "language": language,
            "models": MODEL_TYPES,
            "weights": weights.tolist(),
            "classes": class_names,
            "selection_metric": "development_macro_f1",
            "weight_step": WEIGHT_STEP
        }, file, indent=4)

    # Evaluate the test set only after tuning on development data.
    test_probabilities = []

    for model_type in MODEL_TYPES:
        model = load_one_model(language, model_type)
        probs = get_probabilities(model, test_df["text"].tolist())
        test_probabilities.append(probs)
        del model

    results = []

    for model_type, probs in zip(MODEL_TYPES, test_probabilities):
        results.append(
            evaluate_predictions(
                language, model_type, y_test, probs, class_names
            )
        )

    ensemble_probabilities = sum(
        weights[i] * test_probabilities[i]
        for i in range(3)
    )

    results.append(
        evaluate_predictions(
            language,
            "tuned_ensemble",
            y_test,
            ensemble_probabilities,
            class_names
        )
    )

    return results


if __name__ == "__main__":
    all_results = []

    for language in LANGUAGES:
        all_results.extend(run_experiment(language))

    results_df = pd.DataFrame(all_results)
    results_path = OUTPUT_DIR / "ensemble_comparison.csv"
    results_df.to_csv(results_path, index=False)

    print("\nFINAL TEST COMPARISON")
    print(results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    ))
    print(f"\nResults saved to: {results_path}")
