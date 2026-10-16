
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping


# --------------------------------------------------
# 1. PROJECT SETTINGS
# --------------------------------------------------

SEED = 42
MAX_TOKENS = 10000
MAX_LENGTH = 60
EMBEDDING_DIM = 32
BATCH_SIZE = 64
EPOCHS = 8

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "model_input"
OUTPUT_DIR = BASE_DIR / "results_v2"
MODEL_DIR = OUTPUT_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

os.environ["PYTHONUTF8"] = "1"
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# --------------------------------------------------
# 2. LOAD AND CHECK DATA
# --------------------------------------------------

def load_data(language):
    train_df = pd.read_csv(
        DATA_DIR / f"{language}_train.csv"
    )
    dev_df = pd.read_csv(
        DATA_DIR / f"{language}_dev.csv"
    )
    test_df = pd.read_csv(
        DATA_DIR / f"{language}_test.csv"
    )

    for df in [train_df, dev_df, test_df]:
        if not {"text", "label"}.issubset(df.columns):
            raise ValueError(
                f"{language}: CSV must contain text and label columns."
            )

        df["text"] = df["text"].fillna("").astype(str)
        df["label"] = df["label"].astype(str)

    print(f"\n{'=' * 55}")
    print(f"LANGUAGE: {language.upper()}")
    print(f"Training rows:   {len(train_df)}")
    print(f"Validation rows: {len(dev_df)}")
    print(f"Test rows:       {len(test_df)}")
    print("Training class distribution:")
    print(train_df["label"].value_counts().to_string())

    return train_df, dev_df, test_df


# --------------------------------------------------
# 3. BUILD IMPROVED MODEL
# --------------------------------------------------

def build_model(train_text, number_of_classes, model_type):

    text_input = layers.Input(
        shape=(),
        dtype=tf.string,
        name="text"
    )

    vectorizer = layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=MAX_LENGTH,
        standardize="lower_and_strip_punctuation"
    )

    # Learn vocabulary only from training text.
    vectorizer.adapt(
        tf.data.Dataset.from_tensor_slices(
            train_text
        ).batch(256)
    )

    token_ids = vectorizer(text_input)

    # CNN branch: no masking metadata is passed to Conv1D.
    cnn_embedding = layers.Embedding(
        input_dim=MAX_TOKENS,
        output_dim=EMBEDDING_DIM,
        mask_zero=False,
        name="cnn_embedding"
    )(token_ids)

    cnn_features = layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="cnn"
    )(cnn_embedding)

    # Explicitly exclude padding positions from max pooling.
    valid_tokens = layers.Lambda(
        lambda x: tf.cast(
            tf.not_equal(x, 0), tf.float32
        ),
        name="valid_token_mask"
    )(token_ids)

    valid_tokens = layers.Reshape(
        (MAX_LENGTH, 1)
    )(valid_tokens)

    negative_values = layers.Lambda(
        lambda x: tf.ones_like(x) * (-1e9),
        name="negative_pool_values"
    )(cnn_features)

    masked_cnn_features = layers.Lambda(
        lambda inputs: tf.where(
            tf.cast(inputs[1], tf.bool),
            inputs[0],
            inputs[2]
        ),
        name="exclude_padding"
    )([cnn_features, valid_tokens, negative_values])

    cnn_output = layers.GlobalMaxPooling1D(
        name="cnn_pooling"
    )(masked_cnn_features)

    # Recurrent branch: preserve the padding mask.
    rnn_embedding = layers.Embedding(
        input_dim=MAX_TOKENS,
        output_dim=EMBEDDING_DIM,
        mask_zero=True,
        name="recurrent_embedding"
    )(token_ids)

    if model_type == "CNN-BiLSTM":
        recurrent_output = layers.Bidirectional(
            layers.LSTM(32),
            name="bilstm"
        )(rnn_embedding)

    elif model_type == "CNN-RNN":
        recurrent_output = layers.SimpleRNN(
            32,
            name="simple_rnn"
        )(rnn_embedding)

    elif model_type == "CNN-GRU":
        recurrent_output = layers.GRU(
            32,
            name="gru"
        )(rnn_embedding)

    else:
        raise ValueError(f"Unknown model: {model_type}")

    # Combine CNN and recurrent features.
    combined = layers.Concatenate()(
        [cnn_output, recurrent_output]
    )

    combined = layers.Dense(
        32,
        activation="relu"
    )(combined)

    combined = layers.Dropout(0.3)(combined)

    output = layers.Dense(
        number_of_classes,
        activation="softmax",
        name="classification"
    )(combined)

    model = Model(
        inputs=text_input,
        outputs=output,
        name=model_type.replace("-", "_")
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# --------------------------------------------------
# 4. TRAIN AND EVALUATE
# --------------------------------------------------

def run_experiment(language, model_type):

    train_df, dev_df, test_df = load_data(language)

    encoder = LabelEncoder()
    encoder.fit(train_df["label"])

    known_labels = set(encoder.classes_)

    for name, df in [("validation", dev_df), ("test", test_df)]:
        unknown = set(df["label"]) - known_labels
        if unknown:
            raise ValueError(
                f"Unknown labels in {name} data: {unknown}"
            )

    y_train = encoder.transform(train_df["label"])
    y_dev = encoder.transform(dev_df["label"])
    y_test = encoder.transform(test_df["label"])

    # Give more training importance to minority classes.
    class_ids = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=class_ids,
        y=y_train
    )

    class_weights = {
        int(class_id): float(weight)
        for class_id, weight in zip(class_ids, weights)
    }

    print(f"\nModel: {model_type}")
    print("Class weights:", class_weights)

    model = build_model(
        train_df["text"].to_numpy(),
        len(encoder.classes_),
        model_type
    )

    model.summary()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
        verbose=1
    )

    model.fit(
        train_df["text"].to_numpy(),
        y_train,
        validation_data=(
            dev_df["text"].to_numpy(),
            y_dev
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate only after training/model selection.
    probabilities = model.predict(
        test_df["text"].to_numpy(),
        batch_size=BATCH_SIZE,
        verbose=0
    )

    predictions = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_test, predictions)

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    report = classification_report(
        y_test,
        predictions,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0
    )

    print("\nTest results")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print("\nPer-class report:")
    print(report)

    safe_model_name = model_type.lower().replace("-", "_")

    model_path = (
        MODEL_DIR / f"{language}_{safe_model_name}.keras"
    )
    report_path = (
        OUTPUT_DIR / f"{language}_{safe_model_name}_report.txt"
    )

    model.save(model_path)

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(f"Language: {language}\n")
        file.write(f"Model: {model_type}\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Weighted F1: {weighted_f1:.4f}\n")
        file.write(f"Macro F1: {macro_f1:.4f}\n\n")
        file.write(report)

    return {
        "language": language,
        "model": model_type,
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "model_path": str(model_path)
    }


# --------------------------------------------------
# 5. RUN ALL SIX EXPERIMENTS
# --------------------------------------------------

if __name__ == "__main__":

    results = []

    experiments = [
        ("tamil", "CNN-BiLSTM"),
        ("tamil", "CNN-RNN"),
        ("tamil", "CNN-GRU"),
        ("malayalam", "CNN-BiLSTM"),
        ("malayalam", "CNN-RNN"),
        ("malayalam", "CNN-GRU")
    ]

    for language, model_type in experiments:
        result = run_experiment(language, model_type)
        results.append(result)

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "model_comparison_v2.csv",
        index=False
    )

    print("\nFINAL MODEL COMPARISON")
    print(
        results_df[
            [
                "language",
                "model",
                "accuracy",
                "weighted_f1",
                "macro_f1"
            ]
        ].to_string(index=False)
    )

    print("\nResults saved in:", OUTPUT_DIR)
