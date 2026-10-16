
from pathlib import Path
import json
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

# -----------------------------
# 1. PROJECT SETTINGS
# -----------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "model_input"
OUTPUT_DIR = BASE_DIR / "results"
MODEL_DIR = OUTPUT_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
MAX_TOKENS = 10000
MAX_LENGTH = 60
EMBEDDING_DIM = 32
BATCH_SIZE = 64
EPOCHS = 3

tf.keras.utils.set_random_seed(SEED)

print("TensorFlow:", tf.__version__)
print("Devices:", tf.config.list_physical_devices())
print("Epochs per model:", EPOCHS)

# -----------------------------
# 2. BUILD THE MODEL
# -----------------------------
def build_model(train_text, number_of_classes, architecture):
    vectorizer = layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=MAX_LENGTH
    )

    # Learn vocabulary from training text only.
    text_dataset = tf.data.Dataset.from_tensor_slices(
        train_text
    ).batch(256)

    vectorizer.adapt(text_dataset)

    vocabulary_size = len(vectorizer.get_vocabulary())

    text_input = layers.Input(
        shape=(),
        dtype=tf.string,
        name="text"
    )

    tokens = vectorizer(text_input)

    embedding = layers.Embedding(
        input_dim=vocabulary_size,
        output_dim=EMBEDDING_DIM,
        mask_zero=True
    )(tokens)

    # CNN branch
    cnn = layers.Conv1D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu"
    )(embedding)

    cnn = layers.GlobalMaxPooling1D()(cnn)

    # Recurrent branch
    if architecture == "CNN-BiLSTM":
        recurrent = layers.Bidirectional(
            layers.LSTM(32)
        )(embedding)

    elif architecture == "CNN-RNN":
        recurrent = layers.SimpleRNN(32)(embedding)

    elif architecture == "CNN-GRU":
        recurrent = layers.GRU(32)(embedding)

    else:
        raise ValueError("Unknown architecture: " + architecture)

    combined = layers.Concatenate()([cnn, recurrent])

    hidden = layers.Dense(32, activation="relu")(combined)
    hidden = layers.Dropout(0.3)(hidden)

    output = layers.Dense(
        number_of_classes,
        activation="softmax"
    )(hidden)

    model = Model(inputs=text_input, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# -----------------------------
# 3. LOAD AND TRAIN
# -----------------------------
all_results = []

experiments = [
    ("Tamil", "tamil"),
    ("Malayalam", "malayalam")
]

architectures = [
    "CNN-BiLSTM",
    "CNN-RNN",
    "CNN-GRU"
]

for language, prefix in experiments:
    print("\n" + "=" * 65)
    print("LANGUAGE:", language)
    print("=" * 65)

    train_df = pd.read_csv(
        DATA_DIR / f"{prefix}_train.csv",
        encoding="utf-8"
    )
    dev_df = pd.read_csv(
        DATA_DIR / f"{prefix}_dev.csv",
        encoding="utf-8"
    )
    test_df = pd.read_csv(
        DATA_DIR / f"{prefix}_test.csv",
        encoding="utf-8"
    )

    train_text = train_df["text"].fillna("").astype(str).to_numpy()
    dev_text = dev_df["text"].fillna("").astype(str).to_numpy()
    test_text = test_df["text"].fillna("").astype(str).to_numpy()

    encoder = LabelEncoder()

    y_train = encoder.fit_transform(train_df["label"].astype(str))
    y_dev = encoder.transform(dev_df["label"].astype(str))
    y_test = encoder.transform(test_df["label"].astype(str))

    number_of_classes = len(encoder.classes_)

    print("Training examples:", len(train_text))
    print("Development examples:", len(dev_text))
    print("Test examples:", len(test_text))
    print("Classes:", list(encoder.classes_))

    # Balance classes during training.
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights = {
        int(class_id): float(weight)
        for class_id, weight in zip(
            np.unique(y_train),
            class_weights_array
        )
    }

    for architecture in architectures:
        print("\n" + "-" * 55)
        print(f"Training {language} - {architecture}")
        print("-" * 55)

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)

        model = build_model(
            train_text,
            number_of_classes,
            architecture
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=1,
                restore_best_weights=True
            )
        ]

        history = model.fit(
            train_text,
            y_train,
            validation_data=(dev_text, y_dev),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate on the held-out test set.
        probabilities = model.predict(
            test_text,
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
            labels=np.arange(number_of_classes),
            target_names=encoder.classes_,
            zero_division=0
        )

        experiment_name = f"{prefix}_{architecture.lower()}"

        model_path = MODEL_DIR / f"{experiment_name}.keras"
        model.save(model_path)

        report_path = OUTPUT_DIR / f"{experiment_name}_report.txt"
        report_path.write_text(report, encoding="utf-8")

        label_path = OUTPUT_DIR / f"{prefix}_labels.json"
        label_path.write_text(
            json.dumps(
                encoder.classes_.tolist(),
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

        result = {
            "language": language,
            "model": architecture,
            "accuracy": accuracy,
            "weighted_f1": weighted_f1,
            "macro_f1": macro_f1,
            "epochs_completed": len(history.history["loss"])
        }

        all_results.append(result)

        print("\nTest results:")
        print(f"Accuracy:   {accuracy:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")
        print(f"Macro F1:    {macro_f1:.4f}")
        print("\nPer-class report:")
        print(report)
        print("Saved model:", model_path)

        del model
        del history

# -----------------------------
# 4. SAVE COMPARISON TABLE
# -----------------------------
results_df = pd.DataFrame(all_results)

results_path = OUTPUT_DIR / "model_comparison.csv"

results_df.to_csv(
    results_path,
    index=False,
    encoding="utf-8"
)

print("\n" + "=" * 65)
print("FINAL MODEL COMPARISON")
print("=" * 65)

print(
    results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)

print("\nResults saved to:", results_path)
print("All experiments completed.")
