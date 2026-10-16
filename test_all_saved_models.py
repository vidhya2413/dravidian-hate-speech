import tensorflow as tf
import numpy as np
from pathlib import Path

model_dir = Path("results_v3/models")

models = [
    "tamil_cnn_bilstm.keras",
    "tamil_cnn_rnn.keras",
    "tamil_cnn_gru.keras",
    "malayalam_cnn_bilstm.keras",
    "malayalam_cnn_rnn.keras",
    "malayalam_cnn_gru.keras",
]

sample = np.array(
    ["This is a test comment"],
    dtype=str
)

for filename in models:
    path = model_dir / filename

    print(f"\n--- Testing {filename} ---")

    try:
        model = tf.keras.models.load_model(
            path,
            safe_mode=False,
            compile=False
        )

        for name in [
            "valid_token_mask",
            "negative_pool_values",
            "exclude_padding",
        ]:
            layer = model.get_layer(name)
            layer.function.__globals__["tf"] = tf

        prediction = model(sample, training=False).numpy()

        print("Output shape:", prediction.shape)
        print("Finite scores:", bool(np.isfinite(prediction).all()))
        print("Prediction:", prediction[0])
        print("STATUS: PASS")

    except Exception as error:
        print("STATUS: FAIL")
        print(type(error).__name__, error)