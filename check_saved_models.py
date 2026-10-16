from pathlib import Path
import pandas as pd
import tensorflow as tf

model_path = Path("results_v2/models/tamil_cnn_bilstm.keras")
data_path = Path("data/model_input/tamil_dev.csv")

print("Loading saved model...")

try:
    model = tf.keras.models.load_model(
        model_path,
        safe_mode=False,
        compile=False
    )

    print("Model loaded successfully!")

    data = pd.read_csv(data_path)
    texts = data["text"].fillna("").astype(str).head(3).to_numpy()

    predictions = model.predict(texts, verbose=0)

    print("Prediction shape:", predictions.shape)
    print("Predictions generated successfully!")

except Exception as error:
    print("Error type:", type(error).__name__)
    print("Error details:", error)