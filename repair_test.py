import tensorflow as tf
import numpy as np

path = "results_v3/models/tamil_cnn_bilstm.keras"

print("Loading model...")

model = tf.keras.models.load_model(
    path,
    safe_mode=False,
    compile=False
)

print("Registering TensorFlow for Lambda functions...")

for layer_name in [
    "valid_token_mask",
    "negative_pool_values",
    "exclude_padding",
]:
    layer = model.get_layer(layer_name)
    layer.function.__globals__["tf"] = tf

print("Testing prediction...")

sample = np.array(["This is a test comment"], dtype=str)
prediction = model(sample, training=False).numpy()

print("Prediction shape:", prediction.shape)
print("Prediction scores:", prediction[0])
print("All scores finite:", bool(np.isfinite(prediction).all()))

if prediction.shape[1] == 5 and np.isfinite(prediction).all():
    print("TEST PASSED: Prediction generated.")
else:
    print("TEST FAILED: Output needs further inspection.")