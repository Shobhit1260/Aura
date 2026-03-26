import tensorflow as tf

# Load Keras model
model = tf.keras.models.load_model("tb_model.keras")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save file
with open("tb_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Converted successfully")