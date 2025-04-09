# convert_tflite.py
import tensorflow as tf


def convert_to_tflite(h5_path, tflite_path):
    model = tf.keras.models.load_model(h5_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # optional
    tflite_model = converter.convert()

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"✅ Converted {h5_path} → {tflite_path}")


# # Convert 2 models
# convert_to_tflite('models/model1_binary_recyclable.h5', 'assets/model1.tflite')
# convert_to_tflite('models/model2_multiclass_recyclable.h5', 'assets/model2.tflite')
