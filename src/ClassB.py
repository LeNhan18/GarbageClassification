
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

base_model_1 = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model_1.trainable = False

model_1 = models.Sequential([
    base_model_1,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])
model_2 = model

