# train_model1.py
# Huấn luyện mô hình phân biệt: tái chế vs không tái chế

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# --- Cấu hình ---
data_dir = 'data/'  # Folder chứa "recyclable/" và "non_recyclable/"
img_size = (150, 150)
batch_size = 32
epochs = 10

# --- Tiền xử lý dữ liệu ---
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# --- Xây mô hình CNN đơn giản ---
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- Huấn luyện ---
model.fit(train_generator, validation_data=val_generator, epochs=epochs)

# --- Lưu mô hình ---
model.save('models/model1_binary_recyclable.h5')
print("✅ Đã lưu model1 thành công!")