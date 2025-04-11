# train_model1.py
# Huấn luyện mô hình phân biệt: tái chế vs không tái chế

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import VGG16, ResNet50

# --- Cấu hình ---
data_dir = 'data/'  # Folder chứa "recyclable/" và "non_recyclable/"
img_size = (150, 150)
batch_size = 32
epochs = 10
input_shape = (150, 150, 3)  # Thêm định nghĩa input_shape
dropout_rate = 0.5  # Thêm định nghĩa dropout_rate

# --- Tiền xử lý dữ liệu ---
datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

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

# --- Xây mô hình CNN nâng cao ---
model = models.Sequential([  # Sửa từ 'mmodel' thành 'model'
    # Block 1
    layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Block 2
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Block 3
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Block 4
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Fully connected layers
    layers.Flatten(),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(dropout_rate),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(dropout_rate),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# --- Biên dịch mô hình ---
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'AUC', 'Precision', 'Recall']  # Thêm metrics để đánh giá mô hình tốt hơn
)

# --- Tạo thư mục để lưu mô hình nếu chưa tồn tại ---
os.makedirs('models', exist_ok=True)

# --- Huấn luyện ---
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    verbose=1
)

# --- Lưu mô hình ---
model.save('models/model1_binary_recyclable.h5')
print("✅ Đã lưu model1 thành công!")

# --- Hiển thị tóm tắt kiến trúc mô hình ---
model.summary()