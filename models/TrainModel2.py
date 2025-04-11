# train_model2.py
# Huấn luyện mô hình CNN nâng cao để phân loại các loại rác tái chế (multi-class)

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

# --- Cấu hình ---
data_dir = 'data/recyclable/'  # Folder chứa plastic/, paper/, metal/... bên trong
img_size = (150, 150)
batch_size = 32
epochs = 30
input_shape = (150, 150, 3)

# --- Tiền xử lý dữ liệu ---
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# Validation chỉ cần rescale
val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

num_classes = train_generator.num_classes
print(f"Số lượng lớp phân loại: {num_classes}")
print(f"Tên các lớp: {list(train_generator.class_indices.keys())}")

# --- Tạo thư mục lưu mô hình ---
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)
logs_dir = 'logs'
os.makedirs(logs_dir, exist_ok=True)

# --- Callbacks ---
model_checkpoint = ModelCheckpoint(
    os.path.join(models_dir, 'model2_best.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    verbose=1
)

csv_logger = CSVLogger(os.path.join(logs_dir, 'training_log.csv'))

callbacks = [model_checkpoint, early_stopping, reduce_lr, csv_logger]

# --- Xây dựng mô hình CNN nâng cao ---
model = models.Sequential([
    # Block 1
    layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Block 2
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Block 3
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Block 4
    layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Fully connected layers
    layers.Flatten(),
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')  # Multi-class classification
])

# --- Biên dịch mô hình ---
optimizer = optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', 'AUC', 'Precision', 'Recall']
)

# In tóm tắt mô hình
model.summary()

# --- Huấn luyện ---
print("Bắt đầu huấn luyện mô hình CNN...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)

# --- Lưu mô hình cuối cùng ---
final_model_path = os.path.join(models_dir, 'model2_multiclass_recyclable.h5')
model.save(final_model_path)
print(f"✅ Đã lưu mô hình CNN thành công tại: {final_model_path}")

# --- Đánh giá mô hình ---
evaluation = model.evaluate(val_generator)
print("Đánh giá mô hình trên tập validation:")
for i, metric in enumerate(model.metrics_names):
    print(f"{metric}: {evaluation[i]}")

# --- Lưu thông tin về lớp ---
import json

class_indices = train_generator.class_indices
# Đảo ngược dictionary để lưu mapping từ index sang tên lớp
class_mapping = {v: k for k, v in class_indices.items()}
with open(os.path.join(models_dir, 'class_mapping.json'), 'w') as f:
    json.dump(class_mapping, f)

print("✅ Đã lưu thông tin ánh xạ lớp thành công!")
print("Hoàn thành quá trình huấn luyện.")