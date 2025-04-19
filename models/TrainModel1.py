# train_model1.py
# Huấn luyện mô hình phân biệt: tái chế vs không tái chế

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import tensorflow as tf

# --- Cấu hình ---
data_dir = 'Z:\\GarbageClassification\\data'  # Folder chứa "recyclable/" và "non_recyclable/"
img_size = (128, 128)  # Giảm kích thước ảnh
batch_size = 128  # Tăng batch size
epochs = 20
input_shape = (128, 128, 3)
dropout_rate = 0.5

# Cấu hình GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ Đã cấu hình GPU thành công")
    except RuntimeError as e:
        print(f"❌ Lỗi cấu hình GPU: {e}")

# --- Tiền xử lý dữ liệu với augmentation tối ưu ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,  # Giảm góc xoay
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_generator = val_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# --- Tạo thư mục lưu mô hình ---
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
os.makedirs(models_dir, exist_ok=True)
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(logs_dir, exist_ok=True)

# --- Callbacks tối ưu ---
model_checkpoint = ModelCheckpoint(
    os.path.join(models_dir, 'model1_best.keras'),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,  # Giảm patience
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,  # Giảm patience
    min_lr=1e-6,
    verbose=1
)

callbacks = [model_checkpoint, early_stopping, reduce_lr]

# --- Xây mô hình CNN nâng cao với kiến trúc tối ưu ---
model = models.Sequential([
    # Block 1
    layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Block 2
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Block 3
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Fully connected layers
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(dropout_rate),
    layers.Dense(1, activation='sigmoid')
])

# --- Biên dịch mô hình với optimizer tối ưu ---
optimizer = optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', 'AUC', 'Precision', 'Recall']
)

# --- Huấn luyện với tối ưu hóa ---
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)

# --- Lưu mô hình cuối cùng ---
model.save(os.path.join(models_dir, 'model1_binary_recyclable.keras'))
print("✅ Đã lưu model1 thành công!")

# --- Hiển thị tóm tắt kiến trúc mô hình ---
model.summary()

# --- Đánh giá mô hình ---
print("\nĐánh giá chi tiết mô hình:")
print("1. Đánh giá trên tập validation:")
val_loss, val_accuracy, val_auc, val_precision, val_recall = model.evaluate(val_generator)
print(f"- Loss: {val_loss:.4f}")
print(f"- Accuracy: {val_accuracy:.4f}")
print(f"- AUC: {val_auc:.4f}")
print(f"- Precision: {val_precision:.4f}")
print(f"- Recall: {val_recall:.4f}")

# Vẽ biểu đồ
plt.figure(figsize=(12, 4))

# Biểu đồ accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Biểu đồ loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(logs_dir, 'model1_training_history.png'))
plt.close()

# Lưu kết quả đánh giá
with open(os.path.join(logs_dir, 'model1_evaluation.txt'), 'w') as f:
    f.write(f"Validation Loss: {val_loss:.4f}\n")
    f.write(f"Validation Accuracy: {val_accuracy:.4f}\n")
    f.write(f"Validation AUC: {val_auc:.4f}\n")
    f.write(f"Validation Precision: {val_precision:.4f}\n")
    f.write(f"Validation Recall: {val_recall:.4f}\n")

print("\nĐã lưu kết quả đánh giá và biểu đồ vào thư mục logs")