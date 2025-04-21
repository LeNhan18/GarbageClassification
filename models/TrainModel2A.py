# train_model2A.py
# Huấn luyện mô hình phân loại rác tái chế

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import matplotlib.pyplot as plt
import tensorflow as tf

# --- Cấu hình GPU để tăng tốc độ ---
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        )
        print("✅ Đã cấu hình GPU thành công")
    else:
        print("❌ Không tìm thấy GPU")
except Exception as e:
    print(f"❌ Lỗi cấu hình GPU: {e}")

# --- Cấu hình ---
data_dir = 'Z:\\GarbageClassification\\data\\recyclable'
img_size = (224, 224)  # Giảm kích thước ảnh xuống
batch_size = 32  # Tăng batch size
epochs = 50  # Giảm epochs
input_shape = (224, 224, 3)

# --- Data Augmentation vừa phải ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,  # Giảm rotation
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Tạo generators
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

# --- Tạo thư mục lưu mô hình và logs ---
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# --- Xây dựng mô hình với EfficientNetB0 ---
base_model = EfficientNetB0(  # Đổi về B0 cho nhẹ hơn
    weights='imagenet',
    include_top=False,
    input_shape=input_shape
)

# Fine-tune từ block 6 trở đi
fine_tune_at = 156  # Block 6 của EfficientNetB0
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    
    # First Dense Block - giảm số neurons
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),
    
    # Output Layer
    layers.Dense(len(train_generator.class_indices), activation='softmax')
])

# --- Tối ưu quá trình huấn luyện ---
initial_learning_rate = 1e-4  # Tăng learning rate
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(  # Đổi sang ExponentialDecay
    initial_learning_rate,
    decay_steps=train_generator.samples // batch_size * 5,
    decay_rate=0.9,
    staircase=True
)

optimizer = optimizers.Adam(
    learning_rate=lr_schedule,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
)

# --- Callbacks ---
callbacks = [
    ModelCheckpoint(
        os.path.join(models_dir, 'model2A_best.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=10,  # Giảm patience
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,  # Giảm patience
        min_lr=1e-6,
        verbose=1
    ),
    CSVLogger(os.path.join(logs_dir, 'model2A_training.csv'))
]

# --- Huấn luyện mô hình ---
print("\nBắt đầu huấn luyện Model 2A (Phân loại rác tái chế)...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)

# --- Lưu mô hình và đánh giá ---
model.save(os.path.join(models_dir, 'model2A_final.keras'))
print("✅ Đã lưu mô hình thành công!")

# Đánh giá mô hình
val_metrics = model.evaluate(val_generator, verbose=1)
metrics_names = ['loss', 'accuracy', 'top_2_accuracy']
print("\nKết quả đánh giá Model 2A:")
for name, value in zip(metrics_names, val_metrics):
    print(f"{name}: {value:.4f}")

# Vẽ và lưu biểu đồ
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model 2A Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model 2A Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(logs_dir, 'model2A_training_history.png'))
plt.close()

print("✅ Hoàn thành quá trình huấn luyện Model 2A.") 