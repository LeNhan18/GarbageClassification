# train_model2B.py
# Huấn luyện mô hình phân loại rác không tái chế

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
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
        print("Đã cấu hình GPU thành công")
    else:
        print("Không tìm thấy GPU")
except Exception as e:
    print(f"Lỗi cấu hình GPU: {e}")

# --- Cấu hình ---
data_dir = 'Z:\\GarbageClassification\\data\\non_recyclable'
img_size = (240, 240)
batch_size = 16
epochs = 100  # Tăng epochs
input_shape = (240, 240, 3)

# --- Data Augmentation cân bằng hơn ---
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
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

# --- Tính toán class weights ---
total_samples = sum(train_generator.classes)
class_weights = {}
for class_idx in range(len(train_generator.class_indices)):
    class_samples = np.sum(train_generator.classes == class_idx)
    class_weights[class_idx] = total_samples / (len(train_generator.class_indices) * class_samples)

print("\nClass weights:")
for class_name, class_idx in train_generator.class_indices.items():
    print(f"{class_name}: {class_weights[class_idx]:.2f}")

# --- Tạo thư mục lưu mô hình và logs ---
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# --- Xây dựng mô hình với EfficientNetB2 ---
base_model = EfficientNetB2(
    weights='imagenet',
    include_top=False,
    input_shape=input_shape
)

# Fine-tune nhiều layer hơn
fine_tune_at = 150
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),

    # First Dense Block
    layers.Dense(2048, kernel_regularizer=regularizers.l2(0.0001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),

    # Second Dense Block
    layers.Dense(1024, kernel_regularizer=regularizers.l2(0.0001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.4),

    # Third Dense Block
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),

    # Output Layer
    layers.Dense(len(train_generator.class_indices), activation='softmax')
])

# --- Tối ưu hóa ---
initial_learning_rate = 5e-5

optimizer = optimizers.Adam(
    learning_rate=initial_learning_rate,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')
    ]
)

# --- Callbacks ---
callbacks = [
    ModelCheckpoint(
        os.path.join(models_dir, 'model2B_best.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    CSVLogger(os.path.join(logs_dir, 'model2B_training.csv'))
]

# --- Huấn luyện mô hình ---
print("\nBắt đầu huấn luyện Model 2B (Phân loại rác không tái chế)...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=callbacks,
    class_weight=class_weights,  # Thêm class weights
    verbose=1
)

# --- Lưu mô hình và đánh giá ---
model.save(os.path.join(models_dir, 'model2B_final.keras'))
print("Đã lưu mô hình thành công!")

# Đánh giá mô hình
val_metrics = model.evaluate(val_generator, verbose=1)
metrics_names = ['loss', 'accuracy', 'auc', 'precision', 'recall', 'top_2_accuracy']
print("\nKết quả đánh giá Model 2B:")
for name, value in zip(metrics_names, val_metrics):
    print(f"{name}: {value:.4f}")

# Tạo confusion matrix
y_pred = []
y_true = []

val_generator.reset()
for i in range(len(val_generator)):
    x, y = val_generator[i]
    pred = model.predict(x, verbose=0)
    pred_classes = np.argmax(pred, axis=1)
    y_pred.extend(pred_classes)
    y_true.extend(np.argmax(y, axis=1))
    if len(y_true) >= val_generator.samples:
        break

y_true = y_true[:val_generator.samples]
y_pred = y_pred[:val_generator.samples]

# Vẽ confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Confusion Matrix - Model 2B', fontsize=16)
plt.ylabel('Actual Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)
class_names = list(train_generator.class_indices.keys())
tick_marks = np.arange(len(class_names)) + 0.5
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
plt.tight_layout()
plt.savefig(os.path.join(logs_dir, 'model2B_confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# In classification report
report = classification_report(y_true, y_pred, target_names=class_names)
print("\nClassification Report:")
print(report)

# Lưu report
with open(os.path.join(logs_dir, 'model2B_classification_report.txt'), 'w') as f:
    f.write(report)

# Vẽ learning curves
plt.figure(figsize=(20, 8))

plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Training', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
plt.title('Model Accuracy', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Training', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation', linewidth=2)
plt.title('Model Loss', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

plt.subplot(1, 3, 3)
plt.plot(history.history['auc'], label='Training AUC', linewidth=2)
plt.plot(history.history['val_auc'], label='Validation AUC', linewidth=2)
plt.title('Model AUC', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('AUC', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(logs_dir, 'model2B_training_history.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Hoàn thành quá trình huấn luyện và đánh giá Model 2B.")
