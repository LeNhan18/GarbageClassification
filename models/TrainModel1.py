# train_model1.py
# Huấn luyện mô hình phân loại nhị phân: rác tái chế vs không tái chế
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.applications import EfficientNetB2  # Nâng cấp lên B2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tensorflow as tf

# --- Cấu hình GPU để tăng tốc độ ---
try:
    # Giới hạn GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Giới hạn bộ nhớ GPU xuống 4GB cho RTX 3050
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
data_dir = 'Z:\\GarbageClassification\\data'
img_size = (240, 240)  # Tăng kích thước ảnh một chút
batch_size = 16  # Giảm batch size để cải thiện độ chính xác
epochs = 120  # Tăng số epochs hơn nữa
input_shape = (240, 240, 3)

# --- Data Augmentation mạnh hơn ---
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2,
    rotation_range=30,  # Tăng rotation range thêm
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,  # Tăng zoom range
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],  # Mở rộng brightness range
    fill_mode='nearest'
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
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

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
fine_tune_at = 100  # Giảm số lớp đóng băng nhiều hơn
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),

    # First Dense Block - tăng số neurons
    layers.Dense(1024, kernel_regularizer=regularizers.l2(0.0003)),  # Giảm regularization hơn nữa
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.4),

    # Second Dense Block
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0003)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),

    # Third Dense Block
    layers.Dense(256, kernel_regularizer=regularizers.l2(0.0003)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),

    # Output Layer
    layers.Dense(1, activation='sigmoid')
])

# --- Tối ưu hóa ---
# Sử dụng Adam optimizer với learning rate cố định
initial_learning_rate = 1e-4

optimizer = optimizers.Adam(
    learning_rate=initial_learning_rate,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(), 'Precision', 'Recall', tf.keras.metrics.F1Score(threshold=0.5)]
)

# --- Callbacks ---
callbacks = [
    ModelCheckpoint(
        os.path.join(models_dir, 'model1_best.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=20,  # Tăng patience thêm nữa
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(  # Thêm ReduceLROnPlateau để tự động điều chỉnh learning rate
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    ),
    CSVLogger(os.path.join(logs_dir, 'model1_training.csv'))
]

# --- Huấn luyện mô hình ---
print("\nBắt đầu huấn luyện mô hình phân loại nhị phân (rác tái chế vs không tái chế)...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)

# --- Lưu mô hình và đánh giá ---
model.save(os.path.join(models_dir, 'model1_final.keras'))
print("✅ Đã lưu mô hình thành công!")

# Đánh giá mô hình
val_metrics = model.evaluate(val_generator, verbose=1)
metrics_names = ['loss', 'accuracy', 'auc', 'precision', 'recall', 'f1_score']
print("\nKết quả đánh giá:")
for name, value in zip(metrics_names, val_metrics):
    print(f"{name}: {value:.4f}")

# Tạo confusion matrix cho đánh giá chi tiết
y_pred = []
y_true = []

# Dự đoán trên tập validation
val_generator.reset()
for i in range(len(val_generator)):
    x, y = val_generator[i]
    pred = model.predict(x, verbose=0)
    pred = (pred > 0.5).astype(int)
    y_pred.extend(pred.flatten())
    y_true.extend(y)
    if len(y_true) >= val_generator.samples:
        break

# Vẽ confusion matrix với cải tiến
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Confusion Matrix - Model 1', fontsize=16)
plt.ylabel('Actual Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)
class_names = list(train_generator.class_indices.keys())
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks + 0.5, class_names)
plt.yticks(tick_marks + 0.5, class_names)
plt.savefig(os.path.join(logs_dir, 'model1_confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# In classification report
report = classification_report(y_true, y_pred, target_names=class_names)
print("\nClassification Report:")
print(report)

# Lưu report vào file
with open(os.path.join(logs_dir, 'model1_classification_report.txt'), 'w') as f:
    f.write(report)

# Vẽ và lưu biểu đồ learning curves với cải tiến
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
plt.savefig(os.path.join(logs_dir, 'model1_training_history.png'), dpi=300, bbox_inches='tight')

# Thêm biểu đồ mới cho Precision và Recall
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['precision'], label='Training', linewidth=2)
plt.plot(history.history['val_precision'], label='Validation', linewidth=2)
plt.title('Model Precision', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

plt.subplot(1, 2, 2)
plt.plot(history.history['recall'], label='Training', linewidth=2)
plt.plot(history.history['val_recall'], label='Validation', linewidth=2)
plt.title('Model Recall', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Recall', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(logs_dir, 'model1_precision_recall.png'), dpi=300, bbox_inches='tight')
plt.close()

print("✅ Hoàn thành quá trình huấn luyện và đánh giá Model 1.")