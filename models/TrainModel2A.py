# train_model2a.py
# Huấn luyện mô hình phân loại chi tiết các loại rác tái chế

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tensorflow as tf
import json

# --- Cấu hình GPU để tăng tốc độ ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("✅ Đã cấu hình GPU thành công")
    except RuntimeError as e:
        print(f"❌ Lỗi cấu hình GPU: {e}")

# --- Cấu hình ---
data_dir = 'Z:\\GarbageClassification\\data\\recyclable'
img_size = (224, 224)
batch_size = 64
epochs = 30
input_shape = (224, 224, 3)

# --- Tiền xử lý dữ liệu với augmentation vừa phải ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Tạo data generators
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

# --- Tạo thư mục lưu mô hình và logs ---
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
os.makedirs(models_dir, exist_ok=True)
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(logs_dir, exist_ok=True)

# --- Xây dựng mô hình với Transfer Learning ---
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=input_shape
)

# Đóng băng các layer của base model
for layer in base_model.layers:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
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

# --- Callbacks ---
model_checkpoint = ModelCheckpoint(
    os.path.join(models_dir, 'model2a_best.keras'),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

csv_logger = CSVLogger(os.path.join(logs_dir, 'model2a_training.csv'))

callbacks = [model_checkpoint, early_stopping, reduce_lr, csv_logger]

# --- Huấn luyện mô hình ---
print("Bắt đầu huấn luyện mô hình...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)

# --- Lưu mô hình cuối cùng ---
model.save(os.path.join(models_dir, 'model2a_multiclass_recyclable.keras'))
print("✅ Đã lưu mô hình thành công!")

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
plt.savefig(os.path.join(logs_dir, 'model2a_training_history.png'))
plt.close()

# Lưu kết quả đánh giá
with open(os.path.join(logs_dir, 'model2a_evaluation.txt'), 'w') as f:
    f.write(f"Validation Loss: {val_loss:.4f}\n")
    f.write(f"Validation Accuracy: {val_accuracy:.4f}\n")
    f.write(f"Validation AUC: {val_auc:.4f}\n")
    f.write(f"Validation Precision: {val_precision:.4f}\n")
    f.write(f"Validation Recall: {val_recall:.4f}\n")

print("\n✅ Đã lưu kết quả đánh giá và biểu đồ vào thư mục logs")

# In ma trận nhầm lẫn
y_pred = model.predict(val_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_generator.classes

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(logs_dir, 'model2a_confusion_matrix.png'))
plt.close()

# In báo cáo phân loại
print("\nBáo cáo phân loại:")
print(classification_report(y_true, y_pred_classes, target_names=list(val_generator.class_indices.keys())))

# Lưu báo cáo phân loại
with open(os.path.join(logs_dir, 'model2a_classification_report.txt'), 'w') as f:
    f.write(classification_report(y_true, y_pred_classes, target_names=list(val_generator.class_indices.keys())))

# --- Lưu thông tin về lớp ---
class_indices = train_generator.class_indices
class_mapping = {v: k for k, v in class_indices.items()}
with open(os.path.join(models_dir, 'class_mapping_2a.json'), 'w') as f:
    json.dump(class_mapping, f)

print("✅ Đã lưu thông tin ánh xạ lớp thành công!")
print("✅ Hoàn thành quá trình huấn luyện.") 