# train_model1_resnet50.py
# Huấn luyện mô hình phân loại nhị phân (rác tái chế vs không tái chế) sử dụng ResNet50.

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import matplotlib.pyplot as plt
from datetime import datetime

# --- Cấu hình các tham số chính ---
# THAY ĐỔI ĐƯỜNG DẪN NÀY cho phù hợp với vị trí dataset của bạn
# Dataset này nên có 2 thư mục con, ví dụ: "recyclable" và "non_recyclable"
data_dir = 'Z:\\GarbageClassification\\datas' # Ví dụ: 'path/to/your/binary_classification_data'

img_size = (224, 224)  # Kích thước ảnh tiêu chuẩn cho ResNet50
batch_size = 16
epochs = 50
input_shape = (img_size[0], img_size[1], 3)
num_classes = 1 # Cho phân loại nhị phân (sử dụng sigmoid)

# --- Tạo thư mục lưu mô hình và logs ---
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
# Lưu models và logs trong thư mục con của script này
current_script_dir = os.path.dirname(os.path.abspath(__file__))
output_base_dir_name = "model1_ResNet50_output"

models_dir = os.path.join(current_script_dir, output_base_dir_name, 'models_ResNet50', timestamp)
logs_dir = os.path.join(current_script_dir, output_base_dir_name, 'logs_ResNet50', timestamp)

os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
print(f"Thư mục lưu mô hình: {models_dir}")
print(f"Thư mục lưu logs: {logs_dir}")

# --- Data Augmentation ---
train_datagen = ImageDataGenerator(
    rescale=1./255,             # Chuẩn hóa giá trị pixel
    validation_split=0.2,       # 20% dữ liệu cho validation
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,        # Thường không lật dọc cho các đối tượng có hướng tự nhiên
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# --- Tạo Data Generators ---
print("\nĐang tạo Data Generators...")
try:
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',      # Phân loại nhị phân
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
    print(f"Tìm thấy {train_generator.samples} ảnh cho training.")
    print(f"Tìm thấy {val_generator.samples} ảnh cho validation.")
    print(f"Các lớp tìm thấy: {train_generator.class_indices}")
    # Ví dụ class_indices có thể là: {'non_recyclable': 0, 'recyclable': 1}

except Exception as e:
    print(f"Lỗi khi tạo data generators: {e}")
    print(f"Hãy kiểm tra lại đường dẫn thư mục dữ liệu: {data_dir}")
    print("Thư mục dữ liệu nên có các thư mục con tương ứng với mỗi lớp.")
    exit()

# --- Xây dựng mô hình với ResNet50 (Transfer Learning) ---
print("\nĐang xây dựng kiến trúc mô hình với ResNet50...")
base_model_resnet = ResNet50(
    weights='imagenet',
    include_top=False,      # Bỏ lớp Dense cuối cùng của ResNet50
    input_shape=input_shape
)

# Chiến lược Fine-tuning cho ResNet50:
# ResNet50 có 175 lớp. `conv5_block1_0_conv` là lớp thứ 143.
# Chúng ta sẽ đóng băng các lớp trước khối conv5 (ví dụ, 140 lớp đầu)
# và cho phép huấn luyện (fine-tune) các lớp còn lại.
fine_tune_at_layer_index = 140 # Ví dụ: mở khóa từ khối conv5_block1 trở đi
print(f"Số lớp của ResNet50: {len(base_model_resnet.layers)}")
print(f"Đóng băng {fine_tune_at_layer_index} lớp đầu tiên của ResNet50.")

for layer in base_model_resnet.layers[:fine_tune_at_layer_index]:
    layer.trainable = False
# for layer in base_model_resnet.layers[fine_tune_at_layer_index:]:
#     layer.trainable = True # Mặc định các lớp sau đã là trainable nếu base_model.trainable = True

# Xây dựng mô hình tuần tự
model_resnet = models.Sequential([
    base_model_resnet,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),

    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    layers.Dense(num_classes, activation='sigmoid') # Lớp output cho phân loại nhị phân
])

# --- Cấu hình Optimizer và Biên dịch mô hình ---
initial_learning_rate_resnet = 1e-4 # Learning rate nhỏ cho fine-tuning
optimizer_resnet = optimizers.Adam(learning_rate=initial_learning_rate_resnet)

model_resnet.compile(
    optimizer=optimizer_resnet,
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)
model_resnet.summary()

# --- Callbacks ---
checkpoint_path_resnet = os.path.join(models_dir, 'model1_ResNet50_best.keras')
callbacks_list_resnet = [
    ModelCheckpoint(
        checkpoint_path_resnet,
        monitor='val_auc', # Theo dõi val_auc hoặc val_accuracy
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=15, # Tăng patience
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3, # Giảm LR mạnh hơn
        patience=7,  # Giảm patience
        min_lr=1e-7,
        verbose=1
    ),
    CSVLogger(os.path.join(logs_dir, 'model1_ResNet50_training_log.csv'))
    # tf.keras.callbacks.TensorBoard(log_dir=logs_dir, histogram_freq=1) # Bật nếu cần
]

# --- Bắt đầu huấn luyện ---
print("\nBắt đầu huấn luyện Model 1 (Phân loại nhị phân với ResNet50)...")
history_resnet = model_resnet.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=callbacks_list_resnet,
    verbose=1
)

# --- Lưu mô hình cuối cùng ---
final_model_path_resnet = os.path.join(models_dir, 'model1_ResNet50_final.keras')
model_resnet.save(final_model_path_resnet)
print(f"Đã lưu mô hình cuối cùng tại: {final_model_path_resnet}")

# --- Đánh giá mô hình (sử dụng mô hình tốt nhất đã khôi phục bởi EarlyStopping) ---
print("\nĐang đánh giá Model 1 (ResNet50) trên tập Validation...")
# Model_resnet đã có trọng số tốt nhất do restore_best_weights=True trong EarlyStopping
val_metrics_resnet = model_resnet.evaluate(val_generator, verbose=1)
metrics_names_resnet = ['loss', 'accuracy', 'auc', 'precision', 'recall']
print("\nKết quả đánh giá Model 1 (ResNet50):")
for name, value in zip(metrics_names_resnet, val_metrics_resnet):
    print(f"- {name}: {value:.4f}")

# --- Vẽ và lưu biểu đồ lịch sử huấn luyện ---
print("\nĐang vẽ và lưu biểu đồ lịch sử huấn luyện cho Model 1 (ResNet50)...")
plt.figure(figsize=(15, 12)) # Kích thước cho 5 plots

# Accuracy
plt.subplot(3, 2, 1)
plt.plot(history_resnet.history['accuracy'], label='Training Accuracy')
plt.plot(history_resnet.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model 1 (ResNet50): Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss
plt.subplot(3, 2, 2)
plt.plot(history_resnet.history['loss'], label='Training Loss')
plt.plot(history_resnet.history['val_loss'], label='Validation Loss')
plt.title('Model 1 (ResNet50): Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# AUC
plt.subplot(3, 2, 3)
plt.plot(history_resnet.history['auc'], label='Training AUC')
plt.plot(history_resnet.history['val_auc'], label='Validation AUC')
plt.title('Model 1 (ResNet50): AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)

# Precision
plt.subplot(3, 2, 4)
plt.plot(history_resnet.history['precision'], label='Training Precision')
plt.plot(history_resnet.history['val_precision'], label='Validation Precision')
plt.title('Model 1 (ResNet50): Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)

# Recall
plt.subplot(3, 2, 5)
plt.plot(history_resnet.history['recall'], label='Training Recall')
plt.plot(history_resnet.history['val_recall'], label='Validation Recall')
plt.title('Model 1 (ResNet50): Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()
plt.grid(True)

plt.tight_layout()
plot_save_path = os.path.join(logs_dir, 'model1_ResNet50_training_history.png')
plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Đã lưu biểu đồ lịch sử huấn luyện tại: {plot_save_path}")

print("\nHoàn thành quá trình huấn luyện Model 1 với ResNet50.")