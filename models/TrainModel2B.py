# train_model2B_final_solution.py
# Huấn luyện mô hình phân loại rác không tái chế - Giải pháp với custom generator

import os
import numpy as np
import tensorflow as tf # Nên import tensorflow trước
from cv2.gapi import kernel
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.applications import EfficientNetB2 # Sửa đường dẫn import
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from datetime import datetime # Thêm datetime nếu chưa có

from CovertTFlite import keras_model_path

# --- Cấu hình GPU ---
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Đã cấu hình GPU thành công. Số lượng GPU tìm thấy: {len(gpus)}")
    else:
        print("Không tìm thấy GPU. Đang sử dụng CPU.")
except Exception as e:
    print(f"Lỗi cấu hình GPU: {e}")

# --- Cấu hình các tham số chính ---
data_dir = 'Z:\\GarbageClassification\\data\\non_recyclable'
img_size = (224, 224)  # Cân nhắc (260, 260) là kích thước chuẩn cho EfficientNetB2
batch_size = 16
epochs = 70 # Bạn có thể điều chỉnh lại nếu cần
input_shape = (img_size[0], img_size[1], 3)

# --- Cấu hình Data Augmentation ---
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    brightness_range=[0.7, 1.3]
)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2
)

# --- Tạo Data Generators ---
print("\nĐang tạo Data Generators...")
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

num_classes = len(train_generator.class_indices)
print(f"Số lượng lớp phân loại tìm thấy: {num_classes}")
print(f"Tên các lớp: {train_generator.class_indices}")

# --- Hàm tạo Generator với Sample Weights ---
def create_multi_class_weighted_generator(generator, class_weights_dictionary_local):
    if class_weights_dictionary_local is None:
        print("CẢNH BÁO: class_weights_dictionary là None. Sẽ yield (x, y, trọng số mặc định là 1).")
        while True:
            x_batch, y_batch = next(generator)
            sample_weights = np.ones(y_batch.shape[0], dtype=np.float32)
            yield x_batch, y_batch, sample_weights
    else:
        print("Thông báo: Sử dụng generator tùy chỉnh để yield (x, y, sample_weights).")
        while True:
            x_batch, y_batch = next(generator)
            y_integer_labels = np.argmax(y_batch, axis=1)
            sample_weights = np.array(
                [class_weights_dictionary_local.get(label_idx, 1.0) for label_idx in y_integer_labels],
                dtype=np.float32
            )
            yield x_batch, y_batch, sample_weights

# --- Tính toán Class Weights ---
print("\nĐang tính toán Class Weights...")
train_labels_for_weights = train_generator.classes
unique_classes, counts = np.unique(train_labels_for_weights, return_counts=True)
class_weights_dict = None

if len(unique_classes) >= 2 and len(unique_classes) == num_classes:
    print(f"Phân bố mẫu thực tế từ generator (chỉ số lớp: số lượng): {dict(zip(unique_classes, counts))}")
    try:
        computed_class_weights_array = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=train_labels_for_weights
        )
        class_weights_dict = dict(zip(unique_classes, computed_class_weights_array)) # Sử dụng unique_classes làm key
        print(f"Class weights dictionary sẽ được dùng bởi generator: {class_weights_dict}")
    except Exception as e:
        print(f"Lỗi khi tính toán class weights: {e}. Sẽ không sử dụng class_weight.")
        class_weights_dict = None
else:
    print(f"Không đủ số lớp ({len(unique_classes)}) hoặc không khớp với num_classes ({num_classes}) để tính class_weight. Sẽ không sử dụng class_weight.")
    class_weights_dict = None

# --- Tạo thư mục lưu mô hình và logs (Dành riêng cho Model 2B) ---
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
current_script_dir = os.path.dirname(os.path.abspath(__file__))
output_base_dir_name = f"model2B_EfficientNetB2_output_{timestamp}" # Thêm timestamp để mỗi lần chạy là duy nhất

models_dir = os.path.join(current_script_dir, output_base_dir_name, 'models',timestamp)
logs_dir = os.path.join(current_script_dir, output_base_dir_name, 'logs',timestamp)

os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
print(f"Thư mục lưu mô hình: {models_dir}")
print(f"Thư mục lưu logs: {logs_dir}")

# --- Xây dựng mô hình ---
print("\nĐang xây dựng kiến trúc mô hình với EfficientNetB2...")
base_model = EfficientNetB2(
    weights='imagenet',
    include_top=False,
    input_shape=input_shape
)

fine_tune_at = 100
print(f"Số lớp của EfficientNetB2: {len(base_model.layers)}")
print(f"Đóng băng {fine_tune_at} lớp đầu tiên của EfficientNetB2.")
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(1024, kernel_regularizer=regularizers.l2(0.0005), activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0005), activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(256, kernel_regularizer= regularizers.l2(0.0005), activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# --- Cấu hình Optimizer và Biên dịch ---
initial_learning_rate = 1e-4
optimizer = optimizers.Adam(learning_rate=initial_learning_rate)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)
model.summary()

# --- Cấu hình Callbacks ---
model_checkpoint_path = os.path.join(models_dir, 'model_2B_EfficientNetB2_best.keras') # Đổi tên file
csv_log_path = os.path.join(logs_dir, 'model_2B_EfficientNetB2_training_log.csv') # Đổi tên file

callbacks = [
    ModelCheckpoint(
        model_checkpoint_path,
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
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    ),
    CSVLogger(csv_log_path)
    # tf.keras.callbacks.TensorBoard(log_dir=logs_dir, histogram_freq=1) # Bật nếu cần
]

# --- Tạo effective_train_generator ---
effective_train_generator = create_multi_class_weighted_generator(train_generator, class_weights_dict)

# --- Bắt đầu huấn luyện ---
print("\nBắt đầu huấn luyện Model 2B (sử dụng generator với sample_weights)...")
history = model.fit(
    effective_train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=callbacks,
    # class_weight=class_weights_dict, # <<--- ĐÃ XÓA/COMMENT DÒNG NÀY
    steps_per_epoch=len(train_generator), # Hoặc train_generator.samples // batch_size
    verbose=1
)

# --- Lưu mô hình cuối cùng ---
final_model_save_path = os.path.join(models_dir, 'model_2B_EfficientNetB2_final.keras') # Đổi tên file
model.save(final_model_save_path)
print(f"Đã lưu mô hình cuối cùng thành công tại: {final_model_save_path}")

# --- Đánh giá mô hình ---
print("\nĐang đánh giá Model 2B trên tập Validation...")
val_metrics = model.evaluate(val_generator, verbose=1)
metrics_names = model.metrics_names # Lấy tên metrics từ model đã compile
print("\nKết quả đánh giá Model 2B:")
for name, value in zip(metrics_names, val_metrics):
    print(f"- {name}: {value:.4f}")

# --- In thông tin chi tiết huấn luyện ---
print("\nThông tin chi tiết về quá trình huấn luyện:")
# ... (Giữ nguyên phần in thông tin của bạn, đảm bảo các key trong history.history là đúng)
if 'loss' in history.history: print(f"Số epochs đã chạy: {len(history.history['loss'])}")
if 'loss' in history.history: print(f"Loss cuối cùng (Training): {history.history['loss'][-1]:.4f}")
if 'val_loss' in history.history: print(f"Loss cuối cùng (Validation): {history.history['val_loss'][-1]:.4f}")
if 'accuracy' in history.history: print(f"Accuracy cuối cùng (Training): {history.history['accuracy'][-1]:.4f}")
if 'val_accuracy' in history.history: print(f"Accuracy cuối cùng (Validation): {history.history['val_accuracy'][-1]:.4f}")
if 'top_2_accuracy' in history.history and 'val_top_2_accuracy' in history.history : # Kiểm tra sự tồn tại của key
    print(f"Top-2 Accuracy cuối cùng (Training): {history.history['top_2_accuracy'][-1]:.4f}")
    print(f"Top-2 Accuracy cuối cùng (Validation): {history.history['val_top_2_accuracy'][-1]:.4f}")


# --- Vẽ và lưu biểu đồ ---
print("\nĐang vẽ và lưu biểu đồ lịch sử huấn luyện cho Model 2B...")
plot_save_path = os.path.join(logs_dir, 'model_2B_EfficientNetB2_training_history.png') # Đổi tên file
plt.figure(figsize=(18, 15))

metrics_to_plot = {
    'Accuracy': 'accuracy',
    'Loss': 'loss',
    'Top-2 Accuracy': 'top_2_accuracy',
    'Precision': 'precision',
    'Recall': 'recall',
    'AUC': 'auc'
}
plot_index = 1
for display_name, metric_key in metrics_to_plot.items():
    if metric_key in history.history and f'val_{metric_key}' in history.history:
        plt.subplot(3, 2, plot_index)
        plt.plot(history.history[metric_key], label=f'Training {display_name}')
        plt.plot(history.history[f'val_{metric_key}'], label=f'Validation {display_name}')
        plt.title(f'Model 2B: {display_name}')
        plt.xlabel('Epoch')
        plt.ylabel(display_name)
        plt.legend()
        plt.grid(True)
        plot_index += 1
    elif metric_key in history.history : # Trường hợp chỉ có training metric (ít xảy ra với Keras fit)
         plt.subplot(3, 2, plot_index)
         plt.plot(history.history[metric_key], label=f'Training {display_name}')
         plt.title(f'Model 2B: {display_name} (Training only)')
         plt.xlabel('Epoch')
         plt.ylabel(display_name)
         plt.legend()
         plt.grid(True)
         plot_index += 1

if plot_index == 1: # Không có metric nào được vẽ
    print("Không có đủ dữ liệu metrics để vẽ biểu đồ.")
else:
    plt.tight_layout()
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    print(f"Đã lưu biểu đồ lịch sử huấn luyện tại: {plot_save_path}")
plt.close()

print("\nHoàn thành quá trình huấn luyện Model 2B.")