# train_model_single_phase.py
# Huấn luyện mô hình phân loại nhị phân (rác tái chế vs không tái chế) trong một giai đoạn.

import os
import numpy as np
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import tensorflow as tf
from datetime import datetime
from sklearn.utils import class_weight

# --- Cấu hình GPU để tăng tốc độ ---
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Đã bật Mixed Precision Training (sử dụng float16/float32).")

        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Cấu hình thành công {len(gpus)} GPU")
    else:
        print("Không tìm thấy GPU, sử dụng CPU. Mixed Precision Training sẽ không được bật.")
except Exception as e:
    print(f"Lỗi cấu hình GPU hoặc Mixed Precision: {e}")

# --- Cấu hình các tham số chính ---
data_dir = 'Z:\\GarbageClassification\\datas'
img_size = (180, 180)  # Giảm kích thước ảnh từ 240x240 xuống 180x180
batch_size = 8  # Giảm batch size từ 16 xuống 8
input_shape = (180, 180, 3)  # Cập nhật input shape
epochs = 50  # Giảm số epochs tối đa từ 100 xuống 50
num_classes = 1  # Cho phân loại nhị phân (sử dụng sigmoid)

# --- Tạo thư mục lưu mô hình và logs ---
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_binary_classification_single_phase')
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs_binary_classification_single_phase',
                        timestamp)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
print(f"Thư mục lưu mô hình: {models_dir}")
print(f"Thư mục lưu logs: {logs_dir}")

# --- Data Augmentation Cải tiến ---
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2
)

# Tạo generators
print("\nĐang tạo Data Generators...")
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training',
    shuffle=True,
)

val_generator = val_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# --- Tính toán Class Weights để xử lý mất cân bằng lớp ---
print("\nĐang tính toán Class Weights...")
train_labels = train_generator.labels
if not (0 in train_labels and 1 in train_labels):
    print("Cảnh báo: Tập huấn luyện chỉ chứa một lớp. Không thể tính toán class weights. Đặt class_weights = None.")
    class_weights = None
else:
    classes = np.unique(train_labels)
    computed_class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=train_labels
    )
    class_weights = dict(zip(classes, computed_class_weights))
    print(f"Class Weights: {class_weights}")


# --- Hàm wrapper cho generator để thêm sample_weights ---
def create_weighted_generator(generator, class_weights_dict):
    if class_weights_dict is None:
        return generator

    while True:
        x_batch, y_batch = next(generator)
        sample_weights = np.array([class_weights_dict[label] for label in y_batch])
        yield x_batch, y_batch, sample_weights


# --- Custom F1 Score Metric ---
@tf.keras.utils.register_keras_serializable(package="Custom", name="F1ScoreWithReshape")
class F1ScoreWithReshape(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.f1_metric = tf.keras.metrics.F1Score(threshold=threshold, average='weighted')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_reshaped = tf.expand_dims(tf.cast(y_true, self.dtype), axis=-1)
        self.f1_metric.update_state(y_true_reshaped, y_pred, sample_weight)

    def result(self):
        return self.f1_metric.result()

    def reset_state(self):
        self.f1_metric.reset_state()

    def get_config(self):
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config


# --- Xây dựng mô hình (Single Phase) ---
print("\nĐang xây dựng kiến trúc mô hình (Một giai đoạn)...")
base_model = EfficientNetB2(
    weights='imagenet',
    include_top=False,
    input_shape=input_shape
)

# Fine-tune các lớp của base_model ngay từ đầu
# unfreeze_from_layer là chỉ số của lớp đầu tiên được mở khóa.
# Để fine-tune một phần, chúng ta sẽ mở khóa các lớp cuối cùng.
# Ví dụ: 250 sẽ mở khóa khoảng 80 lớp cuối của EfficientNetB2 (có khoảng 330 lớp).
unfreeze_from_layer = 250
total_base_layers = len(base_model.layers)
num_unfrozen_layers = total_base_layers - unfreeze_from_layer
print(f"Số lớp của EfficientNetB2: {total_base_layers}")
print(f"Mở khóa {num_unfrozen_layers} lớp cuối cùng (từ index {unfreeze_from_layer}) của EfficientNetB2 để fine-tune.")

for layer in base_model.layers[:unfreeze_from_layer]:
    layer.trainable = False
for layer in base_model.layers[unfreeze_from_layer:]:
    layer.trainable = True

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),

    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.4),

    layers.Dense(256, kernel_regularizer=regularizers.l2(0.0001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),

    layers.Dense(128, kernel_regularizer=regularizers.l2(0.0001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),

    layers.Dense(num_classes, activation='sigmoid', dtype='float32')
])

# --- Biên dịch mô hình ---
initial_learning_rate = 5e-5  # Learning rate nhỏ phù hợp cho fine-tuning ngay từ đầu
optimizer = optimizers.Adam(learning_rate=initial_learning_rate)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        F1ScoreWithReshape(threshold=0.5)
    ]
)
model.summary()

# --- Callbacks ---
checkpoint_path = os.path.join(models_dir, 'model_best.keras')
callbacks = [
    ModelCheckpoint(
        checkpoint_path,
        monitor='val_f1_score',  # Theo dõi F1-score để lưu mô hình tốt nhất
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_f1_score',
        patience=10,  # Giảm patience từ 20 xuống 10
        restore_best_weights=True,
        mode='max',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_f1_score',
        factor=0.2,
        patience=5,  # Giảm patience từ 10 xuống 5
        min_lr=1e-7,
        verbose=1
    ),
    CSVLogger(os.path.join(logs_dir, 'training_log.csv'))
]

print("\nBắt đầu huấn luyện mô hình (Một giai đoạn)...")
weighted_train_generator = create_weighted_generator(train_generator, class_weights)

history = model.fit(
    weighted_train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)

# --- Lưu mô hình và đánh giá ---
final_model_path = os.path.join(models_dir, 'model_final.keras')
best_model_loaded = models.load_model(checkpoint_path, custom_objects={'F1ScoreWithReshape': F1ScoreWithReshape})
best_model_loaded.save(final_model_path)
print(f"\nĐã lưu mô hình cuối cùng (là mô hình tốt nhất) tại: {final_model_path}")

print(f"\nĐang tải mô hình tốt nhất để đánh giá: {checkpoint_path}")
best_model = models.load_model(checkpoint_path, custom_objects={'F1ScoreWithReshape': F1ScoreWithReshape})

print("\nĐang đánh giá mô hình tốt nhất trên tập Validation...")
val_metrics = best_model.evaluate(val_generator, verbose=1)
metrics_names = ['loss', 'accuracy', 'auc', 'precision', 'recall', 'f1_score']
print("\nKết quả đánh giá (Mô hình tốt nhất):")
for name, value in zip(metrics_names, val_metrics):
    print(f"- {name}: {value:.4f}")

# --- Tạo dự đoán để đánh giá chi tiết ---
y_pred_probs = []
y_true = []

val_generator.reset()
num_val_batches = int(np.ceil(val_generator.samples / val_generator.batch_size))

for _ in range(num_val_batches):
    x, y = next(val_generator)
    pred = best_model.predict(x, verbose=0)
    y_pred_probs.extend(pred.flatten())
    y_true.extend(y.flatten())

y_true = np.array(y_true[:val_generator.samples])
y_pred_probs = np.array(y_pred_probs[:val_generator.samples])
y_pred = (y_pred_probs > 0.5).astype(int)

# --- Tạo Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=list(train_generator.class_indices.keys()),
            yticklabels=list(train_generator.class_indices.keys()))
plt.title('Confusion Matrix', fontsize=16)
plt.ylabel('Actual Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)
plt.savefig(os.path.join(logs_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# --- Tạo Classification Report ---
class_names = list(train_generator.class_indices.keys())
report = classification_report(y_true, y_pred, target_names=class_names)
print("\nClassification Report:")
print(report)

# Lưu report
with open(os.path.join(logs_dir, 'classification_report.txt'), 'w') as f:
    f.write(report)

# --- Vẽ ROC curve ---
fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic (ROC)', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(logs_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
plt.close()

# --- Vẽ đồ thị lịch sử huấn luyện ---
plt.figure(figsize=(20, 15))

# Accuracy plot
plt.subplot(3, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# Loss plot
plt.subplot(3, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# AUC plot
plt.subplot(3, 2, 3)
plt.plot(history.history['auc'], label='Training AUC', linewidth=2)
plt.plot(history.history['val_auc'], label='Validation AUC', linewidth=2)
plt.title('Model AUC', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('AUC', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# Precision plot
plt.subplot(3, 2, 4)
plt.plot(history.history['precision'], label='Training Precision', linewidth=2)
plt.plot(history.history['val_precision'], label='Validation Precision', linewidth=2)
plt.title('Model Precision', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# Recall plot
plt.subplot(3, 2, 5)
plt.plot(history.history['recall'], label='Training Recall', linewidth=2)
plt.plot(history.history['val_recall'], label='Validation Recall', linewidth=2)
plt.title('Model Recall', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Recall', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# F1 Score plot
plt.subplot(3, 2, 6)
plt.plot(history.history['f1_score'], label='Training F1 Score', linewidth=2)
plt.plot(history.history['val_f1_score'], label='Validation F1 Score', linewidth=2)
plt.title('Model F1 Score', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('F1 Score', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(logs_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
plt.close()

# --- Tạo báo cáo tổng hợp ---
with open(os.path.join(logs_dir, 'summary_results.txt'), 'w') as f:
    f.write("=== KẾT QUẢ HUẤN LUYỆN MÔ HÌNH PHÂN LOẠI RÁC TÁI CHẾ (MỘT GIAI ĐOẠN) ===\n\n")
    f.write(f"Thời gian: {timestamp}\n")
    f.write(f"Mô hình: EfficientNetB2\n")
    f.write(f"Tổng số epochs huấn luyện: {len(history.history['loss'])}\n\n")

    f.write("--- Kết quả đánh giá cuối cùng (Mô hình tốt nhất) ---\n")
    f.write(f"Loss: {val_metrics[metrics_names.index('loss')]:.4f}\n")
    f.write(f"Accuracy: {val_metrics[metrics_names.index('accuracy')]:.4f}\n")
    f.write(f"AUC: {val_metrics[metrics_names.index('auc')]:.4f}\n")
    f.write(f"Precision: {val_metrics[metrics_names.index('precision')]:.4f}\n")
    f.write(f"Recall: {val_metrics[metrics_names.index('recall')]:.4f}\n")
    f.write(f"F1 Score: {val_metrics[metrics_names.index('f1_score')]:.4f}\n\n")

    f.write("=== Classification Report ===\n")
    f.write(report)
    f.write("\n\n=== Confusion Matrix ===\n")
    f.write(str(cm))
    f.write("\n\n=== Thông tin chi tiết Huấn luyện ===\n")
    f.write(f"Thư mục dữ liệu: {data_dir}\n")
    f.write(f"Kích thước ảnh: {img_size}\n")
    f.write(f"Batch size: {batch_size}\n")
    f.write(f"Số lượng dữ liệu training: {train_generator.samples}\n")
    f.write(f"Số lượng dữ liệu validation: {val_generator.samples}\n")
    f.write(f"Class Weights: {class_weights}\n\n")

    f.write("--- Tham số Huấn luyện Một Giai đoạn ---\n")
    f.write(f"Initial Learning Rate: {initial_learning_rate}\n")
    f.write(f"Epochs Tối đa: {epochs}\n")
    f.write(f"Số lớp Base Model được mở khóa Fine-tuning: {num_unfrozen_layers} (từ index {unfreeze_from_layer})\n")
    f.write(f"EarlyStopping Patience: {callbacks[1].patience} (monitor='{callbacks[1].monitor}')\n")
    f.write(
        f"ReduceLROnPlateau Patience: {callbacks[2].patience} (monitor='{callbacks[2].monitor}', factor={callbacks[2].factor})\n\n")

    f.write("--- Cấu hình Data Augmentation ---\n")
    aug_params = train_datagen.__dict__
    for k, v in aug_params.items():
        if not k.startswith('_') and k not in ['flow_from_directory', 'random_transform', 'get_random_transform',
                                               'apply_transform', 'fit', 'idg']:
            f.write(f"  {k}: {v}\n")

print("Hoàn thành quá trình huấn luyện và đánh giá mô hình!")