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
        tf.config.experimental.set_memory_growth(gpus[0],True)
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096))
        print("Cấu hình thành công GPU")
    else:
        print("Không tìm thấy GPU")
except Exception as e:
    print(f"Lỗi cấu hình GPU: {e}")

# --- Cấu hình ---
data_dir = 'Z:\\GarbageClassification\\data'
img_size = (240, 240)  # Tăng kích thước ảnh một chút
batch_size = 16  # Giảm batch size để cải thiện độ chính xác
epochs = 10  # Tăng số epochs hơn nữa (Adjust this based on your training time)
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

# --- Define Custom F1 Score Metric to handle shape mismatch ---
import tensorflow as tf

# Định nghĩa lại lớp F1ScoreWithReshape và đăng ký nó với Keras
from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable(package="Custom", name="F1ScoreWithReshape")
class F1ScoreWithReshape(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        # Use the standard F1Score metric internally
        self.f1_metric = tf.keras.metrics.F1Score(threshold=threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Reshape y_true to match y_pred's last dimension (add a dimension)
        # y_true comes in as (batch_size,) and y_pred as (batch_size, 1)
        # We need y_true to be (batch_size, 1) for the metric
        y_true_reshaped = tf.expand_dims(tf.cast(y_true, self.dtype), axis=-1)
        self.f1_metric.update_state(y_true_reshaped, y_pred, sample_weight)

    def result(self):
        return self.f1_metric.result()

    def reset_state(self):
        self.f1_metric.reset_state()

    # Phương thức get_config được yêu cầu để đảm bảo serialization hoạt động đúng
    def get_config(self):
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config
# Hoặc sử dụng trong các hàm custom_objects nếu cần
# model = load_model('model_path.keras', custom_objects={'F1ScoreWithReshape': F1ScoreWithReshape})
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
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(),
        'Precision',
        'Recall',
        F1ScoreWithReshape(threshold=0.5) # Use the custom F1 Score metric
    ]
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
# Note: When evaluating or predicting manually, you'll still need to be mindful
# of the shapes if using metrics that expect 2D input.
# The custom metric handles this during model.fit/evaluate.
val_metrics = model.evaluate(val_generator, verbose=1)
# Update the metric names list to include the custom metric name
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
    # The model outputs (batch_size, 1), predictions are probabilities
    # Convert predictions to binary (0 or 1) based on threshold
    pred_binary = (pred > 0.5).astype(int)
    y_pred.extend(pred_binary.flatten()) # Flatten to match the expected format for sklearn metrics
    y_true.extend(y.flatten()) # Flatten the true labels as well if they are (batch_size,)

# Check if y_true from generator is (batch_size,) or (batch_size, 1)
# Based on class_mode='binary', it should be (batch_size,), so flattening is correct here.
# If it were (batch_size, 1), flatten() is still appropriate for sklearn.

    if len(y_true) >= val_generator.samples:
        break

# Ensure lengths match if generator does not yield exactly val_generator.samples
y_true = y_true[:val_generator.samples]
y_pred = y_pred[:val_generator.samples]


# Vẽ confusion matrix với cải tiến
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Confusion Matrix - Model 1', fontsize=16)
plt.ylabel('Actual Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)
class_names = list(train_generator.class_indices.keys())
# Adjust tick marks to be centered
tick_marks = np.arange(len(class_names)) + 0.5
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
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
plt.close() # Close the figure after saving

# Thêm biểu đồ mới cho Precision, Recall và F1 Score
plt.figure(figsize=(20, 6)) # Adjusted figure size

plt.subplot(1, 3, 1) # Adjusted subplot layout
plt.plot(history.history['precision'], label='Training', linewidth=2)
plt.plot(history.history['val_precision'], label='Validation', linewidth=2)
plt.title('Model Precision', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

plt.subplot(1, 3, 2) # Adjusted subplot layout
plt.plot(history.history['recall'], label='Training', linewidth=2)
plt.plot(history.history['val_recall'], label='Validation', linewidth=2)
plt.title('Model Recall', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Recall', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

plt.subplot(1, 3, 3) # Added F1 Score plot
plt.plot(history.history['f1_score'], label='Training', linewidth=2) # Use the name of the custom metric
plt.plot(history.history['val_f1_score'], label='Validation', linewidth=2) # Use the name of the custom metric
plt.title('Model F1 Score', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('F1 Score', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)


plt.tight_layout()
plt.savefig(os.path.join(logs_dir, 'model1_precision_recall_f1score.png'), dpi=300, bbox_inches='tight')
plt.close()

print("✅ Hoàn thành quá trình huấn luyện và đánh giá Model 1.")