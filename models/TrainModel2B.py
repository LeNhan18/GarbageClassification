# train_model2B_improved.py
# Huấn luyện mô hình phân loại rác không tái chế

import os
import numpy as np
from keras.src.applications.efficientnet import EfficientNetB2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import matplotlib.pyplot as plt
import tensorflow as tf
# --- Cấu hình các tham số chính ---
data_dir = 'Z:\\GarbageClassification\\data\\non_recyclable' # Thư mục dữ liệu cho rác không tái chế
img_size = (224, 224)  # Kích thước ảnh phù hợp với EfficientNetB2
batch_size = 16        # Giảm batch size có thể cải thiện độ chính xác và phù hợp với VRAM hạn chế
epochs = 100           # Tăng số epochs tối đa. EarlyStopping sẽ dừng sớm nếu mô hình hội tụ
input_shape = (240, 240, 3) # Hình dạng đầu vào của ảnh (chiều cao, chiều rộng, kênh màu)

# --- Cấu hình Data Augmentation mạnh mẽ ---
# Tăng cường khả năng khái quát hóa của mô hình
train_datagen = ImageDataGenerator(
    rescale=1. / 255,           # Chuẩn hóa giá trị pixel về khoảng [0, 1]
    validation_split=0.2,       # Tỷ lệ dữ liệu dùng cho validation
    rotation_range=30,          # Xoay ảnh ngẫu nhiên trong khoảng 30 độ
    width_shift_range=0.2,      # Dịch chuyển chiều ngang ngẫu nhiên (tỷ lệ chiều rộng ảnh)
    height_shift_range=0.2,     # Dịch chuyển chiều dọc ngẫu nhiên (tỷ lệ chiều cao ảnh)
    shear_range=0.15,           # Biến đổi cắt (shear transformation)
    zoom_range=0.2,             # Phóng to/thu nhỏ ngẫu nhiên
    horizontal_flip=True,       # Lật ngang ảnh ngẫu nhiên
    vertical_flip=True,         # Lật dọc ảnh ngẫu nhiên (phù hợp nếu không có hướng cụ thể)
    fill_mode='nearest',        # Cách lấp đầy các pixel mới sau các phép biến đổi
    brightness_range=[0.7, 1.3] # Thay đổi độ sáng ngẫu nhiên
)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,           # Chỉ chuẩn hóa cho tập validation (không augment)
    validation_split=0.2
)

# Tạo generators từ thư mục dữ liệu
print("\nĐang tạo Data Generators...")
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',   # Phân loại đa lớp
    subset='training',
    shuffle=True                # Trộn dữ liệu huấn luyện
)

val_generator = val_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False               # Không trộn dữ liệu validation để kết quả nhất quán
)

num_classes = len(train_generator.class_indices)
print(f"Số lượng lớp phân loại tìm thấy: {num_classes}")
print(f"Tên các lớp: {train_generator.class_indices}")

# --- Tạo thư mục lưu mô hình và logs ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_script_dir, 'model')
logs_dir = os.path.join(current_script_dir, 'logs')
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
print(f"Thư mục lưu mô hình: {models_dir}")
print(f"Thư mục lưu logs: {logs_dir}")

# --- Xây dựng mô hình với EfficientNetB2 (Transfer Learning) ---
print("\nĐang xây dựng kiến trúc mô hình với EfficientNetB2...")
base_model = EfficientNetB2(
    weights='imagenet',     # Sử dụng trọng số tiền huấn luyện trên ImageNet
    include_top=False,      # Không bao gồm các lớp phân loại cuối cùng của ImageNet
    input_shape=input_shape
)

# Chiến lược Fine-tuning:
# Đóng băng các lớp ban đầu của mô hình cơ sở để giữ các đặc trưng chung
# sau đó mở khóa một phần các lớp cuối của base_model để fine-tune các đặc trưng chuyên biệt hơn.
# Giá trị fine_tune_at càng nhỏ thì càng nhiều lớp của base_model được huấn luyện.
# Việc này cần thử nghiệm để tìm giá trị tối ưu cho dataset của bạn.
# Ví dụ: có thể bắt đầu với fine_tune_at = len(base_model.layers) (đóng băng tất cả)
# rồi sau đó giảm dần để fine-tune.
fine_tune_at = 100 # Mở khóa và huấn luyện các lớp từ index 100 trở đi
print(f"Số lớp của EfficientNetB2: {len(base_model.layers)}")
print(f"Đóng băng {fine_tune_at} lớp đầu tiên của EfficientNetB2.")

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
# Kiểm tra để xác nhận các lớp đã được đóng băng/mở khóa đúng cách
# for i, layer in enumerate(base_model.layers):
#     print(f"Lớp {i}: {layer.name}, Trainable: {layer.trainable}")

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(), # Giảm chiều dữ liệu trong khi giữ lại thông tin không gian
    layers.BatchNormalization(),     # Giúp ổn định quá trình huấn luyện

    # First Dense Block
    # Tăng neurons và áp dụng L2 regularization để kiểm soát overfitting
    layers.Dense(1024, kernel_regularizer=regularizers.l2(0.0005), activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),             # Tỷ lệ Dropout giúp giảm overfitting

    # Second Dense Block - thêm một lớp hidden để tăng khả năng học các đặc trưng phức tạp
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0005), activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    # Output Layer: Số neurons bằng số lượng lớp phân loại, activation='softmax' cho phân loại đa lớp
    layers.Dense(num_classes, activation='softmax')
])

# --- Cấu hình trình tối ưu hóa và biên dịch mô hình ---
# Initial Learning Rate là một điểm khởi đầu tốt cho fine-tuning
initial_learning_rate = 1e-4

# Sử dụng Adam optimizer
optimizer = optimizers.Adam(
    learning_rate=initial_learning_rate,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)

# Biên dịch mô hình
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy', # Hàm mất mát cho phân loại đa lớp
    metrics=[
        'accuracy',                     # Độ chính xác
        tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy'), # Độ chính xác Top-2
        tf.keras.metrics.Precision(name='precision'),                         # Độ chính xác
        tf.keras.metrics.Recall(name='recall'),                               # Độ thu hồi
        tf.keras.metrics.AUC(name='auc')                                      # Diện tích dưới đường cong ROC
    ]
)
model.summary() # In tóm tắt cấu trúc mô hình

# --- Cấu hình Callbacks ---
# Các Callbacks giúp quản lý quá trình huấn luyện một cách hiệu quả
callbacks = [
    ModelCheckpoint(
        os.path.join(models_dir, 'model2B_best.keras'), # Lưu mô hình tốt nhất, đổi tên file
        monitor='val_accuracy', # Theo dõi độ chính xác trên tập validation
        save_best_only=True,    # Chỉ lưu khi có cải thiện tốt hơn
        mode='max',             # Theo dõi giá trị lớn nhất (accuracy)
        verbose=1               # Hiển thị thông báo khi lưu
    ),
    EarlyStopping(
        monitor='val_loss',     # Theo dõi hàm mất mát trên tập validation
        patience=15,            # Số epochs chờ đợi trước khi dừng nếu không có cải thiện
        restore_best_weights=True, # Khôi phục trọng số từ epoch tốt nhất
        verbose=1
    ),
    ReduceLROnPlateau(          # Giảm Learning Rate khi hiệu suất ngừng cải thiện
        monitor='val_loss',
        factor=0.5,             # Hệ số giảm LR (LR mới = LR cũ * factor)
        patience=7,             # Số epochs chờ đợi trước khi giảm LR
        min_lr=1e-7,            # Ngưỡng LR tối thiểu
        verbose=1
    ),
    CSVLogger(os.path.join(logs_dir, 'model2b_training.csv')), # Lưu lịch sử huấn luyện vào file CSV, đổi tên file
    # Thêm TensorBoard nếu bạn muốn theo dõi quá trình huấn luyện chi tiết hơn qua giao diện web
    # tf.keras.callbacks.TensorBoard(log_dir=logs_dir, histogram_freq=1)
]

# --- Bắt đầu huấn luyện mô hình ---
print("\nBắt đầu huấn luyện Model 2B (Phân loại rác không tái chế)...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1 # Hiển thị tiến trình huấn luyện
)

# --- Lưu mô hình cuối cùng ---
model.save(os.path.join(models_dir, 'model2B_final.keras')) # Đổi tên file lưu mô hình
print(" Đã lưu mô hình cuối cùng thành công!")

# --- Đánh giá mô hình trên tập validation ---
print("\nĐang đánh giá Model 2B trên tập Validation...")
val_metrics = model.evaluate(val_generator, verbose=1)
metrics_names = ['loss', 'accuracy', 'top_2_accuracy', 'precision', 'recall', 'auc']
print("\nKết quả đánh giá Model 2B:")
for name, value in zip(metrics_names, val_metrics):
    print(f"- {name}: {value:.4f}")

# In thêm thông tin chi tiết về quá trình huấn luyện
print("\nThông tin chi tiết về quá trình huấn luyện:")
print(f"Số epochs đã chạy: {len(history.history['loss'])}")
print(f"Loss cuối cùng (Training): {history.history['loss'][-1]:.4f}")
print(f"Loss cuối cùng (Validation): {history.history['val_loss'][-1]:.4f}")
print(f"Accuracy cuối cùng (Training): {history.history['accuracy'][-1]:.4f}")
print(f"Accuracy cuối cùng (Validation): {history.history['val_accuracy'][-1]:.4f}")
print(f"Top-2 Accuracy cuối cùng (Training): {history.history['top_2_accuracy'][-1]:.4f}")
print(f"Top-2 Accuracy cuối cùng (Validation): {history.history['val_top_2_accuracy'][-1]:.4f}")

# --- Vẽ và lưu biểu đồ lịch sử huấn luyện ---
print("\nĐang vẽ và lưu biểu đồ lịch sử huấn luyện...")
plt.figure(figsize=(18, 15)) # Điều chỉnh kích thước figure cho phù hợp

# Biểu đồ Accuracy
plt.subplot(3, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model 2B: Accuracy') # Đổi tên tiêu đề biểu đồ
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Biểu đồ Loss
plt.subplot(3, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model 2B: Loss') # Đổi tên tiêu đề biểu đồ
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Biểu đồ Top-2 Accuracy
plt.subplot(3, 2, 3)
plt.plot(history.history['top_2_accuracy'], label='Training Top-2 Accuracy')
plt.plot(history.history['val_top_2_accuracy'], label='Validation Top-2 Accuracy')
plt.title('Model 2B: Top-2 Accuracy') # Đổi tên tiêu đề biểu đồ
plt.xlabel('Epoch')
plt.ylabel('Top-2 Accuracy')
plt.legend()
plt.grid(True)

# Biểu đồ Precision
plt.subplot(3, 2, 4)
plt.plot(history.history['precision'], label='Training Precision')
plt.plot(history.history['val_precision'], label='Validation Precision')
plt.title('Model 2B: Precision') # Đổi tên tiêu đề biểu đồ
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)

# Biểu đồ Recall
plt.subplot(3, 2, 5)
plt.plot(history.history['recall'], label='Training Recall')
plt.plot(history.history['val_recall'], label='Validation Recall')
plt.title('Model 2B: Recall') # Đổi tên tiêu đề biểu đồ
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()
plt.grid(True)

# Biểu đồ AUC
plt.subplot(3, 2, 6)
plt.plot(history.history['auc'], label='Training AUC')
plt.plot(history.history['val_auc'], label='Validation AUC')
plt.title('Model 2B: AUC') # Đổi tên tiêu đề biểu đồ
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)

plt.tight_layout() # Tự động điều chỉnh khoảng cách giữa các subplot
plt.savefig(os.path.join(logs_dir, 'model2b_training_history.png'), dpi=300, bbox_inches='tight') # Đổi tên file ảnh
plt.close()

print("\nHoàn thành quá trình huấn luyện Model 2B.")