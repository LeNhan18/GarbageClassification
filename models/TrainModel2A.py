# train_model2A_improved.py
# Huấn luyện mô hình phân loại rác tái chế

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

# --- Cấu hình các tham số chính ---
data_dir = 'Z:\\GarbageClassification\\data\\recyclable'
img_size = (224, 224)  # Kích thước ảnh phù hợp với EfficientNetB2
batch_size = 16  # Giảm batch size có thể cải thiện độ chính xác và phù hợp với VRAM hạn chế
epochs = 100  # Tăng số epochs tối đa. EarlyStopping sẽ dừng sớm nếu mô hình hội tụ
input_shape = (224, 224, 3)  # Hình dạng đầu vào của ảnh (chiều cao, chiều rộng, kênh màu)

# --- Cấu hình Data Augmentation mạnh mẽ ---
# Tăng cường khả năng khái quát hóa của mô hình
train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # Chuẩn hóa giá trị pixel về khoảng [0, 1]
    validation_split=0.2,  # Tỷ lệ dữ liệu dùng cho validation
    rotation_range=30,  # Xoay ảnh ngẫu nhiên trong khoảng 30 độ
    width_shift_range=0.2,  # Dịch chuyển chiều ngang ngẫu nhiên (tỷ lệ chiều rộng ảnh)
    height_shift_range=0.2,  # Dịch chuyển chiều dọc ngẫu nhiên (tỷ lệ chiều cao ảnh)
    shear_range=0.15,  # Biến đổi cắt (shear transformation)
    zoom_range=0.2,  # Phóng to/thu nhỏ ngẫu nhiên
    horizontal_flip=True,  # Lật ngang ảnh ngẫu nhiên
    vertical_flip=True,  # Lật dọc ảnh ngẫu nhiên (phù hợp nếu không có hướng cụ thể)
    fill_mode='nearest',  # Cách lấp đầy các pixel mới sau các phép biến đổi
    brightness_range=[0.7, 1.3]  # Thay đổi độ sáng ngẫu nhiên
)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,  # Chỉ chuẩn hóa cho tập validation (không augment)
    validation_split=0.2
)

# Tạo generators từ thư mục dữ liệu
print("\nĐang tạo Data Generators...")
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',  # Phân loại đa lớp
    subset='training',
    shuffle=True  # Trộn dữ liệu huấn luyện
)

val_generator = val_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False  # Không trộn dữ liệu validation để kết quả nhất quán
)

num_classes = len(train_generator.class_indices)
print(f"Số lượng lớp phân loại tìm thấy: {num_classes}")
print(f"Tên các lớp: {train_generator.class_indices}")
print("")


# ------Tính toán class weight cân bằng dữ liệu
def create_multi_class_weighted_generator(generator, class_weights_dictionary, num_classes_local):
    """
    Tạo một generator bọc quanh generator gốc để yield thêm sample_weights.
    Phù hợp cho trường hợp nhãn là one-hot encoded (từ class_mode='categorical').
    """
    if class_weights_dictionary is None:
        print("Thông báo: class_weights_dictionary là None. Sẽ yield (x, y, trọng số mặc định là 1).")
        while True:
            x_batch, y_batch = next(generator)
            # Tạo trọng số mẫu mặc định là 1 cho mỗi mẫu trong batch
            # Kích thước của sample_weights phải là (batch_size,)
            sample_weights = np.ones(y_batch.shape[0], dtype=np.float32)
            yield x_batch, y_batch, sample_weights
    else:
        print("Thông báo: Sử dụng generator tùy chỉnh để yield (x, y, sample_weights).")
        while True:
            x_batch, y_batch = next(generator)  # y_batch ở đây là one-hot encoded, ví dụ: [0,1,0,0,0]

            # Chuyển đổi nhãn one-hot (y_batch) thành nhãn số nguyên (integer labels)
            # Ví dụ: [0,1,0,0,0] sẽ thành 1
            y_integer_labels = np.argmax(y_batch, axis=1)

            # Tạo sample_weights dựa trên nhãn số nguyên và class_weights_dictionary
            sample_weights = np.array(
                [class_weights_dictionary.get(label_idx, 1.0) for label_idx in y_integer_labels],
                dtype=np.float32
            )
            yield x_batch, y_batch, sample_weights


print("\nĐang tính toán Class Weights...")
train_labels_for_weights = train_generator.classes
unique_classes, counts = np.unique(train_labels_for_weights, return_counts=True)
class_weights_dict = None  # Khởi tạo

if len(unique_classes) >= 2 and len(unique_classes) == num_classes:
    print(f"Phân bố mẫu thực tế từ generator (chỉ số lớp: số lượng): {dict(zip(unique_classes, counts))}")
    try:
        computed_class_weights_array = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=train_labels_for_weights
        )
        class_weights_dict = dict(zip(unique_classes, computed_class_weights_array))
        print(f"Sẽ sử dụng Class Weights: {class_weights_dict}")
    except Exception as e:
        print(f"Lỗi khi tính toán class weights: {e}. Sẽ không sử dụng class_weight.")
        class_weights_dict = None
else:
    print(
        f"Không đủ số lớp ({len(unique_classes)}) hoặc không khớp với num_classes ({num_classes}) để tính class_weight. Sẽ không sử dụng class_weight.")
    class_weights_dict = None

# --- Tạo thư mục lưu mô hình và logs ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_script_dir, 'model_EfficientNetB2')
logs_dir = os.path.join(current_script_dir, 'logs_EfficientNetB2')
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
print(f"Thư mục lưu mô hình: {models_dir}")
print(f"Thư mục lưu logs: {logs_dir}")

# --- Xây dựng mô hình với EfficientNetB2 (Transfer Learning) ---
print("\nĐang xây dựng kiến trúc mô hình với EfficientNetB2...")
base_model = EfficientNetB2(
    weights='imagenet',  # Sử dụng trọng số tiền huấn luyện trên ImageNet
    include_top=False,  # Không bao gồm các lớp phân loại cuối cùng của ImageNet
    input_shape=input_shape
)

# Chiến lược Fine-tuning:
# Đóng băng các lớp ban đầu của mô hình cơ sở để giữ các đặc trưng chung
# sau đó mở khóa một phần các lớp cuối của base_model để fine-tune các đặc trưng chuyên biệt hơn.
# Giá trị fine_tune_at càng nhỏ thì càng nhiều lớp của base_model được huấn luyện.
# Việc này cần thử nghiệm để tìm giá trị tối ưu cho dataset của bạn.
# Ví dụ: có thể bắt đầu với fine_tune_at = len(base_model.layers) (đóng băng tất cả)
# rồi sau đó giảm dần để fine-tune.
fine_tune_at = 80  # Mở khóa và huấn luyện các lớp từ index 80 trở đi (giảm từ 100 để fine-tune nhiều hơn)
print(f"Số lớp của EfficientNetB2: {len(base_model.layers)}")
print(f"Đóng băng {fine_tune_at} lớp đầu tiên của EfficientNetB2.")

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
# Kiểm tra để xác nhận các lớp đã được đóng băng/mở khóa đúng cách
# for i, layer in enumerate(base_model.layers):
#     print(f"Lớp {i}: {layer.name}, Trainable: {layer.trainable}")

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),  # Giảm chiều dữ liệu trong khi giữ lại thông tin không gian
    layers.BatchNormalization(),  # Giúp ổn định quá trình huấn luyện

    # First Dense Block
    # Tăng neurons và áp dụng L2 regularization để kiểm soát overfitting
    layers.Dense(1024, kernel_regularizer=regularizers.l2(0.0005), activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),  # Tỷ lệ Dropout giúp giảm overfitting

    # Second Dense Block - thêm một lớp hidden để tăng khả năng học các đặc trưng phức tạp
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0005), activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

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
    loss='categorical_crossentropy',  # Hàm mất mát cho phân loại đa lớp
    metrics=[
        'accuracy',  # Độ chính xác
        tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy'),  # Độ chính xác Top-2
        tf.keras.metrics.Precision(name='precision'),  # Độ chính xác
        tf.keras.metrics.Recall(name='recall'),  # Độ thu hồi
        tf.keras.metrics.AUC(name='auc')  # Diện tích dưới đường cong ROC
    ]
)
model.summary()  # In tóm tắt cấu trúc mô hình

# --- Cấu hình Callbacks ---
# Các Callbacks giúp quản lý quá trình huấn luyện một cách hiệu quả
callbacks = [
    ModelCheckpoint(
        os.path.join(models_dir, 'model2A_EfficientNetB2.keras'),  # Lưu mô hình tốt nhất
        monitor='val_accuracy',  # Theo dõi độ chính xác trên tập validation
        save_best_only=True,  # Chỉ lưu khi có cải thiện tốt hơn
        mode='max',  # Theo dõi giá trị lớn nhất (accuracy)
        verbose=1  # Hiển thị thông báo khi lưu
    ),
    EarlyStopping(
        monitor='val_loss',  # Theo dõi hàm mất mát trên tập validation
        patience=15,  # Số epochs chờ đợi trước khi dừng nếu không có cải thiện
        restore_best_weights=True,  # Khôi phục trọng số từ epoch tốt nhất
        verbose=1
    ),
    ReduceLROnPlateau(  # Giảm Learning Rate khi hiệu suất ngừng cải thiện
        monitor='val_loss',
        factor=0.5,  # Hệ số giảm LR (LR mới = LR cũ * factor)
        patience=7,  # Số epochs chờ đợi trước khi giảm LR
        min_lr=1e-7,  # Ngưỡng LR tối thiểu
        verbose=1
    ),
    CSVLogger(os.path.join(logs_dir, 'model2AA_training.csv')),  # Lưu lịch sử huấn luyện vào file CSV
    # Thêm TensorBoard nếu bạn muốn theo dõi quá trình huấn luyện chi tiết hơn qua giao diện web
    # tf.keras.callbacks.TensorBoard(log_dir=logs_dir, histogram_freq=1)
]

# --- Tạo weighted generator nếu có class weights ---
if class_weights_dict is not None:
    print("Sử dụng weighted generator với class weights...")
    weighted_train_generator = create_multi_class_weighted_generator(
        train_generator, class_weights_dict, num_classes
    )
    train_data_for_fit = weighted_train_generator
else:
    print("Sử dụng generator thông thường...")
    train_data_for_fit = train_generator

# --- Bắt đầu huấn luyện mô hình ---
print(f"Sử dụng class_weights_dict: {class_weights_dict}")
print("Kiểm tra output của train_generator:")
sample_batch_images, sample_batch_labels = next(train_generator)
print(f"  Hình dạng batch ảnh: {sample_batch_images.shape}")
print(f"  Kiểu dữ liệu batch ảnh: {sample_batch_images.dtype}")
print(f"  Hình dạng batch nhãn: {sample_batch_labels.shape}")
print(f"  Kiểu dữ liệu batch nhãn: {sample_batch_labels.dtype}")
print("\nBắt đầu huấn luyện Model 2A (Phân loại rác tái chế)...")

history = model.fit(
    train_data_for_fit,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1  # Hiển thị tiến trình huấn luyện
)

# --- Lưu mô hình cuối cùng ---
model.save(os.path.join(models_dir, 'model2a_EfficientNetB2.keras'))
print("Đã lưu mô hình cuối cùng thành công!")

# --- Đánh giá mô hình trên tập validation ---
print("\nĐang đánh giá Model 2A trên tập Validation...")
val_metrics = model.evaluate(val_generator, verbose=1)
metrics_names = ['loss', 'accuracy', 'top_2_accuracy', 'precision', 'recall', 'auc']
print("\nKết quả đánh giá Model 2A:")
for name, value in zip(metrics_names, val_metrics):
    print(f"- {name}: {value:.4f}")

# --- Tạo predictions cho evaluation ---
print("\nĐang tạo predictions cho Confusion Matrix và các đánh giá...")
val_generator.reset()  # Reset generator về đầu
steps = val_generator.samples // val_generator.batch_size
if val_generator.samples % val_generator.batch_size != 0:
    steps += 1

y_pred_probs = model.predict(val_generator, steps=steps, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

# Lấy true labels
y_true = val_generator.classes[:len(y_pred)]  # Đảm bảo cùng độ dài

# Lấy tên classes theo đúng thứ tự
class_names_ordered = [k for k, v in sorted(val_generator.class_indices.items(), key=lambda item: item[1])]

print(f"Số lượng predictions: {len(y_pred)}")
print(f"Số lượng true labels: {len(y_true)}")
print(f"Classes: {class_names_ordered}")

# --- Tạo Confusion Matrix ---
print("\nĐang tạo Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=class_names_ordered,
            yticklabels=class_names_ordered)
plt.title('Confusion Matrix - Model 2A EfficientNetB2', fontsize=16)
plt.ylabel('Actual Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(logs_dir, 'confusion_matrix_EfficientNetB2.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Đã lưu Confusion Matrix.")

# --- Tạo Classification Report ---
print("\nĐang tạo Classification Report...")
report = classification_report(y_true, y_pred, target_names=class_names_ordered)
print(report)

# Lưu report
report_path = os.path.join(logs_dir, 'classification_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("Classification Report - Model 2A EfficientNetB2\n")
    f.write("=" * 50 + "\n\n")
    f.write(report)
print(f"Đã lưu Classification Report tại: {report_path}")

# --- Vẽ ROC curve cho multi-class ---
print("\nĐang vẽ ROC curve cho multi-class...")
try:
    # Binarize labels cho multi-class ROC
    y_true_bin = label_binarize(y_true, classes=range(num_classes))

    # Nếu chỉ có 2 classes, label_binarize trả về 1D array
    if num_classes == 2:
        y_true_bin = np.column_stack([1 - y_true_bin, y_true_bin])

    # Compute ROC curve và AUC cho từng class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(12, 10))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink'])

    for i, color in zip(range(num_classes), colors):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {class_names_ordered[i]} (AUC = {roc_auc[i]:.2f})')

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Multi-class ROC Curves - Model 2A EfficientNetB2', fontsize=16)
    plt.legend(loc="lower right", bbox_to_anchor=(1.3, 0))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    roc_curve_path = os.path.join(logs_dir, 'roc_curve_multiclass.png')
    plt.savefig(roc_curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu ROC curve tại: {roc_curve_path}")

except Exception as e:
    print(f"Lỗi khi vẽ ROC curve: {e}")

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
print("\nĐang vẽ và lúu biểu đồ lịch sử huấn luyện...")
plt.figure(figsize=(20, 16))  # Điều chỉnh kích thước figure cho phù hợp

# Biểu đồ Accuracy
plt.subplot(3, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model 2A: Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Biểu đồ Loss
plt.subplot(3, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model 2A: Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Biểu đồ Top-2 Accuracy
plt.subplot(3, 2, 3)
plt.plot(history.history['top_2_accuracy'], label='Training Top-2 Accuracy', linewidth=2)
plt.plot(history.history['val_top_2_accuracy'], label='Validation Top-2 Accuracy', linewidth=2)
plt.title('Model 2A: Top-2 Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Top-2 Accuracy', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Biểu đồ Precision
plt.subplot(3, 2, 4)
plt.plot(history.history['precision'], label='Training Precision', linewidth=2)
plt.plot(history.history['val_precision'], label='Validation Precision', linewidth=2)
plt.title('Model 2A: Precision', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Biểu đồ Recall
plt.subplot(3, 2, 5)
plt.plot(history.history['recall'], label='Training Recall', linewidth=2)
plt.plot(history.history['val_recall'], label='Validation Recall', linewidth=2)
plt.title('Model 2A: Recall', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Recall', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Biểu đồ AUC
plt.subplot(3, 2, 6)
plt.plot(history.history['auc'], label='Training AUC', linewidth=2)
plt.plot(history.history['val_auc'], label='Validation AUC', linewidth=2)
plt.title('Model 2A: AUC', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('AUC', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.suptitle('Model 2A EfficientNetB2 - Training History', fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Tự động điều chỉnh khoảng cách giữa các subplot
plt.savefig(os.path.join(logs_dir, 'model2a_EfficientNetB2_training_history.png'), dpi=300, bbox_inches='tight')
plt.close()

print("\nHoàn thành quá trình huấn luyện Model 2A.")
print(f"Tất cả kết quả đã được lưu trong thư mục: {logs_dir}")
print(f"Mô hình đã được lưu trong thư mục: {models_dir}")