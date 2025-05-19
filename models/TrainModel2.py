# train_model_single_phase.py
# Huấn luyện mô hình phân loại nhị phân (rác tái chế vs không tái chế) trong một giai đoạn.

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from datetime import datetime
from sklearn.utils import class_weight

# --- Cấu hình GPU để tăng tốc độ ---
try:
    # Kiểm tra số lượng GPU vật lý
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Tìm thấy {len(gpus)} GPU vật lý.")
        # Bật Mixed Precision Training (sử dụng float16 cho phần lớn tính toán)
        # Điều này giúp tăng tốc và giảm bộ nhớ cần thiết trên GPU có hỗ trợ Tensor Cores (Pascal+ / Volta+ / Turing+ / Ampere+)
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Đã bật Mixed Precision Training (sử dụng float16/float32).")

        # Cấu hình bộ nhớ tăng trưởng (Growth) để tránh lỗi "out of memory" ban đầu
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Cấu hình thành công Memory Growth cho GPU.")
    else:
        print("Không tìm thấy GPU vật lý. Sử dụng CPU. Mixed Precision Training sẽ không được bật.")
except Exception as e:
    print(f"Lỗi cấu hình GPU hoặc Mixed Precision: {e}")

# --- Cấu hình các tham số chính ---
# data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datas') # <-- Dòng này đã gây lỗi FileNotFoundError
data_dir = 'Z:\\GarbageClassification\\datas' # <-- Sửa lại đường dẫn chính xác của bạn

# Cập nhật kích thước ảnh đầu vào theo yêu cầu
img_size = (240, 240)
batch_size = 8 # Giữ batch size nhỏ như bạn đang dùng, cân nhắc tăng nếu có đủ bộ nhớ GPU
input_shape = (img_size[0], img_size[1], 3) # Cập nhật input shape
epochs = 50
num_classes = 1 # Cho phân loại nhị phân (sử dụng sigmoid)

# --- Tạo thư mục lưu mô hình và logs ---
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
base_script_dir = os.path.dirname(os.path.abspath(__file__))
# Có thể bạn muốn lưu models và logs cùng cấp với thư mục 'datas', không phải bên trong thư mục 'models'
# Nếu bạn muốn thư mục models và logs nằm ở Z:\GarbageClassification\
root_output_dir = 'Z:\\GarbageClassification' # Hoặc os.path.dirname(data_dir) nếu data_dir là Z:\GarbageClassification\datas
models_dir = os.path.join(root_output_dir, 'models', 'model_binary_classification_single_phase')
logs_dir = os.path.join(root_output_dir, 'logs', 'logs_binary_classification_single_phase', timestamp)

# Nếu bạn vẫn muốn models và logs nằm trong Z:\GarbageClassification\models
# models_dir = os.path.join(base_script_dir, 'model_binary_classification_single_phase')
# logs_dir = os.path.join(base_script_dir, 'logs_binary_classification_single_phase', timestamp)


os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
print(f"Thư mục lưu mô hình: {models_dir}")
print(f"Thư mục lưu logs: {logs_dir}")

# --- Data Augmentation Cải tiến ---
# Giữ nguyên các cài đặt augmentation mạnh mẽ để tăng cường tính tổng quát
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2,
    rotation_range=25, # Tăng nhẹ rotation_range
    width_shift_range=0.15, # Tăng nhẹ shift range
    height_shift_range=0.15,
    shear_range=0.15, # Tăng nhẹ shear range
    zoom_range=0.15, # Tăng nhẹ zoom range
    horizontal_flip=True,
    vertical_flip=False, # Cẩn thận với vertical_flip nếu hướng ảnh quan trọng
    brightness_range=[0.6, 1.4], # Mở rộng phạm vi brightness
    channel_shift_range=30.0, # Thêm channel shift
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2
)

# Tạo generators
print("\nĐang tạo Data Generators...")
try:
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size, # Sử dụng kích thước ảnh mới
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True,
    )

    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size, # Sử dụng kích thước ảnh mới
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )
    print(f"Tìm thấy {train_generator.samples} ảnh cho training.")
    print(f"Tìm thấy {val_generator.samples} ảnh cho validation.")
    print(f"Các lớp tìm thấy: {train_generator.class_indices}")

except Exception as e:
    print(f"Lỗi khi tạo data generators: {e}")
    print(f"Kiểm tra lại đường dẫn thư mục dữ liệu: {data_dir}")
    exit() # Thoát chương trình nếu không thể load dữ liệu

# --- Tính toán Class Weights để xử lý mất cân bằng lớp ---
print("\nĐang tính toán Class Weights...")
train_labels = train_generator.labels
unique_classes, counts = np.unique(train_labels, return_counts=True)

if len(unique_classes) < 2:
    print("Cảnh báo: Tập huấn luyện chỉ chứa một lớp. Không thể tính toán class weights. Đặt class_weights = None.")
    class_weights = None
elif len(unique_classes) == 2:
    print(f"Số lượng mẫu cho mỗi lớp: {dict(zip(unique_classes, counts))}")
    try:
        computed_class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=train_labels
        )
        class_weights = dict(zip(unique_classes, computed_class_weights))
        print(f"Class Weights: {class_weights}")
    except Exception as e:
        print(f"Lỗi khi tính toán class weights: {e}")
        class_weights = None
else:
     print("Cảnh báo: Tìm thấy nhiều hơn 2 lớp trong tập huấn luyện. Class weights không được tính toán cho phân loại nhị phân.")
     class_weights = None


# --- Hàm wrapper cho generator để thêm sample_weights ---
def create_weighted_generator(generator, class_weights_dict):
    if class_weights_dict is None:
        return generator

    print("Sử dụng weighted generator.")
    while True:
        # Lấy một batch từ generator gốc
        x_batch, y_batch = next(generator)
        # Tính sample weights dựa trên nhãn của batch đó
        sample_weights = np.array([class_weights_dict.get(label, 1.0) for label in y_batch]) # Dùng .get(label, 1.0) để an toàn
        yield x_batch, y_batch, sample_weights

# --- Xây dựng mô hình (Single Phase) ---
print("\nĐang xây dựng kiến trúc mô hình (Một giai đoạn)...")
base_model = EfficientNetB2(
    weights='imagenet',
    include_top=False, # Không bao gồm lớp phân loại cuối cùng của EfficientNetB2
    input_shape=input_shape # Sử dụng input shape mới
)

# Fine-tune các lớp của base_model ngay từ đầu
# unfreeze_from_layer là chỉ số của lớp đầu tiên được mở khóa.
# Để fine-tune một phần, chúng ta sẽ mở khóa các lớp cuối cùng.
# Giá trị 250 là hợp lý để mở khóa các khối cuối cùng.
unfreeze_from_layer = 250
total_base_layers = len(base_model.layers)
num_unfrozen_layers = total_base_layers - unfreeze_from_layer
print(f"Số lớp của EfficientNetB2: {total_base_layers}")
print(f"Mở khóa {num_unfrozen_layers} lớp cuối cùng (từ index {unfreeze_from_layer}) của EfficientNetB2 để fine-tune.")

for layer in base_model.layers[:unfreeze_from_layer]:
    layer.trainable = False
for layer in base_model.layers[unfreeze_from_layer:]:
    layer.trainable = True

# Xây dựng phần head của mô hình theo yêu cầu 1028 -> 512 -> 256 -> 1
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(), # Giảm kích thước spatial
    layers.BatchNormalization(), # Chuẩn hóa sau Pooling

    # Lớp Dense thứ nhất (1028 neurons)
    layers.Dense(1028, kernel_regularizer=regularizers.l2(0.0001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.4), # Dropout để giảm overfitting

    # Lớp Dense thứ hai (512 neurons)
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3), # Giảm Dropout

    # Lớp Dense thứ ba (256 neurons)
    layers.Dense(256, kernel_regularizer=regularizers.l2(0.0001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2), # Giảm Dropout

    # Lớp output nhị phân
    # Đảm bảo dtype là float32 khi dùng mixed precision cho output layer
    layers.Dense(num_classes, activation='sigmoid', dtype='float32')
])

# --- Biên dịch mô hình ---
# Learning rate nhỏ phù hợp cho fine-tuning ngay từ đầu
initial_learning_rate = 5e-5
optimizer = optimizers.Adam(learning_rate=initial_learning_rate)

# Khi sử dụng mixed_precision, optimizer cần được wrapped để xử lý gradient đúng cách
# Tuy nhiên, với API mới (TensorFlow 2.4+), optimizer thường tự động xử lý mixed_precision
# Nếu gặp vấn đề, có thể cần wrap thủ công: optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
# Hiện tại, giữ nguyên optimizer Adam là đủ.

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy', # Loss function cho phân loại nhị phân
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'), # Area Under the ROC Curve
        tf.keras.metrics.Precision(name='precision'), # Precision
        tf.keras.metrics.Recall(name='recall') # Recall
        # F1 Score đã được loại bỏ theo yêu cầu
    ]
)
model.summary()

# --- Callbacks ---
# Thay đổi từ val_f1_score sang val_auc để theo dõi
checkpoint_path = os.path.join(models_dir, 'model_best.keras')
callbacks = [
    # Lưu mô hình tốt nhất dựa trên val_auc thay vì val_f1_score
    ModelCheckpoint(
        checkpoint_path,
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    # Dừng sớm nếu val_auc không cải thiện sau một số epoch
    EarlyStopping(
        monitor='val_auc',
        patience=10, # Số epoch chờ
        restore_best_weights=True, # Khôi phục trọng số tốt nhất
        mode='max',
        verbose=1
    ),
    # Giảm learning rate nếu val_auc không cải thiện
    ReduceLROnPlateau(
        monitor='val_auc',
        factor=0.2, # Giảm learning rate xuống 20%
        patience=5, # Số epoch chờ
        min_lr=1e-7, # Giới hạn learning rate nhỏ nhất
        verbose=1
    ),
    # Ghi log quá trình huấn luyện ra file CSV
    CSVLogger(os.path.join(logs_dir, 'training_log.csv'))
]

print("\nBắt đầu huấn luyện mô hình (Một giai đoạn)...")
# Sử dụng weighted generator nếu class_weights được tính
effective_train_generator = create_weighted_generator(train_generator, class_weights)

history = model.fit(
    effective_train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)
# --- Lưu mô hình và đánh giá ---
# Tải mô hình tốt nhất đã được lưu bởi ModelCheckpoint
try:
    print(f"\nĐang tải mô hình tốt nhất từ: {checkpoint_path}")
    # Không cần cung cấp custom_objects vì F1ScoreWithReshape đã bị loại bỏ
    best_model_loaded = models.load_model(checkpoint_path)
    final_model_path = os.path.join(models_dir, 'model_final.keras')
    # Lưu lại mô hình tốt nhất với tên final để dễ quản lý
    best_model_loaded.save(final_model_path)
    print(f"Đã lưu mô hình tốt nhất (final) tại: {final_model_path}")

except Exception as e:
    print(f"Lỗi khi tải hoặc lưu mô hình tốt nhất: {e}")
    # Nếu có lỗi, thử tải và lưu mô hình cuối cùng của quá trình fit (có thể không phải tốt nhất)
    print("Thử lưu mô hình cuối cùng của quá trình huấn luyện...")
    try:
         final_model_path_fallback = os.path.join(models_dir, 'model_final_fallback.keras')
         model.save(final_model_path_fallback)
         best_model_loaded = model # Sử dụng mô hình cuối cùng nếu tải mô hình tốt nhất lỗi
         print(f"Đã lưu mô hình cuối cùng của quá trình huấn luyện tại: {final_model_path_fallback}")
    except Exception as e_fallback:
         print(f"Lỗi khi lưu mô hình cuối cùng: {e_fallback}")
         best_model_loaded = None # Không có mô hình để đánh giá chi tiết


if best_model_loaded:
    print("\nĐang đánh giá mô hình tốt nhất trên tập Validation...")
    # Đảm bảo sử dụng generator không shuffle cho đánh giá cuối cùng
    val_generator.reset()
    val_metrics = best_model_loaded.evaluate(val_generator, verbose=1)
    metrics_names = ['loss', 'accuracy', 'auc', 'precision', 'recall'] # Cập nhật danh sách metrics, bỏ f1_score

    print("\nKết quả đánh giá (Mô hình tốt nhất):")
    for name, value in zip(metrics_names, val_metrics):
        print(f"- {name}: {value:.4f}")

    # --- Tạo dự đoán để đánh giá chi tiết ---
    print("\nĐang tạo dự đoán trên tập Validation để phân tích chi tiết...")
    y_pred_probs = []
    y_true = []

    val_generator.reset() # Reset generator trước khi tạo dự đoán
    num_val_batches = int(np.ceil(val_generator.samples / val_generator.batch_size))

    for i in range(num_val_batches):
        x, y = next(val_generator)
        # Sử dụng predict_on_batch hoặc predict trên từng batch nhỏ để quản lý bộ nhớ tốt hơn
        pred = best_model_loaded.predict_on_batch(x)
        y_pred_probs.extend(pred.flatten())
        y_true.extend(y.flatten())
        # print(f"Đã xử lý batch {i+1}/{num_val_batches}") # In tiến trình

    # Cắt bớt nếu số lượng dự đoán vượt quá số lượng mẫu (do làm tròn trong num_val_batches)
    y_true = np.array(y_true[:val_generator.samples])
    y_pred_probs = np.array(y_pred_probs[:val_generator.samples])
    y_pred = (y_pred_probs > 0.5).astype(int) # Chuyển xác suất sang nhãn dự đoán với ngưỡng 0.5

    # Lấy tên lớp từ generator để hiển thị trong báo cáo/biểu đồ
    try:
        class_names_list = list(train_generator.class_indices.keys())
        # Đảm bảo thứ tự tên lớp khớp với nhãn 0 và 1
        # Giả sử 0 là lớp thứ nhất, 1 là lớp thứ hai trong dictionary của class_indices
        # Nếu không chắc, cần kiểm tra train_generator.class_indices
        # Ví dụ: {'Non_Recyclable': 0, 'Recyclable': 1}
        # class_names_ordered = [name for name, index in sorted(train_generator.class_indices.items())] # Cách an toàn hơn nếu cần

        # Cách đơn giản nếu biết 0 và 1 tương ứng với lớp nào
        class_names_ordered = ['Class_0', 'Class_1'] # Đặt tên mặc định
        if 'Non_Recyclable' in train_generator.class_indices and 'Recyclable' in train_generator.class_indices:
             class_names_ordered[train_generator.class_indices['Non_Recyclable']] = 'Non_Recyclable'
             class_names_ordered[train_generator.class_indices['Recyclable']] = 'Recyclable'
        print(f"Tên lớp (theo thứ tự 0, 1) cho báo cáo: {class_names_ordered}")

    except Exception as e:
         print(f"Lỗi khi lấy tên lớp từ generator: {e}. Sử dụng tên mặc định ['0', '1'].")
         class_names_ordered = ['0', '1']


    # --- Tạo Confusion Matrix ---
    print("\nĐang tạo Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=class_names_ordered,
                yticklabels=class_names_ordered)
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('Actual Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.savefig(os.path.join(logs_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Đã lưu Confusion Matrix.")

    # --- Tạo Classification Report ---
    print("\nĐang tạo Classification Report...")
    report = classification_report(y_true, y_pred, target_names=class_names_ordered)
    print(report)

    # Lưu report
    report_path = os.path.join(logs_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Đã lưu Classification Report tại: {report_path}")

    # --- Vẽ ROC curve ---
    print("\nĐang vẽ ROC curve...")
    try:
        # Tính toán lại AUC từ y_true và y_pred_probs để đảm bảo kết quả chính xác
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
        roc_curve_path = os.path.join(logs_dir, 'roc_curve.png')
        plt.savefig(roc_curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Đã lưu ROC curve tại: {roc_curve_path}")
    except Exception as e:
        print(f"Lỗi khi vẽ ROC curve: {e}")


    # --- Vẽ đồ thị lịch sử huấn luyện ---
    print("\nĐang vẽ đồ thị lịch sử huấn luyện...")
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
    if 'auc' in history.history and 'val_auc' in history.history:
        plt.subplot(3, 2, 3)
        plt.plot(history.history['auc'], label='Training AUC', linewidth=2)
        plt.plot(history.history['val_auc'], label='Validation AUC', linewidth=2)
        plt.title('Model AUC', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('AUC', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
    else:
         print("Không tìm thấy dữ liệu AUC trong lịch sử huấn luyện để vẽ đồ thị.")

    # Precision plot
    if 'precision' in history.history and 'val_precision' in history.history:
        plt.subplot(3, 2, 4)
        plt.plot(history.history['precision'], label='Training Precision', linewidth=2)
        plt.plot(history.history['val_precision'], label='Validation Precision', linewidth=2)
        plt.title('Model Precision', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
    else:
         print("Không tìm thấy dữ liệu Precision trong lịch sử huấn luyện để vẽ đồ thị.")

    # Recall plot
    if 'recall' in history.history and 'val_recall' in history.history:
        plt.subplot(3, 2, 5)
        plt.plot(history.history['recall'], label='Training Recall', linewidth=2)
        plt.plot(history.history['val_recall'], label='Validation Recall', linewidth=2)
        plt.title('Model Recall', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Recall', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
    else:
         print("Không tìm thấy dữ liệu Recall trong lịch sử huấn luyện để vẽ đồ thị.")

    # Đã loại bỏ phần vẽ đồ thị F1 Score

    plt.tight_layout()
    history_plot_path = os.path.join(logs_dir, 'training_history.png')
    plt.savefig(history_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu đồ thị lịch sử huấn luyện tại: {history_plot_path}")

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