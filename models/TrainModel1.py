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

# --- Cấu hình các tham số chính ---
data_dir = 'Z:\\GarbageClassification\\datas'
img_size = (224, 224)  # Giữ nguyên kích thước ảnh theo yêu cầu
batch_size = 16
input_shape = (img_size[0], img_size[1], 3)
epochs = 70  # Số epochs tối đa, EarlyStopping sẽ dừng sớm nếu cần
num_classes = 1  # Cho phân loại nhị phân (sử dụng sigmoid)

# --- Tạo thư mục lưu mô hình và logs ---
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
root_output_dir = 'Z:\\GarbageClassification'  # Đặt thư mục gốc theo cấu trúc của bạn
models_dir_base_name = 'Model1_EfficientNetB2_BalancedTune'  # Cập nhật tên để phản ánh điều chỉnh
logs_dir_base_name = f'logs_{models_dir_base_name}'

models_dir = os.path.join(root_output_dir, 'models_Model1_EfficientNetB2', models_dir_base_name, timestamp)
logs_dir = os.path.join(root_output_dir, 'logs_Model1_EfficientNetB2', logs_dir_base_name, timestamp)

os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
print(f"Thư mục lưu mô hình: {models_dir}")
print(f"Thư mục lưu logs: {logs_dir}")

# --- Data Augmentation Cải tiến ---
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=[0.85, 1.15],
    horizontal_flip=True,
    vertical_flip=False,  # Giữ False, quan trọng để tránh làm mất đặc trưng hướng của một số loại rác
    brightness_range=[0.6, 1.4],
    channel_shift_range=30.0,
    fill_mode='reflect',
)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2
)

print("\nĐang tạo Data Generators...")
try:
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
    print(f"Tìm thấy {train_generator.samples} ảnh cho training.")
    print(f"Tìm thấy {val_generator.samples} ảnh cho validation.")
    print(f"Các lớp tìm thấy: {train_generator.class_indices}")

except Exception as e:
    print(f"Lỗi khi tạo data generators: {e}")
    print(f"Kiểm tra lại đường dẫn thư mục dữ liệu: {data_dir}")
    exit()

# --- Xây dựng mô hình (Single Phase) ---
print("\nĐang xây dựng kiến trúc mô hình (Một giai đoạn)...")
base_model = EfficientNetB2(
    weights='imagenet',
    include_top=False,
    input_shape=input_shape
)
# --- Phần head của mô hình ---
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),

    layers.Dense(256, kernel_regularizer=regularizers.l2(0.0005), name="dense_1"),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),

    layers.Dense(128, kernel_regularizer=regularizers.l2(0.0005), name="dense_2"),
    layers.BatchNormalization(name="bn_dense_2"),
    layers.Activation('relu', name="relu_2"),
    layers.Dropout(0.4, name="dropout_2"),

    layers.Dense(num_classes, activation='sigmoid', dtype='float32', name="output_layer")
])

# --- Biên dịch mô hình ---
initial_learning_rate = 3e-5  # Giữ learning rate nhỏ
optimizer = optimizers.Adam(learning_rate=initial_learning_rate)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)
model.summary()

# --- Callbacks ---
model_filename_base_for_callbacks = f'model_{models_dir_base_name}'
checkpoint_path = os.path.join(models_dir, f'{model_filename_base_for_callbacks}_best_{timestamp}.keras')

callbacks = [
    ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=4,
        min_lr=1e-7,
        verbose=1
    ),
    CSVLogger(os.path.join(logs_dir, f'{model_filename_base_for_callbacks}_training_log_{timestamp}.csv')),
    tf.keras.callbacks.TensorBoard(
        log_dir=logs_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=False
    )
]

print("\nBắt đầu huấn luyện mô hình (Một giai đoạn - Điều chỉnh cân bằng)...")  # Cập nhật thông báo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)

# --- Lưu mô hình và đánh giá ---
best_model_loaded = None
try:
    print(f"\nĐang tải mô hình tốt nhất từ: {checkpoint_path}")
    best_model_loaded = models.load_model(checkpoint_path)
    final_model_path = os.path.join(models_dir,
                                    f'{model_filename_base_for_callbacks}_final_from_best_{timestamp}.keras')
    best_model_loaded.save(final_model_path)
    print(f"Đã lưu mô hình tốt nhất (final) tại: {final_model_path}")
except Exception as e:
    print(f"Lỗi khi tải hoặc lưu mô hình tốt nhất: {e}")
    print("Thử lưu mô hình cuối cùng của quá trình huấn luyện...")
    try:
        final_model_path_fallback = os.path.join(models_dir,
                                                 f'{model_filename_base_for_callbacks}_final_fallback_{timestamp}.keras')
        model.save(final_model_path_fallback)
        best_model_loaded = model
        print(f"Đã lưu mô hình cuối cùng của quá trình huấn luyện tại: {final_model_path_fallback}")
    except Exception as e_fallback:
        print(f"Lỗi khi lưu mô hình cuối cùng: {e_fallback}")

if best_model_loaded:
    print("\nĐang đánh giá mô hình tốt nhất trên tập Validation...")
    val_generator.reset()
    val_metrics = best_model_loaded.evaluate(val_generator, verbose=1)
    metrics_names = best_model_loaded.metrics_names

    print("\nKết quả đánh giá (Mô hình tốt nhất):")
    for name, value in zip(metrics_names, val_metrics):
        print(f"- {name}: {value:.4f}")

    print("\nĐang tạo dự đoán trên tập Validation để phân tích chi tiết...")
    y_pred_probs = []
    y_true = []
    val_generator.reset()
    num_val_batches = len(val_generator)

    for i in range(num_val_batches):
        x, y = next(val_generator)
        pred = best_model_loaded.predict_on_batch(x)
        y_pred_probs.extend(pred.flatten())
        y_true.extend(y.flatten())
        if (i + 1) % 20 == 0 or (i + 1) == num_val_batches:
            print(f"Đã xử lý batch dự đoán {i + 1}/{num_val_batches}")

    y_true = np.array(y_true[:val_generator.samples])
    y_pred_probs = np.array(y_pred_probs[:val_generator.samples])
    y_pred = (y_pred_probs > 0.5).astype(int)

    try:
        class_indices = train_generator.class_indices
        class_names_ordered = [None] * len(class_indices)
        for class_name, index in class_indices.items():
            if 0 <= index < len(class_names_ordered):
                class_names_ordered[index] = class_name
            else:
                print(
                    f"Cảnh báo: class index {index} cho lớp '{class_name}' nằm ngoài dự kiến ({len(class_names_ordered)} lớp).")
        if None in class_names_ordered or len(class_names_ordered) != (num_classes if num_classes > 1 else 2):
            print("Không thể xác định đầy đủ tên lớp từ class_indices, sử dụng tên mặc định.")
            class_names_ordered = [f'Lớp {i}' for i in range(2 if num_classes == 1 else num_classes)]
        print(f"Tên lớp (theo thứ tự 0, 1) cho báo cáo: {class_names_ordered}")
    except Exception as e:
        print(f"Lỗi khi lấy tên lớp từ generator: {e}. Sử dụng tên mặc định ['Lớp 0', 'Lớp 1'].")
        class_names_ordered = ['Lớp 0', 'Lớp 1']

    cm_path = os.path.join(logs_dir, f'confusion_matrix_{models_dir_base_name}_{timestamp}.png')
    print(f"\nĐang tạo Confusion Matrix tại: {cm_path}...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=class_names_ordered,
                yticklabels=class_names_ordered)
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('Actual Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Đã lưu Confusion Matrix.")

    print("\nĐang tạo Classification Report...")
    report = classification_report(y_true, y_pred, target_names=class_names_ordered, zero_division=0)
    print(report)
    report_path = os.path.join(logs_dir, f'classification_report_{models_dir_base_name}_{timestamp}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Đã lưu Classification Report tại: {report_path}")

    roc_auc_val = 0.0
    print("\nĐang vẽ ROC curve...")
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
        roc_auc_val = auc(fpr, tpr)
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_val:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('Receiver Operating Characteristic (ROC)', fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        roc_curve_path = os.path.join(logs_dir, f'roc_curve_{models_dir_base_name}_{timestamp}.png')
        plt.savefig(roc_curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Đã lưu ROC curve tại: {roc_curve_path}")
    except ValueError:
        print(f"Lỗi khi vẽ ROC curve: Không thể tính toán ROC AUC. Kiểm tra lại nhãn y_true và dự đoán y_pred_probs.")
        print(f"Số lượng nhãn duy nhất trong y_true: {np.unique(y_true)}")
    except Exception as e:
        print(f"Lỗi không xác định khi vẽ ROC curve: {e}")

    print("\nĐang vẽ đồ thị lịch sử huấn luyện...")
    plt.figure(figsize=(18, 12))

    plot_idx_counter = 1
    num_cols = 2

    available_metrics = [m for m in ['accuracy', 'loss', 'auc', 'precision', 'recall'] if
                         m in history.history and f'val_{m}' in history.history]
    num_metrics_to_plot = len(available_metrics)
    num_rows = int(np.ceil(num_metrics_to_plot / num_cols))
    if num_rows == 0: num_rows = 1


    def plot_metric_local(metric_name, plot_title):
        global plot_idx_counter  # SỬA LỖI: Sử dụng global để tham chiếu đến plot_idx_counter ở phạm vi module
        plt.subplot(num_rows, num_cols, plot_idx_counter)
        plt.plot(history.history[metric_name], label=f'Training {plot_title}', linewidth=2)
        plt.plot(history.history[f'val_{metric_name}'], label=f'Validation {plot_title}', linewidth=2)
        plt.title(f'Model {plot_title}', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel(plot_title, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plot_idx_counter += 1


    if 'accuracy' in available_metrics:
        plot_metric_local('accuracy', 'Accuracy')
    if 'loss' in available_metrics:
        plot_metric_local('loss', 'Loss')
    if 'auc' in available_metrics:
        plot_metric_local('auc', 'AUC')
    if 'precision' in available_metrics:
        plot_metric_local('precision', 'Precision')
    if 'recall' in available_metrics:
        plot_metric_local('recall', 'Recall')

    if num_metrics_to_plot == 0:
        if num_rows > 0 and num_cols > 0:
            ax = plt.subplot(num_rows, num_cols, 1)
            ax.text(0.5, 0.5, 'Không có dữ liệu metrics để vẽ đồ thị.',
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        else:
            plt.text(0.5, 0.5, 'Không có dữ liệu metrics để vẽ đồ thị và không thể tạo subplot.',
                     horizontalalignment='center', verticalalignment='center')

    plt.tight_layout(pad=3.0)
    history_plot_path = os.path.join(logs_dir, f'training_history_{models_dir_base_name}_{timestamp}.png')
    plt.savefig(history_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu đồ thị lịch sử huấn luyện tại: {history_plot_path}")

    summary_file_path = os.path.join(logs_dir, f'summary_results_{models_dir_base_name}_{timestamp}.txt')
    with open(summary_file_path, 'w', encoding='utf-8') as f:
        f.write(
            f"=== KẾT QUẢ HUẤN LUYỆN MÔ HÌNH PHÂN LOẠI RÁC TÁI CHẾ (MỘT GIAI ĐOẠN - ĐIỀU CHỈNH CÂN BẰNG) ===\n\n")  # Cập nhật tiêu đề
        f.write(f"Thời gian chạy: {timestamp}\n")
        f.write(f"Mô hình gốc (Base Model): EfficientNetB2\n")
        f.write(f"Tên mô hình tùy chỉnh (Custom Model Name): {models_dir_base_name}\n")
        f.write(f"Tổng số epochs huấn luyện thực tế: {len(history.history.get('loss', []))}\n\n")

        f.write("--- Kết quả đánh giá cuối cùng (Mô hình tốt nhất trên tập Validation) ---\n")
        for name, value in zip(metrics_names, val_metrics):
            f.write(f"{name.capitalize()}: {value:.4f}\n")
        if 'roc_auc_val' in locals() and roc_auc_val > 0.0:
            f.write(f"ROC AUC (tính toán thủ công từ dự đoán validation): {roc_auc_val:.4f}\n\n")
        else:
            f.write("ROC AUC (tính toán thủ công từ dự đoán validation): Không thể tính toán hoặc không hợp lệ\n\n")

        f.write("=== Classification Report (trên tập Validation) ===\n")
        f.write(report)
        f.write("\n\n=== Confusion Matrix (trên tập Validation) ===\n")
        f.write("Labels: " + str(class_names_ordered) + "\n")
        f.write(np.array2string(cm, separator=', '))
        f.write("\n")

        f.write("\n\n=== Thông tin chi tiết Huấn luyện ===\n")
        f.write(f"Thư mục dữ liệu: {data_dir}\n")
        f.write(f"Kích thước ảnh đầu vào: {img_size}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Số lượng ảnh training: {train_generator.samples}\n")
        f.write(f"Số lượng ảnh validation: {val_generator.samples}\n")
        f.write(f"Các lớp được phát hiện: {train_generator.class_indices}\n")

        f.write("\n--- Tham số Huấn luyện Một Giai đoạn ---\n")
        f.write(f"Initial Learning Rate: {initial_learning_rate}\n")
        f.write(f"Epochs Tối đa được cấu hình: {epochs}\n")

        es_callback = next((cb for cb in callbacks if isinstance(cb, EarlyStopping)), None)
        if es_callback:
            f.write(f"EarlyStopping Patience: {es_callback.patience} (monitor='{es_callback.monitor}')\n")
        rop_callback = next((cb for cb in callbacks if isinstance(cb, ReduceLROnPlateau)), None)
        if rop_callback:
            f.write(
                f"ReduceLROnPlateau Patience: {rop_callback.patience} (monitor='{rop_callback.monitor}', factor={rop_callback.factor}, min_lr={rop_callback.min_lr})\n")

        f.write(f"\n--- Cấu hình Đóng băng Base Model ---\n")
        f.write(
            f"Fine-tune từ block: {fine_tune_from_block if 'fine_tune_from_block' in locals() else 'Toàn bộ base model được huấn luyện (Không đóng băng)'}\n")

        f.write("\n--- Cấu hình Data Augmentation (Training) ---\n")
        aug_params = train_datagen.__dict__
        main_aug_keys = ['rescale', 'validation_split', 'rotation_range', 'width_shift_range', 'height_shift_range',
                         'shear_range', 'zoom_range', 'horizontal_flip', 'vertical_flip',
                         'brightness_range', 'channel_shift_range', 'fill_mode']
        for k in main_aug_keys:
            if k in aug_params:
                value = aug_params[k]
                if k == 'zoom_range' and isinstance(value, (list, tuple)):
                    f.write(f"  {k}: [{value[0]:.2f}, {value[1]:.2f}]\n")
                else:
                    f.write(f"  {k}: {value}\n")

    print(f"Hoàn thành quá trình huấn luyện và đánh giá mô hình! Kết quả lưu tại: {logs_dir} và {models_dir}")
    print(f"Báo cáo tổng hợp đã được lưu tại: {summary_file_path}")

else:
    print("Không có mô hình tốt nhất để đánh giá (best_model_loaded is None).")

