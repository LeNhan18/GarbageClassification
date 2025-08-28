# train_model_mobilenetv2_single_phase_improved.py
# Huấn luyện mô hình phân loại nhị phân (rác tái chế vs không tái chế) sử dụng MobileNetV2 - Phiên bản cải thiện

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from datetime import datetime

# Custom F1Score metric cho binary classification
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()
# --- Cấu hình các tham số chính ---
data_dir = 'Z:\\GarbageClassification\\datas'
img_size = (224, 224)
batch_size = 16  # Tăng batch size để ổn định hơn
input_shape = (img_size[0], img_size[1], 3)
epochs = 70
num_classes = 1

# --- Tạo thư mục lưu mô hình và logs ---
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
root_output_dir = os.path.dirname(data_dir)
models_dir = os.path.join(root_output_dir, 'models', 'model_mobilenetv2_improved',timestamp)
logs_dir = os.path.join(root_output_dir, 'logs', 'logs_mobilenetv2_improved', timestamp)

os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
print(f"Thư mục lưu mô hình: {models_dir}")
print(f"Thư mục lưu logs: {logs_dir}")

# --- Data Augmentation được tối ưu ---
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2,
    rotation_range=20,  # Giảm rotation để tránh biến dạng quá mức
    width_shift_range=0.1,  # Giảm shift range
    height_shift_range=0.1,
    shear_range=0.1,  # Giảm shear
    zoom_range=0.1,  # Giảm zoom
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],  # Giảm brightness range
    channel_shift_range=20.0,  # Giảm channel shift
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2
)


# Tạo generators với error handling
def create_data_generators():
    try:
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='binary',
            subset='training',
            shuffle=True,
            seed=42  # Thêm seed để reproducible
        )

        val_generator = val_datagen.flow_from_directory(
            data_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='binary',
            subset='validation',
            shuffle=False,
            seed=42
        )

        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {val_generator.samples}")
        print(f"Class indices: {train_generator.class_indices}")

        return train_generator, val_generator
    except Exception as e:
        print(f"Lỗi tạo data generators: {e}")
        return None, None


train_generator, val_generator = create_data_generators()
if train_generator is None:
    exit()


# --- Xây dựng mô hình cải thiện ---
def build_improved_model():
    # Base model MobileNetV2
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Fine-tuning strategy: Unfreeze gradually
    total_layers = len(base_model.layers)
    unfreeze_from = int(total_layers * 0.8)  # Unfreeze top 20% layers

    print(f"Total base layers: {total_layers}")
    print(f"Unfreezing from layer {unfreeze_from} ({total_layers - unfreeze_from} layers)")

    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False
    for layer in base_model.layers[unfreeze_from:]:
        layer.trainable = True

    # Improved architecture
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),

        # First dense layer - 1024 neurons
        layers.Dense(1024, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),

        # Second dense layer - 512 neurons
        layers.Dense(512, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.4),

        # Third dense layer - 256 neurons
        layers.Dense(256, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),

        # Output layer
        layers.Dense(num_classes, activation='sigmoid', dtype='float32')
    ])

    return model


model = build_improved_model()

# --- Compilation với custom F1Score ---
initial_lr = 1e-4  # Learning rate hợp lý hơn
optimizer = optimizers.Adam(learning_rate=initial_lr, beta_1=0.9, beta_2=0.999)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        F1Score(name='f1_score')
    ]
)

print("\n=== Model Summary ===")
model.summary()

# --- Improved Callbacks ---
checkpoint_path = os.path.join(models_dir, 'best_model.keras')
callbacks = [
    ModelCheckpoint(
        checkpoint_path,
        monitor='val_f1_score',
        save_best_only=True,
        mode='max',
        verbose=1,
        save_weights_only=False
    ),
    EarlyStopping(
        monitor='val_f1_score',
        patience=15,  # Tăng patience
        restore_best_weights=True,
        mode='max',
        verbose=1,
        min_delta=0.001  # Thêm min_delta
    ),
    ReduceLROnPlateau(
        monitor='val_loss',  # Monitor loss thay vì f1_score
        factor=0.5,  # Giảm ít hơn mỗi lần
        patience=8,
        min_lr=1e-7,
        verbose=1,
        cooldown=3  # Thêm cooldown
    ),
    CSVLogger(os.path.join(logs_dir, 'training_log.csv'), append=True)
]


# --- Training với improved setup ---
print("\n=== Bắt đầu huấn luyện ===")

# Calculate steps
train_steps = max(1, train_generator.samples // batch_size)
val_steps = max(1, val_generator.samples // batch_size)

print(f"Training steps per epoch: {train_steps}")
print(f"Validation steps per epoch: {val_steps}")

try:
    history = model.fit(
        train_generator,
        steps_per_epoch=train_steps,
        validation_data=val_generator,
        validation_steps=val_steps,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    print("Huấn luyện hoàn thành!")

except Exception as e:
    print(f"Lỗi trong quá trình huấn luyện: {e}")
    exit()


# --- Model Loading và Evaluation ---
def load_and_evaluate_model():
    try:
        print(f"\nĐang load model từ: {checkpoint_path}")
        best_model = models.load_model(
            checkpoint_path,
            custom_objects={'F1Score': F1Score}
        )

        # Save final model
        final_path = os.path.join(models_dir, 'final_model.keras')
        best_model.save(final_path)
        print(f"Đã lưu final model tại: {final_path}")

        return best_model
    except Exception as e:
        print(f"Lỗi load model: {e}")
        return model  # Fallback to current model


final_model = load_and_evaluate_model()


# --- Detailed Evaluation ---
def detailed_evaluation(model, val_gen):
    print("\n=== Đánh giá chi tiết ===")

    # Reset generator
    val_gen.reset()

    # Evaluate
    eval_results = model.evaluate(val_gen, verbose=1)
    metric_names = ['loss', 'accuracy', 'auc', 'precision', 'recall', 'f1_score']

    print("\nKết quả đánh giá:")
    results_dict = {}
    for name, value in zip(metric_names, eval_results):
        print(f"- {name}: {value:.4f}")
        results_dict[name] = value

    # Generate predictions
    print("\nTạo predictions...")
    val_gen.reset()

    # Collect all predictions and true labels
    y_true_all = []
    y_pred_all = []

    num_batches = len(val_gen)
    for i in range(num_batches):
        try:
            x_batch, y_batch = next(val_gen)
            predictions = model.predict_on_batch(x_batch)

            y_true_all.extend(y_batch.flatten())
            y_pred_all.extend(predictions.flatten())

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{num_batches} batches")

        except StopIteration:
            break

    y_true = np.array(y_true_all)
    y_pred_probs = np.array(y_pred_all)
    y_pred = (y_pred_probs > 0.5).astype(int)

    return results_dict, y_true, y_pred, y_pred_probs


# Run evaluation
eval_results, y_true, y_pred, y_pred_probs = detailed_evaluation(final_model, val_generator)


# --- Comprehensive Visualization Functions ---
def create_comprehensive_visualizations(history, y_true, y_pred, y_pred_probs, logs_dir):
    # Get class names
    class_names = ['Non_Recyclable', 'Recyclable']
    if hasattr(train_generator, 'class_indices'):
        indices = train_generator.class_indices
        class_names = [''] * 2
        for name, idx in indices.items():
            class_names[idx] = name

    print("Creating comprehensive visualizations...")

    # Set style for better plots
    plt.style.use('default')
    sns.set_palette("husl")

    # 1. Enhanced Confusion Matrix with percentages
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create annotations with both counts and percentages
    annotations = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            row.append(f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)')
        annotations.append(row)

    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix (Count + Percentage)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.savefig(os.path.join(logs_dir, 'confusion_matrix_detailed.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. ROC Curve with additional metrics
    plt.figure(figsize=(12, 10))
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    # Find optimal threshold (Youden's index)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10,
             label=f'Optimal Threshold = {optimal_threshold:.3f}')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve Analysis', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(logs_dir, 'roc_curve_detailed.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Precision-Recall Curve
    from sklearn.metrics import precision_recall_curve, average_precision_score

    plt.figure(figsize=(12, 10))
    precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_true, y_pred_probs)
    avg_precision = average_precision_score(y_true, y_pred_probs)

    plt.plot(recall_vals, precision_vals, color='purple', lw=3,
             label=f'PR Curve (AP = {avg_precision:.3f})')
    plt.axhline(y=np.mean(y_true), color='red', linestyle='--',
                label=f'Baseline (Random) = {np.mean(y_true):.3f}')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(logs_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Comprehensive Training History (6 metrics)
    plt.figure(figsize=(24, 16))

    metrics_to_plot = [
        ('accuracy', 'Accuracy', 'green'),
        ('loss', 'Loss', 'red'),
        ('auc', 'AUC', 'blue'),
        ('precision', 'Precision', 'orange'),
        ('recall', 'Recall', 'purple'),
        ('f1_score', 'F1 Score', 'brown')
    ]

    for i, (metric, title, color) in enumerate(metrics_to_plot, 1):
        plt.subplot(2, 3, i)
        if metric in history.history and f'val_{metric}' in history.history:
            epochs_range = range(1, len(history.history[metric]) + 1)

            plt.plot(epochs_range, history.history[metric],
                     color=color, linewidth=2.5, label=f'Training {title}', alpha=0.8)
            plt.plot(epochs_range, history.history[f'val_{metric}'],
                     color=color, linewidth=2.5, linestyle='--',
                     label=f'Validation {title}', alpha=0.8)

            # Add best point
            best_epoch = np.argmax(history.history[f'val_{metric}']) + 1
            best_val = np.max(history.history[f'val_{metric}'])
            plt.plot(best_epoch, best_val, 'ro', markersize=8)
            plt.annotate(f'Best: {best_val:.3f}\n@Epoch {best_epoch}',
                         xy=(best_epoch, best_val), xytext=(10, 10),
                         textcoords='offset points', fontsize=10,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

            plt.title(f'Model {title}', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel(title, fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)

    plt.suptitle('Training History - All Metrics', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(logs_dir, 'training_history_comprehensive.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Learning Rate Schedule (if available)
    if 'lr' in history.history:
        plt.figure(figsize=(12, 8))
        plt.plot(history.history['lr'], linewidth=2, color='red')
        plt.title('Learning Rate Schedule', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Learning Rate', fontsize=14)
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(logs_dir, 'learning_rate_schedule.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 6. Prediction Distribution Analysis
    plt.figure(figsize=(15, 10))

    # Subplot 1: Prediction histogram
    plt.subplot(2, 2, 1)
    plt.hist(y_pred_probs[y_true == 0], bins=50, alpha=0.7, label=class_names[0], color='red')
    plt.hist(y_pred_probs[y_true == 1], bins=50, alpha=0.7, label=class_names[1], color='blue')
    plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
    plt.xlabel('Predicted Probability', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Prediction Distribution by True Class', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Threshold analysis
    plt.subplot(2, 2, 2)
    thresholds_range = np.arange(0.1, 0.9, 0.01)
    f1_scores = []
    accuracies = []

    for thresh in thresholds_range:
        y_pred_thresh = (y_pred_probs > thresh).astype(int)
        from sklearn.metrics import f1_score, accuracy_score
        f1_scores.append(f1_score(y_true, y_pred_thresh))
        accuracies.append(accuracy_score(y_true, y_pred_thresh))

    plt.plot(thresholds_range, f1_scores, label='F1 Score', linewidth=2)
    plt.plot(thresholds_range, accuracies, label='Accuracy', linewidth=2)
    plt.axvline(x=0.5, color='red', linestyle='--', label='Default (0.5)')

    # Find optimal F1 threshold
    optimal_f1_idx = np.argmax(f1_scores)
    optimal_f1_thresh = thresholds_range[optimal_f1_idx]
    plt.axvline(x=optimal_f1_thresh, color='green', linestyle='--',
                label=f'Optimal F1 ({optimal_f1_thresh:.2f})')

    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Threshold vs Performance Metrics', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 3: Class distribution
    plt.subplot(2, 2, 3)
    class_counts = [np.sum(y_true == 0), np.sum(y_true == 1)]
    colors = ['lightcoral', 'lightskyblue']
    plt.pie(class_counts, labels=class_names, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('True Class Distribution', fontsize=14, fontweight='bold')

    # Subplot 4: Prediction vs True scatter
    plt.subplot(2, 2, 4)
    plt.scatter(y_true, y_pred_probs, alpha=0.6, s=20)
    plt.axhline(y=0.5, color='red', linestyle='--', label='Threshold')
    plt.xlabel('True Label', fontsize=12)
    plt.ylabel('Predicted Probability', fontsize=12)
    plt.title('True vs Predicted', fontsize=14, fontweight='bold')
    plt.xticks([0, 1], class_names)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(logs_dir, 'prediction_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 7. Model Performance Summary Dashboard
    plt.figure(figsize=(16, 12))

    # Calculate additional metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

    final_metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'AUC-ROC': roc_auc,
        'AUC-PR': avg_precision
    }

    # Create dashboard
    gs = plt.GridSpec(3, 3, figure=plt.gcf())

    # Metrics bar chart
    ax1 = plt.subplot(gs[0, :2])
    metrics_names = list(final_metrics.keys())
    metrics_values = list(final_metrics.values())
    bars = ax1.bar(metrics_names, metrics_values,
                   color=['skyblue', 'lightgreen', 'salmon', 'gold', 'plum', 'lightcoral'])
    ax1.set_ylim(0, 1.1)
    ax1.set_title('Final Model Performance Metrics', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12)

    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Mini confusion matrix
    ax2 = plt.subplot(gs[0, 2])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Confusion Matrix', fontsize=12, fontweight='bold')

    # Training progress overview
    ax3 = plt.subplot(gs[1, :])
    epochs_range = range(1, len(history.history['loss']) + 1)
    ax3.plot(epochs_range, history.history['val_accuracy'], 'b-', linewidth=2, label='Val Accuracy')
    ax3.plot(epochs_range, history.history['val_f1_score'], 'r-', linewidth=2, label='Val F1-Score')
    ax3.plot(epochs_range, history.history['val_auc'], 'g-', linewidth=2, label='Val AUC')
    ax3.set_title('Key Validation Metrics Progress', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ROC curve mini
    ax4 = plt.subplot(gs[2, 0])
    ax4.plot(fpr, tpr, 'b-', linewidth=2)
    ax4.plot([0, 1], [0, 1], 'r--', linewidth=1)
    ax4.set_title(f'ROC (AUC={roc_auc:.3f})', fontsize=12, fontweight='bold')
    ax4.set_xlabel('FPR')
    ax4.set_ylabel('TPR')
    ax4.grid(True, alpha=0.3)

    # PR curve mini
    ax5 = plt.subplot(gs[2, 1])
    ax5.plot(recall_vals, precision_vals, 'g-', linewidth=2)
    ax5.set_title(f'PR (AP={avg_precision:.3f})', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Recall')
    ax5.set_ylabel('Precision')
    ax5.grid(True, alpha=0.3)

    # Class distribution
    ax6 = plt.subplot(gs[2, 2])
    ax6.pie(class_counts, labels=class_names, autopct='%1.1f%%', startangle=90)
    ax6.set_title('Class Distribution', fontsize=12, fontweight='bold')

    plt.suptitle('Model Performance Dashboard', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(logs_dir, 'performance_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(os.path.join(logs_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    print("✅ Created comprehensive visualizations:")
    print("  - Detailed Confusion Matrix")
    print("  - ROC Curve with optimal threshold")
    print("  - Precision-Recall Curve")
    print("  - Comprehensive Training History")
    print("  - Learning Rate Schedule (if available)")
    print("  - Prediction Distribution Analysis")
    print("  - Performance Dashboard")

    return cm, report, final_metrics, optimal_f1_thresh

# --- Summary Report ---
def create_summary_report():
    summary_path = os.path.join(logs_dir, 'training_summary.txt')

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=== TRAINING SUMMARY - MOBILENETV2 IMPROVED ===\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: MobileNetV2 + Dense (1024->512->256->1)\n")
        f.write(f"Input size: {img_size}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Total epochs trained: {len(history.history['loss'])}\n\n")

        f.write("=== FINAL RESULTS ===\n")
        for metric, value in eval_results.items():
            f.write(f"{metric}: {value:.4f}\n")

        f.write(f"\n=== DATASET INFO ===\n")
        f.write(f"Training samples: {train_generator.samples}\n")
        f.write(f"Validation samples: {val_generator.samples}\n")

    print(f"Summary report saved to: {summary_path}")


create_summary_report()

print("\n" + "=" * 50)
print("TRAINING COMPLETED SUCCESSFULLY!")
print(f"Best model saved at: {os.path.join(models_dir, 'final_model.keras')}")
print(f"Logs and visualizations saved at: {logs_dir}")
print("=" * 50)