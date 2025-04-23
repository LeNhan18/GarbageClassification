# utils.py
# Các hàm hỗ trợ cho xử lý ảnh, đổi tên, resize, chuẩn hóa...

import os
from PIL import Image
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import json
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def rename_images_in_folder(folder_path, prefix="img", start_index=1):
    """
    Đổi tên tất cả ảnh trong thư mục thành prefix + số thứ tự (VD: plastic1.jpg)
    """
    index = start_index
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            ext = os.path.splitext(filename)[-1]
            new_name = f"{prefix}{index}{ext}"
            os.rename(file_path, os.path.join(folder_path, new_name))
            index += 1
    print(f"✅ Đã đổi tên toàn bộ ảnh trong {folder_path}")

def resize_images_in_folder(folder_path, size=(150, 150)):
    """
    Resize tất cả ảnh trong thư mục về kích thước size
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                img = Image.open(file_path)
                img = img.resize(size)
                img.save(file_path)
            except Exception as e:
                print(f"⚠️ Lỗi với file {filename}: {e}")
    print(f"✅ Đã resize ảnh trong {folder_path} về {size}")

def prepare_all_subfolders(root_dir, rename_prefix="img", resize_size=(150, 150)):
    """
    Resize và rename toàn bộ ảnh trong các thư mục con của root_dir
    """
    for subdir in os.listdir(root_dir):
        sub_path = os.path.join(root_dir, subdir)
        if os.path.isdir(sub_path):
            resize_images_in_folder(sub_path, size=resize_size)
            rename_images_in_folder(sub_path, prefix=rename_prefix)

def load_trained_model(model_path):
    """
    Tải mô hình đã được huấn luyện
    """
    try:
        model = load_model(model_path)
        print(f"✅ Đã tải mô hình thành công từ: {model_path}")
        return model
    except Exception as e:
        print(f"❌ Lỗi khi tải mô hình: {str(e)}")
        return None

def load_class_mapping(mapping_path):
    """
    Tải ánh xạ từ index sang tên lớp
    """
    try:
        with open(mapping_path, 'r') as f:
            class_mapping = json.load(f)
        print(f"✅ Đã tải ánh xạ lớp thành công từ: {mapping_path}")
        return class_mapping
    except Exception as e:
        print(f"❌ Lỗi khi tải ánh xạ lớp: {str(e)}")
        return None

def preprocess_image(img_path, target_size=(150, 150)):
    """
    Tiền xử lý ảnh đầu vào
    """
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Chuẩn hóa
        return img_array
    except Exception as e:
        print(f"❌ Lỗi khi tiền xử lý ảnh: {str(e)}")
        return None

def predict_single_image(model, img_path, class_mapping=None):
    """
    Dự đoán cho một ảnh đơn lẻ
    """
    try:
        # Tiền xử lý ảnh
        img_array = preprocess_image(img_path)
        if img_array is None:
            return None

        # Dự đoán
        predictions = model.predict(img_array)
        
        # Xử lý kết quả
        if len(predictions[0]) == 1:  # Mô hình nhị phân
            probability = predictions[0][0]
            predicted_class = "Tái chế" if probability > 0.5 else "Không tái chế"
            return {
                "predicted_class": predicted_class,
                "probability": float(probability)
            }
        else:  # Mô hình đa lớp
            predicted_class_idx = np.argmax(predictions[0])
            probability = predictions[0][predicted_class_idx]
            if class_mapping:
                predicted_class = class_mapping[str(predicted_class_idx)]
            else:
                predicted_class = str(predicted_class_idx)
            return {
                "predicted_class": predicted_class,
                "probability": float(probability),
                "all_probabilities": {str(i): float(p) for i, p in enumerate(predictions[0])}
            }
    except Exception as e:
        print(f"❌ Lỗi khi dự đoán: {str(e)}")
        return None

def evaluate_model(model, test_generator, class_names=None):
    """
    Đánh giá mô hình trên tập test
    """
    try:
        # Đánh giá tổng thể
        results = model.evaluate(test_generator)
        metrics = dict(zip(model.metrics_names, results))
        
        # Dự đoán
        y_pred = model.predict(test_generator)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = test_generator.classes
        
        # Tính ma trận nhầm lẫn
        cm = confusion_matrix(y_true, y_pred_classes)
        
        # Vẽ ma trận nhầm lẫn
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Lưu biểu đồ
        plt.savefig('logs/confusion_matrix.png')
        plt.close()
        
        # Báo cáo phân loại
        if class_names:
            report = classification_report(y_true, y_pred_classes, target_names=class_names)
        else:
            report = classification_report(y_true, y_pred_classes)
            
        return {
            "metrics": metrics,
            "confusion_matrix": cm.tolist(),
            "classification_report": report
        }
    except Exception as e:
        print(f"❌ Lỗi khi đánh giá mô hình: {str(e)}")
        return None

def visualize_predictions(model, test_generator, class_mapping=None, num_images=5):
    """
    Trực quan hóa kết quả dự đoán
    """
    try:
        # Lấy một số ảnh từ test_generator
        images, labels = next(test_generator)
        
        # Dự đoán
        predictions = model.predict(images)
        
        # Hiển thị kết quả
        plt.figure(figsize=(15, 5))
        for i in range(min(num_images, len(images))):
            plt.subplot(1, num_images, i + 1)
            plt.imshow(images[i])
            
            if len(predictions[i]) == 1:  # Mô hình nhị phân
                pred_class = "Tái chế" if predictions[i][0] > 0.5 else "Không tái chế"
                true_class = "Tái chế" if labels[i][0] > 0.5 else "Không tái chế"
            else:  # Mô hình đa lớp
                pred_idx = np.argmax(predictions[i])
                true_idx = np.argmax(labels[i])
                if class_mapping:
                    pred_class = class_mapping[str(pred_idx)]
                    true_class = class_mapping[str(true_idx)]
                else:
                    pred_class = str(pred_idx)
                    true_class = str(true_idx)
            
            plt.title(f"True: {true_class}\nPred: {pred_class}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('logs/prediction_visualization.png')
        plt.close()
        
        return True
    except Exception as e:
        print(f"❌ Lỗi khi trực quan hóa dự đoán: {str(e)}")
        return False

def save_evaluation_results(results, output_dir='logs'):
    """
    Lưu kết quả đánh giá
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Lưu metrics
        with open(os.path.join(output_dir, 'evaluation_metrics.txt'), 'w') as f:
            for metric, value in results['metrics'].items():
                f.write(f"{metric}: {value}\n")
        
        # Lưu báo cáo phân loại
        with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
            f.write(results['classification_report'])
            
        print(f"✅ Đã lưu kết quả đánh giá vào thư mục: {output_dir}")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi lưu kết quả đánh giá: {str(e)}")
        return False

if __name__ == '__main__':
    # Test nhanh
    folder = input("Nhập đường dẫn thư mục ảnh: ")
    prepare_all_subfolders(folder)
