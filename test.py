import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt
import json
import argparse

def load_model(model_path):
    """Load mô hình từ file"""
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Đã tải mô hình từ {model_path}")
        return model
    except Exception as e:
        print(f"❌ Lỗi khi tải mô hình: {e}")
        return None

def load_class_mapping(mapping_path):
    """Load ánh xạ lớp từ file JSON"""
    try:
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        print(f"✅ Đã tải ánh xạ lớp từ {mapping_path}")
        return mapping
    except Exception as e:
        print(f"❌ Lỗi khi tải ánh xạ lớp: {e}")
        return None

def preprocess_image(img_path, target_size=(224, 224)):
    """Tiền xử lý ảnh đầu vào"""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_image(model, img_array, class_mapping=None):
    """Dự đoán lớp của ảnh"""
    predictions = model.predict(img_array)
    
    if len(predictions[0]) == 1:  # Binary classification
        class_idx = int(predictions[0][0] > 0.5)
        confidence = float(predictions[0][0] if class_idx == 1 else 1 - predictions[0][0])
        if class_mapping:
            class_name = class_mapping[str(class_idx)]
        else:
            class_name = "recyclable" if class_idx == 1 else "non_recyclable"
    else:  # Multi-class classification
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        if class_mapping:
            class_name = class_mapping[str(class_idx)]
        else:
            class_name = f"class_{class_idx}"
    
    return class_name, confidence

def display_result(img_path, class_name, confidence):
    """Hiển thị kết quả"""
    img = image.load_img(img_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(f"Class: {class_name}\nConfidence: {confidence:.2%}")
    plt.axis('off')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Test mô hình phân loại rác')
    parser.add_argument('--image', type=str, required=True, help='Đường dẫn đến ảnh cần test')
    parser.add_argument('--model', type=str, required=True, choices=['model1', 'model2a', 'model2b'], 
                        help='Loại mô hình cần test (model1, model2a, model2b)')
    args = parser.parse_args()

    # Đường dẫn đến các file mô hình và ánh xạ
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'model')
    model_paths = {
        'model1': os.path.join(models_dir, 'model1_binary_recyclable.keras'),
        'model2a': os.path.join(models_dir, 'model2a_multiclass_recyclable.keras'),
        'model2b': os.path.join(models_dir, 'model2b_multiclass_non_recyclable.keras')
    }
    
    mapping_paths = {
        'model2a': os.path.join(models_dir, 'class_mapping_2a.json'),
        'model2b': os.path.join(models_dir, 'class_mapping_2b.json')
    }

    # Load mô hình và ánh xạ lớp
    model = load_model(model_paths[args.model])
    if model is None:
        return

    class_mapping = None
    if args.model in ['model2a', 'model2b']:
        class_mapping = load_class_mapping(mapping_paths[args.model])

    # Tiền xử lý ảnh
    img_array = preprocess_image(args.image)

    # Dự đoán
    class_name, confidence = predict_image(model, img_array, class_mapping)

    # Hiển thị kết quả
    print(f"\nKết quả phân loại:")
    print(f"- Ảnh: {args.image}")
    print(f"- Mô hình: {args.model}")
    print(f"- Loại rác: {class_name}")
    print(f"- Độ tin cậy: {confidence:.2%}")

    display_result(args.image, class_name, confidence)

if __name__ == "__main__":
    main() 