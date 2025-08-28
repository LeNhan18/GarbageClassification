import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image
import argparse

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """
    Tải và tiền xử lý ảnh đầu vào
    """
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Chuẩn hóa pixel về khoảng [0,1]
        return img, img_array
    except Exception as e:
        print(f"Lỗi khi tải ảnh: {e}")
        return None, None

def predict_image(model, img_array, class_names):
    """
    Dự đoán ảnh và trả về kết quả
    """
    try:
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        return predicted_class, confidence, predictions[0]
    except Exception as e:
        print(f"Lỗi khi dự đoán: {e}")
        return None, None, None

def display_results(img, predicted_class, confidence, all_predictions, class_names):
    """
    Hiển thị ảnh và kết quả dự đoán
    """
    plt.figure(figsize=(12, 6))
    
    # Hiển thị ảnh
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f'Dự đoán: {class_names[predicted_class]}\nĐộ tin cậy: {confidence:.2%}')
    plt.axis('off')
    
    # Hiển thị biểu đồ xác suất
    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(class_names))
    plt.barh(y_pos, all_predictions)
    plt.yticks(y_pos, class_names)
    plt.xlabel('Xác suất')
    plt.title('Xác suất cho mỗi lớp')
    
    plt.tight_layout()
    plt.show()

def main():
    # Thiết lập argument parser với giá trị mặc định
    parser = argparse.ArgumentParser(description='Kiểm tra mô hình phân loại rác tái chế')
    parser.add_argument('--model_path', type=str, 
                       default='Z:\\GarbageClassification\\models\\model_EfficientB2\\model2B_EfficientNetB2_final.keras',
                       help='Đường dẫn đến file mô hình (.keras)')
    parser.add_argument('--image_path', type=str, 
                       default='C:\\Users\\Admin\\Downloads\\u-rac-sinh-hoc-con-duoc-goi-la-u-phan-huu-co.jpg',
                       help='Đường dẫn đến ảnh cần kiểm tra')
    parser.add_argument('--classes', type=str, nargs='+',
                       default=['pin', 'rác hữu cơ', 'quần áo cũ', 'giầy dép', 'rác khác'],
                       help='Danh sách tên các lớp theo thứ tự: cardboard glass metal paper plastic')
    args = parser.parse_args()

    # Kiểm tra file mô hình tồn tại
    if not os.path.exists(args.model_path):
        print(f"Không tìm thấy file mô hình tại: {args.model_path}")
        return

    # Kiểm tra file ảnh tồn tại
    if not os.path.exists(args.image_path):
        print(f"Không tìm thấy file ảnh tại: {args.image_path}")
        return

    # Tải mô hình
    print("Đang tải mô hình...")
    try:
        model = load_model(args.model_path)
        print("Đã tải mô hình thành công!")
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return

    # Tải và tiền xử lý ảnh
    print("Đang xử lý ảnh...")
    img, img_array = load_and_preprocess_image(args.image_path)
    if img is None or img_array is None:
        return

    # Thực hiện dự đoán
    print("Đang thực hiện dự đoán...")
    predicted_class, confidence, all_predictions = predict_image(model, img_array, args.classes)
    if predicted_class is None:
        return

    # Hiển thị kết quả
    print("\nKết quả dự đoán:")
    print(f"Lớp dự đoán: {args.classes[predicted_class]}")
    print(f"Độ tin cậy: {confidence:.2%}")
    
    # In xác suất cho tất cả các lớp
    print("\nXác suất cho mỗi lớp:")
    for i, (class_name, prob) in enumerate(zip(args.classes, all_predictions)):
        print(f"{class_name}: {prob:.2%}")

    # Hiển thị ảnh và biểu đồ
    display_results(img, predicted_class, confidence, all_predictions, args.classes)

if __name__ == "__main__":
    main() 