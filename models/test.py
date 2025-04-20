# test_model1_camera.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json
import os
from utils import preprocess_image, predict_single_image

from TrainModel1 import train_generator

# --- Load model ---
model_path = 'model1_binary_recyclable.keras'  # hoặc model1_best.keras
model = load_model(model_path)
# Tạo lại ánh xạ lớp nếu không có file json
class_mapping = train_generator.class_indices
inv_class_mapping = {v: k for k, v in class_mapping.items()}
with open('class_mapping.json', 'w') as f:
    json.dump(inv_class_mapping, f)

# --- Load class mapping ---
# Nếu bạn có file class_mapping.json
try:
    with open('class_mapping.json', 'r') as f:
        class_mapping = json.load(f)
    print("Đã load ánh xạ lớp.")
except:
    # Nếu không có, bạn tự định nghĩa:
    class_mapping = {0: 'Non-Recyclable', 1: 'Recyclable'}  # hoặc ngược lại nếu model bạn học theo chiều ngược


# --- Hàm xử lý ảnh ---
def preprocess_image(img, target_size=(150, 150)):
    img_resized = cv2.resize(img, target_size)
    img_array = np.expand_dims(img_resized, axis=0)
    img_array = img_array / 255.0  # normalize giống lúc train
    return img_array


def test_model():
    print("=== BẮT ĐẦU KIỂM TRA MODEL ===")
    
    # --- Load model và mapping ---
    model_path = 'model1_binary_recyclable.keras'
    mapping_path = 'class_mapping.json'
    
    try:
        model = load_model(model_path)
        print("Đã tải mô hình thành công")
        
        with open(mapping_path, 'r') as f:
            class_mapping = json.load(f)
        print("Đã tải mapping thành công")
    except Exception as e:
        print(f"Lỗi khi tải mô hình hoặc mapping: {e}")
        return

    # --- Mở camera ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Không thể mở camera")
        return

    print("🚀 Camera đã sẵn sàng. Bấm 'q' để thoát.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc frame từ camera")
            break

        # Vẽ khung giữa ảnh
        h, w = frame.shape[:2]
        size = min(h, w) // 2
        x1 = (w - size) // 2
        y1 = (h - size) // 2
        x2 = x1 + size
        y2 = y1 + size
        
        # Lấy ảnh trong khung
        roi = frame[y1:y2, x1:x2]
        
        # Tiền xử lý ảnh
        img = cv2.resize(roi, (128, 128))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Dự đoán
        prediction = model.predict(img)[0]
        class_index = np.argmax(prediction)
        confidence = float(prediction[class_index])
        label = "Tái chế" if class_index == 1 else "Không tái chế"
        
        # Vẽ khung và kết quả
        color = (0, 255, 0) if label == "Tái chế" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", 
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow('Phân Loại Rác Thải', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Đã đóng camera và kết thúc chương trình")

def test_model1_images():
    print("\n=== BẮT ĐẦU KIỂM TRA MODEL 1 VỚI ẢNH ===")
    
    # --- Load model ---
    model_path = os.path.join('model', 'model1_binary_recyclable.keras')
    try:
        model = load_model(model_path)
        print("Đã tải mô hình thành công")
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return
    
    # Thư mục test
    test_dir = 'Z:\\GarbageClassification\\data\\test'
    
    # Test với ảnh từ mỗi thư mục
    for class_name in ['recyclable', 'non_recyclable']:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(class_dir):
            continue
            
        print(f"\nKiểm tra ảnh trong thư mục: {class_name}")
        
        # Lấy 5 ảnh đầu tiên
        for img_name in os.listdir(class_dir)[:5]:
            img_path = os.path.join(class_dir, img_name)
            print(f"\nẢnh: {img_name}")
            print(f"Lớp thực tế: {class_name}")
            
            result = predict_single_image(model, img_path)
            if result:
                print(f"Dự đoán: {result['predicted_class']}")
                print(f"Độ tin cậy: {result['probability']*100:.2f}%")
                
                # Hiển thị ảnh
                img = cv2.imread(img_path)
                if img is not None:
                    color = (0, 255, 0) if result['predicted_class'] == 'Tái chế' else (0, 0, 255)
                    cv2.putText(img, f"{result['predicted_class']} ({result['probability']*100:.1f}%)",
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.imshow(f"Test - {img_name}", img)
                    cv2.waitKey(1000)  # Hiển thị 1 giây
                    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_model()
