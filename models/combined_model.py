import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import time

def load_models():
    try:
        # Load cả 3 model
        model1 = tf.keras.models.load_model('Z:\\GarbageClassification\\models\\model\\model1_best.keras')
        model2a = tf.keras.models.load_model('Z:\\GarbageClassification\\models\\model\\model2a_best.keras')
        model2b = tf.keras.models.load_model('Z:\\GarbageClassification\\models\\model\\model2b_best.keras')
        print("Đã tải tất cả models thành công!")
        return model1, model2a, model2b
    except Exception as e:
        print(f"Lỗi khi tải models: {e}")
        return None, None, None

def preprocess_image(frame, target_size=(224, 224)):
    # Resize frame
    img = cv2.resize(frame, target_size)
    # Convert to array và normalize
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_class_name(class_idx, is_recyclable):
    if is_recyclable:
        classes = {
            0: "Giấy",
            1: "Nhựa",
            2: "Thủy tinh",
            3: "Kim loại"
        }
    else:
        classes = {
            0: "Rác hữu cơ",
            1: "Rác vô cơ",
            2: "Rác nguy hại"
        }
    return classes.get(class_idx, "Không xác định")

def draw_prediction(frame, model1_pred, model2_pred, is_recyclable, fps):
    # Xử lý kết quả từ model1
    pred_value = float(model1_pred[0])
    recyclable = pred_value > 0.5
    confidence1 = pred_value if recyclable else 1 - pred_value
    
    # Xử lý kết quả từ model2
    class_idx = np.argmax(model2_pred[0])
    confidence2 = float(model2_pred[0][class_idx])
    class_name = get_class_name(class_idx, is_recyclable)
    
    # Chọn màu dựa trên kết quả
    color = (0, 255, 0) if recyclable else (0, 0, 255)
    
    # Vẽ thông tin lên frame
    cv2.putText(frame, f"Loai rac: {'Tái chế' if recyclable else 'Không tái chế'}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"Phan loai: {class_name}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"Do tin cay: {confidence1*100:.1f}%", 
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"Phan loai chi tiet: {confidence2*100:.1f}%", 
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", 
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return frame

def main():
    # Tải models
    model1, model2a, model2b = load_models()
    if None in (model1, model2a, model2b):
        return
    
    # Khởi tạo camera
    print("Đang khởi tạo camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở camera!")
        return
    
    print("Đã khởi tạo camera thành công!")
    print("Nhấn 'q' để thoát")
    
    # Biến đếm FPS
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    while True:
        # Đọc frame từ camera
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc frame từ camera!")
            break
        
        # Xử lý frame
        processed_frame = preprocess_image(frame)
        
        # Dự đoán với model1
        model1_pred = model1.predict(processed_frame, verbose=0)
        is_recyclable = float(model1_pred[0]) > 0.5
        
        # Dự đoán với model2 tương ứng
        model2_pred = model2a.predict(processed_frame, verbose=0) if is_recyclable else model2b.predict(processed_frame, verbose=0)
        
        # Tính FPS
        frame_count += 1
        if frame_count >= 30:
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = time.time()
        
        # Vẽ kết quả lên frame
        frame = draw_prediction(frame, model1_pred, model2_pred, is_recyclable, fps)
        
        # Hiển thị frame
        cv2.imshow('Phan loai rac', frame)
        
        # Thoát nếu nhấn 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Đã đóng camera và kết thúc chương trình!")

if __name__ == "__main__":
    main() 