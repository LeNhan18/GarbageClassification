import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import time

def load_and_prepare_model():
    try:
        # Load model
        model = tf.keras.models.load_model('Z:\\GarbageClassification\\models\\models\\model1_binary_recyclable.h5')
        print("✅ Đã tải model thành công!")
        return model
    except Exception as e:
        print(f"❌ Lỗi khi tải model: {e}")
        return None

def preprocess_image(frame, target_size=(224, 224)):
    # Resize frame
    img = cv2.resize(frame, target_size)
    # Convert to array và normalize
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def draw_prediction(frame, prediction, fps):
    # Lấy kết quả dự đoán
    pred_value = float(prediction[0])  # Convert to float
    pred_class = "Rác tái chế" if pred_value > 0.5 else "Rác không tái chế"
    confidence = pred_value if pred_value > 0.5 else 1 - pred_value
    
    # Chọn màu dựa trên kết quả
    color = (0, 255, 0) if pred_class == "Rác tái chế" else (0, 0, 255)
    
    # Vẽ khung và thông tin
    cv2.putText(frame, f"Phan loai: {pred_class}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"Do tin cay: {confidence*100:.1f}%", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return frame

def main():
    # Tải model
    model = load_and_prepare_model()
    if model is None:
        return
    
    # Khởi tạo camera
    print("Đang khởi tạo camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Không thể mở camera!")
        return
    
    print("✅ Đã khởi tạo camera thành công!")
    print("Nhấn 'q' để thoát")
    
    # Biến đếm FPS
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    while True:
        # Đọc frame từ camera
        ret, frame = cap.read()
        if not ret:
            print("❌ Không thể đọc frame từ camera!")
            break
        
        # Xử lý frame
        processed_frame = preprocess_image(frame)
        
        # Dự đoán
        prediction = model.predict(processed_frame, verbose=0)
        
        # Tính FPS
        frame_count += 1
        if frame_count >= 30:
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = time.time()
        
        # Vẽ kết quả lên frame
        frame = draw_prediction(frame, prediction, fps)
        
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