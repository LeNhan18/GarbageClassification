import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2
import time

def load_and_preprocess_image(image, target_size=(224, 224)):
    """Load and preprocess image for model input"""
    img = Image.fromarray(image)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(model, image):
    """Predict image using the model"""
    # Preprocess image
    img_array = load_and_preprocess_image(image)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    
    # Get class with highest probability
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    return predicted_class, confidence

def main():
    # Model path
    model_path = "Z:\\GarbageClassification\\models\\model_mobilenetv2_single_phase\\model_final.keras"
    
    if not os.path.exists(model_path):
        print(f"Model không tìm thấy tại {model_path}")
        return
    
    try:
        # Load model mà không cần metrics
        print("Đang tải model...")
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Compile lại model với metrics đơn giản
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        print("Đã tải model thành công!")
        
        # Khởi tạo camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Không thể mở camera!")
            return
            
        print("Nhấn 'q' để thoát")
        
        while True:
            # Đọc frame từ camera
            ret, frame = cap.read()
            if not ret:
                print("Không thể đọc frame từ camera!")
                break
                
            # Lật frame để hiển thị như gương
            frame = cv2.flip(frame, 1)
            
            # Thực hiện dự đoán
            predicted_class, confidence = predict_image(model, frame)
            
            # Hiển thị kết quả
            result_text = "Tái chế" if predicted_class == 1 else "Không tái chế"
            color = (0, 255, 0) if predicted_class == 1 else (0, 0, 255)
            
            # Vẽ kết quả lên frame
            cv2.putText(frame, f"{result_text}: {confidence:.2f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Hiển thị frame
            cv2.imshow('Phân loại rác', frame)
            
            # Thoát nếu nhấn 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Lỗi: {str(e)}")
    finally:
        # Giải phóng tài nguyên
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 