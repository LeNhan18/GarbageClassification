import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Cấu hình GPU
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Cấu hình GPU thành công")
    else:
        print("Không tìm thấy GPU")
except Exception as e:
    print(f"Lỗi cấu hình GPU: {e}")

# Đường dẫn model
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_EfficientNetB2','model2A_EfficientNetB2.keras')
IMG_SIZE = (224, 224)

# Danh sách các lớp rác không tái chế
CLASS_NAMES = [
    'cardboard' , 'glass','metal','paper','plastic'
]

def load_custom_model():
    """Load model"""
    print("Đang load model...")
    try:
        model = load_model(MODEL_PATH)
        print("Load model thành công!")
        return model
    except Exception as e:
        print(f"Lỗi khi load model: {str(e)}")
        return None
def preprocess_image(image_path):
    """Tiền xử lý ảnh"""
    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
    
    # Resize ảnh
    img = cv2.resize(img, IMG_SIZE)
    
    # Chuyển từ BGR sang RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Chuẩn hóa
    img = img / 255.0
    
    # Thêm batch dimension
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(model, image_path):
    """Dự đoán ảnh"""
    try:
        # Tiền xử lý ảnh
        processed_img = preprocess_image(image_path)
        
        # Dự đoán
        predictions = model.predict(processed_img, verbose=0)
        
        # Lấy kết quả dự đoán
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        class_name = CLASS_NAMES[class_idx]
        
        return {
            'class': class_name,
            'confidence': float(confidence),
            'all_predictions': {name: float(pred) for name, pred in zip(CLASS_NAMES, predictions[0])}
        }
    except Exception as e:
        print(f"Lỗi khi dự đoán: {str(e)}")
        return None

def main():
    # Load model
    model = load_custom_model()
    if model is None:
        return

    # Đường dẫn ảnh cần dự đoán
    image_path = input("Nhập đường dẫn ảnh cần dự đoán: ")
    
    # Kiểm tra file tồn tại
    if not os.path.exists(image_path):
        print("Đường dẫn ảnh không tồn tại")
    # Dự đoán
    result = predict_image(model, image_path)
    
    if result:
        print("\nKết quả dự đoán:")
        print(f"Loại rác: {result['class']}")
        print(f"Độ tin cậy: {result['confidence']*100:.2f}%")
        print("\nChi tiết các lớp:")
        for class_name, confidence in result['all_predictions'].items():
            print(f"{class_name}: {confidence*100:.2f}%")

if __name__ == "__main__":
    main() 