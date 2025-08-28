import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import argparse
import time
import os

def load_and_preprocess_frame(frame, target_size=(224, 224)):
    # Chuyển đổi frame từ BGR sang RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Resize frame
    frame_resized = cv2.resize(frame_rgb, target_size)
    # Chuyển đổi thành array và chuẩn hóa
    frame_array = np.expand_dims(frame_resized, axis=0)
    frame_array = frame_array / 255.0
    return frame_array

def predict_combined(model1, model2a, model2b, frame_array, class_names_2a, class_names_2b, threshold=0.5):
    # Model 1: Phân loại nhị phân
    pred1 = model1.predict(frame_array, verbose=0)
    is_recyclable = pred1[0][0] > threshold
    result = {}
    if is_recyclable:
        # Model 2A: Phân loại chi tiết rác tái chế
        pred2a = model2a.predict(frame_array, verbose=0)
        idx2a = np.argmax(pred2a[0])
        conf2a = pred2a[0][idx2a]
        result['type'] = 'Tái chế'
        result['detail'] = class_names_2a[idx2a]
        result['confidence'] = float(conf2a)
        result['all_probs'] = {name: float(prob) for name, prob in zip(class_names_2a, pred2a[0])}
    else:
        # Model 2B: Phân loại chi tiết rác không tái chế
        pred2b = model2b.predict(frame_array, verbose=0)
        idx2b = np.argmax(pred2b[0])
        conf2b = pred2b[0][idx2b]
        result['type'] = 'Không tái chế'
        result['detail'] = class_names_2b[idx2b]
        result['confidence'] = float(conf2b)
        result['all_probs'] = {name: float(prob) for name, prob in zip(class_names_2b, pred2b[0])}
    result['is_recyclable'] = is_recyclable
    result['binary_prob'] = float(pred1[0][0])
    return result

def draw_prediction(frame, result):
    # Tạo bản sao của frame để vẽ lên
    display_frame = frame.copy()
    
    # Xác định màu dựa trên loại rác
    color = (0, 255, 0) if result['is_recyclable'] else (0, 0, 255)
    
    # Vẽ thông tin dự đoán
    text1 = f"Loại: {result['type']} ({result['binary_prob']:.2%})"
    text2 = f"Chi tiết: {result['detail']} ({result['confidence']:.2%})"
    
    # Thêm text vào frame
    cv2.putText(display_frame, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(display_frame, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return display_frame

def main():
    parser = argparse.ArgumentParser(description='Dự đoán rác bằng camera')
    parser.add_argument('--model1_path', type=str, default='Z:\\GarbageClassification\\models_Model1_EfficientNetB2\\Model1_EfficientNetB2_BalancedTune\\20250527-011116\\model_Model1_EfficientNetB2_BalancedTune_final_from_best_20250527-011116.keras', help='Đường dẫn model 1 (nhị phân)')
    parser.add_argument('--model2a_path', type=str, default='Z:\\GarbageClassification\\models\\model_EfficientNetB2\\model2A_EfficientNetB2.keras', help='Đường dẫn model 2A (tái chế)')
    parser.add_argument('--model2b_path', type=str, default='Z:\\GarbageClassification\\models\\model_EfficientB2\\model2B_EfficientNetB2_final.keras', help='Đường dẫn model 2B (không tái chế)')
    parser.add_argument('--classes2a', type=str, nargs='+', default=['cardboard', 'glass', 'metal', 'paper', 'plastic'], help='Tên các lớp model 2A')
    parser.add_argument('--classes2b', type=str, nargs='+', default=['battery', 'biological', 'clothes', 'shoes', 'trash'], help='Tên các lớp model 2B')
    parser.add_argument('--camera', type=int, default=0, help='Chỉ số camera (mặc định: 0)')
    args = parser.parse_args()

    # Kiểm tra file model
    for path in [args.model1_path, args.model2a_path, args.model2b_path]:
        if not os.path.exists(path):
            print(f'Không tìm thấy: {path}')
            return

    # Load models
    print('Đang tải các model...')
    model1 = load_model(args.model1_path, compile=False)
    model2a = load_model(args.model2a_path, compile=False)
    model2b = load_model(args.model2b_path, compile=False)
    print('Đã tải xong các model!')

    # Khởi tạo camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Không thể mở camera!")
        return

    print("\nNhấn 'q' để thoát")
    print("Nhấn 's' để lưu ảnh")

    last_prediction_time = 0
    prediction_interval = 0.5  # Thời gian giữa các lần dự đoán (giây)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc frame từ camera!")
            break

        # Thực hiện dự đoán mỗi prediction_interval giây
        current_time = time.time()
        if current_time - last_prediction_time >= prediction_interval:
            frame_array = load_and_preprocess_frame(frame)
            result = predict_combined(model1, model2a, model2b, frame_array, args.classes2a, args.classes2b)
            last_prediction_time = current_time

        # Vẽ kết quả dự đoán lên frame
        display_frame = draw_prediction(frame, result)

        # Hiển thị frame
        cv2.imshow('Phân loại rác', display_frame)

        # Xử lý phím nhấn
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Lưu ảnh
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"prediction_{timestamp}.jpg"
            cv2.imwrite(filename, display_frame)
            print(f"Đã lưu ảnh: {filename}")

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 