import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json
import os

def test_model2():
    print("=== BẮT ĐẦU KIỂM TRA MODEL 2 (PHÂN LOẠI ĐA LỚP) ===")
    
    # --- Load model và mapping ---
    model_path = 'model2_multiclass_recyclable.keras'
    mapping_path = 'class_mapping.json'
    
    try:
        model = load_model(model_path)
        print("✅ Đã tải mô hình thành công")
        
        with open(mapping_path, 'r') as f:
            class_mapping = json.load(f)
        print("✅ Đã tải mapping thành công")
    except Exception as e:
        print(f"❌ Lỗi khi tải mô hình hoặc mapping: {e}")
        return

    # --- Mở camera ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Không thể mở camera")
        return

    print("🚀 Camera đã sẵn sàng. Bấm 'q' để thoát.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Không thể đọc frame từ camera")
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
        label = class_mapping[str(class_index)]
        
        # Vẽ khung và kết quả
        color = (0, 255, 0)  # Màu xanh lá cho tất cả các loại rác tái chế
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Hiển thị top 3 dự đoán
        top3_indices = np.argsort(prediction)[-3:][::-1]
        for i, idx in enumerate(top3_indices):
            prob = float(prediction[idx])
            class_name = class_mapping[str(idx)]
            y_pos = y1 - 10 - (i * 20)
            cv2.putText(frame, f"{class_name}: {prob*100:.1f}%", 
                       (x1, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.imshow('Phân Loại Rác Tái Chế', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Đã đóng camera và kết thúc chương trình")

if __name__ == '__main__':
    test_model2() 