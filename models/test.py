# test_model1_camera.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json
import os

from TrainModel1 import train_generator

# --- Load model ---
model_path = 'model1_binary_recyclable.h5'  # hoặc model1_best.keras
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
    print("✅ Đã load ánh xạ lớp.")
except:
    # Nếu không có, bạn tự định nghĩa:
    class_mapping = {0: 'Non-Recyclable', 1: 'Recyclable'}  # hoặc ngược lại nếu model bạn học theo chiều ngược


# --- Hàm xử lý ảnh ---
def preprocess_image(img, target_size=(150, 150)):
    img_resized = cv2.resize(img, target_size)
    img_array = np.expand_dims(img_resized, axis=0)
    img_array = img_array / 255.0  # normalize giống lúc train
    return img_array


# --- Mở camera ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Không mở được camera.")
    exit()

print("🚀 Camera đã sẵn sàng. Bấm 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Vẽ khung giữa ảnh
    h, w, _ = frame.shape
    size = 224
    cx, cy = w // 2, h // 2
    x1, y1 = cx - size // 2, cy - size // 2
    x2, y2 = cx + size // 2, cy + size // 2

    roi = frame[y1:y2, x1:x2]  # Lấy ảnh trong khung giữa
    input_img = preprocess_image(roi, (150, 150))

    # Dự đoán
    prediction = model.predict(input_img)[0]

    # Xử lý kết quả
    class_index = np.argmax(prediction)
    confidence = float(prediction[class_index])
    label = class_mapping[class_index]

    # Hiển thị
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    text = f"{label} ({confidence * 100:.1f}%)"
    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Recycle Classifier (q to quit)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
