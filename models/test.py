import cv2
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("Z:\\GarbageClassification\\models\\models\\model1_binary_recyclable.h5")  # Đổi đúng đường dẫn model
input_size = (150, 150)

# Webcam setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Không thể mở webcam.")
    exit()

print("✅ Webcam đang hoạt động. Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Không thể đọc hình từ webcam.")
        break

    # Tiền xử lý ảnh
    img = cv2.resize(frame, input_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Dự đoán
    prediction = model.predict(img)
    class_name = "Recyclable" if prediction[0][0] > 0.5 else "Non-Recyclable"
    confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

    # Hiển thị kết quả lên màn hình
    label = f"{class_name} ({confidence*100:.1f}%)"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    cv2.imshow("Garbage Classification - Camera", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
