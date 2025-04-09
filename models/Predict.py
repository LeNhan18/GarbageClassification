# predict.py
# Dự đoán 1 ảnh: Tái chế / Không → Nếu tái chế thì phân loại chi tiết

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# --- Cấu hình ---
model1_path = 'models/model1_binary_recyclable.h5'
model2_path = 'models/model2_multiclass_recyclable.h5'
img_size_binary = (150, 150)
img_size_multiclass = (150, 150)  # dùng CNN nên giữ nguyên

# --- Load model ---
model1 = load_model(model1_path)
model2 = load_model(model2_path)

# --- Hàm dự đoán ---
def predict_image(img_path):
    img = image.load_img(img_path, target_size=img_size_binary)
    img_array = image.img_to_array(img) / 255.0
    img_array_expanded = np.expand_dims(img_array, axis=0)

    # Dự đoán tái chế / không tái chế
    pred_binary = model1.predict(img_array_expanded)[0][0]
    if pred_binary < 0.5:
        return "❌ Không tái chế"
    else:
        # Nếu là tái chế → phân loại chi tiết
        img_multiclass = image.load_img(img_path, target_size=img_size_multiclass)
        img_array_multi = image.img_to_array(img_multiclass) / 255.0
        img_array_multi = np.expand_dims(img_array_multi, axis=0)

        pred_multi = model2.predict(img_array_multi)
        class_index = np.argmax(pred_multi)
        class_labels = os.listdir('data/recyclable')
        class_labels.sort()  # đảm bảo đúng thứ tự
        return f"♻️ Tái chế: {class_labels[class_index]}"

# --- Dùng thử ---
if __name__ == '__main__':
    img_path = input("Nhập đường dẫn ảnh cần dự đoán: ")
    if not os.path.exists(img_path):
        print("Ảnh không tồn tại!")
    else:
        result = predict_image(img_path)
        print("\n🎯 Kết quả dự đoán:", result)