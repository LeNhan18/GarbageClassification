import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img, img_array

def predict_combined(model1, model2a, model2b, img_array, class_names_2a, class_names_2b, threshold=0.5):
    # Model 1: Phân loại nhị phân
    pred1 = model1.predict(img_array)
    is_recyclable = pred1[0][0] > threshold
    result = {}
    if is_recyclable:
        # Model 2A: Phân loại chi tiết rác tái chế
        pred2a = model2a.predict(img_array)
        idx2a = np.argmax(pred2a[0])
        conf2a = pred2a[0][idx2a]
        result['type'] = 'Tái chế'
        result['detail'] = class_names_2a[idx2a]
        result['confidence'] = float(conf2a)
        result['all_probs'] = {name: float(prob) for name, prob in zip(class_names_2a, pred2a[0])}
    else:
        # Model 2B: Phân loại chi tiết rác không tái chế
        pred2b = model2b.predict(img_array)
        idx2b = np.argmax(pred2b[0])
        conf2b = pred2b[0][idx2b]
        result['type'] = 'Không tái chế'
        result['detail'] = class_names_2b[idx2b]
        result['confidence'] = float(conf2b)
        result['all_probs'] = {name: float(prob) for name, prob in zip(class_names_2b, pred2b[0])}
    result['is_recyclable'] = is_recyclable
    result['binary_prob'] = float(pred1[0][0])
    return result

def main():
    parser = argparse.ArgumentParser(description='Dự đoán kết hợp 3 model phân loại rác')
    parser.add_argument('--model1_path', type=str, default='Z:\\GarbageClassification\\models\\Model_Final\\Model1_EfficientNetB2.keras', help='Đường dẫn model 1 (nhị phân)')
    parser.add_argument('--model2a_path', type=str, default='Z:\\GarbageClassification\\models\\Model_Final\\model2A_EfficientNetB2.keras', help='Đường dẫn model 2A (tái chế)')
    parser.add_argument('--model2b_path', type=str, default='Z:\\GarbageClassification\\models\\Model_Final\\model2B_EfficientNetB2_final.keras', help='Đường dẫn model 2B (không tái chế)')
    parser.add_argument('--classes2a', type=str, nargs='+', default=['cardboard', 'glass', 'metal', 'paper', 'plastic'], help='Tên các lớp model 2A')
    parser.add_argument('--classes2b', type=str, nargs='+', default=['battery', 'biological', 'clothes', 'shoes', 'trash'], help='Tên các lớp model 2B')
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

    while True:
        # Yêu cầu người dùng nhập đường dẫn ảnh
        image_path = input('\nNhập đường dẫn ảnh cần dự đoán (hoặc nhập "q" để thoát): ').strip()
        
        if image_path.lower() == 'q':
            print('Đã thoát chương trình!')
            break
            
        if not os.path.exists(image_path):
            print(f'Không tìm thấy file ảnh: {image_path}')
            continue

        # Load và tiền xử lý ảnh
        img, img_array = load_and_preprocess_image(image_path)

        # Dự đoán
        result = predict_combined(model1, model2a, model2b, img_array, args.classes2a, args.classes2b)

        # Hiển thị kết quả
        print('\n===== KẾT QUẢ DỰ ĐOÁN =====')
        print(f"Loại rác tổng quát: {result['type']} (Xác suất: {result['binary_prob']:.2%})")
        print(f"Phân loại chi tiết: {result['detail']} (Độ tin cậy: {result['confidence']:.2%})")
        print('\nXác suất cho từng lớp chi tiết:')
        for name, prob in result['all_probs'].items():
            print(f"- {name}: {prob:.2%}")
        print('\n-----------------------------------\n')

        # Vẽ ảnh gốc và biểu đồ xác suất trên cùng một cửa sổ
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        # Hiển thị ảnh gốc
        ax1.imshow(img)
        ax1.axis('off')
        # Overlay loại rác tổng quát và xác suất nhị phân
        label_text = f"{result['type']} ({result['binary_prob']:.2%})"
        color = 'green' if result['is_recyclable'] else 'red'
        ax1.text(0.5, 0.05, label_text, fontsize=14, color=color, ha='center', va='bottom', transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.7, edgecolor=color))
        # Hiển thị biểu đồ xác suất
        class_names = list(result['all_probs'].keys())
        probs = list(result['all_probs'].values())
        ax2.barh(class_names, probs, color=color, alpha=0.7)
        ax2.set_xlabel('Xác suất')
        ax2.set_title('Xác suất cho mỗi lớp')
        ax2.set_xlim(0, 1)
        plt.tight_layout()
        plt.show()
if __name__ == '__main__':
    main() 