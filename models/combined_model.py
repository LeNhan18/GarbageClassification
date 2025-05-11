import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import Metric
import time


# Define and register the custom F1Score metric class
@tf.keras.utils.register_keras_serializable(package="Custom")
class F1ScoreWithReshape(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', threshold=0.5, **kwargs):
        super(F1ScoreWithReshape, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.precision = self.add_weight(name='precision', initializer='zeros')
        self.recall = self.add_weight(name='recall', initializer='zeros')
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Reshape if needed
        if len(y_true.shape) > 1 and y_true.shape[-1] > 1:
            y_true = tf.argmax(y_true, axis=-1)
        if len(y_pred.shape) > 1 and y_pred.shape[-1] > 1:
            y_pred = tf.argmax(y_pred, axis=-1)
        else:
            y_pred = tf.cast(tf.greater_equal(y_pred, self.threshold), tf.float32)

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Calculate true positives, false positives, false negatives
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))

        # Update state variables
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

        # Calculate precision and recall
        precision = tp / (tp + fp + K.epsilon())
        recall = tp / (tp + fn + K.epsilon())

        self.precision.assign(precision)
        self.recall.assign(recall)

    def result(self):
        precision = self.precision
        recall = self.recall
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    def reset_state(self):
        self.true_positives.assign(0.)
        self.false_positives.assign(0.)
        self.false_negatives.assign(0.)
        self.precision.assign(0.)
        self.recall.assign(0.)

    def get_config(self):
        config = super(F1ScoreWithReshape, self).get_config()
        config.update({"threshold": self.threshold})
        return config


def load_models():
    try:
        # Load models with custom_objects to handle the F1Score metric
        custom_objects = {"F1ScoreWithReshape": F1ScoreWithReshape}

        # Thêm compile=False để tránh lỗi khi tải model
        model1 = tf.keras.models.load_model('Z:\\GarbageClassification\\models\\model\\model_final.keras',
                                            custom_objects=custom_objects,
                                            compile=False)
        model2a = tf.keras.models.load_model('Z:\\GarbageClassification\\models\\model\\model2a_final.keras',
                                             custom_objects=custom_objects,
                                             compile=False)
        model2b = tf.keras.models.load_model('Z:\\GarbageClassification\\models\\model\\model2B_final.keras',
                                             custom_objects=custom_objects,
                                             compile=False)
        print("Đã tải tất cả models thành công!")
        return model1, model2a, model2b
    except Exception as e:
        print(f"Lỗi khi tải models: {e}")
        return None, None, None


def preprocess_image(frame, target_size=(240, 240)):
    # Resize frame
    img = cv2.resize(frame, target_size)

    # Chuyển từ BGR sang RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Cải thiện độ tương phản
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    # Giảm nhiễu nhẹ hơn
    img = cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 5, 15)

    # Tăng độ sắc nét
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel)

    # Chuẩn hóa
    img_array = img_to_array(img)
    img_array = img_array / 255.0

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def get_class_name(class_idx, is_recyclable):
    if is_recyclable:
        classes = {
            0: "Giấy (paper)",
            1: "Nhựa (plastic)",
            2: "Thủy tinh (glass)",
            3: "Kim loại (metal)",
            4: "Bìa cứng (cardboard)"
        }
    else:
        classes = {
            0: "Pin (battery)",
            1: "Rác sinh học (biological)",
            2: "Quần áo (clothes)",
            3: "Giày dép (shoes)",
            4: "Rác vụn (trash)"
        }
    return classes.get(class_idx, "Không xác định")


def detect_objects(frame):
    # Resize frame về kích thước 240x240
    frame = cv2.resize(frame, (240, 240))
    
    # Chuyển đổi frame sang grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Cải thiện độ tương phản
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Áp dụng Gaussian blur để giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Phát hiện cạnh bằng Canny với ngưỡng thích ứng
    edges = cv2.Canny(blurred, 20, 150)

    # Mở rộng cạnh để kết nối các cạnh gần nhau
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Tìm contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lọc contours theo diện tích và tỷ lệ khung hình
    min_area = 3000  # Giảm diện tích tối thiểu
    max_area = 200000  # Tăng diện tích tối đa
    valid_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h
            # Lọc theo tỷ lệ khung hình
            if 0.1 < aspect_ratio < 10:  # Mở rộng tỷ lệ cho phép
                valid_contours.append(contour)

    # Tìm bounding boxes
    boxes = []
    for contour in valid_contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Thêm padding
        padding = 15  # Tăng padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2*padding)
        h = min(frame.shape[0] - y, h + 2*padding)
        boxes.append((x, y, w, h))

    return boxes


def draw_prediction(frame, model1_pred, model2_pred, is_recyclable, fps, boxes=None):
    # Xử lý kết quả từ model1
    pred_value = float(model1_pred[0][0]) if isinstance(model1_pred[0], np.ndarray) else float(model1_pred[0])
    recyclable = pred_value > 0.5
    confidence1 = pred_value if recyclable else 1 - pred_value

    # Xử lý kết quả từ model2
    class_idx = np.argmax(model2_pred[0])
    confidence2 = float(model2_pred[0][class_idx])
    class_name = get_class_name(class_idx, recyclable)

    # Tăng ngưỡng độ tin cậy
    min_confidence = 0.75  # Tăng ngưỡng độ tin cậy

    if confidence1 >= min_confidence and confidence2 >= min_confidence:
        # Chọn màu dựa trên kết quả
        color = (0, 255, 0) if recyclable else (0, 0, 255)

        # Vẽ bounding boxes nếu có
        if boxes:
            for box in boxes:
                x, y, w, h = box
                # Vẽ khung
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # Vẽ nhãn với nền
                label = f"{class_name} ({confidence2*100:.1f}%)"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x, y - label_h - 10), (x + label_w, y), color, -1)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Vẽ thông tin tổng quan
        cv2.putText(frame, f"Loai rac: {'Tái chế' if recyclable else 'Không tái chế'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Phan loai: {class_name}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Do tin cay: {confidence1 * 100:.1f}%",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Phan loai chi tiet: {confidence2 * 100:.1f}%",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    else:
        # Hiển thị thông báo khi độ tin cậy thấp
        cv2.putText(frame, "Khong du doan duoc", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)

    # Hiển thị FPS
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

        # Phát hiện vật thể
        boxes = detect_objects(frame)

        # Xử lý frame
        processed_frame = preprocess_image(frame)

        try:
            # Dự đoán với model1
            model1_pred = model1.predict(processed_frame, verbose=0)
            pred_value = float(model1_pred[0][0]) if len(model1_pred.shape) > 1 and model1_pred.shape[1] > 1 else float(model1_pred[0])
            is_recyclable = pred_value > 0.5

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
            frame = draw_prediction(frame, model1_pred, model2_pred, is_recyclable, fps, boxes)

        except Exception as e:
            print(f"Lỗi khi dự đoán: {e}")
            cv2.putText(frame, f"Loi: {str(e)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Hiển thị frame
        cv2.imshow('Phan loai rac', frame)

        # Thoát nếu nhấn 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()
    print(" Đã đóng camera và kết thúc chương trình!")


if __name__ == "__main__":
    main()