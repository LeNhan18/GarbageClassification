import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow.keras.backend as K
import time
import os


# Define the custom F1Score metric class
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


def load_models(model_paths):
    """
    Tải các mô hình từ đường dẫn được cung cấp
    """
    try:
        # Load models with custom_objects to handle the F1Score metric
        custom_objects = {"F1ScoreWithReshape": F1ScoreWithReshape}

        model1 = tf.keras.models.load_model(model_paths['model1'],
                                            custom_objects=custom_objects,
                                            compile=False)
        model2a = tf.keras.models.load_model(model_paths['model2a'],
                                             custom_objects=custom_objects,
                                             compile=False)
        model2b = tf.keras.models.load_model(model_paths['model2b'],
                                             custom_objects=custom_objects,
                                             compile=False)
        print("Đã tải tất cả models thành công!")
        return model1, model2a, model2b
    except Exception as e:
        print(f"Lỗi khi tải models: {e}")
        return None, None, None


def preprocess_image(frame, target_size=(240, 240)):
    """
    Tiền xử lý hình ảnh cho mô hình
    """
    # Resize frame
    img = cv2.resize(frame, target_size)
    # Chuyển từ BGR sang RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert to array và normalize
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def get_class_name(class_idx, is_recyclable):
    """
    Chuyển đổi chỉ số lớp thành tên lớp
    """
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


def draw_prediction(frame, model1_pred, model2_pred, is_recyclable, fps=None):
    """
    Vẽ kết quả dự đoán lên frame
    """
    # Xử lý kết quả từ model1
    pred_value = float(model1_pred[0][0]) if isinstance(model1_pred[0], np.ndarray) else float(model1_pred[0])
    recyclable = pred_value > 0.5
    confidence1 = pred_value if recyclable else 1 - pred_value

    # Xử lý kết quả từ model2
    class_idx = np.argmax(model2_pred[0])
    confidence2 = float(model2_pred[0][class_idx])
    class_name = get_class_name(class_idx, recyclable)

    # Chọn màu dựa trên kết quả
    color = (0, 255, 0) if recyclable else (0, 0, 255)

    # Vẽ thông tin lên frame
    cv2.putText(frame, f"Loai rac: {'Tái chế' if recyclable else 'Không tái chế'}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"Phan loai: {class_name}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"Do tin cay: {confidence1 * 100:.1f}%",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"Phan loai chi tiet: {confidence2 * 100:.1f}%",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return frame


def predict_from_image(image_path, model_paths):
    """
    Dự đoán từ một hình ảnh
    """
    # Tải các mô hình
    model1, model2a, model2b = load_models(model_paths)
    if None in (model1, model2a, model2b):
        print("Không thể tải mô hình!")
        return

    # Đọc hình ảnh
    try:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Không thể đọc hình ảnh từ đường dẫn: {image_path}")
            return
    except Exception as e:
        print(f"Lỗi khi đọc hình ảnh: {e}")
        return

    # Xử lý hình ảnh
    processed_frame = preprocess_image(frame)

    try:
        # Dự đoán với model1
        model1_pred = model1.predict(processed_frame, verbose=0)

        # Xác định loại rác
        pred_value = float(model1_pred[0][0]) if len(model1_pred.shape) > 1 and model1_pred.shape[1] > 1 else float(
            model1_pred[0])
        is_recyclable = pred_value > 0.5

        # Dự đoán với model2 tương ứng
        model2_pred = model2a.predict(processed_frame, verbose=0) if is_recyclable else model2b.predict(processed_frame,
                                                                                                        verbose=0)

        # Vẽ kết quả lên frame
        result_frame = draw_prediction(frame.copy(), model1_pred, model2_pred, is_recyclable)

        # Hiển thị hình ảnh
        cv2.imshow('Kết quả phân loại', result_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return result_frame

    except Exception as e:
        print(f"Lỗi khi dự đoán: {e}")
        return None


def create_empty_frame(width=640, height=480):
    """
    Tạo một frame trống với kích thước cho trước
    """
    # Tạo frame trống màu trắng
    frame = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Thêm văn bản giải thích
    cv2.putText(frame, "Frame trong (không có đối tượng)",
                (width // 2 - 150, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    return frame


def predict_from_camera(model_paths, camera_index=0, duration=10):
    """
    Dự đoán từ camera trong một khoảng thời gian
    """
    # Tải các mô hình
    model1, model2a, model2b = load_models(model_paths)
    if None in (model1, model2a, model2b):
        print("Không thể tải mô hình!")
        return

    # Khởi tạo camera
    print("Đang khởi tạo camera...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Không thể mở camera!")
        return

    print("Đã khởi tạo camera thành công!")
    print(f"Phân loại rác trong {duration} giây...")

    # Biến đếm FPS
    fps = 0
    frame_count = 0
    start_time = time.time()
    end_time = start_time + duration

    while time.time() < end_time:
        # Đọc frame từ camera
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc frame từ camera!")
            break

        # Xử lý frame
        processed_frame = preprocess_image(frame)

        try:
            # Dự đoán với model1
            model1_pred = model1.predict(processed_frame, verbose=0)

            # Xác định loại rác
            pred_value = float(model1_pred[0][0]) if len(model1_pred.shape) > 1 and model1_pred.shape[1] > 1 else float(
                model1_pred[0])
            is_recyclable = pred_value > 0.5

            # Dự đoán với model2 tương ứng
            model2_pred = model2a.predict(processed_frame, verbose=0) if is_recyclable else model2b.predict(
                processed_frame, verbose=0)

            # Tính FPS
            frame_count += 1
            current_time = time.time()
            if current_time - start_time >= 1:
                fps = frame_count / (current_time - start_time)
                frame_count = 0
                start_time = current_time

            # Vẽ kết quả lên frame
            result_frame = draw_prediction(frame, model1_pred, model2_pred, is_recyclable, fps)

            # Hiển thị frame
            cv2.imshow('Phân loại rác', result_frame)

        except Exception as e:
            print(f"Lỗi khi dự đoán: {e}")
            cv2.putText(frame, f"Lỗi: {str(e)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow('Phân loại rác', frame)

        # Thoát nếu nhấn 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Đã đóng camera và kết thúc chương trình!")


def main():
    # Đường dẫn đến các mô hình (cần được cập nhật)
    model_paths = {
        'model1': 'Z:\\GarbageClassification\\models\\model\\model1_final.keras',
        'model2a': 'Z:\\GarbageClassification\\models\\model\\model2a_final.keras',
        'model2b': 'Z:\\GarbageClassification\\models\\model\\model2b_final.keras'
    }

    # Hiển thị menu
    print("===== CHƯƠNG TRÌNH PHÂN LOẠI RÁC =====")
    print("1. Dự đoán từ camera")
    print("2. Dự đoán từ hình ảnh")
    print("3. Tạo frame trống")
    print("4. Thoát")

    choice = input("Chọn chức năng (1-4): ")

    if choice == '1':
        try:
            camera_index = int(input("Nhập chỉ số camera (mặc định 0): ") or 0)
            duration = int(input("Nhập thời gian dự đoán (giây, mặc định 10): ") or 10)
            predict_from_camera(model_paths, camera_index, duration)
        except ValueError:
            print("Vui lòng nhập số hợp lệ!")

    elif choice == '2':
        image_path = input("Nhập đường dẫn đến hình ảnh: ")
        if os.path.exists(image_path):
            predict_from_image(image_path, model_paths)
        else:
            print(f"Không tìm thấy hình ảnh tại: {image_path}")

    elif choice == '3':
        try:
            width = int(input("Nhập chiều rộng frame (mặc định 640): ") or 640)
            height = int(input("Nhập chiều cao frame (mặc định 480): ") or 480)
            frame = create_empty_frame(width, height)
            cv2.imshow("Frame trống", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except ValueError:
            print("Vui lòng nhập số hợp lệ!")

    elif choice == '4':
        print("Cảm ơn bạn đã sử dụng chương trình!")

    else:
        print("Lựa chọn không hợp lệ!")


if __name__ == "__main__":
    main()