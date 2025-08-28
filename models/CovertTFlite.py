import tensorflow as tf
import os # Thêm os để lấy kích thước file (tùy chọn)

# --- Cấu hình ---
# Đường dẫn đến model Keras đã lưu của bạn
keras_model_path = 'Z:\\GarbageClassification\\models_Model1_EfficientNetB2\\Model1_EfficientNetB2_BalancedTune\\20250527-011116\\model_Model1_EfficientNetB2_BalancedTune_final_from_best_20250527-011116.keras'
# Đường dẫn bạn muốn lưu model TFLite
tflite_model_path = 'Z:\\GarbageClassification\\src\\common\\RecycleTrashApp\\Assets\\ML_Models\\Model1.tflite'

try:
    # 1. Tải model Keras
    print(f"Đang tải model Keras từ: {keras_model_path}")
    model = tf.keras.models.load_model(keras_model_path)
    print("Tải model Keras thành công.")
    # 2. Khởi tạo TFLiteConverter từ model Keras
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # 3. (Tùy chọn) Áp dụng các tối ưu hóa
    # Ví dụ: Tối ưu hóa mặc định (có thể bao gồm dynamic range quantization)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Ví dụ: Quantization sang Float16 (giảm kích thước, tốt cho GPU, ít mất độ chính xác)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
    # print("Đã áp dụng tối ưu hóa Float16.")

    # 4. Chuyển đổi model
    print("Đang chuyển đổi model sang TFLite...")
    tflite_model_content = converter.convert() # Đổi tên biến để tránh nhầm lẫn với module
    print("Chuyển đổi TFLite thành công.")

    # 5. Lưu model TFLite
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model_content)
    print(f"Model TFLite đã được lưu tại: {tflite_model_path}")
    if os.path.exists(tflite_model_path):
        print(f"Kích thước file TFLite: {os.path.getsize(tflite_model_path)/1024:.2f} KB")
    else:
        print("Không thể tìm thấy file TFLite đã lưu để kiểm tra kích thước.")

except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file model Keras tại '{keras_model_path}'. Vui lòng kiểm tra lại đường dẫn.")
except Exception as e:
    print(f"Đã xảy ra lỗi trong quá trình chuyển đổi: {e}")