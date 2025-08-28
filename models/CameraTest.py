import os
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from starlette.responses import JSONResponse

# --- CẤU HÌNH ---
# Thay đổi các đường dẫn và tên lớp nếu cần
MODEL1_PATH = 'Z:\\GarbageClassification\\models\\Model_Final\\Model1_EfficientNetB2.keras'
MODEL2A_PATH = 'Z:\\GarbageClassification\\models\\Model_Final\\model2A_EfficientNetB2.keras'
MODEL2B_PATH = 'Z:\\GarbageClassification\\models\\Model_Final\\model2B_EfficientNetB2_final.keras'
CLASS_NAMES_2A = ['cardboard', 'glass', 'metal', 'paper', 'plastic']
CLASS_NAMES_2B = ['battery', 'biological', 'clothes', 'shoes', 'trash']

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="Garbage Classification API",
    description="API để phân loại rác thải thành rác tái chế và không tái chế, sau đó phân loại chi tiết.",
    version="1.0.0"
)

# Biến toàn cục để lưu các model đã tải
models = {}


# --- HÀM TẢI MODEL KHI SERVER KHỞI ĐỘNG ---
@app.on_event("startup")
async def load_models():
    """
    Tải các model vào bộ nhớ khi ứng dụng FastAPI khởi động.
    """
    print("Đang tải các model...")
    try:
        models['model1'] = load_model(MODEL1_PATH, compile=False)
        models['model2a'] = load_model(MODEL2A_PATH, compile=False)
        models['model2b'] = load_model(MODEL2B_PATH, compile=False)
        print("Đã tải xong các model!")
    except Exception as e:
        print(f"Lỗi khi tải model: {e}")
        # Nếu không tải được model, API không thể hoạt động
        raise RuntimeError(f"Không thể tải model. Lỗi: {e}")


# --- CÁC HÀM HỖ TRỢ ---
def read_and_preprocess_image(file_bytes: bytes, target_size=(224, 224)):
    """
    Đọc file ảnh từ bytes, tiền xử lý và chuẩn bị cho model.
    """
    # Mở ảnh từ bytes
    img = Image.open(io.BytesIO(file_bytes))
    # Resize nếu cần và chuyển thành mảng numpy
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Chuẩn hóa ảnh
    img_array = img_array / 255.0
    return img_array


def predict_combined(img_array, threshold=0.5):
    """
    Thực hiện dự đoán kết hợp 3 model.
    """
    # Lấy model từ biến toàn cục
    model1 = models['model1']
    model2a = models['model2a']
    model2b = models['model2b']

    # Model 1: Phân loại nhị phân
    pred1 = model1.predict(img_array)
    binary_prob = float(pred1[0][0])
    is_recyclable = binary_prob > threshold

    result = {}
    if is_recyclable:
        # Model 2A: Phân loại chi tiết rác tái chế
        pred2a = model2a.predict(img_array)
        idx2a = np.argmax(pred2a[0])
        conf2a = pred2a[0][idx2a]
        result['type'] = 'Tái chế'
        result['detail'] = CLASS_NAMES_2A[idx2a]
        result['confidence'] = float(conf2a)
        result['all_probs'] = {name: float(prob) for name, prob in zip(CLASS_NAMES_2A, pred2a[0])}
    else:
        # Model 2B: Phân loại chi tiết rác không tái chế
        pred2b = model2b.predict(img_array)
        idx2b = np.argmax(pred2b[0])
        conf2b = pred2b[0][idx2b]
        result['type'] = 'Không tái chế'
        result['detail'] = CLASS_NAMES_2B[idx2b]
        result['confidence'] = float(conf2b)
        result['all_probs'] = {name: float(prob) for name, prob in zip(CLASS_NAMES_2B, pred2b[0])}

    result['is_recyclable'] = is_recyclable
    result['binary_prob'] = binary_prob
    return result


# --- ENDPOINT CỦA API ---
@app.post("/predict/", summary="Phân loại ảnh rác thải")
async def predict_image(file: UploadFile = File(...)):
    """
    Nhận một file ảnh, thực hiện phân loại và trả về kết quả.

    - **file**: File ảnh cần phân loại (định dạng: jpeg, png, ...).
    """
    # Kiểm tra xem model đã được tải chưa
    if not all(k in models for k in ['model1', 'model2a', 'model2b']):
        raise HTTPException(status_code=503,
                            detail="Các model đang được tải hoặc tải thất bại. Vui lòng thử lại sau giây lát.")

    # Đọc nội dung file ảnh
    file_bytes = await file.read()

    try:
        # Tiền xử lý ảnh
        img_array = read_and_preprocess_image(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không thể xử lý file ảnh. Lỗi: {e}")

    # Dự đoán
    try:
        prediction_result = predict_combined(img_array)
        return JSONResponse(content=prediction_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi trong quá trình dự đoán. Lỗi: {e}")


# --- LỆNH ĐỂ CHẠY SERVER (DÀNH CHO VIỆC DEBUG) ---
if __name__ == '__main__':
    # Chạy server API trên cổng 8000
    # Dùng lệnh `uvicorn main.api:app --reload` trong terminal để có tính năng tự động tải lại khi code thay đổi
    uvicorn.run(app, host="127.0.0.1", port=8000)