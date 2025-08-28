# main.py
import io
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO

# --- CẤU HÌNH ---
# Khởi tạo ứng dụng FastAPI
app = FastAPI(title="YOLOv8 Inference API", description="API để chạy dự đoán với mô hình YOLOv8")
MODEL_PATH = 'C:\\Users\\Admin\\PycharmProjects\\pythonProject\\runs\\detect\\train4\\weights\\last.pt'  # <-- THAY ĐỔI ĐƯỜNG DẪN NÀY

try:
    model = YOLO(MODEL_PATH)
    print("Tải mô hình thành công!")
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    model = None


# --- API ENDPOINT ---
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    """
    Endpoint nhận một file ảnh và trả về kết quả nhận diện.
    - file: File ảnh được tải lên.
    """
    if not model:
        return {"error": "Mô hình không khả dụng, vui lòng kiểm tra lại đường dẫn model."}

    # Đọc nội dung file ảnh từ request
    contents = await file.read()

    # Chuyển đổi dữ liệu byte thành ảnh mà mô hình có thể xử lý
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        return {"error": f"Không thể đọc file ảnh: {e}"}

    # Chạy dự đoán với mô hình
    results = model(image)

    # Xử lý kết quả trả về
    # Lấy kết quả từ lần dự đoán đầu tiên (vì chúng ta chỉ xử lý 1 ảnh)
    result = results[0]

    # Chuẩn bị một list để chứa các đối tượng được phát hiện
    detections = []

    # Lặp qua các bounding box được phát hiện
    for box in result.boxes:
        # Lấy tọa độ của box (x1, y1, x2, y2)
        coords = box.xyxy[0].tolist()

        # Lấy class_id và độ tin cậy (confidence)
        class_id = int(box.cls[0].item())
        confidence = box.conf[0].item()

        # Lấy tên của lớp đối tượng từ model
        class_name = model.names[class_id]

        # Thêm thông tin của đối tượng vào list
        detections.append({
            "class_name": class_name,
            "confidence": round(confidence, 4),
            "bounding_box": {
                "x1": round(coords[0], 2),
                "y1": round(coords[1], 2),
                "x2": round(coords[2], 2),
                "y2": round(coords[3], 2),
            }
        })

    # Trả về kết quả dưới dạng JSON
    return {"detections": detections}

@app.get("/")
def read_root():
    return {"message": "Chào mừng đến với API nhận diện của YOLOv8!"}

