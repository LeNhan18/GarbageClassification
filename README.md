git clone https://github.com/LeNhan18/GarbageClassification.git
# README.md

## ♻️ Hệ thống phân loại rác bằng 2 mô hình CNN (Python + Flutter)

### 📁 Cấu trúc thư mục dữ liệu
```
data/
├── binary/               # Dùng cho model1
│   ├── recyclable/
│   └── non_recyclable/
└── recyclable/           # Dùng cho model2a
    ├── plastic/
    ├── paper/
    ├── metal/
    └── glass/
    └── cardboard/
└── non_recyclable/       # Dùng cho model2b
    ├── trash/
    ├── battery/
    ├── clothes/
    └── shoes/
    └── biological/
```

---

### 📦 Các file Python chính

| File name         | Chức năng                                      |
|-------------------|------------------------------------------------|
| `train_model1.py` | Phân loại Tái chế / Không Tái chế (binary)    |
| `train_model2a.py` | Phân loại chi tiết các loại rác tái chế (CNN) |
| `train_model2b.py` | Phân loại chi tiết các loại rác không tái chế (CNN) |
| `predict.py`      | Dự đoán ảnh đầu vào qua 2 bước                 |
| `utils.py`        | Hỗ trợ rename, resize, xử lý thư mục ảnh       |
| `convert_tflite.py` | Convert model `.h5` sang `.tflite` để dùng Flutter |

---

### ⚙️ Cài đặt môi trường (Python >= 3.8)
```bash
pip install tensorflow pillow
```

---

### 🧠 Huấn luyện mô hình
#### 1. Mô hình phân loại tái chế / không tái chế (Binary)
```bash
python train_model1.py
```
#### 2. Mô hình phân loại chi tiết các loại rác tái chế (Multi-class)
```bash
python train_model2a.py
```
#### 3. Mô hình phân loại chi tiết các loại rác không tái chế (Multi-class)
```bash
python train_model2b.py
```
> Kết quả sẽ được lưu vào thư mục `models/`

---

### 🔄 Convert mô hình sang TensorFlow Lite để dùng Flutter
```bash
python convert_tflite.py
```
> Kết quả: `assets/model1.tflite` và `assets/model2.tflite`

---

### 📱 Kết nối với Flutter App
#### Các bước:
1. Thêm thư viện:
```yaml
dependencies:
  tflite_flutter: ^0.10.4
  image_picker: ^1.0.4
```
2. Đặt model `.tflite` vào `assets/` và khai báo trong `pubspec.yaml`
3. Sử dụng `Interpreter` từ `tflite_flutter` để load và chạy model
4. Resize ảnh, đưa vào mô hình để lấy dự đoán

👉 *Chi tiết mã Flutter sẽ được viết ở thư mục `flutter_app/`*

---

### 🛠️ Xử lý ảnh: Đổi tên, resize
```bash
python utils.py
```
> Nhập đường dẫn thư mục chứa nhiều lớp con (như `data/recyclable` hoặc `data/binary/recyclable`...)

---

### ✅ Gợi ý dữ liệu
- Tên folder ảnh không được đặt tiếng Việt có dấu
- Các ảnh nên resize về 150x150
- Nên có >200 ảnh mỗi lớp để mô hình hoạt động tốt

---
