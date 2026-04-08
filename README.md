<div align="center">
  <img src="Image/logo.jpg" alt="Garbage Classification Banner" width="100%"/>
</div>

# Garbage Classification System

Hệ thống phân loại rác bằng thị giác máy tính, kết hợp:
- YOLOv8 cho phát hiện đối tượng
- EfficientNetB2 cho phân loại theo pipeline nhiều tầng
- FastAPI cho backend inference
- Flutter cho ứng dụng di động đa nền tảng

## Muc Luc

- [1. Tong Quan](#1-tong-quan)
- [2. Kien Truc He Thong](#2-kien-truc-he-thong)
- [3. Cong Nghe Su Dung](#3-cong-nghe-su-dung)
- [4. Cau Truc Thu Muc](#4-cau-truc-thu-muc)
- [5. Cai Dat](#5-cai-dat)
- [6. Chay Ung Dung](#6-chay-ung-dung)
- [7. API Endpoints](#7-api-endpoints)
- [8. Mo Hinh Da Huấn Luyen](#8-mo-hinh-da-huan-luyen)
- [9. Hinh Anh Minh Hoa](#9-hinh-anh-minh-hoa)
- [10. Huong Dan Luyen Mo Hinh](#10-huong-dan-luyen-mo-hinh)
- [11. Ung Dung Flutter](#11-ung-dung-flutter)
- [12. Dong Gop](#12-dong-gop)
- [13. License](#13-license)

## 1. Tong Quan

Du an huong den bai toan phan loai rac trong thuc te voi 2 giai doan:

1. Phan loai nhi phan: recyclable vs non_recyclable
2. Phan loai chi tiet theo nhom con

Muc tieu:
- Tang toc do phan loai anh
- Ho tro tich hop backend va mobile
- De mo rong them nhom rac moi trong tuong lai

## 2. Kien Truc He Thong

Pipeline tong quan:

1. Nguoi dung gui anh (API hoac app mobile)
2. YOLOv8 (tuy chon) phat hien vung doi tuong
3. Model 1 phan loai recyclable / non_recyclable
4. Model 2A hoac 2B phan loai chi tiet
5. Tra ket qua gom nhan va do tin cay

## 3. Cong Nghe Su Dung

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=flat&logo=opencv&logoColor=white)
![Flutter](https://img.shields.io/badge/Flutter-%2302569B.svg?style=flat&logo=Flutter&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?style=flat&logo=YOLO&logoColor=black)
![EfficientNetB2](https://img.shields.io/badge/EfficientNetB2-FF6F00?style=flat&logo=tensorflow&logoColor=white)

## 4. Cau Truc Thu Muc

```text
GarbageClassification/
|-- data/
|   |-- recyclable/
|   |   |-- cardboard/
|   |   |-- metal/
|   |   |-- paper/
|   |   `-- plastic/
|   `-- non_recyclable/
|       `-- trash/
|-- docs/
|-- Image/
|-- models/
|   |-- main.py
|   |-- MainYolo.py
|   |-- utils.py
|   |-- TrainModel1.py
|   |-- TrainModel2.py
|   |-- train_model1_improved.py
|   |-- predict_image.py
|   |-- predict_camera.py
|   |-- CovertTFlite.py
|   |-- logs/
|   |-- logs_EfficientNetB2/
|   |-- model/
|   `-- Model1/
|-- src/common/RecycleTrashApp/
|   |-- lib/
|   `-- android/ ios/ web/ windows/ macos/ linux/
|-- requirements.txt
`-- README.md
```

## 5. Cai Dat

### Yeu cau

- Python 3.8+
- pip moi
- Khuyen nghi RAM toi thieu 4GB

### Cai thu vien

```bash
git clone https://github.com/LeNhan18/GarbageClassification.git
cd GarbageClassification
pip install -r requirements.txt
```

## 6. Chay Ung Dung

### Chay FastAPI backend

```bash
python -m uvicorn models.main:app --reload --host 0.0.0.0 --port 8000
```

Sau khi chay thanh cong:
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

## 7. API Endpoints

| Method | Endpoint | Mo ta |
|---|---|---|
| GET | / | Thong tin co ban cua API |
| GET | /health | Kiem tra trang thai model |
| POST | /predict | Phat hien doi tuong bang YOLOv8 |
| POST | /classify_garbage | Phan loai rac bang EfficientNetB2 |

### Vi du goi API

```bash
curl -X POST "http://localhost:8000/classify_garbage" \
  -F "file=@path/to/image.jpg"
```

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/image.jpg"
```

## 8. Mo Hinh Da Huan Luyen

Model da train duoc luu tai:
- models/model/model1_binary_recyclable.keras
- models/model/model1_best.keras
- models/model/model2_best.keras
- models/model/model2b_best.keras

### Mo hinh tren Hugging Face

Repository:
- https://huggingface.co/LeNhan18/ClassifyGarbageEfficientNetB2

Vi du tai model:

```python
from huggingface_hub import hf_hub_download

local_path = hf_hub_download(
    repo_id="LeNhan18/ClassifyGarbageEfficientNetB2",
    filename="model.keras"
)
print(local_path)
```

## 9. Hinh Anh Minh Hoa

| Main Interface | EfficientNetB2 Classification |
|:--:|:--:|
| ![Main Interface](Image/GiaodienChinh.jpg) | ![Classification Interface](Image/GiaodienPhanLoaiEB2.jpg) |

| YOLOv8 Detection | History View |
|:--:|:--:|
| ![YOLO Interface](Image/YOLOV8.jpg) | ![History Interface](Image/LichSu.jpg) |

## 10. Huong Dan Luyen Mo Hinh

Mot so script training co san:

```bash
python models/TrainModel1.py
python models/TrainModel2.py
python models/train_model1_improved.py
python models/Train_Model2B_ResNet.py
```

Chuyen doi sang TensorFlow Lite:

```bash
python models/CovertTFlite.py
```

## 11. Ung Dung Flutter

Duong dan app mobile:
- src/common/RecycleTrashApp/

Chay app Flutter:

```bash
cd src/common/RecycleTrashApp
flutter pub get
flutter run
```

## 12. Dong Gop

1. Fork repository
2. Tao nhanh moi: git checkout -b feature/ten-tinh-nang
3. Commit: git commit -m "feat: mo ta thay doi"
4. Push nhanh: git push origin feature/ten-tinh-nang
5. Tao Pull Request

## 13. License

Du an duoc phat hanh theo MIT License.
