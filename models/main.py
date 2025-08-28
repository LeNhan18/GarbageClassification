import os
import io
import numpy as np
import cv2
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from fastapi.middleware.cors import CORSMiddleware
import traceback
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from PIL import Image
try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None  # ultralytics may be optional if only classification is used

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cấu hình từ environment variables
MODEL_BASE_PATH = os.getenv("MODEL_BASE_PATH", r"Z:\GarbageClassification\models\Model_Final")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024))  # 10MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", r"C:\\Users\\Admin\\PycharmProjects\\pythonProject\\runs\\detect\\train4\\weights\\last.pt")

# Đường dẫn đến các file model
MODEL_PATHS = {
    'recyclable': os.path.join(MODEL_BASE_PATH, 'Model1_EfficientNetB2.keras'),
    'recyclable_detail': os.path.join(MODEL_BASE_PATH, 'model2A_EfficientNetB2.keras'),
    'non_recyclable_detail': os.path.join(MODEL_BASE_PATH, 'model2B_EfficientNetB2_final.keras')
}

# Kích thước ảnh và ngưỡng
IMG_SIZE = (224, 224)
BINARY_CLASSIFICATION_THRESHOLD = 0.5

# Định nghĩa các nhãn
LABELS_RECYCLABLE = {0: 'Khong tai che', 1: 'Tai che'}
LABELS_RECYCLABLE_DETAIL = {
    0: 'Bia cung ', 1: 'thuy tinh', 2: 'kim loai', 3: 'Giay', 4: 'Nhua '
}
LABELS_NON_RECYCLABLE_DETAIL = {
    0: 'Pin', 1: 'Rac huu co', 2: 'Quan Ao cu', 3: 'Giay cu', 4: 'Rac Khac'
}

# Global models storage
models: Dict[str, Optional[tf.keras.Model]] = {}
yolo_model: Optional[Any] = None


# Pydantic models
class WasteClassificationResult(BaseModel):
    category: str
    categoryConfidence: float
    recyclable: bool
    recyclableConfidence: float


class ClassificationResponse(BaseModel):
    results: List[WasteClassificationResult]
    message: str = "Classification successful"


class HealthCheckResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    message: str


def load_models():
    """Load all classification models"""
    global models
    models_loaded = {}

    for model_type, model_path in MODEL_PATHS.items():
        try:
            if not os.path.exists(model_path):
                logger.warning(f"Model {model_type} not found at {model_path}")
                models[model_type] = None
                models_loaded[model_type] = False
                continue

            logger.info(f"Loading model {model_type} from: {model_path}")
            models[model_type] = tf.keras.models.load_model(model_path, compile=False)
            models_loaded[model_type] = True
            logger.info(f"Successfully loaded model {model_type}")

        except Exception as e:
            logger.error(f"Error loading model {model_type}: {e}")
            models[model_type] = None
            models_loaded[model_type] = False

    return models_loaded


def load_yolo_model() -> bool:
    """Load YOLO model if ultralytics is available and path is configured"""
    global yolo_model
    if YOLO is None:
        logger.warning("ultralytics not installed; YOLO endpoint will be unavailable")
        yolo_model = None
        return False
    try:
        if not YOLO_MODEL_PATH or not os.path.exists(YOLO_MODEL_PATH):
            logger.warning(f"YOLO model path not found: {YOLO_MODEL_PATH}")
            yolo_model = None
            return False
        yolo_model = YOLO(YOLO_MODEL_PATH)
        logger.info("YOLO model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        yolo_model = None
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    logger.info("Starting up Waste Classification API...")
    models_status = load_models()
    yolo_status = load_yolo_model()

    # Check if at least one model is loaded
    if not any(models_status.values()):
        logger.error("No models could be loaded. API will not function properly.")
    else:
        logger.info("API startup completed successfully")

    yield

    # Shutdown (cleanup if needed)
    logger.info("Shutting down Waste Classification API...")
    # Clear models from memory
    global models
    models.clear()
    global yolo_model
    yolo_model = None


# Khởi tạo FastAPI app
app = FastAPI(
    title="Waste Classification API",
    description="API for classifying waste into recyclable and non-recyclable categories",
    version="1.0.0",
    lifespan=lifespan
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên chỉ định cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def validate_file(file: UploadFile) -> None:
    """Validate uploaded file"""
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type not supported. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Check content type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")


def preprocess_classification_image(
        image_np: np.ndarray,
        target_size: tuple = IMG_SIZE,
        normalize_option: str = '0_1',
        to_rgb: bool = True
) -> np.ndarray:
    """Preprocess image for classification"""
    try:
        # Resize image
        processed_img = cv2.resize(image_np, target_size)

        # Convert color space if needed
        if to_rgb and len(processed_img.shape) == 3:
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

        # Convert to float32
        processed_img = processed_img.astype('float32')

        # Normalize
        if normalize_option == '0_1':
            processed_img /= 255.0
        elif normalize_option == '-1_1':
            processed_img = (processed_img / 127.5) - 1

        # Add batch dimension
        processed_img = np.expand_dims(processed_img, axis=0)

        return processed_img

    except Exception as e:
        logger.error(f"Error in image preprocessing: {e}")
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {e}")


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    models_status = {name: model is not None for name, model in models.items()}
    all_loaded = all(models_status.values())

    return HealthCheckResponse(
        status="healthy" if all_loaded else "degraded",
        models_loaded=models_status,
        message="All models loaded" if all_loaded else "Some models failed to load"
    )


@app.post("/classify_garbage", response_model=ClassificationResponse)
async def classify_garbage(file: UploadFile = File(...)):
    """Classify waste image into recyclable/non-recyclable categories"""
    try:
        # Validate file
        validate_file(file)

        # Check if required models are loaded
        required_models = ['recyclable', 'recyclable_detail', 'non_recyclable_detail']
        missing_models = [name for name in required_models if models.get(name) is None]

        if missing_models:
            raise HTTPException(
                status_code=503,
                detail=f"Required models not loaded: {', '.join(missing_models)}"
            )

        # Read and decode image
        contents = await file.read()

        # Check file size
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
            )

        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Cannot decode image file")

        # Preprocess image
        processed_img = preprocess_classification_image(
            img,
            normalize_option='0_1',
            to_rgb=True
        )

        # Primary classification (recyclable/non-recyclable)
        recyclable_preds = models['recyclable'].predict(processed_img, verbose=0)
        recyclable_prob = float(recyclable_preds[0][0])
        is_recyclable = recyclable_prob > BINARY_CLASSIFICATION_THRESHOLD

        # Detailed classification
        if is_recyclable:
            detail_preds = models['recyclable_detail'].predict(processed_img, verbose=0)
            detail_class_index = np.argmax(detail_preds[0])
            predicted_category = LABELS_RECYCLABLE_DETAIL.get(
                detail_class_index,
                "unknown_recyclable"
            )
            confidence = float(np.max(detail_preds[0]))
        else:
            detail_preds = models['non_recyclable_detail'].predict(processed_img, verbose=0)
            detail_class_index = np.argmax(detail_preds[0])
            predicted_category = LABELS_NON_RECYCLABLE_DETAIL.get(
                detail_class_index,
                "unknown_non_recyclable"
            )
            confidence = float(np.max(detail_preds[0]))

        # Create result
        results_list = [WasteClassificationResult(
            category=predicted_category,
            categoryConfidence=confidence,
            recyclable=is_recyclable,
            recyclableConfidence=recyclable_prob if is_recyclable else (1 - recyclable_prob)
        )]

        return ClassificationResponse(results=results_list)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in classify_garbage: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")


# YOLO predict endpoint (object detection)
class YoloBox(BaseModel):
    class_name: str
    confidence: float
    bounding_box: Dict[str, float]


class YoloResponse(BaseModel):
    detections: List[YoloBox]


@app.post("/predict", response_model=YoloResponse)
async def predict(file: UploadFile = File(...)):
    """YOLO object detection endpoint at /predict for backward compatibility"""
    # Validate file
    validate_file(file)

    if yolo_model is None:
        raise HTTPException(status_code=503, detail="YOLO model not loaded or ultralytics missing")

    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {e}")

    try:
        results = yolo_model(image)
        result = results[0]
        detections: List[YoloBox] = []
        for box in result.boxes:
            coords = box.xyxy[0].tolist()
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = yolo_model.names[class_id]
            detections.append(YoloBox(
                class_name=class_name,
                confidence=round(confidence, 4),
                bounding_box={
                    "x1": round(coords[0], 2),
                    "y1": round(coords[1], 2),
                    "x2": round(coords[2], 2),
                    "y2": round(coords[3], 2)
                }
            ))
        return YoloResponse(detections=detections)
    except Exception as e:
        logger.error(f"YOLO inference error: {e}")
        raise HTTPException(status_code=500, detail="YOLO inference failed")


@app.get("/", include_in_schema=False)
async def read_root():
    """Root endpoint"""
    return {
        "message": "Waste Classification API is running",
        "version": "1.0.0",
        "endpoints": {
            "classify": "/classify_garbage",
            "health": "/health",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )