import os
from pathlib import Path

class Settings:
    # Server configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    
    # Model configuration
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "plant_disease_model.pt"))
    MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
    
    # Image preprocessing
    IMAGE_SIZE = (224, 224)  # Standard input size for most CNN models
    MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
    STD = [0.229, 0.224, 0.225]   # ImageNet std
    
    # Logging
    LOG_DB_PATH = os.getenv("LOG_DB_PATH", str(BASE_DIR / "logs" / "predictions.db"))
    
    # Disease classes - Update this based on your trained model
    CLASS_NAMES = [
        "Apple___Apple_scab",
        "Apple___Black_rot",
        "Apple___Cedar_apple_rust",
        "Apple___healthy",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Corn_(maize)___Common_rust_",
        "Corn_(maize)___Northern_Leaf_Blight",
        "Corn_(maize)___healthy",
        "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "Grape___healthy",
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy"
    ]
    
    # File validation
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

settings = Settings()
