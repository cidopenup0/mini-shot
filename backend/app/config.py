import os
import json
from pathlib import Path

class Settings:
    # Server configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    
    # Model configuration
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "plant_disease_model_new.pt"))
    MODEL_VERSION = os.getenv("MODEL_VERSION", "2.0.0")
    
    # Image preprocessing
    IMAGE_SIZE = (224, 224)  # Standard input size for most CNN models
    MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
    STD = [0.229, 0.224, 0.225]   # ImageNet std
    
    # Logging
    LOG_DB_PATH = os.getenv("LOG_DB_PATH", str(BASE_DIR / "logs" / "predictions.db"))
    
    # Disease classes - Load from class_names_new.json
    CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names_new.json"
    
    def _load_class_names(self):
        """Load class names from JSON file"""
        try:
            with open(self.CLASS_NAMES_PATH, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: class_names.json not found at {self.CLASS_NAMES_PATH}")
            # Fallback to the 15 classes from your dataset
            return [
                "Pepper__bell___Bacterial_spot",
                "Pepper__bell___healthy",
                "Potato___Early_blight",
                "Potato___Late_blight",
                "Potato___healthy",
                "Tomato_Bacterial_spot",
                "Tomato_Early_blight",
                "Tomato_Late_blight",
                "Tomato_Leaf_Mold",
                "Tomato_Septoria_leaf_spot",
                "Tomato_Spider_mites_Two_spotted_spider_mite",
                "Tomato__Target_Spot",
                "Tomato__Tomato_YellowLeaf__Curl_Virus",
                "Tomato__Tomato_mosaic_virus",
                "Tomato_healthy"
            ]
    
    CLASS_NAMES = None  # Will be loaded dynamically
    
    # File validation
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

settings = Settings()
# Load class names after settings initialization
settings.CLASS_NAMES = settings._load_class_names()
