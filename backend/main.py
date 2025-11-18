from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime
import os

from app.model_loader import ModelLoader
from app.preprocessor import ImagePreprocessor
from app.inference import InferenceEngine
from app.logger import PredictionLogger
from app.config import settings

app = FastAPI(
    title="Plant Disease Detection API",
    description="Backend API for detecting plant diseases from leaf images",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
model_loader = ModelLoader(settings.MODEL_PATH)
preprocessor = ImagePreprocessor()
inference_engine = InferenceEngine(model_loader)
logger = PredictionLogger(settings.LOG_DB_PATH)


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    print("Loading model...")
    model_loader.load_model()
    print("Model loaded successfully")
    logger.initialize()
    print("Logger initialized")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Plant Disease Detection API",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model_loader.is_loaded(),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    """
    Predict plant disease from uploaded leaf image
    
    Args:
        file: Uploaded image file (JPEG, PNG)
    
    Returns:
        JSON with prediction, confidence, and metadata
    """
    try:
        # Validate file format
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Please upload an image file."
            )
        
        allowed_formats = ["image/jpeg", "image/jpg", "image/png"]
        if file.content_type not in allowed_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format. Allowed: JPEG, PNG"
            )
        
        # Read and preprocess image
        image_bytes = await file.read()
        preprocessed_tensor = preprocessor.preprocess(image_bytes)
        
        # Run inference
        prediction_result = inference_engine.predict(preprocessed_tensor)
        
        # Prepare response
        response = {
            "success": True,
            "prediction": {
                "disease": prediction_result["class_name"],
                "confidence": prediction_result["confidence"],
                "class_id": prediction_result["class_id"]
            },
            "metadata": {
                "filename": file.filename,
                "timestamp": datetime.now().isoformat(),
                "model_version": settings.MODEL_VERSION
            }
        }
        
        # Log prediction
        logger.log_prediction(
            filename=file.filename,
            disease=prediction_result["class_name"],
            confidence=prediction_result["confidence"],
            timestamp=datetime.now()
        )
        
        return JSONResponse(content=response)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/history")
async def get_prediction_history(limit: int = 50):
    """
    Get recent prediction history
    
    Args:
        limit: Number of recent predictions to return
    
    Returns:
        List of recent predictions
    """
    try:
        history = logger.get_recent_predictions(limit)
        return {
            "success": True,
            "count": len(history),
            "predictions": history
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve history: {str(e)}"
        )


@app.get("/classes")
async def get_disease_classes():
    """Get list of supported disease classes"""
    return {
        "success": True,
        "classes": settings.CLASS_NAMES,
        "total": len(settings.CLASS_NAMES)
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
