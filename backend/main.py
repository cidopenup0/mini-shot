from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional
import uvicorn
from datetime import datetime
import os
import shutil
from pathlib import Path

from app.model_loader import ModelLoader
from app.preprocessor import ImagePreprocessor
from app.inference import InferenceEngine
from app.logger import PredictionLogger
from app.config import settings

# Initialize components
model_loader = ModelLoader(settings.MODEL_PATH)
preprocessor = ImagePreprocessor()
inference_engine = InferenceEngine(model_loader)
logger = PredictionLogger(settings.LOG_DB_PATH)

# Feedback storage directory
FEEDBACK_DIR = Path("feedback_data")
FEEDBACK_DIR.mkdir(exist_ok=True)

# Pydantic model for feedback
class FeedbackRequest(BaseModel):
    prediction_id: Optional[int] = None
    filename: str
    predicted_class: str
    correct_class: str
    is_correct: bool
    confidence: float
    timestamp: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    print("Loading model...")
    model_loader.load_model()
    print("Model loaded successfully")
    logger.initialize()
    print("Logger initialized")
    yield
    # Shutdown (cleanup if needed)
    print("Shutting down...")


app = FastAPI(
    title="Plant Disease Detection API",
    description="Backend API for detecting plant diseases from leaf images",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        
        current_time = datetime.now()
        
        # Log prediction and get ID
        prediction_id = logger.log_prediction(
            filename=file.filename,
            disease=prediction_result["class_name"],
            confidence=prediction_result["confidence"],
            timestamp=current_time
        )
        
        # Save image for potential retraining (with prediction_id in filename)
        if prediction_id:
            image_filename = f"pred_{prediction_id}_{file.filename}"
            image_save_path = FEEDBACK_DIR / "uploads" / image_filename
            image_save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the uploaded image
            with open(image_save_path, "wb") as f:
                f.write(image_bytes)
        
        # Prepare response
        response = {
            "class": prediction_result["class_name"],
            "confidence": prediction_result["confidence"],
            "class_id": prediction_result["class_id"],
            "filename": file.filename,
            "timestamp": current_time.isoformat(),
            "prediction_id": prediction_id
        }
        
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


@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """
    Submit user feedback for a prediction
    
    Args:
        feedback: Feedback data including predicted and correct classes
    
    Returns:
        Success status and feedback ID
    """
    try:
        # Find the saved image
        image_path = None
        if feedback.prediction_id:
            image_pattern = f"pred_{feedback.prediction_id}_*"
            uploads_dir = FEEDBACK_DIR / "uploads"
            matching_files = list(uploads_dir.glob(image_pattern))
            
            if matching_files:
                original_image = matching_files[0]
                
                # Organize by correct class for easy retraining
                correct_class_dir = FEEDBACK_DIR / "labeled" / feedback.correct_class.replace(" ", "_").replace("/", "-")
                correct_class_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy image to labeled directory
                new_image_path = correct_class_dir / f"{feedback.prediction_id}_{feedback.filename}"
                shutil.copy2(original_image, new_image_path)
                image_path = str(new_image_path)
        
        # Log feedback
        success = logger.log_feedback(
            prediction_id=feedback.prediction_id,
            filename=feedback.filename,
            predicted_class=feedback.predicted_class,
            correct_class=feedback.correct_class,
            is_correct=feedback.is_correct,
            confidence=feedback.confidence,
            original_timestamp=feedback.timestamp,
            image_path=image_path
        )
        
        if success:
            return {
                "success": True,
                "message": "Feedback recorded successfully",
                "image_saved": image_path is not None
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to save feedback"
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process feedback: {str(e)}"
        )


@app.get("/feedback/export")
async def export_feedback():
    """
    Export feedback data for model retraining
    
    Returns:
        List of all feedback with corrected labels
    """
    try:
        feedback_data = logger.get_feedback_for_retraining()
        return {
            "success": True,
            "count": len(feedback_data),
            "feedback": feedback_data
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export feedback: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
