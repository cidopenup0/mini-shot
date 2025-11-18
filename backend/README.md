# Plant Disease Detection Backend

FastAPI backend for plant disease detection using deep learning.

## Features

- REST API endpoints for image upload and disease prediction
- PyTorch CNN model integration
- Image preprocessing pipeline
- SQLite-based prediction logging
- CORS support for frontend integration
- Modular architecture

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── config.py           # Configuration settings
│   ├── model_loader.py     # Model loading logic
│   ├── preprocessor.py     # Image preprocessing
│   ├── inference.py        # Prediction engine
│   └── logger.py           # Prediction logging
├── models/
│   └── plant_disease_model.pt  # Trained model (add your model here)
├── logs/
│   └── predictions.db      # SQLite database (auto-created)
├── main.py                 # FastAPI application
├── requirements.txt
└── .env.example

```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your trained model in `models/plant_disease_model.pt`

3. Configure environment variables (optional):
```bash
cp .env.example .env
# Edit .env with your settings
```

4. Run the server:
```bash
python main.py
```

Or with uvicorn:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### POST /predict
Upload an image for disease prediction.

**Request:**
- Form-data with `file` field containing the image

**Response:**
```json
{
  "success": true,
  "prediction": {
    "disease": "Tomato___Late_blight",
    "confidence": 0.95,
    "class_id": 17
  },
  "metadata": {
    "filename": "leaf.jpg",
    "timestamp": "2025-11-18T10:30:00",
    "model_version": "1.0.0"
  }
}
```

### GET /health
Health check endpoint.

### GET /history?limit=50
Get recent predictions.

### GET /classes
Get list of supported disease classes.

## Model Requirements

The model file should be a PyTorch model (`.pt` file) that:
- Accepts input tensors of shape `(batch_size, 3, 224, 224)`
- Returns logits for the disease classes
- Was trained on the same classes listed in `config.py`

## Configuration

Edit `app/config.py` to:
- Update `CLASS_NAMES` with your model's disease classes
- Adjust image preprocessing parameters
- Configure server settings

## Logging

Predictions are automatically logged to SQLite database at `logs/predictions.db` with:
- Filename
- Predicted disease
- Confidence score
- Timestamp
