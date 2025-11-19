# Plant Disease Detection System

A full-stack project for detecting plant diseases from leaf images using deep learning (ResNet50), FastAPI backend, and React + Vite + Tailwind frontend.

## Features

- Upload plant leaf images (JPEG/PNG, <10MB)
- Predict disease class and confidence
- View prediction history and feedback
- See all supported disease classes
- Check backend/model health status
- Submit feedback for model improvement
- Visualize dataset and training metrics

## Project Structure

```
mini/
├── backend/        # FastAPI backend, model, training, dataset, feedback
├── frontend/       # React + Vite + Tailwind frontend
```

## Quick Start

### 1. Backend

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

- Model: Place `plant_disease_model_new.pt` and `class_names_new.json` in `backend/models/`
- Dataset: Place your dataset in `backend/data/`

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

- Access at [http://localhost:3000](http://localhost:3000)

## Training the Model

- Use `backend/training/train_full_model.py` to train ResNet50 on your dataset.
- Training history and plots are saved in `backend/models/`.

## Visualizing Dataset & Results

- Run `backend/visualize_dataset.py` to see class distribution and sample images.
- Run `backend/training/visualize_results.py` to plot training metrics.

## Feedback & Retraining

- User feedback is stored in `backend/feedback_data/` for future retraining.
- Use feedback export for model improvement.

## API Endpoints

- `POST /predict`
- `GET /history`
- `GET /classes`
- `GET /health`
- `POST /feedback`
- `GET /feedback/export`

**For more details, see `backend/README.md` and `frontend/README.md`.**
