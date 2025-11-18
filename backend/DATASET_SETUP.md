# Quick Start Guide - Dataset Preparation

## Step 1: Install Dependencies

```bash
# Activate virtual environment
.venv\Scripts\activate

# Install all requirements
pip install -r requirements.txt
```

## Step 2: Set Up Kaggle API

1. Go to https://www.kaggle.com/account
2. Scroll to "API" section
3. Click "Create New API Token"
4. Save the downloaded `kaggle.json` to: `C:\Users\<YourUsername>\.kaggle\kaggle.json`

## Step 3: Download Dataset

```bash
cd data_preparation
python download_dataset.py
```

This downloads the Plant Disease dataset from:
https://www.kaggle.com/datasets/emmarex/plantdisease

## Step 4: Split Dataset

```bash
python split_dataset.py
```

**Output:**
- `data/processed/train/` - 70% of images (for training)
- `data/processed/val/` - 15% of images (for validation)
- `data/processed/test/` - 15% of images (for testing)
- `data/processed/split_statistics.json` - Split statistics

## Step 5: Analyze Dataset

```bash
python analyze_dataset.py
```

Generates:
- Console summary with class distributions
- Distribution visualization: `data/processed/dataset_distribution.png`

## Expected Directory Structure

```
mini/backend/
├── data/
│   ├── raw/
│   │   └── PlantVillage/          # Downloaded dataset
│   │       ├── Tomato___healthy/
│   │       ├── Potato___Late_blight/
│   │       └── ... (38 classes)
│   │
│   └── processed/
│       ├── train/                  # 70% training data
│       ├── val/                    # 15% validation data
│       ├── test/                   # 15% test data
│       ├── split_statistics.json
│       └── dataset_distribution.png
│
├── data_preparation/
│   ├── download_dataset.py
│   ├── split_dataset.py
│   └── analyze_dataset.py
│
└── models/
    └── plant_disease_model.pt      # Your trained model goes here
```

## Customization

Edit `split_dataset.py` to change ratios:

```python
TRAIN_RATIO = 0.7   # 70% training
VAL_RATIO = 0.15    # 15% validation
TEST_RATIO = 0.15   # 15% testing
```

## Next Steps

After dataset preparation:
1. Train your model using the split dataset
2. Save trained model as `models/plant_disease_model.pt`
3. Update `app/config.py` CLASS_NAMES if your classes differ
4. Run the backend: `python main.py`
