import os
from pathlib import Path
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

NUM_EPOCHS = 25              # Training epochs
BATCH_SIZE = 32              # Batch size
LEARNING_RATE = 0.001        # Learning rate
MODEL_ARCHITECTURE = 'resnet50'  # or 'resnet18', 'efficientnet_b0'

def download_dataset():
    """
    Download the Plant Disease dataset from Kaggle
    """
    print("Plant Disease Dataset Downloader")
    print("="*60 + "\n")
    
    # Set credentials BEFORE importing kaggle
    os.environ['KAGGLE_USERNAME'] = 'cidopenup0'
    os.environ['KAGGLE_KEY'] = '64cdee528f2e3125f68d051dfecc709a'
    
    # Check if kaggle is installed
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("Error: Kaggle API not found!")
        print("Install it with: pip install kaggle")
        return
    
    # Create data directories
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading dataset from Kaggle...")
    print("Dataset: emmarex/plantdisease\n")
    
    try:
        # Initialize and authenticate Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        print("Authentication successful!")
        print("Starting download...\n")
        
        # Download dataset
        api.dataset_download_files(
            'emmarex/plantdisease',
            path=str(raw_dir),
            unzip=True
        )
        
        print("\n" + "="*60)
        print("Download Complete!")
        print("="*60)
        print(f"Dataset saved to: {raw_dir.absolute()}")
        
        # Check the structure
        print("\nDataset structure:")
        for item in raw_dir.iterdir():
            if item.is_dir():
                num_classes = len([d for d in item.iterdir() if d.is_dir()])
                print(f"  {item.name}/ - {num_classes} classes")
        
        print("\nNext step: Run split_dataset.py to split into train/val/test")
        
    except Exception as e:
        print(f"\nError downloading dataset: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure you have accepted the dataset terms on Kaggle")
        print("2. Verify your kaggle.json credentials are set up correctly")
        print("3. Check your internet connection")


def split_dataset(raw_dir, test_size=0.2, random_state=42):
    """
    Split the dataset into train/val/test
    """
    print("Splitting dataset into train/val/test...")
    raw_path = raw_dir / "train"
    test_path = raw_dir / "test"
    val_path = raw_dir / "val"
    
    # Create directories
    test_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)
    
    # Split the dataset
    train_data, test_data = train_test_split(
        list(raw_path.iterdir()),
        test_size=test_size,
        random_state=random_state
    )
    
    # Create train/val/test directories
    for path in train_data:
        path.mkdir(parents=True, exist_ok=True)
    for path in test_data:
        path.mkdir(parents=True, exist_ok=True)
    
    print(f"Train set: {len(train_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    print(f"Val set: {len(test_data)} samples")


def train_model(model, train_loader, val_loader, epochs=NUM_EPOCHS):
    """
    Train the model
    """
    print("Training model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss / len(train_loader):.4f} - Val Loss: {val_loss / len(val_loader):.4f}")
    
    return train_losses, val_losses


if __name__ == "__main__":
    download_dataset()
