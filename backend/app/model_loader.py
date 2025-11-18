import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


class ModelLoader:
    """
    Handles loading and management of the PyTorch model
    """
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model: Optional[nn.Module] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(self) -> None:
        """Load the trained model from disk"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        
        try:
            # Load the model
            self.model = torch.load(
                self.model_path,
                map_location=self.device
            )
            
            # Set to evaluation mode
            self.model.eval()
            
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def get_model(self) -> nn.Module:
        """Get the loaded model"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def get_device(self) -> torch.device:
        """Get the device (CPU/GPU) being used"""
        return self.device
