import torch
from torchvision import transforms
from PIL import Image
import io
from typing import Union

from app.config import settings


class ImagePreprocessor:
    """
    Handles image preprocessing for model input
    """
    
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(settings.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=settings.MEAN,
                std=settings.STD
            )
        ])
    
    def preprocess(self, image_data: Union[bytes, str]) -> torch.Tensor:
        """
        Preprocess image data into model-ready tensor
        
        Args:
            image_data: Raw image bytes or file path
        
        Returns:
            Preprocessed tensor ready for model input
        """
        try:
            # Load image
            if isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            else:
                image = Image.open(image_data)
            
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Apply transformations
            tensor = self.transform(image)
            
            # Add batch dimension
            tensor = tensor.unsqueeze(0)
            
            return tensor
            
        except Exception as e:
            raise ValueError(f"Image preprocessing failed: {str(e)}")
    
    def validate_image(self, image_data: bytes) -> bool:
        """
        Validate if the data is a valid image
        
        Args:
            image_data: Raw image bytes
        
        Returns:
            True if valid, raises exception otherwise
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            image.verify()
            return True
        except Exception as e:
            raise ValueError(f"Invalid image data: {str(e)}")
