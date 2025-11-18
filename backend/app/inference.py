import torch
import torch.nn.functional as F
from typing import Dict

from app.model_loader import ModelLoader
from app.config import settings


class InferenceEngine:
    """
    Handles model inference and prediction
    """
    
    def __init__(self, model_loader: ModelLoader):
        self.model_loader = model_loader
    
    def predict(self, input_tensor: torch.Tensor) -> Dict:
        """
        Run inference on preprocessed image tensor
        
        Args:
            input_tensor: Preprocessed image tensor
        
        Returns:
            Dictionary containing prediction results
        """
        try:
            model = self.model_loader.get_model()
            device = self.model_loader.get_device()
            
            # Move tensor to device
            input_tensor = input_tensor.to(device)
            
            # Perform inference
            with torch.no_grad():
                outputs = model(input_tensor)
                
                # Apply softmax to get probabilities
                probabilities = F.softmax(outputs, dim=1)
                
                # Get prediction
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                confidence_value = confidence.item()
                class_id = predicted_idx.item()
                class_name = settings.CLASS_NAMES[class_id]
                
                # Get top 3 predictions
                top3_prob, top3_idx = torch.topk(probabilities, 3)
                top3_predictions = [
                    {
                        "class": settings.CLASS_NAMES[idx.item()],
                        "confidence": prob.item()
                    }
                    for prob, idx in zip(top3_prob[0], top3_idx[0])
                ]
            
            return {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(confidence_value),
                "top3_predictions": top3_predictions
            }
            
        except Exception as e:
            raise RuntimeError(f"Inference failed: {str(e)}")
    
    def batch_predict(self, input_tensors: torch.Tensor) -> list:
        """
        Run inference on batch of images
        
        Args:
            input_tensors: Batch of preprocessed image tensors
        
        Returns:
            List of prediction dictionaries
        """
        try:
            model = self.model_loader.get_model()
            device = self.model_loader.get_device()
            
            input_tensors = input_tensors.to(device)
            
            with torch.no_grad():
                outputs = model(input_tensors)
                probabilities = F.softmax(outputs, dim=1)
                confidences, predicted_indices = torch.max(probabilities, 1)
            
            results = []
            for i in range(len(input_tensors)):
                class_id = predicted_indices[i].item()
                results.append({
                    "class_id": class_id,
                    "class_name": settings.CLASS_NAMES[class_id],
                    "confidence": float(confidences[i].item())
                })
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"Batch inference failed: {str(e)}")
