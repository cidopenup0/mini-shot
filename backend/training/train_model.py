import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import json
from datetime import datetime
import time


class ModelTrainer:
    """
    Handles training of the plant disease detection model
    """
    
    def __init__(
        self,
        data_dir: str,
        model_save_path: str,
        num_classes: int,
        batch_size: int = 32,
        num_epochs: int = 25,
        learning_rate: float = 0.001,
        device: str = None
    ):
        """
        Initialize trainer
        
        Args:
            data_dir: Path to processed data directory (containing train/val/test)
            model_save_path: Path to save the trained model
            num_classes: Number of disease classes
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            learning_rate: Initial learning rate
            device: Device to use ('cuda' or 'cpu'), auto-detect if None
        """
        self.data_dir = Path(data_dir)
        self.model_save_path = Path(model_save_path)
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.class_names = []
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epochs': []
        }
    
    def setup_data_loaders(self):
        """Setup data loaders with appropriate transforms"""
        # Data augmentation for training
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Standard transforms for validation and test
        val_test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load datasets
        train_dataset = datasets.ImageFolder(
            self.data_dir / 'train',
            transform=train_transform
        )
        val_dataset = datasets.ImageFolder(
            self.data_dir / 'val',
            transform=val_test_transform
        )
        test_dataset = datasets.ImageFolder(
            self.data_dir / 'test',
            transform=val_test_transform
        )
        
        # Store class names
        self.class_names = train_dataset.classes
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"\nDataset loaded:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
        print(f"  Number of classes: {len(self.class_names)}")
    
    def build_model(self, model_name: str = 'resnet50', pretrained: bool = True):
        """
        Build the CNN model
        
        Args:
            model_name: Name of the model architecture (resnet50, resnet18, efficientnet_b0)
            pretrained: Whether to use pretrained weights
        """
        print(f"\nBuilding model: {model_name}")
        print(f"Pretrained: {pretrained}")
        
        if model_name == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.model = models.resnet50(weights=weights)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, self.num_classes)
        
        elif model_name == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.model = models.resnet18(weights=weights)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, self.num_classes)
        
        elif model_name == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.model = models.efficientnet_b0(weights=weights)
            num_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_features, self.num_classes)
        
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=3,
            verbose=True
        )
        
        print(f"Model built successfully!")
    
    def train_epoch(self) -> tuple:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in self.train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self) -> tuple:
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        
        best_val_acc = 0.0
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['epochs'].append(epoch + 1)
            
            epoch_time = time.time() - epoch_start
            
            # Print progress
            print(f"\nEpoch [{epoch+1}/{self.num_epochs}] - {epoch_time:.2f}s")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(is_best=True)
                print(f"  ✓ Best model saved! (Val Acc: {val_acc*100:.2f}%)")
        
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
    
    def test(self) -> dict:
        """Evaluate on test set"""
        print("\n" + "="*60)
        print("Testing Model")
        print("="*60)
        
        self.model.eval()
        correct = 0
        total = 0
        class_correct = [0] * self.num_classes
        class_total = [0] * self.num_classes
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Per-class accuracy
                c = (predicted == labels).squeeze()
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        
        test_acc = correct / total
        
        print(f"\nOverall Test Accuracy: {test_acc*100:.2f}%")
        print("\nPer-Class Accuracy:")
        print("-" * 60)
        
        results = {
            'overall_accuracy': test_acc,
            'per_class_accuracy': {}
        }
        
        for i in range(self.num_classes):
            if class_total[i] > 0:
                acc = class_correct[i] / class_total[i]
                results['per_class_accuracy'][self.class_names[i]] = acc
                print(f"{self.class_names[i]:40s} {acc*100:.2f}%")
        
        return results
    
    def save_model(self, is_best: bool = False):
        """Save the trained model"""
        self.model_save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if is_best:
            save_path = self.model_save_path
        else:
            save_path = self.model_save_path.parent / f"{self.model_save_path.stem}_final.pt"
        
        torch.save(self.model, save_path)
    
    def save_history(self):
        """Save training history"""
        history_path = self.model_save_path.parent / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"\nTraining history saved to: {history_path}")
    
    def save_class_names(self):
        """Save class names for reference"""
        class_names_path = self.model_save_path.parent / "class_names.json"
        with open(class_names_path, 'w') as f:
            json.dump(self.class_names, f, indent=2)
        print(f"Class names saved to: {class_names_path}")


def main():
    """Main execution function"""
    print("Plant Disease Model Training")
    print("="*60 + "\n")
    
    # Configuration
    DATA_DIR = "data/processed"
    MODEL_SAVE_PATH = "models/plant_disease_model.pt"
    NUM_EPOCHS = 25
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    MODEL_ARCHITECTURE = 'resnet50'  # Options: resnet50, resnet18, efficientnet_b0
    
    # Get number of classes from data
    train_dir = Path(DATA_DIR) / 'train'
    class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
    NUM_CLASSES = len(class_dirs)
    
    print(f"Configuration:")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Model save path: {MODEL_SAVE_PATH}")
    print(f"  Number of classes: {NUM_CLASSES}")
    print(f"  Model architecture: {MODEL_ARCHITECTURE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    
    # Initialize trainer
    trainer = ModelTrainer(
        data_dir=DATA_DIR,
        model_save_path=MODEL_SAVE_PATH,
        num_classes=NUM_CLASSES,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE
    )
    
    # Setup data loaders
    trainer.setup_data_loaders()
    
    # Build model
    trainer.build_model(model_name=MODEL_ARCHITECTURE, pretrained=True)
    
    # Train model
    trainer.train()
    
    # Test model
    test_results = trainer.test()
    
    # Save results
    trainer.save_history()
    trainer.save_class_names()
    
    # Save test results
    results_path = Path(MODEL_SAVE_PATH).parent / "test_results.json"
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"Test results saved to: {results_path}")
    
    print("\n" + "="*60)
    print("All Done!")
    print("="*60)


if __name__ == "__main__":
    main()
