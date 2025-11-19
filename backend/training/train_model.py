"""
Training script for Plant Disease Detection Model
Complete 38-class dataset training with EfficientNet
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import json
from datetime import datetime
import time
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class PlantDiseaseTrainer:
    """
    Complete training pipeline for plant disease detection
    """
    
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        output_dir: str,
        model_name: str = "resnet50",
        batch_size: int = 32,
        num_epochs: int = 10,
        learning_rate: float = 0.001,
        device: str = None
    ):
        """
        Initialize trainer
        
        Args:
            train_dir: Path to training data directory
            val_dir: Path to validation data directory
            output_dir: Directory to save outputs (model, plots, logs)
            model_name: Model architecture (resnet50)
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            learning_rate: Initial learning rate
            device: Device to use ('cuda' or 'cpu')
        """
        self.train_dir = Path(train_dir)
        self.val_dir = Path(val_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"{'='*60}")
        print(f"Plant Disease Detection Model Training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Model: {model_name}")
        print(f"Batch Size: {batch_size}")
        print(f"Epochs: {num_epochs}")
        print(f"Learning Rate: {learning_rate}")
        print(f"{'='*60}\n")
        
        # Initialize components
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.class_names = []
        self.num_classes = 0
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'train_precision': [],
            'train_recall': [],
            'train_f1': [],
            'val_loss': [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'learning_rates': [],
            'epochs': []
        }
        
        self.best_val_acc = 0.0
        self.best_model_wts = None
    
    def setup_data_loaders(self):
        """Setup data loaders with augmentation"""
        print("Setting up data loaders...")
        
        # Training augmentation
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Validation transform (no augmentation)
        val_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load datasets
        train_dataset = datasets.ImageFolder(self.train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(self.val_dir, transform=val_transform)
        
        # Get class names
        self.class_names = train_dataset.classes
        self.num_classes = len(self.class_names)
        
        print(f"Number of classes: {self.num_classes}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print("Data loaders ready!\n")
    
    def setup_model(self):
        """Initialize ResNet50 model architecture"""
        print(f"Setting up ResNet50 model...")
        
        # Load ResNet50 with ImageNet pretrained weights
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Replace final fully connected layer for our number of classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)
        
        # Move model to device (GPU/CPU)
        self.model = self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        print(f"Model ready! Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}\n")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for inputs, labels in self.train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        epoch_loss = running_loss / len(all_labels)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        return epoch_loss, accuracy * 100, precision, recall, f1
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        epoch_loss = running_loss / len(all_labels)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        return epoch_loss, accuracy * 100, precision, recall, f1
    
    def train(self):
        """Complete training loop"""
        print("Starting training...\n")
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc, train_precision, train_recall, train_f1 = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_precision, val_recall, val_f1 = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_precision'].append(train_precision)
            self.history['train_recall'].append(train_recall)
            self.history['train_f1'].append(train_f1)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_precision'].append(val_precision)
            self.history['val_recall'].append(val_recall)
            self.history['val_f1'].append(val_f1)
            self.history['learning_rates'].append(current_lr)
            self.history['epochs'].append(epoch + 1)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_wts = self.model.state_dict().copy()
            
            # Print progress
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1}/{self.num_epochs} - {epoch_time:.2f}s")
            print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | P: {train_precision*100:.2f}% | R: {train_recall*100:.2f}% | F1: {train_f1*100:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | P: {val_precision*100:.2f}% | R: {val_recall*100:.2f}% | F1: {val_f1*100:.2f}%")
            print(f"  LR: {current_lr:.6f} | Best Val Acc: {self.best_val_acc:.2f}%")
            print()
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        
        # Load best model weights
        self.model.load_state_dict(self.best_model_wts)
    
    def save_model(self):
        """Save the trained model"""
        model_path = self.output_dir / "plant_disease_model_new.pt"
        torch.save(self.model, model_path)
        print(f"Model saved to: {model_path}")
        
        # Save class names
        class_names_path = self.output_dir / "class_names_new.json"
        with open(class_names_path, 'w') as f:
            json.dump(self.class_names, f, indent=2)
        print(f"Class names saved to: {class_names_path}")
        
        # Save training history
        history_path = self.output_dir / "training_history_new.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to: {history_path}")
    
    def plot_training_curves(self):
        """Plot comprehensive training curves"""
        # Create a 2x2 subplot
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Loss curves with CrossEntropy annotation
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(self.history['epochs'], self.history['train_loss'], 
                label='Train Loss (CrossEntropy)', marker='o', linewidth=2, markersize=6)
        ax1.plot(self.history['epochs'], self.history['val_loss'], 
                label='Val Loss (CrossEntropy)', marker='s', linewidth=2, markersize=6)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('CrossEntropy Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss (CrossEntropy)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Add text box with loss info
        textstr = f'Final Train Loss: {self.history["train_loss"][-1]:.4f}\nFinal Val Loss: {self.history["val_loss"][-1]:.4f}'
        ax1.text(0.95, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. Accuracy curves
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(self.history['epochs'], self.history['train_acc'], 
                label='Train Acc', marker='o', linewidth=2, markersize=6, color='green')
        ax2.plot(self.history['epochs'], self.history['val_acc'], 
                label='Val Acc', marker='s', linewidth=2, markersize=6, color='orange')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=self.best_val_acc, color='r', linestyle='--', 
                   label=f'Best Val Acc: {self.best_val_acc:.2f}%', alpha=0.7)
        ax2.legend(fontsize=11)
        
        # 3. Learning rate schedule
        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(self.history['epochs'], self.history['learning_rates'], 
                marker='d', linewidth=2, markersize=6, color='purple')
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Learning Rate', fontsize=12)
        ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # 4. Loss vs Accuracy comparison
        ax4 = plt.subplot(2, 2, 4)
        ax4_twin = ax4.twinx()
        
        line1 = ax4.plot(self.history['epochs'], self.history['val_loss'], 
                        label='Val Loss', marker='s', linewidth=2, 
                        markersize=6, color='red', alpha=0.7)
        line2 = ax4_twin.plot(self.history['epochs'], self.history['val_acc'], 
                             label='Val Acc', marker='o', linewidth=2, 
                             markersize=6, color='blue', alpha=0.7)
        
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Validation Loss', fontsize=12, color='red')
        ax4_twin.set_ylabel('Validation Accuracy (%)', fontsize=12, color='blue')
        ax4.set_title('Validation Loss vs Accuracy', fontsize=14, fontweight='bold')
        ax4.tick_params(axis='y', labelcolor='red')
        ax4_twin.tick_params(axis='y', labelcolor='blue')
        ax4.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='center right', fontsize=11)
        
        plt.tight_layout()
        plot_path = self.output_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {plot_path}")
        plt.close()
        
        # Additional plots
        self._plot_training_summary()
        self._plot_loss_analysis()
    
    def _plot_training_summary(self):
        """Create a summary plot with key metrics"""
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        
        # Title
        fig.suptitle('Training Summary', fontsize=18, fontweight='bold', y=0.98)
        
        # Summary text
        summary_text = f"""
Model Architecture: {self.model_name}
Number of Classes: {self.num_classes}
Total Epochs: {self.num_epochs}
Batch Size: {self.batch_size}
Initial Learning Rate: {self.learning_rate}
Final Learning Rate: {self.history['learning_rates'][-1]:.6f}

{'='*50}
FINAL RESULTS
{'='*50}

Training Accuracy: {self.history['train_acc'][-1]:.2f}%
Validation Accuracy: {self.history['val_acc'][-1]:.2f}%
Best Validation Accuracy: {self.best_val_acc:.2f}%

Training Loss: {self.history['train_loss'][-1]:.4f}
Validation Loss: {self.history['val_loss'][-1]:.4f}

{'='*50}
TRAINING PROGRESS
{'='*50}

Initial Train Acc: {self.history['train_acc'][0]:.2f}%
Final Train Acc: {self.history['train_acc'][-1]:.2f}%
Improvement: {self.history['train_acc'][-1] - self.history['train_acc'][0]:.2f}%

Initial Val Acc: {self.history['val_acc'][0]:.2f}%
Final Val Acc: {self.history['val_acc'][-1]:.2f}%
Improvement: {self.history['val_acc'][-1] - self.history['val_acc'][0]:.2f}%

Device: {self.device}
        """
        
        ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        summary_path = self.output_dir / "training_summary.png"
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        print(f"Training summary saved to: {summary_path}")
        plt.close()
    
    def _plot_loss_analysis(self):
        """Create detailed CrossEntropy loss analysis plot"""
        fig = plt.figure(figsize=(14, 10))
        
        # 1. Loss curve with confidence intervals
        ax1 = plt.subplot(2, 2, 1)
        epochs = self.history['epochs']
        train_loss = self.history['train_loss']
        val_loss = self.history['val_loss']
        
        ax1.plot(epochs, train_loss, 'b-o', label='Training Loss', linewidth=2, markersize=5)
        ax1.plot(epochs, val_loss, 'r-s', label='Validation Loss', linewidth=2, markersize=5)
        ax1.fill_between(epochs, train_loss, alpha=0.2, color='blue')
        ax1.fill_between(epochs, val_loss, alpha=0.2, color='red')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('CrossEntropy Loss', fontsize=12)
        ax1.set_title('CrossEntropy Loss Over Time', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # 2. Loss difference (overfitting indicator)
        ax2 = plt.subplot(2, 2, 2)
        loss_diff = [v - t for v, t in zip(val_loss, train_loss)]
        ax2.plot(epochs, loss_diff, 'g-^', linewidth=2, markersize=6)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.fill_between(epochs, loss_diff, alpha=0.3, color='green')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss Difference (Val - Train)', fontsize=12)
        ax2.set_title('Overfitting Indicator', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Add annotation
        avg_diff = np.mean(loss_diff)
        ax2.text(0.05, 0.95, f'Avg Difference: {avg_diff:.4f}', 
                transform=ax2.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        # 3. Loss reduction rate
        ax3 = plt.subplot(2, 2, 3)
        train_loss_reduction = [0] + [train_loss[i-1] - train_loss[i] for i in range(1, len(train_loss))]
        val_loss_reduction = [0] + [val_loss[i-1] - val_loss[i] for i in range(1, len(val_loss))]
        
        ax3.bar([e - 0.2 for e in epochs], train_loss_reduction, width=0.4, 
               label='Train Loss Reduction', alpha=0.7, color='blue')
        ax3.bar([e + 0.2 for e in epochs], val_loss_reduction, width=0.4, 
               label='Val Loss Reduction', alpha=0.7, color='red')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Loss Reduction', fontsize=12)
        ax3.set_title('Loss Reduction per Epoch', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Loss statistics table
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')
        
        stats_text = f"""
CrossEntropy Loss Statistics
{'='*45}

TRAINING LOSS:
  Initial:    {train_loss[0]:.4f}
  Final:      {train_loss[-1]:.4f}
  Minimum:    {min(train_loss):.4f}
  Reduction:  {train_loss[0] - train_loss[-1]:.4f}
  Reduction%: {((train_loss[0] - train_loss[-1])/train_loss[0]*100):.2f}%

VALIDATION LOSS:
  Initial:    {val_loss[0]:.4f}
  Final:      {val_loss[-1]:.4f}
  Minimum:    {min(val_loss):.4f}
  Reduction:  {val_loss[0] - val_loss[-1]:.4f}
  Reduction%: {((val_loss[0] - val_loss[-1])/val_loss[0]*100):.2f}%

GENERALIZATION GAP:
  Initial Gap:  {abs(val_loss[0] - train_loss[0]):.4f}
  Final Gap:    {abs(val_loss[-1] - train_loss[-1]):.4f}
  Avg Gap:      {np.mean([abs(v-t) for v,t in zip(val_loss, train_loss)]):.4f}

LOSS FUNCTION:
  Type: CrossEntropyLoss
  Formula: -Σ y_true * log(y_pred)
  Properties: 
    • Penalizes confident wrong predictions
    • Works well for multi-class classification
    • Combines LogSoftmax + NLLLoss
        """
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        loss_analysis_path = self.output_dir / "loss_analysis.png"
        plt.savefig(loss_analysis_path, dpi=300, bbox_inches='tight')
        print(f"Loss analysis saved to: {loss_analysis_path}")
        plt.close()


def main():
    """Main training function"""
    # Configuration
    TRAIN_DIR = r"C:\Users\Manjunatha H M\Downloads\New Plant Diseases Dataset(Augmented)\train"
    VAL_DIR = r"C:\Users\Manjunatha H M\Desktop\Dev\mini\backend\data\val"  # Use split validation set
    OUTPUT_DIR = r"C:\Users\Manjunatha H M\Desktop\Dev\mini\backend\models"
    
    # Training parameters
    config = {
        'model_name': 'resnet50',  # Using ResNet50 architecture
        'batch_size': 32,
        'num_epochs': 10,
        'learning_rate': 0.001
    }
    
    # Create trainer
    trainer = PlantDiseaseTrainer(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        output_dir=OUTPUT_DIR,
        **config
    )
    
    # Setup
    trainer.setup_data_loaders()
    trainer.setup_model()
    
    # Train
    trainer.train()
    
    # Save
    trainer.save_model()
    trainer.plot_training_curves()
    
    print("\n" + "="*60)
    print("Training pipeline completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
