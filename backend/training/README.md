# Model Training Guide

## Prerequisites

1. Dataset must be split into train/val/test:
```bash
python data_preparation/split_dataset.py
```

2. Install training dependencies:
```bash
pip install torch torchvision matplotlib seaborn
```

## Training the Model

### Quick Start

```bash
python training/train_model.py
```

### Configuration

Edit the configuration in `train_model.py`:

```python
# Training parameters
NUM_EPOCHS = 25              # Number of training epochs
BATCH_SIZE = 32              # Batch size
LEARNING_RATE = 0.001        # Initial learning rate

# Model architecture options:
MODEL_ARCHITECTURE = 'resnet50'  # resnet50, resnet18, or efficientnet_b0
```

### Model Architectures

- **resnet50** (default) - Best accuracy, slower training
- **resnet18** - Faster training, good accuracy
- **efficientnet_b0** - Balanced speed and accuracy

## Training Process

The training script will:

1. ✅ Load and augment training data
2. ✅ Build CNN model with pretrained weights
3. ✅ Train for specified epochs
4. ✅ Validate after each epoch
5. ✅ Save best model based on validation accuracy
6. ✅ Test on test set
7. ✅ Save training history and results

## Output Files

After training, the following files are created in `models/`:

```
models/
├── plant_disease_model.pt      # Best model (highest val accuracy)
├── plant_disease_model_final.pt # Final model after all epochs
├── training_history.json       # Loss and accuracy per epoch
├── test_results.json          # Test set performance
├── class_names.json           # List of disease classes
├── training_curves.png        # Visualization of training progress
└── per_class_accuracy.png     # Per-class test accuracy
```

## Visualize Results

Generate plots after training:

```bash
python training/visualize_results.py
```

This creates:
- **training_curves.png** - Loss and accuracy curves
- **per_class_accuracy.png** - Bar chart of per-class accuracy

## Training Tips

### GPU Acceleration

The script automatically uses GPU if available:
```
Using device: cuda
```

For CPU-only training:
```
Using device: cpu
```

### Data Augmentation

Training uses:
- Random horizontal flip
- Random rotation (±10°)
- Color jitter (brightness, contrast, saturation)
- Normalization (ImageNet stats)

### Learning Rate Scheduling

- Uses ReduceLROnPlateau scheduler
- Reduces LR by 10x when validation loss plateaus
- Patience: 3 epochs

### Early Stopping

Best model is saved when validation accuracy improves:
```
✓ Best model saved! (Val Acc: 95.23%)
```

## Monitor Training

Watch for:
- **Overfitting**: Train accuracy >> Val accuracy
- **Underfitting**: Both accuracies are low
- **Good fit**: Train and val accuracies are close and high

## Example Output

```
Epoch [10/25] - 45.32s
  Train Loss: 0.1234 | Train Acc: 96.45%
  Val Loss:   0.2156 | Val Acc:   94.23%
  ✓ Best model saved! (Val Acc: 94.23%)

Training Complete!
Total time: 18.75 minutes
Best validation accuracy: 96.78%

Testing Model
Overall Test Accuracy: 95.67%
```

## Troubleshooting

### Out of Memory (GPU)
Reduce `BATCH_SIZE` to 16 or 8

### Slow Training
- Use GPU if available
- Reduce `num_workers` in DataLoader
- Use smaller model (resnet18)

### Poor Accuracy
- Increase `NUM_EPOCHS`
- Try different model architecture
- Check data quality and balance

## Next Steps

After training:
1. Check `test_results.json` for accuracy
2. Review `training_curves.png` for overfitting
3. Update `app/config.py` CLASS_NAMES if needed
4. Run backend: `python main.py`
