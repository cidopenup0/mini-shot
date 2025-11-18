import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import numpy as np


def plot_training_history(history_path: str, save_path: str = None):
    """
    Plot training history (loss and accuracy curves)
    
    Args:
        history_path: Path to training_history.json
        save_path: Optional path to save the plot
    """
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = history['epochs']
    
    # Plot loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot accuracy
    train_acc = [acc * 100 for acc in history['train_acc']]
    val_acc = [acc * 100 for acc in history['val_acc']]
    
    axes[1].plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    else:
        plt.show()


def plot_per_class_accuracy(results_path: str, save_path: str = None):
    """
    Plot per-class accuracy from test results
    
    Args:
        results_path: Path to test_results.json
        save_path: Optional path to save the plot
    """
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    per_class = results['per_class_accuracy']
    
    # Sort by accuracy
    sorted_classes = sorted(per_class.items(), key=lambda x: x[1], reverse=True)
    classes = [c[0] for c in sorted_classes]
    accuracies = [c[1] * 100 for c in sorted_classes]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create horizontal bar chart
    colors = ['#2ecc71' if acc >= 90 else '#f39c12' if acc >= 80 else '#e74c3c' for acc in accuracies]
    bars = ax.barh(classes, accuracies, color=colors, alpha=0.8)
    
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_ylabel('Disease Class', fontsize=12)
    ax.set_title(f'Per-Class Test Accuracy\nOverall: {results["overall_accuracy"]*100:.2f}%', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax.text(acc + 1, i, f'{acc:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class accuracy plot saved to: {save_path}")
    else:
        plt.show()


def generate_all_plots(model_dir: str = "models"):
    """
    Generate all visualization plots
    
    Args:
        model_dir: Directory containing training results
    """
    model_path = Path(model_dir)
    
    print("Generating visualization plots...")
    
    # Plot training history
    history_path = model_path / "training_history.json"
    if history_path.exists():
        plot_training_history(
            str(history_path),
            save_path=str(model_path / "training_curves.png")
        )
    else:
        print(f"Training history not found at {history_path}")
    
    # Plot per-class accuracy
    results_path = model_path / "test_results.json"
    if results_path.exists():
        plot_per_class_accuracy(
            str(results_path),
            save_path=str(model_path / "per_class_accuracy.png")
        )
    else:
        print(f"Test results not found at {results_path}")
    
    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    generate_all_plots()
