import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import numpy as np


def plot_training_history(history_path: str, save_path: str = None):
    """
    Plot training history (loss, accuracy, and learning rate)
    
    Args:
        history_path: Path to training_history.json
        save_path: Optional path to save the plot
    """
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Training Metrics History', fontsize=16, fontweight='bold')
    
    epochs = history['epochs']
    
    # Plot 1: Loss (Cross-Entropy)
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training', linewidth=2, marker='o', markersize=4)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2, marker='s', markersize=4)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Cross-Entropy Loss', fontsize=11)
    axes[0].set_title('Loss (Cross-Entropy)', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot 2: Accuracy
    # Check if accuracy values are already in percentage
    train_acc = history['train_acc']
    val_acc = history['val_acc']
    if max(train_acc + val_acc) > 1.0:
        # Already in percentage
        train_acc_plot = train_acc
        val_acc_plot = val_acc
    else:
        # Convert to percentage
        train_acc_plot = [acc * 100 for acc in train_acc]
        val_acc_plot = [acc * 100 for acc in val_acc]
    
    axes[1].plot(epochs, train_acc_plot, 'b-', label='Training', linewidth=2, marker='o', markersize=4)
    axes[1].plot(epochs, val_acc_plot, 'r-', label='Validation', linewidth=2, marker='s', markersize=4)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Accuracy (%)', fontsize=11)
    axes[1].set_title('Accuracy', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Plot 3: Learning Rate
    if 'learning_rates' in history:
        axes[2].plot(epochs, history['learning_rates'], 'g-', linewidth=2, marker='d', markersize=4)
        axes[2].set_xlabel('Epoch', fontsize=11)
        axes[2].set_ylabel('Learning Rate', fontsize=11)
        axes[2].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        axes[2].set_yscale('log')
        axes[2].grid(alpha=0.3)
    else:
        # Hide the third plot if no learning rate data
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    else:
        plt.show()


def plot_per_class_metrics(results_path: str, save_path: str = None):
    """
    Plot per-class metrics from test results
    
    Args:
        results_path: Path to test_results.json
        save_path: Optional path to save the plot
    """
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    per_class = results['per_class_metrics']
    
    # Extract metrics
    classes = list(per_class.keys())
    accuracies = [per_class[c]['accuracy'] * 100 for c in classes]
    precisions = [per_class[c]['precision'] * 100 for c in classes]
    recalls = [per_class[c]['recall'] * 100 for c in classes]
    f1_scores = [per_class[c]['f1'] * 100 for c in classes]
    
    # Sort by F1-score
    sorted_indices = sorted(range(len(f1_scores)), key=lambda i: f1_scores[i], reverse=True)
    classes = [classes[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    precisions = [precisions[i] for i in sorted_indices]
    recalls = [recalls[i] for i in sorted_indices]
    f1_scores = [f1_scores[i] for i in sorted_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Set up the bar positions
    y_pos = np.arange(len(classes))
    bar_width = 0.2
    
    # Create bars
    bars1 = ax.barh(y_pos - 1.5*bar_width, accuracies, bar_width, label='Accuracy', color='#3498db', alpha=0.8)
    bars2 = ax.barh(y_pos - 0.5*bar_width, precisions, bar_width, label='Precision', color='#2ecc71', alpha=0.8)
    bars3 = ax.barh(y_pos + 0.5*bar_width, recalls, bar_width, label='Recall', color='#f39c12', alpha=0.8)
    bars4 = ax.barh(y_pos + 1.5*bar_width, f1_scores, bar_width, label='F1-Score', color='#9b59b6', alpha=0.8)
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel('Score (%)', fontsize=12)
    ax.set_ylabel('Disease Class', fontsize=12)
    ax.set_title(f'Per-Class Test Metrics\nOverall - Acc: {results["overall_accuracy"]*100:.2f}% | P: {results["overall_precision"]*100:.2f}% | R: {results["overall_recall"]*100:.2f}% | F1: {results["overall_f1"]*100:.2f}%', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, 105)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class metrics plot saved to: {save_path}")
    else:
        plt.show()


def plot_confusion_matrix(results_path: str, class_names_path: str, save_path: str = None):
    """
    Plot confusion matrix heatmap
    
    Args:
        results_path: Path to test_results.json
        class_names_path: Path to class_names.json
        save_path: Optional path to save the plot
    """
    # Load results and class names
    with open(results_path, 'r') as f:
        results = json.load(f)
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    
    cm = np.array(results['confusion_matrix'])
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 18))
    
    # Create heatmap
    sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count'}, ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to: {save_path}")
    else:
        plt.show()


def generate_all_plots(model_dir: str = "models"):
    """
    Generate all visualization plots
    
    Args:
        model_dir: Directory containing training results
    """
    model_path = Path(model_dir)
    
    print("Generating comprehensive visualization plots...")
    print("="*60)
    
    # Try both old and new file names
    history_files = [
        model_path / "training_history_new.json",
        model_path / "training_history.json"
    ]
    class_names_files = [
        model_path / "class_names_new.json",
        model_path / "class_names.json"
    ]
    
    # Plot training history
    history_path = None
    for file_path in history_files:
        if file_path.exists():
            history_path = file_path
            break
    
    if history_path:
        print(f"\n1. Generating training metrics history from {history_path.name}...")
        plot_training_history(
            str(history_path),
            save_path=str(model_path / "training_metrics_new.png")
        )
    else:
        print(f"⚠ Training history not found. Tried: {', '.join([f.name for f in history_files])}")
    
    # Plot per-class metrics
    results_path = model_path / "test_results.json"
    
    # Find class names file
    class_names_path = None
    for file_path in class_names_files:
        if file_path.exists():
            class_names_path = file_path
            break
    
    if results_path.exists():
        print(f"\n2. Generating per-class metrics from {results_path.name}...")
        plot_per_class_metrics(
            str(results_path),
            save_path=str(model_path / "per_class_metrics_new.png")
        )
        
        if class_names_path:
            print(f"\n3. Generating confusion matrix using {class_names_path.name}...")
            plot_confusion_matrix(
                str(results_path),
                str(class_names_path),
                save_path=str(model_path / "confusion_matrix_new.png")
            )
        else:
            print(f"⚠ Class names not found. Tried: {', '.join([f.name for f in class_names_files])}")
    else:
        print(f"⚠ Test results not found at {results_path}")
    
    print("\n" + "="*60)
    print("✓ All plots generated successfully!")
    print("="*60)


if __name__ == "__main__":
    generate_all_plots()
