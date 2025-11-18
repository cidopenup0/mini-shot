import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import Counter


class DatasetAnalyzer:
    """
    Analyzes and visualizes the split dataset
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize analyzer
        
        Args:
            data_dir: Path to the processed data directory
        """
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / "train"
        self.val_dir = self.data_dir / "val"
        self.test_dir = self.data_dir / "test"
    
    def count_images_per_class(self, split_dir: Path) -> dict:
        """
        Count images per class in a split
        
        Args:
            split_dir: Path to split directory (train/val/test)
        
        Returns:
            Dictionary mapping class names to image counts
        """
        counts = {}
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
                images = [f for f in class_dir.iterdir() if f.suffix in image_extensions]
                counts[class_dir.name] = len(images)
        return counts
    
    def analyze_distribution(self) -> dict:
        """
        Analyze the distribution across all splits
        
        Returns:
            Dictionary with distribution statistics
        """
        train_counts = self.count_images_per_class(self.train_dir)
        val_counts = self.count_images_per_class(self.val_dir)
        test_counts = self.count_images_per_class(self.test_dir)
        
        analysis = {
            "train": train_counts,
            "val": val_counts,
            "test": test_counts,
            "summary": {
                "total_train": sum(train_counts.values()),
                "total_val": sum(val_counts.values()),
                "total_test": sum(test_counts.values()),
                "num_classes": len(train_counts)
            }
        }
        
        return analysis
    
    def plot_distribution(self, save_path: str = None):
        """
        Plot the distribution of images across classes
        
        Args:
            save_path: Optional path to save the plot
        """
        analysis = self.analyze_distribution()
        
        # Prepare data for plotting
        classes = sorted(analysis["train"].keys())
        train_counts = [analysis["train"][c] for c in classes]
        val_counts = [analysis["val"][c] for c in classes]
        test_counts = [analysis["test"][c] for c in classes]
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        
        # Plot 1: Stacked bar chart
        x = range(len(classes))
        axes[0].bar(x, train_counts, label='Train', alpha=0.8)
        axes[0].bar(x, val_counts, bottom=train_counts, label='Validation', alpha=0.8)
        axes[0].bar(x, test_counts, bottom=[t+v for t,v in zip(train_counts, val_counts)], 
                    label='Test', alpha=0.8)
        axes[0].set_xlabel('Classes', fontsize=12)
        axes[0].set_ylabel('Number of Images', fontsize=12)
        axes[0].set_title('Distribution of Images Across Classes', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(classes, rotation=90, ha='right', fontsize=8)
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Plot 2: Split comparison
        summary = analysis["summary"]
        splits = ['Train', 'Validation', 'Test']
        totals = [summary['total_train'], summary['total_val'], summary['total_test']]
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        axes[1].bar(splits, totals, color=colors, alpha=0.8)
        axes[1].set_ylabel('Number of Images', fontsize=12)
        axes[1].set_title('Dataset Split Distribution', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (split, total) in enumerate(zip(splits, totals)):
            axes[1].text(i, total + 100, str(total), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()
    
    def print_summary(self):
        """Print a summary of the dataset"""
        analysis = self.analyze_distribution()
        summary = analysis["summary"]
        
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        print(f"Number of classes: {summary['num_classes']}")
        print(f"Total images: {summary['total_train'] + summary['total_val'] + summary['total_test']}")
        print("\nSplit Distribution:")
        print(f"  Training:   {summary['total_train']:5d} images ({summary['total_train']/(summary['total_train']+summary['total_val']+summary['total_test'])*100:.1f}%)")
        print(f"  Validation: {summary['total_val']:5d} images ({summary['total_val']/(summary['total_train']+summary['total_val']+summary['total_test'])*100:.1f}%)")
        print(f"  Test:       {summary['total_test']:5d} images ({summary['total_test']/(summary['total_train']+summary['total_val']+summary['total_test'])*100:.1f}%)")
        
        print("\nPer-Class Distribution (Train/Val/Test):")
        print("-" * 60)
        for class_name in sorted(analysis["train"].keys()):
            train = analysis["train"][class_name]
            val = analysis["val"][class_name]
            test = analysis["test"][class_name]
            total = train + val + test
            print(f"{class_name:40s} {train:4d} / {val:4d} / {test:4d}  (Total: {total})")


def main():
    """Main execution function"""
    DATA_DIR = "data/processed"
    
    print("Dataset Analysis")
    print("="*60 + "\n")
    
    analyzer = DatasetAnalyzer(DATA_DIR)
    
    # Print summary
    analyzer.print_summary()
    
    # Create visualization
    print("\nGenerating distribution plots...")
    analyzer.plot_distribution(save_path="data/processed/dataset_distribution.png")


if __name__ == "__main__":
    main()
