import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random
from typing import Tuple, List
import json


class DatasetSplitter:
    """
    Splits the Plant Disease dataset into train, validation, and test sets
    """
    
    def __init__(
        self,
        source_dir: str,
        output_dir: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ):
        """
        Initialize dataset splitter
        
        Args:
            source_dir: Path to the original dataset directory
            output_dir: Path where split datasets will be saved
            train_ratio: Proportion of data for training (default: 0.7)
            val_ratio: Proportion of data for validation (default: 0.15)
            test_ratio: Proportion of data for testing (default: 0.15)
            seed: Random seed for reproducibility
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        # Validate ratios
        if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
            raise ValueError("Train, val, and test ratios must sum to 1.0")
        
        # Set random seed
        random.seed(seed)
    
    def get_class_directories(self) -> List[Path]:
        """Get all class directories from source"""
        class_dirs = [d for d in self.source_dir.iterdir() if d.is_dir()]
        return class_dirs
    
    def split_class_images(self, class_dir: Path) -> Tuple[List[Path], List[Path], List[Path]]:
        """
        Split images of a single class into train/val/test
        
        Args:
            class_dir: Path to class directory
        
        Returns:
            Tuple of (train_images, val_images, test_images)
        """
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        images = [
            f for f in class_dir.iterdir()
            if f.suffix in image_extensions
        ]
        
        # Shuffle images
        random.shuffle(images)
        
        # Calculate split indices
        total = len(images)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)
        
        # Split
        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]
        
        return train_images, val_images, test_images
    
    def copy_images(self, images: List[Path], dest_dir: Path, class_name: str) -> None:
        """
        Copy images to destination directory
        
        Args:
            images: List of image paths to copy
            dest_dir: Destination directory
            class_name: Name of the class
        """
        # Create class directory
        class_dest = dest_dir / class_name
        class_dest.mkdir(parents=True, exist_ok=True)
        
        # Copy images
        for img in images:
            dest_path = class_dest / img.name
            shutil.copy2(img, dest_path)
    
    def split_dataset(self) -> dict:
        """
        Perform the complete dataset split
        
        Returns:
            Dictionary with split statistics
        """
        print("Starting dataset split...")
        print(f"Source: {self.source_dir}")
        print(f"Output: {self.output_dir}")
        print(f"Ratios - Train: {self.train_ratio}, Val: {self.val_ratio}, Test: {self.test_ratio}\n")
        
        # Create output directories
        train_dir = self.output_dir / "train"
        val_dir = self.output_dir / "val"
        test_dir = self.output_dir / "test"
        
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all class directories
        class_dirs = self.get_class_directories()
        
        if not class_dirs:
            raise ValueError(f"No class directories found in {self.source_dir}")
        
        print(f"Found {len(class_dirs)} classes\n")
        
        # Statistics
        stats = {
            "classes": [],
            "total_train": 0,
            "total_val": 0,
            "total_test": 0,
            "total_images": 0
        }
        
        # Process each class
        for class_dir in class_dirs:
            class_name = class_dir.name
            print(f"Processing class: {class_name}")
            
            # Split images
            train_images, val_images, test_images = self.split_class_images(class_dir)
            
            # Copy to respective directories
            self.copy_images(train_images, train_dir, class_name)
            self.copy_images(val_images, val_dir, class_name)
            self.copy_images(test_images, test_dir, class_name)
            
            # Update statistics
            class_stats = {
                "name": class_name,
                "train": len(train_images),
                "val": len(val_images),
                "test": len(test_images),
                "total": len(train_images) + len(val_images) + len(test_images)
            }
            
            stats["classes"].append(class_stats)
            stats["total_train"] += len(train_images)
            stats["total_val"] += len(val_images)
            stats["total_test"] += len(test_images)
            stats["total_images"] += class_stats["total"]
            
            print(f"  Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}\n")
        
        # Save statistics
        stats_file = self.output_dir / "split_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print("\n" + "="*50)
        print("Dataset Split Complete!")
        print("="*50)
        print(f"Total images: {stats['total_images']}")
        print(f"Training set: {stats['total_train']} ({stats['total_train']/stats['total_images']*100:.1f}%)")
        print(f"Validation set: {stats['total_val']} ({stats['total_val']/stats['total_images']*100:.1f}%)")
        print(f"Test set: {stats['total_test']} ({stats['total_test']/stats['total_images']*100:.1f}%)")
        print(f"\nStatistics saved to: {stats_file}")
        
        return stats
    
    def verify_split(self) -> bool:
        """
        Verify the split was successful
        
        Returns:
            True if verification passed
        """
        train_dir = self.output_dir / "train"
        val_dir = self.output_dir / "val"
        test_dir = self.output_dir / "test"
        
        if not all([train_dir.exists(), val_dir.exists(), test_dir.exists()]):
            print("Error: Not all split directories exist")
            return False
        
        train_classes = set(d.name for d in train_dir.iterdir() if d.is_dir())
        val_classes = set(d.name for d in val_dir.iterdir() if d.is_dir())
        test_classes = set(d.name for d in test_dir.iterdir() if d.is_dir())
        
        if not (train_classes == val_classes == test_classes):
            print("Error: Class mismatch between splits")
            return False
        
        print("\nVerification passed!")
        print(f"All {len(train_classes)} classes present in train, val, and test sets")
        return True


def main():
    """Main execution function"""
    # Configuration
    SOURCE_DIR = "data/raw/PlantVillage"  # Update with your dataset path
    OUTPUT_DIR = "data/processed"
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    print("Plant Disease Dataset Splitter")
    print("="*50 + "\n")
    
    # Create splitter
    splitter = DatasetSplitter(
        source_dir=SOURCE_DIR,
        output_dir=OUTPUT_DIR,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=42
    )
    
    # Perform split
    try:
        stats = splitter.split_dataset()
        
        # Verify split
        splitter.verify_split()
        
        # Print class list
        print("\nClass Names:")
        print("-" * 50)
        for i, class_info in enumerate(stats["classes"], 1):
            print(f"{i:2d}. {class_info['name']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
