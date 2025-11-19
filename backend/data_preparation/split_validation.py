"""
Simple dataset splitter for the augmented Plant Disease dataset
Splits the existing validation set into validation and test sets
"""

import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random
import json


def split_val_to_val_test(source_val_dir, output_base_dir, test_ratio=0.5, seed=42):
    """
    Split existing validation into val + test
    
    Current structure:
        Downloads/New Plant Diseases Dataset(Augmented)/
            ├── train/          (70,295 images - keep as is)
            └── valid/          (17,572 images - split this)
    
    New structure:
        backend/data/
            ├── val/            (~8,786 images)
            └── test/           (~8,786 images)
    
    Args:
        source_val_dir: Path to current validation folder
        output_base_dir: Where to create val/ and test/ folders
        test_ratio: What % to use for test (0.5 = 50/50 split)
        seed: Random seed
    """
    random.seed(seed)
    
    source = Path(source_val_dir)
    output = Path(output_base_dir)
    
    val_out = output / "val"
    test_out = output / "test"
    
    # Create output dirs
    val_out.mkdir(parents=True, exist_ok=True)
    test_out.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Dataset Splitting")
    print("="*60)
    print(f"Source: {source}")
    print(f"Output: {output}")
    print(f"Test ratio: {test_ratio*100}%")
    print("="*60)
    print()
    
    # Get all disease classes
    classes = sorted([d.name for d in source.iterdir() if d.is_dir()])
    print(f"Found {len(classes)} disease classes\n")
    
    stats = {"val": {}, "test": {}}
    total_val = 0
    total_test = 0
    
    for i, class_name in enumerate(classes, 1):
        print(f"[{i}/{len(classes)}] Processing: {class_name}")
        
        class_dir = source / class_name
        
        # Get all images
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG")) + \
                 list(class_dir.glob("*.png")) + list(class_dir.glob("*.PNG"))
        
        if not images:
            print(f"  ⚠️  No images found, skipping")
            continue
        
        # Split into val and test
        val_imgs, test_imgs = train_test_split(
            images, 
            test_size=test_ratio, 
            random_state=seed
        )
        
        # Create class folders
        (val_out / class_name).mkdir(exist_ok=True)
        (test_out / class_name).mkdir(exist_ok=True)
        
        # Copy files
        for img in val_imgs:
            shutil.copy2(img, val_out / class_name / img.name)
        
        for img in test_imgs:
            shutil.copy2(img, test_out / class_name / img.name)
        
        # Update stats
        stats["val"][class_name] = len(val_imgs)
        stats["test"][class_name] = len(test_imgs)
        total_val += len(val_imgs)
        total_test += len(test_imgs)
        
        print(f"  ✓ Val: {len(val_imgs):4d} | Test: {len(test_imgs):4d}")
    
    print()
    print("="*60)
    print("Summary")
    print("="*60)
    print(f"Validation images: {total_val:,}")
    print(f"Test images: {total_test:,}")
    print(f"Total processed: {total_val + total_test:,}")
    print()
    print(f"✓ Validation folder: {val_out}")
    print(f"✓ Test folder: {test_out}")
    
    # Save statistics
    summary = {
        "total_val": total_val,
        "total_test": total_test,
        "num_classes": len(classes),
        "test_ratio": test_ratio,
        "seed": seed,
        "classes": stats
    }
    
    summary_file = output / "split_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Summary saved: {summary_file}")
    print("="*60)


def main():
    """Run the dataset split"""
    
    # Paths
    SOURCE_VALID = r"C:\Users\Manjunatha H M\Downloads\New Plant Diseases Dataset(Augmented)\valid"
    OUTPUT_DIR = r"C:\Users\Manjunatha H M\Desktop\Dev\mini\backend\data"
    
    # Split validation into val + test (50/50)
    split_val_to_val_test(
        source_val_dir=SOURCE_VALID,
        output_base_dir=OUTPUT_DIR,
        test_ratio=0.5,
        seed=42
    )
    
    print("\n✅ Dataset split complete!")
    print("\nNext steps:")
    print("1. Training data is still at:")
    print(f"   C:\\Users\\Manjunatha H M\\Downloads\\New Plant Diseases Dataset(Augmented)\\train")
    print("2. Use the new val/test folders for training/evaluation")
    print("3. Run: python training/train_full_model.py")


if __name__ == "__main__":
    main()
