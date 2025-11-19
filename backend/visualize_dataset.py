import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from pathlib import Path

DATASET_DIR = r"C:\Users\Manjunatha H M\Downloads\New Plant Diseases Dataset(Augmented)"
SPLITS = ["train", "valid"]
SAMPLES_PER_CLASS = 1


def get_class_dirs(split_dir):
    return [d for d in Path(split_dir).iterdir() if d.is_dir()]


def visualize_samples(split="train", samples_per_class=SAMPLES_PER_CLASS, seed=42):
    random.seed(seed)
    split_dir = Path(DATASET_DIR) / split
    class_dirs = get_class_dirs(split_dir)
    print(f"Found {len(class_dirs)} classes in '{split}' split.")

    # Show only 1 sample per class, arrange in a grid (e.g., 6 columns)
    num_classes = len(class_dirs)
    samples_per_class = 1
    ncols = 6
    nrows = (num_classes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    fig.suptitle(f"{split.capitalize()} Set: {num_classes} Classes, 1 Sample Each", fontsize=18, fontweight="bold")

    def format_class_name(name):
        # Replace triple underscores with ' - ', double with ': ', single with space
        name = name.replace('___', ' - ')
        name = name.replace('__', ': ')
        name = name.replace('_', ' ')
        name = name.replace('(', '')
        name = name.replace(')', '')
        name = name.replace('  ', ' ')
        name = name.strip()
        # Capitalize each word
        name = ' '.join([w.capitalize() for w in name.split()])
        # Wrap long names
        if len(name) > 22:
            name = '\n'.join([name[i:i+22] for i in range(0, len(name), 22)])
        return name

    for idx, class_dir in enumerate(class_dirs):
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        ax = axes[idx // ncols, idx % ncols]
        if len(image_files) == 0:
            ax.axis('off')
            continue
        chosen_file = random.choice(image_files)
        img = mpimg.imread(chosen_file)
        ax.imshow(img)
        ax.set_title(format_class_name(class_dir.name), fontsize=11, fontweight="bold")
        ax.axis('off')

    # Hide unused axes
    for idx in range(num_classes, nrows * ncols):
        axes[idx // ncols, idx % ncols].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def visualize_class_distribution(split="train"):
    split_dir = Path(DATASET_DIR) / split
    class_dirs = get_class_dirs(split_dir)
    class_counts = {d.name: len(list(d.glob("*.jpg"))) + len(list(d.glob("*.png"))) for d in class_dirs}
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    plt.figure(figsize=(12, 10))
    plt.barh(classes, counts, color="skyblue")
    plt.xlabel("Number of Images")
    plt.ylabel("Class Name")
    plt.title(f"Class Distribution in {split.capitalize()} Set")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Visualizing dataset samples and class distribution...")
    visualize_class_distribution("train")
    visualize_samples("train", samples_per_class=5)
    # You can also visualize 'val' or 'test' splits by changing the argument
    # visualize_class_distribution("val")
    # visualize_samples("val", samples_per_class=5)
