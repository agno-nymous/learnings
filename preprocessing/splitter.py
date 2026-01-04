"""
Dataset splitter for train/eval splits.
"""
import random
import shutil
from pathlib import Path

from core.config import IMAGE_EXTENSIONS


def split_dataset(
    source_dir: Path,
    train_dir: Path,
    eval_dir: Path,
    train_ratio: float = 0.8,
    copy: bool = False,
    seed: int = 42,
):
    """
    Split images into train/eval sets.
    
    Args:
        source_dir: Directory containing images
        train_dir: Destination for training images
        eval_dir: Destination for eval images
        train_ratio: Fraction for training (default 80%)
        copy: If True, copy files; else move them
        seed: Random seed for reproducibility
    """
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(source_dir.glob(ext))
    
    if not images:
        print(f"No images found in {source_dir}")
        return 0, 0
    
    # Shuffle deterministically
    random.seed(seed)
    random.shuffle(images)
    
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    eval_images = images[split_idx:]
    
    op = shutil.copy2 if copy else shutil.move
    op_name = "Copying" if copy else "Moving"
    
    print(f"{op_name} {len(train_images)} images to train/")
    for img in train_images:
        op(img, train_dir / img.name)
    
    print(f"{op_name} {len(eval_images)} images to eval/")
    for img in eval_images:
        op(img, eval_dir / img.name)
    
    print(f"Done: {len(train_images)} train, {len(eval_images)} eval")
    return len(train_images), len(eval_images)


def split_welllog(source_dir: Path = None, train_ratio: float = 0.8):
    """Split welllog cropped headers into train/eval."""
    if source_dir is None:
        source_dir = Path("./cropped_headers")
    
    return split_dataset(
        source_dir=source_dir,
        train_dir=Path("./datasets/welllog/train"),
        eval_dir=Path("./datasets/welllog/eval"),
        train_ratio=train_ratio,
        copy=False,  # Move to save space
    )


if __name__ == "__main__":
    split_welllog()
