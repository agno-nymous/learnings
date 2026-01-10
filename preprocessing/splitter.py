"""Dataset splitter for train/eval splits.

Provides a Preprocessor implementation for splitting image datasets
into training and evaluation sets with deterministic randomization.
"""

import random
import shutil
from pathlib import Path

from core.config import IMAGE_EXTENSIONS
from core.types import ProcessingResult

from .base import Preprocessor, PreprocessorRegistry


class Splitter(Preprocessor):
    """Split images into train/eval sets.

    Divides a directory of images into training and evaluation sets
    with configurable split ratio and deterministic shuffling.

    Attributes:
        train_ratio: Fraction of images for training (default 0.8).
        copy: If True, copy files; if False, move them.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        train_ratio: float = 0.8,
        copy: bool = False,
        seed: int = 42,
    ) -> None:
        """Initialize the splitter.

        Args:
            train_ratio: Fraction of images for training (0.0-1.0).
            copy: If True, copy files; if False, move them.
            seed: Random seed for reproducibility.
        """
        self.train_ratio = train_ratio
        self.copy = copy
        self.seed = seed

    @property
    def name(self) -> str:
        """Human-readable name."""
        return "Dataset Splitter"

    def _find_images(self, source_dir: Path) -> list[Path]:
        """Find all image files in the source directory.

        Args:
            source_dir: Directory to search.

        Returns:
            List of image file paths.
        """
        images = []
        for ext in IMAGE_EXTENSIONS:
            images.extend(source_dir.glob(ext))
        return images

    def process(
        self,
        input_dir: Path,
        output_dir: Path | None = None,
        workers: int = 4,  # noqa: ARG002
    ) -> ProcessingResult:
        """Split images into train/eval directories.

        Note: output_dir should be the parent directory containing
        'train' and 'eval' subdirectories.

        Args:
            input_dir: Directory containing source images.
            output_dir: Parent directory for train/eval subdirs.
            workers: Not used (splitting is sequential).

        Returns:
            ProcessingResult with split statistics.
        """
        input_dir = Path(input_dir)

        if output_dir is None:
            # Default to creating train/eval in same directory
            output_dir = input_dir.parent

        output_dir = Path(output_dir)
        train_dir = output_dir / "train"
        eval_dir = output_dir / "eval"

        train_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)

        # Find all images
        images = self._find_images(input_dir)

        if not images:
            print(f"No images found in {input_dir}")
            return ProcessingResult(processed=0)

        # Shuffle deterministically
        random.seed(self.seed)
        random.shuffle(images)

        split_idx = int(len(images) * self.train_ratio)
        train_images = images[:split_idx]
        eval_images = images[split_idx:]

        op = shutil.copy2 if self.copy else shutil.move
        op_name = "Copying" if self.copy else "Moving"

        print(f"{op_name} {len(train_images)} images to train/")
        for img in train_images:
            op(img, train_dir / img.name)

        print(f"{op_name} {len(eval_images)} images to eval/")
        for img in eval_images:
            op(img, eval_dir / img.name)

        total = len(train_images) + len(eval_images)
        print(f"Done: {len(train_images)} train, {len(eval_images)} eval")

        return ProcessingResult(processed=total)


# Register with the registry
PreprocessorRegistry.register("split", Splitter)


# Backward-compatible functions
def split_dataset(
    source_dir: Path,
    train_dir: Path,
    eval_dir: Path,
    train_ratio: float = 0.8,
    copy: bool = False,
    seed: int = 42,
) -> tuple[int, int]:
    """Split images into train/eval sets.

    This is a backward-compatible wrapper.

    Args:
        source_dir: Directory containing images
        train_dir: Destination for training images
        eval_dir: Destination for eval images
        train_ratio: Fraction for training (default 80%)
        copy: If True, copy files; else move them
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_count, eval_count)
    """
    # Use the direct implementation for full control over paths
    train_dir = Path(train_dir)
    eval_dir = Path(eval_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(Path(source_dir).glob(ext))

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


def split_welllog(source_dir: Path = None, train_ratio: float = 0.8) -> tuple[int, int]:
    """Split welllog cropped headers into train/eval.

    Args:
        source_dir: Source directory (default: ./cropped_headers)
        train_ratio: Fraction for training

    Returns:
        Tuple of (train_count, eval_count)
    """
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
