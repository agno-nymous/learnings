"""Dataset loading utilities for Qwen3-VL training."""

import base64
from io import BytesIO
from typing import Any

from PIL import Image

from datasets import load_dataset as hf_load_dataset


def b64_to_image(b64_str: str) -> Image.Image:
    """Convert base64 string to PIL RGB Image.

    Args:
        b64_str: Base64-encoded image string.

    Returns:
        PIL Image in RGB mode.
    """
    return Image.open(BytesIO(base64.b64decode(b64_str))).convert("RGB")


class LazyVisionDataset:
    """Memory-efficient dataset with lazy image loading.

    Defers image decoding until __getitem__ is called to avoid
    loading all images into RAM at once.
    """

    def __init__(self, hf_dataset: Any):
        """Initialize with HuggingFace dataset.

        Args:
            hf_dataset: HuggingFace dataset with 'image_base64', 'instruction', 'answer' fields.
        """
        self.data = hf_dataset

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """Get a single training sample.

        Returns:
            Dict with 'messages' (for Unsloth) and 'images' (lazy-loaded PIL Images).
        """
        sample = self.data[idx]
        img = b64_to_image(sample["image_base64"])

        # Format required by PaddleOCR/Unsloth - image must be included in content
        return {
            "images": [img],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": sample["instruction"]},
                        {"type": "image", "image": img},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample["answer"]}],
                },
            ],
        }


def load_dataset(
    dataset_name: str, train_subset: int = -1, eval_subset: int = -1
) -> tuple[LazyVisionDataset, LazyVisionDataset]:
    """Load training and evaluation datasets.

    Args:
        dataset_name: HuggingFace dataset identifier.
        train_subset: Number of training samples to use (-1 for all).
        eval_subset: Number of eval samples to use (-1 for all).

    Returns:
        Tuple of (train_dataset, eval_dataset) as LazyVisionDataset.
    """
    raw_dataset = hf_load_dataset(dataset_name)

    train_split = raw_dataset["train"]
    eval_split = raw_dataset["eval"]

    if train_subset > 0:
        train_split = train_split.select(range(min(train_subset, len(train_split))))
    if eval_subset > 0:
        eval_split = eval_split.select(range(min(eval_subset, len(eval_split))))

    return LazyVisionDataset(train_split), LazyVisionDataset(eval_split)
