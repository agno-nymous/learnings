"""Tests for dataset loading utilities."""

import pytest
from training.dataset import LazyVisionDataset, b64_to_image, load_dataset
from datasets import load_dataset as hf_load_dataset
from PIL import Image


def test_b64_to_image():
    # Tiny valid 1x1 PNG base64
    b64_str = "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCwAB//2Q=="
    img = b64_to_image(b64_str)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_lazy_vision_dataset_length():
    raw_dataset = hf_load_dataset("wrath/well-log-headers-ocr", split="train[:5]")
    dataset = LazyVisionDataset(raw_dataset)
    assert len(dataset) == 5


def test_lazy_vision_dataset_getitem():
    raw_dataset = hf_load_dataset("wrath/well-log-headers-ocr", split="train[:1]")
    dataset = LazyVisionDataset(raw_dataset)
    sample = dataset[0]
    assert "messages" in sample
    assert "images" in sample
    assert len(sample["messages"]) == 2
    assert sample["messages"][0]["role"] == "user"
    assert sample["messages"][1]["role"] == "assistant"


def test_load_dataset():
    train_ds, eval_ds = load_dataset("wrath/well-log-headers-ocr", train_subset=10, eval_subset=5)
    assert len(train_ds) == 10
    assert len(eval_ds) == 5
