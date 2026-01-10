"""Tests for core/config.py - DatasetRegistry and configuration."""

from pathlib import Path

import pytest

from core.config import (
    DATASETS,
    DEFAULT_MODEL,
    IMAGE_EXTENSIONS,
    OCR_INSTRUCTION,
    DatasetRegistry,
    get_dataset,
)
from core.types import DatasetConfig


class TestOCRInstruction:
    """Tests for OCR instruction configuration."""

    def test_instruction_is_string(self):
        """OCR instruction should be a non-empty string."""
        assert isinstance(OCR_INSTRUCTION, str)
        assert len(OCR_INSTRUCTION) > 0

    def test_instruction_contains_markdown(self):
        """OCR instruction should mention markdown conversion."""
        assert "markdown" in OCR_INSTRUCTION.lower()


class TestDatasetRegistry:
    """Tests for DatasetRegistry class."""

    def test_default_datasets_registered(self):
        """Default datasets should be pre-registered."""
        expected = ["welllog", "welllog-train", "welllog-eval", "olmocr-train", "olmocr-eval"]
        for name in expected:
            assert name in DatasetRegistry.list_all()

    def test_get_existing_dataset(self):
        """Getting a registered dataset should return DatasetConfig."""
        config = DatasetRegistry.get("welllog")
        assert isinstance(config, DatasetConfig)
        assert isinstance(config.images_dir, Path)
        assert isinstance(config.output, Path)

    def test_get_unknown_dataset_raises(self):
        """Getting an unknown dataset should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            DatasetRegistry.get("nonexistent-dataset")

    def test_register_new_dataset(self):
        """Registering a new dataset should make it available."""
        DatasetRegistry.register(
            "test-dataset",
            DatasetConfig(
                images_dir=Path("./test/images"),
                output=Path("./test/output.jsonl"),
                description="Test dataset",
            ),
        )
        assert "test-dataset" in DatasetRegistry.list_all()
        config = DatasetRegistry.get("test-dataset")
        assert config.images_dir == Path("./test/images")

    def test_list_all_returns_list(self):
        """list_all should return a list of strings."""
        result = DatasetRegistry.list_all()
        assert isinstance(result, list)
        assert all(isinstance(name, str) for name in result)

    def test_as_dict_returns_copy(self):
        """as_dict should return a copy of the registry."""
        original = DatasetRegistry.as_dict()
        original["fake"] = None  # type: ignore
        assert "fake" not in DatasetRegistry.list_all()


class TestBackwardCompatibility:
    """Tests for backward compatibility with old API."""

    def test_datasets_dict_exists(self):
        """DATASETS dict should exist for backward compat."""
        assert isinstance(DATASETS, dict)
        assert "welllog" in DATASETS

    def test_get_dataset_function(self):
        """get_dataset function should work like before."""
        result = get_dataset("welllog")
        assert isinstance(result, dict)
        assert "images_dir" in result
        assert "output" in result

    def test_get_dataset_raises_for_unknown(self):
        """get_dataset should raise ValueError for unknown dataset."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_dataset("nonexistent")


class TestImageExtensions:
    """Tests for image extension configuration."""

    def test_extensions_are_tuple(self):
        """IMAGE_EXTENSIONS should be a tuple."""
        assert isinstance(IMAGE_EXTENSIONS, tuple)

    def test_common_formats_included(self):
        """Common image formats should be included."""
        formats_str = " ".join(IMAGE_EXTENSIONS)
        assert "tif" in formats_str
        assert "png" in formats_str
        assert "jpg" in formats_str


class TestGeminiModels:
    """Tests for Gemini model configuration."""

    def test_default_model_is_string(self):
        """DEFAULT_MODEL should be a string."""
        assert isinstance(DEFAULT_MODEL, str)
        assert len(DEFAULT_MODEL) > 0
