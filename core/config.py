"""Centralized configuration for OCR pipeline.

This module is the single source of truth for:
- OCR instruction prompts
- Image format definitions
- Model configurations
- Dataset registry
"""

from pathlib import Path

from core.types import DatasetConfig

# ===================================================================
# OCR Prompt - Single source of truth
# ===================================================================
OCR_INSTRUCTION = """Convert the following document to markdown.
Return only the markdown with no explanation text. Do not include delimiters like ```markdown or ```html.

RULES:
  - You must include all information on the page. Do not exclude headers, footers, or subtext.
  - Return tables in an HTML format.
  - Charts & infographics must be interpreted to a markdown format. Prefer table format when applicable.
  - Prefer using ☐ and ☑ for check boxes."""

# ===================================================================
# Image Formats
# ===================================================================
IMAGE_EXTENSIONS = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")
TIFF_EXTENSIONS = ("*.tif", "*.tiff")

# ===================================================================
# Gemini Models
# ===================================================================
GEMINI_MODELS = {
    "1": ("gemini-2.5-pro", "Gemini 2.5 Pro - Stable"),
    "2": ("gemini-3-pro-preview", "Gemini 3 Pro Preview - Best quality"),
    "3": ("gemini-3-flash-preview", "Gemini 3 Flash - Fast"),
}
DEFAULT_MODEL = "gemini-3-pro-preview"

# ===================================================================
# Default Paths
# ===================================================================
DEFAULT_INPUT_DIR = Path("./cropped_headers")
DEFAULT_OUTPUT_JSONL = Path("./ocr_dataset.jsonl")
DEFAULT_DOWNLOADS_DIR = Path("./elog_downloads")


# ===================================================================
# Dataset Registry
# ===================================================================


class DatasetRegistry:
    """Registry for dataset configurations.

    Provides a type-safe way to register and retrieve dataset configurations.
    Follows the Open/Closed Principle - add new datasets without modifying
    existing code.

    Example:
        DatasetRegistry.register("my-dataset", DatasetConfig(
            images_dir=Path("./my-images"),
            output=Path("./my-output.jsonl"),
            description="My custom dataset"
        ))

        config = DatasetRegistry.get("my-dataset")
    """

    _registry: dict[str, DatasetConfig] = {}

    @classmethod
    def register(cls, name: str, config: DatasetConfig) -> None:
        """Register a dataset configuration.

        Args:
            name: Unique identifier for the dataset.
            config: DatasetConfig with paths and metadata.
        """
        cls._registry[name] = config

    @classmethod
    def get(cls, name: str) -> DatasetConfig:
        """Get dataset configuration by name.

        Args:
            name: Dataset identifier.

        Returns:
            DatasetConfig for the requested dataset.

        Raises:
            ValueError: If dataset name is not registered.
        """
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown dataset: {name}. Available: {available}")
        return cls._registry[name]

    @classmethod
    def list_all(cls) -> list[str]:
        """List all registered dataset names.

        Returns:
            List of registered dataset identifiers.
        """
        return list(cls._registry.keys())

    @classmethod
    def as_dict(cls) -> dict[str, DatasetConfig]:
        """Get all datasets as a dictionary.

        Returns:
            Dictionary mapping names to DatasetConfig objects.
        """
        return cls._registry.copy()


# ===================================================================
# Register Default Datasets
# ===================================================================

# Legacy dataset - uses old paths for backward compatibility
DatasetRegistry.register(
    "welllog",
    DatasetConfig(
        images_dir=Path("./cropped_headers"),
        output=Path("./well_log_header.jsonl"),
        description="Legacy welllog dataset (backward compat)",
    ),
)

# Welllog train/eval splits
DatasetRegistry.register(
    "welllog-train",
    DatasetConfig(
        images_dir=Path("./datasets/welllog/train"),
        output=Path("./datasets/welllog/train.jsonl"),
        description="Welllog training set",
    ),
)

DatasetRegistry.register(
    "welllog-eval",
    DatasetConfig(
        images_dir=Path("./datasets/welllog/eval"),
        output=Path("./datasets/welllog/eval.jsonl"),
        description="Welllog evaluation set",
    ),
)

# olmOCR train/eval splits
DatasetRegistry.register(
    "olmocr-train",
    DatasetConfig(
        images_dir=Path("./datasets/olmocr/train"),
        output=Path("./datasets/olmocr/train.jsonl"),
        description="olmOCR training set",
    ),
)

DatasetRegistry.register(
    "olmocr-eval",
    DatasetConfig(
        images_dir=Path("./datasets/olmocr/eval"),
        output=Path("./datasets/olmocr/eval.jsonl"),
        description="olmOCR evaluation set",
    ),
)


# ===================================================================
# Backward Compatibility
# ===================================================================

# Keep DATASETS dict for backward compatibility
DATASETS = {
    name: {"images_dir": cfg.images_dir, "output": cfg.output}
    for name, cfg in DatasetRegistry.as_dict().items()
}


def get_dataset(name: str) -> dict:
    """Get dataset config by name (backward compatible).

    Args:
        name: Dataset identifier.

    Returns:
        Dictionary with 'images_dir' and 'output' keys.

    Raises:
        ValueError: If dataset name is not registered.
    """
    config = DatasetRegistry.get(name)
    return {"images_dir": config.images_dir, "output": config.output}
