"""Centralized configuration for OCR pipeline."""
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# OCR Prompt - Single source of truth
# ═══════════════════════════════════════════════════════════════
OCR_INSTRUCTION = """Convert the following document to markdown.
Return only the markdown with no explanation text. Do not include delimiters like ```markdown or ```html.

RULES:
  - You must include all information on the page. Do not exclude headers, footers, or subtext.
  - Return tables in an HTML format.
  - Charts & infographics must be interpreted to a markdown format. Prefer table format when applicable.
  - Prefer using ☐ and ☑ for check boxes."""

# ═══════════════════════════════════════════════════════════════
# Image Formats
# ═══════════════════════════════════════════════════════════════
IMAGE_EXTENSIONS = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")
TIFF_EXTENSIONS = ("*.tif", "*.tiff")

# ═══════════════════════════════════════════════════════════════
# Gemini Models
# ═══════════════════════════════════════════════════════════════
GEMINI_MODELS = {
    "1": ("gemini-2.5-pro", "Gemini 2.5 Pro - Stable"),
    "2": ("gemini-3-pro-preview", "Gemini 3 Pro Preview - Best quality"),
    "3": ("gemini-3-flash-preview", "Gemini 3 Flash - Fast"),
}
DEFAULT_MODEL = "gemini-3-pro-preview"

# ═══════════════════════════════════════════════════════════════
# Default Paths
# ═══════════════════════════════════════════════════════════════
DEFAULT_INPUT_DIR = Path("./cropped_headers")
DEFAULT_OUTPUT_JSONL = Path("./ocr_dataset.jsonl")
DEFAULT_DOWNLOADS_DIR = Path("./elog_downloads")

# ═══════════════════════════════════════════════════════════════
# Dataset Registry - Open/Closed: add new datasets here
# ═══════════════════════════════════════════════════════════════
DATASETS = {
    # Legacy - uses old paths for backward compatibility
    "welllog": {
        "images_dir": Path("./cropped_headers"),
        "output": Path("./well_log_header.jsonl"),
    },
    # New structured datasets
    "welllog-train": {
        "images_dir": Path("./datasets/welllog/train"),
        "output": Path("./datasets/welllog/train.jsonl"),
    },
    "welllog-eval": {
        "images_dir": Path("./datasets/welllog/eval"),
        "output": Path("./datasets/welllog/eval.jsonl"),
    },
    "olmocr-train": {
        "images_dir": Path("./datasets/olmocr/train"),
        "output": Path("./datasets/olmocr/train.jsonl"),
    },
    "olmocr-eval": {
        "images_dir": Path("./datasets/olmocr/eval"),
        "output": Path("./datasets/olmocr/eval.jsonl"),
    },
}

def get_dataset(name: str) -> dict:
    """Get dataset config by name."""
    if name not in DATASETS:
        available = ", ".join(DATASETS.keys())
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")
    return DATASETS[name]

