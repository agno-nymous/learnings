"""Preprocessing module for image preparation pipeline.

Provides:
    - Preprocessor: Abstract base class for preprocessors
    - PreprocessorRegistry: Registration-based factory (Open/Closed Principle)
    - Downloader: Download elog scans from URLs
    - Rotator: Rotate TIFF images 90 degrees
    - Cropper: Crop headers from well log scans
    - Splitter: Split datasets into train/eval sets

Backward-compatible functions:
    - download_scans, rotate_tiffs, crop_headers, split_dataset, split_welllog
"""

from .base import Preprocessor, PreprocessorRegistry
from .cropper import Cropper, crop_headers
from .downloader import Downloader, download_scans
from .rotator import Rotator, rotate_tiffs
from .splitter import Splitter, split_dataset, split_welllog

__all__ = [
    # Base classes
    "Preprocessor",
    "PreprocessorRegistry",
    # Preprocessor classes
    "Downloader",
    "Rotator",
    "Cropper",
    "Splitter",
    # Backward-compatible functions
    "download_scans",
    "rotate_tiffs",
    "crop_headers",
    "split_dataset",
    "split_welllog",
]
