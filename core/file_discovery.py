"""File discovery utilities."""

from pathlib import Path

from .config import IMAGE_EXTENSIONS, TIFF_EXTENSIONS


def find_images(directory: Path, extensions: tuple = IMAGE_EXTENSIONS) -> list[Path]:
    """Find all image files in a directory.

    Args:
        directory: Path to search
        extensions: Glob patterns to match (default: all image types)

    Returns:
        Sorted list of matching file paths
    """
    files = []
    for pattern in extensions:
        files.extend(directory.glob(pattern))
    return sorted(files)


def find_tiff_files(directory: Path) -> list[Path]:
    """Find TIFF files only."""
    return find_images(directory, TIFF_EXTENSIONS)
