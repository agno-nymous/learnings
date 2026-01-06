"""Rotate TIFF files 90 degrees anticlockwise."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from core.config import DEFAULT_DOWNLOADS_DIR
from core.file_discovery import find_tiff_files
from core.image_utils import Image


def _rotate_file(filepath: Path) -> str:
    """Rotate a single TIFF file 90 degrees anticlockwise and overwrite it."""
    try:
        with Image.open(filepath) as img:
            rotated = img.rotate(90, expand=True)
            rotated.save(filepath)
        return f"✓ {filepath.name}"
    except Exception as e:
        return f"✗ {filepath.name}: {e}"


def rotate_tiffs(
    input_dir: Path = DEFAULT_DOWNLOADS_DIR,
    workers: int = 4,
) -> dict:
    """
    Rotate all TIFF files in directory 90 degrees anticlockwise.

    Args:
        input_dir: Directory containing TIFF files
        workers: Number of parallel workers

    Returns:
        Dict with 'processed' count
    """
    tiff_files = find_tiff_files(input_dir)

    if not tiff_files:
        print(f"No TIFF files found in {input_dir}")
        return {"processed": 0}

    print(f"Found {len(tiff_files)} TIFF files to rotate")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = executor.map(_rotate_file, tiff_files)
        for result in results:
            print(result)

    print("Done!")
    return {"processed": len(tiff_files)}
