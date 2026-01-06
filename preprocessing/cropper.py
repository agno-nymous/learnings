"""Crop well log headers from rotated TIFF scans."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from core.config import DEFAULT_DOWNLOADS_DIR, DEFAULT_INPUT_DIR
from core.file_discovery import find_tiff_files
from core.image_utils import Image


def _crop_first_page(args: tuple) -> str:
    """Crop the first 4:3 page from a wide image."""
    input_path, output_path = args
    try:
        with Image.open(input_path) as img:
            width, height = img.size

            # 4:3 aspect ratio: page_width = height * (4/3)
            page_width = int(height * (4 / 3))
            crop_width = min(page_width, width)
            cropped = img.crop((0, 0, crop_width, height))

            output_path.parent.mkdir(parents=True, exist_ok=True)
            cropped.save(output_path)

        return f"✓ {input_path.name} → {output_path.name} ({crop_width}x{height})"
    except Exception as e:
        return f"✗ {input_path.name}: {e}"


def crop_headers(
    input_dir: Path = DEFAULT_DOWNLOADS_DIR,
    output_dir: Path = DEFAULT_INPUT_DIR,
    workers: int = 4,
) -> dict:
    """
    Crop well log headers from rotated TIFF scans.

    Splits wide images into 4:3 aspect ratio pages and saves
    only the first page (header section).

    Args:
        input_dir: Directory containing source TIFF files
        output_dir: Directory for cropped outputs
        workers: Number of parallel workers

    Returns:
        Dict with 'processed' count
    """
    tiff_files = find_tiff_files(input_dir)

    if not tiff_files:
        print(f"No TIFF files found in {input_dir}")
        return {"processed": 0}

    print(f"Found {len(tiff_files)} TIFF files to process")
    print(f"Output directory: {output_dir}")

    tasks = [(f, output_dir / f.name) for f in tiff_files]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = executor.map(_crop_first_page, tasks)
        for result in results:
            print(result)

    print("Done!")
    return {"processed": len(tiff_files)}
