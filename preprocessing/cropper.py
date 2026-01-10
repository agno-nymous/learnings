"""Crop well log headers from rotated TIFF scans.

Provides a Preprocessor implementation for extracting the header
section (first 4:3 page) from wide well log scans.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from core.config import DEFAULT_DOWNLOADS_DIR, DEFAULT_INPUT_DIR
from core.file_discovery import find_tiff_files
from core.image_utils import Image
from core.types import ProcessingResult

from .base import Preprocessor, PreprocessorRegistry


def _crop_first_page(args: tuple[Path, Path]) -> tuple[bool, str]:
    """Crop the first 4:3 page from a wide image.

    Args:
        args: Tuple of (input_path, output_path).

    Returns:
        Tuple of (success, message).
    """
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

        return True, f"OK {input_path.name} -> {output_path.name} ({crop_width}x{height})"
    except Exception as e:
        return False, f"FAIL {input_path.name}: {e}"


class Cropper(Preprocessor):
    """Crop well log headers from rotated TIFF scans.

    Extracts the first page (header section) from wide well log scans
    using 4:3 aspect ratio calculation.
    """

    @property
    def name(self) -> str:
        """Human-readable name."""
        return "Header Cropper"

    def process(
        self,
        input_dir: Path,
        output_dir: Path | None = None,
        workers: int = 4,
    ) -> ProcessingResult:
        """Crop headers from all TIFF files.

        Args:
            input_dir: Directory containing source TIFF files.
            output_dir: Directory for cropped outputs.
            workers: Number of parallel workers.

        Returns:
            ProcessingResult with cropping statistics.
        """
        input_dir = Path(input_dir)

        if output_dir is None:
            output_dir = DEFAULT_INPUT_DIR
        output_dir = Path(output_dir)

        tiff_files = find_tiff_files(input_dir)

        if not tiff_files:
            print(f"No TIFF files found in {input_dir}")
            return ProcessingResult(processed=0)

        print(f"Found {len(tiff_files)} TIFF files to process")
        print(f"Output directory: {output_dir}")

        tasks = [(f, output_dir / f.name) for f in tiff_files]

        processed = 0
        failed = 0

        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = executor.map(_crop_first_page, tasks)
            for success, message in results:
                print(message)
                if success:
                    processed += 1
                else:
                    failed += 1

        print(f"Done! Cropped: {processed}, Failed: {failed}")
        return ProcessingResult(processed=processed, failed=failed)


# Register with the registry
PreprocessorRegistry.register("crop", Cropper)


# Backward-compatible function
def crop_headers(
    input_dir: Path = DEFAULT_DOWNLOADS_DIR,
    output_dir: Path = DEFAULT_INPUT_DIR,
    workers: int = 4,
) -> dict:
    """Crop well log headers from rotated TIFF scans.

    This is a backward-compatible wrapper around the Cropper class.

    Args:
        input_dir: Directory containing source TIFF files
        output_dir: Directory for cropped outputs
        workers: Number of parallel workers

    Returns:
        Dict with 'processed' count
    """
    cropper = Cropper()
    result = cropper.process(input_dir=input_dir, output_dir=output_dir, workers=workers)
    return {"processed": result.processed}
