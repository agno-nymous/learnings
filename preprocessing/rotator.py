"""Rotate TIFF files 90 degrees anticlockwise.

Provides a Preprocessor implementation for rotating TIFF images
in-place to correct orientation.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from core.config import DEFAULT_DOWNLOADS_DIR
from core.file_discovery import find_tiff_files
from core.image_utils import Image
from core.types import ProcessingResult

from .base import Preprocessor, PreprocessorRegistry


def _rotate_file(filepath: Path) -> tuple[bool, str]:
    """Rotate a single TIFF file 90 degrees anticlockwise and overwrite it.

    Args:
        filepath: Path to the TIFF file.

    Returns:
        Tuple of (success, message).
    """
    try:
        with Image.open(filepath) as img:
            rotated = img.rotate(90, expand=True)
            rotated.save(filepath)
        return True, f"OK {filepath.name}"
    except Exception as e:
        return False, f"FAIL {filepath.name}: {e}"


class Rotator(Preprocessor):
    """Rotate TIFF files 90 degrees anticlockwise.

    Rotates images in-place to correct orientation issues from scanning.
    Uses multiprocessing for parallel processing.
    """

    @property
    def name(self) -> str:
        """Human-readable name."""
        return "TIFF Rotator"

    def process(
        self,
        input_dir: Path,
        output_dir: Path | None = None,  # noqa: ARG002
        workers: int = 4,
    ) -> ProcessingResult:
        """Rotate all TIFF files in directory 90 degrees anticlockwise.

        Note: output_dir is not used - rotation is done in-place.

        Args:
            input_dir: Directory containing TIFF files.
            output_dir: Not used (rotation is in-place).
            workers: Number of parallel workers.

        Returns:
            ProcessingResult with rotation statistics.
        """
        input_dir = Path(input_dir)
        tiff_files = find_tiff_files(input_dir)

        if not tiff_files:
            print(f"No TIFF files found in {input_dir}")
            return ProcessingResult(processed=0)

        print(f"Found {len(tiff_files)} TIFF files to rotate")

        processed = 0
        failed = 0

        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = executor.map(_rotate_file, tiff_files)
            for success, message in results:
                print(message)
                if success:
                    processed += 1
                else:
                    failed += 1

        print(f"Done! Rotated: {processed}, Failed: {failed}")
        return ProcessingResult(processed=processed, failed=failed)


# Register with the registry
PreprocessorRegistry.register("rotate", Rotator)


# Backward-compatible function
def rotate_tiffs(
    input_dir: Path = DEFAULT_DOWNLOADS_DIR,
    workers: int = 4,
) -> dict:
    """Rotate all TIFF files in directory 90 degrees anticlockwise.

    This is a backward-compatible wrapper around the Rotator class.

    Args:
        input_dir: Directory containing TIFF files
        workers: Number of parallel workers

    Returns:
        Dict with 'processed' count
    """
    rotator = Rotator()
    result = rotator.process(input_dir=input_dir, workers=workers)
    return {"processed": result.processed}
