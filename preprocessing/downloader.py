"""Download TIFF/PDF files from elog scans.

Provides a Preprocessor implementation for downloading well log scans
from a CSV file containing URLs.
"""

import csv
import socket
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypedDict

from core.config import DEFAULT_DOWNLOADS_DIR
from core.types import ProcessingResult

from .base import Preprocessor, PreprocessorRegistry

# Timeout after 15 seconds if a download is stuck
socket.setdefaulttimeout(15)


class DownloadResult(TypedDict):
    """Result from downloading a single file."""

    ok: bool
    file: str
    status: str


class Downloader(Preprocessor):
    """Download elog scan files from URLs in a CSV file.

    Reads URLs from a CSV file and downloads files to the output directory.
    Supports parallel downloads and skips existing files.

    Attributes:
        input_file: Path to CSV file containing URLs.
        limit: Maximum number of files to download.
    """

    def __init__(self, input_file: Path, limit: int = 500) -> None:
        """Initialize the downloader.

        Args:
            input_file: Path to CSV file with URLs (SCAN_URL column).
            limit: Maximum files to download.
        """
        self.input_file = Path(input_file)
        self.limit = limit

    @property
    def name(self) -> str:
        """Human-readable name."""
        return "Elog Scan Downloader"

    def _download_one_file(self, url: str, output_dir: Path) -> DownloadResult:
        """Download a single file and return the result.

        Args:
            url: URL to download from.
            output_dir: Directory to save the file.

        Returns:
            DownloadResult with status information.
        """
        filename = url.split("/")[-1]
        filepath = output_dir / filename

        if filepath.exists():
            return {"ok": True, "file": filename, "status": "exists"}

        try:
            urllib.request.urlretrieve(url, filepath)  # nosec: B310 - URL from trusted source
            return {"ok": True, "file": filename, "status": "downloaded"}
        except Exception as e:
            return {"ok": False, "file": filename, "status": str(e)}

    def _read_urls(self) -> list[str]:
        """Read URLs from the CSV file.

        Returns:
            List of unique URLs from the CSV.
        """
        urls = []
        with open(self.input_file) as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                url = row[-2].strip()  # SCAN_URL column
                if url.startswith("http") and url not in urls:
                    urls.append(url)
                if len(urls) >= self.limit * 3:  # Read 3x in case of failures
                    break
        return urls

    def process(
        self,
        input_dir: Path,  # noqa: ARG002
        output_dir: Path | None = None,
        workers: int = 4,
    ) -> ProcessingResult:
        """Download files from the CSV.

        Note: input_dir is not used for this preprocessor - uses self.input_file instead.

        Args:
            input_dir: Not used (kept for interface compatibility).
            output_dir: Directory to save downloaded files.
            workers: Number of parallel download workers.

        Returns:
            ProcessingResult with download statistics.
        """
        if output_dir is None:
            output_dir = DEFAULT_DOWNLOADS_DIR

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        urls = self._read_urls()
        print(f"Downloading {self.limit} files using {workers} workers...")

        processed = 0
        failed = 0
        skipped = 0

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self._download_one_file, url, output_dir) for url in urls]

            for future in futures:
                if processed + skipped >= self.limit:
                    break

                result = future.result()
                if result["ok"]:
                    if result["status"] == "exists":
                        skipped += 1
                    else:
                        processed += 1
                    print(
                        f"[{processed + skipped}/{self.limit}] {result['status']}: {result['file']}"
                    )
                else:
                    failed += 1
                    print(f"[FAIL] {result['file']}: {result['status']}")

        print(f"\nDone! Downloaded: {processed}, Skipped: {skipped}, Failed: {failed}")
        return ProcessingResult(processed=processed, failed=failed, skipped=skipped)


# Register with the registry
PreprocessorRegistry.register("download", Downloader)


# Backward-compatible function
def download_scans(
    input_file: Path,
    output_dir: Path = DEFAULT_DOWNLOADS_DIR,
    limit: int = 500,
    workers: int = 4,
) -> dict:
    """Download elog scan files.

    This is a backward-compatible wrapper around the Downloader class.

    Args:
        input_file: Path to CSV file with URLs
        output_dir: Output directory
        limit: Maximum files to download
        workers: Number of parallel workers

    Returns:
        Dict with 'success' and 'failed' counts
    """
    downloader = Downloader(input_file=input_file, limit=limit)
    result = downloader.process(input_dir=Path("."), output_dir=output_dir, workers=workers)
    return {"success": result.processed + result.skipped, "failed": result.failed}
