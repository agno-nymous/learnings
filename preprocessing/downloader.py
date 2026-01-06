"""Download TIFF/PDF files from elog scans."""

import csv
import socket
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from core.config import DEFAULT_DOWNLOADS_DIR

# Timeout after 15 seconds if a download is stuck
socket.setdefaulttimeout(15)


def _download_one_file(url: str, output_dir: Path) -> dict:
    """Download a single file and return the result."""
    filename = url.split("/")[-1]
    filepath = output_dir / filename

    if filepath.exists():
        return {"ok": True, "file": filename, "status": "exists"}

    try:
        urllib.request.urlretrieve(url, filepath)  # nosec: B310 - URL from trusted source
        return {"ok": True, "file": filename, "status": "downloaded"}
    except Exception as e:
        return {"ok": False, "file": filename, "status": str(e)}


def download_scans(
    input_file: Path,
    output_dir: Path = DEFAULT_DOWNLOADS_DIR,
    limit: int = 500,
    workers: int = 4,
) -> dict:
    """
    Download elog scan files.

    Args:
        input_file: Path to CSV file with URLs
        output_dir: Output directory
        limit: Maximum files to download
        workers: Number of parallel workers

    Returns:
        Dict with 'success' and 'failed' counts
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read URLs from CSV
    urls = []
    with open(input_file) as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            url = row[-2].strip()  # SCAN_URL column
            if url.startswith("http") and url not in urls:
                urls.append(url)
            if len(urls) >= limit * 3:  # Read 3x in case of failures
                break

    print(f"Downloading {limit} files using {workers} workers...")

    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_download_one_file, url, output_dir) for url in urls]

        for future in futures:
            if success_count >= limit:
                break

            result = future.result()
            if result["ok"]:
                success_count += 1
                print(f"[{success_count}/{limit}] {result['status']}: {result['file']}")
            else:
                fail_count += 1
                print(f"[FAIL] {result['file']}: {result['status']}")

    print(f"\nDone! Success: {success_count}, Failed: {fail_count}")
    return {"success": success_count, "failed": fail_count}
