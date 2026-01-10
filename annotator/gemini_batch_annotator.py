"""Batch API Gemini OCR annotator (50% cheaper).

Processes images using the Gemini Batch API for cost-efficient
large-scale OCR operations.
"""

import io
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TypedDict

from core.config import OCR_INSTRUCTION
from core.image_utils import load_as_png_bytes
from core.types import AnnotationEntry, AnnotationResult

from .base import Annotator
from .client import GeminiClientMixin


class UploadResult(TypedDict):
    """Result from uploading a single image."""

    filename: str
    file_uri: str | None
    status: str


class GeminiBatchAnnotator(GeminiClientMixin, Annotator):
    """Batch OCR using Gemini Batch API.

    50% cheaper than real-time API. Workflow:
    1. Upload images to Files API
    2. Create JSONL batch input
    3. Submit batch job
    4. Wait for completion
    5. Download and parse results

    Attributes:
        model: Gemini model identifier.
        workers: Number of parallel workers for uploads.
        poll_interval: Seconds between batch job status checks.
    """

    def __init__(self, model: str, workers: int = 4, poll_interval: int = 60) -> None:
        """Initialize the Gemini batch annotator.

        Args:
            model: Gemini model identifier (e.g., 'gemini-3-flash-preview').
            workers: Number of parallel workers for image uploads.
            poll_interval: Seconds between batch job status checks.
        """
        self.model = model
        self.workers = workers
        self.poll_interval = poll_interval
        self._client = None  # Initialize for mixin

    @property
    def name(self) -> str:
        """Human-readable annotator name."""
        return f"Gemini Batch API ({self.model})"

    @property
    def mode(self) -> str:
        """Annotation mode identifier."""
        return "batch"

    def _upload_image(self, image_path: Path) -> UploadResult:
        """Upload a single image to Files API.

        Args:
            image_path: Path to image file.

        Returns:
            UploadResult with filename, file_uri, and status.
        """
        try:
            png_bytes = load_as_png_bytes(image_path)
            buffer = io.BytesIO(png_bytes)

            uploaded = self.client.files.upload(
                file=buffer,
                config={"display_name": image_path.name, "mime_type": "image/png"},
            )
            return {
                "filename": image_path.name,
                "file_uri": uploaded.name,
                "status": "success",
            }
        except Exception as e:
            return {
                "filename": image_path.name,
                "file_uri": None,
                "status": f"error: {e}",
            }

    def _upload_images(self, image_paths: list[Path]) -> dict[str, str]:
        """Upload images to Files API in parallel.

        Args:
            image_paths: List of image file paths.

        Returns:
            Dict mapping filename to file_uri for successful uploads.
        """
        total = len(image_paths)
        uploads: dict[str, str] = {}

        print("\n" + "=" * 60)
        print("STEP 1: Uploading images to Gemini Files API...")
        print("=" * 60)

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(self._upload_image, img): img for img in image_paths}
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result["status"] == "success" and result["file_uri"]:
                    uploads[result["filename"]] = result["file_uri"]
                    print(f"[{i}/{total}] OK {result['filename']}")
                else:
                    print(f"[{i}/{total}] FAIL {result['filename']}: {result['status']}")

        print(f"\nUploaded: {len(uploads)} files")
        return uploads

    def _create_batch_input(self, uploads: dict[str, str], batch_input_path: Path) -> None:
        """Create JSONL batch input file.

        Args:
            uploads: Dict mapping filename to file_uri.
            batch_input_path: Path to write the batch input file.
        """
        print("\n" + "=" * 60)
        print("STEP 2: Creating batch input JSONL...")
        print("=" * 60)

        with open(batch_input_path, "w") as f:
            for filename, file_uri in uploads.items():
                if file_uri.startswith("files/"):
                    full_uri = f"https://generativelanguage.googleapis.com/v1beta/{file_uri}"
                else:
                    full_uri = file_uri

                request = {
                    "key": filename,
                    "request": {
                        "contents": [
                            {
                                "parts": [
                                    {"file_data": {"file_uri": full_uri, "mime_type": "image/png"}},
                                    {"text": OCR_INSTRUCTION},
                                ]
                            }
                        ]
                    },
                }
                f.write(json.dumps(request) + "\n")

        print(f"Created: {batch_input_path} ({len(uploads)} requests)")

    def _submit_batch_job(self, batch_input_path: Path) -> str:
        """Submit batch job and return job name.

        Args:
            batch_input_path: Path to the batch input JSONL file.

        Returns:
            Batch job name for tracking.
        """
        print("\n" + "=" * 60)
        print("STEP 3: Submitting batch job...")
        print("=" * 60)

        with open(batch_input_path, "rb") as f:
            input_file = self.client.files.upload(
                file=f,
                config={"display_name": "batch_ocr_input.jsonl", "mime_type": "application/jsonl"},
            )
        print(f"Uploaded input file: {input_file.name}")

        batch_job = self.client.batches.create(
            model=self.model,
            src=input_file.name,
            config={"display_name": "well_log_ocr"},
        )
        print(f"Batch job created: {batch_job.name}")
        return batch_job.name

    def _wait_for_completion(self, job_name: str) -> str | None:
        """Wait for batch job completion.

        Args:
            job_name: Batch job name to monitor.

        Returns:
            Destination file name on success, None on failure.
        """
        print("\n" + "=" * 60)
        print("STEP 4: Waiting for batch job to complete...")
        print("=" * 60)

        while True:
            batch_job = self.client.batches.get(name=job_name)
            state = str(batch_job.state)
            print(f"State: {state}")

            if "SUCCEEDED" in state:
                print("Batch job completed!")
                return batch_job.dest.file_name
            elif "FAILED" in state or "CANCELLED" in state:
                print(f"Batch job failed: {state}")
                return None

            print(f"Waiting {self.poll_interval}s...")
            time.sleep(self.poll_interval)

    def _parse_results(self, dest_file: str, output_path: Path) -> AnnotationResult:
        """Download and parse batch results.

        Args:
            dest_file: Destination file name from batch job.
            output_path: Path to output JSONL file.

        Returns:
            AnnotationResult with success and error counts.
        """
        print("\n" + "=" * 60)
        print("STEP 5: Downloading results...")
        print("=" * 60)

        content = self.client.files.download(file=dest_file)

        success = 0
        errors = 0

        with open(output_path, "a") as out_f:
            for line in content.decode("utf-8").strip().split("\n"):
                entry = self._parse_result_line(line)
                if entry:
                    out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    if entry["status"] == "success":
                        success += 1
                    else:
                        errors += 1
                else:
                    errors += 1

        return {"success": success, "errors": errors}

    def _parse_result_line(self, line: str) -> AnnotationEntry | None:
        """Parse a single result line from batch output.

        Args:
            line: JSON line from batch results.

        Returns:
            AnnotationEntry or None if parsing fails.
        """
        try:
            result = json.loads(line)
            key = result.get("key", "unknown")

            if "error" in result:
                return {
                    "filename": key,
                    "instruction": OCR_INSTRUCTION,
                    "answer": "",
                    "status": f"error: {result['error']}",
                    "model": self.model,
                }

            response = result.get("response", {})
            candidates = response.get("candidates", [])
            text = (
                candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if candidates
                else ""
            )

            if text:
                return {
                    "filename": key,
                    "instruction": OCR_INSTRUCTION,
                    "answer": text,
                    "status": "success",
                    "model": self.model,
                }
            else:
                return {
                    "filename": key,
                    "instruction": OCR_INSTRUCTION,
                    "answer": "",
                    "status": "error: empty response",
                    "model": self.model,
                }
        except json.JSONDecodeError:
            return None

    def annotate(self, image_paths: list[Path], output_path: Path) -> AnnotationResult:
        """Process images via batch API.

        Args:
            image_paths: List of image file paths to process.
            output_path: Path to output JSONL file (append mode).

        Returns:
            AnnotationResult with success and error counts.
        """
        total = len(image_paths)
        batch_input_path = Path("./batch_input_temp.jsonl")

        try:
            # Step 1: Upload images
            uploads = self._upload_images(image_paths)
            if not uploads:
                print("No files uploaded successfully!")
                return {"success": 0, "errors": total}

            # Step 2: Create batch input
            self._create_batch_input(uploads, batch_input_path)

            # Step 3: Submit batch job
            job_name = self._submit_batch_job(batch_input_path)

            # Step 4: Wait for completion
            dest_file = self._wait_for_completion(job_name)
            if not dest_file:
                return {"success": 0, "errors": total}

            # Step 5: Parse results
            result = self._parse_results(dest_file, output_path)

            print(f"\n{'=' * 60}")
            print(f"DONE! Success: {result['success']}, Errors: {result['errors']}")
            print(f"{'=' * 60}")

            return result

        finally:
            # Cleanup temp file
            batch_input_path.unlink(missing_ok=True)
