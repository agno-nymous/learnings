"""Batch API Gemini OCR annotator (50% cheaper)."""

import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from core.config import OCR_INSTRUCTION
from core.image_utils import load_as_png_bytes

from .base import Annotator

load_dotenv()


class GeminiBatchAnnotator(Annotator):
    """Batch OCR using Gemini Batch API.

    50% cheaper than real-time API. Workflow:
    1. Upload images to Files API
    2. Create JSONL batch input
    3. Submit batch job
    4. Wait for completion
    5. Download and parse results
    """

    def __init__(self, model: str, workers: int = 4, poll_interval: int = 60):
        self.model = model
        self.workers = workers
        self.poll_interval = poll_interval
        self._client = None

    @property
    def name(self) -> str:
        return f"Gemini Batch API ({self.model})"

    @property
    def mode(self) -> str:
        return "batch"

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not set")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def _upload_image(self, image_path: Path) -> dict:
        """Upload a single image to Files API."""
        try:
            png_bytes = load_as_png_bytes(image_path)
            buffer = io.BytesIO(png_bytes)

            uploaded = self.client.files.upload(
                file=buffer, config={"display_name": image_path.name, "mime_type": "image/png"}
            )
            return {"filename": image_path.name, "file_uri": uploaded.name, "status": "success"}
        except Exception as e:
            return {"filename": image_path.name, "file_uri": None, "status": f"error: {e}"}

    def annotate(self, image_paths: list[Path], output_path: Path) -> dict[str, int]:
        """Process images via batch API."""
        total = len(image_paths)

        # ═══════════════════════════════════════════════════════════════
        # STEP 1: Upload images
        # ═══════════════════════════════════════════════════════════════
        print("\n" + "=" * 60)
        print("STEP 1: Uploading images to Gemini Files API...")
        print("=" * 60)

        uploads = {}
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(self._upload_image, img): img for img in image_paths}
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result["status"] == "success":
                    uploads[result["filename"]] = result["file_uri"]
                    print(f"[{i}/{total}] ✓ {result['filename']}")
                else:
                    print(f"[{i}/{total}] ✗ {result['filename']}: {result['status']}")

        if not uploads:
            print("No files uploaded successfully!")
            return {"success": 0, "errors": total}

        print(f"\nUploaded: {len(uploads)} files")

        # ═══════════════════════════════════════════════════════════════
        # STEP 2: Create JSONL batch input
        # ═══════════════════════════════════════════════════════════════
        print("\n" + "=" * 60)
        print("STEP 2: Creating batch input JSONL...")
        print("=" * 60)

        batch_input_path = Path("./batch_input_temp.jsonl")
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

        # ═══════════════════════════════════════════════════════════════
        # STEP 3: Submit batch job
        # ═══════════════════════════════════════════════════════════════
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
            model=self.model, src=input_file.name, config={"display_name": "well_log_ocr"}
        )
        print(f"Batch job created: {batch_job.name}")

        # ═══════════════════════════════════════════════════════════════
        # STEP 4: Wait for completion
        # ═══════════════════════════════════════════════════════════════
        print("\n" + "=" * 60)
        print("STEP 4: Waiting for batch job to complete...")
        print("=" * 60)

        while True:
            batch_job = self.client.batches.get(name=batch_job.name)
            state = str(batch_job.state)
            print(f"State: {state}")

            if "SUCCEEDED" in state:
                print("✓ Batch job completed!")
                break
            elif "FAILED" in state or "CANCELLED" in state:
                print(f"✗ Batch job failed: {state}")
                batch_input_path.unlink(missing_ok=True)
                return {"success": 0, "errors": total}

            print(f"Waiting {self.poll_interval}s...")
            time.sleep(self.poll_interval)

        # ═══════════════════════════════════════════════════════════════
        # STEP 5: Download and parse results
        # ═══════════════════════════════════════════════════════════════
        print("\n" + "=" * 60)
        print("STEP 5: Downloading results...")
        print("=" * 60)

        content = self.client.files.download(file=batch_job.dest.file_name)

        success = 0
        errors = 0

        with open(output_path, "a") as out_f:
            for line in content.decode("utf-8").strip().split("\n"):
                try:
                    result = json.loads(line)
                    key = result.get("key", "unknown")

                    if "error" in result:
                        errors += 1
                        continue

                    response = result.get("response", {})
                    candidates = response.get("candidates", [])
                    text = (
                        candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                        if candidates
                        else ""
                    )

                    if text:
                        training_row = {
                            "filename": key,
                            "instruction": OCR_INSTRUCTION,
                            "answer": text,
                            "status": "success",
                            "model": self.model,
                        }
                        out_f.write(json.dumps(training_row, ensure_ascii=False) + "\n")
                        success += 1
                    else:
                        error_row = {
                            "filename": key,
                            "instruction": OCR_INSTRUCTION,
                            "answer": "",
                            "status": "error: empty response",
                            "model": self.model,
                        }
                        out_f.write(json.dumps(error_row, ensure_ascii=False) + "\n")
                        errors += 1
                except json.JSONDecodeError:
                    errors += 1

        batch_input_path.unlink(missing_ok=True)

        print(f"\n{'=' * 60}")
        print(f"DONE! Success: {success}, Errors: {errors}")
        print(f"{'=' * 60}")

        return {"success": success, "errors": errors}
