"""Real-time Gemini OCR annotator.

Processes images using the standard Gemini generate_content API
with parallel workers for throughput.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google.genai import types

from core.config import OCR_INSTRUCTION
from core.image_utils import load_as_png_base64
from core.types import AnnotationEntry, AnnotationResult

from .base import Annotator
from .client import GeminiClientMixin


class GeminiAnnotator(GeminiClientMixin, Annotator):
    """Real-time OCR using Gemini API.

    Processes images with parallel workers using the standard
    generate_content API. Good for small to medium batches.

    Attributes:
        model: Gemini model identifier.
        workers: Number of parallel workers.
    """

    def __init__(self, model: str, workers: int = 4) -> None:
        """Initialize the Gemini real-time annotator.

        Args:
            model: Gemini model identifier (e.g., 'gemini-3-flash-preview').
            workers: Number of parallel workers for processing.
        """
        self.model = model
        self.workers = workers
        self._client = None  # Initialize for mixin

    @property
    def name(self) -> str:
        """Human-readable annotator name."""
        return f"Gemini Real-time ({self.model})"

    @property
    def mode(self) -> str:
        """Annotation mode identifier."""
        return "realtime"

    def _process_single(self, image_path: Path) -> AnnotationEntry:
        """Process a single image through OCR.

        Args:
            image_path: Path to image file.

        Returns:
            AnnotationEntry with OCR results or error status.
        """
        try:
            image_data = load_as_png_base64(image_path)

            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    types.Content(
                        parts=[
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type="image/png",
                                    data=image_data,
                                )
                            ),
                            types.Part(text=OCR_INSTRUCTION),
                        ]
                    )
                ],
            )

            answer = response.text if response.text else ""

            return {
                "filename": image_path.name,
                "instruction": OCR_INSTRUCTION,
                "answer": answer,
                "status": "success",
                "model": self.model,
            }
        except Exception as e:
            return {
                "filename": image_path.name,
                "instruction": OCR_INSTRUCTION,
                "answer": "",
                "status": f"error: {str(e)}",
                "model": self.model,
            }

    def annotate(self, image_paths: list[Path], output_path: Path) -> AnnotationResult:
        """Process images using parallel workers.

        Args:
            image_paths: List of image file paths to process.
            output_path: Path to output JSONL file (append mode).

        Returns:
            AnnotationResult with success and error counts.
        """
        success_count = 0
        error_count = 0
        total = len(image_paths)

        with (
            open(output_path, "a") as out_file,
            ThreadPoolExecutor(max_workers=self.workers) as executor,
        ):
            future_to_path = {
                executor.submit(self._process_single, img): img for img in image_paths
            }

            for i, future in enumerate(as_completed(future_to_path), 1):
                img_path = future_to_path[future]
                try:
                    result = future.result()
                    out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_file.flush()

                    if result["status"] == "success":
                        success_count += 1
                        print(f"[{i}/{total}] OK {img_path.name}")
                    else:
                        error_count += 1
                        print(f"[{i}/{total}] FAIL {img_path.name}: {result['status']}")
                except Exception as e:
                    error_count += 1
                    print(f"[{i}/{total}] FAIL {img_path.name}: {e}")

                time.sleep(0.1)  # Rate limiting

        return {"success": success_count, "errors": error_count}
