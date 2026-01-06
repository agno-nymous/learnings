"""Real-time Gemini OCR annotator."""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from core.config import OCR_INSTRUCTION
from core.image_utils import load_as_png_base64

from .base import Annotator

load_dotenv()


class GeminiAnnotator(Annotator):
    """Real-time OCR using Gemini API.

    Processes images one at a time (with parallel workers) using
    the standard generate_content API.
    """

    def __init__(self, model: str, workers: int = 4):
        """Initialize the Gemini real-time annotator.

        Args:
            model: Gemini model identifier.
            workers: Number of parallel workers (not used in real-time mode).
        """
        self.model = model
        self.workers = workers
        self._client = None

    @property
    def name(self) -> str:
        """Return the annotator name."""
        return f"Gemini Real-time ({self.model})"

    @property
    def mode(self) -> str:
        """Return the annotation mode."""
        return "realtime"

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not set")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def _process_single(self, image_path: Path) -> dict:
        """Process a single image."""
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

    def annotate(self, image_paths: list[Path], output_path: Path) -> dict[str, int]:
        """Process images using parallel workers."""
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
                        print(f"[{i}/{total}] ✓ {img_path.name}")
                    else:
                        error_count += 1
                        print(f"[{i}/{total}] ✗ {img_path.name}: {result['status']}")
                except Exception as e:
                    error_count += 1
                    print(f"[{i}/{total}] ✗ {img_path.name}: {e}")

                time.sleep(0.1)  # Rate limiting

        return {"success": success_count, "errors": error_count}
