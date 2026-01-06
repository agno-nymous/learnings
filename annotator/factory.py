"""Factory for creating annotators."""

from core.config import DEFAULT_MODEL

from .base import Annotator
from .gemini_annotator import GeminiAnnotator
from .gemini_batch_annotator import GeminiBatchAnnotator


class AnnotatorFactory:
    """Factory for creating OCR annotators.

    Supports:
    - "realtime": Standard Gemini API (faster, good for small batches)
    - "batch": Gemini Batch API (50% cheaper, good for large batches)
    """

    MODES = {
        "realtime": GeminiAnnotator,
        "batch": GeminiBatchAnnotator,
    }

    @classmethod
    def create(
        cls,
        mode: str = "realtime",
        model: str = DEFAULT_MODEL,
        workers: int = 4,
        poll_interval: int = 60,
    ) -> Annotator:
        """
        Create an annotator instance.

        Args:
            mode: "realtime" or "batch"
            model: Gemini model name
            workers: Number of parallel workers
            poll_interval: Seconds between batch job status checks (batch mode only)

        Returns:
            Configured Annotator instance

        Raises:
            ValueError: If mode is not recognized
        """
        if mode not in cls.MODES:
            valid = ", ".join(cls.MODES.keys())
            raise ValueError(f"Unknown mode: {mode}. Valid modes: {valid}")

        annotator_class = cls.MODES[mode]

        if mode == "batch":
            return annotator_class(model=model, workers=workers, poll_interval=poll_interval)
        else:
            return annotator_class(model=model, workers=workers)

    @classmethod
    def list_modes(cls) -> list:
        """Return list of available modes."""
        return list(cls.MODES.keys())
