"""Abstract base class for OCR annotators."""

from abc import ABC, abstractmethod
from pathlib import Path


class Annotator(ABC):
    """Abstract base class for OCR annotators.

    Implements the Template Method pattern - subclasses implement
    the actual annotation logic while the base class handles common
    workflow steps.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the annotator."""
        pass

    @property
    @abstractmethod
    def mode(self) -> str:
        """Mode identifier: 'realtime' or 'batch'."""
        pass

    @abstractmethod
    def annotate(self, image_paths: list[Path], output_path: Path) -> dict[str, int]:
        """
        Process images and write results to output JSONL.

        Args:
            image_paths: List of image file paths to process
            output_path: Path to output JSONL file (append mode)

        Returns:
            Dict with 'success' and 'errors' counts
        """
        pass
