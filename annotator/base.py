"""Abstract base class for OCR annotators."""

from abc import ABC, abstractmethod
from pathlib import Path

from core.types import AnnotationResult


class Annotator(ABC):
    """Abstract base class for OCR annotators.

    Implements the Template Method pattern - subclasses implement
    the actual annotation logic while the base class defines the
    interface contract.

    Subclasses must implement:
        - name: Human-readable name of the annotator
        - mode: Mode identifier ('realtime', 'batch', etc.)
        - annotate: Process images and write results

    Example:
        class MyAnnotator(Annotator):
            @property
            def name(self) -> str:
                return "My Custom Annotator"

            @property
            def mode(self) -> str:
                return "custom"

            def annotate(self, image_paths, output_path) -> AnnotationResult:
                # Process images...
                return {"success": 10, "errors": 0}
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the annotator."""
        ...

    @property
    @abstractmethod
    def mode(self) -> str:
        """Mode identifier: 'realtime', 'batch', etc."""
        ...

    @abstractmethod
    def annotate(self, image_paths: list[Path], output_path: Path) -> AnnotationResult:
        """Process images and write results to output JSONL.

        Args:
            image_paths: List of image file paths to process.
            output_path: Path to output JSONL file (append mode).

        Returns:
            AnnotationResult with 'success' and 'errors' counts.
        """
        ...
