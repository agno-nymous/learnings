"""Abstract base class for preprocessing operations.

Provides a common interface for all preprocessing steps in the pipeline,
enabling consistent execution and result reporting.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from core.types import ProcessingResult


class Preprocessor(ABC):
    """Abstract base class for preprocessing operations.

    All preprocessing steps (download, rotate, crop, split) should inherit
    from this class to ensure a consistent interface.

    Example:
        class MyPreprocessor(Preprocessor):
            @property
            def name(self) -> str:
                return "My Preprocessor"

            def process(self, input_dir, output_dir, workers) -> ProcessingResult:
                # Implementation
                return ProcessingResult(processed=10)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this preprocessor.

        Returns:
            Name string used for logging and UI display.
        """
        pass

    @abstractmethod
    def process(
        self,
        input_dir: Path,
        output_dir: Path | None = None,
        workers: int = 4,
    ) -> ProcessingResult:
        """Execute the preprocessing operation.

        Args:
            input_dir: Directory containing input files.
            output_dir: Directory for output files (optional, some operations
                       modify in-place).
            workers: Number of parallel workers for processing.

        Returns:
            ProcessingResult with counts of processed, failed, and skipped items.
        """
        pass

    def __repr__(self) -> str:
        """String representation of the preprocessor."""
        return f"{self.__class__.__name__}(name={self.name!r})"


class PreprocessorRegistry:
    """Registry for preprocessor classes.

    Allows registration and lookup of preprocessors by name,
    following the Open/Closed Principle.

    Example:
        PreprocessorRegistry.register("rotate", Rotator)
        rotator = PreprocessorRegistry.create("rotate")
    """

    _registry: dict[str, type[Preprocessor]] = {}

    @classmethod
    def register(cls, name: str, preprocessor_class: type[Preprocessor]) -> None:
        """Register a preprocessor class.

        Args:
            name: Unique name for this preprocessor.
            preprocessor_class: The preprocessor class to register.
        """
        cls._registry[name] = preprocessor_class

    @classmethod
    def create(cls, name: str, **kwargs) -> Preprocessor:
        """Create a preprocessor instance by name.

        Args:
            name: Registered name of the preprocessor.
            **kwargs: Arguments passed to the preprocessor constructor.

        Returns:
            Configured preprocessor instance.

        Raises:
            ValueError: If name is not registered.
        """
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown preprocessor: {name}. Available: {available}")
        return cls._registry[name](**kwargs)

    @classmethod
    def list_all(cls) -> list[str]:
        """List all registered preprocessor names.

        Returns:
            List of registered names.
        """
        return list(cls._registry.keys())
