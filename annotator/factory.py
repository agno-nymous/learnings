"""Factory for creating annotators with registration-based pattern.

Supports the Open/Closed Principle - new annotators can be added
without modifying existing code by using the register decorator.
"""

from collections.abc import Callable
from typing import Any, TypeVar

from core.config import DEFAULT_MODEL

from .base import Annotator

# Type variable for annotator classes
T = TypeVar("T", bound=Annotator)


class AnnotatorRegistry:
    """Registry for annotator classes.

    Provides a central registration point for annotator implementations,
    enabling the Open/Closed Principle.

    Example:
        # Register a new annotator
        @AnnotatorRegistry.register("custom")
        class CustomAnnotator(Annotator):
            ...

        # Create an instance
        annotator = AnnotatorRegistry.create("custom", model="gemini-pro")
    """

    _annotators: dict[str, type[Annotator]] = {}
    _default_kwargs: dict[str, dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        mode: str,
        **default_kwargs: Any,
    ) -> Callable[[type[T]], type[T]]:
        """Register an annotator class for a mode.

        Args:
            mode: Mode identifier (e.g., 'realtime', 'batch').
            **default_kwargs: Default keyword arguments for this annotator.

        Returns:
            Decorator function that registers the class.

        Raises:
            ValueError: If mode is already registered.
        """

        def decorator(annotator_class: type[T]) -> type[T]:
            if mode in cls._annotators:
                raise ValueError(f"Mode '{mode}' is already registered")
            cls._annotators[mode] = annotator_class
            cls._default_kwargs[mode] = default_kwargs
            return annotator_class

        return decorator

    @classmethod
    def create(
        cls,
        mode: str = "realtime",
        model: str = DEFAULT_MODEL,
        **kwargs: Any,
    ) -> Annotator:
        """Create an annotator instance.

        Args:
            mode: Annotator mode (e.g., 'realtime', 'batch').
            model: Model identifier.
            **kwargs: Additional arguments passed to annotator constructor.

        Returns:
            Configured Annotator instance.

        Raises:
            ValueError: If mode is not registered.
        """
        if mode not in cls._annotators:
            valid = ", ".join(cls._annotators.keys())
            raise ValueError(f"Unknown mode: {mode}. Valid modes: {valid}")

        annotator_class = cls._annotators[mode]

        # Merge default kwargs with provided kwargs
        merged_kwargs = {**cls._default_kwargs.get(mode, {}), **kwargs}

        return annotator_class(model=model, **merged_kwargs)

    @classmethod
    def list_modes(cls) -> list[str]:
        """Return list of registered modes."""
        return list(cls._annotators.keys())

    @classmethod
    def get_annotator_class(cls, mode: str) -> type[Annotator]:
        """Get the annotator class for a mode.

        Args:
            mode: Mode identifier.

        Returns:
            Annotator class.

        Raises:
            ValueError: If mode is not registered.
        """
        if mode not in cls._annotators:
            valid = ", ".join(cls._annotators.keys())
            raise ValueError(f"Unknown mode: {mode}. Valid modes: {valid}")
        return cls._annotators[mode]

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (useful for testing)."""
        cls._annotators.clear()
        cls._default_kwargs.clear()


# Register built-in annotators
# Import here to avoid circular imports
from .gemini_annotator import GeminiAnnotator  # noqa: E402
from .gemini_batch_annotator import GeminiBatchAnnotator  # noqa: E402

AnnotatorRegistry._annotators["realtime"] = GeminiAnnotator
AnnotatorRegistry._annotators["batch"] = GeminiBatchAnnotator
AnnotatorRegistry._default_kwargs["realtime"] = {"workers": 4}
AnnotatorRegistry._default_kwargs["batch"] = {"workers": 4, "poll_interval": 60}


# Backward-compatible factory class
class AnnotatorFactory:
    """Factory for creating OCR annotators.

    This is a backward-compatible wrapper around AnnotatorRegistry.
    New code should use AnnotatorRegistry directly.

    Supports:
    - "realtime": Standard Gemini API (faster, good for small batches)
    - "batch": Gemini Batch API (50% cheaper, good for large batches)
    """

    @classmethod
    def create(
        cls,
        mode: str = "realtime",
        model: str = DEFAULT_MODEL,
        workers: int = 4,
        poll_interval: int = 60,
    ) -> Annotator:
        """Create an annotator instance.

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
        if mode == "batch":
            return AnnotatorRegistry.create(
                mode=mode,
                model=model,
                workers=workers,
                poll_interval=poll_interval,
            )
        else:
            return AnnotatorRegistry.create(
                mode=mode,
                model=model,
                workers=workers,
            )

    @classmethod
    def list_modes(cls) -> list[str]:
        """Return list of available modes."""
        return AnnotatorRegistry.list_modes()
