"""Type definitions for the OCR pipeline.

Provides TypedDicts and dataclasses for type-safe data structures.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

# ===================================================================
# Annotation Types
# ===================================================================


class AnnotationEntry(TypedDict):
    """Full annotation entry with metadata."""

    filename: str
    instruction: str
    answer: str
    status: str
    model: str


class CleanEntry(TypedDict):
    """Clean annotation entry for training (no metadata)."""

    filename: str
    instruction: str
    answer: str


class AnnotationResult(TypedDict):
    """Result from annotation operations."""

    success: int
    errors: int


# ===================================================================
# Dataset Types
# ===================================================================


@dataclass
class DatasetConfig:
    """Configuration for a dataset in the registry.

    Attributes:
        images_dir: Directory containing images for this dataset.
        output: Path to the output JSONL file.
        description: Human-readable description.
    """

    images_dir: Path
    output: Path
    description: str = ""

    def __post_init__(self) -> None:
        """Ensure paths are Path objects."""
        self.images_dir = Path(self.images_dir)
        self.output = Path(self.output)


# ===================================================================
# Processing Types
# ===================================================================


@dataclass
class ProcessingResult:
    """Standard result from processing operations.

    Attributes:
        processed: Number of items successfully processed.
        failed: Number of items that failed.
        skipped: Number of items skipped (e.g., already exists).
    """

    processed: int
    failed: int = 0
    skipped: int = 0

    @property
    def total(self) -> int:
        """Total items attempted."""
        return self.processed + self.failed + self.skipped

    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        if self.total == 0:
            return 0.0
        return (self.processed / self.total) * 100


# ===================================================================
# Training Types
# ===================================================================


@dataclass
class CostInfo:
    """GPU cost tracking information.

    Attributes:
        gpu_hourly_rate: Cost per hour in USD.
        elapsed_hours: Total elapsed time in hours.
        total_cost: Total accumulated cost in USD.
    """

    gpu_hourly_rate: float
    elapsed_hours: float = 0.0
    total_cost: float = 0.0


class CheckpointInfo(TypedDict):
    """Information about a training checkpoint."""

    name: str
    step: int
    eval_loss: float | None
