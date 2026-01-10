"""Tests for core/types.py - Type definitions."""

from pathlib import Path

from core.types import (
    AnnotationEntry,
    AnnotationResult,
    CheckpointInfo,
    CleanEntry,
    CostInfo,
    DatasetConfig,
    ProcessingResult,
)


class TestDatasetConfig:
    """Tests for DatasetConfig dataclass."""

    def test_create_with_paths(self):
        """Should create config with Path objects."""
        config = DatasetConfig(
            images_dir=Path("./images"),
            output=Path("./output.jsonl"),
        )
        assert config.images_dir == Path("./images")
        assert config.output == Path("./output.jsonl")
        assert config.description == ""

    def test_create_with_strings_converts_to_paths(self):
        """Should convert string paths to Path objects."""
        config = DatasetConfig(
            images_dir="./images",  # type: ignore
            output="./output.jsonl",  # type: ignore
        )
        assert isinstance(config.images_dir, Path)
        assert isinstance(config.output, Path)

    def test_description_optional(self):
        """Description should be optional with empty default."""
        config = DatasetConfig(
            images_dir=Path("./images"),
            output=Path("./output.jsonl"),
            description="Custom description",
        )
        assert config.description == "Custom description"


class TestProcessingResult:
    """Tests for ProcessingResult dataclass."""

    def test_create_with_defaults(self):
        """Should create with default values for failed and skipped."""
        result = ProcessingResult(processed=10)
        assert result.processed == 10
        assert result.failed == 0
        assert result.skipped == 0

    def test_total_property(self):
        """Total should sum all counts."""
        result = ProcessingResult(processed=10, failed=2, skipped=3)
        assert result.total == 15

    def test_success_rate_all_processed(self):
        """Success rate should be 100% when all processed."""
        result = ProcessingResult(processed=10)
        assert result.success_rate == 100.0

    def test_success_rate_partial(self):
        """Success rate should calculate correctly with failures."""
        result = ProcessingResult(processed=8, failed=2)
        assert result.success_rate == 80.0

    def test_success_rate_zero_total(self):
        """Success rate should be 0 when total is 0."""
        result = ProcessingResult(processed=0)
        assert result.success_rate == 0.0


class TestCostInfo:
    """Tests for CostInfo dataclass."""

    def test_create_with_defaults(self):
        """Should create with default values."""
        cost = CostInfo(gpu_hourly_rate=0.40)
        assert cost.gpu_hourly_rate == 0.40
        assert cost.elapsed_hours == 0.0
        assert cost.total_cost == 0.0

    def test_create_with_all_values(self):
        """Should create with all values specified."""
        cost = CostInfo(gpu_hourly_rate=0.40, elapsed_hours=2.5, total_cost=1.0)
        assert cost.elapsed_hours == 2.5
        assert cost.total_cost == 1.0


class TestTypedDicts:
    """Tests for TypedDict definitions."""

    def test_annotation_entry_structure(self):
        """AnnotationEntry should have expected keys."""
        entry: AnnotationEntry = {
            "filename": "test.png",
            "instruction": "Convert to markdown",
            "answer": "# Test",
            "status": "success",
            "model": "gemini-3",
        }
        assert entry["filename"] == "test.png"
        assert entry["status"] == "success"

    def test_clean_entry_structure(self):
        """CleanEntry should have expected keys."""
        entry: CleanEntry = {
            "filename": "test.png",
            "instruction": "Convert to markdown",
            "answer": "# Test",
        }
        assert "status" not in entry
        assert "model" not in entry

    def test_annotation_result_structure(self):
        """AnnotationResult should have expected keys."""
        result: AnnotationResult = {"success": 10, "errors": 2}
        assert result["success"] == 10
        assert result["errors"] == 2

    def test_checkpoint_info_structure(self):
        """CheckpointInfo should have expected keys."""
        info: CheckpointInfo = {"name": "checkpoint-100", "step": 100, "eval_loss": 0.5}
        assert info["step"] == 100
        assert info["eval_loss"] == 0.5
