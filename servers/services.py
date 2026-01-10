"""Service layer for the OCR Pipeline web server.

Provides service classes that encapsulate business logic and state,
enabling dependency injection with FastAPI's Depends.
"""

import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.config import DATASETS, get_dataset
from core.jsonl_utils import read_entries

logger = logging.getLogger(__name__)


@dataclass
class JobStatus:
    """Status of a background job."""

    running: bool = False
    type: str | None = None
    dataset: str | None = None
    process: subprocess.Popen | None = field(default=None, repr=False)


class JobService:
    """Service for managing background OCR jobs.

    Handles job execution, status tracking, and logging.
    Uses the Single Responsibility Principle - only manages job lifecycle.

    Attributes:
        project_root: Root directory of the project.
        log_file: Path to the execution log file.
        status: Current job status.
    """

    def __init__(self, project_root: Path, log_file: Path | None = None) -> None:
        """Initialize the job service.

        Args:
            project_root: Root directory of the project.
            log_file: Path to log file (default: project_root/ocr_execution.log).
        """
        self.project_root = Path(project_root)
        self.log_file = log_file or (self.project_root / "ocr_execution.log")
        self.status = JobStatus()

    def is_running(self) -> bool:
        """Check if a job is currently running."""
        return self.status.running

    def get_status(self) -> dict[str, Any]:
        """Get current job status as a dict."""
        return {
            "running": self.status.running,
            "type": self.status.type,
            "dataset": self.status.dataset,
        }

    def run_annotation(
        self,
        model: str,
        mode: str,
        dataset: str,
        workers: int = 4,
    ) -> None:
        """Run annotation using the unified CLI.

        Args:
            model: Gemini model identifier.
            mode: Annotation mode ('realtime' or 'batch').
            dataset: Dataset name to annotate.
            workers: Number of parallel workers.
        """
        self.status.running = True
        self.status.type = mode
        self.status.dataset = dataset

        try:
            cmd = [
                sys.executable,
                str(self.project_root / "annotate.py"),
                "--dataset",
                dataset,
                "--mode",
                mode,
                "--model",
                model,
                "--workers",
                str(workers),
            ]
            self._run_command(f"{mode} OCR on {dataset} with {model}", cmd)
        finally:
            self._reset_status()

    def run_pipeline_command(self, name: str, cmd: list[str]) -> None:
        """Run a pipeline command in background.

        Args:
            name: Human-readable name for the command.
            cmd: Command and arguments to execute.
        """
        self.status.running = True
        self.status.type = name
        self.status.dataset = None

        try:
            self._run_command(name, cmd)
        finally:
            self._reset_status()

    def _run_command(self, name: str, cmd: list[str]) -> None:
        """Execute a command and log output.

        Args:
            name: Human-readable name for logging.
            cmd: Command and arguments.
        """
        logger.info(f"Starting: {name}")
        logger.info(f"Command: {' '.join(cmd)}")

        try:
            with open(self.log_file, "a") as f:
                f.write(f"\n--- Starting {name} ---\n")
                f.flush()
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=str(self.project_root),
                )
                self.status.process = process
                process.wait()
                f.write(f"--- Finished {name} ---\n")
                f.flush()
        except Exception as e:
            logger.error(f"Failed to run {name}: {e}")
            with open(self.log_file, "a") as f:
                f.write(f"ERROR: {e}\n")

    def _reset_status(self) -> None:
        """Reset job status to idle."""
        self.status.running = False
        self.status.type = None
        self.status.dataset = None
        self.status.process = None

    def get_logs(self, lines: int = 200) -> str:
        """Get recent execution logs.

        Args:
            lines: Number of recent lines to return.

        Returns:
            Log content as string.
        """
        if not self.log_file.exists():
            return "No execution logs yet."

        with open(self.log_file) as f:
            all_lines = f.readlines()
            return "".join(all_lines[-lines:])


class DatasetService:
    """Service for dataset operations.

    Provides methods for querying dataset information and statistics.
    Uses the Single Responsibility Principle - only handles dataset queries.

    Attributes:
        project_root: Root directory of the project.
    """

    def __init__(self, project_root: Path) -> None:
        """Initialize the dataset service.

        Args:
            project_root: Root directory of the project.
        """
        self.project_root = Path(project_root)

    def list_datasets(self) -> dict[str, Any]:
        """List available datasets.

        Returns:
            Dict with 'datasets' list and 'default' key.
        """
        return {
            "datasets": list(DATASETS.keys()),
            "default": "welllog",
        }

    def get_entries(self, dataset: str) -> list[dict] | dict[str, str]:
        """Get entries for a specific dataset.

        Args:
            dataset: Dataset name.

        Returns:
            List of entries or error dict.
        """
        try:
            ds = get_dataset(dataset)
        except ValueError as e:
            return {"error": str(e)}

        output_path = self.project_root / ds["output"]
        clean_path = output_path.with_suffix(".clean.jsonl")

        if clean_path.exists():
            return list(read_entries(clean_path))
        if output_path.exists():
            return list(read_entries(output_path))
        return []

    def get_stats(self, dataset: str) -> dict[str, Any]:
        """Get OCR statistics for a dataset.

        Args:
            dataset: Dataset name.

        Returns:
            Dict with total, success, and error counts.
        """
        try:
            ds = get_dataset(dataset)
        except ValueError as e:
            return {"error": str(e)}

        output_path = self.project_root / ds["output"]
        if not output_path.exists():
            return {"total": 0, "success": 0, "error": 0}

        total, success, error = 0, 0, 0
        for entry in read_entries(output_path):
            total += 1
            if entry.get("status") == "success":
                success += 1
            else:
                error += 1

        return {"total": total, "success": success, "error": error}

    def get_images_dir(self, dataset: str) -> Path:
        """Get the images directory for a dataset.

        Args:
            dataset: Dataset name.

        Returns:
            Path to images directory.
        """
        try:
            ds = get_dataset(dataset)
            return self.project_root / ds["images_dir"]
        except ValueError:
            # Fallback to legacy location
            return self.project_root / "cropped_headers"


# Singleton instances for dependency injection
_project_root = Path(__file__).parent.parent
_job_service: JobService | None = None
_dataset_service: DatasetService | None = None


def get_job_service() -> JobService:
    """Get the JobService singleton for FastAPI dependency injection.

    Returns:
        JobService instance.
    """
    global _job_service
    if _job_service is None:
        _job_service = JobService(project_root=_project_root)
    return _job_service


def get_dataset_service() -> DatasetService:
    """Get the DatasetService singleton for FastAPI dependency injection.

    Returns:
        DatasetService instance.
    """
    global _dataset_service
    if _dataset_service is None:
        _dataset_service = DatasetService(project_root=_project_root)
    return _dataset_service
