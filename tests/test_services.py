"""Tests for servers/services.py."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from servers.services import DatasetService, JobService, JobStatus


class TestJobStatus:
    """Tests for JobStatus dataclass."""

    def test_default_values(self):
        """Should have correct default values."""
        status = JobStatus()
        assert status.running is False
        assert status.type is None
        assert status.dataset is None
        assert status.process is None


class TestJobService:
    """Tests for JobService."""

    def test_init(self):
        """Should initialize with project_root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = JobService(project_root=Path(tmpdir))
            assert service.project_root == Path(tmpdir)
            assert service.log_file == Path(tmpdir) / "ocr_execution.log"

    def test_init_custom_log_file(self):
        """Should accept custom log file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "custom.log"
            service = JobService(project_root=Path(tmpdir), log_file=log_file)
            assert service.log_file == log_file

    def test_is_running_default(self):
        """Should return False when no job running."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = JobService(project_root=Path(tmpdir))
            assert service.is_running() is False

    def test_get_status_default(self):
        """Should return idle status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = JobService(project_root=Path(tmpdir))
            status = service.get_status()

            assert status["running"] is False
            assert status["type"] is None
            assert status["dataset"] is None

    def test_get_logs_no_file(self):
        """Should return message when no log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = JobService(project_root=Path(tmpdir))
            logs = service.get_logs()
            assert "No execution logs" in logs

    def test_get_logs_with_content(self):
        """Should return log content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            log_file.write_text("Line 1\nLine 2\nLine 3\n")

            service = JobService(project_root=Path(tmpdir), log_file=log_file)
            logs = service.get_logs()

            assert "Line 1" in logs
            assert "Line 2" in logs
            assert "Line 3" in logs

    def test_get_logs_limits_lines(self):
        """Should limit returned lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            # Write many lines
            lines = [f"Line {i}\n" for i in range(300)]
            log_file.write_text("".join(lines))

            service = JobService(project_root=Path(tmpdir), log_file=log_file)
            logs = service.get_logs(lines=10)

            # Should only have last 10 lines
            assert "Line 290" in logs
            assert "Line 299" in logs
            assert "Line 0" not in logs

    def test_reset_status(self):
        """_reset_status should clear all status fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = JobService(project_root=Path(tmpdir))

            # Set some status
            service.status.running = True
            service.status.type = "test"
            service.status.dataset = "welllog"
            service.status.process = MagicMock()

            service._reset_status()

            assert service.status.running is False
            assert service.status.type is None
            assert service.status.dataset is None
            assert service.status.process is None


class TestDatasetService:
    """Tests for DatasetService."""

    def test_init(self):
        """Should initialize with project_root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = DatasetService(project_root=Path(tmpdir))
            assert service.project_root == Path(tmpdir)

    def test_list_datasets(self):
        """Should list available datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = DatasetService(project_root=Path(tmpdir))
            result = service.list_datasets()

            assert "datasets" in result
            assert "default" in result
            assert isinstance(result["datasets"], list)

    def test_get_entries_invalid_dataset(self):
        """Should return error for invalid dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = DatasetService(project_root=Path(tmpdir))
            result = service.get_entries("nonexistent")

            assert "error" in result

    def test_get_stats_invalid_dataset(self):
        """Should return error for invalid dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = DatasetService(project_root=Path(tmpdir))
            result = service.get_stats("nonexistent")

            assert "error" in result

    def test_get_images_dir_fallback(self):
        """Should fallback for invalid dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = DatasetService(project_root=Path(tmpdir))
            result = service.get_images_dir("nonexistent")

            # Should fallback to cropped_headers
            assert "cropped_headers" in str(result)


class TestDependencyInjection:
    """Tests for dependency injection functions."""

    def test_get_job_service_singleton(self):
        """get_job_service should return same instance."""
        from servers.services import get_job_service

        service1 = get_job_service()
        service2 = get_job_service()

        assert service1 is service2

    def test_get_dataset_service_singleton(self):
        """get_dataset_service should return same instance."""
        from servers.services import get_dataset_service

        service1 = get_dataset_service()
        service2 = get_dataset_service()

        assert service1 is service2
