"""Tests for preprocessing module."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from core.types import ProcessingResult
from preprocessing import (
    Cropper,
    Downloader,
    Preprocessor,
    PreprocessorRegistry,
    Rotator,
    Splitter,
)


class TestPreprocessorBase:
    """Tests for Preprocessor abstract base class."""

    def test_cannot_instantiate_directly(self):
        """Should not be able to instantiate abstract class."""
        with pytest.raises(TypeError):
            Preprocessor()

    def test_concrete_implementation(self):
        """Should be able to create a concrete implementation."""

        class TestPreprocessor(Preprocessor):
            @property
            def name(self) -> str:
                return "Test Preprocessor"

            def process(
                self,
                input_dir: Path,  # noqa: ARG002
                output_dir: Path | None = None,  # noqa: ARG002
                workers: int = 4,  # noqa: ARG002
            ) -> ProcessingResult:
                return ProcessingResult(processed=10, failed=1, skipped=2)

        preprocessor = TestPreprocessor()
        assert preprocessor.name == "Test Preprocessor"

        result = preprocessor.process(Path("/tmp"))
        assert result.processed == 10
        assert result.failed == 1
        assert result.skipped == 2
        assert result.total == 13

    def test_repr(self):
        """repr should include class name and preprocessor name."""

        class TestPreprocessor(Preprocessor):
            @property
            def name(self) -> str:
                return "Test"

            def process(
                self,
                input_dir: Path,  # noqa: ARG002
                output_dir: Path | None = None,  # noqa: ARG002
                workers: int = 4,  # noqa: ARG002
            ) -> ProcessingResult:
                return ProcessingResult(processed=0)

        preprocessor = TestPreprocessor()
        assert "TestPreprocessor" in repr(preprocessor)
        assert "Test" in repr(preprocessor)


class TestPreprocessorRegistry:
    """Tests for PreprocessorRegistry."""

    def setup_method(self):
        """Store original registry state."""
        self._original_registry = PreprocessorRegistry._registry.copy()

    def teardown_method(self):
        """Restore original registry state."""
        PreprocessorRegistry._registry = self._original_registry

    def test_builtin_preprocessors_registered(self):
        """Built-in preprocessors should be registered."""
        registered = PreprocessorRegistry.list_all()
        assert "download" in registered
        assert "rotate" in registered
        assert "crop" in registered
        assert "split" in registered

    def test_create_rotator(self):
        """Should create Rotator instance."""
        preprocessor = PreprocessorRegistry.create("rotate")
        assert isinstance(preprocessor, Rotator)

    def test_create_cropper(self):
        """Should create Cropper instance."""
        preprocessor = PreprocessorRegistry.create("crop")
        assert isinstance(preprocessor, Cropper)

    def test_create_splitter(self):
        """Should create Splitter instance."""
        preprocessor = PreprocessorRegistry.create("split")
        assert isinstance(preprocessor, Splitter)

    def test_create_unknown_raises(self):
        """Should raise ValueError for unknown preprocessor."""
        with pytest.raises(ValueError, match="Unknown preprocessor"):
            PreprocessorRegistry.create("unknown")

    def test_register_custom(self):
        """Should register custom preprocessor."""

        class CustomPreprocessor(Preprocessor):
            @property
            def name(self) -> str:
                return "Custom"

            def process(
                self,
                input_dir: Path,  # noqa: ARG002
                output_dir: Path | None = None,  # noqa: ARG002
                workers: int = 4,  # noqa: ARG002
            ) -> ProcessingResult:
                return ProcessingResult(processed=0)

        PreprocessorRegistry.register("custom_test", CustomPreprocessor)
        assert "custom_test" in PreprocessorRegistry.list_all()

        preprocessor = PreprocessorRegistry.create("custom_test")
        assert isinstance(preprocessor, CustomPreprocessor)


class TestDownloader:
    """Tests for Downloader preprocessor."""

    def test_init(self):
        """Should initialize with input_file and limit."""
        downloader = Downloader(input_file=Path("test.csv"), limit=100)
        assert downloader.input_file == Path("test.csv")
        assert downloader.limit == 100

    def test_name_property(self):
        """name should return descriptive string."""
        downloader = Downloader(input_file=Path("test.csv"))
        assert "Downloader" in downloader.name

    @patch("preprocessing.downloader.urllib.request.urlretrieve")
    def test_download_one_file_success(self, mock_urlretrieve):  # noqa: ARG002
        """Should download file successfully."""
        downloader = Downloader(input_file=Path("test.csv"))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            result = downloader._download_one_file("http://example.com/test.tif", output_dir)

            assert result["ok"] is True
            assert result["file"] == "test.tif"
            assert result["status"] == "downloaded"

    def test_download_one_file_exists(self):
        """Should skip existing file."""
        downloader = Downloader(input_file=Path("test.csv"))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            # Create existing file
            (output_dir / "test.tif").touch()

            result = downloader._download_one_file("http://example.com/test.tif", output_dir)

            assert result["ok"] is True
            assert result["status"] == "exists"


class TestRotator:
    """Tests for Rotator preprocessor."""

    def test_init(self):
        """Should initialize without arguments."""
        rotator = Rotator()
        assert rotator.name == "TIFF Rotator"

    def test_name_property(self):
        """name should return descriptive string."""
        rotator = Rotator()
        assert "Rotator" in rotator.name

    def test_process_empty_dir(self):
        """Should handle empty directory."""
        rotator = Rotator()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = rotator.process(input_dir=Path(tmpdir))

            assert result.processed == 0
            assert result.failed == 0


class TestCropper:
    """Tests for Cropper preprocessor."""

    def test_init(self):
        """Should initialize without arguments."""
        cropper = Cropper()
        assert cropper.name == "Header Cropper"

    def test_name_property(self):
        """name should return descriptive string."""
        cropper = Cropper()
        assert "Cropper" in cropper.name

    def test_process_empty_dir(self):
        """Should handle empty directory."""
        cropper = Cropper()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = cropper.process(input_dir=Path(tmpdir))

            assert result.processed == 0
            assert result.failed == 0


class TestSplitter:
    """Tests for Splitter preprocessor."""

    def test_init_defaults(self):
        """Should initialize with default values."""
        splitter = Splitter()
        assert splitter.train_ratio == 0.8
        assert splitter.copy is False
        assert splitter.seed == 42

    def test_init_custom(self):
        """Should initialize with custom values."""
        splitter = Splitter(train_ratio=0.7, copy=True, seed=123)
        assert splitter.train_ratio == 0.7
        assert splitter.copy is True
        assert splitter.seed == 123

    def test_name_property(self):
        """name should return descriptive string."""
        splitter = Splitter()
        assert "Splitter" in splitter.name

    def test_process_empty_dir(self):
        """Should handle empty directory."""
        splitter = Splitter()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = splitter.process(input_dir=Path(tmpdir))

            assert result.processed == 0

    def test_process_creates_train_eval_dirs(self):
        """Should create train and eval directories."""
        splitter = Splitter(copy=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "source"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()

            # Create some test images
            for i in range(10):
                (input_dir / f"image_{i}.png").touch()

            result = splitter.process(input_dir=input_dir, output_dir=output_dir)

            assert result.processed == 10
            assert (output_dir / "train").exists()
            assert (output_dir / "eval").exists()

    def test_process_respects_ratio(self):
        """Should split according to train_ratio."""
        splitter = Splitter(train_ratio=0.8, copy=True, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "source"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()

            # Create 10 test images
            for i in range(10):
                (input_dir / f"image_{i}.png").touch()

            splitter.process(input_dir=input_dir, output_dir=output_dir)

            train_count = len(list((output_dir / "train").glob("*.png")))
            eval_count = len(list((output_dir / "eval").glob("*.png")))

            assert train_count == 8  # 80%
            assert eval_count == 2  # 20%

    def test_process_deterministic(self):
        """Should produce same split with same seed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "source"
            input_dir.mkdir()

            for i in range(10):
                (input_dir / f"image_{i}.png").touch()

            # Run twice with same seed
            results = []
            for _ in range(2):
                output_dir = Path(tmpdir) / f"output_{_}"
                splitter = Splitter(copy=True, seed=42)
                splitter.process(input_dir=input_dir, output_dir=output_dir)

                train_files = sorted(f.name for f in (output_dir / "train").glob("*.png"))
                results.append(train_files)

            assert results[0] == results[1]
