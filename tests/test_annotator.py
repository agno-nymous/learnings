"""Tests for annotator module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from annotator import (
    Annotator,
    AnnotatorFactory,
    AnnotatorRegistry,
    GeminiAnnotator,
    GeminiBatchAnnotator,
    GeminiClientMixin,
)
from core.types import AnnotationResult


class TestAnnotatorBase:
    """Tests for Annotator abstract base class."""

    def test_cannot_instantiate_directly(self):
        """Should not be able to instantiate abstract class."""
        with pytest.raises(TypeError):
            Annotator()

    def test_concrete_implementation(self):
        """Should be able to create a concrete implementation."""

        class TestAnnotator(Annotator):
            @property
            def name(self) -> str:
                return "Test Annotator"

            @property
            def mode(self) -> str:
                return "test"

            def annotate(
                self,
                image_paths: list[Path],
                output_path: Path,  # noqa: ARG002
            ) -> AnnotationResult:
                return {"success": len(image_paths), "errors": 0}

        annotator = TestAnnotator()
        assert annotator.name == "Test Annotator"
        assert annotator.mode == "test"

        result = annotator.annotate([Path("test.png")], Path("output.jsonl"))
        assert result["success"] == 1
        assert result["errors"] == 0


class TestGeminiClientMixin:
    """Tests for GeminiClientMixin."""

    def test_client_lazy_initialization(self):
        """Client should be lazily initialized."""

        class TestClass(GeminiClientMixin):
            def __init__(self):
                self._client = None

        obj = TestClass()
        assert obj._client is None

    def test_client_raises_without_api_key(self):
        """Should raise ValueError if GOOGLE_API_KEY not set."""

        class TestClass(GeminiClientMixin):
            def __init__(self):
                self._client = None

        obj = TestClass()

        with (
            patch.dict("os.environ", {}, clear=True),
            patch("annotator.client.os.getenv", return_value=None),
            pytest.raises(ValueError, match="GOOGLE_API_KEY not set"),
        ):
            _ = obj.client

    def test_reset_client(self):
        """reset_client should set _client to None."""

        class TestClass(GeminiClientMixin):
            def __init__(self):
                self._client = MagicMock()

        obj = TestClass()
        obj._client = MagicMock()
        obj.reset_client()
        assert obj._client is None


class TestAnnotatorRegistry:
    """Tests for AnnotatorRegistry."""

    def setup_method(self):
        """Store original registry state."""
        self._original_annotators = AnnotatorRegistry._annotators.copy()
        self._original_kwargs = AnnotatorRegistry._default_kwargs.copy()

    def teardown_method(self):
        """Restore original registry state."""
        AnnotatorRegistry._annotators = self._original_annotators
        AnnotatorRegistry._default_kwargs = self._original_kwargs

    def test_builtin_modes_registered(self):
        """Built-in modes should be registered."""
        modes = AnnotatorRegistry.list_modes()
        assert "realtime" in modes
        assert "batch" in modes

    def test_create_realtime(self):
        """Should create GeminiAnnotator for realtime mode."""
        annotator = AnnotatorRegistry.create(mode="realtime", model="test-model")
        assert isinstance(annotator, GeminiAnnotator)
        assert annotator.model == "test-model"

    def test_create_batch(self):
        """Should create GeminiBatchAnnotator for batch mode."""
        annotator = AnnotatorRegistry.create(mode="batch", model="test-model")
        assert isinstance(annotator, GeminiBatchAnnotator)
        assert annotator.model == "test-model"

    def test_create_with_kwargs(self):
        """Should pass kwargs to constructor."""
        annotator = AnnotatorRegistry.create(
            mode="realtime",
            model="test-model",
            workers=8,
        )
        assert annotator.workers == 8

    def test_create_unknown_mode_raises(self):
        """Should raise ValueError for unknown mode."""
        with pytest.raises(ValueError, match="Unknown mode"):
            AnnotatorRegistry.create(mode="unknown")

    def test_get_annotator_class(self):
        """Should return the correct class."""
        cls = AnnotatorRegistry.get_annotator_class("realtime")
        assert cls == GeminiAnnotator

    def test_register_decorator(self):
        """Should register new annotator via decorator."""

        @AnnotatorRegistry.register("custom_test")
        class CustomAnnotator(Annotator):
            def __init__(self, model: str, custom_param: str = "default"):
                self.model = model
                self.custom_param = custom_param

            @property
            def name(self) -> str:
                return "Custom"

            @property
            def mode(self) -> str:
                return "custom_test"

            def annotate(
                self,
                image_paths: list[Path],  # noqa: ARG002
                output_path: Path,  # noqa: ARG002
            ) -> AnnotationResult:
                return {"success": 0, "errors": 0}

        assert "custom_test" in AnnotatorRegistry.list_modes()

        annotator = AnnotatorRegistry.create(
            mode="custom_test",
            model="test",
            custom_param="value",
        )
        assert isinstance(annotator, CustomAnnotator)
        assert annotator.custom_param == "value"

    def test_register_duplicate_raises(self):
        """Should raise ValueError for duplicate registration."""
        with pytest.raises(ValueError, match="already registered"):

            @AnnotatorRegistry.register("realtime")
            class DuplicateAnnotator(Annotator):
                @property
                def name(self) -> str:
                    return "Duplicate"

                @property
                def mode(self) -> str:
                    return "realtime"

                def annotate(
                    self,
                    image_paths: list[Path],  # noqa: ARG002
                    output_path: Path,  # noqa: ARG002
                ) -> AnnotationResult:
                    return {"success": 0, "errors": 0}


class TestAnnotatorFactory:
    """Tests for backward-compatible AnnotatorFactory."""

    def test_create_realtime(self):
        """Should create GeminiAnnotator."""
        annotator = AnnotatorFactory.create(mode="realtime", model="test-model")
        assert isinstance(annotator, GeminiAnnotator)

    def test_create_batch(self):
        """Should create GeminiBatchAnnotator."""
        annotator = AnnotatorFactory.create(mode="batch", model="test-model")
        assert isinstance(annotator, GeminiBatchAnnotator)

    def test_list_modes(self):
        """Should list available modes."""
        modes = AnnotatorFactory.list_modes()
        assert "realtime" in modes
        assert "batch" in modes

    def test_create_with_workers(self):
        """Should pass workers parameter."""
        annotator = AnnotatorFactory.create(
            mode="realtime",
            model="test",
            workers=16,
        )
        assert annotator.workers == 16

    def test_create_batch_with_poll_interval(self):
        """Should pass poll_interval to batch annotator."""
        annotator = AnnotatorFactory.create(
            mode="batch",
            model="test",
            poll_interval=30,
        )
        assert annotator.poll_interval == 30


class TestGeminiAnnotator:
    """Tests for GeminiAnnotator."""

    def test_init(self):
        """Should initialize with model and workers."""
        annotator = GeminiAnnotator(model="test-model", workers=8)
        assert annotator.model == "test-model"
        assert annotator.workers == 8
        assert annotator._client is None

    def test_name_property(self):
        """name should include model."""
        annotator = GeminiAnnotator(model="gemini-pro")
        assert "gemini-pro" in annotator.name
        assert "Real-time" in annotator.name

    def test_mode_property(self):
        """mode should be 'realtime'."""
        annotator = GeminiAnnotator(model="test")
        assert annotator.mode == "realtime"


class TestGeminiBatchAnnotator:
    """Tests for GeminiBatchAnnotator."""

    def test_init(self):
        """Should initialize with model, workers, and poll_interval."""
        annotator = GeminiBatchAnnotator(
            model="test-model",
            workers=8,
            poll_interval=120,
        )
        assert annotator.model == "test-model"
        assert annotator.workers == 8
        assert annotator.poll_interval == 120
        assert annotator._client is None

    def test_name_property(self):
        """name should include model."""
        annotator = GeminiBatchAnnotator(model="gemini-pro")
        assert "gemini-pro" in annotator.name
        assert "Batch" in annotator.name

    def test_mode_property(self):
        """mode should be 'batch'."""
        annotator = GeminiBatchAnnotator(model="test")
        assert annotator.mode == "batch"

    def test_parse_result_line_success(self):
        """Should parse successful result line."""
        annotator = GeminiBatchAnnotator(model="test-model")

        line = (
            '{"key": "test.png", "response": {"candidates": [{"content": '
            '{"parts": [{"text": "OCR result"}]}}]}}'
        )
        result = annotator._parse_result_line(line)

        assert result is not None
        assert result["filename"] == "test.png"
        assert result["answer"] == "OCR result"
        assert result["status"] == "success"
        assert result["model"] == "test-model"

    def test_parse_result_line_error(self):
        """Should parse error result line."""
        annotator = GeminiBatchAnnotator(model="test-model")

        line = '{"key": "test.png", "error": "API error"}'
        result = annotator._parse_result_line(line)

        assert result is not None
        assert result["filename"] == "test.png"
        assert result["status"] == "error: API error"
        assert result["answer"] == ""

    def test_parse_result_line_empty_response(self):
        """Should handle empty response."""
        annotator = GeminiBatchAnnotator(model="test-model")

        line = '{"key": "test.png", "response": {"candidates": []}}'
        result = annotator._parse_result_line(line)

        assert result is not None
        assert result["status"] == "error: empty response"

    def test_parse_result_line_invalid_json(self):
        """Should return None for invalid JSON."""
        annotator = GeminiBatchAnnotator(model="test-model")

        result = annotator._parse_result_line("invalid json")
        assert result is None
