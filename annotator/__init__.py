"""Annotator module for OCR using Gemini.

Provides:
    - Annotator: Abstract base class for annotators
    - AnnotatorRegistry: Registration-based factory (Open/Closed Principle)
    - AnnotatorFactory: Backward-compatible factory wrapper
    - GeminiAnnotator: Real-time Gemini API annotator
    - GeminiBatchAnnotator: Batch API annotator (50% cheaper)
    - GeminiClientMixin: Mixin for lazy Gemini client initialization
"""

from .base import Annotator
from .client import GeminiClientMixin
from .factory import AnnotatorFactory, AnnotatorRegistry
from .gemini_annotator import GeminiAnnotator
from .gemini_batch_annotator import GeminiBatchAnnotator

__all__ = [
    "Annotator",
    "AnnotatorFactory",
    "AnnotatorRegistry",
    "GeminiAnnotator",
    "GeminiBatchAnnotator",
    "GeminiClientMixin",
]
