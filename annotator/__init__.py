"""Annotator module for OCR using Gemini."""

from .base import Annotator
from .factory import AnnotatorFactory

__all__ = ["AnnotatorFactory", "Annotator"]
