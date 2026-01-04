"""Annotator module for OCR using Gemini."""
from .factory import AnnotatorFactory
from .base import Annotator

__all__ = ["AnnotatorFactory", "Annotator"]
