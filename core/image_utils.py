"""Image utilities for OCR pipeline."""

import base64
import io
from pathlib import Path

from PIL import Image as PILImage

# Disable decompression bomb check once, globally
PILImage.MAX_IMAGE_PIXELS = None

# Re-export Image class with MAX_IMAGE_PIXELS already disabled
Image = PILImage


def load_as_png_bytes(image_path: Path) -> bytes:
    """Convert any image to PNG bytes for API submission.

    Handles:
    - TIFF files (not supported by most APIs)
    - RGBA/LA/P mode images (converted to RGB)
    """
    with PILImage.open(image_path) as img:
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()


def load_as_png_base64(image_path: Path) -> str:
    """Convert image to base64-encoded PNG string."""
    return base64.standard_b64encode(load_as_png_bytes(image_path)).decode("utf-8")


def get_mime_type(path: Path) -> str:
    """Get MIME type for image file."""
    suffix = path.suffix.lower()
    return {
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(suffix, "application/octet-stream")
