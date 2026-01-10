"""Gemini API client utilities.

Provides a mixin class for lazy-loaded Gemini client initialization,
reducing code duplication across annotator implementations.
"""

import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from google import genai

load_dotenv()


class GeminiClientMixin:
    """Mixin providing lazy-loaded Gemini API client.

    Subclasses can access self.client to get a configured Gemini client.
    The client is initialized lazily on first access.

    Example:
        class MyAnnotator(Annotator, GeminiClientMixin):
            def annotate(self, images, output):
                response = self.client.models.generate_content(...)
    """

    _client: "genai.Client | None" = None

    @property
    def client(self) -> "genai.Client":
        """Get or create the Gemini API client.

        Returns:
            Configured genai.Client instance.

        Raises:
            ValueError: If GOOGLE_API_KEY environment variable is not set.
        """
        if self._client is None:
            from google import genai

            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "GOOGLE_API_KEY not set. " "Get one at https://aistudio.google.com/app/apikey"
                )
            self._client = genai.Client(api_key=api_key)
        return self._client

    def reset_client(self) -> None:
        """Reset the client (useful for testing or reconnection)."""
        self._client = None
