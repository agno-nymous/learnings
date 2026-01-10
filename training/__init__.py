"""Training module for vision-language models.

Provides utilities for:
- Model training with QLoRA adapters
- Checkpoint management and cleanup
- Early stopping and high-loss sample logging
- Cost tracking and budget monitoring

Note: Callbacks require transformers to be installed. Import them
directly from training.callbacks if needed.
"""

from training.checkpoint import CheckpointManager
from training.metrics import compute_cer, compute_wer

__all__ = [
    "CheckpointManager",
    "compute_cer",
    "compute_wer",
]


def __getattr__(name: str):
    """Lazy imports for optional dependencies."""
    if name == "CheckpointCallback":
        from training.callbacks.checkpoint import CheckpointCallback

        return CheckpointCallback
    if name == "CheckpointRetentionCallback":
        # Backward compatibility alias
        from training.callbacks.checkpoint import CheckpointCallback

        return CheckpointCallback
    if name == "EarlyStoppingCallback":
        from training.callbacks.early_stopping import EarlyStoppingCallback

        return EarlyStoppingCallback
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
