"""Training module for vision-language models.

Provides utilities for:
- Model training with QLoRA adapters
- Checkpoint management and cleanup
- Early stopping and high-loss sample logging
- Cost tracking and budget monitoring
"""

from training.checkpoint import CheckpointManager, CheckpointRetentionCallback
from training.early_stopping import EarlyStoppingCallback
from training.metrics import compute_cer, compute_wer
from training.train import main

__all__ = [
    "CheckpointManager",
    "CheckpointRetentionCallback",
    "EarlyStoppingCallback",
    "compute_cer",
    "compute_wer",
    "main",
]
