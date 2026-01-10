"""Checkpoint management callback for training.

Integrates with CheckpointManager to clean up old checkpoints
based on retention policy.
"""

import logging
from typing import Any

from transformers import TrainerCallback

from training.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


class CheckpointCallback(TrainerCallback):
    """Callback to manage checkpoint retention during training.

    After each evaluation, updates the checkpoint manager with the
    current eval loss and triggers cleanup based on retention policy.

    Attributes:
        checkpoint_manager: Manager for checkpoint cleanup.
        checkpoints: Dict mapping checkpoint names to eval_loss values.
    """

    def __init__(self, checkpoint_manager: CheckpointManager) -> None:
        """Initialize the callback.

        Args:
            checkpoint_manager: CheckpointManager instance for cleanup.
        """
        self.checkpoint_manager = checkpoint_manager
        self.checkpoints: dict[str, float] = {}

    def on_evaluate(
        self,
        _args: Any,
        state: Any,
        _control: Any,
        metrics: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> None:
        """After evaluation, update checkpoint retention.

        Args:
            _args: Training arguments (unused).
            state: Trainer state with global_step.
            _control: Trainer control (unused).
            metrics: Evaluation metrics including eval_loss.
            **_kwargs: Additional arguments (unused).
        """
        # Update checkpoints dict with current eval loss
        checkpoint_name = f"checkpoint-{state.global_step}"
        if metrics and "eval_loss" in metrics:
            self.checkpoints[checkpoint_name] = metrics["eval_loss"]
            logger.debug(f"Recorded {checkpoint_name} with eval_loss={metrics['eval_loss']:.4f}")

        # Run cleanup
        self.checkpoint_manager.cleanup(self.checkpoints)
