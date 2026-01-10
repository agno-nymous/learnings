"""Early stopping callback for training.

Provides flexible early stopping based on:
- Training loss threshold
- Evaluation loss patience
"""

import logging
from typing import Any

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class EarlyStoppingCallback(TrainerCallback):
    """Early stopping callback with train loss threshold and eval loss patience.

    Stops training when:
    1. Training loss drops below threshold (default 0.2), OR
    2. Eval loss doesn't improve for N consecutive evaluations (patience)

    This is the consolidated implementation combining features from both
    the simple patience-based and threshold-based approaches.

    Attributes:
        train_loss_threshold: Stop when train_loss drops below this value.
        patience: Number of evals to wait without improvement.
        min_evals: Minimum evaluations before early stopping applies.
        min_delta: Minimum change to qualify as improvement.
    """

    def __init__(
        self,
        train_loss_threshold: float = 0.2,
        patience: int = 3,
        min_evals: int = 2,
        min_delta: float = 0.001,
    ) -> None:
        """Initialize early stopping callback.

        Args:
            train_loss_threshold: Stop training when train_loss drops below this.
            patience: Number of evals to wait without improvement before stopping.
            min_evals: Minimum number of evaluations before early stopping applies.
            min_delta: Minimum change to qualify as improvement.
        """
        self.train_loss_threshold = train_loss_threshold
        self.patience = patience
        self.min_evals = min_evals
        self.min_delta = min_delta

        # State
        self.best_eval_loss: float = float("inf")
        self.evals_without_improvement: int = 0
        self._threshold_reached: bool = False
        self._eval_count: int = 0

    def on_log(
        self,
        _args: Any,
        _state: Any,
        control: Any,
        logs: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> None:
        """Check training loss threshold after each step.

        Args:
            _args: Training arguments (unused).
            _state: Trainer state (unused).
            control: Trainer control object.
            logs: Current log values including loss.
            **_kwargs: Additional arguments (unused).
        """
        if logs is None:
            return

        # Check if train_loss is below threshold
        train_loss = logs.get("loss")
        if (
            train_loss is not None
            and train_loss < self.train_loss_threshold
            and not self._threshold_reached
        ):
            self._threshold_reached = True
            logger.info(
                f"Train loss {train_loss:.4f} below threshold "
                f"{self.train_loss_threshold}. Will run eval and stop."
            )
            control.should_evaluate = True
            control.should_training_stop = True

    def on_evaluate(
        self,
        _args: Any,
        state: Any,
        control: Any,
        metrics: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> None:
        """Check for eval loss improvement and apply patience-based stopping.

        Args:
            _args: Training arguments (unused).
            state: Trainer state with global_step.
            control: Trainer control object.
            metrics: Evaluation metrics including eval_loss.
            **_kwargs: Additional arguments (unused).
        """
        if metrics is None:
            return

        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return

        self._eval_count += 1

        # Track best eval loss with min_delta threshold
        if eval_loss < self.best_eval_loss - self.min_delta:
            self.best_eval_loss = eval_loss
            self.evals_without_improvement = 0
            logger.info(f"New best eval_loss: {eval_loss:.4f}")
        else:
            self.evals_without_improvement += 1
            logger.info(
                f"Eval loss {eval_loss:.4f} not improved. "
                f"{self.evals_without_improvement}/{self.patience} without improvement"
            )

        # Check if we should stop due to patience (only after min_evals)
        if (
            state.global_step > 0
            and self._eval_count >= self.min_evals
            and self.evals_without_improvement >= self.patience
        ):
            logger.info(
                f"Early stopping triggered: eval_loss hasn't improved for "
                f"{self.patience} evaluations. Best eval_loss: {self.best_eval_loss:.4f}"
            )
            control.should_training_stop = True

    def reset(self) -> None:
        """Reset callback state (useful for retraining)."""
        self.best_eval_loss = float("inf")
        self.evals_without_improvement = 0
        self._threshold_reached = False
        self._eval_count = 0
