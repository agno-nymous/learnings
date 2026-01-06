"""Early stopping callback for training."""

import logging

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class EarlyStoppingCallback(TrainerCallback):
    """Stop training if validation loss plateaus."""

    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        """Initialize early stopping.

        Args:
            patience: Number of evals with no improvement before stopping.
            min_delta: Minimum change to qualify as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.wait = 0
        self.stopped = False

    def on_evaluate(self, _args, _state, control, metrics=None, **_kwargs):
        """Check if we should stop training.

        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control.
            metrics: Evaluation metrics.
            **kwargs: Additional keyword arguments.
        """
        if metrics is None or "eval_loss" not in metrics:
            return

        current_loss = metrics["eval_loss"]

        if self.best_loss is None:
            self.best_loss = current_loss
        elif current_loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = current_loss
            self.wait = 0
            logger.info(f"New best loss: {current_loss:.4f}")
        else:
            # No improvement
            self.wait += 1
            logger.info(
                f"No improvement for {self.wait} evals. Best: {self.best_loss:.4f}, Current: {current_loss:.4f}"
            )

            if self.wait >= self.patience:
                logger.info(
                    f"Early stopping triggered after {self.patience} evals with no improvement"
                )
                control.should_training_stop = True
                self.stopped = True
