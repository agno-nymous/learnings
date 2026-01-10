"""Cost logging callback for training.

Tracks GPU compute costs and logs to W&B.
"""

import logging
from typing import Any

import wandb
from transformers import TrainerCallback

from training.cost_tracker import CostTracker

logger = logging.getLogger(__name__)


class CostLoggingCallback(TrainerCallback):
    """Log training cost to W&B during training.

    Tracks elapsed time and calculates cost based on GPU hourly rate.
    Logs to W&B and warns when approaching budget cap.

    Attributes:
        cost_tracker: CostTracker instance for monitoring compute costs.
        budget_cap: Maximum budget in USD before warnings.
    """

    def __init__(self, cost_tracker: CostTracker, budget_cap: float) -> None:
        """Initialize the cost logging callback.

        Args:
            cost_tracker: CostTracker instance for monitoring compute costs.
            budget_cap: Maximum budget in USD before warnings.
        """
        self.cost_tracker = cost_tracker
        self.budget_cap = budget_cap

    def on_log(
        self,
        _args: Any,
        _state: Any,
        _control: Any,
        _logs: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> None:
        """Log current cost to W&B.

        Args:
            _args: Training arguments (unused).
            _state: Trainer state (unused).
            _control: Trainer control (unused).
            _logs: Current logs (unused).
            **_kwargs: Additional arguments (unused).
        """
        current_cost = self.cost_tracker.update()

        if wandb.run is not None:
            wandb.log({"cost_usd": current_cost})

        # Warn if approaching budget
        if current_cost > self.budget_cap * 0.9:
            logger.warning(f"Approaching budget cap: ${current_cost:.2f} / ${self.budget_cap:.2f}")
