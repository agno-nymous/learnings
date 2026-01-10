"""Cost tracking for training runs.

Tracks GPU compute costs based on hourly rate and elapsed time.
"""

import time
from dataclasses import dataclass, field


@dataclass
class CostTracker:
    """Track training costs based on GPU hourly rate.

    Attributes:
        gpu_hourly_rate: Cost per hour in USD (e.g., 0.40 for RTX 4090 spot).
        start_time: Timestamp when tracking started.
        total_cost: Accumulated cost in USD.
    """

    gpu_hourly_rate: float
    start_time: float = field(default=0.0, repr=False)
    total_cost: float = 0.0

    def start(self) -> None:
        """Start tracking training time."""
        self.start_time = time.time()

    def update(self) -> float:
        """Update total cost and return current value.

        Returns:
            Current accumulated cost in USD.
        """
        if self.start_time == 0.0:
            return 0.0
        elapsed_hours = (time.time() - self.start_time) / 3600
        self.total_cost = elapsed_hours * self.gpu_hourly_rate
        return self.total_cost

    @property
    def elapsed_hours(self) -> float:
        """Get elapsed time in hours.

        Returns:
            Hours since start() was called.
        """
        if self.start_time == 0.0:
            return 0.0
        return (time.time() - self.start_time) / 3600


# GPU hourly rates for spot instances (approximate USD)
GPU_RATES = {
    "RTX_3090": 0.25,
    "RTX_4090": 0.40,
    "A100": 1.50,
    "H100": 2.50,
}


def get_gpu_rate(gpu_type: str) -> float:
    """Get hourly rate for a GPU type.

    Args:
        gpu_type: GPU identifier (e.g., "RTX_4090", "A100").

    Returns:
        Hourly rate in USD, defaults to 0.40 if unknown.
    """
    return GPU_RATES.get(gpu_type, 0.40)
