"""Training callbacks package.

Provides reusable callback classes for the training loop:
- EarlyStoppingCallback: Stop training based on loss thresholds
- CheckpointCallback: Manage checkpoint retention
- CostLoggingCallback: Track and log GPU costs
- EvalLoggingCallback: Log evaluation samples to W&B

Note: These callbacks require transformers to be installed.
Import them directly from submodules when needed.
"""

__all__ = [
    "CheckpointCallback",
    "CostLoggingCallback",
    "EarlyStoppingCallback",
    "EvalWithHighLossLoggingCallback",
]


def __getattr__(name: str):
    """Lazy imports for callbacks that require transformers."""
    if name == "CheckpointCallback":
        from training.callbacks.checkpoint import CheckpointCallback

        return CheckpointCallback
    if name == "CostLoggingCallback":
        from training.callbacks.cost_logging import CostLoggingCallback

        return CostLoggingCallback
    if name == "EarlyStoppingCallback":
        from training.callbacks.early_stopping import EarlyStoppingCallback

        return EarlyStoppingCallback
    if name == "EvalWithHighLossLoggingCallback":
        from training.callbacks.eval_logging import EvalWithHighLossLoggingCallback

        return EvalWithHighLossLoggingCallback
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
