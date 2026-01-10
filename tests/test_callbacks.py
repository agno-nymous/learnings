"""Tests for training/callbacks/ package."""

import pytest

# Skip tests if transformers not installed
pytest.importorskip("transformers")

from training.callbacks.early_stopping import EarlyStoppingCallback  # noqa: E402
from training.cost_tracker import CostTracker, get_gpu_rate  # noqa: E402


class MockControl:
    """Mock Trainer control object."""

    def __init__(self):
        self.should_training_stop = False
        self.should_evaluate = False


class MockState:
    """Mock Trainer state object."""

    def __init__(self, global_step: int = 100):
        self.global_step = global_step


class TestEarlyStoppingCallback:
    """Tests for EarlyStoppingCallback."""

    def test_init_defaults(self):
        """Should initialize with default values."""
        callback = EarlyStoppingCallback()
        assert callback.train_loss_threshold == 0.2
        assert callback.patience == 3
        assert callback.min_evals == 2
        assert callback.best_eval_loss == float("inf")

    def test_train_loss_threshold_triggers_stop(self):
        """Should trigger stop when train loss below threshold."""
        callback = EarlyStoppingCallback(train_loss_threshold=0.5)
        control = MockControl()

        # Loss above threshold - no stop
        callback.on_log(None, None, control, logs={"loss": 0.8})
        assert not control.should_training_stop

        # Loss below threshold - stop
        callback.on_log(None, None, control, logs={"loss": 0.3})
        assert control.should_training_stop
        assert control.should_evaluate

    def test_eval_loss_improvement_resets_counter(self):
        """Should reset counter when eval loss improves."""
        callback = EarlyStoppingCallback(patience=3)
        control = MockControl()
        state = MockState()

        # First eval - sets baseline
        callback.on_evaluate(None, state, control, metrics={"eval_loss": 1.0})
        assert callback.best_eval_loss == 1.0
        assert callback.evals_without_improvement == 0

        # Worse eval - increments counter
        callback.on_evaluate(None, state, control, metrics={"eval_loss": 1.1})
        assert callback.evals_without_improvement == 1

        # Better eval - resets counter
        callback.on_evaluate(None, state, control, metrics={"eval_loss": 0.8})
        assert callback.best_eval_loss == 0.8
        assert callback.evals_without_improvement == 0

    def test_patience_triggers_stop(self):
        """Should stop after patience exceeded."""
        callback = EarlyStoppingCallback(patience=2, min_evals=2)
        control = MockControl()
        state = MockState()

        # First eval - sets baseline
        callback.on_evaluate(None, state, control, metrics={"eval_loss": 1.0})

        # Two evals without improvement
        callback.on_evaluate(None, state, control, metrics={"eval_loss": 1.1})
        assert not control.should_training_stop

        callback.on_evaluate(None, state, control, metrics={"eval_loss": 1.2})
        assert control.should_training_stop

    def test_min_evals_prevents_early_stop(self):
        """Should not stop before min_evals reached."""
        callback = EarlyStoppingCallback(patience=1, min_evals=5)
        control = MockControl()
        state = MockState()

        # Set baseline and trigger patience
        callback.on_evaluate(None, state, control, metrics={"eval_loss": 1.0})
        callback.on_evaluate(None, state, control, metrics={"eval_loss": 1.1})

        # Should not stop because min_evals not reached
        assert not control.should_training_stop

    def test_reset_clears_state(self):
        """reset() should clear all state."""
        callback = EarlyStoppingCallback()
        callback.best_eval_loss = 0.5
        callback.evals_without_improvement = 3
        callback._threshold_reached = True
        callback._eval_count = 10

        callback.reset()

        assert callback.best_eval_loss == float("inf")
        assert callback.evals_without_improvement == 0
        assert not callback._threshold_reached
        assert callback._eval_count == 0


class TestCostTracker:
    """Tests for CostTracker."""

    def test_init(self):
        """Should initialize with hourly rate."""
        tracker = CostTracker(gpu_hourly_rate=0.40)
        assert tracker.gpu_hourly_rate == 0.40
        assert tracker.total_cost == 0.0

    def test_update_before_start_returns_zero(self):
        """Should return 0 if not started."""
        tracker = CostTracker(gpu_hourly_rate=0.40)
        assert tracker.update() == 0.0

    def test_elapsed_hours_before_start(self):
        """Should return 0 elapsed hours if not started."""
        tracker = CostTracker(gpu_hourly_rate=0.40)
        assert tracker.elapsed_hours == 0.0


class TestGetGpuRate:
    """Tests for get_gpu_rate function."""

    def test_known_gpu(self):
        """Should return correct rate for known GPUs."""
        assert get_gpu_rate("RTX_4090") == 0.40
        assert get_gpu_rate("A100") == 1.50

    def test_unknown_gpu_returns_default(self):
        """Should return default rate for unknown GPUs."""
        assert get_gpu_rate("UNKNOWN_GPU") == 0.40
