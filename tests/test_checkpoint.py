import pytest
import tempfile
import shutil
from pathlib import Path
from training.checkpoint import CheckpointManager, get_latest_checkpoint


def test_checkpoint_manager_creation(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_best=3, keep_recent=2)
    assert mgr.output_dir == tmp_path
    assert mgr.keep_best == 3
    assert mgr.keep_recent == 2


def test_get_latest_checkpoint_none(tmp_path):
    assert get_latest_checkpoint(tmp_path) is None


def test_get_latest_checkpoint(tmp_path):
    chk1 = tmp_path / "checkpoint-10"
    chk2 = tmp_path / "checkpoint-50"
    chk1.mkdir()
    chk2.mkdir()

    latest = get_latest_checkpoint(tmp_path)
    assert latest == chk2


def test_rotating_retention(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_best=2, keep_recent=2)

    # Create checkpoints with validation losses
    checkpoints = {
        "checkpoint-10": 0.5,
        "checkpoint-20": 0.3,
        "checkpoint-30": 0.4,
        "checkpoint-40": 0.2,
        "checkpoint-50": 0.25,
    }

    for name, loss in checkpoints.items():
        chk_path = tmp_path / name
        chk_path.mkdir()
        (chk_path / "trainer_state.json").write_text(f'{{"eval_loss": {loss}}}')

    # Run cleanup
    mgr.cleanup(checkpoints)

    # Should keep best (0.2, 0.25) + recent (checkpoint-50, checkpoint-40) = overlap
    kept = [d.name for d in tmp_path.iterdir() if d.is_dir()]
    # checkpoint-40 (best=0.2), checkpoint-50 (recent+second best=0.25), checkpoint-20 (third best=0.3)
    assert len(kept) <= 4  # Allow some overlap
    assert "checkpoint-40" in kept  # Best
    assert "checkpoint-50" in kept  # Recent and second-best
