"""Checkpoint management for training runs with rotating retention."""

import json
import shutil
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoint saving and cleanup with retention policy."""

    def __init__(self, output_dir: Path, keep_best: int = 5, keep_recent: int = 5):
        """Initialize checkpoint manager.

        Args:
            output_dir: Directory where checkpoints are saved.
            keep_best: Number of best (lowest eval loss) checkpoints to keep.
            keep_recent: Number of most recent checkpoints to keep.
        """
        self.output_dir = Path(output_dir)
        self.keep_best = keep_best
        self.keep_recent = keep_recent
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_checkpoint_losses(self) -> Dict[str, float]:
        """Get validation loss for all checkpoints.

        Returns:
            Dict mapping checkpoint name to eval_loss.
        """
        checkpoints = {}
        for chk_dir in self.output_dir.glob("checkpoint-*"):
            state_file = chk_dir / "trainer_state.json"
            if state_file.exists():
                try:
                    state = json.loads(state_file.read_text())
                    if "eval_loss" in state:
                        checkpoints[chk_dir.name] = state["eval_loss"]
                except (json.JSONDecodeError, KeyError):
                    logger.warning(f"Could not read eval_loss from {state_file}")
        return checkpoints

    def cleanup(self, checkpoints: Dict[str, float]) -> None:
        """Remove checkpoints outside retention policy.

        Keeps:
        - Top `keep_best` checkpoints by lowest eval_loss
        - Most recent `keep_recent` checkpoints

        Args:
            checkpoints: Dict mapping checkpoint name to eval_loss.
        """
        if len(checkpoints) <= self.keep_best + self.keep_recent:
            return  # Nothing to clean

        # Sort by loss (ascending) - keep best
        sorted_by_loss = sorted(checkpoints.items(), key=lambda x: x[1])
        best_names = {name for name, _ in sorted_by_loss[:self.keep_best]}

        # Sort by step number (descending) - keep recent
        sorted_by_step = sorted(
            checkpoints.items(),
            key=lambda x: int(x[0].split("-")[1]) if x[0].split("-")[1].isdigit() else 0,
            reverse=True
        )
        recent_names = {name for name, _ in sorted_by_step[:self.keep_recent]}

        # Union of best + recent to keep
        keep_names = best_names | recent_names

        # Remove others
        for chk_name in checkpoints:
            if chk_name not in keep_names:
                chk_path = self.output_dir / chk_name
                shutil.rmtree(chk_path)
                logger.info(f"Removed checkpoint: {chk_name}")


def get_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    """Get the most recent checkpoint by step number.

    Args:
        output_dir: Directory containing checkpoints.

    Returns:
        Path to latest checkpoint, or None if no checkpoints exist.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return None

    checkpoints = list(output_dir.glob("checkpoint-*"))
    if not checkpoints:
        return None

    # Sort by step number (descending)
    checkpoints.sort(
        key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else 0,
        reverse=True
    )
    return checkpoints[0]
