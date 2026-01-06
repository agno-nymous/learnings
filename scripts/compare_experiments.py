#!/usr/bin/env python3
"""Compare training experiments from W&B or local logs."""

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare training experiments")
    parser.add_argument("--runs", nargs="+", required=True, help="W&B run IDs or checkpoint paths")
    parser.add_argument(
        "--metrics", nargs="+", default=["eval_loss", "cer", "wer"], help="Metrics to compare"
    )
    return parser.parse_args()


def load_run_metrics(run_id: str) -> dict[str, Any]:
    """Load metrics from a run (W&B or local).

    Args:
        run_id: W&B run ID or local checkpoint path.

    Returns:
        Dict with metrics.
    """
    # Try W&B first
    try:
        import wandb

        api = wandb.Api()
        run = api.run(run_id)
        return run.summary
    except Exception:  # noqa: S110 - W&B API not available, will fallback to local
        pass

    # Fallback to local checkpoint
    path = Path(run_id)
    if path.exists():
        state_file = path / "trainer_state.json"
        if state_file.exists():
            return json.loads(state_file.read_text())

    raise ValueError(f"Could not load metrics for {run_id}")


def main() -> None:
    """Compare metrics across multiple experiment runs."""
    args = parse_args()

    print(f"{'Run ID':<40} {'Eval Loss':<12} {'CER':<8} {'WER':<8}")
    print("-" * 70)

    for run_id in args.runs:
        try:
            metrics = load_run_metrics(run_id)
            eval_loss = metrics.get("eval_loss", "N/A")
            cer = metrics.get("cer", "N/A")
            wer = metrics.get("wer", "N/A")

            print(f"{run_id:<40} {str(eval_loss):<12} {str(cer):<8} {str(wer):<8}")
        except Exception as e:
            print(f"{run_id:<40} ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
