"""Utilities for evaluation and high-loss sample logging."""

import base64
from io import BytesIO
from typing import List, Dict, Any
import wandb

from training.metrics import compute_cer, compute_wer


def compute_sample_metrics(prediction: str, reference: str) -> Dict[str, float]:
    """Compute CER and WER for a single sample.

    Args:
        prediction: Model output text.
        reference: Ground truth text.

    Returns:
        Dict with 'cer' and 'wer' keys.
    """
    return {
        "cer": compute_cer(reference, prediction),
        "wer": compute_wer(reference, prediction),
    }


def log_high_loss_samples(
    samples: List[Dict[str, Any]],
    predictions: List[str],
    references: List[str],
    losses: List[float],
    top_k: int = 20,
    table_name: str = "high_loss_samples",
) -> None:
    """Log worst samples to W&B as a table.

    Args:
        samples: List of dataset samples (with images).
        predictions: List of model predictions.
        references: List of ground truth texts.
        losses: List of loss values per sample.
        top_k: Number of worst samples to log.
        table_name: W&B table name.
    """
    # Sort by loss (descending)
    indexed = list(enumerate(losses))
    indexed.sort(key=lambda x: x[1], reverse=True)
    worst_indices = [i for i, _ in indexed[:top_k]]

    # Create W&B table
    columns = ["index", "loss", "cer", "wer", "image", "prediction", "reference"]
    table = wandb.Table(columns=columns)

    for idx in worst_indices:
        sample = samples[idx]
        pred = predictions[idx]
        ref = references[idx]
        loss = losses[idx]

        # Compute metrics
        metrics = compute_sample_metrics(pred, ref)

        # Convert image to wandb.Image
        img = sample["images"][0]
        table.add_data(
            idx,
            round(loss, 4),
            round(metrics["cer"], 4),
            round(metrics["wer"], 4),
            wandb.Image(img),
            pred[:500] + "..." if len(pred) > 500 else pred,
            ref[:500] + "..." if len(ref) > 500 else ref,
        )

    wandb.log({table_name: table})
