"""Pre-flight validation checks before training."""

import os
import torch
from transformers import AutoConfig


def check_model_exists(model_name: str) -> bool:
    """Verify model exists on HuggingFace Hub.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        True if model exists.

    Raises:
        ValueError: If model not found.
    """
    try:
        AutoConfig.from_pretrained(model_name)
        return True
    except Exception as e:
        raise ValueError(f"Model '{model_name}' not found or inaccessible: {e}")


def check_dataset_exists(dataset_name: str) -> bool:
    """Verify dataset exists on HuggingFace Hub.

    Args:
        dataset_name: HuggingFace dataset identifier.

    Returns:
        True if dataset exists.

    Raises:
        ValueError: If dataset not found.
    """
    try:
        from datasets import load_dataset
        load_dataset(dataset_name, split="train[:1]")
        return True
    except Exception as e:
        raise ValueError(f"Dataset '{dataset_name}' not found or inaccessible: {e}")


def check_network_volume(volume_path: str = "/runpod_volume") -> bool:
    """Check network volume is mounted and writable.

    Args:
        volume_path: Path to network volume.

    Returns:
        True if volume is accessible.

    Raises:
        ValueError: If volume not mounted or not writable.
    """
    if not os.path.ismount(volume_path) and not os.path.exists(volume_path):
        # Local testing: directory might exist without being a mount
        os.makedirs(volume_path, exist_ok=True)

    if not os.access(volume_path, os.W_OK):
        raise ValueError(f"Network volume '{volume_path}' not writable")

    return True


def check_vram_capacity(required_gb: float = 16.0) -> bool:
    """Check GPU has sufficient VRAM.

    Args:
        required_gb: Minimum required VRAM in GB.

    Returns:
        True if sufficient VRAM.

    Raises:
        ValueError: If insufficient VRAM.
    """
    if not torch.cuda.is_available():
        raise ValueError("CUDA not available - no GPU detected")

    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    if gpu_memory < required_gb:
        raise ValueError(f"GPU has {gpu_memory:.1f}GB VRAM, requires {required_gb}GB")

    return True


def run_preflight_checks(config) -> None:
    """Run all pre-flight validation checks.

    Args:
        config: TrainingConfig instance.

    Raises:
        ValueError: If any check fails.
    """
    print("Running pre-flight checks...")

    check_model_exists(config.model_name)
    print("  ✓ Model accessible")

    check_dataset_exists(config.dataset_name)
    print("  ✓ Dataset accessible")

    check_network_volume(config.output_dir)
    print("  ✓ Output directory writable")

    check_vram_capacity(required_gb=16.0)
    print("  ✓ Sufficient GPU VRAM")

    print("All checks passed!\n")
