#!/usr/bin/env python3
"""Training script for Qwen3-VL QLoRA fine-tuning.

Extracted from finetune_qwen3_vl_qlora.ipynb for production use.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.base import TrainingConfig
# from training.dataset import load_dataset  # TODO: uncomment when dependencies installed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train Qwen3-VL model with QLoRA")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config Python file (e.g., configs/experiments/qwen3_qlora_r16.py)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from",
    )
    return parser.parse_args()


def load_config(config_path: str) -> TrainingConfig:
    """Load config from Python file.

    Args:
        config_path: Import path like 'configs.experiments.quick_val' or file path.

    Returns:
        TrainingConfig instance.
    """
    # Support both import paths and file paths
    if config_path.startswith("configs/") or config_path.startswith("configs."):
        # Import path - load module and get 'config' variable
        module_path = config_path.replace(".py", "").replace("/", ".")
        import importlib
        module = importlib.import_module(module_path)
        return module.config
    else:
        # File path - exec and get 'config' variable
        config_file = Path(config_path)
        if not config_file.is_absolute():
            # Relative to project root
            config_file = project_root / config_path

        import importlib.util
        spec = importlib.util.spec_from_file_location("config_module", config_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.config


def main():
    """Main training entry point."""
    args = parse_args()

    # Load config
    config = load_config(args.config)

    print(f"=== Training Configuration ===")
    print(f"Experiment: {config.experiment_name}")
    print(f"Model: {config.model_name}")
    print(f"Max Steps: {config.max_steps}")
    print(f"Batch Size: {config.per_device_train_batch_size} × {config.gradient_accumulation_steps} = {config.per_device_train_batch_size * config.gradient_accumulation_steps} effective")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Dataset: {config.dataset_name}")
    print(f"W&B Project: {config.wandb_project}")
    print(f"Output Dir: {config.output_dir}")
    print(f"=============================\n")

    # TODO: Load model, setup W&B, train
    print("Training implementation in next task...")


if __name__ == "__main__":
    main()
