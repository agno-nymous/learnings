#!/usr/bin/env python3
"""Training script for Qwen3-VL QLoRA fine-tuning.

Extracted from finetune_qwen3_vl_qlora.ipynb for production use.
"""

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.base import TrainingConfig
# from training.dataset import load_dataset  # TODO: uncomment when dependencies installed


def parse_args() -> argparse.Namespace:
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

    Raises:
        ValueError: If config file cannot be loaded or doesn't contain valid config.
    """
    try:
        if config_path.startswith("configs/") or config_path.startswith("configs."):
            # Import path - try importing as module first
            module_path = config_path.replace(".py", "").replace("/", ".")

            try:
                # First, try to import the full path as a module
                module = importlib.import_module(module_path)
                config = getattr(module, "config")
            except (ModuleNotFoundError, ImportError):
                # If that fails, try treating the last part as a variable name
                parts = module_path.split(".")
                module_name = ".".join(parts[:-1])
                var_name = parts[-1]

                module = importlib.import_module(module_name)
                config = getattr(module, var_name)
        else:
            # File path - exec and get 'config' variable
            config_file = Path(config_path)
            if not config_file.is_absolute():
                # Relative to project root
                config_file = project_root / config_path

            spec = importlib.util.spec_from_file_location("config_module", config_file)
            if spec is None or spec.loader is None:
                raise ValueError(f"Cannot load config from path: {config_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            config = getattr(module, "config")

        # Validate config type
        if not isinstance(config, TrainingConfig):
            raise ValueError(
                f"Config object must be TrainingConfig instance, got {type(config).__name__}"
            )

        return config

    except (ModuleNotFoundError, ImportError) as e:
        raise ValueError(f"Failed to import config module '{config_path}': {e}")
    except AttributeError as e:
        raise ValueError(f"Config file must contain a 'config' variable: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load config from '{config_path}': {e}")


def main() -> None:
    """Main training entry point."""
    # Note: --resume parameter used in Task 5 for checkpoint resumption
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
