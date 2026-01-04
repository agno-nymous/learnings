#!/usr/bin/env python3
"""Training script for Qwen3-VL QLoRA fine-tuning.

Extracted from finetune_qwen3_vl_qlora.ipynb for production use.
"""

import argparse
import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import wandb
from configs.base import TrainingConfig
# from training.dataset import load_dataset  # TODO: uncomment when dependencies installed
from trl import SFTConfig, SFTTrainer
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator

logger = logging.getLogger(__name__)


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


def setup_model_and_tokenizer(config: TrainingConfig):
    """Load model and tokenizer with QLoRA.

    Args:
        config: Training configuration.

    Returns:
        Tuple of (model, tokenizer).
    """
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=config.model_name,
        load_in_4bit=config.load_in_4bit,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=config.finetune_vision_layers,
        finetune_language_layers=config.finetune_language_layers,
        finetune_attention_modules=config.finetune_attention_modules,
        finetune_mlp_modules=config.finetune_mlp_modules,
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        random_state=config.seed,
        use_rslora=False,
        loftq_config=None,
    )

    return model, tokenizer


def setup_wandb(config: TrainingConfig):
    """Initialize Weights & Biases logging.

    Args:
        config: Training configuration.
    """
    if config.report_to_wandb:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.experiment_name,
            config=config.__dict__,
        )


def create_trainer(
    config: TrainingConfig,
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    resume_from_checkpoint: Optional[str] = None,
):
    """Create SFTTrainer with configuration.

    Args:
        config: Training configuration.
        model: LoRA model.
        tokenizer: Tokenizer.
        train_dataset: Training dataset.
        eval_dataset: Evaluation dataset.
        resume_from_checkpoint: Optional checkpoint path to resume from.

    Returns:
        SFTTrainer instance.
    """
    FastVisionModel.for_training(model)

    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            learning_rate=config.learning_rate,
            fp16=config.fp16,
            bf16=config.bf16,
            logging_steps=config.logging_steps,
            eval_strategy="steps",
            eval_steps=config.eval_steps,
            optim=config.optim,
            weight_decay=config.weight_decay,
            lr_scheduler_type=config.lr_scheduler_type,
            seed=config.seed,
            output_dir=config.output_dir,
            report_to="wandb" if config.report_to_wandb else "none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=config.dataset_num_proc,
            max_seq_length=config.max_seq_length,
        ),
    )


def main() -> None:
    """Main training entry point."""
    args = parse_args()
    config = load_config(args.config)

    print(f"=== Training Configuration ===")
    print(f"Experiment: {config.experiment_name}")
    print(f"Model: {config.model_name}")
    print(f"Max Steps: {config.max_steps}")
    print(
        f"Batch Size: {config.per_device_train_batch_size} × {config.gradient_accumulation_steps} = {config.per_device_train_batch_size * config.gradient_accumulation_steps} effective"
    )
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Dataset: {config.dataset_name}")
    print(f"W&B Project: {config.wandb_project}")
    print(f"Output Dir: {config.output_dir}")
    print(f"=============================\n")

    # Setup W&B
    setup_wandb(config)

    # Load model and tokenizer
    print("Loading model...")
    model, tokenizer = setup_model_and_tokenizer(config)

    # Load datasets
    print("Loading datasets...")
    train_dataset, eval_dataset = load_dataset(
        config.dataset_name,
        train_subset=config.train_subset,
        eval_subset=config.eval_subset,
    )
    print(f"Train: {len(train_dataset)} samples")
    print(f"Eval: {len(eval_dataset)} samples")

    # Create trainer
    print("Creating trainer...")
    trainer = create_trainer(
        config,
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        resume_from_checkpoint=args.resume,
    )

    # Train
    print("Starting training...")
    gpu_stats = torch.cuda.is_available()
    if gpu_stats:
        gpu_props = torch.cuda.get_device_properties(0)
        max_memory = round(gpu_props.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU: {gpu_props.name}, Max Memory: {max_memory} GB")
    else:
        print("No GPU detected")

    trainer_stats = trainer.train(resume_from_checkpoint=args.resume)

    used_memory = (
        round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        if gpu_stats
        else 0
    )
    print(f"\nTraining Time: {round(trainer_stats.metrics['train_runtime'] / 60, 2)} min")
    if used_memory > 0:
        print(f"Peak Memory: {used_memory} GB ({round(used_memory / max_memory * 100, 1)}%)")

    # Save final model
    print("\nSaving model...")
    output_path = f"{config.output_dir}/{config.experiment_name}"
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Model saved to {output_path}")

    if config.report_to_wandb:
        wandb.finish()

    print("Training complete!")


if __name__ == "__main__":
    main()
