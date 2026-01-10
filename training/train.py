#!/usr/bin/env python3
"""Training script for Qwen3-VL QLoRA fine-tuning.

This script orchestrates the training pipeline:
1. Load configuration from Python file
2. Setup model with QLoRA adapters
3. Load and prepare datasets
4. Train with callbacks for checkpointing, early stopping, and cost tracking
5. Save final model

Usage:
    python training/train.py --config configs/experiments/paddleocr_vl.py
    python training/train.py --config configs/experiments/paddleocr_vl.py --resume checkpoints/checkpoint-100
"""

# CRITICAL: Import unsloth BEFORE torch/transformers/trl
# Unsloth patches these libraries at import time for optimizations
import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from unsloth import FastVisionModel  # isort: skip
from unsloth.trainer import UnslothVisionDataCollator  # isort: skip

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch  # noqa: E402
import wandb  # noqa: E402
from tenacity import retry, stop_after_attempt, wait_exponential  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

from configs.base import TrainingConfig  # noqa: E402
from configs.loader import load_config  # noqa: E402
from training.callbacks import (  # noqa: E402
    CheckpointCallback,
    CostLoggingCallback,
    EarlyStoppingCallback,
    EvalWithHighLossLoggingCallback,
)
from training.checkpoint import CheckpointManager  # noqa: E402
from training.cost_tracker import CostTracker, get_gpu_rate  # noqa: E402
from training.dataset import load_dataset  # noqa: E402
from training.preflight import run_preflight_checks  # noqa: E402

logger = logging.getLogger(__name__)


# ===================================================================
# Retry Wrappers
# ===================================================================


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
def load_dataset_with_retry(
    dataset_name: str, train_subset: int = -1, eval_subset: int = -1
) -> tuple[Any, Any]:
    """Load dataset with exponential backoff retry.

    Args:
        dataset_name: HuggingFace dataset identifier.
        train_subset: Number of training samples (-1 for all).
        eval_subset: Number of eval samples (-1 for all).

    Returns:
        Tuple of (train_dataset, eval_dataset).
    """
    return load_dataset(dataset_name, train_subset=train_subset, eval_subset=eval_subset)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
def wandb_init_with_retry(**kwargs) -> Any:
    """Initialize W&B with retry.

    Args:
        **kwargs: Arguments passed to wandb.init().

    Returns:
        wandb.Run instance.
    """
    return wandb.init(**kwargs)


# ===================================================================
# CLI
# ===================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments with config path and optional resume checkpoint.
    """
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


# ===================================================================
# Model Setup
# ===================================================================


def setup_model_and_tokenizer(config: TrainingConfig) -> tuple[Any, Any, Any | None]:
    """Load model and tokenizer with QLoRA adapters.

    Args:
        config: Training configuration.

    Returns:
        Tuple of (model, tokenizer, processor). Processor is only returned for
        PaddleOCR models, None otherwise.
    """
    from transformers import AutoModelForCausalLM

    is_paddleocr = "PaddleOCR" in config.model_name

    # Load base model
    if is_paddleocr:
        model, tokenizer = FastVisionModel.from_pretrained(
            config.model_name,
            max_seq_length=config.max_seq_length,
            load_in_4bit=False,  # PaddleOCR doesn't support 4bit
            load_in_8bit=False,
            full_finetuning=True,
            auto_model=AutoModelForCausalLM,
            trust_remote_code=True,
            use_gradient_checkpointing=config.use_gradient_checkpointing,
        )
    else:
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=config.model_name,
            load_in_4bit=config.load_in_4bit,
            use_gradient_checkpointing=config.use_gradient_checkpointing,
        )

    # Apply LoRA adapters
    model = FastVisionModel.get_peft_model(
        model,
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        random_state=config.seed,
        use_rslora=False,
        target_modules=(_get_paddleocr_target_modules() if is_paddleocr else None),
        finetune_vision_layers=config.finetune_vision_layers if not is_paddleocr else None,
        finetune_language_layers=config.finetune_language_layers if not is_paddleocr else None,
        finetune_attention_modules=config.finetune_attention_modules if not is_paddleocr else None,
        finetune_mlp_modules=config.finetune_mlp_modules if not is_paddleocr else None,
    )

    # Mark PaddleOCR-VL as vision model for SFTTrainer detection
    if is_paddleocr:
        model.is_vision_model = True
        if hasattr(model, "model"):
            model.model.is_vision_model = True

    # Load processor for PaddleOCR
    processor = None
    if is_paddleocr:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(config.model_name, trust_remote_code=True)

    return model, tokenizer, processor


def _get_paddleocr_target_modules() -> list[str]:
    """Get LoRA target modules for PaddleOCR model.

    Returns:
        List of module names to apply LoRA to.
    """
    return [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "out_proj",
        "fc1",
        "fc2",
        "linear_1",
        "linear_2",
    ]


# ===================================================================
# Trainer Setup
# ===================================================================


def setup_wandb(config: TrainingConfig) -> None:
    """Initialize Weights & Biases logging.

    Args:
        config: Training configuration.
    """
    if config.report_to_wandb:
        wandb_init_with_retry(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.experiment_name,
            config=config.__dict__,
        )


def create_trainer(
    config: TrainingConfig,
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any,
    processor: Any = None,
) -> SFTTrainer:
    """Create SFTTrainer with configuration.

    Args:
        config: Training configuration.
        model: LoRA model.
        tokenizer: Tokenizer.
        train_dataset: Training dataset.
        eval_dataset: Evaluation dataset.
        processor: Optional processor for vision models.

    Returns:
        Configured SFTTrainer instance.
    """
    FastVisionModel.for_training(model)

    # Configure data collator based on model type
    is_paddleocr = "PaddleOCR" in config.model_name
    if is_paddleocr and processor is not None:
        data_collator = UnslothVisionDataCollator(
            model=model,
            processor=processor,
            ignore_index=-100,
            max_seq_length=4096,
            train_on_responses_only=True,
            instruction_part="User: ",
            response_part="\nAssistant:",
            pad_to_multiple_of=8,
        )
    else:
        data_collator = UnslothVisionDataCollator(model, tokenizer)

    return SFTTrainer(
        model=model,
        tokenizer=processor.tokenizer if processor else tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=config.warmup_steps,
            num_train_epochs=config.num_train_epochs,
            learning_rate=config.learning_rate,
            fp16=config.fp16,
            bf16=config.bf16,
            logging_steps=config.logging_steps,
            eval_strategy="epoch",
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
        ),
    )


def add_callbacks(
    trainer: SFTTrainer,
    config: TrainingConfig,
    eval_dataset: Any,
    processor: Any,
) -> None:
    """Add all training callbacks to the trainer.

    Args:
        trainer: SFTTrainer instance.
        config: Training configuration.
        eval_dataset: Evaluation dataset for logging.
        processor: Model processor for inference in logging.
    """
    # Checkpoint management
    checkpoint_mgr = CheckpointManager(
        Path(config.output_dir),
        keep_best=5,
        keep_recent=5,
    )
    trainer.add_callback(CheckpointCallback(checkpoint_mgr))

    # Early stopping
    trainer.add_callback(
        EarlyStoppingCallback(
            train_loss_threshold=config.train_loss_threshold,
            patience=config.early_stopping_patience,
            min_evals=config.early_stopping_min_evals,
        )
    )

    # Cost tracking
    cost_tracker = CostTracker(gpu_hourly_rate=get_gpu_rate(config.gpu_type))
    cost_tracker.start()
    trainer.add_callback(CostLoggingCallback(cost_tracker, config.budget_cap))

    # Evaluation logging (only with W&B)
    if config.report_to_wandb:
        trainer.add_callback(
            EvalWithHighLossLoggingCallback(
                eval_dataset=eval_dataset,
                processor=processor,
                top_k=5,
            )
        )


# ===================================================================
# Main
# ===================================================================


def print_config_summary(config: TrainingConfig) -> None:
    """Print training configuration summary.

    Args:
        config: Training configuration to display.
    """
    effective_batch = config.per_device_train_batch_size * config.gradient_accumulation_steps
    print("=== Training Configuration ===")
    print(f"Experiment: {config.experiment_name}")
    print(f"Model: {config.model_name}")
    print(f"Epochs: {config.num_train_epochs}")
    print(
        f"Batch Size: {config.per_device_train_batch_size} x "
        f"{config.gradient_accumulation_steps} = {effective_batch} effective"
    )
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Dataset: {config.dataset_name}")
    print(f"W&B Project: {config.wandb_project}")
    print(f"Output Dir: {config.output_dir}")
    print("=============================\n")


def print_gpu_info() -> tuple[bool, float]:
    """Print GPU information and return availability and max memory.

    Returns:
        Tuple of (has_gpu, max_memory_gb).
    """
    has_gpu = torch.cuda.is_available()
    max_memory = 0.0

    if has_gpu:
        gpu_props = torch.cuda.get_device_properties(0)
        max_memory = round(gpu_props.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU: {gpu_props.name}, Max Memory: {max_memory} GB")
    else:
        print("No GPU detected")

    return has_gpu, max_memory


def main() -> None:
    """Run the training pipeline."""
    args = parse_args()
    config = load_config(args.config, project_root)

    print_config_summary(config)
    run_preflight_checks(config)
    setup_wandb(config)

    try:
        # Load model
        print("Loading model...")
        model, tokenizer, processor = setup_model_and_tokenizer(config)

        # Load datasets
        print("Loading datasets...")
        train_dataset, eval_dataset = load_dataset_with_retry(
            config.dataset_name,
            train_subset=config.train_subset,
            eval_subset=config.eval_subset,
        )
        print(f"Train: {len(train_dataset)} samples")
        print(f"Eval: {len(eval_dataset)} samples")

        # Create trainer with callbacks
        print("Creating trainer...")
        trainer = create_trainer(config, model, tokenizer, train_dataset, eval_dataset, processor)
        add_callbacks(trainer, config, eval_dataset, processor)

        # Train
        print("Starting training...")
        has_gpu, max_memory = print_gpu_info()

        trainer_stats = trainer.train(resume_from_checkpoint=args.resume)

        # Report results
        if has_gpu:
            used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            print(f"\nTraining Time: {round(trainer_stats.metrics['train_runtime'] / 60, 2)} min")
            print(f"Peak Memory: {used_memory} GB ({round(used_memory / max_memory * 100, 1)}%)")

        # Save final model
        print("\nSaving model...")
        output_path = f"{config.output_dir}/{config.experiment_name}"
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        print(f"Model saved to {output_path}")

    finally:
        if config.report_to_wandb:
            wandb.finish()

    print("Training complete!")


if __name__ == "__main__":
    main()
