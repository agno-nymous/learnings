#!/usr/bin/env python3
"""Training script for Qwen3-VL QLoRA fine-tuning.

Extracted from finetune_qwen3_vl_qlora.ipynb for production use.
"""

# CRITICAL: Import unsloth BEFORE torch/transformers/trl
# Unsloth patches these libraries at import time for optimizations
import argparse
import contextlib
import importlib
import importlib.util
import logging
import sys
import time
from dataclasses import dataclass
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
from transformers import TrainerCallback  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

from configs.base import TrainingConfig  # noqa: E402
from training.checkpoint import CheckpointManager  # noqa: E402
from training.dataset import (  # TODO: uncomment when dependencies installed  # noqa: E402
    load_dataset,
)
from training.eval_utils import log_high_loss_samples  # noqa: E402
from training.preflight import run_preflight_checks  # noqa: E402

logger = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
def load_dataset_with_retry(dataset_name: str, train_subset: int = -1, eval_subset: int = -1):
    """Load dataset with exponential backoff retry.

    Args:
        dataset_name: HuggingFace dataset identifier.
        train_subset: Number of training samples.
        eval_subset: Number of eval samples.

    Returns:
        Tuple of (train_dataset, eval_dataset).
    """
    return load_dataset(dataset_name, train_subset=train_subset, eval_subset=eval_subset)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
def wandb_init_with_retry(**kwargs) -> None:
    """Initialize W&B with retry.

    Args:
        **kwargs: Arguments passed to wandb.init().
    """
    return wandb.init(**kwargs)


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
                config = module.config
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
            config = module.config

        # Validate config type
        if not isinstance(config, TrainingConfig):
            raise ValueError(
                f"Config object must be TrainingConfig instance, got {type(config).__name__}"
            )

        return config

    except (ModuleNotFoundError, ImportError) as e:
        raise ValueError(f"Failed to import config module '{config_path}': {e}") from e
    except AttributeError as e:
        raise ValueError(f"Config file must contain a 'config' variable: {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to load config from '{config_path}': {e}") from e


@dataclass
class CostTracker:
    """Track training costs based on GPU hourly rate."""

    gpu_hourly_rate: float  # e.g., 0.40 for RTX 4090 spot
    start_time: float = 0.0
    total_cost: float = 0.0

    def start(self) -> None:
        """Start tracking training time."""
        self.start_time = time.time()

    def update(self) -> float:
        """Update total cost and return current value.

        Returns:
            Current accumulated cost in USD.
        """
        elapsed_hours = (time.time() - self.start_time) / 3600
        self.total_cost = elapsed_hours * self.gpu_hourly_rate
        return self.total_cost


def get_gpu_rate(config: TrainingConfig) -> float:
    """Get hourly GPU rate from config.

    Args:
        config: Training configuration.

    Returns:
        Hourly rate in USD.
    """
    # Approximate rates for spot instances (USD)
    rates = {
        "RTX_3090": 0.25,
        "RTX_4090": 0.40,
        "A100": 1.50,
        "H100": 2.50,
    }
    return rates.get(config.gpu_type, 0.40)


class CostLoggingCallback(TrainerCallback):
    """Log training cost to W&B during training."""

    def __init__(self, cost_tracker: CostTracker, budget_cap: float):
        """Initialize the cost logging callback.

        Args:
            cost_tracker: CostTracker instance for monitoring compute costs.
            budget_cap: Maximum budget in USD before warnings.
        """
        self.cost_tracker = cost_tracker
        self.budget_cap = budget_cap

    def on_log(self, _args, _state, _control, _logs=None, **_kwargs):
        """Log current cost to W&B."""
        current_cost = self.cost_tracker.update()
        if wandb.run is not None:
            wandb.log({"cost_usd": current_cost})

        # Warn if approaching budget
        if current_cost > self.budget_cap * 0.9:
            logger.warning(f"Approaching budget cap: ${current_cost:.2f} / ${self.budget_cap:.2f}")


class EarlyStoppingCallback(TrainerCallback):
    """Early stopping callback with train loss threshold and eval loss patience.

    Stops training when:
    1. Training loss drops below threshold (default 0.2), OR
    2. Eval loss doesn't improve for N consecutive evaluations (patience)
    """

    def __init__(
        self,
        train_loss_threshold: float = 0.2,
        patience: int = 3,
        min_evals: int = 2,
    ) -> None:
        """Initialize early stopping callback.

        Args:
            train_loss_threshold: Stop training when train_loss drops below this.
            patience: Number of evals to wait without improvement before stopping.
            min_evals: Minimum number of evaluations before early stopping applies.
        """
        self.train_loss_threshold = train_loss_threshold
        self.patience = patience
        self.min_evals = min_evals
        self.best_eval_loss = float("inf")
        self.evals_without_improvement = 0
        self._threshold_reached = False

    def on_log(self, _args, _state, control, logs=None, **_kwargs):
        """Check training loss threshold after each step."""
        if logs is None:
            return

        # Check if train_loss is below threshold
        train_loss = logs.get("loss")
        if (
            train_loss is not None
            and train_loss < self.train_loss_threshold
            and not self._threshold_reached
        ):
            self._threshold_reached = True
            logger.info(
                f"Train loss {train_loss:.4f} below threshold "
                f"{self.train_loss_threshold}. Will run eval and stop."
            )
            control.should_evaluate = True
            control.should_training_stop = True

    def on_evaluate(
        self,
        _args: Any,
        state: Any,
        control: Any,
        metrics: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> None:
        """Check for eval loss improvement and apply patience-based stopping."""
        if metrics is None:
            return

        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return

        # Track best eval loss
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.evals_without_improvement = 0
            logger.info(f"New best eval_loss: {eval_loss:.4f}")
        else:
            self.evals_without_improvement += 1
            logger.info(
                f"Eval loss {eval_loss:.4f} not improved. "
                f"{self.evals_without_improvement}/{self.patience} without improvement"
            )

        # Check if we should stop due to patience (only after min_evals)
        if (
            state.global_step > 0
            and self.evals_without_improvement >= self.patience
            and self.evals_without_improvement >= self.min_evals
        ):
            logger.info(
                f"Early stopping triggered: eval_loss hasn't improved for "
                f"{self.patience} evaluations. Best eval_loss: {self.best_eval_loss:.4f}"
            )
            control.should_training_stop = True


class EvalWithHighLossLoggingCallback(TrainerCallback):
    """Run evaluation with high-loss sample logging to W&B.

    After each evaluation, runs actual model inference on sampled eval data
    and logs predictions vs references to W&B for visual inspection.
    """

    def __init__(self, eval_dataset: Any, processor: Any, top_k: int = 20) -> None:
        """Initialize the callback.

        Args:
            eval_dataset: The evaluation dataset to sample from.
            processor: The model processor for inference.
            top_k: Number of samples to log.
        """
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.top_k = top_k

    def _run_inference(self, model: Any, image: Any) -> str:
        """Run inference on a single image.

        Args:
            model: The model to use for inference.
            image: PIL Image to process.

        Returns:
            Generated text output.
        """
        instruction = """Convert the following document to markdown.
Return only the markdown with no explanation text. Do not include delimiters like ```markdown or ```html.

RULES:
- You must include all information on the page. Do not exclude headers, footers, or subtext.
- Return tables in an HTML format.
- Charts & infographics must be interpreted to a markdown format. Prefer table format when applicable.
- Prefer using ☐ and ☑ for check boxes."""

        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": instruction}],
            }
        ]
        text_prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            image, text_prompt, add_special_tokens=False, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=2048,
                use_cache=True,
                temperature=1.5,
                min_p=0.1,
            )
        return self.processor.tokenizer.decode(output[0], skip_special_tokens=True)

    def on_evaluate(
        self,
        _args: Any,
        state: Any,
        _control: Any,
        _metrics: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """After evaluation, run inference and log samples to W&B."""
        if not wandb.run:
            return

        model = kwargs.get("model")
        if model is None:
            return

        logger.info(f"Running inference on {self.top_k} eval samples for W&B logging...")

        try:
            import random

            # Sample eval set
            sample_size = min(self.top_k, len(self.eval_dataset))
            indices = random.sample(range(len(self.eval_dataset)), sample_size)
            samples = [self.eval_dataset[i] for i in indices]

            # Switch to inference mode
            FastVisionModel.for_inference(model)

            predictions = []
            references = []
            losses = []

            for i, sample in enumerate(samples):
                img = sample["images"][0]
                # Extract reference from messages (assistant content)
                ref = ""
                for msg in sample.get("messages", []):
                    if msg.get("role") == "assistant":
                        for content in msg.get("content", []):
                            if content.get("type") == "text":
                                ref = content.get("text", "")
                                break

                # Run actual inference
                pred = self._run_inference(model, img)
                predictions.append(pred)
                references.append(ref)
                losses.append(1.0)  # Placeholder loss (we don't track per-sample loss)

                if (i + 1) % 5 == 0:
                    logger.info(f"  Processed {i + 1}/{sample_size} samples")

            # Switch back to training mode
            FastVisionModel.for_training(model)

            log_high_loss_samples(
                samples=samples,
                predictions=predictions,
                references=references,
                losses=losses,
                top_k=self.top_k,
                table_name=f"high_loss_samples_step_{state.global_step}",
            )
            logger.info(f"Logged {sample_size} samples with predictions to W&B")

        except Exception as e:
            logger.warning(f"Failed to log high-loss samples: {e}")
            # Ensure we return to training mode even on error
            with contextlib.suppress(Exception):
                FastVisionModel.for_training(model)


def setup_model_and_tokenizer(config: TrainingConfig) -> tuple[Any, Any]:
    """Load model and tokenizer with QLoRA.

    Args:
        config: Training configuration.

    Returns:
        Tuple of (model, tokenizer).
    """
    from transformers import AutoModelForCausalLM

    # PaddleOCR-VL requires special loading parameters
    is_paddleocr = "PaddleOCR" in config.model_name

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

    model = FastVisionModel.get_peft_model(
        model,
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        random_state=config.seed,
        use_rslora=False,
        # PaddleOCR uses explicit target_modules instead of finetune_* params
        target_modules=(
            [
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
            if is_paddleocr
            else None
        ),
        finetune_vision_layers=config.finetune_vision_layers if not is_paddleocr else None,
        finetune_language_layers=config.finetune_language_layers if not is_paddleocr else None,
        finetune_attention_modules=config.finetune_attention_modules if not is_paddleocr else None,
        finetune_mlp_modules=config.finetune_mlp_modules if not is_paddleocr else None,
    )

    # CRITICAL: Mark PaddleOCR-VL as vision model for SFTTrainer detection
    # Unsloth's SFTTrainer checks for vision model attributes when dataset has images
    # After PEFT wrapping, we need to explicitly set these attributes
    if is_paddleocr:
        model.is_vision_model = True
        if hasattr(model, "model"):
            model.model.is_vision_model = True

    # For PaddleOCR, also load the processor
    if is_paddleocr:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            config.model_name, trust_remote_code=True
        )  # nosec: B615
        return model, tokenizer, processor

    return model, tokenizer, None


def setup_wandb(config: TrainingConfig) -> None:
    """Initialize Weights & Biases logging with retry.

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
        SFTTrainer instance.
    """
    FastVisionModel.for_training(model)

    # Use processor-based collator for PaddleOCR
    is_paddleocr = "PaddleOCR" in config.model_name
    if is_paddleocr and processor is not None:
        data_collator = UnslothVisionDataCollator(
            model=model,
            processor=processor,
            ignore_index=-100,
            max_seq_length=4096,  # Per Colab notebook for PaddleOCR
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


class CheckpointCallback(TrainerCallback):
    """Callback to manage checkpoint retention during training.

    Attributes:
        checkpoint_manager: Manages checkpoint cleanup with retention policy.
        checkpoints: Dict mapping checkpoint names to eval_loss values.
    """

    def __init__(self, checkpoint_manager: CheckpointManager) -> None:
        """Initialize the callback.

        Args:
            checkpoint_manager: CheckpointManager instance for cleanup.
        """
        self.checkpoint_manager = checkpoint_manager
        self.checkpoints: dict[str, float] = {}

    def on_evaluate(
        self,
        _args: Any,
        state: Any,
        _control: Any,
        metrics: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> None:
        """After evaluation, update checkpoint retention.

        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control.
            metrics: Evaluation metrics including eval_loss.
            **kwargs: Additional keyword arguments.
        """
        # Update checkpoints dict with current eval loss
        checkpoint_name = f"checkpoint-{state.global_step}"
        if metrics and "eval_loss" in metrics:
            self.checkpoints[checkpoint_name] = metrics["eval_loss"]

        # Run cleanup
        self.checkpoint_manager.cleanup(self.checkpoints)


def main() -> None:
    """Run the training pipeline."""
    args = parse_args()
    config = load_config(args.config)

    print("=== Training Configuration ===")
    print(f"Experiment: {config.experiment_name}")
    print(f"Model: {config.model_name}")
    print(f"Epochs: {config.num_train_epochs}")
    effective_batch = config.per_device_train_batch_size * config.gradient_accumulation_steps
    print(
        f"Batch Size: {config.per_device_train_batch_size} × {config.gradient_accumulation_steps} = {effective_batch} effective"
    )
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Dataset: {config.dataset_name}")
    print(f"W&B Project: {config.wandb_project}")
    print(f"Output Dir: {config.output_dir}")
    print("=============================\n")

    # Run pre-flight checks
    run_preflight_checks(config)

    # Setup W&B
    setup_wandb(config)

    try:
        # Load model and tokenizer
        print("Loading model...")
        model, tokenizer, processor = setup_model_and_tokenizer(config)

        # Load datasets with retry
        print("Loading datasets...")
        train_dataset, eval_dataset = load_dataset_with_retry(
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
            processor=processor,
        )

        # Create checkpoint manager
        checkpoint_mgr = CheckpointManager(
            Path(config.output_dir),
            keep_best=5,
            keep_recent=5,
        )

        # Add checkpoint callback
        trainer.add_callback(CheckpointCallback(checkpoint_mgr))

        # Add early stopping callback
        trainer.add_callback(
            EarlyStoppingCallback(
                train_loss_threshold=config.train_loss_threshold,
                patience=config.early_stopping_patience,
                min_evals=config.early_stopping_min_evals,
            )
        )

        # Add high-loss logging callback (only if W&B is enabled)
        if config.report_to_wandb:
            trainer.add_callback(
                EvalWithHighLossLoggingCallback(
                    eval_dataset=eval_dataset,
                    processor=processor,
                    top_k=5,
                )
            )

        # Setup cost tracking
        cost_tracker = CostTracker(gpu_hourly_rate=get_gpu_rate(config))
        cost_tracker.start()

        # Add cost logging callback
        trainer.add_callback(CostLoggingCallback(cost_tracker, config.budget_cap))

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
            round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3) if gpu_stats else 0
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

    finally:
        # Always cleanup W&B, even if training fails
        if config.report_to_wandb:
            wandb.finish()

    print("Training complete!")


if __name__ == "__main__":
    main()
