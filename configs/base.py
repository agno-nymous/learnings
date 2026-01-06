from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Base configuration for Qwen3-VL QLoRA training."""

    # === Model ===
    model_name: str = "unsloth/PaddleOCR-VL"  # PaddleOCR-VL for OCR tasks
    load_in_4bit: bool = True
    use_gradient_checkpointing: str = "unsloth"

    # === LoRA ===
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0
    finetune_vision_layers: bool = True
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True

    # === Training ===
    max_steps: int = 500
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5

    # === Optimizer ===
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"

    # === Precision ===
    fp16: bool = False
    bf16: bool = True

    # === Data ===
    dataset_name: str = "wrath/well-log-headers-ocr"
    train_subset: int = -1  # -1 = full dataset
    eval_subset: int = -1

    # === Evaluation ===
    eval_steps: int = 50
    logging_steps: int = 1

    # === Early Stopping ===
    train_loss_threshold: float = 0.2  # Stop training when train_loss drops below this
    early_stopping_patience: int = 3  # Stop after N evals without eval_loss improvement
    early_stopping_min_evals: int = 2  # Minimum evals before early stopping applies

    # === Output ===
    output_dir: str = "./checkpoints"  # Use relative path; mount network volume here if needed
    experiment_name: str | None = None  # Auto-generated if None

    # === Hardware/RunPod ===
    gpu_type: str = "RTX_4090"
    spot_instance: bool = True
    budget_cap: float = 10.0

    # === Monitoring ===
    wandb_project: str = "well-log-ocr"
    wandb_entity: str | None = None
    report_to_wandb: bool = True

    # === Misc ===
    seed: int = 3407
    max_seq_length: int = 2048
    dataset_num_proc: int = 4

    def __post_init__(self):
        """Auto-generate experiment name if not provided."""
        if self.experiment_name is None:
            self.experiment_name = self._generate_experiment_name()

    def _generate_experiment_name(self) -> str:
        """Generate unique experiment name from config."""
        from datetime import datetime

        model_short = self.model_name.split("/")[1].split("-")[0].lower()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{model_short}-qlora-r{self.r}-lr{self.learning_rate:.0e}-steps{self.max_steps}-{timestamp}"
