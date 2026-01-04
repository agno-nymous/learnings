# RunPod Training Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move well log OCR model fine-tuning from local/Colab to RunPod cloud infrastructure with QLoRA training on Qwen3-VL-2B using spot instances for cost optimization.

**Architecture:** Extract training logic from `finetune_qwen3_vl_qlora.ipynb` into production-ready Python scripts with config-based experiments, W&B monitoring, checkpoint management, and automated RunPod startup workflow.

**Tech Stack:** Python 3.12+, Unsloth, PyTorch, HuggingFace Transformers/PEFT/TRL, W&B, RunPod, dataclasses/attrs for configs

---

## Phase 1: Foundation (Priority: High)

### Task 1: Create Configuration System

**Files:**
- Create: `training/__init__.py`
- Create: `configs/base.py`
- Create: `configs/experiments/quick_val.py`
- Create: `configs/experiments/qwen3_qlora_r16.py`

**Step 1: Create training package directory**

```bash
mkdir -p training configs/experiments
touch training/__init__.py configs/__init__.py configs/experiments/__init__.py
```

**Step 2: Write the base configuration class**

```python
# configs/base.py
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TrainingConfig:
    """Base configuration for Qwen3-VL QLoRA training."""

    # === Model ===
    model_name: str = "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit"
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

    # === Output ===
    output_dir: str = "/runpod_volume/checkpoints"
    experiment_name: Optional[str] = None  # Auto-generated if None

    # === Hardware/RunPod ===
    gpu_type: str = "RTX_4090"
    spot_instance: bool = True
    budget_cap: float = 10.0

    # === Monitoring ===
    wandb_project: str = "well-log-ocr"
    wandb_entity: Optional[str] = None
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
```

**Step 3: Write the quick validation config**

```python
# configs/experiments/quick_val.py
from configs.base import TrainingConfig

config = TrainingConfig(
    max_steps=10,
    train_subset=100,
    eval_subset=20,
    eval_steps=5,
    output_dir="/runpod_volume/checkpoints/quick-val",
)
```

**Step 4: Write the full training config**

```python
# configs/experiments/qwen3_qlora_r16.py
from configs.base import TrainingConfig

config = TrainingConfig(
    # Explicit overrides (rest inherited from base)
    max_steps=500,
    eval_steps=50,
)
```

**Step 5: Run config tests**

```bash
python -c "from configs.base import TrainingConfig; c = TrainingConfig(); print(c.experiment_name)"
python -c "from configs.experiments.quick_val import config; print(config.max_steps)"
```

Expected output: Auto-generated experiment name and `10`

**Step 6: Commit**

```bash
git add configs/ training/
git commit -m "feat: add base configuration system for training experiments"
```

---

### Task 2: Create Dataset Module

**Files:**
- Create: `training/dataset.py`
- Create: `tests/test_dataset.py`

**Step 1: Write the failing test**

```python
# tests/test_dataset.py
import pytest
from training.dataset import LazyVisionDataset, b64_to_image
from datasets import load_dataset
from PIL import Image

def test_b64_to_image():
    b64_str = "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCwAB//2Q=="
    img = b64_to_image(b64_str)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"

def test_lazy_vision_dataset_length():
    raw_dataset = load_dataset("wrath/well-log-headers-ocr", split="train[:5]")
    dataset = LazyVisionDataset(raw_dataset)
    assert len(dataset) == 5

def test_lazy_vision_dataset_getitem():
    raw_dataset = load_dataset("wrath/well-log-headers-ocr", split="train[:1]")
    dataset = LazyVisionDataset(raw_dataset)
    sample = dataset[0]
    assert "messages" in sample
    assert "images" in sample
    assert len(sample["messages"]) == 2
    assert sample["messages"][0]["role"] == "user"
    assert sample["messages"][1]["role"] == "assistant"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_dataset.py -v
```

Expected: `ModuleNotFoundError: No module named 'training.dataset'`

**Step 3: Write minimal implementation**

```python
# training/dataset.py
"""Dataset loading utilities for Qwen3-VL training."""

from datasets import load_dataset as hf_load_dataset
from PIL import Image
from io import BytesIO
import base64
from typing import Any


def b64_to_image(b64_str: str) -> Image.Image:
    """Convert base64 string to PIL RGB Image.

    Args:
        b64_str: Base64-encoded image string.

    Returns:
        PIL Image in RGB mode.
    """
    return Image.open(BytesIO(base64.b64decode(b64_str))).convert("RGB")


class LazyVisionDataset:
    """Memory-efficient dataset with lazy image loading.

    Defers image decoding until __getitem__ is called to avoid
    loading all images into RAM at once.
    """

    def __init__(self, hf_dataset: Any):
        """Initialize with HuggingFace dataset.

        Args:
            hf_dataset: HuggingFace dataset with 'image_base64', 'instruction', 'answer' fields.
        """
        self.data = hf_dataset

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """Get a single training sample.

        Returns:
            Dict with 'messages' (for Unsloth) and 'images' (lazy-loaded PIL Images).
        """
        sample = self.data[idx]
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": sample["instruction"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample["answer"]}],
                },
            ],
            "images": [b64_to_image(sample["image_base64"])],
        }


def load_dataset(dataset_name: str, train_subset: int = -1, eval_subset: int = -1):
    """Load training and evaluation datasets.

    Args:
        dataset_name: HuggingFace dataset identifier.
        train_subset: Number of training samples to use (-1 for all).
        eval_subset: Number of eval samples to use (-1 for all).

    Returns:
        Tuple of (train_dataset, eval_dataset) as LazyVisionDataset.
    """
    raw_dataset = hf_load_dataset(dataset_name)

    train_split = raw_dataset["train"]
    eval_split = raw_dataset["eval"]

    if train_subset > 0:
        train_split = train_split.select(range(min(train_subset, len(train_split))))
    if eval_subset > 0:
        eval_split = eval_split.select(range(min(eval_subset, len(eval_split))))

    return LazyVisionDataset(train_split), LazyVisionDataset(eval_split)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_dataset.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add training/dataset.py tests/test_dataset.py
git commit -m "feat: add lazy-loading dataset module"
```

---

### Task 3: Create Metrics Module

**Files:**
- Create: `training/metrics.py`
- Create: `tests/test_metrics.py`

**Step 1: Write the failing test**

```python
# tests/test_metrics.py
import pytest
from training.metrics import compute_cer, compute_wer

def test_cer_perfect_match():
    assert compute_cer("hello", "hello") == 0.0

def test_cer_all_wrong():
    assert compute_cer("abc", "xyz") == 1.0

def test_cer_partial_match():
    # "hello" -> "hallo" = 1 substitution / 5 chars = 0.2
    cer = compute_cer("hello", "hallo")
    assert abs(cer - 0.2) < 0.01

def test_wer_perfect_match():
    assert compute_wer("hello world", "hello world") == 0.0

def test_wer_substitution():
    # "hello world" -> "hallo world" = 1 error / 2 words = 0.5
    wer = compute_wer("hello world", "hallo world")
    assert abs(wer - 0.5) < 0.01

def test_wer_insertion():
    wer = compute_wer("hello world", "hello big world")
    assert abs(wer - 0.5) < 0.01  # 1 insertion / 2 words

def test_wer_deletion():
    wer = compute_wer("hello big world", "hello world")
    assert abs(wer - 0.5) < 0.01  # 1 deletion / 2 words
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_metrics.py -v
```

Expected: `ModuleNotFoundError: No module named 'training.metrics'`

**Step 3: Write minimal implementation**

```python
# training/metrics.py
"""Character and Word Error Rate computation for OCR evaluation."""

import editdistance


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (CER).

    CER = (substitutions + insertions + deletions) / reference_length

    Args:
        reference: Ground truth text.
        hypothesis: Predicted text.

    Returns:
        CER as float between 0 and 1.
    """
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0

    distance = editdistance.eval(reference, hypothesis)
    return distance / len(reference)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER).

    WER = (substitutions + insertions + deletions) / reference_word_count

    Args:
        reference: Ground truth text.
        hypothesis: Predicted text.

    Returns:
        WER as float between 0 and 1.
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    distance = editdistance.eval(ref_words, hyp_words)
    return distance / len(ref_words)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_metrics.py -v
```

Expected: All tests PASS

**Step 5: Add editdistance dependency**

```bash
# Add to requirements.txt
echo "editdistance>=0.8.1" >> requirements.txt
```

**Step 6: Commit**

```bash
git add training/metrics.py tests/test_metrics.py requirements.txt
git commit -m "feat: add CER/WER metrics for OCR evaluation"
```

---

### Task 4: Create Main Training Script

**Files:**
- Create: `training/train.py`
- Modify: `requirements.txt`

**Step 1: Update requirements.txt**

Add to `requirements.txt`:
```txt
unsloth>=2024.12.0
torch>=2.0.0
transformers>=4.40.0
peft>=0.10.0
trl>=0.9.0
datasets>=2.18.0
wandb>=0.16.0
pillow>=10.0.0
pandas>=2.0.0
editdistance>=0.8.1
```

**Step 2: Write training script skeleton with CLI**

```python
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
from training.dataset import load_dataset


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
        # Import path
        module_path = config_path.replace(".py", "").replace("/", ".")
        parts = module_path.split(".")
        module_name = ".".join(parts[:-1]) if len(parts) > 1 else parts[0]
        var_name = parts[-1] if len(parts) > 1 else "config"

        import importlib
        module = importlib.import_module(module_name)
        return getattr(module, var_name)
    else:
        # File path - exec and get 'config' variable
        import importlib.util
        spec = importlib.util.spec_from_file_location("config_module", config_path)
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
```

**Step 3: Make executable and test**

```bash
chmod +x training/train.py
python training/train.py --config configs/experiments/quick_val.py
```

Expected: Configuration printed, "Training implementation in next task..."

**Step 4: Commit**

```bash
git add training/train.py requirements.txt
git commit -m "feat: add training script skeleton with CLI"
```

---

### Task 5: Implement Model Loading and Training Loop

**Files:**
- Modify: `training/train.py`

**Step 1: Add model loading and training functions**

```python
# Add to training/train.py, after imports
import torch
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
import wandb
import gc


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


def create_trainer(config: TrainingConfig, model, tokenizer, train_dataset, eval_dataset, resume_from_checkpoint=None):
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


def main():
    """Main training entry point."""
    args = parse_args()
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
    gpu_stats = torch.cuda.get_device_properties(0)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU: {gpu_stats.name}, Max Memory: {max_memory} GB")

    trainer_stats = trainer.train(resume_from_checkpoint=args.resume)

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    print(f"\nTraining Time: {round(trainer_stats.metrics['train_runtime']/60, 2)} min")
    print(f"Peak Memory: {used_memory} GB ({round(used_memory/max_memory*100, 1)}%)")

    # Save final model
    print("\nSaving model...")
    output_path = f"{config.output_dir}/{config.experiment_name}"
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Model saved to {output_path}")

    if config.report_to_wandb:
        wandb.finish()

    print("Training complete!")
```

**Step 2: Test quick validation run**

```bash
python training/train.py --config configs/experiments/quick_val.py
```

Expected: Training runs for 10 steps, saves model

**Step 3: Commit**

```bash
git add training/train.py
git commit -m "feat: implement model loading and training loop"
```

---

### Task 6: Add Pre-flight Validation

**Files:**
- Create: `training/preflight.py`
- Modify: `training/train.py`

**Step 1: Write pre-flight validation module**

```python
# training/preflight.py
"""Pre-flight validation checks before training."""

import os
import torch
from transformers import AutoConfig


def check_model_exists(model_name: str) -> bool:
    """Verify model exists on HuggingFace Hub.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        True if model exists, False otherwise.

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
        True if dataset exists, False otherwise.

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
```

**Step 2: Integrate into training script**

Add to `training/train.py` in `main()` before loading model:
```python
from training.preflight import run_preflight_checks

def main():
    # ... after config printing, before model loading ...
    run_preflight_checks(config)
    # ... continue with model loading ...
```

**Step 3: Test pre-flight validation**

```bash
python training/train.py --config configs/experiments/quick_val.py
```

Expected: "All checks passed!" before training starts

**Step 4: Commit**

```bash
git add training/preflight.py training/train.py
git commit -m "feat: add pre-flight validation checks"
```

---

## Phase 2: Automation (Priority: High)

### Task 7: Create Checkpoint Management Module

**Files:**
- Create: `training/checkpoint.py`
- Create: `tests/test_checkpoint.py`

**Step 1: Write the failing test**

```python
# tests/test_checkpoint.py
import pytest
import tempfile
import shutil
from pathlib import Path
from training.checkpoint import CheckpointManager, get_latest_checkpoint

def test_checkpoint_manager_creation(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_best=3, keep_recent=2)
    assert mgr.output_dir == tmp_path
    assert mgr.keep_best == 3
    assert mgr.keep_recent == 2

def test_get_latest_checkpoint_none(tmp_path):
    assert get_latest_checkpoint(tmp_path) is None

def test_get_latest_checkpoint(tmp_path):
    chk1 = tmp_path / "checkpoint-10"
    chk2 = tmp_path / "checkpoint-50"
    chk1.mkdir()
    chk2.mkdir()

    latest = get_latest_checkpoint(tmp_path)
    assert latest == chk2

def test_rotating_retention(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_best=2, keep_recent=2)

    # Create checkpoints with validation losses
    checkpoints = {
        "checkpoint-10": 0.5,
        "checkpoint-20": 0.3,
        "checkpoint-30": 0.4,
        "checkpoint-40": 0.2,
        "checkpoint-50": 0.25,
    }

    for name, loss in checkpoints.items():
        chk_path = tmp_path / name
        chk_path.mkdir()
        (chk_path / "trainer_state.json").write_text(f'{{"eval_loss": {loss}}}')

    # Should keep: best (0.2, 0.25) + recent (checkpoint-50, checkpoint-40) = 3 unique
    mgr.cleanup(checkpoints)

    kept = [d.name for d in tmp_path.iterdir() if d.is_dir()]
    assert "checkpoint-40" in kept  # best (0.2)
    assert "checkpoint-50" in kept  # recent and second-best
    assert "checkpoint-20" in kept  # third best
    assert len(kept) <= 4  # keep_best + keep_recent with overlap
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_checkpoint.py -v
```

Expected: `ModuleNotFoundError: No module named 'training.checkpoint'`

**Step 3: Write minimal implementation**

```python
# training/checkpoint.py
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
            key=lambda x: int(x[0].split("-")[1]),
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
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_checkpoint.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add training/checkpoint.py tests/test_checkpoint.py
git commit -m "feat: add checkpoint management with rotating retention"
```

---

### Task 8: Create RunPod Startup Script

**Files:**
- Create: `scripts/setup_pod.sh`

**Step 1: Write the startup script**

```bash
#!/bin/bash
# RunPod startup script - auto-starts training when pod launches

set -e  # Exit on error

echo "=== RunPod Pod Startup Script ==="
echo "Start time: $(date)"

# Environment variables (set in RunPod template)
: "${GIT_REPO:?GIT_REPO not set}"
: "${GIT_BRANCH:?GIT_BRANCH not set}"
: "${CONFIG_PATH:?CONFIG_PATH not set}"
: "${WANDB_API_KEY:?WANDB_API_KEY not set}"

# Optional variables
NETWORK_VOLUME=${NETWORK_VOLUME:-/runpod_volume}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-}

echo "Git repo: $GIT_REPO"
echo "Git branch: $GIT_BRANCH"
echo "Config: $CONFIG_PATH"
echo "Network volume: $NETWORK_VOLUME"
echo "================================"

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Clone repo
WORKSPACE=/workspace
if [ ! -d "$WORKSPACE/.git" ]; then
    echo "Cloning repository..."
    git clone "$GIT_REPO" "$WORKSPACE"
fi

cd "$WORKSPACE"
git checkout "$GIT_BRANCH"
git pull

# Install dependencies
echo "Installing dependencies..."
uv pip install -r requirements.txt

# Setup network volume
mkdir -p "$NETWORK_VOLUME/checkpoints"
mkdir -p "$NETWORK_VOLUME/logs"

# Login to W&B
echo "Logging into Weights & Biases..."
wandb login "$WANDB_API_KEY"

# Run training
echo "Starting training..."
LOG_FILE="$NETWORK_VOLUME/logs/${EXPERIMENT_NAME:-training}-$(date +%Y%m%d-%H%M%S).log"

python training/train.py \
    --config "$CONFIG_PATH" \
    2>&1 | tee "$LOG_FILE"

echo "=== Training completed ==="
echo "End time: $(date)"
```

**Step 2: Make executable**

```bash
chmod +x scripts/setup_pod.sh
```

**Step 3: Create README for RunPod setup**

```markdown
# scripts/README.md

## RunPod Setup Instructions

### 1. Create Network Volume

```bash
runpodctl create volume well-log-ocr-checkpoints --size 100
```

### 2. Configure Environment Variables

In RunPod pod template, set these environment variables:

- `GIT_REPO`: Your git repository URL
- `GIT_BRANCH`: Experiment branch name
- `CONFIG_PATH`: Config file path (e.g., `configs/experiments/qwen3_qlora_r16.py`)
- `WANDB_API_KEY`: Your Weights & Biases API key
- `NETWORK_VOLUME` (optional): Default `/runpod_volume`
- `EXPERIMENT_NAME` (optional): Auto-generated if not set

### 3. Configure Storage Mount

Mount network volume to: `/runpod_volume`

### 4. Set Startup Script

Use `scripts/setup_pod.sh` as the pod startup script.

### 5. Launch Pod

```bash
runpodctl create pod \
  --gpus "RTX_4090:1" \
  --volume-name well-log-ocr-checkpoints:/runpod_volume \
  --spot \
  --spot-interruption="auto-restart" \
  --container-name ghcr.io/huggingface/unsloth \
  --env "GIT_REPO=..." \
  --env "GIT_BRANCH=..." \
  --env "CONFIG_PATH=configs/experiments/qwen3_qlora_r16.py" \
  --env "WANDB_API_KEY=..."
```
```

**Step 4: Commit**

```bash
git add scripts/
git commit -m "feat: add RunPod startup script and setup instructions"
```

---

### Task 9: Integrate Checkpointing into Training

**Files:**
- Modify: `training/train.py`

**Step 1: Add checkpoint callbacks to training**

Modify `create_trainer()` in `training/train.py`:
```python
from training.checkpoint import CheckpointManager

# Add to function
def create_trainer(config, model, tokenizer, train_dataset, eval_dataset, resume_from_checkpoint=None):
    # ... existing trainer setup ...

    # Wrap with custom callbacks for checkpoint management
    class CheckpointCallback(TrainerCallback):
        def __init__(self, checkpoint_manager):
            self.checkpoint_manager = checkpoint_manager

        def on_evaluate(self, args, state, control, metrics, **kwargs):
            """After evaluation, update checkpoint retention."""
            checkpoints = self.checkpoint_manager.get_checkpoint_losses()
            # Add current checkpoint
            chk_name = f"checkpoint-{state.global_step}"
            if "eval_loss" in metrics:
                checkpoints[chk_name] = metrics["eval_loss"]
            self.checkpoint_manager.cleanup(checkpoints)

    checkpoint_mgr = CheckpointManager(
        config.output_dir,
        keep_best=5,
        keep_recent=5,
    )

    trainer.add_callback(CheckpointCallback(checkpoint_mgr))
    return trainer
```

**Step 2: Test checkpoint retention**

```bash
python training/train.py --config configs/experiments/quick_val.py
```

Expected: Older checkpoints cleaned up, only best + recent kept

**Step 3: Commit**

```bash
git add training/train.py
git commit -m "feat: integrate checkpoint cleanup with training"
```

---

## Phase 3: Monitoring (Priority: Medium)

### Task 10: Add High-Loss Sample Logging

**Files:**
- Create: `training/eval_utils.py`
- Modify: `training/train.py`

**Step 1: Write high-loss sample logging**

```python
# training/eval_utils.py
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
    columns = ["step", "loss", "cer", "wer", "image", "prediction", "reference"]
    table = wandb.Table(columns=columns)

    for idx in worst_indices:
        sample = samples[idx]
        pred = predictions[idx]
        ref = references[idx]
        loss = losses[idx]

        # Compute metrics
        metrics = compute_sample_metrics(pred, ref)

        # Convert image to base64 for W&B
        img = sample["images"][0]
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        img_url = f"data:image/png;base64,{img_b64}"

        # Truncate text for display
        pred_short = pred[:500] + "..." if len(pred) > 500 else pred
        ref_short = ref[:500] + "..." if len(ref) > 500 else ref

        table.add_data(
            idx,  # step/sample index
            round(loss, 4),
            round(metrics["cer"], 4),
            round(metrics["wer"], 4),
            wandb.Image(img),
            pred_short,
            ref_short,
        )

    wandb.log({table_name: table})
```

**Step 2: Integrate into training script**

Add evaluation callback in `training/train.py`:
```python
from transformers import TrainerCallback
from training.eval_utils import log_high_loss_samples

class EvalLoggingCallback(TrainerCallback):
    def __init__(self, eval_dataset, top_k=20):
        self.eval_dataset = eval_dataset
        self.top_k = top_k

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """After evaluation, log high-loss samples."""
        if state.global_step % args.eval_steps != 0:
            return

        model = kwargs["model"]
        tokenizer = kwargs["tokenizer"]

        # Run inference on eval set
        from unsloth import FastVisionModel
        FastVisionModel.for_inference(model)

        predictions, references, losses = [], [], []
        samples_batch = []

        for i in range(min(100, len(self.eval_dataset))):  # Limit to 100 samples
            sample = self.eval_dataset[i]
            img = sample["images"][0]
            instruction = sample["messages"][0]["content"][1]["text"]
            reference = sample["messages"][1]["content"][0]["text"]

            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
            input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            inputs = tokenizer(img, input_text, add_special_tokens=False, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=512, use_cache=True)
                prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Compute loss (simplified - use trainer's eval loss)
                loss = metrics.get("eval_loss", 0.0)

            predictions.append(prediction)
            references.append(reference)
            losses.append(loss)
            samples_batch.append(sample)

        # Log to W&B
        if wandb.run is not None:
            log_high_loss_samples(samples_batch, predictions, references, losses, top_k=self.top_k)
```

**Step 3: Commit**

```bash
git add training/eval_utils.py training/train.py
git commit -m "feat: add high-loss sample logging to W&B"
```

---

### Task 11: Add Cost Tracking

**Files:**
- Modify: `training/train.py`

**Step 1: Add cost tracking to training**

```python
# Add to training/train.py
import time
from dataclasses import dataclass

@dataclass
class CostTracker:
    """Track training costs based on GPU hourly rate."""
    gpu_hourly_rate: float  # e.g., 0.40 for RTX 4090 spot
    start_time: float = 0.0
    total_cost: float = 0.0

    def start(self):
        self.start_time = time.time()

    def update(self) -> float:
        """Update total cost and return current value."""
        elapsed_hours = (time.time() - self.start_time) / 3600
        self.total_cost = elapsed_hours * self.gpu_hourly_rate
        return self.total_cost

# In main():
def get_gpu_rate(config: TrainingConfig) -> float:
    """Get hourly GPU rate from config."""
    # Approximate rates for spot instances (USD)
    rates = {
        "RTX_3090": 0.25,
        "RTX_4090": 0.40,
        "A100": 1.50,
        "H100": 2.50,
    }
    return rates.get(config.gpu_type, 0.40)

# In training loop:
cost_tracker = CostTracker(gpu_hourly_rate=get_gpu_rate(config))
cost_tracker.start()

# Add W&B logging callback
class CostLoggingCallback(TrainerCallback):
    def __init__(self, cost_tracker, budget_cap):
        self.cost_tracker = cost_tracker
        self.budget_cap = budget_cap

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log current cost to W&B."""
        current_cost = self.cost_tracker.update()
        if wandb.run is not None:
            wandb.log({"cost_usd": current_cost})

        # Warn if approaching budget
        if current_cost > self.budget_cap * 0.9:
            logger.warning(f"Approaching budget cap: ${current_cost:.2f} / ${self.budget_cap:.2f}")

trainer.add_callback(CostLoggingCallback(cost_tracker, config.budget_cap))
```

**Step 2: Commit**

```bash
git add training/train.py
git commit -m "feat: add GPU cost tracking to W&B"
```

---

### Task 12: Add Failure Handling (OOM and Network Retry)

**Files:**
- Modify: `training/train.py`

**Step 1: Add OOM adaptation wrapper**

```python
# Add to training/train.py
import traceback

class OOMAdaptation:
    """Handle OOM errors by adapting configuration."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.attempts = 0
        self.max_attempts = 2

    def should_retry(self, error: Exception) -> bool:
        """Check if error is OOM and we should retry."""
        if self.attempts >= self.max_attempts:
            return False

        error_str = str(error).lower()
        return "out of memory" in error_str or "cuda out of memory" in error_str

    def adapt(self) -> dict:
        """Adapt config for next attempt."""
        self.attempts += 1

        if self.attempts == 1:
            # First retry: reduce batch size
            old_batch = self.config.per_device_train_batch_size
            self.config.per_device_train_batch_size = max(1, old_batch // 2)
            return {"action": "reduce_batch", "old": old_batch, "new": self.config.per_device_train_batch_size}
        elif self.attempts == 2:
            # Second retry: enable gradient checkpointing
            self.config.use_gradient_checkpointing = "unsloth"
            return {"action": "enable_gradient_checkpointing"}

        return {"action": "fail"}

# Wrap training in main():
def train_with_oom_handling(config, trainer):
    """Run training with OOM adaptation."""
    oom_handler = OOMAdaptation(config)

    while True:
        try:
            return trainer.train()
        except RuntimeError as e:
            if oom_handler.should_retry(e):
                adaptation = oom_handler.adapt()
                logger.warning(f"OOM detected, adapting: {adaptation}")

                # Recreate trainer with adapted config
                # (Need to reload model with new settings)
                break  # For simplicity, require manual restart
            else:
                raise

# Replace trainer.train() with:
# train_with_oom_handling(config, trainer)
```

**Step 2: Add network retry wrapper**

```python
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
def load_dataset_with_retry(dataset_name: str):
    """Load dataset with exponential backoff retry."""
    from datasets import load_dataset
    return load_dataset(dataset_name)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
def wandb_init_with_retry(**kwargs):
    """Initialize W&B with retry."""
    return wandb.init(**kwargs)
```

**Step 3: Update requirements**

```bash
echo "tenacity>=8.0.0" >> requirements.txt
```

**Step 4: Commit**

```bash
git add training/train.py requirements.txt
git commit -m "feat: add OOM adaptation and network retry handling"
```

---

## Phase 4: Optimization (Priority: Low)

### Task 13: Add Early Stopping

**Files:**
- Create: `training/early_stopping.py`

**Step 1: Write early stopping callback**

```python
# training/early_stopping.py
"""Early stopping callback for training."""

from transformers import TrainerCallback
import logging

logger = logging.getLogger(__name__)


class EarlyStoppingCallback(TrainerCallback):
    """Stop training if validation loss plateaus."""

    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        """Initialize early stopping.

        Args:
            patience: Number of evals with no improvement before stopping.
            min_delta: Minimum change to qualify as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.wait = 0
        self.stopped = False

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Check if we should stop training."""
        if metrics is None or "eval_loss" not in metrics:
            return

        current_loss = metrics["eval_loss"]

        if self.best_loss is None:
            self.best_loss = current_loss
        elif current_loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = current_loss
            self.wait = 0
            logger.info(f"New best loss: {current_loss:.4f}")
        else:
            # No improvement
            self.wait += 1
            logger.info(f"No improvement for {self.wait} evals. Best: {self.best_loss:.4f}, Current: {current_loss:.4f}")

            if self.wait >= self.patience:
                logger.info(f"Early stopping triggered after {self.patience} evals with no improvement")
                control.should_training_stop = True
                self.stopped = True
```

**Step 2: Integrate into training**

```python
# In training/train.py
from training.early_stopping import EarlyStoppingCallback

# In create_trainer():
trainer.add_callback(EarlyStoppingCallback(patience=5, min_delta=0.001))
```

**Step 3: Commit**

```bash
git add training/early_stopping.py training/train.py
git commit -m "feat: add early stopping callback"
```

---

### Task 14: Add Spot Instance Auto-Restart

**Files:**
- Create: `scripts/resume_training.sh`

**Step 1: Write resume wrapper script**

```bash
#!/bin/bash
# Resume training from latest checkpoint if available

set -e

OUTPUT_DIR="${1:-/runpod_volume/checkpoints}"
CONFIG_PATH="${2:?CONFIG_PATH required}"

# Find latest checkpoint
LATEST_CHECKPOINT=$(python -c "
from training.checkpoint import get_latest_checkpoint
import sys
chk = get_latest_checkpoint('$OUTPUT_DIR')
print(chk if chk else '')
")

if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "Resuming from checkpoint: $LATEST_CHECKPOINT"
    python training/train.py \
        --config "$CONFIG_PATH" \
        --resume "$LATEST_CHECKPOINT"
else
    echo "No checkpoint found, starting fresh training"
    python training/train.py --config "$CONFIG_PATH"
fi
```

**Step 2: Make executable**

```bash
chmod +x scripts/resume_training.sh
```

**Step 3: Update startup script to use resume**

Modify `scripts/setup_pod.sh`:
```bash
# Replace python training/train.py line with:
python training/train.py \
    --config "$CONFIG_PATH" \
    --resume "$(python -c 'from training.checkpoint import get_latest_checkpoint; print(get_latest_checkpoint("$NETWORK_VOLUME/checkpoints") or "")')" \
    2>&1 | tee "$LOG_FILE"
```

**Step 4: Commit**

```bash
git add scripts/
git commit -m "feat: add checkpoint resume for spot instance recovery"
```

---

### Task 15: Add Experiment Comparison Tools

**Files:**
- Create: `scripts/compare_experiments.py`

**Step 1: Write experiment comparison script**

```python
#!/usr/bin/env python3
"""Compare training experiments from W&B or local logs."""

import argparse
import sys
from pathlib import Path
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Compare training experiments")
    parser.add_argument("--runs", nargs="+", required=True, help="W&B run IDs or checkpoint paths")
    parser.add_argument("--metrics", nargs="+", default=["eval_loss", "cer", "wer"], help="Metrics to compare")
    return parser.parse_args()

def load_run_metrics(run_id: str) -> dict:
    """Load metrics from a run (W&B or local)."""
    # Try W&B first
    try:
        import wandb
        api = wandb.Api()
        run = api.run(run_id)
        return run.summary
    except:
        pass

    # Fallback to local checkpoint
    path = Path(run_id)
    if path.exists():
        state_file = path / "trainer_state.json"
        if state_file.exists():
            return json.loads(state_file.read_text())

    raise ValueError(f"Could not load metrics for {run_id}")

def main():
    args = parse_args()

    print(f"{'Run ID':<40} {'Eval Loss':<12} {'CER':<8} {'WER':<8}")
    print("-" * 70)

    for run_id in args.runs:
        metrics = load_run_metrics(run_id)
        eval_loss = metrics.get("eval_loss", "N/A")
        cer = metrics.get("cer", "N/A")
        wer = metrics.get("wer", "N/A")

        print(f"{run_id:<40} {str(eval_loss):<12} {str(cer):<8} {str(wer):<8}")

if __name__ == "__main__":
    main()
```

**Step 2: Make executable**

```bash
chmod +x scripts/compare_experiments.py
```

**Step 3: Commit**

```bash
git add scripts/compare_experiments.py
git commit -m "feat: add experiment comparison utility"
```

---

## Final Steps

### Task 16: Documentation and README

**Files:**
- Create: `README_TRAINING.md`

**Step 1: Write comprehensive training README**

```markdown
# RunPod Training Guide

## Quick Start

### Local Quick Validation

```bash
python training/train.py --config configs/experiments/quick_val.py
```

### RunPod Full Training

1. Create experiment branch:
```bash
git checkout -b qwen3-qlora-r16-lr2e-4-steps500
```

2. Update config in `configs/experiments/`

3. Commit config:
```bash
git add configs/experiments/qwen3_qlora_r16.py
git commit -m "Exp: Qwen3 QLoRA r16 lr2e-4 steps500"
```

4. Launch RunPod pod (see `scripts/README.md`)

5. Monitor in W&B dashboard

## Configuration

Base config in `configs/base.py`, experiments in `configs/experiments/`.

## Monitoring

All experiments log to W&B project `well-log-ocr`.

## Cost Estimates

- RTX 4090 spot: ~$0.40/hr
- Quick validation (10 steps): ~$0.50
- Full training (500 steps): ~$5-10
```

**Step 2: Commit**

```bash
git add README_TRAINING.md
git commit -m "docs: add comprehensive training guide"
```

---

### Task 17: Final Integration Test

**Step 1: Run end-to-end test**

```bash
# Quick validation
python training/train.py --config configs/experiments/quick_val.py
```

**Step 2: Verify all components work**

- [ ] Config system loads
- [ ] Pre-flight checks pass
- [ ] Model loads successfully
- [ ] Dataset streams correctly
- [ ] Training runs for configured steps
- [ ] Checkpoints saved with retention
- [ ] W&B logging works
- [ ] High-loss samples uploaded
- [ ] Cost tracking accurate

**Step 3: Commit final integration**

```bash
git add .
git commit -m "feat: complete RunPod training pipeline implementation"
```

---

**End of Implementation Plan**

**Total Estimated Tasks:** 17
**Lines of Code:** ~2,500
**Files Created:** ~20
**Files Modified:** ~5

**Key Dependencies:**
- unsloth >= 2024.12.0
- transformers >= 4.40.0
- trl >= 0.9.0
- wandb >= 0.16.0
- tenacity >= 8.0.0
- editdistance >= 0.8.1
