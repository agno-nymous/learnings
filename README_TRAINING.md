# RunPod Training Guide

Comprehensive guide for training Well Log OCR models on RunPod using QLoRA fine-tuning.

## Quick Start

### Local Quick Validation

Validate your training pipeline locally before launching on RunPod:

```bash
python training/train.py --config configs/experiments/quick_val.py
```

This runs a minimal training loop (10 steps, 100 samples) to verify:
- Dataset loading works
- Model initializes correctly
- Checkpointing functions
- W&B logging connects

### RunPod Full Training

Once validated locally, launch full training on RunPod:

1. **Create experiment branch:**
```bash
git checkout -b qwen3-qlora-r16-lr2e-4-steps500
```

2. **Update config in `configs/experiments/`:**
```python
# configs/experiments/my_experiment.py
from configs.base import TrainingConfig

config = TrainingConfig(
    max_steps=500,
    learning_rate=2e-4,
    r=16,
    # ... other overrides
)
```

3. **Commit config:**
```bash
git add configs/experiments/my_experiment.py
git commit -m "Exp: Qwen3 QLoRA r16 lr2e-4 steps500"
```

4. **Launch RunPod pod** (see [scripts/README.md](scripts/README.md) for detailed instructions)

5. **Monitor in W&B dashboard:**
   - Project: `well-log-ocr`
   - Real-time metrics: loss, learning rate, cost tracking

## Configuration

### Base Configuration

Base configuration is defined in [`configs/base.py`](configs/base.py):

```python
@dataclass
class TrainingConfig:
    # Model
    model_name: str = "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit"
    load_in_4bit: bool = True

    # LoRA
    r: int = 16                    # LoRA rank
    lora_alpha: int = 16           # LoRA alpha
    lora_dropout: float = 0

    # Training
    max_steps: int = 500
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4

    # Data
    dataset_name: str = "wrath/well-log-headers-ocr"

    # Monitoring
    wandb_project: str = "well-log-ocr"
    report_to_wandb: bool = True
```

### Experiment Configurations

Experiment-specific configs in [`configs/experiments/`](configs/experiments/):

- **`quick_val.py`** - Fast validation (10 steps, 100 samples)
- **`qwen3_qlora_r16.py`** - Full training (500 steps)

Create your own by importing `TrainingConfig` and overriding specific parameters:

```python
from configs.base import TrainingConfig

config = TrainingConfig(
    max_steps=1000,
    learning_rate=1e-4,
    r=32,
    train_subset=1000,
    eval_subset=200,
)
```

## Monitoring

### Weights & Biases (W&B)

All experiments log to W&B project `well-log-ocr`:

- **URL**: https://wandb.ai/your-entity/well-log-ocr
- **Metrics tracked**:
  - `train_loss` - Training loss per step
  - `eval_loss` - Validation loss
  - `learning_rate` - Current learning rate
  - `cost_usd` - Real-time GPU cost tracking
  - `train/epoch` - Current epoch
  - `train/global_step` - Global step number

### GPU Utilization

Monitor GPU usage during training:

```bash
# SSH into RunPod pod
nvidia-smi -l 1  # Update every second
```

### Checkpoint Monitoring

List checkpoints saved to network volume:

```bash
ls -lh /runpod_volume/checkpoints/
```

Checkpoints are automatically managed:
- **Best 5** kept (by lowest eval_loss)
- **Recent 5** kept (by step number)
- Older checkpoints auto-deleted to save space

## Cost Estimates

### GPU Hourly Rates (Spot Instances)

| GPU     | VRAM   | Hourly Rate | 500 Steps Estimate |
|---------|--------|-------------|-------------------|
| RTX 3090 | 24GB  | ~$0.25/hr   | ~$3-5             |
| RTX 4090 | 24GB  | ~$0.40/hr   | ~$5-8             |
| A100     | 80GB  | ~$1.50/hr   | ~$15-25           |
| H100     | 80GB  | ~$2.50/hr   | ~$25-40           |

### Example Costs

- **Quick validation** (10 steps, 100 samples): ~$0.50 on RTX 4090
- **Full training** (500 steps, full dataset): ~$5-10 on RTX 4090
- **Extended training** (1000 steps): ~$10-20 on RTX 4090

### Cost Tracking

Training cost is automatically tracked and logged to W&B:

```python
# Cost calculated from: (elapsed_hours * gpu_hourly_rate)
# Logged as 'cost_usd' metric in W&B
# Warns when approaching budget_cap (default: $10)
```

## Features

### Automated Checkpointing

- **Saves best checkpoints**: Automatically keeps top 5 by eval_loss
- **Keeps recent checkpoints**: Retains last 5 checkpoints
- **Auto-cleanup**: Deletes old checkpoints to save storage
- **Spot recovery**: Auto-resumes from checkpoint after preemption

### Cost Tracking

Real-time GPU spending logged to W&B:
- Tracks elapsed time
- Multiplies by GPU hourly rate
- Warns at 90% of budget cap
- Logs every training step

### Pre-flight Validation

Catches config errors before training starts:
- Validates dataset accessibility
- Checks W&B API key
- Verifies output directory is writable
- Confirms model exists on HuggingFace

```python
# Automatically runs before training
from training.preflight import run_preflight_checks
run_preflight_checks(config)
```

### Network Retry

Automatic retry for transient network failures:
- Dataset loading retries 3x with exponential backoff
- W&B initialization retries 3x with exponential backoff
- Prevents training failure from temporary network issues

### Spot Instance Recovery

Auto-resume from checkpoint after preemption:
- Checkpoints saved to network volume (`/runpod_volume/checkpoints`)
- Pod auto-restarts (if configured with auto-restart)
- Training resumes from last checkpoint
- No manual intervention required

### High-Loss Sample Logging

Worst performing samples logged to W&B for review:
- Automatically logs samples with highest loss during evaluation
- Visual inspection in W&B dashboard
- Identify problematic data patterns

## Module Structure

```
training/
├── train.py              # Main training script
├── dataset.py            # Dataset loading utilities
├── metrics.py            # CER/WER computation
├── checkpoint.py         # Checkpoint management
├── preflight.py          # Pre-flight validation
├── eval_utils.py         # Evaluation utilities
└── early_stopping.py     # Early stopping callback
```

### Key Modules

- **`train.py`** - Main entry point, orchestrates training loop
- **`dataset.py`** - HuggingFace dataset loading and preprocessing
- **`metrics.py`** - Character Error Rate (CER) and Word Error Rate (WER)
- **`checkpoint.py`** - Checkpoint retention policy management
- **`preflight.py`** - Pre-training validation checks
- **`eval_utils.py`** - Evaluation helper functions
- **`early_stopping.py`** - Early stopping based on eval loss plateau

## Experiment Workflow

### 1. Experiment Design

Define your hypothesis and parameters:

```python
# Example: Higher LoRA rank might improve accuracy
config = TrainingConfig(
    r=32,              # Increased from 16
    learning_rate=1e-4,  # Lower LR for higher rank
    max_steps=1000,
    experiment_name="qwen3-qlora-r32-lr1e4"
)
```

### 2. Create Branch

```bash
git checkout -b qwen3-qlora-r32-lr1e4
```

### 3. Update Config

Create new experiment config in `configs/experiments/`:

```bash
# configs/experiments/qwen3_qlora_r32.py
from configs.base import TrainingConfig

config = TrainingConfig(
    r=32,
    learning_rate=1e-4,
    max_steps=1000,
)
```

### 4. Commit Config

```bash
git add configs/experiments/qwen3_qlora_r32.py
git commit -m "Exp: Qwen3 QLoRA r32 lr1e-4 steps1000"
```

### 5. Launch RunPod Pod

Follow instructions in [scripts/README.md](scripts/README.md):

1. Create network volume (if not exists)
2. Configure environment variables:
   - `GIT_REPO` - Your repository URL
   - `GIT_BRANCH` - `qwen3-qlora-r32-lr1e4`
   - `CONFIG_PATH` - `configs/experiments/qwen3_qlora_r32.py`
   - `WANDB_API_KEY` - Your W&B API key
3. Mount network volume to `/runpod_volume`
4. Set startup script to `scripts/setup_pod.sh`
5. Deploy pod

### 6. Monitor Training

- **W&B Dashboard**: https://wandb.ai/your-entity/well-log-ocr
- **SSH into pod**: View logs with `tail -f /runpod_volume/logs/*.log`
- **Check checkpoints**: `ls -lh /runpod_volume/checkpoints/`

### 7. Review Results

After training completes:

```bash
# Download final model from RunPod
scp -r user@pod-ip:/runpod_volume/checkpoints/qwen3-qlora-r32 ./models/

# Evaluate on test set
python training/evaluate.py --model ./models/qwen3-qlora-r32
```

### 8. Merge and Document

```bash
# Merge to main if successful
git checkout main
git merge qwen3-qlora-r32-lr1e4

# Document findings
# Update README with results, hyperparameters, observations
```

## Advanced Usage

### Resume from Checkpoint

If training is interrupted:

```bash
python training/train.py \
  --config configs/experiments/qwen3_qlora_r16.py \
  --resume /runpod_volume/checkpoints/checkpoint-250
```

### Custom Dataset

Use your own dataset:

```python
config = TrainingConfig(
    dataset_name="your-org/your-dataset",
    train_subset=-1,  # Use all training data
    eval_subset=-1,   # Use all eval data
)
```

### Multiple GPUs

For multi-GPU training (requires appropriate GPU):

```python
config = TrainingConfig(
    per_device_train_batch_size=4,  # Adjust based on GPU memory
    gradient_accumulation_steps=2,
)
```

Then run with distributed training:

```bash
torchrun --nproc_per_node=2 training/train.py --config configs/experiments/my_experiment.py
```

### Disable W&B Logging

For testing without W&B:

```python
config = TrainingConfig(
    report_to_wandb=False,
)
```

## Troubleshooting

### Out of Memory Errors

**Symptoms**: CUDA out of memory, training crashes

**Solutions**:
- Reduce `per_device_train_batch_size` (try 1 instead of 2)
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Use model with less parameters
- Use GPU with more VRAM (A100 80GB instead of RTX 4090)

### Slow Training

**Symptoms**: Training takes longer than expected

**Solutions**:
- Increase `per_device_train_batch_size` if memory allows
- Reduce `logging_steps` and `eval_steps` for less frequent evaluation
- Use faster GPU (A100/H100 instead of RTX 4090)
- Check GPU utilization with `nvidia-smi`

### Dataset Loading Errors

**Symptoms**: "Dataset not found" or authentication errors

**Solutions**:
- Verify dataset name is correct
- Login to HuggingFace: `huggingface-cli login`
- Check dataset access permissions
- Use `train_subset` parameter to test with smaller data

### W&B Connection Issues

**Symptoms**: "W&B init failed" or metrics not logging

**Solutions**:
- Verify `WANDB_API_KEY` is set correctly
- Check internet connectivity from pod
- Test W&B login: `wandb login`
- Set `report_to_wandb=False` to disable

### Spot Instance Preemption

**Symptoms**: Pod stops unexpectedly, training interrupted

**Solutions**:
- Enable auto-restart in pod configuration
- Ensure network volume is mounted (`/runpod_volume`)
- Checkpoints auto-save to network volume
- Training auto-resumes when pod restarts
- No manual intervention required

## Best Practices

1. **Always test locally first** - Use `quick_val.py` config to validate pipeline
2. **Use spot instances** - 50-80% cost savings
3. **Enable auto-restart** - Automatically resume after preemption
4. **Monitor W&B dashboard** - Real-time metrics and cost tracking
5. **Set budget caps** - Prevent overspending (default: $10)
6. **Commit experiment configs** - Track hyperparameters in git
7. **Document findings** - Record what worked and what didn't
8. **Start small** - Test with fewer samples, then scale up

## Additional Resources

- **RunPod Setup**: [scripts/README.md](scripts/README.md) - Detailed RunPod configuration
- **Base Config**: [`configs/base.py`](configs/base.py) - All configuration options
- **Training Script**: [`training/train.py`](training/train.py) - Main training logic
- **W&B Project**: https://wandb.ai/your-entity/well-log-ocr - Experiment tracking

## Support

For issues or questions:
- Check logs: `/runpod_volume/logs/training-*.log`
- Review W&B runs for detailed metrics
- Verify config parameters match your intent
- Test with `quick_val.py` config before full training
