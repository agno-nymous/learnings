# RunPod Cloud Training Specification

**Project:** Well Log OCR Model Training on RunPod
**Date:** 2025-01-03
**Status:** Ready for Implementation

## Executive Summary

Move well log OCR model fine-tuning from local/Colab to RunPod cloud infrastructure. Focus on Qwen3-VL-2B QLoRA training with cost optimization, faster iteration cycles, and production pipeline capabilities. Use consumer-grade GPUs (RTX 3090/4090) with spot instances for cost efficiency.

---

## 1. Objectives

### Primary Goals
1. **Cost Optimization:** Use RunPod spot instances at ~$0.20-0.40/hr vs local/Colab alternatives
2. **Faster Iteration:** Enable parallel training jobs and reliable on-demand resources
3. **Production Pipeline:** Build systematic, automated training workflow with minimal manual intervention
4. **Model Selection:** Train Qwen3-VL-2B first, evaluate, then decide on PaddleOCR-VL

### Success Criteria
- Training cost: $5-10 per experiment (middle ground)
- Iteration time: <1 hour for quick validation runs
- Automated recovery from spot preemptions
- Reproducible experiments tracked in W&B with cost logging
- Manual review workflow for trained models before deployment

---

## 2. Technical Architecture

### 2.1 Hardware Configuration

**GPU Selection:** Consumer-grade (RTX 3090/4090)
- Cost: ~$0.20-0.40/hour
- VRAM: 24GB (sufficient for Qwen3-VL-2B QLoRA)
- Scaling: Start single-GPU, scale to multi-GPU if needed

**Instance Type:** Spot instances (aggressive)
- Accept preemptions with auto-restart from checkpoint
- Lowest cost, acceptable latency for training
- Fallback: Switch to dedicated for final production runs if needed

**Storage Strategy:** Network Volume with rotating retention
- Capacity: 50-100GB (checkpoints + high-loss samples)
- Retention: Keep 5-10 checkpoints (best + recent)
- Cost: ~$0.00015/GB/month = ~$0.015/month for 100GB

### 2.2 Environment Setup

**Base Image:** RunPod template + startup script
- Primary: Unsloth base image (if available)
- Fallback: HF PyTorch image
- Package Manager: `uv` for fast dependency installation

**Startup Script Responsibilities:**
```bash
1. Install dependencies via uv (requirements.txt)
2. Clone git repo (branch per experiment)
3. Mount network volume to /runpod_volume
4. Initialize W&B (login + project setup)
5. Execute training script automatically
```

**Key Dependencies:**
- `unsloth` (latest)
- `torch` (with CUDA support)
- `transformers`, `peft`, `trl`
- `datasets` (HuggingFace)
- `wandb`
- `pillow`, `pandas`

### 2.3 Data Strategy

**Dataset:** `wrath/well-log-headers-ocr` from HuggingFace
- Training: 501 samples (full), 100-200 samples (quick validation)
- Evaluation: 126 samples
- Format: Base64-encoded images in JSONL
- Loading: Streaming via `datasets.load_dataset()` (no local storage needed)

**Data Flow:**
```
HuggingFace Hub → Streaming load → Lazy decoding → Training batch
```

**Quick Validation Subset:**
- 5-10 training steps for very quick validation
- 100-200 samples from train split
- Verify: code runs, loss decreases, eval loop works

---

## 3. Training Configuration

### 3.1 Model Configuration

**Primary Model:** Qwen3-VL-2B (Unsloth optimized)
```python
model_name = "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit"
load_in_4bit = True  # QLoRA
use_gradient_checkpointing = "unsloth"
```

**LoRA Configuration:**
```python
r = 16              # Rank
lora_alpha = 16     # Scaling (assume r=alpha)
lora_dropout = 0
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
```

### 3.2 Training Hyperparameters

**Base Configuration (Full Training):**
```python
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
effective_batch_size = 8

warmup_steps = 5
max_steps = 500  # Full training
learning_rate = 2e-4

optim = "adamw_8bit"
weight_decay = 0.01
lr_scheduler_type = "linear"

fp16 = not is_bf16_supported()
bf16 = is_bf16_supported()

logging_steps = 1
eval_steps = 50  # Moderate validation frequency
max_seq_length = 2048
```

**Quick Validation Configuration:**
- Override: `max_steps = 10`
- Subset: First 100-200 training samples
- Purpose: Verify setup, not convergence

### 3.3 Evaluation Configuration

**Validation Frequency:** Every 50 steps (moderate)
- More frequent than notebooks (10 steps)
- Less overhead than very frequent validation
- Adjust empirically based on validation time

**Metrics Tracked:**
1. **Validation Loss** (primary optimization target)
2. **Character Error Rate (CER)** - edit distance normalized by length
3. **Word Error Rate (WER)** - word-level accuracy
4. **Per-step metrics** - loss, learning rate, GPU utilization

**High-Loss Sample Saving:**
- Top 20-50 samples by loss per evaluation
- Saved to W&B Artifacts (not network volume)
- Include: input image, model prediction, ground truth, loss, CER, WER
- Purpose: Manual review of worst failures

---

## 4. Automation & Workflow

### 4.1 Experiment Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Create experiment branch                                  │
│    git checkout -b qwen3-qlora-r16-lr2e-4-steps500          │
├─────────────────────────────────────────────────────────────┤
│ 2. Update config for experiment                             │
│    Edit configs/experiments/qwen3_qlora_r16.py              │
├─────────────────────────────────────────────────────────────┤
│ 3. Commit config on milestone                               │
│    git commit -am "Exp: Qwen3 QLoRA r16 lr2e-4"             │
├─────────────────────────────────────────────────────────────┤
│ 4. Launch RunPod pod                                        │
│    - Select RTX 4090 spot instance                          │
│    - Set network volume mount                               │
│    - Configure startup script                               │
│    - Set budget cap: $5-10                                  │
├─────────────────────────────────────────────────────────────┤
│ 5. Auto-start training                                      │
│    - Startup script clones repo                             │
│    - Installs dependencies via uv                           │
│    - Initializes W&B                                       │
│    - Runs train.py with experiment config                   │
├─────────────────────────────────────────────────────────────┤
│ 6. Monitor via W&B                                          │
│    - Live training dashboard                                │
│    - Loss curves, metrics, GPU utilization                  │
│    - Cost tracking                                          │
├─────────────────────────────────────────────────────────────┤
│ 7. Completion or Failure                                    │
│    - Notification via RunPod logs                           │
│    - Checkpoints saved to network volume                    │
│    - High-loss samples to W&B artifacts                     │
├─────────────────────────────────────────────────────────────┤
│ 8. Manual Review                                            │
│    - Download best checkpoint from network volume           │
│    - Review high-loss samples in W&B                        │
│    - Decide: deploy, retrain, or iterate                    │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Configuration Management

**Structure:** Base + Experiment Configs
```
configs/
├── base.py              # Full training config (base)
└── experiments/
    ├── qwen3_qlora_r16.py       # Extends base for specific run
    ├── qwen3_qlora_r32.py       # Different rank
    └── quick_val.py             # Quick validation overrides
```

**Config Format:** Python (dataclass/attrs)
```python
# configs/base.py
@dataclass
class TrainingConfig:
    # Model
    model_name: str = "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit"
    load_in_4bit: bool = True

    # LoRA
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0

    # Training
    max_steps: int = 500
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4

    # Data
    dataset_name: str = "wrath/well-log-headers-ocr"
    train_subset: int = -1  # -1 = full dataset

    # Hardware
    gpu_type: str = "RTX_4090"
    spot_instance: bool = True
    budget_cap: float = 10.0  # dollars

    # Monitoring
    wandb_project: str = "well-log-ocr"
    eval_steps: int = 50
```

**Experiment Config Example:**
```python
# configs/experiments/qwen3_qlora_r16.py
from configs.base import TrainingConfig

config = TrainingConfig(
    # Override specific params
    r=16,
    lora_alpha=16,
    learning_rate=2e-4,
    max_steps=500,
    # Rest inherited from base
)
```

### 4.3 Branch Strategy

**Per-Experiment Branching:**
- Format: `{model}-{method}-r{rank}-lr{lr}-steps{steps}`
- Examples:
  - `qwen3-qlora-r16-lr2e-4-steps500`
  - `qwen3-qlora-r32-lr1e-4-steps1000`

**Commit Strategy:** Commit on milestones
1. After successful quick validation
2. After achieving target metric (CER < threshold)
3. After major config changes

**Git Workflow:**
```
main (stable configs)
  └── qwen3-qlora-r16-lr2e-4-steps500 (experiment branch)
        ├── commit: Initial config for r16 experiment
        └── commit: Validation passed, ready for full training
```

---

## 5. Monitoring & Observability

### 5.1 Weights & Biases (W&B) Integration

**Project:** `well-log-ocr`

**Experiment Tracking:**
- Auto-naming from config: `qwen3-qlora-r16-lr2e-4-steps500-20250103-143052`
- Logged parameters: Full config snapshot
- Logged metrics:
  - Training/eval loss per step
  - Learning rate
  - GPU utilization (if available)
  - CER, WER per evaluation
  - Training time, cost estimate

**Cost Logging:**
- Track: pod start/end time, GPU type, hourly rate
- Calculate: `duration_hours * hourly_rate`
- Display in W&B dashboard alongside metrics
- Alert if approaching budget cap

**Visualizations:**
- Loss curves (train/eval)
- CER/WER over time
- Sample predictions (before/after)
- High-loss sample gallery

### 5.2 Alerts & Notifications

**Completion/Failure Alerts:**
- Primary: RunPod native notifications (console logs)
- Secondary: RunPod webhook (can integrate with email/Slack if needed)
- Email: Skip for now (too much setup)

**Alert Triggers:**
- Training completed successfully
- Training failed (exception, OOM, network error)
- Budget cap approaching (via W&B)
- Pod preemption (spot instance)

### 5.3 Logging Strategy

**Primary:** W&B captures most logs
- Trainer outputs (loss, metrics)
- Console outputs (print statements)
- Exception tracebacks

**Secondary:** Minimal file logging
- Save detailed tracebacks for debugging failures
- Log location: `/runpod_volume/logs/{experiment_name}.log`
- Rotation: Keep last 5 log files

---

## 6. Failure Handling & Recovery

### 6.1 Spot Instance Preemptions

**Auto-Restart Strategy:**
1. Checkpoint saved every eval_steps (50 steps)
2. On preemption, pod terminates
3. New pod auto-starts from same git branch
4. Training script detects last checkpoint in network volume
5. Resumes from last saved step

**Restart Behavior:**
- Load last checkpoint from network volume
- Continue training with same config
- W&B run resumes (same run ID)

### 6.2 Out of Memory (OOM) Adaptation

**Auto-Adaptation Attempts (in order):**
1. Reduce batch size by half (2 → 1)
2. Enable gradient checkpointing (if not enabled)
3. If still OOM after 2 attempts: Fail and notify

**Implementation:**
```python
try:
    trainer.train()
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        if attempt < 2:
            # Adapt config and retry
            config.per_device_train_batch_size //= 2
            # Retry from checkpoint
        else:
            # Log and notify
            send_alert("OOM after 2 adaptation attempts")
            raise
```

### 6.3 Other Failures

**Network Errors (HF, W&B):**
- Auto-retry with exponential backoff
- Max retries: 3
- Backoff: 1s, 2s, 4s

**Manual Review Required:**
- Dataset corruption
- Invalid model name
- Code bugs (syntax, logic)
- Hardware incompatibility

**Failure Recovery Workflow:**
```
Failure occurs
  ↓
Log exception to file and W&B
  ↓
Send alert via RunPod logs
  ↓
Manual investigation
  ↓
Fix config or code
  ↓
Commit fix to experiment branch
  ↓
Restart pod (manual restart from checkpoint)
```

---

## 7. Cost Control

### 7.1 Budget Management

**Per-Experiment Budget:** $5-10
- Quick validation (5-10 steps): ~$0.50-1
- Full training (500 steps): ~$5-10
- Depends on actual training time

**Enforcement:** RunPod platform limit
- Set maximum pod runtime in RunPod template
- Example: 4 hours = $1.60 (at $0.40/hr)
- Or: Set RunPod spending limit for account

**Cost Tracking:**
- Real-time: W&B dashboard shows accumulated cost
- Post-training: Summary in W&B run summary
- Per-experiment: CSV log in network volume

### 7.2 Cost Optimization Strategies

**Spot Instances:**
- Primary: Use spot for all training (aggressive)
- Savings: ~70-80% vs dedicated
- Tradeoff: Potential preemptions, auto-restart adds time

**Early Stopping:**
- Trigger: Validation loss plateaus
- Metric: No improvement for 5 consecutive evals
- Action: Stop training, save current best

**Quick Validation:**
- Strategy: 5-10 step runs before full training
- Purpose: Catch config errors before expensive runs
- Cost: ~$0.50 vs $5-10 for full training

**Rotating Checkpoint Retention:**
- Keep: Best + recent 5-10 checkpoints
- Delete: Older checkpoints automatically
- Savings: Network volume storage costs

---

## 8. Checkpointing & Model Management

### 8.1 Checkpoint Strategy

**Saving Frequency:**
- Every eval_steps (50 steps)
- Plus: final checkpoint at end
- Plus: best model by validation loss

**Checkpoint Format:** LoRA adapters only
```python
model.save_pretrained(f"{output_dir}/checkpoint-{step}")
tokenizer.save_pretrained(f"{output_dir}/checkpoint-{step}")
```

**Storage Location:** Network volume
```
/runpod_volume/
├── checkpoints/
│   ├── qwen3-qlora-r16-lr2e-4-steps500/
│   │   ├── checkpoint-50/
│   │   ├── checkpoint-100/
│   │   ├── best-model/          # Best by validation loss
│   │   └── final-checkpoint/
│   └── qwen3-qlora-r32-lr1e-4-steps1000/
│       └── ...
└── logs/
    └── *.log
```

**Rotating Retention:**
- Keep: Best, last, and first checkpoints
- Keep: Top 5 by validation loss
- Keep: Last 5 checkpoints (recent history)
- Auto-delete: Checkpoints outside retention policy

### 8.2 Post-Training Workflow

**After Training Completes:**
1. **Save final checkpoint** to network volume
2. **Upload to W&B:**
   - High-loss samples (as artifacts)
   - Final metrics (CER, WER, loss)
   - Sample predictions (before/after)
3. **Push to HuggingFace (optional):**
   - Best model only
   - Manual decision after review
4. **Manual review:**
   - View high-loss samples in W&B
   - Compare metrics to baseline
   - Decide: deploy, retrain, iterate

**Manual Review Criteria:**
- Quantitative: CER, WER, validation loss
- Qualitative: Visual inspection of high-loss samples
- Practical: Inference speed, model size
- Cost: Training cost, potential ROI

---

## 9. Pre-Flight Validation

### 9.1 Validation Checks

Before starting expensive training runs:

**1. Model Exists Check:**
```python
try:
    AutoModelForCausalLM.from_pretrained(config.model_name)
except Exception:
    raise ValueError(f"Model {config.model_name} not found")
```

**2. Dataset Access Check:**
```python
try:
    dataset = load_dataset(config.dataset_name, split="train[:1]")
    assert len(dataset) > 0
except Exception:
    raise ValueError(f"Dataset {config.dataset_name} not accessible")
```

**3. Network Volume Check:**
```python
if not os.path.ismount("/runpod_volume"):
    raise ValueError("Network volume not mounted")
if not os.access("/runpod_volume", os.W_OK):
    raise ValueError("Network volume not writable")
```

**4. VRAM Capacity Check:**
```python
gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
required_memory = estimate_memory_requirements(config)
if gpu_memory < required_memory:
    raise ValueError(f"GPU has {gpu_memory}GB, needs {required_memory}GB")
```

### 9.2 Config Summary Display

Before training starts, display:
```
=== Training Configuration ===
Model: qwen3-qlora-r16-lr2e-4-steps500
Dataset: wrath/well-log-headers-ocr (train: 501, eval: 126)
GPU: RTX 4090 (24GB VRAM)
Max Steps: 500
Batch Size: 2 × 4 grad_accum = 8 effective
Learning Rate: 2e-4
Budget Cap: $10.00
Estimated Time: ~4 hours (~$1.60)
Estimated Cost: ~$1.60

Press Enter to continue or Ctrl+C to cancel...
```

---

## 10. Implementation Roadmap

### Phase 1: Foundation (Priority: High)
1. **Convert notebook to training script**
   - Extract training logic from `finetune_qwen3_vl_qlora.ipynb`
   - Create `train.py` with CLI/config file support
   - Implement lazy dataset loading

2. **Setup configuration system**
   - Create `configs/base.py` with all defaults
   - Create `configs/experiments/quick_val.py` for validation
   - Implement config validation

3. **Basic W&B integration**
   - Initialize W&B in training script
   - Log metrics (loss, CER, WER)
   - Display config in W&B UI

### Phase 2: Automation (Priority: High)
4. **RunPod startup script**
   - Dependency installation via uv
   - Git repo cloning
   - Network volume mounting
   - W&B initialization
   - Auto-start training

5. **Checkpoint management**
   - Save LoRA adapters to network volume
   - Implement rotating retention
   - Auto-resume from checkpoint

6. **Pre-flight validation**
   - Model/dataset checks
   - VRAM estimation
   - Config summary display

### Phase 3: Monitoring (Priority: Medium)
7. **Advanced W&B features**
   - Cost logging
   - High-loss sample artifact upload
   - Custom visualizations

8. **Failure handling**
   - OOM adaptation
   - Network retry with backoff
   - Detailed error logging

9. **Alerting**
   - RunPod webhook configuration
   - Budget alerts via W&B

### Phase 4: Optimization (Priority: Low)
10. **Cost optimization**
    - Early stopping implementation
    - Spot instance auto-restart
    - Per-experiment budget enforcement

11. **Experiment management**
    - Branch naming automation
    - Git tagging for experiments
    - Experiment comparison tools

---

## 11. Known Tradeoffs

### 11.1 Cost vs Speed
**Tradeoff:** Spot instances are cheaper but can be preempted
**Decision:** Aggressive spot usage with auto-restart
**Rationale:** 70-80% cost savings worth potential restart delays
**Mitigation:** Frequent checkpointing (every 50 steps)

### 11.2 Storage Cost vs Convenience
**Tradeoff:** Network volume costs money vs pod storage is free but ephemeral
**Decision:** Network volume for checkpoints, W&B for samples
**Rationale:** Checkpoints need persistence for resume capability
**Mitigation:** Rotating retention limits storage growth

### 11.3 Reproducibility vs Flexibility
**Tradeoff:** Strict versioning (containers) vs quick iteration
**Decision:** Git-tracked configs, flexible environment
**Rationale:** Fast iteration more important than exact environment reproducibility
**Mitigation:** Commit configs on milestones, W&B tracks full config snapshot

### 11.4 Automation vs Control
**Tradeoff:** Fully automated (CI/CD) vs manual control
**Decision:** Script-based automation with manual review
**Rationale:** Balance of automation with human oversight for quality
**Mitigation:** Pre-flight validation prevents wasted runs

### 11.5 Evaluation Frequency
**Tradeoff:** Frequent eval (10 steps) vs training speed
**Decision:** Moderate frequency (50 steps)
**Rationale:** Balance visibility with minimal overhead
**Mitigation:** Adjust empirically based on validation time

---

## 12. Edge Cases & Considerations

### 12.1 Dataset Issues
- **Empty subset:** If subset size > dataset size, use full dataset
- **Missing splits:** If eval split missing, create from train split
- **Corrupted images:** Skip gracefully, log to W&B

### 12.2 GPU Limitations
- **VRAM insufficient:** Auto-reduce batch size, then gradient checkpointing
- **GPU unavailable:** Wait with retry, or fail with alert
- **Multi-GPU:** Future enhancement, not initial implementation

### 12.3 Network Issues
- **HF hub down:** Retry with backoff, or use cached dataset
- **W&B upload failed:** Continue training, retry upload later
- **Git clone failed:** Fail immediately, check repo URL

### 12.4 Configuration Errors
- **Invalid model name:** Pre-flight check, fail before training
- **Negative steps:** Validation in config, reject invalid values
- **Conflicting overrides:** Last override wins, log warning

### 12.5 Time-Based Issues
- **Training exceeds budget:** Stop at budget cap, save checkpoint
- **Very long training:** Implement mid-training notifications
- **Clock changes:** Use monotonic time for duration tracking

---

## 13. Security Considerations

### 13.1 Credentials Management
- **HuggingFace token:** Environment variable `HF_TOKEN`
- **W&B API key:** Environment variable `WANDB_API_KEY`
- **Git credentials:** SSH key or token (never commit to repo)

### 13.2 Network Volume Security
- **Permissions:** User-isolated, no cross-pod access
- **Encryption:** RunPod handles encryption at rest
- **Access:** Only pod with correct mount credentials

### 13.3 Code Security
- **Trusted remote code:** Only for known models (PaddleOCR-VL)
- **Git repo:** Private repository for training code
- **Secrets scanning:** Never log tokens or credentials

---

## 14. Success Metrics

### 14.1 Technical Metrics
- **CER:** < 10% (target)
- **WER:** < 20% (target)
- **Training stability:** < 5% failure rate
- **Resume success:** 100% (from valid checkpoint)

### 14.2 Cost Metrics
- **Per-experiment cost:** $5-10 (target)
- **Spot savings:** > 70% vs dedicated
- **Storage cost:** < $1/month
- **Total monthly cost:** < $50 (for ~5-10 experiments)

### 14.3 Time Metrics
- **Quick validation:** < 30 min end-to-end
- **Full training:** < 8 hours
- **Iteration cycle:** < 1 day (including review)
- **Setup time:** < 1 day initial, < 1 hour per new experiment

### 14.4 Workflow Metrics
- **Manual intervention:** < 20% of runs
- **Reproducibility:** 100% (config in git, W&B tracking)
- **Review time:** < 1 hour per trained model

---

## 15. Open Questions

### 15.1 To Be Determined
1. **Exact budget per experiment:** Start with $10, adjust based on actual training times
2. **Validation frequency:** Start with 50 steps, adjust based on validation duration
3. **High-loss sample count:** Start with 20-50, adjust based on value
4. **Multi-GPU need:** Determine after single-GPU baseline performance

### 15.2 Future Considerations
1. **PaddleOCR-VL training:** After Qwen3-VL evaluation
2. **Model comparison:** If both models trained, systematic comparison
3. **Production deployment:** Inference serving strategy (RunPod Serverless, HF API, etc.)
4. **Dataset expansion:** Collect more training data if needed
5. **Hyperparameter sweeps:** If single run insufficient, consider automated search

---

## Appendix A: Quick Reference Commands

### RunPod Setup
```bash
# Create network volume
runpodctl create volume well-log-ocr-checkpoints --size 100

# Build custom template (if needed)
runpodctl build template -f Dockerfile .

# Launch pod with spot instance
runpodctl create pod --gpus "RTX_4090:1" \
  --volume-name well-log-ocr-checkpoints \
  --template-id <template_id> \
  --spot \
  --spot-interruption="auto-restart"
```

### Training Commands
```bash
# Quick validation (5-10 steps)
python train.py --config configs/experiments/quick_val.py

# Full training
python train.py --config configs/experiments/qwen3_qlora_r16.py

# Resume from checkpoint
python train.py --config configs/experiments/qwen3_qlora_r16.py \
  --resume /runpod_volume/checkpoints/.../checkpoint-100
```

### Git Workflow
```bash
# Create experiment branch
git checkout -b qwen3-qlora-r16-lr2e-4-steps500

# Commit config
git add configs/experiments/qwen3_qlora_r16.py
git commit -m "Exp: Qwen3 QLoRA r16 lr2e-4 steps500"

# Tag successful experiment
git tag v1.0-qwen3-qlora-r16-cer8.5
```

### W&B Commands
```bash
# Login
wandb login

# View runs
wandb online

# Compare experiments
wandb dashboard well-log-ocr
```

---

## Appendix B: File Structure

```
well-log-ocr/
├── configs/
│   ├── base.py                          # Base training config
│   └── experiments/
│       ├── quick_val.py                 # Quick validation (5-10 steps)
│       ├── qwen3_qlora_r16.py          # Qwen3 QLoRA rank 16
│       └── qwen3_qlora_r32.py          # Qwen3 QLoRA rank 32
├── training/
│   ├── train.py                         # Main training script
│   ├── dataset.py                       # Dataset loading utilities
│   ├── metrics.py                       # CER/WER computation
│   └── checkpoint.py                    # Checkpoint management
├── scripts/
│   ├── setup_pod.sh                     # RunPod startup script
│   ├── preflight_check.py               # Pre-flight validation
│   └── notify.sh                        # Notification script
├── finetune_qwen3_vl_qlora.ipynb       # Original notebook (reference)
├── finetune_paddleocr_vl.ipynb         # Original notebook (reference)
├── requirements.txt                     # Python dependencies
├── CLAUDE.md                            # Project documentation
└── RUNPOD_TRAINING_SPEC.md              # This document
```

---

**Document Version:** 1.0
**Last Updated:** 2025-01-03
**Status:** Ready for Implementation
