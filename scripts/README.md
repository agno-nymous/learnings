# RunPod Setup Instructions

This guide explains how to configure and launch RunPod pods for automated training.

## 1. Create Network Volume

First, create a network volume to store checkpoints and logs:

```bash
runpodctl create volume well-log-ocr-checkpoints --size 100
```

This creates a 100GB volume. Adjust size based on your needs.

## 2. Configure Environment Variables

In your RunPod pod template, set these environment variables:

### Required Variables

- **`GIT_REPO`**: Your git repository URL
  - Example: `https://github.com/your-org/well-log-ocr.git`
  - Or: `git@github.com:your-org/well-log-ocr.git`

- **`GIT_BRANCH`**: Experiment branch name
  - Example: `feature/experiment-qwen3-qlora`
  - Example: `main`

- **`CONFIG_PATH`**: Config file path (relative to repo root)
  - Example: `configs/experiments/qwen3_qlora_r16.py`
  - Example: `configs/experiments/quick_val.py`

- **`WANDB_API_KEY`**: Your Weights & Biases API key
  - Get from: https://wandb.ai/settings
  - Format: `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX`

### Optional Variables

- **`NETWORK_VOLUME`**: Mount path for network volume
  - Default: `/runpod_volume`
  - Usually no need to change this

- **`EXPERIMENT_NAME`**: Custom name for log files
  - Auto-generated if not set (format: `training-YYYYMMDD-HHMMSS`)
  - Example: `qwen3-qlora-r16-welllog`

## 3. Configure Storage Mount

Mount your network volume to the pod:

**Mount Point**: `/runpod_volume`
**Volume**: `well-log-ocr-checkpoints` (or whatever you named it)

This ensures:
- Checkpoints are saved persistently
- Logs are stored safely
- Training can resume after spot interruptions

## 4. Set Startup Script

Use `scripts/setup_pod.sh` as the pod startup script.

The script will:
1. Validate required environment variables
2. Install `uv` package manager if needed
3. Clone or update your git repository
4. Install Python dependencies
5. Setup network volume directories
6. Login to Weights & Biases
7. Start training with logging

## 5. Launch Pod

### Using runpodctl CLI

```bash
runpodctl create pod \
  --name "well-log-ocr-training" \
  --gpus "RTX_4090:1" \
  --volume-name well-log-ocr-checkpoints:/runpod_volume \
  --spot \
  --spot-interruption="auto-restart" \
  --container-name ghcr.io/huggingface/unsloth \
  --env "GIT_REPO=https://github.com/your-org/well-log-ocr.git" \
  --env "GIT_BRANCH=feature/experiment-qwen3-qlora" \
  --env "CONFIG_PATH=configs/experiments/qwen3_qlora_r16.py" \
  --env "WANDB_API_KEY=YOUR_WANDB_API_KEY_HERE" \  # pragma: allowlist secret
  --env "EXPERIMENT_NAME=qwen3-qlora-r16"
```

### Using RunPod Web UI

1. Go to https://www.runpod.io/console/pods
2. Click "Deploy" → "Custom Pod"
3. Configure:
   - **Container Image**: `ghcr.io/huggingface/unsloth`
   - **GPU**: Select your desired GPU (e.g., RTX 4090, A100 80GB)
   - **Volume Mount**: `/runpod_volume` → `well-log-ocr-checkpoints`
   - **Spot Instance**: Enabled (for cost savings)
   - **Interruption**: Auto-restart
4. Add environment variables (see section 2)
5. Set startup script to the contents of `scripts/setup_pod.sh`
6. Deploy!

## 6. Monitor Training

### Check Logs

SSH into the pod and view logs:

```bash
# View latest log
tail -f /runpod_volume/logs/training-*.log

# Or use journalctl if using systemd
journalctl -u runpod-startup -f
```

### Monitor W&B

Training metrics are automatically logged to Weights & Biases:
- URL: https://wandb.ai/your-entity/well-log-ocr
- Real-time loss, learning rate, and GPU metrics
- Sample predictions during evaluation

### Check Checkpoints

```bash
ls -lh /runpod_volume/checkpoints/
```

## 7. Resume Training

If training is interrupted (spot preemption or manual stop):

1. Checkpoint is automatically saved to `/runpod_volume/checkpoints/`
2. Pod auto-restarts (if spot with auto-restart enabled)
3. Training resumes from last checkpoint

Manual resume:
```bash
python training/train.py \
  --config configs/experiments/qwen3_qlora_r16.py \
  --resume /runpod_volume/checkpoints/checkpoint-500
```

## Troubleshooting

### Pod fails to start

Check environment variables are set correctly:
```bash
echo $GIT_REPO
echo $GIT_BRANCH
echo $CONFIG_PATH
echo $WANDB_API_KEY
```

### Training fails immediately

1. Check logs: `cat /runpod_volume/logs/*.log`
2. Verify config file exists in repo
3. Ensure W&B API key is valid
4. Check dataset paths are correct

### Out of memory errors

- Reduce `per_device_train_batch_size` in config
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Use a GPU with more VRAM

### Slow training

- Increase `per_device_train_batch_size` if memory allows
- Reduce `logging_steps` and `eval_steps` for less frequent evaluation
- Use a faster GPU (e.g., A100 instead of RTX 4090)

## Cost Optimization Tips

1. **Use Spot Instances**: 50-80% cheaper than on-demand
2. **Enable Auto-Restart**: Automatically resumes after preemption
3. **Network Volume**: Persistent storage avoids data loss
4. **Right-size GPU**: Don't over-provision; test locally first
5. **Monitor Training**: Stop early if converged to save costs

## Example Configurations

### Quick Test (RTX 4090, 24GB)

```bash
--env "CONFIG_PATH=configs/experiments/quick_val.py"
```

- ~5 minutes
- Validates pipeline
- Good for debugging

### Full Training (A100 80GB)

```bash
--env "CONFIG_PATH=configs/experiments/qwen3_qlora_r16.py"
```

- ~2-4 hours
- Full dataset
- Production model
