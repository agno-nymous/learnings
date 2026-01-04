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
