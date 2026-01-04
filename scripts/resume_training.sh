#!/bin/bash
# Resume training from latest checkpoint if available

set -e  # Exit on error

OUTPUT_DIR="${1:-/runpod_volume/checkpoints}"
CONFIG_PATH="${2:?CONFIG_PATH required}"

echo "Checking for existing checkpoints in $OUTPUT_DIR..."

# Find latest checkpoint
LATEST_CHECKPOINT=$(python -c "
from training.checkpoint import get_latest_checkpoint
import sys
chk = get_latest_checkpoint('$OUTPUT_DIR')
print(chk if chk else '')
")

if [ -n "$LATEST_CHECKPOINT" ] && [ -d "$LATEST_CHECKPOINT" ]; then
    echo "Resuming from checkpoint: $LATEST_CHECKPOINT"
    python training/train.py \
        --config "$CONFIG_PATH" \
        --resume "$LATEST_CHECKPOINT"
else
    echo "No checkpoint found, starting fresh training"
    python training/train.py --config "$CONFIG_PATH"
fi
