.PHONY: setup download rotate crop preprocess annotate annotate-batch app clean help olmocr-download split-welllog train train-paddleocr train-qwen setup-pod resume download-from-runpod download-final-only inference

# Default settings
N ?= 500
WORKERS ?= 4
PYTHON ?= .venv/bin/python3
MODE ?= realtime
MODEL ?= gemini-3-flash-preview
DATASET ?= welllog-train

# Early stopping settings
TRAIN_LOSS_THRESHOLD ?= 0.2
EARLY_STOP_PATIENCE ?= 3

# Training config
CONFIG ?= configs/experiments/paddleocr_vl.py
RESUME ?=

# Directories (legacy)
DOWNLOADS_DIR ?= ./elog_downloads
HEADERS_DIR ?= ./cropped_headers
OUTPUT ?= ./well_log_header.jsonl

help:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║              OCR Training Pipeline Commands                  ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  Setup:"
	@echo "    make setup              Create venv and install dependencies"
	@echo ""
	@echo "  Welllog Preprocessing:"
	@echo "    make download           Download 500 files (default)"
	@echo "    make rotate             Rotate TIFFs 90° anticlockwise"
	@echo "    make crop               Crop headers (4:3 first page)"
	@echo "    make preprocess         Run all preprocessing steps"
	@echo "    make split-welllog      Split welllog into train/eval"
	@echo ""
	@echo "  olmOCR Dataset:"
	@echo "    make olmocr-download    Download olmOCR (50 per subset)"
	@echo ""
	@echo "  Annotation (DATASET=welllog-train|olmocr-train|...):"
	@echo "    make annotate           OCR with Gemini (real-time)"
	@echo "    make annotate-batch     OCR with Batch API (50% cheaper)"
	@echo ""
	@echo "  Training:"
	@echo "    make train              Train with default config (PaddleOCR-VL)"
	@echo "    make train CONFIG=configs/experiments/qwen3_qlora_r16.py"
	@echo "    make train RESUME=checkpoints/checkpoint-100"
	@echo ""
	@echo "    Early stopping: train_loss < $(TRAIN_LOSS_THRESHOLD) or"
	@echo "                   $(EARLY_STOP_PATIENCE) evals without improvement"
	@echo ""
	@echo "  Download from RunPod:"
	@echo "    make download-from-runpod RUNPOD_HOST=user@1.2.3.4"
	@echo "    make download-final-only RUNPOD_HOST=user@1.2.3.4"
	@echo ""
	@echo "    Optional: RUNPOD_SSH_KEY=~/.ssh/key RUNPOD_PORT=2222"
	@echo ""
	@echo "  Web App:"
	@echo "    make app                Start web app (port 8000)"
	@echo ""

# ═══════════════════════════════════════════════════════════════
# Setup
# ═══════════════════════════════════════════════════════════════

setup:
	uv venv .venv
	.venv/bin/python3 -m ensurepip
	.venv/bin/python3 -m pip install uv
	.venv/bin/python3 -m uv pip install google-genai python-dotenv pillow fastapi uvicorn datasets pymupdf requests

# ═══════════════════════════════════════════════════════════════
# Welllog Preprocessing
# ═══════════════════════════════════════════════════════════════

download:
	$(PYTHON) preprocess.py download --limit $(N) --workers $(WORKERS)

rotate:
	$(PYTHON) preprocess.py rotate --workers $(WORKERS)

crop:
	$(PYTHON) preprocess.py crop --workers $(WORKERS)

preprocess:
	$(PYTHON) preprocess.py all --limit $(N) --workers $(WORKERS)

split-welllog:
	$(PYTHON) preprocess.py split-welllog

# ═══════════════════════════════════════════════════════════════
# olmOCR Dataset
# ═══════════════════════════════════════════════════════════════

olmocr-download:
	$(PYTHON) preprocess.py olmocr --limit 50

# ═══════════════════════════════════════════════════════════════
# Annotation (supports DATASET variable)
# ═══════════════════════════════════════════════════════════════

annotate:
	$(PYTHON) annotate.py --dataset $(DATASET) --mode realtime --model $(MODEL) -w $(WORKERS)

annotate-batch:
	$(PYTHON) annotate.py --dataset $(DATASET) --mode batch --model $(MODEL) -w $(WORKERS)

annotate-interactive:
	$(PYTHON) annotate.py --dataset $(DATASET) --select-model

# ═══════════════════════════════════════════════════════════════
# Web App
# ═══════════════════════════════════════════════════════════════

app:
	$(PYTHON) servers/app.py

# ═══════════════════════════════════════════════════════════════
# Cleanup
# ═══════════════════════════════════════════════════════════════

clean:
	rm -rf $(DOWNLOADS_DIR) $(HEADERS_DIR)
	rm -f *.jsonl ocr_execution.log batch_input_temp.jsonl

# ═══════════════════════════════════════════════════════════════
# Training (RunPod)
# ═══════════════════════════════════════════════════════════════

train:
	python training/train.py --config $(CONFIG) $(if $(RESUME),--resume $(RESUME))

train-paddleocr:
	python training/train.py --config configs/experiments/paddleocr_vl.py

train-qwen:
	python training/train.py --config configs/experiments/qwen3_qlora_r16.py

resume:
	python training/train.py --config $(CONFIG) --resume $(RESUME)

# Inference
MODEL_PATH ?=
IMAGE ?=
NUM_SAMPLES ?= 5

inference:
	@if [ -z "$(MODEL_PATH)" ]; then \
		echo "Error: MODEL_PATH is required"; \
		echo "Usage: make inference MODEL_PATH=checkpoints/your-model"; \
		echo "       make inference MODEL_PATH=checkpoints/your-model IMAGE=path/to/image.png"; \
		echo "       make inference MODEL_PATH=checkpoints/your-model NUM_SAMPLES=10  # eval set"; \
		exit 1; \
	fi
	@if [ -n "$(IMAGE)" ]; then \
		python training/inference.py --model $(MODEL_PATH) --image $(IMAGE); \
	else \
		python training/inference.py --model $(MODEL_PATH) --eval --num-samples $(NUM_SAMPLES); \
	fi

# Download from RunPod
RUNPOD_HOST ?=
RUNPOD_SOURCE ?= checkpoints
RUNPOD_DEST ?= ./checkpoints
RUNPOD_SSH_KEY ?=
RUNPOD_PORT ?= 22

download-from-runpod:
	@if [ -z "$(RUNPOD_HOST)" ]; then \
		echo "Error: RUNPOD_HOST is required"; \
		echo "Usage: make download-from-runpod RUNPOD_HOST=user@1.2.3.4"; \
		exit 1; \
	fi
	python scripts/download_from_runpod.py $(RUNPOD_HOST) \
		--source $(RUNPOD_SOURCE) \
		--dest $(RUNPOD_DEST) \
		$(if $(RUNPOD_SSH_KEY),--ssh-key $(RUNPOD_SSH_KEY)) \
		--port $(RUNPOD_PORT)

download-final-only:
	@if [ -z "$(RUNPOD_HOST)" ]; then \
		echo "Error: RUNPOD_HOST is required"; \
		echo "Usage: make download-final-only RUNPOD_HOST=user@1.2.3.4"; \
		exit 1; \
	fi
	python scripts/download_from_runpod.py $(RUNPOD_HOST) \
		--source $(RUNPOD_SOURCE) \
		--dest $(RUNPOD_DEST) \
		$(if $(RUNPOD_SSH_KEY),--ssh-key $(RUNPOD_SSH_KEY)) \
		--port $(RUNPOD_PORT) \
		--final-only

setup-pod:
	pip install -r requirements.txt
	@if [ -n "$$WANDB_API_KEY" ]; then wandb login $$WANDB_API_KEY; fi
	@echo "Setup complete. Run 'make train' to start training."
	@echo ""
	@echo "Early stopping configured:"
	@echo "  - Stop when train_loss < $(TRAIN_LOSS_THRESHOLD)"
	@echo "  - Stop after $(EARLY_STOP_PATIENCE) evals without improvement"
