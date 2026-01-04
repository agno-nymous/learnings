.PHONY: setup download rotate crop preprocess annotate annotate-batch app clean help olmocr-download split-welllog

# Default settings
N ?= 500
WORKERS ?= 4
PYTHON ?= .venv/bin/python3
MODE ?= realtime
MODEL ?= gemini-3-flash-preview
DATASET ?= welllog-train

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

