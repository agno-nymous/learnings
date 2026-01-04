# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Well Log OCR Processing and Analysis Project - processes scanned well log header images using Google Gemini OCR, manages output data, and provides a web-based viewer for analyzing results.

## Commands

Run `make help` to see all available commands.

### Setup
- `make setup` - Create virtual environment and install dependencies

### Preprocessing Pipeline (Welllog)
- `make download N=500` - Download elog scan images (default: 500)
- `make rotate` - Rotate TIFF images 90° anticlockwise
- `make crop` - Crop headers from images (4:3 ratio from first page)
- `make preprocess` - Run all preprocessing steps (download → rotate → crop)
- `make split-welllog` - Split welllog data into train/eval sets (80/20)

### olmOCR Dataset
- `make olmocr-download` - Download olmOCR dataset from HuggingFace (default: 50 per subset)

### OCR Annotation
- `make annotate DATASET=welllog-train` - Run OCR with real-time API
- `make annotate-batch DATASET=olmocr-train` - Run OCR with Batch API (50% cheaper)
- `make annotate-interactive DATASET=welllog-train` - Interactive model selection

### Web Interface
- `make app` - Start FastAPI server on port 8000 (serves frontend dashboard)

### Variables
- `N` - Number of files to download (default: 500)
- `WORKERS` - Parallel workers (default: 4)
- `MODEL` - Gemini model (default: gemini-3-flash-preview)
- `DATASET` - Dataset to process (default: welllog-train)

### Cleanup
- `make clean` - Remove downloads, headers, and output files

### Pre-commit Hooks
- `pre-commit install` - Install git pre-commit hooks (run once after setup)
- `pre-commit run --all-files` - Run all hooks manually on all files
- `pre-commit autoupdate` - Update hook versions

**Important**: When installing Python packages, always use `uv`:
```bash
source .venv/bin/activate
python3 -m uv pip install <package>
```

## Architecture

### Dataset Registry (`core/config.py`)

The project uses a centralized dataset registry. To add a new dataset, update `DATASETS` in `core/config.py`:

```python
DATASETS = {
    "my-dataset": {
        "images_dir": Path("./datasets/my-dataset"),
        "output": Path("./datasets/my-dataset/output.jsonl"),
    },
}
```

Available datasets: `welllog`, `welllog-train`, `welllog-eval`, `olmocr-train`, `olmocr-eval`

### Annotation System (`annotator/`)

Factory pattern for creating OCR annotators via `AnnotatorFactory.create()`:
- **Real-time mode**: `GeminiAnnotator` - Standard Gemini API (faster, good for small batches)
- **Batch mode**: `GeminiBatchAnnotator` - Gemini Batch API (50% cheaper, async job polling)

The base `Annotator` interface defines `annotate(images, output)` method.

### CLI Entry Points

- **`annotate.py`** - Unified OCR annotation CLI with dataset registry support
- **`preprocess.py`** - Preprocessing pipeline (download, rotate, crop, split, olmocr)

### Web Server (`servers/app.py`)

FastAPI server with:
- REST API endpoints for OCR control, dataset stats, and data retrieval
- Background task execution for long-running OCR jobs
- Frontend serving (single-page app from `frontend/`)

Key endpoints: `/api/datasets`, `/api/stats`, `/api/run-ocr`, `/images/{filename}`

### Frontend Interfaces

- **`frontend/`** - Main dashboard for dataset selection, OCR job control, and monitoring
- **`viewer/`** - Side-by-side image/markdown comparison viewer for OCR results

### Core Utilities (`core/`)

- **`config.py`** - Centralized configuration (OCR instruction, models, dataset registry)
- **`jsonl_utils.py`** - JSONL file reading/writing utilities
- **`image_utils.py`** - Image processing (TIFF to PNG conversion, etc.)
- **`file_discovery.py`** - File discovery tools for finding images

### Data Flow

1. **Download** → Source URLs → `elog_downloads/` (or olmOCR → `datasets/olmocr/`)
2. **Preprocess** → Rotate + Crop → `cropped_headers/` (legacy) or `datasets/welllog/`
3. **Split** → Organize into train/eval sets
4. **Annotate** → Gemini OCR → JSONL output files
5. **View** → Web interface displays results

## Environment

Requires `.env` file with:
- `GOOGLE_API_KEY` - Get from https://aistudio.google.com/app/apikey
- `HF_TOKEN` - HuggingFace token for dataset uploads

## Tech Stack

- **Python 3.13** with `uv` package manager
- **FastAPI** for web server
- **Google Gemini API** for OCR
- **Pillow** for image processing
- **PyMuPDF (fitz)** for PDF operations
- **JSONL** format for dataset storage
- **Vanilla JavaScript** (no frameworks) for frontend

## Key Patterns

- **Dataset-first**: All operations use the dataset registry for paths
- **Factory pattern**: `AnnotatorFactory.create(mode, model, workers)` for annotator instantiation
- **Resume by default**: `annotate.py` skips already-processed files (use `--no-resume` to override)
- **Clean datasets**: Automatically creates `.clean.jsonl` files with only successful entries
