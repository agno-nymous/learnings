#!/usr/bin/env python3
"""
OCR Pipeline Web App.

Single unified server providing:
- REST API endpoints for OCR control and data
- Frontend serving for the single-page application
- Dataset switching support
"""

import json
import os
import subprocess
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, BackgroundTasks, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

from core.jsonl_utils import read_entries
from core.image_utils import load_as_png_bytes
from core.config import DATASETS, get_dataset

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OCR Pipeline", description="OCR Training Data Generation")

PROJECT_ROOT = Path(__file__).parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
LOG_FILE = PROJECT_ROOT / "ocr_execution.log"


class ProcessStatus:
    def __init__(self):
        self.running = False
        self.type = None
        self.dataset = None
        self.process = None


status = ProcessStatus()


class OCRRequest(BaseModel):
    model: str
    mode: str = "realtime"
    dataset: str = "welllog"
    workers: int = 4


def run_annotation(model: str, mode: str, dataset: str, workers: int = 4):
    """Run annotation using the unified CLI."""
    global status
    status.running = True
    status.type = mode
    status.dataset = dataset

    try:
        cmd = [
            sys.executable, str(PROJECT_ROOT / "annotate.py"),
            "--dataset", dataset,
            "--mode", mode,
            "--model", model,
            "--workers", str(workers),
        ]
        logger.info(f"Starting OCR: {' '.join(cmd)}")

        with open(LOG_FILE, "a") as f:
            f.write(f"\n--- Starting {mode} OCR on {dataset} with {model} ---\n")
            f.flush()
            process = subprocess.Popen(
                cmd, stdout=f, stderr=subprocess.STDOUT,
                text=True, cwd=str(PROJECT_ROOT),
            )
            status.process = process
            process.wait()
            f.write(f"--- Finished {mode} OCR on {dataset} ---\n")
            f.flush()
    except Exception as e:
        logger.error(f"Failed to run: {e}")
        with open(LOG_FILE, "a") as f:
            f.write(f"ERROR: {e}\n")
    finally:
        status.running = False
        status.type = None
        status.dataset = None
        status.process = None


# ═══════════════════════════════════════════════════════════════
# API Endpoints
# ═══════════════════════════════════════════════════════════════

@app.get("/api/datasets")
async def list_datasets():
    """List available datasets."""
    return {
        "datasets": list(DATASETS.keys()),
        "default": "welllog",
    }


@app.get("/api/entries")
async def get_entries(dataset: str = Query("welllog")):
    """Get entries for a specific dataset."""
    try:
        ds = get_dataset(dataset)
    except ValueError as e:
        return {"error": str(e)}
    
    output_path = PROJECT_ROOT / ds["output"]
    clean_path = output_path.with_suffix(".clean.jsonl")
    
    if clean_path.exists():
        return list(read_entries(clean_path))
    if output_path.exists():
        return list(read_entries(output_path))
    return []


@app.get("/api/stats")
async def get_stats(dataset: str = Query("welllog")):
    """Get OCR statistics for a dataset."""
    try:
        ds = get_dataset(dataset)
    except ValueError as e:
        return {"error": str(e)}
    
    output_path = PROJECT_ROOT / ds["output"]
    if not output_path.exists():
        return {"total": 0, "success": 0, "error": 0}
    
    total, success, error = 0, 0, 0
    for entry in read_entries(output_path):
        total += 1
        if entry.get("status") == "success":
            success += 1
        else:
            error += 1
    return {"total": total, "success": success, "error": error}


@app.get("/api/status")
async def get_status():
    """Get current job status."""
    return {
        "running": status.running,
        "type": status.type,
        "dataset": status.dataset,
    }


@app.post("/api/run-ocr")
async def start_ocr(req: OCRRequest, background_tasks: BackgroundTasks):
    """Start an OCR job."""
    if status.running:
        return {"message": "A process is already running"}

    background_tasks.add_task(run_annotation, req.model, req.mode, req.dataset, req.workers)
    return {"message": f"Started {req.mode} OCR on {req.dataset} ({req.workers} workers)"}


@app.get("/api/logs/execution")
async def get_execution_logs():
    """Get execution logs."""
    if LOG_FILE.exists():
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
            return {"logs": "".join(lines[-200:])}
    return {"logs": "No execution logs yet."}


# ═══════════════════════════════════════════════════════════════
# Pipeline Actions (unified - work on any dataset)
# ═══════════════════════════════════════════════════════════════

def run_pipeline_command(name: str, cmd: list):
    """Run a pipeline command in background."""
    global status
    status.running = True
    status.type = name
    status.dataset = None

    try:
        logger.info(f"Running {name}: {' '.join(cmd)}")
        with open(LOG_FILE, "a") as f:
            f.write(f"\n--- Starting {name} ---\n")
            f.flush()
            process = subprocess.Popen(
                cmd, stdout=f, stderr=subprocess.STDOUT,
                text=True, cwd=str(PROJECT_ROOT),
            )
            status.process = process
            process.wait()
            f.write(f"--- Finished {name} ---\n")
            f.flush()
    except Exception as e:
        logger.error(f"Failed: {e}")
        with open(LOG_FILE, "a") as f:
            f.write(f"ERROR: {e}\n")
    finally:
        status.running = False
        status.type = None
        status.process = None


class DatasetActionRequest(BaseModel):
    dataset: str  # e.g., "olmocr", "welllog"
    limit: int = 50


@app.post("/api/download")
async def download_dataset(req: DatasetActionRequest, background_tasks: BackgroundTasks):
    """Download data for the specified dataset type."""
    if status.running:
        return {"message": "A process is already running"}
    
    # Determine dataset type from the dataset name
    ds_type = req.dataset.split("-")[0]  # "welllog-train" -> "welllog"
    
    if ds_type == "olmocr":
        cmd = [sys.executable, str(PROJECT_ROOT / "preprocess.py"), "olmocr", "--limit", str(req.limit)]
        name = f"Download olmOCR ({req.limit}/subset)"
    elif ds_type == "welllog":
        cmd = [sys.executable, str(PROJECT_ROOT / "preprocess.py"), "all", "--limit", str(req.limit)]
        name = f"Download Welllog ({req.limit})"
    else:
        return {"message": f"Unknown dataset type: {ds_type}"}
    
    background_tasks.add_task(run_pipeline_command, name, cmd)
    return {"message": f"Started: {name}"}


@app.post("/api/split")
async def split_dataset(req: DatasetActionRequest, background_tasks: BackgroundTasks):
    """Split the specified dataset into train/eval."""
    if status.running:
        return {"message": "A process is already running"}
    
    ds_type = req.dataset.split("-")[0]
    
    if ds_type == "welllog":
        cmd = [sys.executable, str(PROJECT_ROOT / "preprocess.py"), "split-welllog"]
        name = "Split Welllog"
    elif ds_type == "olmocr":
        # olmOCR is already split during download
        return {"message": "olmOCR is automatically split during download"}
    else:
        return {"message": f"Unknown dataset type: {ds_type}"}
    
    background_tasks.add_task(run_pipeline_command, name, cmd)
    return {"message": f"Started: {name}"}


# ═══════════════════════════════════════════════════════════════
# Image Serving (supports dataset parameter)
# ═══════════════════════════════════════════════════════════════

@app.get("/images/{filename}")
async def get_image(filename: str, dataset: str = Query("welllog")):
    """Serve image from the specified dataset."""
    try:
        ds = get_dataset(dataset)
    except ValueError:
        ds = {"images_dir": Path("./cropped_headers")}  # Fallback
    
    images_dir = PROJECT_ROOT / ds["images_dir"]
    safe_name = os.path.basename(filename)
    image_path = images_dir / safe_name

    if not image_path.exists():
        return {"error": "Image not found"}

    ext = image_path.suffix.lower()
    if ext in (".tif", ".tiff"):
        png_data = load_as_png_bytes(image_path)
        return Response(content=png_data, media_type="image/png")

    return FileResponse(image_path)


# ═══════════════════════════════════════════════════════════════
# Frontend Serving
# ═══════════════════════════════════════════════════════════════

@app.get("/")
async def read_index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/styles.css")
async def read_styles():
    return FileResponse(FRONTEND_DIR / "styles.css")


@app.get("/app.js")
async def read_app_js():
    return FileResponse(FRONTEND_DIR / "app.js")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("  OCR PIPELINE")
    print("  → http://localhost:8000")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)

