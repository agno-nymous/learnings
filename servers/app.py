#!/usr/bin/env python3
"""OCR Pipeline Web App.

Single unified server providing:
- REST API endpoints for OCR control and data
- Frontend serving for the single-page application
- Dataset switching support

Uses dependency injection with FastAPI Depends for service access.
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv  # noqa: E402
from fastapi import BackgroundTasks, Depends, FastAPI, Query  # noqa: E402
from fastapi.responses import FileResponse, Response  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from core.image_utils import load_as_png_bytes  # noqa: E402
from servers.services import (  # noqa: E402
    DatasetService,
    JobService,
    get_dataset_service,
    get_job_service,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OCR Pipeline", description="OCR Training Data Generation")

PROJECT_ROOT = Path(__file__).parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"


# ═══════════════════════════════════════════════════════════════
# Request/Response Models
# ═══════════════════════════════════════════════════════════════


class OCRRequest(BaseModel):
    """Request model for running OCR annotation."""

    model: str
    mode: str = "realtime"
    dataset: str = "welllog"
    workers: int = 4


class DatasetActionRequest(BaseModel):
    """Request model for dataset actions."""

    dataset: str  # e.g., "olmocr", "welllog"
    limit: int = 50


# ═══════════════════════════════════════════════════════════════
# API Endpoints - Dataset Operations
# ═══════════════════════════════════════════════════════════════


@app.get("/api/datasets")
async def list_datasets(
    dataset_service: DatasetService = Depends(get_dataset_service),
):
    """List available datasets."""
    return dataset_service.list_datasets()


@app.get("/api/entries")
async def get_entries(
    dataset: str = Query("welllog"),
    dataset_service: DatasetService = Depends(get_dataset_service),
):
    """Get entries for a specific dataset."""
    return dataset_service.get_entries(dataset)


@app.get("/api/stats")
async def get_stats(
    dataset: str = Query("welllog"),
    dataset_service: DatasetService = Depends(get_dataset_service),
):
    """Get OCR statistics for a dataset."""
    return dataset_service.get_stats(dataset)


# ═══════════════════════════════════════════════════════════════
# API Endpoints - Job Operations
# ═══════════════════════════════════════════════════════════════


@app.get("/api/status")
async def get_status(
    job_service: JobService = Depends(get_job_service),
):
    """Get current job status."""
    return job_service.get_status()


@app.post("/api/run-ocr")
async def start_ocr(
    req: OCRRequest,
    background_tasks: BackgroundTasks,
    job_service: JobService = Depends(get_job_service),
):
    """Start an OCR job."""
    if job_service.is_running():
        return {"message": "A process is already running"}

    background_tasks.add_task(
        job_service.run_annotation,
        req.model,
        req.mode,
        req.dataset,
        req.workers,
    )
    return {"message": f"Started {req.mode} OCR on {req.dataset} ({req.workers} workers)"}


@app.get("/api/logs/execution")
async def get_execution_logs(
    job_service: JobService = Depends(get_job_service),
):
    """Get execution logs."""
    return {"logs": job_service.get_logs()}


# ═══════════════════════════════════════════════════════════════
# API Endpoints - Pipeline Actions
# ═══════════════════════════════════════════════════════════════


@app.post("/api/download")
async def download_dataset(
    req: DatasetActionRequest,
    background_tasks: BackgroundTasks,
    job_service: JobService = Depends(get_job_service),
):
    """Download data for the specified dataset type."""
    if job_service.is_running():
        return {"message": "A process is already running"}

    # Determine dataset type from the dataset name
    ds_type = req.dataset.split("-")[0]  # "welllog-train" -> "welllog"

    if ds_type == "olmocr":
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "preprocess.py"),
            "olmocr",
            "--limit",
            str(req.limit),
        ]
        name = f"Download olmOCR ({req.limit}/subset)"
    elif ds_type == "welllog":
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "preprocess.py"),
            "all",
            "--limit",
            str(req.limit),
        ]
        name = f"Download Welllog ({req.limit})"
    else:
        return {"message": f"Unknown dataset type: {ds_type}"}

    background_tasks.add_task(job_service.run_pipeline_command, name, cmd)
    return {"message": f"Started: {name}"}


@app.post("/api/split")
async def split_dataset(
    req: DatasetActionRequest,
    background_tasks: BackgroundTasks,
    job_service: JobService = Depends(get_job_service),
):
    """Split the specified dataset into train/eval."""
    if job_service.is_running():
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

    background_tasks.add_task(job_service.run_pipeline_command, name, cmd)
    return {"message": f"Started: {name}"}


# ═══════════════════════════════════════════════════════════════
# Image Serving
# ═══════════════════════════════════════════════════════════════


@app.get("/images/{filename}")
async def get_image(
    filename: str,
    dataset: str = Query("welllog"),
    dataset_service: DatasetService = Depends(get_dataset_service),
):
    """Serve image from the specified dataset."""
    images_dir = dataset_service.get_images_dir(dataset)
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
    """Return the main index.html page."""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/styles.css")
async def read_styles():
    """Return the styles.css file."""
    return FileResponse(FRONTEND_DIR / "styles.css")


@app.get("/app.js")
async def read_app_js():
    """Return the app.js file."""
    return FileResponse(FRONTEND_DIR / "app.js")


# ═══════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════


if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("  OCR PIPELINE")
    print("  → http://localhost:8000")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
