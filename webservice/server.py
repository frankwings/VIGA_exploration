"""
SAM3D Web Service — FastAPI backend.

Endpoints:
  POST /api/upload          — upload image, get job_id
  POST /api/segment/{id}    — run SAM segmentation
  POST /api/reconstruct/{id}/{mask_id}?trellis=1|2  — run TRELLIS on one mask
  GET  /api/jobs/{id}/status — job status + mask metadata
  GET  /api/files/{id}/{path} — serve output files
  GET  /                    — serve frontend
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from sam3d_service import SAM3DService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
JOBS_DIR = HERE / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="SAM3D Web Service")
app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

service = SAM3DService(project_root=PROJECT_ROOT, jobs_dir=JOBS_DIR)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    return (HERE / "static" / "index.html").read_text(encoding="utf-8")


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """Accept an image upload, return a job_id."""
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename).suffix or ".png"
    image_path = job_dir / f"input{ext}"
    data = await file.read()
    image_path.write_bytes(data)

    # Save job metadata
    meta = {"job_id": job_id, "image": str(image_path), "status": "uploaded", "masks": []}
    (job_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log.info("Job %s created — %s (%d bytes)", job_id, file.filename, len(data))
    return {"job_id": job_id, "status": "uploaded"}


@app.post("/api/segment/{job_id}")
async def segment_image(job_id: str):
    """Run SAM segmentation on the uploaded image."""
    job_dir = JOBS_DIR / job_id
    meta_path = job_dir / "meta.json"
    if not meta_path.exists():
        raise HTTPException(404, f"Job {job_id} not found")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["status"] == "segmenting":
        raise HTTPException(409, "Segmentation already in progress")

    meta["status"] = "segmenting"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    try:
        result = await asyncio.to_thread(
            service.segment, job_id, meta["image"]
        )
        meta["status"] = "segmented"
        meta["masks"] = result["masks"]
        meta["overlay"] = result["overlay"]
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return {"status": "segmented", "masks": result["masks"], "overlay": f"/api/files/{job_id}/overlay.png"}
    except Exception as e:
        meta["status"] = "error"
        meta["error"] = str(e)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        log.exception("Segmentation failed for %s", job_id)
        raise HTTPException(500, str(e))


@app.post("/api/reconstruct/{job_id}/{mask_id}")
async def reconstruct_mask(job_id: str, mask_id: int, trellis: int = Query(1, ge=1, le=2)):
    """Run TRELLIS reconstruction on a specific mask."""
    job_dir = JOBS_DIR / job_id
    meta_path = job_dir / "meta.json"
    if not meta_path.exists():
        raise HTTPException(404, f"Job {job_id} not found")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if mask_id >= len(meta["masks"]):
        raise HTTPException(404, f"Mask {mask_id} not found (have {len(meta['masks'])} masks)")

    mask_info = meta["masks"][mask_id]
    if mask_info.get("glb"):
        # Already reconstructed — just return it
        return {"status": "done", "glb": f"/api/files/{job_id}/{Path(mask_info['glb']).name}"}

    mask_info["reconstruct_status"] = "running"
    mask_info["trellis_version"] = trellis
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    try:
        glb_path = await asyncio.to_thread(
            service.reconstruct, job_id, mask_id, meta["image"], mask_info, trellis
        )
        mask_info["glb"] = str(glb_path)
        mask_info["reconstruct_status"] = "done"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return {"status": "done", "glb": f"/api/files/{job_id}/{Path(glb_path).name}"}
    except Exception as e:
        mask_info["reconstruct_status"] = "error"
        mask_info["error"] = str(e)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        log.exception("Reconstruction failed for %s mask %d", job_id, mask_id)
        raise HTTPException(500, str(e))


@app.get("/api/jobs/{job_id}/status")
async def job_status(job_id: str):
    meta_path = JOBS_DIR / job_id / "meta.json"
    if not meta_path.exists():
        raise HTTPException(404, f"Job {job_id} not found")
    return JSONResponse(json.loads(meta_path.read_text(encoding="utf-8")))


@app.get("/api/files/{job_id}/{filename:path}")
async def serve_file(job_id: str, filename: str):
    file_path = JOBS_DIR / job_id / filename
    if not file_path.exists():
        raise HTTPException(404, f"File not found: {filename}")
    return FileResponse(str(file_path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
