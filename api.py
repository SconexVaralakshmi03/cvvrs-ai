from __future__ import annotations

import json
import os
import shutil
import tempfile
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from main import GadgetDetectionPipeline

app = FastAPI(
    title="Loco Pilot Distraction Detection API",
    version="2.0.0",
)

# ✅ CORS must be registered immediately after app creation, before any routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_jobs: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=1)


# ── Routes ────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "success",
        "message": "Loco Pilot Distraction Detection API is running",
        "health": "/health",
        "docs": "/docs",
    }

# ✅ Only one /health definition
@app.get("/health", tags=["status"])
def health() -> dict:
    return {"status": "ok"}


def _run_pipeline(job_id: str, tmp_path: str, analysis_id: str, train_detail_id: int) -> None:
    _jobs[job_id]["status"] = "processing"
    try:
        pipeline = GadgetDetectionPipeline(
            source=tmp_path,
            analysis_id=analysis_id,
            train_detail_id=train_detail_id,
            save=False,
            display=False,
        )
        report_path = pipeline.run()

        if not report_path or not os.path.isfile(report_path):
            raise RuntimeError("Pipeline completed but report file was not created.")

        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)

        _jobs[job_id]["status"] = "done"
        _jobs[job_id]["result"] = report

    except Exception as exc:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = f"{exc}\n{traceback.format_exc()}"

    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.post("/analyze", tags=["analysis"])
async def analyze_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    video_id: Optional[str] = Form(default=None),
    train_detail_id: int = Form(default=0),
) -> JSONResponse:
    suffix = os.path.splitext(video.filename or "video.mp4")[1] or ".mp4"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)

    try:
        with os.fdopen(tmp_fd, "wb") as tmp_file:
            shutil.copyfileobj(video.file, tmp_file)
    except Exception as exc:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}")

    if video_id and video_id.strip():
        analysis_id = video_id.strip()
    else:
        stem = os.path.splitext(video.filename or "video")[0]
        analysis_id = stem or "analysis"

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued", "result": None, "error": None}
    _executor.submit(_run_pipeline, job_id, tmp_path, analysis_id, train_detail_id)

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": "queued",
            "message": f"Job accepted. Poll GET /status/{job_id} for progress.",
        },
    )


@app.get("/status/{job_id}", tags=["analysis"])
def job_status(job_id: str) -> JSONResponse:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    response: dict = {"job_id": job_id, "status": job["status"]}
    if job["status"] == "failed":
        response["error"] = job["error"]
    return JSONResponse(content=response)


@app.get("/result/{job_id}", tags=["analysis"])
def job_result(job_id: str) -> JSONResponse:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job["status"] == "failed":
        raise HTTPException(status_code=500, detail=job["error"])
    if job["status"] in ("queued", "processing"):
        raise HTTPException(status_code=409, detail=f"Job is still '{job['status']}'. Try again later.")
    result = _jobs.pop(job_id)["result"]
    return JSONResponse(content=result)