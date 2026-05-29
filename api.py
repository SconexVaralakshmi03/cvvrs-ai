
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

# ── Pipeline import ──────────────────────────────────────────────
from main import GadgetDetectionPipeline
# ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Loco Pilot Distraction Detection API",
    description=(
        "Upload a video to run the full distraction-detection pipeline "
        "(phone use, seat absence, drowsiness) and receive a structured "
        "JSON report with all violations.\n\n"
        "**For large videos:** POST to `/analyze` returns a `job_id` immediately. "
        "Poll `GET /status/{job_id}` until `status` is `done` or `failed`, "
        "then fetch the result from `GET /result/{job_id}`."
    ),
    version="2.0.0",
)

# ── CORS ─────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job store ───────────────────────────────────────────
# Structure: { job_id: { "status": "queued"|"processing"|"done"|"failed",
#                        "result": dict | None,
#                        "error":  str  | None } }
_jobs: dict[str, dict] = {}

# Single-threaded executor so the GPU isn't thrashed by concurrent jobs
_executor = ThreadPoolExecutor(max_workers=1)


# ── Background worker ─────────────────────────────────────────────
def _run_pipeline(
    job_id: str,
    tmp_path: str,
    analysis_id: str,
    train_detail_id: int,
) -> None:
    """Runs in a thread-pool thread; never raises — stores result/error in _jobs."""
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
        # Always clean up the temp video file
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ── Health check ──────────────────────────────────────────────────
@app.get("/health", tags=["status"])
def health() -> dict:
    return {"status": "ok"}


# ── Submit job ────────────────────────────────────────────────────
@app.post("/analyze", tags=["analysis"])
async def analyze_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(..., description="Video file to analyse"),
    video_id: Optional[str] = Form(default=None),
    train_detail_id: int = Form(default=0),
) -> JSONResponse:
    """
    Accepts a video upload and immediately returns a `job_id`.
    The pipeline runs in the background — poll `/status/{job_id}` for progress.
    """
    suffix = os.path.splitext(video.filename or "video.mp4")[1] or ".mp4"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)

    # Save uploaded file to disk before the request closes
    try:
        with os.fdopen(tmp_fd, "wb") as tmp_file:
            shutil.copyfileobj(video.file, tmp_file)
    except Exception as exc:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}")

    # Build analysis_id
    if video_id and video_id.strip():
        analysis_id = video_id.strip()
    else:
        stem = os.path.splitext(video.filename or "video")[0]
        analysis_id = stem or "analysis"

    # Register job
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued", "result": None, "error": None}

    # Submit to thread pool (non-blocking)
    _executor.submit(_run_pipeline, job_id, tmp_path, analysis_id, train_detail_id)

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": "queued",
            "message": (
                f"Job accepted. Poll GET /status/{job_id} for progress, "
                f"then GET /result/{job_id} when done."
            ),
        },
    )


# ── Poll status ───────────────────────────────────────────────────
@app.get("/status/{job_id}", tags=["analysis"])
def job_status(job_id: str) -> JSONResponse:
    """
    Returns `queued`, `processing`, `done`, or `failed`.
    Check this endpoint until status is `done` or `failed`.
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    response: dict = {"job_id": job_id, "status": job["status"]}
    if job["status"] == "failed":
        response["error"] = job["error"]
    return JSONResponse(content=response)


# ── Fetch result ──────────────────────────────────────────────────
@app.get("/result/{job_id}", tags=["analysis"])
def job_result(job_id: str) -> JSONResponse:
    """
    Returns the full JSON report once the job is `done`.
    Returns 404 if job not found, 409 if still processing.
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if job["status"] == "failed":
        raise HTTPException(status_code=500, detail=job["error"])

    if job["status"] in ("queued", "processing"):
        raise HTTPException(
            status_code=409,
            detail=f"Job is still '{job['status']}'. Try again later.",
        )

    # Clean up job from memory after retrieval (optional — comment out to keep history)
    result = _jobs.pop(job_id)["result"]
    return JSONResponse(content=result)
