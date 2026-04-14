"""
api.py — FastAPI wrapper for the Loco Pilot Distraction Detection pipeline.

Endpoints
---------
POST /analyze
    Accepts a video file upload + optional form fields.
    Runs GadgetDetectionPipeline synchronously (blocking until done).
    Returns the analysis JSON report on completion.

Usage
-----
    uvicorn api:app --host 0.0.0.0 --port 8000

Request (multipart/form-data)
------------------------------
    video           : video file  (required)
    video_id        : str         (optional — used as analysis_id; defaults to filename stem)
    train_detail_id : int         (optional, default 0)

Response (200 OK)
------------------
    The full analysis_report.json produced by ViolationStore.finalize(),
    exactly matching the documented JSON schema.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import traceback
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

# ── Pipeline import ──────────────────────────────────────────────
# Assumes api.py lives alongside the project root (same level as
# config/, detector/, utils/).  Adjust sys.path if your layout
# differs.
from app import GadgetDetectionPipeline

# ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Loco Pilot Distraction Detection API",
    description=(
        "Upload a video to run the full distraction-detection pipeline "
        "(phone use, seat absence, drowsiness) and receive a structured "
        "JSON report with all violations."
    ),
    version="1.0.0",
)


# ── Health check ─────────────────────────────────────────────────

@app.get("/health", tags=["status"])
def health() -> dict:
    return {"status": "ok"}


# ── Main analysis endpoint ────────────────────────────────────────

@app.post("/analyze", tags=["analysis"])
async def analyze_video(
    video: UploadFile = File(..., description="Video file to analyse"),
    video_id: Optional[str] = Form(
        default=None,
        description="Analysis ID / video identifier. Defaults to the filename stem.",
    ),
    train_detail_id: int = Form(
        default=0,
        description="Train detail ID stored alongside results in the database.",
    ),
) -> JSONResponse:
    """
    Run the full distraction-detection pipeline on the uploaded video.

    - Saves the upload to a temporary file.
    - Runs GadgetDetectionPipeline (phone · absence · drowsiness).
    - Returns the finalized JSON report (also written to
      outputs/<analysis_id>/analysis_report.json and uploaded to S3/DB).
    - Cleans up the temporary file after processing.
    """

    # ── 1. Persist the upload to a temp file ─────────────────────
    suffix = os.path.splitext(video.filename or "video.mp4")[1] or ".mp4"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)

    try:
        # Write upload to disk
        with os.fdopen(tmp_fd, "wb") as tmp_file:
            shutil.copyfileobj(video.file, tmp_file)

        # ── 2. Determine analysis_id ──────────────────────────────
        if video_id and video_id.strip():
            analysis_id = video_id.strip()
        else:
            stem        = os.path.splitext(video.filename or "video")[0]
            analysis_id = stem or "analysis"

        # ── 3. Run the pipeline ───────────────────────────────────
        pipeline = GadgetDetectionPipeline(
            source          = tmp_path,
            analysis_id     = analysis_id,
            train_detail_id = train_detail_id,
            save            = False,   # no annotated output video
            display         = False,   # headless — no GUI
        )

        try:
            report_path = pipeline.run()
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline error: {exc}\n{traceback.format_exc()}",
            )

        # ── 4. Read and return the JSON report ────────────────────
        if not report_path or not os.path.isfile(report_path):
            raise HTTPException(
                status_code=500,
                detail="Pipeline completed but report file was not created.",
            )

        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)

        return JSONResponse(content=report)

    finally:
        # Always clean up the temporary upload file
        try:
            os.remove(tmp_path)
        except OSError:
            pass