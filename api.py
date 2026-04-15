from __future__ import annotations

import json
import os
import shutil
import tempfile
import traceback
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Pipeline import ──────────────────────────────────────────────
from main import GadgetDetectionPipeline

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

# ── CORS ─────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health check ─────────────────────────────────────────────────
@app.get("/health", tags=["status"])
def health() -> dict:
    return {"status": "ok"}


# ── Main analysis endpoint ────────────────────────────────────────
@app.post("/analyze", tags=["analysis"])
async def analyze_video(
    video: UploadFile = File(..., description="Video file to analyse"),
    video_id: Optional[str] = Form(default=None),
    train_detail_id: int = Form(default=0),
) -> JSONResponse:

    suffix = os.path.splitext(video.filename or "video.mp4")[1] or ".mp4"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)

    try:
        # Save uploaded file
        with os.fdopen(tmp_fd, "wb") as tmp_file:
            shutil.copyfileobj(video.file, tmp_file)

        # Generate analysis ID
        if video_id and video_id.strip():
            analysis_id = video_id.strip()
        else:
            stem = os.path.splitext(video.filename or "video")[0]
            analysis_id = stem or "analysis"

        # Run pipeline
        pipeline = GadgetDetectionPipeline(
            source=tmp_path,
            analysis_id=analysis_id,
            train_detail_id=train_detail_id,
            save=False,
            display=False,
        )

        try:
            report_path = pipeline.run()
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline error: {exc}\n{traceback.format_exc()}",
            )

        # Validate report
        if not report_path or not os.path.isfile(report_path):
            raise HTTPException(
                status_code=500,
                detail="Pipeline completed but report file was not created.",
            )

        # Read JSON report
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)

        return JSONResponse(content=report)

    finally:
        # Cleanup temp file
        try:
            os.remove(tmp_path)
        except OSError:
            pass
