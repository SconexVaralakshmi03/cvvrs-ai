from __future__ import annotations

"""
api.py — Loco Pilot Distraction Detection  (DB-driven batch runner)
====================================================================

Videos are NOT uploaded through this API.
This service polls the database for video_files rows where
process_flag = 'N', downloads each video from S3, runs the detection
pipeline synchronously, and returns the final result directly.

Flag lifecycle
--------------
  N  →  pending   (set by the frontend when a video is uploaded to S3)
  I  →  in-progress (set here before the pipeline starts)
  Y  →  done      (set here only on successful pipeline completion)

If the pipeline crashes the flag stays at 'I' so operators can see
exactly which video failed.

Endpoints
---------
  GET  /         — health / welcome
  GET  /health   — {"status": "ok"}
  POST /trigger  — scan DB → download from S3 → run pipeline → return result

Response shape
--------------
{
  "batch_id":          "444e4af5ff5f",
  "total_videos":      2,
  "completed":         2,
  "failed":            0,
  "folders_processed": 1,
  "folders": [
    {
      "train_detail_id":  22801,
      "folder_name":      "22801-05-06-2026",
      "videos_in_folder": 2,
      "report": {
        "analysis_id":     "22801-05-06-2026",
        "train_detail_id": 22801,
        "processing_time": 108.124,
        "video_info":      [ ... ],   ← list with one entry per video
        "violations":      [ ... ]    ← all violations across ALL videos
      }
    }
  ]
}

Timestamp logic
---------------
For each folder group, videos are processed in seq_no order.
A running time_offset and frame_offset accumulate as each video finishes.

  global_timestamp  = local_video_time  + time_offset
  global_frame      = local_frame_index + frame_offset

In the report:
  "timestamp"                  = global_timestamp  (HH:MM:SS into the full recording)
  "original_video_timestamp"   = "<db_filename> <local_time>"  (time within that file)
"""

import json
import os
import tempfile
import traceback
import uuid
from typing import Any, Dict, List, Optional, Tuple

import cv2

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from main import PilotMonitoringPipeline
from utils.violation_store import ViolationStore
from utils.db_s3_uploader import (
    download_video_from_s3,
    get_pending_videos,
    set_process_flag,
)

# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title   = "Loco Pilot Distraction Detection API",
    version = "3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root() -> dict:
    return {
        "status":  "success",
        "message": "Loco Pilot Distraction Detection API is running",
        "health":  "/health",
        "docs":    "/docs",
        "trigger": "POST /trigger  — scan DB, process all pending videos, return results",
    }


@app.get("/health", tags=["status"])
def health() -> dict:
    return {"status": "ok"}


# ── Main trigger ───────────────────────────────────────────────────────────────

@app.post("/trigger", tags=["batch"])
def trigger_batch() -> JSONResponse:
    """
    Process ALL pending folders one at a time, in upload_timestamp ASC order.

    Flow per iteration
    ──────────────────
    1. get_pending_videos()  →  returns videos of the ONE oldest pending folder
                                (ORDER BY upload_timestamp ASC LIMIT 1 on folder,
                                 then all its videos by seq_no)
    2. Process that folder completely  (N → I → pipeline → Y)
    3. Loop again — because those rows are now 'Y', the next call to
       get_pending_videos() naturally returns the NEXT oldest folder
    4. Repeat until get_pending_videos() returns []  →  all done

    Result: one POST /trigger processes every pending folder,
    strictly one folder at a time, in upload order.
    """
    batch_id     = uuid.uuid4().hex[:12]
    total_videos = 0
    completed    = 0
    failed       = 0
    folders_out: List[Dict[str, Any]] = []
    folder_index = 0

    print(f"[Batch:{batch_id}] Starting — processing all pending folders one by one (upload_timestamp ASC).")

    while True:
        # ── Fetch the next oldest pending folder ─────────────────────────
        try:
            pending = get_pending_videos()   # always returns 1 folder or []
        except Exception as exc:
            raise HTTPException(
                status_code = 500,
                detail      = f"DB query failed (folder #{folder_index + 1}): {exc}",
            )

        if not pending:
            break   # no more pending folders

        folder_index    += 1
        folder_name      = pending[0]["folder_name"]
        train_detail_id  = pending[0]["train_detail_id"]
        n_videos         = len(pending)
        total_videos    += n_videos

        print(f"[Batch:{batch_id}] ── Folder {folder_index}: '{folder_name}'  "
              f"train={train_detail_id}  videos={n_videos} ──")

        # ── Process this folder completely ───────────────────────────────
        report, error, n_ok, n_fail = _process_folder_group(
            train_detail_id = train_detail_id,
            folder_name     = folder_name,
            rows            = pending,
        )
        completed += n_ok
        failed    += n_fail

        folder_entry: Dict[str, Any] = {
            "train_detail_id":  train_detail_id,
            "folder_name":      folder_name,
            "videos_in_folder": n_videos,
        }
        if report is not None:
            folder_entry["report"] = report
        if error:
            folder_entry["error"] = error

        folders_out.append(folder_entry)
        print(f"[Batch:{batch_id}] Folder '{folder_name}' done — checking for next pending folder...")

    # ── All done ─────────────────────────────────────────────────────────
    if not folders_out:
        return JSONResponse(content={
            "batch_id":          batch_id,
            "total_videos":      0,
            "completed":         0,
            "failed":            0,
            "folders_processed": 0,
            "folders":           [],
            "message":           "No pending videos found (process_flag = 'N').",
        })

    print(f"[Batch:{batch_id}] All folders done.  "
          f"folders={len(folders_out)}  videos={total_videos}  "
          f"completed={completed}  failed={failed}")

    return JSONResponse(content={
        "batch_id":          batch_id,
        "total_videos":      total_videos,
        "completed":         completed,
        "failed":            failed,
        "folders_processed": len(folders_out),
        "folders":           folders_out,
    })


# ── Per-folder-group processor ─────────────────────────────────────────────────

def _video_duration_seconds(path: str) -> float:
    """Read the duration of a video file using OpenCV."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0.0
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return total / fps if fps > 0 and total > 0 else 0.0


def _process_folder_group(
    train_detail_id: int,
    folder_name:     str,
    rows:            List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[str], int, int]:
    """
    Process one folder group — all videos that share the same
    (train_detail_id, folder_name), ordered by seq_no.

    Key design
    ──────────
    • One ViolationStore is created for the whole folder and shared
      across every pipeline run.  This means violations from all videos
      accumulate in a single store.
    • time_offset and frame_offset grow after each video so timestamps
      are continuous across the whole recording session.
    • finalize() is called ONCE after all videos are done.

    Returns (report_dict, error_str, n_completed, n_failed).
    """
    n_videos = len(rows)
    print(
        f"\n[Batch] ── Folder: train={train_detail_id}  "
        f"folder='{folder_name}'  ({n_videos} video(s)) ──"
    )

    analysis_id = folder_name

    # Step 1 — mark all rows in-progress
    for row in rows:
        try:
            set_process_flag(row["id"], "I")
        except Exception as exc:
            print(f"[Batch] Could not mark row {row['id']} as 'I': {exc}")

    # Step 2 — download every video from S3 to a temp file
    # We also record the DB filename alongside each temp path.
    tmp_entries: List[Tuple[str, str]] = []   # [(tmp_path, db_filename), ...]
    try:
        for row in rows:
            suffix = os.path.splitext(row["filename"] or "video.mp4")[1] or ".mp4"
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            os.close(tmp_fd)
            download_video_from_s3(row["s3_video_path"], tmp_path)
            tmp_entries.append((tmp_path, row["filename"] or ""))
    except Exception as exc:
        err = f"folder='{folder_name}' download failed: {exc}"
        print(f"[Batch] {err}")
        _cleanup_temps([p for p, _ in tmp_entries])
        return None, err, 0, n_videos

    # Step 3 — create ONE shared ViolationStore for the entire folder
    shared_vstore = ViolationStore(
        analysis_id     = analysis_id,
        train_detail_id = train_detail_id,
        # No video_info here — each pipeline run will call add_video_info()
    )

    # Step 4 — run the pipeline over each video in seq_no order,
    #           accumulating time_offset and frame_offset between videos.
    time_offset  = 0.0
    frame_offset = 0
    total_processing_time = 0.0

    try:
        for idx, (row, (tmp_path, db_filename)) in enumerate(zip(rows, tmp_entries)):
            print(
                f"[Batch]   [{idx + 1}/{n_videos}]  "
                f"{db_filename}  (seq={row['seq_no']})  "
                f"time_offset={time_offset:.2f}s  frame_offset={frame_offset}"
            )

            import time as _time
            t0 = _time.time()

            pipeline = PilotMonitoringPipeline(
                source          = tmp_path,
                analysis_id     = analysis_id,
                train_detail_id = train_detail_id,
                save            = False,
                display         = False,
                shared_vstore   = shared_vstore,
                time_offset     = time_offset,
                frame_offset    = frame_offset,
                source_filename = db_filename,
            )
            pipeline.run()   # returns "" in batch mode; vstore is NOT finalized here

            video_duration = _video_duration_seconds(tmp_path)
            video_frames   = _get_frame_count(tmp_path)

            total_processing_time += _time.time() - t0

            # Advance offsets for the next video
            time_offset  += video_duration
            frame_offset += video_frames

        # Step 5 — finalize the shared store ONCE with total processing time
        report_path = shared_vstore.finalize(processing_time=round(total_processing_time, 3))

        # Step 6 — mark all rows done
        for row in rows:
            try:
                set_process_flag(row["id"], "Y")
            except Exception as exc:
                print(f"[Batch] Could not mark row {row['id']} as 'Y': {exc}")

        # Step 7 — read and return the report
        if report_path and os.path.isfile(report_path):
            with open(report_path, encoding="utf-8") as f:
                return json.load(f), None, n_videos, 0
        else:
            err = f"folder='{folder_name}' no report file after finalize"
            print(f"[Batch] {err}")
            return None, err, 0, n_videos

    except Exception as exc:
        # flags stay at 'I' — intentional so operator can inspect
        err = (
            f"folder='{folder_name}' pipeline error: {exc}\n"
            + traceback.format_exc()
        )
        print(f"[Batch] {err}")
        return None, err, 0, n_videos

    finally:
        _cleanup_temps([p for p, _ in tmp_entries])


def _get_frame_count(path: str) -> int:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


# ── Temp file cleanup ──────────────────────────────────────────────────────────

def _cleanup_temps(paths: List[str]) -> None:
    for p in paths:
        try:
            if os.path.isfile(p):
                os.remove(p)
        except OSError:
            pass