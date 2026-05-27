"""
api.py — FastAPI wrapper for the Loco Pilot Distraction Detection pipeline.

POST /analyze-batch
    Upload N videos (count decided by user on frontend).
    Processes them SEQUENTIALLY. Returns combined JSON report.

POST /analyze-batch-async
    Same. Returns batch_id immediately, processes in background.

GET  /batch-status/{batch_id}   — live progress
GET  /batch-result/{batch_id}   — full report when done
GET  /test                      — simple HTML test page (no Swagger needed)

Usage
-----
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import traceback
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from main import GadgetDetectionPipeline
from utils.violation_store import ViolationStore

app = FastAPI(
    title="Loco Pilot Distraction Detection API",
    version="2.0.0",
    description=(
        "Upload N videos (count entered by user). "
        "Videos are always processed sequentially — video 2 starts only after video 1 finishes."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_batch_jobs: Dict[str, "BatchJob"] = {}
_batch_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────
# BATCH JOB STATE
# ─────────────────────────────────────────────────────────────────

class BatchJob:
    def __init__(self, batch_id: str, total: int) -> None:
        self.batch_id         = batch_id
        self.total_videos     = total
        self.current_index    = 0
        self.current_video_id: Optional[str] = None
        self.completed        = 0
        self.failed           = 0
        self.status           = "running"
        self.results: List[dict] = []
        self._lock            = threading.Lock()

    def start_video(self, index: int, video_id: str) -> None:
        with self._lock:
            self.current_index    = index
            self.current_video_id = video_id

    def record_success(self, index: int, video_id: str, report: dict) -> None:
        with self._lock:
            self.results.append({"index": index, "video_id": video_id,
                                  "status": "success", "report": report, "error": None})
            self.completed += 1

    def record_failure(self, index: int, video_id: str, error: str) -> None:
        with self._lock:
            self.results.append({"index": index, "video_id": video_id,
                                  "status": "failed", "report": None, "error": error})
            self.failed += 1

    def finish(self) -> None:
        with self._lock:
            self.status           = "done"
            self.current_video_id = None

    def to_status_dict(self) -> dict:
        with self._lock:
            return {
                "batch_id":         self.batch_id,
                "total_videos":     self.total_videos,
                "current_index":    self.current_index,
                "current_video_id": self.current_video_id,
                "completed":        self.completed,
                "failed":           self.failed,
                "status":           self.status,
            }

    def to_final_dict(self) -> dict:
        with self._lock:
            return {
                "batch_id":     self.batch_id,
                "total_videos": self.total_videos,
                "completed":    self.completed,
                "failed":       self.failed,
                "status":       self.status,
                "results":      sorted(self.results, key=lambda r: r["index"]),
            }


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def _save_upload(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or "video.mp4")[1] or ".mp4"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return path


def _run_pipeline(
    tmp_path: str,
    analysis_id: str,
    train_detail_id: int,
    time_offset: float = 0.0,
    frame_offset: int  = 0,
    shared_vstore=None,
    original_filename: Optional[str] = None,
) -> tuple:
    """
    Run the pipeline for one video.
    Returns (report_path, video_duration_seconds, total_frame_count).
    In batch mode (shared_vstore is set), report_path is "" — caller finalizes.
    duration_s and total come directly from pipeline.run() which already
    measures them at startup — no need to re-open the (possibly deleted) temp file.
    """
    pipeline = GadgetDetectionPipeline(
        source=tmp_path,
        analysis_id=analysis_id,
        train_detail_id=train_detail_id,
        save=False,
        display=False,
        time_offset=time_offset,
        frame_offset=frame_offset,
        shared_vstore=shared_vstore,
        original_filename=original_filename,
    )
    report_path, duration, frame_count = pipeline.run()
    return report_path, duration, frame_count


def _make_id(filename: Optional[str], fallback: str) -> str:
    if filename:
        stem = os.path.splitext(filename)[0]
        if stem:
            return stem
    return fallback


# ─────────────────────────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Status"])
def health() -> dict:
    return {"status": "ok"}


# ─────────────────────────────────────────────────────────────────
# BATCH — blocking
# ─────────────────────────────────────────────────────────────────

@app.post("/analyze-batch", tags=["Analysis"])
async def analyze_batch(
    videos:          List[UploadFile] = File(...),
    train_detail_id: int              = Form(default=0),
    train_label:     str              = Form(default=""),
) -> JSONResponse:
    """
    Upload N video files (N decided by the user on the frontend).
    Blocks until all videos are processed, then returns the full combined report.
    Videos are processed one after another — video 2 starts only after video 1 finishes.

    train_label  — human-readable folder name, e.g. "12345-27/05/2026".
                   Slashes are replaced with dashes so it is a valid folder name.
                   If omitted, falls back to train_{train_detail_id}_{batch_id[:8]}.
    """
    if not videos:
        raise HTTPException(status_code=400, detail="No videos provided.")

    batch_id = uuid.uuid4().hex[:12]
    job      = BatchJob(batch_id=batch_id, total=len(videos))

    # Save all uploads to disk first
    saved: List[tuple] = []
    for u in videos:
        saved.append((_save_upload(u), u.filename or "video"))

    # ── Single shared analysis_id / folder name for the whole batch
    # train_label e.g. "12345-27/05/2026" → sanitise slashes → "12345-27-05-2026"
    safe_label = train_label.strip().replace("/", "-").replace("\\", "-") if train_label.strip() else ""
    batch_analysis_id = (
        safe_label
        if safe_label
        else (f"train_{train_detail_id}_{batch_id[:8]}" if train_detail_id else f"batch_{batch_id[:8]}")
    )

    # Create ONE shared ViolationStore for all videos in this batch
    shared_vstore = ViolationStore(
        analysis_id     = batch_analysis_id,
        train_detail_id = train_detail_id,
        video_info      = None,   # video infos added per-video via add_video_info()
    )

    import time as _time
    batch_start      = _time.monotonic()        # ← for correct elapsed processing_time
    cumulative_time   = 0.0   # running total of video durations (seconds)
    cumulative_frames = 0     # running total of frame counts across videos

    # Process one by one — strictly sequential
    for i, (tmp_path, filename) in enumerate(saved):
        analysis_id = _make_id(filename, f"video_{i+1}")
        job.start_video(i, analysis_id)
        print(f"[Batch {batch_id}] ▶ Video {i+1}/{len(saved)}: "
              f"{filename!r} → id={analysis_id!r}  "
              f"time_offset={cumulative_time:.3f}s  frame_offset={cumulative_frames}")
        try:
            _report_path, duration, frame_count = _run_pipeline(
                tmp_path, batch_analysis_id, train_detail_id,
                time_offset=cumulative_time,
                frame_offset=cumulative_frames,
                shared_vstore=shared_vstore,
                original_filename=filename,
            )
            cumulative_time   += duration
            cumulative_frames += frame_count
            job.record_success(i, analysis_id, {"status": "processed", "duration": duration})
            print(f"[Batch {batch_id}] ✓ Video {i+1} done  ({duration:.1f}s / {frame_count} frames)  "
                  f"cumulative={cumulative_time:.1f}s  frames={cumulative_frames}")
        except Exception as exc:
            job.record_failure(i, analysis_id, f"{exc}\n{traceback.format_exc()}")
            print(f"[Batch {batch_id}] ✗ Video {i+1} failed: {exc}")
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    # Finalize the ONE shared report after all videos
    report_path = shared_vstore.finalize(processing_time=_time.monotonic() - batch_start)
    final_report: dict = {}
    if report_path and os.path.isfile(report_path):
        with open(report_path, encoding="utf-8") as f:
            final_report = json.load(f)

    job.finish()
    return JSONResponse(content={**job.to_final_dict(), "report": final_report})


# ─────────────────────────────────────────────────────────────────
# BATCH ASYNC — non-blocking
# ─────────────────────────────────────────────────────────────────

@app.post("/analyze-batch-async", tags=["Analysis"])
async def analyze_batch_async(
    videos:          List[UploadFile] = File(...),
    train_detail_id: int              = Form(default=0),
    train_label:     str              = Form(default=""),
) -> JSONResponse:
    """
    Upload N video files. Returns batch_id immediately.
    Videos are processed sequentially in the background.
    Poll GET /batch-status/{batch_id} → when done, GET /batch-result/{batch_id}.

    train_label  — human-readable folder name, e.g. "12345-27/05/2026".
                   Slashes are replaced with dashes so it is a valid folder name.
    """
    if not videos:
        raise HTTPException(status_code=400, detail="No videos provided.")

    # Save uploads to disk before handing to background thread
    saved: List[tuple] = []
    for u in videos:
        saved.append((_save_upload(u), u.filename or "video"))

    batch_id = uuid.uuid4().hex[:12]
    job      = BatchJob(batch_id=batch_id, total=len(videos))

    with _batch_lock:
        _batch_jobs[batch_id] = job

    # ── Single shared analysis_id / folder name for the whole batch
    safe_label = train_label.strip().replace("/", "-").replace("\\", "-") if train_label.strip() else ""
    batch_analysis_id = (
        safe_label
        if safe_label
        else (f"train_{train_detail_id}_{batch_id[:8]}" if train_detail_id else f"batch_{batch_id[:8]}")
    )

    # Create ONE shared ViolationStore for all videos in this batch
    shared_vstore = ViolationStore(
        analysis_id     = batch_analysis_id,
        train_detail_id = train_detail_id,
        video_info      = None,
    )

    def _worker():
        import time as _time
        batch_start      = _time.monotonic()    # ← for correct elapsed processing_time
        cumulative_time   = 0.0
        cumulative_frames = 0
        for i, (tmp_path, filename) in enumerate(saved):
            analysis_id = _make_id(filename, f"video_{i+1}")
            job.start_video(i, analysis_id)
            print(f"[Batch {batch_id}] ▶ Video {i+1}/{len(saved)}: "
                  f"{filename!r} → id={analysis_id!r}  "
                  f"time_offset={cumulative_time:.3f}s  frame_offset={cumulative_frames}")
            try:
                _report_path, duration, frame_count = _run_pipeline(
                    tmp_path, batch_analysis_id, train_detail_id,
                    time_offset=cumulative_time,
                    frame_offset=cumulative_frames,
                    shared_vstore=shared_vstore,
                    original_filename=filename,
                )
                cumulative_time   += duration
                cumulative_frames += frame_count
                job.record_success(i, analysis_id, {"status": "processed", "duration": duration})
                print(f"[Batch {batch_id}] ✓ Video {i+1} done  ({duration:.1f}s / {frame_count} frames)  "
                      f"cumulative={cumulative_time:.1f}s  frames={cumulative_frames}")
            except Exception as exc:
                job.record_failure(i, analysis_id, f"{exc}\n{traceback.format_exc()}")
                print(f"[Batch {batch_id}] ✗ Video {i+1} failed: {exc}")
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        # Finalize the ONE shared report after all videos are done
        shared_vstore.finalize(processing_time=_time.monotonic() - batch_start)
        job.finish()
        print(f"[Batch {batch_id}] ✅ All {len(saved)} videos done.")

    threading.Thread(target=_worker, daemon=True, name=f"Batch-{batch_id}").start()

    return JSONResponse(
        status_code=202,
        content={
            "batch_id":     batch_id,
            "total_videos": len(videos),
            "status":       "running",
            "poll_url":     f"/batch-status/{batch_id}",
            "result_url":   f"/batch-result/{batch_id}",
            "message": (
                f"{len(videos)} video(s) queued. "
                f"Poll /batch-status/{batch_id} for live progress."
            ),
        },
    )


# ─────────────────────────────────────────────────────────────────
# STATUS + RESULT
# ─────────────────────────────────────────────────────────────────

@app.get("/batch-status/{batch_id}", tags=["Analysis"])
def batch_status(batch_id: str) -> JSONResponse:
    """Live progress for an async batch job."""
    with _batch_lock:
        job = _batch_jobs.get(batch_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found.")
    return JSONResponse(content=job.to_status_dict())


@app.get("/batch-result/{batch_id}", tags=["Analysis"])
def batch_result(batch_id: str) -> JSONResponse:
    """Full batch report. Returns 409 if the job is still running."""
    with _batch_lock:
        job = _batch_jobs.get(batch_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found.")
    if job.status != "done":
        raise HTTPException(
            status_code=409,
            detail=(f"Still running — {job.completed}/{job.total_videos} done. "
                    f"Poll /batch-status/{batch_id} first."),
        )
    return JSONResponse(content=job.to_final_dict())


# ─────────────────────────────────────────────────────────────────
# BUILT-IN TEST PAGE  →  http://localhost:8000/test
# ─────────────────────────────────────────────────────────────────

@app.get("/test", response_class=HTMLResponse, tags=["Status"])
def test_page():
    """
    Simple HTML test UI. Open http://localhost:8000/test in your browser.
    Enter a count, pick that many video files, submit — no Swagger needed.
    """
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Loco Pilot — Batch Test</title>
<style>
  body { font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; }
  h2   { color: #1a1a2e; }
  input[type=number], input[type=file] { margin: 6px 0; display: block; }
  button { margin-top: 14px; padding: 10px 28px; background: #0a6ebd;
           color: #fff; border: none; border-radius: 5px; cursor: pointer; font-size: 15px; }
  button:hover { background: #084f8c; }
  #file-slots { margin: 14px 0; }
  #file-slots label { font-weight: bold; display: block; margin-top: 10px; }
  #status  { margin-top: 20px; padding: 12px; background: #f0f4ff;
             border-left: 4px solid #0a6ebd; white-space: pre-wrap; font-size: 13px; }
  #result  { margin-top: 16px; padding: 12px; background: #f6fff0;
             border-left: 4px solid #2ecc71; white-space: pre-wrap;
             font-size: 12px; max-height: 400px; overflow-y: auto; }
  .hidden  { display: none; }
</style>
</head>
<body>

<h2>🚂 Loco Pilot — Batch Video Analysis</h2>

<label><b>Step 1 — Enter number of videos:</b></label>
<input type="number" id="count" min="1" max="20" value="1"
       style="width:80px; padding:6px; font-size:15px;">
<button onclick="generateSlots()">Generate Upload Slots</button>

<div id="file-slots"></div>

<div id="submit-area" class="hidden">
  <label><b>Train Detail ID:</b></label>
  <input type="number" id="train_detail_id" value="0" style="width:100px; padding:6px;">
  <br><br>
  <label><b>Train Number:</b></label>
  <input type="text" id="train_number" placeholder="e.g. 12345"
         style="width:120px; padding:6px; margin:4px 0;"
         oninput="updateTrainLabel()">
  <label style="margin-left:16px;"><b>Journey Date:</b></label>
  <input type="date" id="journey_date"
         style="padding:6px; margin:4px 0;"
         oninput="updateTrainLabel()">
  <br>
  <label><b>Folder name preview:</b></label>
  <input type="text" id="train_label" readonly
         style="width:220px; padding:6px; margin:4px 0;
                background:#f0f0f0; color:#333; font-family:monospace;">
  <br>
  <label>
    <input type="checkbox" id="async_mode"> Use async mode
    (returns immediately — polls for result)
  </label>
  <br>
  <button onclick="submitBatch()">▶ Start Processing</button>
</div>

<div id="status" class="hidden"></div>
<div id="result" class="hidden"></div>

<script>
  let pollTimer = null;

  function generateSlots() {
    const count = parseInt(document.getElementById('count').value);
    if (!count || count < 1) { alert('Enter a valid number.'); return; }

    const container = document.getElementById('file-slots');
    container.innerHTML = '';
    for (let i = 1; i <= count; i++) {
      const lbl = document.createElement('label');
      lbl.textContent = `Video ${i}:`;
      const inp = document.createElement('input');
      inp.type   = 'file';
      inp.id     = `video_${i}`;
      inp.accept = 'video/*';
      container.appendChild(lbl);
      container.appendChild(inp);
    }
    document.getElementById('submit-area').classList.remove('hidden');
    document.getElementById('status').classList.add('hidden');
    document.getElementById('result').classList.add('hidden');
  }

  async function submitBatch() {
    const count   = parseInt(document.getElementById('count').value);
    const formData = new FormData();
    let   filled  = 0;

    for (let i = 1; i <= count; i++) {
      const inp = document.getElementById(`video_${i}`);
      if (inp && inp.files[0]) {
        formData.append('videos', inp.files[0]);
        filled++;
      }
    }

    if (filled === 0) { alert('Please select at least one video file.'); return; }

    const trainId = document.getElementById('train_detail_id').value || '0';
    formData.append('train_detail_id', trainId);
    const trainLabel = document.getElementById('train_label').value || '';
    formData.append('train_label', trainLabel);
    // train_label is auto-built by updateTrainLabel() below

    const asyncMode = document.getElementById('async_mode').checked;
    const endpoint  = asyncMode ? '/analyze-batch-async' : '/analyze-batch';

    showStatus(`Uploading ${filled} video(s) to ${endpoint} ...`);

    try {
      const res  = await fetch(endpoint, { method: 'POST', body: formData });
      const data = await res.json();

      if (!asyncMode) {
        showStatus(`✅ Done! ${data.completed}/${data.total_videos} succeeded, ${data.failed} failed.`);
        showResult(data);
      } else {
        showStatus(`⏳ Batch started. batch_id = ${data.batch_id}\\nPolling for progress...`);
        pollStatus(data.batch_id);
      }
    } catch (err) {
      showStatus(`❌ Error: ${err}`);
    }
  }

  function pollStatus(batchId) {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(async () => {
      try {
        const res  = await fetch(`/batch-status/${batchId}`);
        const data = await res.json();
        showStatus(
          `⏳ Running...\n` +
          `  Current video : ${data.current_index + 1} / ${data.total_videos}  (id: ${data.current_video_id})\n` +
          `  Completed     : ${data.completed}\n` +
          `  Failed        : ${data.failed}\n` +
          `  Status        : ${data.status}`
        );
        if (data.status === 'done') {
          clearInterval(pollTimer);
          const rRes    = await fetch(`/batch-result/${batchId}`);
          const rData   = await rRes.json();
          showStatus(`✅ All done! ${rData.completed}/${rData.total_videos} succeeded, ${rData.failed} failed.`);
          showResult(rData);
        }
      } catch (err) {
        showStatus(`❌ Poll error: ${err}`);
        clearInterval(pollTimer);
      }
    }, 4000);
  }

  function showStatus(msg) {
    const el = document.getElementById('status');
    el.classList.remove('hidden');
    el.textContent = msg;
  }

  function showResult(data) {
    const el = document.getElementById('result');
    el.classList.remove('hidden');
    el.textContent = JSON.stringify(data, null, 2);
  }

  function updateTrainLabel() {
    const num  = document.getElementById('train_number').value.trim();
    const date = document.getElementById('journey_date').value;  // yyyy-mm-dd
    let label  = '';
    if (num && date) {
      // Convert yyyy-mm-dd → dd-mm-yyyy to match your format
      const [yyyy, mm, dd] = date.split('-');
      label = `${num}-${dd}-${mm}-${yyyy}`;
    } else if (num) {
      label = num;
    }
    document.getElementById('train_label').value = label;
  }

  // Set today's date as default
  window.addEventListener('load', () => {
    const today = new Date();
    const yyyy  = today.getFullYear();
    const mm    = String(today.getMonth() + 1).padStart(2, '0');
    const dd    = String(today.getDate()).padStart(2, '0');
    document.getElementById('journey_date').value = `${yyyy}-${mm}-${dd}`;
  });
</script>
</body>
</html>
"""
    return HTMLResponse(content=html)