# # # """
# # # api.py — FastAPI wrapper for the Loco Pilot Distraction Detection pipeline.

# # # POST /analyze-batch
# # #     Upload N videos (count decided by user on frontend).
# # #     Processes them SEQUENTIALLY. Returns combined JSON report.

# # # POST /analyze-batch-async
# # #     Same. Returns batch_id immediately, processes in background.

# # # GET  /batch-status/{batch_id}   — live progress
# # # GET  /batch-result/{batch_id}   — full report when done
# # # GET  /test                      — simple HTML test page (no Swagger needed)

# # # Usage
# # # -----
# # #     uvicorn api:app --host 0.0.0.0 --port 8000
# # # """

# # # from __future__ import annotations

# # # import json
# # # import os
# # # import shutil
# # # import tempfile
# # # import threading
# # # import traceback
# # # import uuid
# # # from typing import Dict, List, Optional

# # # from fastapi import FastAPI, File, Form, HTTPException, UploadFile
# # # from fastapi.middleware.cors import CORSMiddleware
# # # from fastapi.responses import HTMLResponse, JSONResponse

# # # from main import GadgetDetectionPipeline
# # # from utils.violation_store import ViolationStore

# # # app = FastAPI(
# # #     title="Loco Pilot Distraction Detection API",
# # #     version="2.0.0",
# # #     description=(
# # #         "Upload N videos (count entered by user). "
# # #         "Videos are always processed sequentially — video 2 starts only after video 1 finishes."
# # #     ),
# # # )

# # # app.add_middleware(
# # #     CORSMiddleware,
# # #     allow_origins=["*"],
# # #     allow_credentials=True,
# # #     allow_methods=["*"],
# # #     allow_headers=["*"],
# # # )

# # # _batch_jobs: Dict[str, "BatchJob"] = {}
# # # _batch_lock = threading.Lock()


# # # # ─────────────────────────────────────────────────────────────────
# # # # BATCH JOB STATE
# # # # ─────────────────────────────────────────────────────────────────

# # # class BatchJob:
# # #     def __init__(self, batch_id: str, total: int) -> None:
# # #         self.batch_id         = batch_id
# # #         self.total_videos     = total
# # #         self.current_index    = 0
# # #         self.current_video_id: Optional[str] = None
# # #         self.completed        = 0
# # #         self.failed           = 0
# # #         self.status           = "running"
# # #         self.results: List[dict] = []
# # #         self._lock            = threading.Lock()

# # #     def start_video(self, index: int, video_id: str) -> None:
# # #         with self._lock:
# # #             self.current_index    = index
# # #             self.current_video_id = video_id

# # #     def record_success(self, index: int, video_id: str, report: dict) -> None:
# # #         with self._lock:
# # #             self.results.append({"index": index, "video_id": video_id,
# # #                                   "status": "success", "report": report, "error": None})
# # #             self.completed += 1

# # #     def record_failure(self, index: int, video_id: str, error: str) -> None:
# # #         with self._lock:
# # #             self.results.append({"index": index, "video_id": video_id,
# # #                                   "status": "failed", "report": None, "error": error})
# # #             self.failed += 1

# # #     def finish(self) -> None:
# # #         with self._lock:
# # #             self.status           = "done"
# # #             self.current_video_id = None

# # #     def to_status_dict(self) -> dict:
# # #         with self._lock:
# # #             return {
# # #                 "batch_id":         self.batch_id,
# # #                 "total_videos":     self.total_videos,
# # #                 "current_index":    self.current_index,
# # #                 "current_video_id": self.current_video_id,
# # #                 "completed":        self.completed,
# # #                 "failed":           self.failed,
# # #                 "status":           self.status,
# # #             }

# # #     def to_final_dict(self) -> dict:
# # #         with self._lock:
# # #             return {
# # #                 "batch_id":     self.batch_id,
# # #                 "total_videos": self.total_videos,
# # #                 "completed":    self.completed,
# # #                 "failed":       self.failed,
# # #                 "status":       self.status,
# # #                 "results":      sorted(self.results, key=lambda r: r["index"]),
# # #             }


# # # # ─────────────────────────────────────────────────────────────────
# # # # HELPERS
# # # # ─────────────────────────────────────────────────────────────────

# # # def _save_upload(upload: UploadFile) -> str:
# # #     suffix = os.path.splitext(upload.filename or "video.mp4")[1] or ".mp4"
# # #     fd, path = tempfile.mkstemp(suffix=suffix)
# # #     with os.fdopen(fd, "wb") as f:
# # #         shutil.copyfileobj(upload.file, f)
# # #     return path


# # # def _run_pipeline(
# # #     tmp_path: str,
# # #     analysis_id: str,
# # #     train_detail_id: int,
# # #     time_offset: float = 0.0,
# # #     frame_offset: int  = 0,
# # #     shared_vstore=None,
# # #     original_filename: Optional[str] = None,
# # # ) -> tuple:
# # #     """
# # #     Run the pipeline for one video.
# # #     Returns (report_path, video_duration_seconds, total_frame_count).
# # #     In batch mode (shared_vstore is set), report_path is "" — caller finalizes.
# # #     duration_s and total come directly from pipeline.run() which already
# # #     measures them at startup — no need to re-open the (possibly deleted) temp file.
# # #     """
# # #     pipeline = GadgetDetectionPipeline(
# # #         source=tmp_path,
# # #         analysis_id=analysis_id,
# # #         train_detail_id=train_detail_id,
# # #         save=False,
# # #         display=False,
# # #         time_offset=time_offset,
# # #         frame_offset=frame_offset,
# # #         shared_vstore=shared_vstore,
# # #         original_filename=original_filename,
# # #     )
# # #     report_path, duration, frame_count = pipeline.run()
# # #     return report_path, duration, frame_count


# # # def _make_id(filename: Optional[str], fallback: str) -> str:
# # #     if filename:
# # #         stem = os.path.splitext(filename)[0]
# # #         if stem:
# # #             return stem
# # #     return fallback


# # # # ─────────────────────────────────────────────────────────────────
# # # # HEALTH
# # # # ─────────────────────────────────────────────────────────────────

# # # @app.get("/health", tags=["Status"])
# # # def health() -> dict:
# # #     return {"status": "ok"}


# # # # ─────────────────────────────────────────────────────────────────
# # # # BATCH — blocking
# # # # ─────────────────────────────────────────────────────────────────

# # # @app.post("/analyze-batch", tags=["Analysis"])
# # # async def analyze_batch(
# # #     videos:          List[UploadFile] = File(...),
# # #     train_detail_id: int              = Form(default=0),
# # #     train_label:     str              = Form(default=""),
# # # ) -> JSONResponse:
# # #     """
# # #     Upload N video files (N decided by the user on the frontend).
# # #     Blocks until all videos are processed, then returns the full combined report.
# # #     Videos are processed one after another — video 2 starts only after video 1 finishes.

# # #     train_label  — human-readable folder name, e.g. "12345-27/05/2026".
# # #                    Slashes are replaced with dashes so it is a valid folder name.
# # #                    If omitted, falls back to train_{train_detail_id}_{batch_id[:8]}.
# # #     """
# # #     if not videos:
# # #         raise HTTPException(status_code=400, detail="No videos provided.")

# # #     batch_id = uuid.uuid4().hex[:12]
# # #     job      = BatchJob(batch_id=batch_id, total=len(videos))

# # #     # Save all uploads to disk first
# # #     saved: List[tuple] = []
# # #     for u in videos:
# # #         saved.append((_save_upload(u), u.filename or "video"))

# # #     # ── Single shared analysis_id / folder name for the whole batch
# # #     # train_label e.g. "12345-27/05/2026" → sanitise slashes → "12345-27-05-2026"
# # #     safe_label = train_label.strip().replace("/", "-").replace("\\", "-") if train_label.strip() else ""
# # #     batch_analysis_id = (
# # #         safe_label
# # #         if safe_label
# # #         else (f"train_{train_detail_id}_{batch_id[:8]}" if train_detail_id else f"batch_{batch_id[:8]}")
# # #     )

# # #     # Create ONE shared ViolationStore for all videos in this batch
# # #     shared_vstore = ViolationStore(
# # #         analysis_id     = batch_analysis_id,
# # #         train_detail_id = train_detail_id,
# # #         video_info      = None,   # video infos added per-video via add_video_info()
# # #     )

# # #     import time as _time
# # #     batch_start      = _time.monotonic()        # ← for correct elapsed processing_time
# # #     cumulative_time   = 0.0   # running total of video durations (seconds)
# # #     cumulative_frames = 0     # running total of frame counts across videos

# # #     # Process one by one — strictly sequential
# # #     for i, (tmp_path, filename) in enumerate(saved):
# # #         analysis_id = _make_id(filename, f"video_{i+1}")
# # #         job.start_video(i, analysis_id)
# # #         print(f"[Batch {batch_id}] ▶ Video {i+1}/{len(saved)}: "
# # #               f"{filename!r} → id={analysis_id!r}  "
# # #               f"time_offset={cumulative_time:.3f}s  frame_offset={cumulative_frames}")
# # #         try:
# # #             _report_path, duration, frame_count = _run_pipeline(
# # #                 tmp_path, batch_analysis_id, train_detail_id,
# # #                 time_offset=cumulative_time,
# # #                 frame_offset=cumulative_frames,
# # #                 shared_vstore=shared_vstore,
# # #                 original_filename=filename,
# # #             )
# # #             cumulative_time   += duration
# # #             cumulative_frames += frame_count
# # #             job.record_success(i, analysis_id, {"status": "processed", "duration": duration})
# # #             print(f"[Batch {batch_id}] ✓ Video {i+1} done  ({duration:.1f}s / {frame_count} frames)  "
# # #                   f"cumulative={cumulative_time:.1f}s  frames={cumulative_frames}")
# # #         except Exception as exc:
# # #             job.record_failure(i, analysis_id, f"{exc}\n{traceback.format_exc()}")
# # #             print(f"[Batch {batch_id}] ✗ Video {i+1} failed: {exc}")
# # #         finally:
# # #             try:
# # #                 os.remove(tmp_path)
# # #             except OSError:
# # #                 pass

# # #     # Finalize the ONE shared report after all videos
# # #     report_path = shared_vstore.finalize(processing_time=_time.monotonic() - batch_start)
# # #     final_report: dict = {}
# # #     if report_path and os.path.isfile(report_path):
# # #         with open(report_path, encoding="utf-8") as f:
# # #             final_report = json.load(f)

# # #     job.finish()
# # #     return JSONResponse(content={**job.to_final_dict(), "report": final_report})


# # # # ─────────────────────────────────────────────────────────────────
# # # # BATCH ASYNC — non-blocking
# # # # ─────────────────────────────────────────────────────────────────

# # # @app.post("/analyze-batch-async", tags=["Analysis"])
# # # async def analyze_batch_async(
# # #     videos:          List[UploadFile] = File(...),
# # #     train_detail_id: int              = Form(default=0),
# # #     train_label:     str              = Form(default=""),
# # # ) -> JSONResponse:
# # #     """
# # #     Upload N video files. Returns batch_id immediately.
# # #     Videos are processed sequentially in the background.
# # #     Poll GET /batch-status/{batch_id} → when done, GET /batch-result/{batch_id}.

# # #     train_label  — human-readable folder name, e.g. "12345-27/05/2026".
# # #                    Slashes are replaced with dashes so it is a valid folder name.
# # #     """
# # #     if not videos:
# # #         raise HTTPException(status_code=400, detail="No videos provided.")

# # #     # Save uploads to disk before handing to background thread
# # #     saved: List[tuple] = []
# # #     for u in videos:
# # #         saved.append((_save_upload(u), u.filename or "video"))

# # #     batch_id = uuid.uuid4().hex[:12]
# # #     job      = BatchJob(batch_id=batch_id, total=len(videos))

# # #     with _batch_lock:
# # #         _batch_jobs[batch_id] = job

# # #     # ── Single shared analysis_id / folder name for the whole batch
# # #     safe_label = train_label.strip().replace("/", "-").replace("\\", "-") if train_label.strip() else ""
# # #     batch_analysis_id = (
# # #         safe_label
# # #         if safe_label
# # #         else (f"train_{train_detail_id}_{batch_id[:8]}" if train_detail_id else f"batch_{batch_id[:8]}")
# # #     )

# # #     # Create ONE shared ViolationStore for all videos in this batch
# # #     shared_vstore = ViolationStore(
# # #         analysis_id     = batch_analysis_id,
# # #         train_detail_id = train_detail_id,
# # #         video_info      = None,
# # #     )

# # #     def _worker():
# # #         import time as _time
# # #         batch_start      = _time.monotonic()    # ← for correct elapsed processing_time
# # #         cumulative_time   = 0.0
# # #         cumulative_frames = 0
# # #         for i, (tmp_path, filename) in enumerate(saved):
# # #             analysis_id = _make_id(filename, f"video_{i+1}")
# # #             job.start_video(i, analysis_id)
# # #             print(f"[Batch {batch_id}] ▶ Video {i+1}/{len(saved)}: "
# # #                   f"{filename!r} → id={analysis_id!r}  "
# # #                   f"time_offset={cumulative_time:.3f}s  frame_offset={cumulative_frames}")
# # #             try:
# # #                 _report_path, duration, frame_count = _run_pipeline(
# # #                     tmp_path, batch_analysis_id, train_detail_id,
# # #                     time_offset=cumulative_time,
# # #                     frame_offset=cumulative_frames,
# # #                     shared_vstore=shared_vstore,
# # #                     original_filename=filename,
# # #                 )
# # #                 cumulative_time   += duration
# # #                 cumulative_frames += frame_count
# # #                 job.record_success(i, analysis_id, {"status": "processed", "duration": duration})
# # #                 print(f"[Batch {batch_id}] ✓ Video {i+1} done  ({duration:.1f}s / {frame_count} frames)  "
# # #                       f"cumulative={cumulative_time:.1f}s  frames={cumulative_frames}")
# # #             except Exception as exc:
# # #                 job.record_failure(i, analysis_id, f"{exc}\n{traceback.format_exc()}")
# # #                 print(f"[Batch {batch_id}] ✗ Video {i+1} failed: {exc}")
# # #             finally:
# # #                 try:
# # #                     os.remove(tmp_path)
# # #                 except OSError:
# # #                     pass
# # #         # Finalize the ONE shared report after all videos are done
# # #         shared_vstore.finalize(processing_time=_time.monotonic() - batch_start)
# # #         job.finish()
# # #         print(f"[Batch {batch_id}] ✅ All {len(saved)} videos done.")

# # #     threading.Thread(target=_worker, daemon=True, name=f"Batch-{batch_id}").start()

# # #     return JSONResponse(
# # #         status_code=202,
# # #         content={
# # #             "batch_id":     batch_id,
# # #             "total_videos": len(videos),
# # #             "status":       "running",
# # #             "poll_url":     f"/batch-status/{batch_id}",
# # #             "result_url":   f"/batch-result/{batch_id}",
# # #             "message": (
# # #                 f"{len(videos)} video(s) queued. "
# # #                 f"Poll /batch-status/{batch_id} for live progress."
# # #             ),
# # #         },
# # #     )


# # # # ─────────────────────────────────────────────────────────────────
# # # # STATUS + RESULT
# # # # ─────────────────────────────────────────────────────────────────

# # # @app.get("/batch-status/{batch_id}", tags=["Analysis"])
# # # def batch_status(batch_id: str) -> JSONResponse:
# # #     """Live progress for an async batch job."""
# # #     with _batch_lock:
# # #         job = _batch_jobs.get(batch_id)
# # #     if not job:
# # #         raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found.")
# # #     return JSONResponse(content=job.to_status_dict())


# # # @app.get("/batch-result/{batch_id}", tags=["Analysis"])
# # # def batch_result(batch_id: str) -> JSONResponse:
# # #     """Full batch report. Returns 409 if the job is still running."""
# # #     with _batch_lock:
# # #         job = _batch_jobs.get(batch_id)
# # #     if not job:
# # #         raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found.")
# # #     if job.status != "done":
# # #         raise HTTPException(
# # #             status_code=409,
# # #             detail=(f"Still running — {job.completed}/{job.total_videos} done. "
# # #                     f"Poll /batch-status/{batch_id} first."),
# # #         )
# # #     return JSONResponse(content=job.to_final_dict())


# # # # ─────────────────────────────────────────────────────────────────
# # # # BUILT-IN TEST PAGE  →  http://localhost:8000/test
# # # # ─────────────────────────────────────────────────────────────────

# # # @app.get("/test", response_class=HTMLResponse, tags=["Status"])
# # # def test_page():
# # #     """
# # #     Simple HTML test UI. Open http://localhost:8000/test in your browser.
# # #     Enter a count, pick that many video files, submit — no Swagger needed.
# # #     """
# # #     html = """
# # # <!DOCTYPE html>
# # # <html lang="en">
# # # <head>
# # # <meta charset="UTF-8">
# # # <title>Loco Pilot — Batch Test</title>
# # # <style>
# # #   body { font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; }
# # #   h2   { color: #1a1a2e; }
# # #   input[type=number], input[type=file] { margin: 6px 0; display: block; }
# # #   button { margin-top: 14px; padding: 10px 28px; background: #0a6ebd;
# # #            color: #fff; border: none; border-radius: 5px; cursor: pointer; font-size: 15px; }
# # #   button:hover { background: #084f8c; }
# # #   #file-slots { margin: 14px 0; }
# # #   #file-slots label { font-weight: bold; display: block; margin-top: 10px; }
# # #   #status  { margin-top: 20px; padding: 12px; background: #f0f4ff;
# # #              border-left: 4px solid #0a6ebd; white-space: pre-wrap; font-size: 13px; }
# # #   #result  { margin-top: 16px; padding: 12px; background: #f6fff0;
# # #              border-left: 4px solid #2ecc71; white-space: pre-wrap;
# # #              font-size: 12px; max-height: 400px; overflow-y: auto; }
# # #   .hidden  { display: none; }
# # # </style>
# # # </head>
# # # <body>

# # # <h2>🚂 Loco Pilot — Batch Video Analysis</h2>

# # # <label><b>Step 1 — Enter number of videos:</b></label>
# # # <input type="number" id="count" min="1" max="20" value="1"
# # #        style="width:80px; padding:6px; font-size:15px;">
# # # <button onclick="generateSlots()">Generate Upload Slots</button>

# # # <div id="file-slots"></div>

# # # <div id="submit-area" class="hidden">
# # #   <label><b>Train Detail ID:</b></label>
# # #   <input type="number" id="train_detail_id" value="0" style="width:100px; padding:6px;">
# # #   <br><br>
# # #   <label><b>Train Number:</b></label>
# # #   <input type="text" id="train_number" placeholder="e.g. 12345"
# # #          style="width:120px; padding:6px; margin:4px 0;"
# # #          oninput="updateTrainLabel()">
# # #   <label style="margin-left:16px;"><b>Journey Date:</b></label>
# # #   <input type="date" id="journey_date"
# # #          style="padding:6px; margin:4px 0;"
# # #          oninput="updateTrainLabel()">
# # #   <br>
# # #   <label><b>Folder name preview:</b></label>
# # #   <input type="text" id="train_label" readonly
# # #          style="width:220px; padding:6px; margin:4px 0;
# # #                 background:#f0f0f0; color:#333; font-family:monospace;">
# # #   <br>
# # #   <label>
# # #     <input type="checkbox" id="async_mode"> Use async mode
# # #     (returns immediately — polls for result)
# # #   </label>
# # #   <br>
# # #   <button onclick="submitBatch()">▶ Start Processing</button>
# # # </div>

# # # <div id="status" class="hidden"></div>
# # # <div id="result" class="hidden"></div>

# # # <script>
# # #   let pollTimer = null;

# # #   function generateSlots() {
# # #     const count = parseInt(document.getElementById('count').value);
# # #     if (!count || count < 1) { alert('Enter a valid number.'); return; }

# # #     const container = document.getElementById('file-slots');
# # #     container.innerHTML = '';
# # #     for (let i = 1; i <= count; i++) {
# # #       const lbl = document.createElement('label');
# # #       lbl.textContent = `Video ${i}:`;
# # #       const inp = document.createElement('input');
# # #       inp.type   = 'file';
# # #       inp.id     = `video_${i}`;
# # #       inp.accept = 'video/*';
# # #       container.appendChild(lbl);
# # #       container.appendChild(inp);
# # #     }
# # #     document.getElementById('submit-area').classList.remove('hidden');
# # #     document.getElementById('status').classList.add('hidden');
# # #     document.getElementById('result').classList.add('hidden');
# # #   }

# # #   async function submitBatch() {
# # #     const count   = parseInt(document.getElementById('count').value);
# # #     const formData = new FormData();
# # #     let   filled  = 0;

# # #     for (let i = 1; i <= count; i++) {
# # #       const inp = document.getElementById(`video_${i}`);
# # #       if (inp && inp.files[0]) {
# # #         formData.append('videos', inp.files[0]);
# # #         filled++;
# # #       }
# # #     }

# # #     if (filled === 0) { alert('Please select at least one video file.'); return; }

# # #     const trainId = document.getElementById('train_detail_id').value || '0';
# # #     formData.append('train_detail_id', trainId);
# # #     const trainLabel = document.getElementById('train_label').value || '';
# # #     formData.append('train_label', trainLabel);
# # #     // train_label is auto-built by updateTrainLabel() below

# # #     const asyncMode = document.getElementById('async_mode').checked;
# # #     const endpoint  = asyncMode ? '/analyze-batch-async' : '/analyze-batch';

# # #     showStatus(`Uploading ${filled} video(s) to ${endpoint} ...`);

# # #     try {
# # #       const res  = await fetch(endpoint, { method: 'POST', body: formData });
# # #       const data = await res.json();

# # #       if (!asyncMode) {
# # #         showStatus(`✅ Done! ${data.completed}/${data.total_videos} succeeded, ${data.failed} failed.`);
# # #         showResult(data);
# # #       } else {
# # #         showStatus(`⏳ Batch started. batch_id = ${data.batch_id}\\nPolling for progress...`);
# # #         pollStatus(data.batch_id);
# # #       }
# # #     } catch (err) {
# # #       showStatus(`❌ Error: ${err}`);
# # #     }
# # #   }

# # #   function pollStatus(batchId) {
# # #     if (pollTimer) clearInterval(pollTimer);
# # #     pollTimer = setInterval(async () => {
# # #       try {
# # #         const res  = await fetch(`/batch-status/${batchId}`);
# # #         const data = await res.json();
# # #         showStatus(
# # #           `⏳ Running...\n` +
# # #           `  Current video : ${data.current_index + 1} / ${data.total_videos}  (id: ${data.current_video_id})\n` +
# # #           `  Completed     : ${data.completed}\n` +
# # #           `  Failed        : ${data.failed}\n` +
# # #           `  Status        : ${data.status}`
# # #         );
# # #         if (data.status === 'done') {
# # #           clearInterval(pollTimer);
# # #           const rRes    = await fetch(`/batch-result/${batchId}`);
# # #           const rData   = await rRes.json();
# # #           showStatus(`✅ All done! ${rData.completed}/${rData.total_videos} succeeded, ${rData.failed} failed.`);
# # #           showResult(rData);
# # #         }
# # #       } catch (err) {
# # #         showStatus(`❌ Poll error: ${err}`);
# # #         clearInterval(pollTimer);
# # #       }
# # #     }, 4000);
# # #   }

# # #   function showStatus(msg) {
# # #     const el = document.getElementById('status');
# # #     el.classList.remove('hidden');
# # #     el.textContent = msg;
# # #   }

# # #   function showResult(data) {
# # #     const el = document.getElementById('result');
# # #     el.classList.remove('hidden');
# # #     el.textContent = JSON.stringify(data, null, 2);
# # #   }

# # #   function updateTrainLabel() {
# # #     const num  = document.getElementById('train_number').value.trim();
# # #     const date = document.getElementById('journey_date').value;  // yyyy-mm-dd
# # #     let label  = '';
# # #     if (num && date) {
# # #       // Convert yyyy-mm-dd → dd-mm-yyyy to match your format
# # #       const [yyyy, mm, dd] = date.split('-');
# # #       label = `${num}-${dd}-${mm}-${yyyy}`;
# # #     } else if (num) {
# # #       label = num;
# # #     }
# # #     document.getElementById('train_label').value = label;
# # #   }

# # #   // Set today's date as default
# # #   window.addEventListener('load', () => {
# # #     const today = new Date();
# # #     const yyyy  = today.getFullYear();
# # #     const mm    = String(today.getMonth() + 1).padStart(2, '0');
# # #     const dd    = String(today.getDate()).padStart(2, '0');
# # #     document.getElementById('journey_date').value = `${yyyy}-${mm}-${dd}`;
# # #   });
# # # </script>
# # # </body>
# # # </html>
# # # """
# # #     return HTMLResponse(content=html)



# # """
# # api.py — FastAPI wrapper for the Loco Pilot Distraction Detection pipeline.

# # FLOW (new)
# # ──────────
# # Frontend / register_videos.py already inserted rows into video_files with
# #     folder_name, filename, video_length, file_size, s3_video_path,
# #     process_flag = 'N'

# # This API:
# #     POST /process-pending          blocking  — process all flag=N, return full report
# #     POST /process-pending-async    async     — kick off in background, poll for progress
# #     GET  /batch-status/{batch_id}  live progress
# #     GET  /batch-result/{batch_id}  final report when done

# # Flag lifecycle per video:
# #     N  →  I   before pipeline starts
# #     I  →  Y   after pipeline + S3 upload succeeds
# #     I  stays I on failure — operator must review and requeue manually

# # Existing endpoints (unchanged):
# #     POST /analyze-batch
# #     POST /analyze-batch-async
# #     GET  /health
# #     GET  /test
# # """

# # from __future__ import annotations

# # import json
# # import os
# # import shutil
# # import tempfile
# # import threading
# # import traceback
# # import uuid
# # from collections import defaultdict
# # from typing import Dict, List, Optional

# # from fastapi import FastAPI, File, Form, HTTPException, UploadFile
# # from fastapi.middleware.cors import CORSMiddleware
# # from fastapi.responses import HTMLResponse, JSONResponse

# # from main import GadgetDetectionPipeline
# # from utils.violation_store import ViolationStore
# # from utils.db_s3_uploader import (
# #     get_pending_videos,
# #     set_process_flag,
# #     download_video_from_s3,
# # )

# # app = FastAPI(
# #     title="Loco Pilot Distraction Detection API",
# #     version="3.0.0",
# #     description=(
# #         "POST /process-pending — reads flag=N rows from DB, processes sequentially. "
# #         "POST /analyze-batch   — direct upload without DB (original endpoint)."
# #     ),
# # )

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # _batch_jobs: Dict[str, "BatchJob"] = {}
# # _batch_lock = threading.Lock()


# # # ─────────────────────────────────────────────────────────────────
# # # BATCH JOB STATE
# # # ─────────────────────────────────────────────────────────────────

# # class BatchJob:
# #     def __init__(self, batch_id: str, total: int) -> None:
# #         self.batch_id          = batch_id
# #         self.total_videos      = total
# #         self.current_index     = 0
# #         self.current_filename: Optional[str] = None
# #         self.completed         = 0
# #         self.failed            = 0
# #         self.status            = "running"
# #         self.results: List[dict] = []
# #         self._lock             = threading.Lock()

# #     def start_video(self, index: int, filename: str) -> None:
# #         with self._lock:
# #             self.current_index    = index
# #             self.current_filename = filename

# #     def record_success(self, index: int, filename: str, duration: float) -> None:
# #         with self._lock:
# #             self.results.append({
# #                 "index":    index,
# #                 "filename": filename,
# #                 "status":   "success",
# #                 "duration": duration,
# #                 "error":    None,
# #             })
# #             self.completed += 1

# #     def record_failure(self, index: int, filename: str, error: str) -> None:
# #         with self._lock:
# #             self.results.append({
# #                 "index":    index,
# #                 "filename": filename,
# #                 "status":   "failed",
# #                 "duration": 0.0,
# #                 "error":    error,
# #             })
# #             self.failed += 1

# #     def finish(self) -> None:
# #         with self._lock:
# #             self.status           = "done"
# #             self.current_filename = None

# #     def to_status_dict(self) -> dict:
# #         with self._lock:
# #             return {
# #                 "batch_id":         self.batch_id,
# #                 "total_videos":     self.total_videos,
# #                 "current_index":    self.current_index,
# #                 "current_filename": self.current_filename,
# #                 "completed":        self.completed,
# #                 "failed":           self.failed,
# #                 "status":           self.status,
# #             }

# #     def to_final_dict(self) -> dict:
# #         with self._lock:
# #             return {
# #                 "batch_id":     self.batch_id,
# #                 "total_videos": self.total_videos,
# #                 "completed":    self.completed,
# #                 "failed":       self.failed,
# #                 "status":       self.status,
# #                 "results":      sorted(self.results, key=lambda r: r["index"]),
# #             }


# # # ─────────────────────────────────────────────────────────────────
# # # INTERNAL HELPERS
# # # ─────────────────────────────────────────────────────────────────

# # def _run_pipeline(
# #     tmp_path:          str,
# #     analysis_id:       str,
# #     train_detail_id:   int,
# #     time_offset:       float = 0.0,
# #     frame_offset:      int   = 0,
# #     shared_vstore:     Optional[ViolationStore] = None,
# #     original_filename: Optional[str] = None,
# # ) -> tuple:
# #     """
# #     Run the detection pipeline for one video.
# #     Returns (report_path, duration_seconds, total_frame_count).
# #     report_path is '' in batch mode — caller calls shared_vstore.finalize().
# #     """
# #     pipeline = GadgetDetectionPipeline(
# #         source            = tmp_path,
# #         analysis_id       = analysis_id,
# #         train_detail_id   = train_detail_id,
# #         save              = False,
# #         display           = False,
# #         time_offset       = time_offset,
# #         frame_offset      = frame_offset,
# #         shared_vstore     = shared_vstore,
# #         original_filename = original_filename,
# #     )
# #     return pipeline.run()


# # def _save_upload(upload: UploadFile) -> str:
# #     """Save an UploadFile to a temp path and return the path."""
# #     suffix = os.path.splitext(upload.filename or "video.mp4")[1] or ".mp4"
# #     fd, path = tempfile.mkstemp(suffix=suffix)
# #     with os.fdopen(fd, "wb") as f:
# #         shutil.copyfileobj(upload.file, f)
# #     return path


# # def _process_folder(
# #     folder_name:     str,
# #     video_rows:      List[dict],
# #     train_detail_id: int,
# #     job:             BatchJob,
# #     job_index_start: int = 0,
# # ) -> dict:
# #     """
# #     Process all videos in one folder sequentially by seq_no.

# #     This is the single function that owns the entire N→I→Y flag logic
# #     and the cumulative time/frame offset accumulation.
# #     Both the blocking and async endpoints call this same function.

# #     video_rows  — list of dicts from get_pending_videos(), ordered by seq_no.
# #     job_index_start — offset into the overall BatchJob when multiple
# #                       folders are processed in one call.

# #     Returns the parsed final report dict (or {} if no videos succeeded).
# #     """
# #     import time as _time

# #     # One shared ViolationStore for the whole folder = one combined report
# #     shared_vstore = ViolationStore(
# #         analysis_id     = folder_name,
# #         train_detail_id = train_detail_id,
# #         video_info      = None,
# #     )

# #     batch_start       = _time.monotonic()
# #     cumulative_time   = 0.0    # sum of durations of completed videos
# #     cumulative_frames = 0      # sum of frame counts of completed videos

# #     for i, row in enumerate(video_rows):
# #         video_id      = row["id"]
# #         filename      = row["filename"]
# #         s3_video_path = row["s3_video_path"]
# #         global_index  = job_index_start + i

# #         job.start_video(global_index, filename)

# #         # 1. Mark IN PROGRESS — do this before download so it's
# #         #    visible immediately in the DB / check script
# #         set_process_flag(video_id, "I")

# #         suffix   = os.path.splitext(filename)[1] or ".mp4"
# #         tmp_path = tempfile.mktemp(suffix=suffix)

# #         print(
# #             f"[ProcessFolder] ▶ {i+1}/{len(video_rows)}: {filename!r}  "
# #             f"seq={row['seq_no']}  "
# #             f"time_offset={cumulative_time:.1f}s  "
# #             f"frame_offset={cumulative_frames}"
# #         )

# #         try:
# #             # 2. Download video from S3
# #             download_video_from_s3(s3_video_path, tmp_path)

# #             # 3. Run detection pipeline
# #             _report_path, duration, frame_count = _run_pipeline(
# #                 tmp_path          = tmp_path,
# #                 analysis_id       = folder_name,
# #                 train_detail_id   = train_detail_id,
# #                 time_offset       = cumulative_time,
# #                 frame_offset      = cumulative_frames,
# #                 shared_vstore     = shared_vstore,
# #                 original_filename = filename,
# #             )

# #             cumulative_time   += duration
# #             cumulative_frames += frame_count

# #             # 4. Mark DONE
# #             set_process_flag(video_id, "Y")
# #             job.record_success(global_index, filename, duration)
# #             print(
# #                 f"[ProcessFolder] ✓ {filename!r}  "
# #                 f"duration={duration:.1f}s  frames={frame_count}"
# #             )

# #         except Exception as exc:
# #             # Flag intentionally stays 'I' on failure.
# #             # Operator can see exactly which video failed via --check
# #             # and decide whether to manually requeue.
# #             job.record_failure(global_index, filename, str(exc))
# #             print(
# #                 f"[ProcessFolder] ✗ {filename!r} FAILED — "
# #                 f"flag stays 'I': {exc}"
# #             )
# #             # Continue processing remaining videos in this folder

# #         finally:
# #             try:
# #                 os.remove(tmp_path)
# #             except OSError:
# #                 pass

# #     # 5. Finalize: write JSON + save frames + upload results to S3
# #     #    ViolationStore.finalize() internally calls
# #     #    db_s3_uploader.finalize_and_upload() which:
# #     #       - uploads outputs/<folder>/ to S3 under <folder>/results/
# #     #       - updates result_s3_path in DB for all rows in this folder
# #     report_path = shared_vstore.finalize(
# #         processing_time = _time.monotonic() - batch_start
# #     )

# #     final_report: dict = {}
# #     if report_path and os.path.isfile(report_path):
# #         with open(report_path, encoding="utf-8") as f:
# #             final_report = json.load(f)

# #     return final_report


# # # ─────────────────────────────────────────────────────────────────
# # # HEALTH
# # # ─────────────────────────────────────────────────────────────────

# # @app.get("/health", tags=["Status"])
# # def health() -> dict:
# #     return {"status": "ok"}


# # # ─────────────────────────────────────────────────────────────────
# # # PROCESS PENDING — blocking
# # # ─────────────────────────────────────────────────────────────────

# # @app.post("/process-pending", tags=["Processing"])
# # async def process_pending(
# #     train_detail_id: int           = Form(...),
# #     folder_name:     Optional[str] = Form(default=None),
# # ) -> JSONResponse:
# #     """
# #     Blocking. Waits until all videos are done then returns.

# #     Reads all video_files rows where:
# #         train_detail_id = <given>
# #         process_flag    = 'N'
# #         folder_name     = <given>  (optional — omit to process ALL pending)

# #     For each folder (grouped by folder_name, ordered by seq_no):
# #         flag N → I → pipeline → Y
# #         flag stays I on failure

# #     After each folder: writes analysis_report.json locally,
# #     uploads results to S3, updates result_s3_path in DB.
# #     """
# #     pending = get_pending_videos(
# #         train_detail_id = train_detail_id,
# #         folder_name     = folder_name,
# #     )

# #     if not pending:
# #         return JSONResponse(content={
# #             "message":         "No pending videos found (process_flag='N').",
# #             "folder_name":     folder_name,
# #             "train_detail_id": train_detail_id,
# #         })

# #     # Group rows by folder_name
# #     folders: Dict[str, list] = defaultdict(list)
# #     for row in pending:
# #         folders[row["folder_name"]].append(row)

# #     batch_id = uuid.uuid4().hex[:12]
# #     job      = BatchJob(batch_id=batch_id, total=len(pending))

# #     with _batch_lock:
# #         _batch_jobs[batch_id] = job

# #     folder_results = []
# #     job_index      = 0

# #     for f_name, rows in folders.items():
# #         print(f"[ProcessPending] Folder '{f_name}'  {len(rows)} video(s)")
# #         report = _process_folder(
# #             folder_name     = f_name,
# #             video_rows      = rows,
# #             train_detail_id = train_detail_id,
# #             job             = job,
# #             job_index_start = job_index,
# #         )
# #         folder_results.append({
# #             "folder_name":      f_name,
# #             "videos_in_folder": len(rows),
# #             "report":           report,
# #         })
# #         job_index += len(rows)

# #     job.finish()

# #     return JSONResponse(content={
# #         "batch_id":          batch_id,
# #         "train_detail_id":   train_detail_id,
# #         "total_videos":      len(pending),
# #         "completed":         job.completed,
# #         "failed":            job.failed,
# #         "folders_processed": len(folder_results),
# #         "folders":           folder_results,
# #     })


# # # ─────────────────────────────────────────────────────────────────
# # # PROCESS PENDING ASYNC — non-blocking
# # # ─────────────────────────────────────────────────────────────────

# # @app.post("/process-pending-async", tags=["Processing"])
# # async def process_pending_async(
# #     train_detail_id: int           = Form(...),
# #     folder_name:     Optional[str] = Form(default=None),
# # ) -> JSONResponse:
# #     """
# #     Same as /process-pending but returns immediately with a batch_id.
# #     Poll GET /batch-status/{batch_id} every few seconds.
# #     Fetch GET /batch-result/{batch_id} when status = 'done'.
# #     """
# #     pending = get_pending_videos(
# #         train_detail_id = train_detail_id,
# #         folder_name     = folder_name,
# #     )

# #     if not pending:
# #         return JSONResponse(content={
# #             "message":         "No pending videos found.",
# #             "folder_name":     folder_name,
# #             "train_detail_id": train_detail_id,
# #         })

# #     batch_id = uuid.uuid4().hex[:12]
# #     job      = BatchJob(batch_id=batch_id, total=len(pending))

# #     with _batch_lock:
# #         _batch_jobs[batch_id] = job

# #     def _worker():
# #         folders: Dict[str, list] = defaultdict(list)
# #         for row in pending:
# #             folders[row["folder_name"]].append(row)

# #         job_index = 0
# #         for f_name, rows in folders.items():
# #             print(f"[AsyncWorker] Folder '{f_name}'  {len(rows)} video(s)")
# #             _process_folder(
# #                 folder_name     = f_name,
# #                 video_rows      = rows,
# #                 train_detail_id = train_detail_id,
# #                 job             = job,
# #                 job_index_start = job_index,
# #             )
# #             job_index += len(rows)

# #         job.finish()
# #         print(
# #             f"[AsyncWorker] ✅ Batch {batch_id} complete — "
# #             f"{job.completed} succeeded, {job.failed} failed."
# #         )

# #     threading.Thread(
# #         target = _worker,
# #         daemon = True,
# #         name   = f"PendingWorker-{batch_id}",
# #     ).start()

# #     return JSONResponse(
# #         status_code=202,
# #         content={
# #             "batch_id":     batch_id,
# #             "total_videos": len(pending),
# #             "status":       "running",
# #             "poll_url":     f"/batch-status/{batch_id}",
# #             "result_url":   f"/batch-result/{batch_id}",
# #             "message": (
# #                 f"{len(pending)} pending video(s) queued. "
# #                 f"Poll /batch-status/{batch_id} for progress."
# #             ),
# #         },
# #     )


# # # ─────────────────────────────────────────────────────────────────
# # # STATUS + RESULT
# # # ─────────────────────────────────────────────────────────────────

# # @app.get("/batch-status/{batch_id}", tags=["Processing"])
# # def batch_status(batch_id: str) -> JSONResponse:
# #     """Live progress for any batch job (blocking or async)."""
# #     with _batch_lock:
# #         job = _batch_jobs.get(batch_id)
# #     if not job:
# #         raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found.")
# #     return JSONResponse(content=job.to_status_dict())


# # @app.get("/batch-result/{batch_id}", tags=["Processing"])
# # def batch_result(batch_id: str) -> JSONResponse:
# #     """Full result. Returns 409 if the job is still running."""
# #     with _batch_lock:
# #         job = _batch_jobs.get(batch_id)
# #     if not job:
# #         raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found.")
# #     if job.status != "done":
# #         raise HTTPException(
# #             status_code=409,
# #             detail=(
# #                 f"Still running — {job.completed}/{job.total_videos} done. "
# #                 f"Poll /batch-status/{batch_id} first."
# #             ),
# #         )
# #     return JSONResponse(content=job.to_final_dict())


# # # ─────────────────────────────────────────────────────────────────
# # # EXISTING — ANALYZE BATCH direct upload (unchanged)
# # # ─────────────────────────────────────────────────────────────────

# # @app.post("/analyze-batch", tags=["Analysis"])
# # async def analyze_batch(
# #     videos:          List[UploadFile] = File(...),
# #     train_detail_id: int              = Form(default=0),
# #     train_label:     str              = Form(default=""),
# # ) -> JSONResponse:
# #     """
# #     Original direct-upload endpoint. Completely unchanged.
# #     Upload N videos, process immediately, return combined report.
# #     """
# #     if not videos:
# #         raise HTTPException(status_code=400, detail="No videos provided.")

# #     batch_id = uuid.uuid4().hex[:12]
# #     job      = BatchJob(batch_id=batch_id, total=len(videos))
# #     saved    = [(_save_upload(u), u.filename or "video") for u in videos]

# #     safe_label = train_label.strip().replace("/", "-").replace("\\", "-")
# #     batch_analysis_id = (
# #         safe_label if safe_label
# #         else (f"train_{train_detail_id}_{batch_id[:8]}" if train_detail_id
# #               else f"batch_{batch_id[:8]}")
# #     )

# #     shared_vstore = ViolationStore(
# #         analysis_id     = batch_analysis_id,
# #         train_detail_id = train_detail_id,
# #         video_info      = None,
# #     )

# #     import time as _time
# #     batch_start       = _time.monotonic()
# #     cumulative_time   = 0.0
# #     cumulative_frames = 0

# #     for i, (tmp_path, filename) in enumerate(saved):
# #         job.start_video(i, filename)
# #         try:
# #             _, duration, frame_count = _run_pipeline(
# #                 tmp_path          = tmp_path,
# #                 analysis_id       = batch_analysis_id,
# #                 train_detail_id   = train_detail_id,
# #                 time_offset       = cumulative_time,
# #                 frame_offset      = cumulative_frames,
# #                 shared_vstore     = shared_vstore,
# #                 original_filename = filename,
# #             )
# #             cumulative_time   += duration
# #             cumulative_frames += frame_count
# #             job.record_success(i, filename, duration)
# #         except Exception as exc:
# #             job.record_failure(i, filename, f"{exc}\n{traceback.format_exc()}")
# #         finally:
# #             try:
# #                 os.remove(tmp_path)
# #             except OSError:
# #                 pass

# #     report_path = shared_vstore.finalize(
# #         processing_time = _time.monotonic() - batch_start
# #     )
# #     final_report: dict = {}
# #     if report_path and os.path.isfile(report_path):
# #         with open(report_path, encoding="utf-8") as f:
# #             final_report = json.load(f)

# #     job.finish()
# #     return JSONResponse(content={**job.to_final_dict(), "report": final_report})


# # # ─────────────────────────────────────────────────────────────────
# # # EXISTING — ANALYZE BATCH ASYNC direct upload (unchanged)
# # # ─────────────────────────────────────────────────────────────────

# # @app.post("/analyze-batch-async", tags=["Analysis"])
# # async def analyze_batch_async(
# #     videos:          List[UploadFile] = File(...),
# #     train_detail_id: int              = Form(default=0),
# #     train_label:     str              = Form(default=""),
# # ) -> JSONResponse:
# #     """Original async direct-upload endpoint. Completely unchanged."""
# #     if not videos:
# #         raise HTTPException(status_code=400, detail="No videos provided.")

# #     saved    = [(_save_upload(u), u.filename or "video") for u in videos]
# #     batch_id = uuid.uuid4().hex[:12]
# #     job      = BatchJob(batch_id=batch_id, total=len(videos))

# #     with _batch_lock:
# #         _batch_jobs[batch_id] = job

# #     safe_label = train_label.strip().replace("/", "-").replace("\\", "-")
# #     batch_analysis_id = (
# #         safe_label if safe_label
# #         else (f"train_{train_detail_id}_{batch_id[:8]}" if train_detail_id
# #               else f"batch_{batch_id[:8]}")
# #     )

# #     shared_vstore = ViolationStore(
# #         analysis_id     = batch_analysis_id,
# #         train_detail_id = train_detail_id,
# #         video_info      = None,
# #     )

# #     def _worker():
# #         import time as _time
# #         batch_start       = _time.monotonic()
# #         cumulative_time   = 0.0
# #         cumulative_frames = 0
# #         for i, (tmp_path, filename) in enumerate(saved):
# #             job.start_video(i, filename)
# #             try:
# #                 _, duration, frame_count = _run_pipeline(
# #                     tmp_path          = tmp_path,
# #                     analysis_id       = batch_analysis_id,
# #                     train_detail_id   = train_detail_id,
# #                     time_offset       = cumulative_time,
# #                     frame_offset      = cumulative_frames,
# #                     shared_vstore     = shared_vstore,
# #                     original_filename = filename,
# #                 )
# #                 cumulative_time   += duration
# #                 cumulative_frames += frame_count
# #                 job.record_success(i, filename, duration)
# #             except Exception as exc:
# #                 job.record_failure(i, filename, f"{exc}\n{traceback.format_exc()}")
# #             finally:
# #                 try:
# #                     os.remove(tmp_path)
# #                 except OSError:
# #                     pass
# #         shared_vstore.finalize(processing_time=_time.monotonic() - batch_start)
# #         job.finish()

# #     threading.Thread(target=_worker, daemon=True, name=f"Batch-{batch_id}").start()

# #     return JSONResponse(
# #         status_code=202,
# #         content={
# #             "batch_id":     batch_id,
# #             "total_videos": len(videos),
# #             "status":       "running",
# #             "poll_url":     f"/batch-status/{batch_id}",
# #             "result_url":   f"/batch-result/{batch_id}",
# #         },
# #     )


# # # ─────────────────────────────────────────────────────────────────
# # # TEST PAGE  →  http://localhost:8000/test
# # # ─────────────────────────────────────────────────────────────────

# # @app.get("/test", response_class=HTMLResponse, tags=["Status"])
# # def test_page():
# #     html = """
# # <!DOCTYPE html>
# # <html lang="en">
# # <head>
# # <meta charset="UTF-8">
# # <title>Loco Pilot — Process Pending</title>
# # <style>
# #   body  { font-family: Arial, sans-serif; max-width: 680px; margin: 40px auto; padding: 0 20px; }
# #   h2    { color: #1a1a2e; }
# #   label { display:block; margin-top:12px; font-weight:bold; }
# #   input[type=number], input[type=text] { padding:6px; margin-top:4px; font-size:14px; }
# #   button { margin-top:14px; padding:10px 24px; background:#0a6ebd;
# #            color:#fff; border:none; border-radius:5px; cursor:pointer; font-size:14px; }
# #   button:hover { background:#084f8c; }
# #   .box  { margin-top:16px; padding:12px; white-space:pre-wrap; font-size:12px;
# #           max-height:400px; overflow-y:auto; }
# #   .info { background:#f0f4ff; border-left:4px solid #0a6ebd; }
# #   .ok   { background:#f6fff0; border-left:4px solid #2ecc71; }
# #   .hidden { display:none; }
# #   p.note { color:#666; font-size:13px; }
# # </style>
# # </head>
# # <body>
# # <h2>🚂 Loco Pilot — Process Pending Videos</h2>
# # <p class="note">
# #   Videos are already registered in DB with <code>process_flag='N'</code>
# #   (by frontend or <code>register_videos.py</code>).<br>
# #   Use this page to trigger processing.
# # </p>

# # <label>Train Detail ID</label>
# # <input type="number" id="tid" value="1" style="width:100px;">

# # <label>
# #   Folder name
# #   <span style="font-weight:normal">
# #     (optional — leave blank to process ALL pending for this train)
# #   </span>
# # </label>
# # <input type="text" id="folder" placeholder="e.g. 12345-27-05-2026" style="width:300px;">

# # <label>
# #   <input type="checkbox" id="async_mode">
# #   Use async mode (returns immediately — poll for result)
# # </label>

# # <button onclick="run()">▶ Process Pending</button>

# # <div id="status" class="box info hidden"></div>
# # <div id="result" class="box ok  hidden"></div>

# # <script>
# #   let pollTimer = null;

# #   async function run() {
# #     const tid    = document.getElementById('tid').value;
# #     const folder = document.getElementById('folder').value.trim();
# #     const async_ = document.getElementById('async_mode').checked;
# #     const ep     = async_ ? '/process-pending-async' : '/process-pending';

# #     const fd = new FormData();
# #     fd.append('train_detail_id', tid);
# #     if (folder) fd.append('folder_name', folder);

# #     show('status', `Sending to ${ep} ...`);
# #     document.getElementById('result').classList.add('hidden');

# #     try {
# #       const res  = await fetch(ep, { method: 'POST', body: fd });
# #       const data = await res.json();

# #       if (!async_) {
# #         show('status', '✅ Done');
# #         showResult(data);
# #       } else {
# #         show('status', `⏳ batch_id = ${data.batch_id}\nPolling every 4 seconds ...`);
# #         if (pollTimer) clearInterval(pollTimer);
# #         pollTimer = setInterval(() => poll(data.batch_id), 4000);
# #       }
# #     } catch(e) {
# #       show('status', '❌ Error: ' + e);
# #     }
# #   }

# #   async function poll(id) {
# #     try {
# #       const res  = await fetch('/batch-status/' + id);
# #       const data = await res.json();
# #       show('status',
# #         `⏳ ${data.completed} / ${data.total_videos} done  ` +
# #         `(${data.failed} failed)  status = ${data.status}\n` +
# #         `Current file: ${data.current_filename || '—'}`
# #       );
# #       if (data.status === 'done') {
# #         clearInterval(pollTimer);
# #         const r = await fetch('/batch-result/' + id);
# #         showResult(await r.json());
# #       }
# #     } catch(e) {
# #       show('status', '❌ Poll error: ' + e);
# #       clearInterval(pollTimer);
# #     }
# #   }

# #   function show(id, msg) {
# #     const el = document.getElementById(id);
# #     el.classList.remove('hidden');
# #     el.textContent = msg;
# #   }

# #   function showResult(data) {
# #     const el = document.getElementById('result');
# #     el.classList.remove('hidden');
# #     el.textContent = JSON.stringify(data, null, 2);
# #   }
# # </script>
# # </body>
# # </html>
# # """
# #     return HTMLResponse(content=html)

# """
# api.py

# Frontend inserts rows into video_files with process_flag = 'N'.

# POST /process-pending picks up ALL pending rows (no train_detail_id filter),
# groups them by (train_detail_id, folder_name), and processes each group
# sequentially.
# """

# from __future__ import annotations

# import json
# import os
# import shutil
# import tempfile
# import threading
# import traceback
# import uuid
# from collections import defaultdict
# from typing import Dict, List, Optional, Tuple

# from fastapi import FastAPI, File, Form, HTTPException, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import HTMLResponse, JSONResponse

# from main import GadgetDetectionPipeline
# from utils.violation_store import ViolationStore
# from utils.db_s3_uploader import (
#     get_pending_videos,
#     set_process_flag,
#     download_video_from_s3,
# )

# app = FastAPI(title="Loco Pilot Distraction Detection API", version="3.0.0")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# _batch_jobs: Dict[str, "BatchJob"] = {}
# _batch_lock = threading.Lock()


# # ─────────────────────────────────────────────────────────────────
# # BATCH JOB STATE
# # ─────────────────────────────────────────────────────────────────

# class BatchJob:
#     def __init__(self, batch_id: str, total: int) -> None:
#         self.batch_id          = batch_id
#         self.total_videos      = total
#         self.current_index     = 0
#         self.current_filename: Optional[str] = None
#         self.completed         = 0
#         self.failed            = 0
#         self.status            = "running"
#         self.results: List[dict] = []
#         self._lock             = threading.Lock()

#     def start_video(self, index: int, filename: str) -> None:
#         with self._lock:
#             self.current_index    = index
#             self.current_filename = filename

#     def record_success(self, index: int, filename: str, duration: float) -> None:
#         with self._lock:
#             self.results.append({
#                 "index": index, "filename": filename,
#                 "status": "success", "duration": duration, "error": None,
#             })
#             self.completed += 1

#     def record_failure(self, index: int, filename: str, error: str) -> None:
#         with self._lock:
#             self.results.append({
#                 "index": index, "filename": filename,
#                 "status": "failed", "duration": 0.0, "error": error,
#             })
#             self.failed += 1

#     def finish(self) -> None:
#         with self._lock:
#             self.status           = "done"
#             self.current_filename = None

#     def to_status_dict(self) -> dict:
#         with self._lock:
#             return {
#                 "batch_id":         self.batch_id,
#                 "total_videos":     self.total_videos,
#                 "current_index":    self.current_index,
#                 "current_filename": self.current_filename,
#                 "completed":        self.completed,
#                 "failed":           self.failed,
#                 "status":           self.status,
#             }

#     def to_final_dict(self) -> dict:
#         with self._lock:
#             return {
#                 "batch_id":     self.batch_id,
#                 "total_videos": self.total_videos,
#                 "completed":    self.completed,
#                 "failed":       self.failed,
#                 "status":       self.status,
#                 "results":      sorted(self.results, key=lambda r: r["index"]),
#             }


# # ─────────────────────────────────────────────────────────────────
# # INTERNAL: run one video through the pipeline
# # ─────────────────────────────────────────────────────────────────

# def _run_pipeline(
#     tmp_path:          str,
#     analysis_id:       str,
#     train_detail_id:   int,
#     time_offset:       float = 0.0,
#     frame_offset:      int   = 0,
#     shared_vstore:     Optional[ViolationStore] = None,
#     original_filename: Optional[str] = None,
# ) -> tuple:
#     pipeline = GadgetDetectionPipeline(
#         source            = tmp_path,
#         analysis_id       = analysis_id,
#         train_detail_id   = train_detail_id,
#         save              = False,
#         display           = False,
#         time_offset       = time_offset,
#         frame_offset      = frame_offset,
#         shared_vstore     = shared_vstore,
#         original_filename = original_filename,
#     )
#     return pipeline.run()


# # ─────────────────────────────────────────────────────────────────
# # CORE: process all videos in ONE folder sequentially
# # ─────────────────────────────────────────────────────────────────

# def _process_folder(
#     folder_name:     str,
#     train_detail_id: int,
#     video_rows:      List[dict],   # already ordered by seq_no
#     job:             BatchJob,
#     job_index_start: int = 0,
# ) -> dict:
#     """
#     Processes every video in this folder one by one in seq_no order.

#     For each video:
#         flag N → I   (before pipeline)
#         flag I → Y   (after success)
#         flag stays I on failure — operator must review

#     After all videos:
#         ViolationStore.finalize() → writes analysis_report.json locally
#         → calls finalize_and_upload() → uploads to S3 → saves result_s3_path in DB
#     """
#     import time as _time

#     shared_vstore = ViolationStore(
#         analysis_id     = folder_name,
#         train_detail_id = train_detail_id,
#         video_info      = None,
#     )

#     batch_start       = _time.monotonic()
#     cumulative_time   = 0.0
#     cumulative_frames = 0

#     for i, row in enumerate(video_rows):
#         video_id      = row["id"]
#         filename      = row["filename"]
#         s3_video_path = row["s3_video_path"]
#         global_index  = job_index_start + i

#         job.start_video(global_index, filename)

#         print(f"\n[{folder_name}] ▶ {i+1}/{len(video_rows)}  "
#               f"seq={row['seq_no']}  {filename}")

#         # flag → I
#         set_process_flag(video_id, "I")

#         suffix   = os.path.splitext(filename)[1] or ".mp4"
#         tmp_path = tempfile.mktemp(suffix=suffix)

#         try:
#             download_video_from_s3(s3_video_path, tmp_path)

#             _, duration, frame_count = _run_pipeline(
#                 tmp_path          = tmp_path,
#                 analysis_id       = folder_name,
#                 train_detail_id   = train_detail_id,
#                 time_offset       = cumulative_time,
#                 frame_offset      = cumulative_frames,
#                 shared_vstore     = shared_vstore,
#                 original_filename = filename,
#             )

#             cumulative_time   += duration
#             cumulative_frames += frame_count

#             # flag → Y
#             set_process_flag(video_id, "Y")
#             job.record_success(global_index, filename, duration)
#             print(f"[{folder_name}] ✓ {filename}  {duration:.1f}s")

#         except Exception as exc:
#             # Flag stays 'I' — intentional
#             job.record_failure(global_index, filename, str(exc))
#             print(f"[{folder_name}] ✗ FAILED — flag stays 'I': {exc}")

#         finally:
#             try:
#                 os.remove(tmp_path)
#             except OSError:
#                 pass

#     # finalize → writes JSON + frames + uploads to S3
#     report_path = shared_vstore.finalize(
#         processing_time = _time.monotonic() - batch_start
#     )

#     final_report: dict = {}
#     if report_path and os.path.isfile(report_path):
#         with open(report_path, encoding="utf-8") as f:
#             final_report = json.load(f)

#     return final_report


# # ─────────────────────────────────────────────────────────────────
# # SHARED LOGIC: build groups from pending rows
# # ─────────────────────────────────────────────────────────────────

# def _build_groups(
#     pending: List[dict],
# ) -> Dict[Tuple[int, str], List[dict]]:
#     """
#     Group pending rows by (train_detail_id, folder_name).

#     Example — if DB has:
#         train=1  folder=22801-05-06-2026  seq=1  ch01.mp4
#         train=1  folder=22801-05-06-2026  seq=2  ch02.mp4
#         train=2  folder=12345-01-06-2026  seq=1  ch01.mp4

#     Returns:
#         {
#           (1, "22801-05-06-2026"): [row1, row2],
#           (2, "12345-01-06-2026"): [row3],
#         }

#     Rows within each group are already in seq_no order
#     (guaranteed by ORDER BY in get_pending_videos).
#     """
#     groups: Dict[Tuple[int, str], List[dict]] = defaultdict(list)
#     for row in pending:
#         key = (row["train_detail_id"], row["folder_name"])
#         groups[key].append(row)
#     return groups


# # ─────────────────────────────────────────────────────────────────
# # HEALTH
# # ─────────────────────────────────────────────────────────────────

# @app.get("/health", tags=["Status"])
# def health() -> dict:
#     return {"status": "ok"}


# # ─────────────────────────────────────────────────────────────────
# # PROCESS PENDING — blocking
# # ─────────────────────────────────────────────────────────────────

# @app.post("/process-pending", tags=["Processing"])
# async def process_pending() -> JSONResponse:
#     """
#     Picks up ALL rows where process_flag = 'N' across every train.
#     Groups them by (train_detail_id, folder_name).
#     Processes each group's videos sequentially by seq_no.
#     Blocking — waits until everything is done then returns.

#     No parameters needed — just POST to this endpoint.
#     """
#     pending = get_pending_videos()

#     if not pending:
#         return JSONResponse(content={
#             "message": "No pending videos found (process_flag='N').",
#         })

#     groups = _build_groups(pending)

#     print(f"\n[ProcessPending] {len(pending)} video(s) across "
#           f"{len(groups)} folder(s):")
#     for (tid, fname), rows in groups.items():
#         print(f"  train={tid}  folder={fname}  videos={len(rows)}")

#     batch_id = uuid.uuid4().hex[:12]
#     job      = BatchJob(batch_id=batch_id, total=len(pending))
#     with _batch_lock:
#         _batch_jobs[batch_id] = job

#     folder_results = []
#     job_index      = 0

#     for (train_id, f_name), rows in groups.items():
#         print(f"\n[ProcessPending] ═══ train={train_id}  folder={f_name} ═══")
#         report = _process_folder(
#             folder_name     = f_name,
#             train_detail_id = train_id,
#             video_rows      = rows,
#             job             = job,
#             job_index_start = job_index,
#         )
#         folder_results.append({
#             "train_detail_id":  train_id,
#             "folder_name":      f_name,
#             "videos_in_folder": len(rows),
#             "report":           report,
#         })
#         job_index += len(rows)

#     job.finish()

#     return JSONResponse(content={
#         "batch_id":          batch_id,
#         "total_videos":      len(pending),
#         "completed":         job.completed,
#         "failed":            job.failed,
#         "folders_processed": len(folder_results),
#         "folders":           folder_results,
#     })


# # ─────────────────────────────────────────────────────────────────
# # PROCESS PENDING ASYNC — non-blocking
# # ─────────────────────────────────────────────────────────────────

# @app.post("/process-pending-async", tags=["Processing"])
# async def process_pending_async() -> JSONResponse:
#     """
#     Same as /process-pending but returns immediately with a batch_id.
#     Poll GET /batch-status/{batch_id} for progress.
#     GET  /batch-result/{batch_id} when status = 'done'.
#     """
#     pending = get_pending_videos()

#     if not pending:
#         return JSONResponse(content={
#             "message": "No pending videos found.",
#         })

#     groups   = _build_groups(pending)
#     batch_id = uuid.uuid4().hex[:12]
#     job      = BatchJob(batch_id=batch_id, total=len(pending))
#     with _batch_lock:
#         _batch_jobs[batch_id] = job

#     def _worker():
#         job_index = 0
#         for (train_id, f_name), rows in groups.items():
#             print(f"\n[AsyncWorker] ═══ train={train_id}  folder={f_name} ═══")
#             _process_folder(
#                 folder_name     = f_name,
#                 train_detail_id = train_id,
#                 video_rows      = rows,
#                 job             = job,
#                 job_index_start = job_index,
#             )
#             job_index += len(rows)

#         job.finish()
#         print(f"\n[AsyncWorker] ✅ batch={batch_id}  "
#               f"done={job.completed}  failed={job.failed}")

#     threading.Thread(
#         target = _worker,
#         daemon = True,
#         name   = f"PendingWorker-{batch_id}",
#     ).start()

#     return JSONResponse(
#         status_code=202,
#         content={
#             "batch_id":      batch_id,
#             "total_videos":  len(pending),
#             "total_folders": len(groups),
#             "status":        "running",
#             "poll_url":      f"/batch-status/{batch_id}",
#             "result_url":    f"/batch-result/{batch_id}",
#             "message":       f"{len(pending)} video(s) across {len(groups)} folder(s) queued.",
#         },
#     )


# # ─────────────────────────────────────────────────────────────────
# # STATUS + RESULT
# # ─────────────────────────────────────────────────────────────────

# @app.get("/batch-status/{batch_id}", tags=["Processing"])
# def batch_status(batch_id: str) -> JSONResponse:
#     with _batch_lock:
#         job = _batch_jobs.get(batch_id)
#     if not job:
#         raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found.")
#     return JSONResponse(content=job.to_status_dict())


# @app.get("/batch-result/{batch_id}", tags=["Processing"])
# def batch_result(batch_id: str) -> JSONResponse:
#     with _batch_lock:
#         job = _batch_jobs.get(batch_id)
#     if not job:
#         raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found.")
#     if job.status != "done":
#         raise HTTPException(
#             status_code=409,
#             detail=f"Still running — {job.completed}/{job.total_videos} done.",
#         )
#     return JSONResponse(content=job.to_final_dict())


# # ─────────────────────────────────────────────────────────────────
# # EXISTING — ANALYZE BATCH direct upload (unchanged)
# # ─────────────────────────────────────────────────────────────────

# def _save_upload(upload: UploadFile) -> str:
#     suffix = os.path.splitext(upload.filename or "video.mp4")[1] or ".mp4"
#     fd, path = tempfile.mkstemp(suffix=suffix)
#     with os.fdopen(fd, "wb") as f:
#         shutil.copyfileobj(upload.file, f)
#     return path


# @app.post("/analyze-batch", tags=["Analysis"])
# async def analyze_batch(
#     videos:          List[UploadFile] = File(...),
#     train_detail_id: int              = Form(default=0),
#     train_label:     str              = Form(default=""),
# ) -> JSONResponse:
#     if not videos:
#         raise HTTPException(status_code=400, detail="No videos provided.")

#     batch_id = uuid.uuid4().hex[:12]
#     job      = BatchJob(batch_id=batch_id, total=len(videos))
#     saved    = [(_save_upload(u), u.filename or "video") for u in videos]

#     safe_label = train_label.strip().replace("/", "-").replace("\\", "-")
#     batch_analysis_id = (
#         safe_label if safe_label
#         else (f"train_{train_detail_id}_{batch_id[:8]}" if train_detail_id
#               else f"batch_{batch_id[:8]}")
#     )
#     shared_vstore = ViolationStore(
#         analysis_id     = batch_analysis_id,
#         train_detail_id = train_detail_id,
#         video_info      = None,
#     )

#     import time as _time
#     batch_start       = _time.monotonic()
#     cumulative_time   = 0.0
#     cumulative_frames = 0

#     for i, (tmp_path, filename) in enumerate(saved):
#         job.start_video(i, filename)
#         try:
#             _, duration, frame_count = _run_pipeline(
#                 tmp_path          = tmp_path,
#                 analysis_id       = batch_analysis_id,
#                 train_detail_id   = train_detail_id,
#                 time_offset       = cumulative_time,
#                 frame_offset      = cumulative_frames,
#                 shared_vstore     = shared_vstore,
#                 original_filename = filename,
#             )
#             cumulative_time   += duration
#             cumulative_frames += frame_count
#             job.record_success(i, filename, duration)
#         except Exception as exc:
#             job.record_failure(i, filename, f"{exc}\n{traceback.format_exc()}")
#         finally:
#             try:
#                 os.remove(tmp_path)
#             except OSError:
#                 pass

#     report_path = shared_vstore.finalize(
#         processing_time = _time.monotonic() - batch_start
#     )
#     final_report: dict = {}
#     if report_path and os.path.isfile(report_path):
#         with open(report_path, encoding="utf-8") as f:
#             final_report = json.load(f)

#     job.finish()
#     return JSONResponse(content={**job.to_final_dict(), "report": final_report})


# @app.post("/analyze-batch-async", tags=["Analysis"])
# async def analyze_batch_async(
#     videos:          List[UploadFile] = File(...),
#     train_detail_id: int              = Form(default=0),
#     train_label:     str              = Form(default=""),
# ) -> JSONResponse:
#     if not videos:
#         raise HTTPException(status_code=400, detail="No videos provided.")

#     saved    = [(_save_upload(u), u.filename or "video") for u in videos]
#     batch_id = uuid.uuid4().hex[:12]
#     job      = BatchJob(batch_id=batch_id, total=len(videos))
#     with _batch_lock:
#         _batch_jobs[batch_id] = job

#     safe_label = train_label.strip().replace("/", "-").replace("\\", "-")
#     batch_analysis_id = (
#         safe_label if safe_label
#         else (f"train_{train_detail_id}_{batch_id[:8]}" if train_detail_id
#               else f"batch_{batch_id[:8]}")
#     )
#     shared_vstore = ViolationStore(
#         analysis_id     = batch_analysis_id,
#         train_detail_id = train_detail_id,
#         video_info      = None,
#     )

#     def _worker():
#         import time as _time
#         batch_start       = _time.monotonic()
#         cumulative_time   = 0.0
#         cumulative_frames = 0
#         for i, (tmp_path, filename) in enumerate(saved):
#             job.start_video(i, filename)
#             try:
#                 _, duration, frame_count = _run_pipeline(
#                     tmp_path          = tmp_path,
#                     analysis_id       = batch_analysis_id,
#                     train_detail_id   = train_detail_id,
#                     time_offset       = cumulative_time,
#                     frame_offset      = cumulative_frames,
#                     shared_vstore     = shared_vstore,
#                     original_filename = filename,
#                 )
#                 cumulative_time   += duration
#                 cumulative_frames += frame_count
#                 job.record_success(i, filename, duration)
#             except Exception as exc:
#                 job.record_failure(i, filename, f"{exc}\n{traceback.format_exc()}")
#             finally:
#                 try:
#                     os.remove(tmp_path)
#                 except OSError:
#                     pass
#         shared_vstore.finalize(processing_time=_time.monotonic() - batch_start)
#         job.finish()

#     threading.Thread(target=_worker, daemon=True, name=f"Batch-{batch_id}").start()
#     return JSONResponse(
#         status_code=202,
#         content={
#             "batch_id":     batch_id,
#             "total_videos": len(videos),
#             "status":       "running",
#             "poll_url":     f"/batch-status/{batch_id}",
#             "result_url":   f"/batch-result/{batch_id}",
#         },
#     )


# # ─────────────────────────────────────────────────────────────────
# # TEST PAGE
# # ─────────────────────────────────────────────────────────────────

# @app.get("/test", response_class=HTMLResponse, tags=["Status"])
# def test_page():
#     html = """
# <!DOCTYPE html>
# <html lang="en">
# <head>
# <meta charset="UTF-8">
# <title>Loco Pilot — Process Pending</title>
# <style>
#   body  { font-family: Arial, sans-serif; max-width: 620px; margin: 40px auto; padding: 0 20px; }
#   h2    { color: #1a1a2e; }
#   label { display:block; margin-top:12px; font-weight:bold; }
#   button { margin-top:14px; padding:10px 28px; background:#0a6ebd;
#            color:#fff; border:none; border-radius:5px; cursor:pointer; font-size:14px; }
#   button:hover { background:#084f8c; }
#   .box  { margin-top:16px; padding:12px; white-space:pre-wrap; font-size:12px;
#           max-height:420px; overflow-y:auto; border-radius:4px; }
#   .info { background:#f0f4ff; border-left:4px solid #0a6ebd; }
#   .ok   { background:#f6fff0; border-left:4px solid #2ecc71; }
#   .hidden { display:none; }
# </style>
# </head>
# <body>
# <h2>🚂 Process Pending Videos</h2>
# <p style="color:#555;font-size:13px;">
#   Picks up ALL <code>process_flag='N'</code> rows from the DB,
#   groups them by folder, and processes each folder sequentially.
#   No parameters needed.
# </p>

# <label>
#   <input type="checkbox" id="async_mode">
#   Async mode (returns immediately — poll for result)
# </label>

# <button onclick="run()">▶ Process All Pending</button>

# <div id="status" class="box info hidden"></div>
# <div id="result" class="box ok  hidden"></div>

# <script>
#   let pollTimer = null;

#   async function run() {
#     const async_ = document.getElementById('async_mode').checked;
#     const ep     = async_ ? '/process-pending-async' : '/process-pending';

#     show('status', 'Sending to ' + ep + ' ...');
#     document.getElementById('result').classList.add('hidden');

#     try {
#       const res  = await fetch(ep, { method: 'POST' });
#       const data = await res.json();
#       if (!async_) {
#         show('status',
#           '✅ Done — ' + (data.completed||0) + ' succeeded, ' +
#           (data.failed||0) + ' failed across ' +
#           (data.folders_processed||0) + ' folder(s).');
#         showResult(data);
#       } else {
#         show('status',
#           '⏳ batch_id = ' + data.batch_id +
#           '\\n' + data.message +
#           '\\nPolling every 4s ...');
#         if (pollTimer) clearInterval(pollTimer);
#         pollTimer = setInterval(() => poll(data.batch_id), 4000);
#       }
#     } catch(e) { show('status', '❌ ' + e); }
#   }

#   async function poll(id) {
#     try {
#       const res  = await fetch('/batch-status/' + id);
#       const data = await res.json();
#       show('status',
#         '⏳ ' + data.completed + ' / ' + data.total_videos + ' done  ' +
#         '(' + data.failed + ' failed)  status=' + data.status +
#         '\\nCurrent: ' + (data.current_filename || '—'));
#       if (data.status === 'done') {
#         clearInterval(pollTimer);
#         const r = await fetch('/batch-result/' + id);
#         showResult(await r.json());
#       }
#     } catch(e) {
#       show('status', '❌ Poll error: ' + e);
#       clearInterval(pollTimer);
#     }
#   }

#   function show(id, msg) {
#     const el = document.getElementById(id);
#     el.classList.remove('hidden');
#     el.textContent = msg;
#   }
#   function showResult(data) {
#     const el = document.getElementById('result');
#     el.classList.remove('hidden');
#     el.textContent = JSON.stringify(data, null, 2);
#   }
# </script>
# </body>
# </html>
# """
#     return HTMLResponse(content=html)



# # # # # # from __future__ import annotations

# # # # # # """
# # # # # # api.py — Loco Pilot Distraction Detection  (DB-driven batch runner)
# # # # # # ====================================================================

# # # # # # Videos are NO LONGER uploaded through this API.
# # # # # # Instead, this service polls the database for video_files rows where
# # # # # # process_flag = 'N', downloads each video from S3, runs the detection
# # # # # # pipeline, and marks the row done (process_flag = 'Y').

# # # # # # Flag lifecycle
# # # # # # --------------
# # # # # #   N  →  pending (set by the frontend / ingestion service when uploading)
# # # # # #   I  →  in-progress (set here, immediately before the pipeline starts)
# # # # # #   Y  →  done (set here, only on successful pipeline completion)

# # # # # # If the pipeline crashes the flag stays at 'I' so operators can see
# # # # # # exactly which video failed — it is NOT silently reset to 'N'.

# # # # # # Endpoints
# # # # # # ---------
# # # # # #   GET  /              — health / welcome
# # # # # #   GET  /health        — {"status": "ok"}
# # # # # #   POST /trigger       — kick off a DB-scan + batch run immediately
# # # # # #                         (returns job_id; poll /status/<job_id>)
# # # # # #   GET  /status/<id>   — queued | processing | done | failed
# # # # # #   GET  /result/<id>   — final JSON report (consumed once; deleted from memory)
# # # # # # """

# # # # # # import json
# # # # # # import os
# # # # # # import tempfile
# # # # # # import traceback
# # # # # # import uuid
# # # # # # from concurrent.futures import ThreadPoolExecutor
# # # # # # from itertools import groupby
# # # # # # from typing import Any, Dict, List, Optional

# # # # # # from fastapi import BackgroundTasks, FastAPI, HTTPException
# # # # # # from fastapi.middleware.cors import CORSMiddleware
# # # # # # from fastapi.responses import JSONResponse

# # # # # # from main import GadgetDetectionPipeline
# # # # # # from utils.db_s3_uploader import (
# # # # # #     download_video_from_s3,
# # # # # #     get_pending_videos,
# # # # # #     set_process_flag,
# # # # # # )

# # # # # # # ── App setup ──────────────────────────────────────────────────────────────────

# # # # # # app = FastAPI(
# # # # # #     title   = "Loco Pilot Distraction Detection API",
# # # # # #     version = "3.0.0",
# # # # # # )

# # # # # # app.add_middleware(
# # # # # #     CORSMiddleware,
# # # # # #     allow_origins     = ["*"],
# # # # # #     allow_credentials = True,
# # # # # #     allow_methods     = ["*"],
# # # # # #     allow_headers     = ["*"],
# # # # # # )

# # # # # # # In-memory job registry  {job_id: {"status": str, "result": dict|None, "error": str|None}}
# # # # # # _jobs:    Dict[str, Dict[str, Any]] = {}
# # # # # # _executor = ThreadPoolExecutor(max_workers=1)  # one batch at a time


# # # # # # # ── Routes ─────────────────────────────────────────────────────────────────────

# # # # # # @app.get("/")
# # # # # # def root() -> dict:
# # # # # #     return {
# # # # # #         "status":  "success",
# # # # # #         "message": "Loco Pilot Distraction Detection API is running",
# # # # # #         "health":  "/health",
# # # # # #         "docs":    "/docs",
# # # # # #         "trigger": "POST /trigger  — start a DB-scan + batch run",
# # # # # #     }


# # # # # # @app.get("/health", tags=["status"])
# # # # # # def health() -> dict:
# # # # # #     return {"status": "ok"}


# # # # # # # ── Batch trigger ──────────────────────────────────────────────────────────────

# # # # # # @app.post("/trigger", tags=["batch"])
# # # # # # async def trigger_batch(background_tasks: BackgroundTasks) -> JSONResponse:
# # # # # #     """
# # # # # #     Scan the DB for all pending videos (process_flag='N') and process them
# # # # # #     as a batch.  Returns immediately with a job_id; poll /status/<job_id>.
# # # # # #     """
# # # # # #     job_id = str(uuid.uuid4())
# # # # # #     _jobs[job_id] = {"status": "queued", "result": None, "error": None}
# # # # # #     _executor.submit(_run_batch, job_id)
# # # # # #     return JSONResponse(
# # # # # #         status_code = 202,
# # # # # #         content = {
# # # # # #             "job_id":  job_id,
# # # # # #             "status":  "queued",
# # # # # #             "message": f"Batch job accepted. Poll GET /status/{job_id} for progress.",
# # # # # #         },
# # # # # #     )


# # # # # # @app.get("/status/{job_id}", tags=["batch"])
# # # # # # def job_status(job_id: str) -> JSONResponse:
# # # # # #     job = _jobs.get(job_id)
# # # # # #     if job is None:
# # # # # #         raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
# # # # # #     resp: Dict[str, Any] = {"job_id": job_id, "status": job["status"]}
# # # # # #     if job["status"] == "failed":
# # # # # #         resp["error"] = job["error"]
# # # # # #     return JSONResponse(content=resp)


# # # # # # @app.get("/result/{job_id}", tags=["batch"])
# # # # # # def job_result(job_id: str) -> JSONResponse:
# # # # # #     job = _jobs.get(job_id)
# # # # # #     if job is None:
# # # # # #         raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
# # # # # #     if job["status"] == "failed":
# # # # # #         raise HTTPException(status_code=500, detail=job["error"])
# # # # # #     if job["status"] in ("queued", "processing"):
# # # # # #         raise HTTPException(
# # # # # #             status_code=409,
# # # # # #             detail=f"Job is still '{job['status']}'. Try again later.",
# # # # # #         )
# # # # # #     result = _jobs.pop(job_id)["result"]
# # # # # #     return JSONResponse(content=result)


# # # # # # # ── Batch worker ───────────────────────────────────────────────────────────────

# # # # # # def _run_batch(job_id: str) -> None:
# # # # # #     """
# # # # # #     Entry point executed in the thread pool.

# # # # # #     1. Query DB for all rows with process_flag = 'N'.
# # # # # #     2. Group them by (train_detail_id, folder_name) — each group is one
# # # # # #        logical analysis (the videos in a folder are one continuous recording).
# # # # # #     3. For every group:
# # # # # #          a. Mark every row as 'I' (in-progress).
# # # # # #          b. Download each video from S3 to a temp file.
# # # # # #          c. Run GadgetDetectionPipeline over every video in sequence.
# # # # # #          d. On success: mark every row as 'Y' and collect the report.
# # # # # #          e. On failure: leave rows at 'I' (flag stays for operator inspection).
# # # # # #     4. Collect per-group reports and write them to the job registry.
# # # # # #     """
# # # # # #     _jobs[job_id]["status"] = "processing"
# # # # # #     all_reports: List[Dict[str, Any]] = []

# # # # # #     try:
# # # # # #         pending = get_pending_videos()

# # # # # #         if not pending:
# # # # # #             print("[Batch] No pending videos found.")
# # # # # #             _jobs[job_id]["status"] = "done"
# # # # # #             _jobs[job_id]["result"] = {"message": "No pending videos.", "reports": []}
# # # # # #             return

# # # # # #         # Group by (train_detail_id, folder_name) — same order as the DB query
# # # # # #         def _group_key(row: Dict[str, Any]):
# # # # # #             return (row["train_detail_id"], row["folder_name"])

# # # # # #         groups = [
# # # # # #             (key, list(rows))
# # # # # #             for key, rows in groupby(pending, key=_group_key)
# # # # # #         ]
# # # # # #         print(f"[Batch] {len(pending)} video(s) across {len(groups)} folder group(s).")

# # # # # #         for (train_detail_id, folder_name), rows in groups:
# # # # # #             report = _process_folder_group(
# # # # # #                 train_detail_id = train_detail_id,
# # # # # #                 folder_name     = folder_name,
# # # # # #                 rows            = rows,
# # # # # #             )
# # # # # #             if report:
# # # # # #                 all_reports.append(report)

# # # # # #         _jobs[job_id]["status"] = "done"
# # # # # #         _jobs[job_id]["result"] = {
# # # # # #             "message": f"Processed {len(all_reports)} folder group(s).",
# # # # # #             "reports": all_reports,
# # # # # #         }

# # # # # #     except Exception as exc:
# # # # # #         _jobs[job_id]["status"] = "failed"
# # # # # #         _jobs[job_id]["error"]  = f"{exc}\n{traceback.format_exc()}"
# # # # # #         print(f"[Batch] Fatal error: {exc}")


# # # # # # def _process_folder_group(
# # # # # #     train_detail_id: int,
# # # # # #     folder_name:     str,
# # # # # #     rows:            List[Dict[str, Any]],
# # # # # # ) -> Optional[Dict[str, Any]]:
# # # # # #     """
# # # # # #     Process one folder group (all videos that belong to a single analysis).

# # # # # #     Rows are already ordered by seq_no (guaranteed by the DB query).
# # # # # #     Each video is processed in sequence so frame offsets accumulate
# # # # # #     correctly across files.

# # # # # #     Returns the JSON report dict on success, None on failure.
# # # # # #     """
# # # # # #     print(
# # # # # #         f"\n[Batch] ── Folder group: train={train_detail_id}  "
# # # # # #         f"folder='{folder_name}'  ({len(rows)} video(s)) ──"
# # # # # #     )

# # # # # #     # Use folder_name as the analysis_id (unique per recording session)
# # # # # #     analysis_id = folder_name

# # # # # #     # ── Step 1: mark all rows as in-progress ─────────────────────────────────
# # # # # #     for row in rows:
# # # # # #         try:
# # # # # #             set_process_flag(row["id"], "I")
# # # # # #         except Exception as exc:
# # # # # #             print(f"[Batch] Could not mark row {row['id']} as 'I': {exc}")
# # # # # #             # Non-fatal — continue; the row stays 'N' which is safer than crashing

# # # # # #     # ── Step 2: download videos to temp files ─────────────────────────────────
# # # # # #     tmp_paths: List[str] = []
# # # # # #     try:
# # # # # #         for row in rows:
# # # # # #             suffix    = os.path.splitext(row["filename"] or "video.mp4")[1] or ".mp4"
# # # # # #             tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
# # # # # #             os.close(tmp_fd)
# # # # # #             download_video_from_s3(row["s3_video_path"], tmp_path)
# # # # # #             tmp_paths.append(tmp_path)
# # # # # #     except Exception as exc:
# # # # # #         print(f"[Batch] Download failed for folder '{folder_name}': {exc}")
# # # # # #         _cleanup_temps(tmp_paths)
# # # # # #         return None

# # # # # #     # ── Step 3: run the pipeline over each video in sequence ──────────────────
# # # # # #     report_path: str = ""
# # # # # #     try:
# # # # # #         # The pipeline processes one video at a time but shares the same
# # # # # #         # ViolationStore (via analysis_id = folder_name) so violations from
# # # # # #         # every video land in the same report.
# # # # # #         for idx, (row, tmp_path) in enumerate(zip(rows, tmp_paths)):
# # # # # #             print(
# # # # # #                 f"[Batch]   [{idx+1}/{len(rows)}] Processing  "
# # # # # #                 f"{row['filename']}  (seq={row['seq_no']})"
# # # # # #             )
# # # # # #             pipeline = GadgetDetectionPipeline(
# # # # # #                 source          = tmp_path,
# # # # # #                 analysis_id     = analysis_id,
# # # # # #                 train_detail_id = train_detail_id,
# # # # # #                 save            = False,
# # # # # #                 display         = False,
# # # # # #             )
# # # # # #             # run() returns the path to analysis_report.json
# # # # # #             report_path = pipeline.run()

# # # # # #         # ── Step 4: mark all rows as done ────────────────────────────────────
# # # # # #         for row in rows:
# # # # # #             try:
# # # # # #                 set_process_flag(row["id"], "Y")
# # # # # #             except Exception as exc:
# # # # # #                 print(f"[Batch] Could not mark row {row['id']} as 'Y': {exc}")

# # # # # #         # ── Step 5: read and return the report ───────────────────────────────
# # # # # #         if report_path and os.path.isfile(report_path):
# # # # # #             with open(report_path, encoding="utf-8") as f:
# # # # # #                 return json.load(f)
# # # # # #         else:
# # # # # #             print(f"[Batch] Report file missing for folder '{folder_name}'")
# # # # # #             return None

# # # # # #     except Exception as exc:
# # # # # #         # Leave flags at 'I' so the operator can see which group failed
# # # # # #         print(
# # # # # #             f"[Batch] Pipeline error for folder '{folder_name}': {exc}\n"
# # # # # #             + traceback.format_exc()
# # # # # #         )
# # # # # #         return None

# # # # # #     finally:
# # # # # #         _cleanup_temps(tmp_paths)


# # # # # # def _cleanup_temps(paths: List[str]) -> None:
# # # # # #     for p in paths:
# # # # # #         try:
# # # # # #             if os.path.isfile(p):
# # # # # #                 os.remove(p)
# # # # # #         except OSError:
# # # # # #             pass



# # # # # from __future__ import annotations

# # # # # """
# # # # # api.py — Loco Pilot Distraction Detection  (DB-driven batch runner)
# # # # # ====================================================================

# # # # # Videos are NOT uploaded through this API.
# # # # # This service polls the database for video_files rows where
# # # # # process_flag = 'N', downloads each video from S3, runs the detection
# # # # # pipeline synchronously, and returns the final result directly.

# # # # # Flag lifecycle
# # # # # --------------
# # # # #   N  →  pending   (set by the frontend when a video is uploaded to S3)
# # # # #   I  →  in-progress (set here before the pipeline starts)
# # # # #   Y  →  done      (set here only on successful pipeline completion)

# # # # # If the pipeline crashes the flag stays at 'I' so operators can see
# # # # # exactly which video failed.

# # # # # Endpoints
# # # # # ---------
# # # # #   GET  /         — health / welcome
# # # # #   GET  /health   — {"status": "ok"}
# # # # #   POST /trigger  — scan DB → download from S3 → run pipeline → return result
# # # # # """

# # # # # import json
# # # # # import os
# # # # # import tempfile
# # # # # import traceback
# # # # # from itertools import groupby
# # # # # from typing import Any, Dict, List, Optional

# # # # # from fastapi import FastAPI, HTTPException
# # # # # from fastapi.middleware.cors import CORSMiddleware
# # # # # from fastapi.responses import JSONResponse

# # # # # from main import GadgetDetectionPipeline
# # # # # from utils.db_s3_uploader import (
# # # # #     download_video_from_s3,
# # # # #     get_pending_videos,
# # # # #     set_process_flag,
# # # # # )

# # # # # # ── App setup ──────────────────────────────────────────────────────────────────

# # # # # app = FastAPI(
# # # # #     title   = "Loco Pilot Distraction Detection API",
# # # # #     version = "3.0.0",
# # # # # )

# # # # # app.add_middleware(
# # # # #     CORSMiddleware,
# # # # #     allow_origins     = ["*"],
# # # # #     allow_credentials = True,
# # # # #     allow_methods     = ["*"],
# # # # #     allow_headers     = ["*"],
# # # # # )


# # # # # # ── Routes ─────────────────────────────────────────────────────────────────────

# # # # # @app.get("/")
# # # # # def root() -> dict:
# # # # #     return {
# # # # #         "status":  "success",
# # # # #         "message": "Loco Pilot Distraction Detection API is running",
# # # # #         "health":  "/health",
# # # # #         "docs":    "/docs",
# # # # #         "trigger": "POST /trigger  — scan DB, process all pending videos, return results",
# # # # #     }


# # # # # @app.get("/health", tags=["status"])
# # # # # def health() -> dict:
# # # # #     return {"status": "ok"}


# # # # # # ── Main trigger ───────────────────────────────────────────────────────────────

# # # # # @app.post("/trigger", tags=["batch"])
# # # # # def trigger_batch() -> JSONResponse:
# # # # #     """
# # # # #     Scan DB for all rows with process_flag = 'N'.
# # # # #     Group by (train_detail_id, folder_name) — each group is one logical
# # # # #     analysis (a folder of sequential videos from one recording session).
# # # # #     Process every group in sequence, then return all reports together.

# # # # #     Flag lifecycle per video row:
# # # # #       N  →  I  (before pipeline starts)
# # # # #       I  →  Y  (after pipeline succeeds)
# # # # #       stays I   (if pipeline fails — visible to operators)
# # # # #     """
# # # # #     try:
# # # # #         pending = get_pending_videos()
# # # # #     except Exception as exc:
# # # # #         raise HTTPException(
# # # # #             status_code=500,
# # # # #             detail=f"Failed to query pending videos from DB: {exc}",
# # # # #         )

# # # # #     if not pending:
# # # # #         return JSONResponse(content={
# # # # #             "status":  "ok",
# # # # #             "message": "No pending videos found (process_flag = 'N').",
# # # # #             "reports": [],
# # # # #         })

# # # # #     def _group_key(row: Dict[str, Any]):
# # # # #         return (row["train_detail_id"], row["folder_name"])

# # # # #     groups = [
# # # # #         (key, list(rows))
# # # # #         for key, rows in groupby(pending, key=_group_key)
# # # # #     ]
# # # # #     print(f"[Batch] {len(pending)} video(s) across {len(groups)} folder group(s).")

# # # # #     all_reports: List[Dict[str, Any]] = []
# # # # #     errors:      List[str]            = []

# # # # #     for (train_detail_id, folder_name), rows in groups:
# # # # #         report, error = _process_folder_group(
# # # # #             train_detail_id = train_detail_id,
# # # # #             folder_name     = folder_name,
# # # # #             rows            = rows,
# # # # #         )
# # # # #         if report is not None:
# # # # #             all_reports.append(report)
# # # # #         if error:
# # # # #             errors.append(error)

# # # # #     return JSONResponse(content={
# # # # #         "status":         "ok" if not errors else "partial",
# # # # #         "groups_total":   len(groups),
# # # # #         "groups_success": len(all_reports),
# # # # #         "groups_failed":  len(errors),
# # # # #         "errors":         errors,
# # # # #         "reports":        all_reports,
# # # # #     })


# # # # # # ── Per-folder-group processor ─────────────────────────────────────────────────

# # # # # def _process_folder_group(
# # # # #     train_detail_id: int,
# # # # #     folder_name:     str,
# # # # #     rows:            List[Dict[str, Any]],
# # # # # ) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
# # # # #     """
# # # # #     Process one folder group — all videos that share the same
# # # # #     (train_detail_id, folder_name), ordered by seq_no.

# # # # #     Returns (report_dict, None) on success, (None, error_str) on failure.
# # # # #     """
# # # # #     print(
# # # # #         f"\n[Batch] ── Folder: train={train_detail_id}  "
# # # # #         f"folder='{folder_name}'  ({len(rows)} video(s)) ──"
# # # # #     )

# # # # #     # folder_name is unique per recording session → use as analysis_id
# # # # #     analysis_id = folder_name

# # # # #     # Step 1 — mark all rows in-progress
# # # # #     for row in rows:
# # # # #         try:
# # # # #             set_process_flag(row["id"], "I")
# # # # #         except Exception as exc:
# # # # #             print(f"[Batch] Could not mark row {row['id']} as 'I': {exc}")

# # # # #     # Step 2 — download every video from S3 to a temp file
# # # # #     tmp_paths: List[str] = []
# # # # #     try:
# # # # #         for row in rows:
# # # # #             suffix = os.path.splitext(row["filename"] or "video.mp4")[1] or ".mp4"
# # # # #             tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
# # # # #             os.close(tmp_fd)
# # # # #             download_video_from_s3(row["s3_video_path"], tmp_path)
# # # # #             tmp_paths.append(tmp_path)
# # # # #     except Exception as exc:
# # # # #         err = f"folder='{folder_name}' download failed: {exc}"
# # # # #         print(f"[Batch] {err}")
# # # # #         _cleanup_temps(tmp_paths)
# # # # #         return None, err

# # # # #     # Step 3 — run the pipeline over each video in seq_no order
# # # # #     report_path = ""
# # # # #     try:
# # # # #         for idx, (row, tmp_path) in enumerate(zip(rows, tmp_paths)):
# # # # #             print(
# # # # #                 f"[Batch]   [{idx + 1}/{len(rows)}]  "
# # # # #                 f"{row['filename']}  (seq={row['seq_no']})"
# # # # #             )
# # # # #             pipeline = GadgetDetectionPipeline(
# # # # #                 source          = tmp_path,
# # # # #                 analysis_id     = analysis_id,
# # # # #                 train_detail_id = train_detail_id,
# # # # #                 save            = False,
# # # # #                 display         = False,
# # # # #             )
# # # # #             report_path = pipeline.run()

# # # # #         # Step 4 — mark all rows done
# # # # #         for row in rows:
# # # # #             try:
# # # # #                 set_process_flag(row["id"], "Y")
# # # # #             except Exception as exc:
# # # # #                 print(f"[Batch] Could not mark row {row['id']} as 'Y': {exc}")

# # # # #         # Step 5 — read and return the report
# # # # #         if report_path and os.path.isfile(report_path):
# # # # #             with open(report_path, encoding="utf-8") as f:
# # # # #                 return json.load(f), None
# # # # #         else:
# # # # #             err = f"folder='{folder_name}' pipeline returned no report file"
# # # # #             print(f"[Batch] {err}")
# # # # #             return None, err

# # # # #     except Exception as exc:
# # # # #         # flags stay at 'I' — intentional, so operator can inspect
# # # # #         err = (
# # # # #             f"folder='{folder_name}' pipeline error: {exc}\n"
# # # # #             + traceback.format_exc()
# # # # #         )
# # # # #         print(f"[Batch] {err}")
# # # # #         return None, err

# # # # #     finally:
# # # # #         _cleanup_temps(tmp_paths)


# # # # # # ── Temp file cleanup ──────────────────────────────────────────────────────────

# # # # # def _cleanup_temps(paths: List[str]) -> None:
# # # # #     for p in paths:
# # # # #         try:
# # # # #             if os.path.isfile(p):
# # # # #                 os.remove(p)
# # # # #         except OSError:
# # # # #             pass


# # # # from __future__ import annotations

# # # # """
# # # # api.py — Loco Pilot Distraction Detection  (DB-driven batch runner)
# # # # ====================================================================

# # # # Videos are NOT uploaded through this API.
# # # # This service polls the database for video_files rows where
# # # # process_flag = 'N', downloads each video from S3, runs the detection
# # # # pipeline synchronously, and returns the final result directly.

# # # # Flag lifecycle
# # # # --------------
# # # #   N  →  pending   (set by the frontend when a video is uploaded to S3)
# # # #   I  →  in-progress (set here before the pipeline starts)
# # # #   Y  →  done      (set here only on successful pipeline completion)

# # # # If the pipeline crashes the flag stays at 'I' so operators can see
# # # # exactly which video failed.

# # # # Endpoints
# # # # ---------
# # # #   GET  /         — health / welcome
# # # #   GET  /health   — {"status": "ok"}
# # # #   POST /trigger  — scan DB → download from S3 → run pipeline → return result
# # # # """

# # # # import json
# # # # import os
# # # # import tempfile
# # # # import traceback
# # # # import uuid
# # # # from itertools import groupby
# # # # from typing import Any, Dict, List, Optional, Tuple

# # # # from fastapi import FastAPI, HTTPException
# # # # from fastapi.middleware.cors import CORSMiddleware
# # # # from fastapi.responses import JSONResponse

# # # # from main import GadgetDetectionPipeline
# # # # from utils.db_s3_uploader import (
# # # #     download_video_from_s3,
# # # #     get_pending_videos,
# # # #     set_process_flag,
# # # # )

# # # # # ── App setup ──────────────────────────────────────────────────────────────────

# # # # app = FastAPI(
# # # #     title   = "Loco Pilot Distraction Detection API",
# # # #     version = "3.0.0",
# # # # )

# # # # app.add_middleware(
# # # #     CORSMiddleware,
# # # #     allow_origins     = ["*"],
# # # #     allow_credentials = True,
# # # #     allow_methods     = ["*"],
# # # #     allow_headers     = ["*"],
# # # # )


# # # # # ── Routes ─────────────────────────────────────────────────────────────────────

# # # # @app.get("/")
# # # # def root() -> dict:
# # # #     return {
# # # #         "status":  "success",
# # # #         "message": "Loco Pilot Distraction Detection API is running",
# # # #         "health":  "/health",
# # # #         "docs":    "/docs",
# # # #         "trigger": "POST /trigger  — scan DB, process all pending videos, return results",
# # # #     }


# # # # @app.get("/health", tags=["status"])
# # # # def health() -> dict:
# # # #     return {"status": "ok"}


# # # # # ── Main trigger ───────────────────────────────────────────────────────────────

# # # # @app.post("/trigger", tags=["batch"])
# # # # def trigger_batch() -> JSONResponse:
# # # #     """
# # # #     Scan DB for all rows with process_flag = 'N'.
# # # #     Group by (train_detail_id, folder_name) — each group is one logical
# # # #     analysis (a folder of sequential videos from one recording session).
# # # #     Process every group in sequence, then return all results in the
# # # #     target batch envelope format.

# # # #     Response shape:
# # # #     {
# # # #       "batch_id":          "<hex>",
# # # #       "total_videos":      N,
# # # #       "completed":         N,
# # # #       "failed":            N,
# # # #       "folders_processed": N,
# # # #       "folders": [
# # # #         {
# # # #           "train_detail_id":   22803,
# # # #           "folder_name":       "22803-05-06-2026",
# # # #           "videos_in_folder":  8,
# # # #           "report": { ... }   ← full analysis_report.json content
# # # #         },
# # # #         ...
# # # #       ]
# # # #     }
# # # #     """
# # # #     try:
# # # #         pending = get_pending_videos()
# # # #     except Exception as exc:
# # # #         raise HTTPException(
# # # #             status_code = 500,
# # # #             detail      = f"Failed to query pending videos from DB: {exc}",
# # # #         )

# # # #     if not pending:
# # # #         return JSONResponse(content={
# # # #             "batch_id":          uuid.uuid4().hex[:12],
# # # #             "total_videos":      0,
# # # #             "completed":         0,
# # # #             "failed":            0,
# # # #             "folders_processed": 0,
# # # #             "folders":           [],
# # # #             "message":           "No pending videos found (process_flag = 'N').",
# # # #         })

# # # #     batch_id = uuid.uuid4().hex[:12]

# # # #     def _group_key(row: Dict[str, Any]):
# # # #         return (row["train_detail_id"], row["folder_name"])

# # # #     groups = [
# # # #         (key, list(grp_rows))
# # # #         for key, grp_rows in groupby(pending, key=_group_key)
# # # #     ]
# # # #     print(f"[Batch:{batch_id}] {len(pending)} video(s) across {len(groups)} folder group(s).")

# # # #     total_videos = len(pending)
# # # #     completed    = 0
# # # #     failed       = 0
# # # #     folders_out: List[Dict[str, Any]] = []

# # # #     for (train_detail_id, folder_name), rows in groups:
# # # #         report, error, n_completed, n_failed = _process_folder_group(
# # # #             train_detail_id = train_detail_id,
# # # #             folder_name     = folder_name,
# # # #             rows            = rows,
# # # #         )
# # # #         completed += n_completed
# # # #         failed    += n_failed

# # # #         folder_entry: Dict[str, Any] = {
# # # #             "train_detail_id":  train_detail_id,
# # # #             "folder_name":      folder_name,
# # # #             "videos_in_folder": len(rows),
# # # #         }
# # # #         if report is not None:
# # # #             folder_entry["report"] = report
# # # #         if error:
# # # #             folder_entry["error"] = error

# # # #         folders_out.append(folder_entry)

# # # #     return JSONResponse(content={
# # # #         "batch_id":          batch_id,
# # # #         "total_videos":      total_videos,
# # # #         "completed":         completed,
# # # #         "failed":            failed,
# # # #         "folders_processed": len(folders_out),
# # # #         "folders":           folders_out,
# # # #     })


# # # # # ── Per-folder-group processor ─────────────────────────────────────────────────

# # # # def _process_folder_group(
# # # #     train_detail_id: int,
# # # #     folder_name:     str,
# # # #     rows:            List[Dict[str, Any]],
# # # # ) -> Tuple[Optional[Dict[str, Any]], Optional[str], int, int]:
# # # #     """
# # # #     Process one folder group — all videos that share the same
# # # #     (train_detail_id, folder_name), ordered by seq_no.

# # # #     Returns (report_dict, error_str, n_completed, n_failed).
# # # #     """
# # # #     n_videos = len(rows)
# # # #     print(
# # # #         f"\n[Batch] ── Folder: train={train_detail_id}  "
# # # #         f"folder='{folder_name}'  ({n_videos} video(s)) ──"
# # # #     )

# # # #     # folder_name is unique per recording session → use as analysis_id
# # # #     analysis_id = folder_name

# # # #     # Step 1 — mark all rows in-progress
# # # #     for row in rows:
# # # #         try:
# # # #             set_process_flag(row["id"], "I")
# # # #         except Exception as exc:
# # # #             print(f"[Batch] Could not mark row {row['id']} as 'I': {exc}")

# # # #     # Step 2 — download every video from S3 to a temp file
# # # #     tmp_paths: List[str] = []
# # # #     try:
# # # #         for row in rows:
# # # #             suffix = os.path.splitext(row["filename"] or "video.mp4")[1] or ".mp4"
# # # #             tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
# # # #             os.close(tmp_fd)
# # # #             download_video_from_s3(row["s3_video_path"], tmp_path)
# # # #             tmp_paths.append(tmp_path)
# # # #     except Exception as exc:
# # # #         err = f"folder='{folder_name}' download failed: {exc}"
# # # #         print(f"[Batch] {err}")
# # # #         _cleanup_temps(tmp_paths)
# # # #         return None, err, 0, n_videos

# # # #     # Step 3 — run the pipeline over each video in seq_no order
# # # #     report_path = ""
# # # #     try:
# # # #         for idx, (row, tmp_path) in enumerate(zip(rows, tmp_paths)):
# # # #             print(
# # # #                 f"[Batch]   [{idx + 1}/{n_videos}]  "
# # # #                 f"{row['filename']}  (seq={row['seq_no']})"
# # # #             )
# # # #             pipeline = GadgetDetectionPipeline(
# # # #                 source          = tmp_path,
# # # #                 analysis_id     = analysis_id,
# # # #                 train_detail_id = train_detail_id,
# # # #                 save            = False,
# # # #                 display         = False,
# # # #             )
# # # #             report_path = pipeline.run()

# # # #         # Step 4 — mark all rows done
# # # #         for row in rows:
# # # #             try:
# # # #                 set_process_flag(row["id"], "Y")
# # # #             except Exception as exc:
# # # #                 print(f"[Batch] Could not mark row {row['id']} as 'Y': {exc}")

# # # #         # Step 5 — read and return the report
# # # #         if report_path and os.path.isfile(report_path):
# # # #             with open(report_path, encoding="utf-8") as f:
# # # #                 return json.load(f), None, n_videos, 0
# # # #         else:
# # # #             err = f"folder='{folder_name}' pipeline returned no report file"
# # # #             print(f"[Batch] {err}")
# # # #             return None, err, 0, n_videos

# # # #     except Exception as exc:
# # # #         # flags stay at 'I' — intentional, so operator can inspect
# # # #         err = (
# # # #             f"folder='{folder_name}' pipeline error: {exc}\n"
# # # #             + traceback.format_exc()
# # # #         )
# # # #         print(f"[Batch] {err}")
# # # #         return None, err, 0, n_videos

# # # #     finally:
# # # #         _cleanup_temps(tmp_paths)


# # # # # ── Temp file cleanup ──────────────────────────────────────────────────────────

# # # # def _cleanup_temps(paths: List[str]) -> None:
# # # #     for p in paths:
# # # #         try:
# # # #             if os.path.isfile(p):
# # # #                 os.remove(p)
# # # #         except OSError:
# # # #             pass

# # # from __future__ import annotations

# # # """
# # # api.py — Loco Pilot Distraction Detection  (DB-driven batch runner)
# # # ====================================================================

# # # Videos are NOT uploaded through this API.
# # # This service polls the database for video_files rows where
# # # process_flag = 'N', downloads each video from S3, runs the detection
# # # pipeline synchronously, and returns the final result directly.

# # # Flag lifecycle
# # # --------------
# # #   N  →  pending   (set by the frontend when a video is uploaded to S3)
# # #   I  →  in-progress (set here before the pipeline starts)
# # #   Y  →  done      (set here only on successful pipeline completion)

# # # If the pipeline crashes the flag stays at 'I' so operators can see
# # # exactly which video failed.

# # # Endpoints
# # # ---------
# # #   GET  /         — health / welcome
# # #   GET  /health   — {"status": "ok"}
# # #   POST /trigger  — scan DB → download from S3 → run pipeline → return result

# # # Response shape
# # # --------------
# # # {
# # #   "batch_id":          "444e4af5ff5f",
# # #   "total_videos":      2,
# # #   "completed":         2,
# # #   "failed":            0,
# # #   "folders_processed": 1,
# # #   "folders": [
# # #     {
# # #       "train_detail_id":  22801,
# # #       "folder_name":      "22801-05-06-2026",
# # #       "videos_in_folder": 2,
# # #       "report": {
# # #         "analysis_id":     "22801-05-06-2026",
# # #         "train_detail_id": 22801,
# # #         "processing_time": 108.124,
# # #         "video_info":      [ ... ],   ← list with one entry per video
# # #         "violations":      [ ... ]    ← all violations across ALL videos
# # #       }
# # #     }
# # #   ]
# # # }

# # # Timestamp logic
# # # ---------------
# # # For each folder group, videos are processed in seq_no order.
# # # A running time_offset and frame_offset accumulate as each video finishes.

# # #   global_timestamp  = local_video_time  + time_offset
# # #   global_frame      = local_frame_index + frame_offset

# # # In the report:
# # #   "timestamp"                  = global_timestamp  (HH:MM:SS into the full recording)
# # #   "original_video_timestamp"   = "<db_filename> <local_time>"  (time within that file)
# # # """

# # # import json
# # # import os
# # # import tempfile
# # # import traceback
# # # import uuid
# # # from itertools import groupby
# # # from typing import Any, Dict, List, Optional, Tuple

# # # import cv2

# # # from fastapi import FastAPI, HTTPException
# # # from fastapi.middleware.cors import CORSMiddleware
# # # from fastapi.responses import JSONResponse

# # # from main import GadgetDetectionPipeline
# # # from utils.violation_store import ViolationStore
# # # from utils.db_s3_uploader import (
# # #     download_video_from_s3,
# # #     get_pending_videos,
# # #     set_process_flag,
# # # )

# # # # ── App setup ──────────────────────────────────────────────────────────────────

# # # app = FastAPI(
# # #     title   = "Loco Pilot Distraction Detection API",
# # #     version = "3.0.0",
# # # )

# # # app.add_middleware(
# # #     CORSMiddleware,
# # #     allow_origins     = ["*"],
# # #     allow_credentials = True,
# # #     allow_methods     = ["*"],
# # #     allow_headers     = ["*"],
# # # )


# # # # ── Routes ─────────────────────────────────────────────────────────────────────

# # # @app.get("/")
# # # def root() -> dict:
# # #     return {
# # #         "status":  "success",
# # #         "message": "Loco Pilot Distraction Detection API is running",
# # #         "health":  "/health",
# # #         "docs":    "/docs",
# # #         "trigger": "POST /trigger  — scan DB, process all pending videos, return results",
# # #     }


# # # @app.get("/health", tags=["status"])
# # # def health() -> dict:
# # #     return {"status": "ok"}


# # # # ── Main trigger ───────────────────────────────────────────────────────────────

# # # @app.post("/trigger", tags=["batch"])
# # # def trigger_batch() -> JSONResponse:
# # #     """
# # #     Scan DB for all rows with process_flag = 'N'.
# # #     Group by (train_detail_id, folder_name).
# # #     Process every group in sequence, return the batch envelope.
# # #     """
# # #     try:
# # #         pending = get_pending_videos()
# # #     except Exception as exc:
# # #         raise HTTPException(
# # #             status_code = 500,
# # #             detail      = f"Failed to query pending videos from DB: {exc}",
# # #         )

# # #     batch_id = uuid.uuid4().hex[:12]

# # #     if not pending:
# # #         return JSONResponse(content={
# # #             "batch_id":          batch_id,
# # #             "total_videos":      0,
# # #             "completed":         0,
# # #             "failed":            0,
# # #             "folders_processed": 0,
# # #             "folders":           [],
# # #             "message":           "No pending videos found (process_flag = 'N').",
# # #         })

# # #     def _group_key(row: Dict[str, Any]):
# # #         return (row["train_detail_id"], row["folder_name"])

# # #     groups = [
# # #         (key, list(grp_rows))
# # #         for key, grp_rows in groupby(pending, key=_group_key)
# # #     ]
# # #     print(f"[Batch:{batch_id}] {len(pending)} video(s) across {len(groups)} folder group(s).")

# # #     total_videos = len(pending)
# # #     completed    = 0
# # #     failed       = 0
# # #     folders_out: List[Dict[str, Any]] = []

# # #     for (train_detail_id, folder_name), rows in groups:
# # #         report, error, n_ok, n_fail = _process_folder_group(
# # #             train_detail_id = train_detail_id,
# # #             folder_name     = folder_name,
# # #             rows            = rows,
# # #         )
# # #         completed += n_ok
# # #         failed    += n_fail

# # #         folder_entry: Dict[str, Any] = {
# # #             "train_detail_id":  train_detail_id,
# # #             "folder_name":      folder_name,
# # #             "videos_in_folder": len(rows),
# # #         }
# # #         if report is not None:
# # #             folder_entry["report"] = report
# # #         if error:
# # #             folder_entry["error"] = error

# # #         folders_out.append(folder_entry)

# # #     return JSONResponse(content={
# # #         "batch_id":          batch_id,
# # #         "total_videos":      total_videos,
# # #         "completed":         completed,
# # #         "failed":            failed,
# # #         "folders_processed": len(folders_out),
# # #         "folders":           folders_out,
# # #     })


# # # # ── Per-folder-group processor ─────────────────────────────────────────────────

# # # def _video_duration_seconds(path: str) -> float:
# # #     """Read the duration of a video file using OpenCV."""
# # #     cap = cv2.VideoCapture(path)
# # #     if not cap.isOpened():
# # #         return 0.0
# # #     fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
# # #     total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# # #     cap.release()
# # #     return total / fps if fps > 0 and total > 0 else 0.0


# # # def _process_folder_group(
# # #     train_detail_id: int,
# # #     folder_name:     str,
# # #     rows:            List[Dict[str, Any]],
# # # ) -> Tuple[Optional[Dict[str, Any]], Optional[str], int, int]:
# # #     """
# # #     Process one folder group — all videos that share the same
# # #     (train_detail_id, folder_name), ordered by seq_no.

# # #     Key design
# # #     ──────────
# # #     • One ViolationStore is created for the whole folder and shared
# # #       across every pipeline run.  This means violations from all videos
# # #       accumulate in a single store.
# # #     • time_offset and frame_offset grow after each video so timestamps
# # #       are continuous across the whole recording session.
# # #     • finalize() is called ONCE after all videos are done.

# # #     Returns (report_dict, error_str, n_completed, n_failed).
# # #     """
# # #     n_videos = len(rows)
# # #     print(
# # #         f"\n[Batch] ── Folder: train={train_detail_id}  "
# # #         f"folder='{folder_name}'  ({n_videos} video(s)) ──"
# # #     )

# # #     analysis_id = folder_name

# # #     # Step 1 — mark all rows in-progress
# # #     for row in rows:
# # #         try:
# # #             set_process_flag(row["id"], "I")
# # #         except Exception as exc:
# # #             print(f"[Batch] Could not mark row {row['id']} as 'I': {exc}")

# # #     # Step 2 — download every video from S3 to a temp file
# # #     # We also record the DB filename alongside each temp path.
# # #     tmp_entries: List[Tuple[str, str]] = []   # [(tmp_path, db_filename), ...]
# # #     try:
# # #         for row in rows:
# # #             suffix = os.path.splitext(row["filename"] or "video.mp4")[1] or ".mp4"
# # #             tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
# # #             os.close(tmp_fd)
# # #             download_video_from_s3(row["s3_video_path"], tmp_path)
# # #             tmp_entries.append((tmp_path, row["filename"] or ""))
# # #     except Exception as exc:
# # #         err = f"folder='{folder_name}' download failed: {exc}"
# # #         print(f"[Batch] {err}")
# # #         _cleanup_temps([p for p, _ in tmp_entries])
# # #         return None, err, 0, n_videos

# # #     # Step 3 — create ONE shared ViolationStore for the entire folder
# # #     shared_vstore = ViolationStore(
# # #         analysis_id     = analysis_id,
# # #         train_detail_id = train_detail_id,
# # #         # No video_info here — each pipeline run will call add_video_info()
# # #     )

# # #     # Step 4 — run the pipeline over each video in seq_no order,
# # #     #           accumulating time_offset and frame_offset between videos.
# # #     time_offset  = 0.0
# # #     frame_offset = 0
# # #     total_processing_time = 0.0

# # #     try:
# # #         for idx, (row, (tmp_path, db_filename)) in enumerate(zip(rows, tmp_entries)):
# # #             print(
# # #                 f"[Batch]   [{idx + 1}/{n_videos}]  "
# # #                 f"{db_filename}  (seq={row['seq_no']})  "
# # #                 f"time_offset={time_offset:.2f}s  frame_offset={frame_offset}"
# # #             )

# # #             import time as _time
# # #             t0 = _time.time()

# # #             pipeline = GadgetDetectionPipeline(
# # #                 source          = tmp_path,
# # #                 analysis_id     = analysis_id,
# # #                 train_detail_id = train_detail_id,
# # #                 save            = False,
# # #                 display         = False,
# # #                 shared_vstore   = shared_vstore,
# # #                 time_offset     = time_offset,
# # #                 frame_offset    = frame_offset,
# # #                 source_filename = db_filename,
# # #             )
# # #             pipeline.run()   # returns "" in batch mode; vstore is NOT finalized here

# # #             video_duration = _video_duration_seconds(tmp_path)
# # #             video_frames   = _get_frame_count(tmp_path)

# # #             total_processing_time += _time.time() - t0

# # #             # Advance offsets for the next video
# # #             time_offset  += video_duration
# # #             frame_offset += video_frames

# # #         # Step 5 — finalize the shared store ONCE with total processing time
# # #         report_path = shared_vstore.finalize(processing_time=round(total_processing_time, 3))

# # #         # Step 6 — mark all rows done
# # #         for row in rows:
# # #             try:
# # #                 set_process_flag(row["id"], "Y")
# # #             except Exception as exc:
# # #                 print(f"[Batch] Could not mark row {row['id']} as 'Y': {exc}")

# # #         # Step 7 — read and return the report
# # #         if report_path and os.path.isfile(report_path):
# # #             with open(report_path, encoding="utf-8") as f:
# # #                 return json.load(f), None, n_videos, 0
# # #         else:
# # #             err = f"folder='{folder_name}' no report file after finalize"
# # #             print(f"[Batch] {err}")
# # #             return None, err, 0, n_videos

# # #     except Exception as exc:
# # #         # flags stay at 'I' — intentional so operator can inspect
# # #         err = (
# # #             f"folder='{folder_name}' pipeline error: {exc}\n"
# # #             + traceback.format_exc()
# # #         )
# # #         print(f"[Batch] {err}")
# # #         return None, err, 0, n_videos

# # #     finally:
# # #         _cleanup_temps([p for p, _ in tmp_entries])


# # # def _get_frame_count(path: str) -> int:
# # #     cap = cv2.VideoCapture(path)
# # #     if not cap.isOpened():
# # #         return 0
# # #     count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# # #     cap.release()
# # #     return count


# # # # ── Temp file cleanup ──────────────────────────────────────────────────────────

# # # def _cleanup_temps(paths: List[str]) -> None:
# # #     for p in paths:
# # #         try:
# # #             if os.path.isfile(p):
# # #                 os.remove(p)
# # #         except OSError:
# # #             pass

# # # # # from __future__ import annotations

# # # # # """
# # # # # api.py — Loco Pilot Distraction Detection  (DB-driven batch runner)
# # # # # ====================================================================

# # # # # Videos are NO LONGER uploaded through this API.
# # # # # Instead, this service polls the database for video_files rows where
# # # # # process_flag = 'N', downloads each video from S3, runs the detection
# # # # # pipeline, and marks the row done (process_flag = 'Y').

# # # # # Flag lifecycle
# # # # # --------------
# # # # #   N  →  pending (set by the frontend / ingestion service when uploading)
# # # # #   I  →  in-progress (set here, immediately before the pipeline starts)
# # # # #   Y  →  done (set here, only on successful pipeline completion)

# # # # # If the pipeline crashes the flag stays at 'I' so operators can see
# # # # # exactly which video failed — it is NOT silently reset to 'N'.

# # # # # Endpoints
# # # # # ---------
# # # # #   GET  /              — health / welcome
# # # # #   GET  /health        — {"status": "ok"}
# # # # #   POST /trigger       — kick off a DB-scan + batch run immediately
# # # # #                         (returns job_id; poll /status/<job_id>)
# # # # #   GET  /status/<id>   — queued | processing | done | failed
# # # # #   GET  /result/<id>   — final JSON report (consumed once; deleted from memory)
# # # # # """

# # # # # import json
# # # # # import os
# # # # # import tempfile
# # # # # import traceback
# # # # # import uuid
# # # # # from concurrent.futures import ThreadPoolExecutor
# # # # # from itertools import groupby
# # # # # from typing import Any, Dict, List, Optional

# # # # # from fastapi import BackgroundTasks, FastAPI, HTTPException
# # # # # from fastapi.middleware.cors import CORSMiddleware
# # # # # from fastapi.responses import JSONResponse

# # # # # from main import GadgetDetectionPipeline
# # # # # from utils.db_s3_uploader import (
# # # # #     download_video_from_s3,
# # # # #     get_pending_videos,
# # # # #     set_process_flag,
# # # # # )

# # # # # # ── App setup ──────────────────────────────────────────────────────────────────

# # # # # app = FastAPI(
# # # # #     title   = "Loco Pilot Distraction Detection API",
# # # # #     version = "3.0.0",
# # # # # )

# # # # # app.add_middleware(
# # # # #     CORSMiddleware,
# # # # #     allow_origins     = ["*"],
# # # # #     allow_credentials = True,
# # # # #     allow_methods     = ["*"],
# # # # #     allow_headers     = ["*"],
# # # # # )

# # # # # # In-memory job registry  {job_id: {"status": str, "result": dict|None, "error": str|None}}
# # # # # _jobs:    Dict[str, Dict[str, Any]] = {}
# # # # # _executor = ThreadPoolExecutor(max_workers=1)  # one batch at a time


# # # # # # ── Routes ─────────────────────────────────────────────────────────────────────

# # # # # @app.get("/")
# # # # # def root() -> dict:
# # # # #     return {
# # # # #         "status":  "success",
# # # # #         "message": "Loco Pilot Distraction Detection API is running",
# # # # #         "health":  "/health",
# # # # #         "docs":    "/docs",
# # # # #         "trigger": "POST /trigger  — start a DB-scan + batch run",
# # # # #     }


# # # # # @app.get("/health", tags=["status"])
# # # # # def health() -> dict:
# # # # #     return {"status": "ok"}


# # # # # # ── Batch trigger ──────────────────────────────────────────────────────────────

# # # # # @app.post("/trigger", tags=["batch"])
# # # # # async def trigger_batch(background_tasks: BackgroundTasks) -> JSONResponse:
# # # # #     """
# # # # #     Scan the DB for all pending videos (process_flag='N') and process them
# # # # #     as a batch.  Returns immediately with a job_id; poll /status/<job_id>.
# # # # #     """
# # # # #     job_id = str(uuid.uuid4())
# # # # #     _jobs[job_id] = {"status": "queued", "result": None, "error": None}
# # # # #     _executor.submit(_run_batch, job_id)
# # # # #     return JSONResponse(
# # # # #         status_code = 202,
# # # # #         content = {
# # # # #             "job_id":  job_id,
# # # # #             "status":  "queued",
# # # # #             "message": f"Batch job accepted. Poll GET /status/{job_id} for progress.",
# # # # #         },
# # # # #     )


# # # # # @app.get("/status/{job_id}", tags=["batch"])
# # # # # def job_status(job_id: str) -> JSONResponse:
# # # # #     job = _jobs.get(job_id)
# # # # #     if job is None:
# # # # #         raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
# # # # #     resp: Dict[str, Any] = {"job_id": job_id, "status": job["status"]}
# # # # #     if job["status"] == "failed":
# # # # #         resp["error"] = job["error"]
# # # # #     return JSONResponse(content=resp)


# # # # # @app.get("/result/{job_id}", tags=["batch"])
# # # # # def job_result(job_id: str) -> JSONResponse:
# # # # #     job = _jobs.get(job_id)
# # # # #     if job is None:
# # # # #         raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
# # # # #     if job["status"] == "failed":
# # # # #         raise HTTPException(status_code=500, detail=job["error"])
# # # # #     if job["status"] in ("queued", "processing"):
# # # # #         raise HTTPException(
# # # # #             status_code=409,
# # # # #             detail=f"Job is still '{job['status']}'. Try again later.",
# # # # #         )
# # # # #     result = _jobs.pop(job_id)["result"]
# # # # #     return JSONResponse(content=result)


# # # # # # ── Batch worker ───────────────────────────────────────────────────────────────

# # # # # def _run_batch(job_id: str) -> None:
# # # # #     """
# # # # #     Entry point executed in the thread pool.

# # # # #     1. Query DB for all rows with process_flag = 'N'.
# # # # #     2. Group them by (train_detail_id, folder_name) — each group is one
# # # # #        logical analysis (the videos in a folder are one continuous recording).
# # # # #     3. For every group:
# # # # #          a. Mark every row as 'I' (in-progress).
# # # # #          b. Download each video from S3 to a temp file.
# # # # #          c. Run GadgetDetectionPipeline over every video in sequence.
# # # # #          d. On success: mark every row as 'Y' and collect the report.
# # # # #          e. On failure: leave rows at 'I' (flag stays for operator inspection).
# # # # #     4. Collect per-group reports and write them to the job registry.
# # # # #     """
# # # # #     _jobs[job_id]["status"] = "processing"
# # # # #     all_reports: List[Dict[str, Any]] = []

# # # # #     try:
# # # # #         pending = get_pending_videos()

# # # # #         if not pending:
# # # # #             print("[Batch] No pending videos found.")
# # # # #             _jobs[job_id]["status"] = "done"
# # # # #             _jobs[job_id]["result"] = {"message": "No pending videos.", "reports": []}
# # # # #             return

# # # # #         # Group by (train_detail_id, folder_name) — same order as the DB query
# # # # #         def _group_key(row: Dict[str, Any]):
# # # # #             return (row["train_detail_id"], row["folder_name"])

# # # # #         groups = [
# # # # #             (key, list(rows))
# # # # #             for key, rows in groupby(pending, key=_group_key)
# # # # #         ]
# # # # #         print(f"[Batch] {len(pending)} video(s) across {len(groups)} folder group(s).")

# # # # #         for (train_detail_id, folder_name), rows in groups:
# # # # #             report = _process_folder_group(
# # # # #                 train_detail_id = train_detail_id,
# # # # #                 folder_name     = folder_name,
# # # # #                 rows            = rows,
# # # # #             )
# # # # #             if report:
# # # # #                 all_reports.append(report)

# # # # #         _jobs[job_id]["status"] = "done"
# # # # #         _jobs[job_id]["result"] = {
# # # # #             "message": f"Processed {len(all_reports)} folder group(s).",
# # # # #             "reports": all_reports,
# # # # #         }

# # # # #     except Exception as exc:
# # # # #         _jobs[job_id]["status"] = "failed"
# # # # #         _jobs[job_id]["error"]  = f"{exc}\n{traceback.format_exc()}"
# # # # #         print(f"[Batch] Fatal error: {exc}")


# # # # # def _process_folder_group(
# # # # #     train_detail_id: int,
# # # # #     folder_name:     str,
# # # # #     rows:            List[Dict[str, Any]],
# # # # # ) -> Optional[Dict[str, Any]]:
# # # # #     """
# # # # #     Process one folder group (all videos that belong to a single analysis).

# # # # #     Rows are already ordered by seq_no (guaranteed by the DB query).
# # # # #     Each video is processed in sequence so frame offsets accumulate
# # # # #     correctly across files.

# # # # #     Returns the JSON report dict on success, None on failure.
# # # # #     """
# # # # #     print(
# # # # #         f"\n[Batch] ── Folder group: train={train_detail_id}  "
# # # # #         f"folder='{folder_name}'  ({len(rows)} video(s)) ──"
# # # # #     )

# # # # #     # Use folder_name as the analysis_id (unique per recording session)
# # # # #     analysis_id = folder_name

# # # # #     # ── Step 1: mark all rows as in-progress ─────────────────────────────────
# # # # #     for row in rows:
# # # # #         try:
# # # # #             set_process_flag(row["id"], "I")
# # # # #         except Exception as exc:
# # # # #             print(f"[Batch] Could not mark row {row['id']} as 'I': {exc}")
# # # # #             # Non-fatal — continue; the row stays 'N' which is safer than crashing

# # # # #     # ── Step 2: download videos to temp files ─────────────────────────────────
# # # # #     tmp_paths: List[str] = []
# # # # #     try:
# # # # #         for row in rows:
# # # # #             suffix    = os.path.splitext(row["filename"] or "video.mp4")[1] or ".mp4"
# # # # #             tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
# # # # #             os.close(tmp_fd)
# # # # #             download_video_from_s3(row["s3_video_path"], tmp_path)
# # # # #             tmp_paths.append(tmp_path)
# # # # #     except Exception as exc:
# # # # #         print(f"[Batch] Download failed for folder '{folder_name}': {exc}")
# # # # #         _cleanup_temps(tmp_paths)
# # # # #         return None

# # # # #     # ── Step 3: run the pipeline over each video in sequence ──────────────────
# # # # #     report_path: str = ""
# # # # #     try:
# # # # #         # The pipeline processes one video at a time but shares the same
# # # # #         # ViolationStore (via analysis_id = folder_name) so violations from
# # # # #         # every video land in the same report.
# # # # #         for idx, (row, tmp_path) in enumerate(zip(rows, tmp_paths)):
# # # # #             print(
# # # # #                 f"[Batch]   [{idx+1}/{len(rows)}] Processing  "
# # # # #                 f"{row['filename']}  (seq={row['seq_no']})"
# # # # #             )
# # # # #             pipeline = GadgetDetectionPipeline(
# # # # #                 source          = tmp_path,
# # # # #                 analysis_id     = analysis_id,
# # # # #                 train_detail_id = train_detail_id,
# # # # #                 save            = False,
# # # # #                 display         = False,
# # # # #             )
# # # # #             # run() returns the path to analysis_report.json
# # # # #             report_path = pipeline.run()

# # # # #         # ── Step 4: mark all rows as done ────────────────────────────────────
# # # # #         for row in rows:
# # # # #             try:
# # # # #                 set_process_flag(row["id"], "Y")
# # # # #             except Exception as exc:
# # # # #                 print(f"[Batch] Could not mark row {row['id']} as 'Y': {exc}")

# # # # #         # ── Step 5: read and return the report ───────────────────────────────
# # # # #         if report_path and os.path.isfile(report_path):
# # # # #             with open(report_path, encoding="utf-8") as f:
# # # # #                 return json.load(f)
# # # # #         else:
# # # # #             print(f"[Batch] Report file missing for folder '{folder_name}'")
# # # # #             return None

# # # # #     except Exception as exc:
# # # # #         # Leave flags at 'I' so the operator can see which group failed
# # # # #         print(
# # # # #             f"[Batch] Pipeline error for folder '{folder_name}': {exc}\n"
# # # # #             + traceback.format_exc()
# # # # #         )
# # # # #         return None

# # # # #     finally:
# # # # #         _cleanup_temps(tmp_paths)


# # # # # def _cleanup_temps(paths: List[str]) -> None:
# # # # #     for p in paths:
# # # # #         try:
# # # # #             if os.path.isfile(p):
# # # # #                 os.remove(p)
# # # # #         except OSError:
# # # # #             pass



# # # # from __future__ import annotations

# # # # """
# # # # api.py — Loco Pilot Distraction Detection  (DB-driven batch runner)
# # # # ====================================================================

# # # # Videos are NOT uploaded through this API.
# # # # This service polls the database for video_files rows where
# # # # process_flag = 'N', downloads each video from S3, runs the detection
# # # # pipeline synchronously, and returns the final result directly.

# # # # Flag lifecycle
# # # # --------------
# # # #   N  →  pending   (set by the frontend when a video is uploaded to S3)
# # # #   I  →  in-progress (set here before the pipeline starts)
# # # #   Y  →  done      (set here only on successful pipeline completion)

# # # # If the pipeline crashes the flag stays at 'I' so operators can see
# # # # exactly which video failed.

# # # # Endpoints
# # # # ---------
# # # #   GET  /         — health / welcome
# # # #   GET  /health   — {"status": "ok"}
# # # #   POST /trigger  — scan DB → download from S3 → run pipeline → return result
# # # # """

# # # # import json
# # # # import os
# # # # import tempfile
# # # # import traceback
# # # # from itertools import groupby
# # # # from typing import Any, Dict, List, Optional

# # # # from fastapi import FastAPI, HTTPException
# # # # from fastapi.middleware.cors import CORSMiddleware
# # # # from fastapi.responses import JSONResponse

# # # # from main import GadgetDetectionPipeline
# # # # from utils.db_s3_uploader import (
# # # #     download_video_from_s3,
# # # #     get_pending_videos,
# # # #     set_process_flag,
# # # # )

# # # # # ── App setup ──────────────────────────────────────────────────────────────────

# # # # app = FastAPI(
# # # #     title   = "Loco Pilot Distraction Detection API",
# # # #     version = "3.0.0",
# # # # )

# # # # app.add_middleware(
# # # #     CORSMiddleware,
# # # #     allow_origins     = ["*"],
# # # #     allow_credentials = True,
# # # #     allow_methods     = ["*"],
# # # #     allow_headers     = ["*"],
# # # # )


# # # # # ── Routes ─────────────────────────────────────────────────────────────────────

# # # # @app.get("/")
# # # # def root() -> dict:
# # # #     return {
# # # #         "status":  "success",
# # # #         "message": "Loco Pilot Distraction Detection API is running",
# # # #         "health":  "/health",
# # # #         "docs":    "/docs",
# # # #         "trigger": "POST /trigger  — scan DB, process all pending videos, return results",
# # # #     }


# # # # @app.get("/health", tags=["status"])
# # # # def health() -> dict:
# # # #     return {"status": "ok"}


# # # # # ── Main trigger ───────────────────────────────────────────────────────────────

# # # # @app.post("/trigger", tags=["batch"])
# # # # def trigger_batch() -> JSONResponse:
# # # #     """
# # # #     Scan DB for all rows with process_flag = 'N'.
# # # #     Group by (train_detail_id, folder_name) — each group is one logical
# # # #     analysis (a folder of sequential videos from one recording session).
# # # #     Process every group in sequence, then return all reports together.

# # # #     Flag lifecycle per video row:
# # # #       N  →  I  (before pipeline starts)
# # # #       I  →  Y  (after pipeline succeeds)
# # # #       stays I   (if pipeline fails — visible to operators)
# # # #     """
# # # #     try:
# # # #         pending = get_pending_videos()
# # # #     except Exception as exc:
# # # #         raise HTTPException(
# # # #             status_code=500,
# # # #             detail=f"Failed to query pending videos from DB: {exc}",
# # # #         )

# # # #     if not pending:
# # # #         return JSONResponse(content={
# # # #             "status":  "ok",
# # # #             "message": "No pending videos found (process_flag = 'N').",
# # # #             "reports": [],
# # # #         })

# # # #     def _group_key(row: Dict[str, Any]):
# # # #         return (row["train_detail_id"], row["folder_name"])

# # # #     groups = [
# # # #         (key, list(rows))
# # # #         for key, rows in groupby(pending, key=_group_key)
# # # #     ]
# # # #     print(f"[Batch] {len(pending)} video(s) across {len(groups)} folder group(s).")

# # # #     all_reports: List[Dict[str, Any]] = []
# # # #     errors:      List[str]            = []

# # # #     for (train_detail_id, folder_name), rows in groups:
# # # #         report, error = _process_folder_group(
# # # #             train_detail_id = train_detail_id,
# # # #             folder_name     = folder_name,
# # # #             rows            = rows,
# # # #         )
# # # #         if report is not None:
# # # #             all_reports.append(report)
# # # #         if error:
# # # #             errors.append(error)

# # # #     return JSONResponse(content={
# # # #         "status":         "ok" if not errors else "partial",
# # # #         "groups_total":   len(groups),
# # # #         "groups_success": len(all_reports),
# # # #         "groups_failed":  len(errors),
# # # #         "errors":         errors,
# # # #         "reports":        all_reports,
# # # #     })


# # # # # ── Per-folder-group processor ─────────────────────────────────────────────────

# # # # def _process_folder_group(
# # # #     train_detail_id: int,
# # # #     folder_name:     str,
# # # #     rows:            List[Dict[str, Any]],
# # # # ) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
# # # #     """
# # # #     Process one folder group — all videos that share the same
# # # #     (train_detail_id, folder_name), ordered by seq_no.

# # # #     Returns (report_dict, None) on success, (None, error_str) on failure.
# # # #     """
# # # #     print(
# # # #         f"\n[Batch] ── Folder: train={train_detail_id}  "
# # # #         f"folder='{folder_name}'  ({len(rows)} video(s)) ──"
# # # #     )

# # # #     # folder_name is unique per recording session → use as analysis_id
# # # #     analysis_id = folder_name

# # # #     # Step 1 — mark all rows in-progress
# # # #     for row in rows:
# # # #         try:
# # # #             set_process_flag(row["id"], "I")
# # # #         except Exception as exc:
# # # #             print(f"[Batch] Could not mark row {row['id']} as 'I': {exc}")

# # # #     # Step 2 — download every video from S3 to a temp file
# # # #     tmp_paths: List[str] = []
# # # #     try:
# # # #         for row in rows:
# # # #             suffix = os.path.splitext(row["filename"] or "video.mp4")[1] or ".mp4"
# # # #             tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
# # # #             os.close(tmp_fd)
# # # #             download_video_from_s3(row["s3_video_path"], tmp_path)
# # # #             tmp_paths.append(tmp_path)
# # # #     except Exception as exc:
# # # #         err = f"folder='{folder_name}' download failed: {exc}"
# # # #         print(f"[Batch] {err}")
# # # #         _cleanup_temps(tmp_paths)
# # # #         return None, err

# # # #     # Step 3 — run the pipeline over each video in seq_no order
# # # #     report_path = ""
# # # #     try:
# # # #         for idx, (row, tmp_path) in enumerate(zip(rows, tmp_paths)):
# # # #             print(
# # # #                 f"[Batch]   [{idx + 1}/{len(rows)}]  "
# # # #                 f"{row['filename']}  (seq={row['seq_no']})"
# # # #             )
# # # #             pipeline = GadgetDetectionPipeline(
# # # #                 source          = tmp_path,
# # # #                 analysis_id     = analysis_id,
# # # #                 train_detail_id = train_detail_id,
# # # #                 save            = False,
# # # #                 display         = False,
# # # #             )
# # # #             report_path = pipeline.run()

# # # #         # Step 4 — mark all rows done
# # # #         for row in rows:
# # # #             try:
# # # #                 set_process_flag(row["id"], "Y")
# # # #             except Exception as exc:
# # # #                 print(f"[Batch] Could not mark row {row['id']} as 'Y': {exc}")

# # # #         # Step 5 — read and return the report
# # # #         if report_path and os.path.isfile(report_path):
# # # #             with open(report_path, encoding="utf-8") as f:
# # # #                 return json.load(f), None
# # # #         else:
# # # #             err = f"folder='{folder_name}' pipeline returned no report file"
# # # #             print(f"[Batch] {err}")
# # # #             return None, err

# # # #     except Exception as exc:
# # # #         # flags stay at 'I' — intentional, so operator can inspect
# # # #         err = (
# # # #             f"folder='{folder_name}' pipeline error: {exc}\n"
# # # #             + traceback.format_exc()
# # # #         )
# # # #         print(f"[Batch] {err}")
# # # #         return None, err

# # # #     finally:
# # # #         _cleanup_temps(tmp_paths)


# # # # # ── Temp file cleanup ──────────────────────────────────────────────────────────

# # # # def _cleanup_temps(paths: List[str]) -> None:
# # # #     for p in paths:
# # # #         try:
# # # #             if os.path.isfile(p):
# # # #                 os.remove(p)
# # # #         except OSError:
# # # #             pass


# # # from __future__ import annotations

# # # """
# # # api.py — Loco Pilot Distraction Detection  (DB-driven batch runner)
# # # ====================================================================

# # # Videos are NOT uploaded through this API.
# # # This service polls the database for video_files rows where
# # # process_flag = 'N', downloads each video from S3, runs the detection
# # # pipeline synchronously, and returns the final result directly.

# # # Flag lifecycle
# # # --------------
# # #   N  →  pending   (set by the frontend when a video is uploaded to S3)
# # #   I  →  in-progress (set here before the pipeline starts)
# # #   Y  →  done      (set here only on successful pipeline completion)

# # # If the pipeline crashes the flag stays at 'I' so operators can see
# # # exactly which video failed.

# # # Endpoints
# # # ---------
# # #   GET  /         — health / welcome
# # #   GET  /health   — {"status": "ok"}
# # #   POST /trigger  — scan DB → download from S3 → run pipeline → return result
# # # """

# # # import json
# # # import os
# # # import tempfile
# # # import traceback
# # # import uuid
# # # from itertools import groupby
# # # from typing import Any, Dict, List, Optional, Tuple

# # # from fastapi import FastAPI, HTTPException
# # # from fastapi.middleware.cors import CORSMiddleware
# # # from fastapi.responses import JSONResponse

# # # from main import GadgetDetectionPipeline
# # # from utils.db_s3_uploader import (
# # #     download_video_from_s3,
# # #     get_pending_videos,
# # #     set_process_flag,
# # # )

# # # # ── App setup ──────────────────────────────────────────────────────────────────

# # # app = FastAPI(
# # #     title   = "Loco Pilot Distraction Detection API",
# # #     version = "3.0.0",
# # # )

# # # app.add_middleware(
# # #     CORSMiddleware,
# # #     allow_origins     = ["*"],
# # #     allow_credentials = True,
# # #     allow_methods     = ["*"],
# # #     allow_headers     = ["*"],
# # # )


# # # # ── Routes ─────────────────────────────────────────────────────────────────────

# # # @app.get("/")
# # # def root() -> dict:
# # #     return {
# # #         "status":  "success",
# # #         "message": "Loco Pilot Distraction Detection API is running",
# # #         "health":  "/health",
# # #         "docs":    "/docs",
# # #         "trigger": "POST /trigger  — scan DB, process all pending videos, return results",
# # #     }


# # # @app.get("/health", tags=["status"])
# # # def health() -> dict:
# # #     return {"status": "ok"}


# # # # ── Main trigger ───────────────────────────────────────────────────────────────

# # # @app.post("/trigger", tags=["batch"])
# # # def trigger_batch() -> JSONResponse:
# # #     """
# # #     Scan DB for all rows with process_flag = 'N'.
# # #     Group by (train_detail_id, folder_name) — each group is one logical
# # #     analysis (a folder of sequential videos from one recording session).
# # #     Process every group in sequence, then return all results in the
# # #     target batch envelope format.

# # #     Response shape:
# # #     {
# # #       "batch_id":          "<hex>",
# # #       "total_videos":      N,
# # #       "completed":         N,
# # #       "failed":            N,
# # #       "folders_processed": N,
# # #       "folders": [
# # #         {
# # #           "train_detail_id":   22803,
# # #           "folder_name":       "22803-05-06-2026",
# # #           "videos_in_folder":  8,
# # #           "report": { ... }   ← full analysis_report.json content
# # #         },
# # #         ...
# # #       ]
# # #     }
# # #     """
# # #     try:
# # #         pending = get_pending_videos()
# # #     except Exception as exc:
# # #         raise HTTPException(
# # #             status_code = 500,
# # #             detail      = f"Failed to query pending videos from DB: {exc}",
# # #         )

# # #     if not pending:
# # #         return JSONResponse(content={
# # #             "batch_id":          uuid.uuid4().hex[:12],
# # #             "total_videos":      0,
# # #             "completed":         0,
# # #             "failed":            0,
# # #             "folders_processed": 0,
# # #             "folders":           [],
# # #             "message":           "No pending videos found (process_flag = 'N').",
# # #         })

# # #     batch_id = uuid.uuid4().hex[:12]

# # #     def _group_key(row: Dict[str, Any]):
# # #         return (row["train_detail_id"], row["folder_name"])

# # #     groups = [
# # #         (key, list(grp_rows))
# # #         for key, grp_rows in groupby(pending, key=_group_key)
# # #     ]
# # #     print(f"[Batch:{batch_id}] {len(pending)} video(s) across {len(groups)} folder group(s).")

# # #     total_videos = len(pending)
# # #     completed    = 0
# # #     failed       = 0
# # #     folders_out: List[Dict[str, Any]] = []

# # #     for (train_detail_id, folder_name), rows in groups:
# # #         report, error, n_completed, n_failed = _process_folder_group(
# # #             train_detail_id = train_detail_id,
# # #             folder_name     = folder_name,
# # #             rows            = rows,
# # #         )
# # #         completed += n_completed
# # #         failed    += n_failed

# # #         folder_entry: Dict[str, Any] = {
# # #             "train_detail_id":  train_detail_id,
# # #             "folder_name":      folder_name,
# # #             "videos_in_folder": len(rows),
# # #         }
# # #         if report is not None:
# # #             folder_entry["report"] = report
# # #         if error:
# # #             folder_entry["error"] = error

# # #         folders_out.append(folder_entry)

# # #     return JSONResponse(content={
# # #         "batch_id":          batch_id,
# # #         "total_videos":      total_videos,
# # #         "completed":         completed,
# # #         "failed":            failed,
# # #         "folders_processed": len(folders_out),
# # #         "folders":           folders_out,
# # #     })


# # # # ── Per-folder-group processor ─────────────────────────────────────────────────

# # # def _process_folder_group(
# # #     train_detail_id: int,
# # #     folder_name:     str,
# # #     rows:            List[Dict[str, Any]],
# # # ) -> Tuple[Optional[Dict[str, Any]], Optional[str], int, int]:
# # #     """
# # #     Process one folder group — all videos that share the same
# # #     (train_detail_id, folder_name), ordered by seq_no.

# # #     Returns (report_dict, error_str, n_completed, n_failed).
# # #     """
# # #     n_videos = len(rows)
# # #     print(
# # #         f"\n[Batch] ── Folder: train={train_detail_id}  "
# # #         f"folder='{folder_name}'  ({n_videos} video(s)) ──"
# # #     )

# # #     # folder_name is unique per recording session → use as analysis_id
# # #     analysis_id = folder_name

# # #     # Step 1 — mark all rows in-progress
# # #     for row in rows:
# # #         try:
# # #             set_process_flag(row["id"], "I")
# # #         except Exception as exc:
# # #             print(f"[Batch] Could not mark row {row['id']} as 'I': {exc}")

# # #     # Step 2 — download every video from S3 to a temp file
# # #     tmp_paths: List[str] = []
# # #     try:
# # #         for row in rows:
# # #             suffix = os.path.splitext(row["filename"] or "video.mp4")[1] or ".mp4"
# # #             tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
# # #             os.close(tmp_fd)
# # #             download_video_from_s3(row["s3_video_path"], tmp_path)
# # #             tmp_paths.append(tmp_path)
# # #     except Exception as exc:
# # #         err = f"folder='{folder_name}' download failed: {exc}"
# # #         print(f"[Batch] {err}")
# # #         _cleanup_temps(tmp_paths)
# # #         return None, err, 0, n_videos

# # #     # Step 3 — run the pipeline over each video in seq_no order
# # #     report_path = ""
# # #     try:
# # #         for idx, (row, tmp_path) in enumerate(zip(rows, tmp_paths)):
# # #             print(
# # #                 f"[Batch]   [{idx + 1}/{n_videos}]  "
# # #                 f"{row['filename']}  (seq={row['seq_no']})"
# # #             )
# # #             pipeline = GadgetDetectionPipeline(
# # #                 source          = tmp_path,
# # #                 analysis_id     = analysis_id,
# # #                 train_detail_id = train_detail_id,
# # #                 save            = False,
# # #                 display         = False,
# # #             )
# # #             report_path = pipeline.run()

# # #         # Step 4 — mark all rows done
# # #         for row in rows:
# # #             try:
# # #                 set_process_flag(row["id"], "Y")
# # #             except Exception as exc:
# # #                 print(f"[Batch] Could not mark row {row['id']} as 'Y': {exc}")

# # #         # Step 5 — read and return the report
# # #         if report_path and os.path.isfile(report_path):
# # #             with open(report_path, encoding="utf-8") as f:
# # #                 return json.load(f), None, n_videos, 0
# # #         else:
# # #             err = f"folder='{folder_name}' pipeline returned no report file"
# # #             print(f"[Batch] {err}")
# # #             return None, err, 0, n_videos

# # #     except Exception as exc:
# # #         # flags stay at 'I' — intentional, so operator can inspect
# # #         err = (
# # #             f"folder='{folder_name}' pipeline error: {exc}\n"
# # #             + traceback.format_exc()
# # #         )
# # #         print(f"[Batch] {err}")
# # #         return None, err, 0, n_videos

# # #     finally:
# # #         _cleanup_temps(tmp_paths)


# # # # ── Temp file cleanup ──────────────────────────────────────────────────────────

# # # def _cleanup_temps(paths: List[str]) -> None:
# # #     for p in paths:
# # #         try:
# # #             if os.path.isfile(p):
# # #                 os.remove(p)
# # #         except OSError:
# # #             pass

# # from __future__ import annotations

# # """
# # api.py — Loco Pilot Distraction Detection  (DB-driven batch runner)
# # ====================================================================

# # Videos are NOT uploaded through this API.
# # This service polls the database for video_files rows where
# # process_flag = 'N', downloads each video from S3, runs the detection
# # pipeline synchronously, and returns the final result directly.

# # Flag lifecycle
# # --------------
# #   N  →  pending   (set by the frontend when a video is uploaded to S3)
# #   I  →  in-progress (set here before the pipeline starts)
# #   Y  →  done      (set here only on successful pipeline completion)

# # If the pipeline crashes the flag stays at 'I' so operators can see
# # exactly which video failed.

# # Endpoints
# # ---------
# #   GET  /         — health / welcome
# #   GET  /health   — {"status": "ok"}
# #   POST /trigger  — scan DB → download from S3 → run pipeline → return result

# # Response shape
# # --------------
# # {
# #   "batch_id":          "444e4af5ff5f",
# #   "total_videos":      2,
# #   "completed":         2,
# #   "failed":            0,
# #   "folders_processed": 1,
# #   "folders": [
# #     {
# #       "train_detail_id":  22801,
# #       "folder_name":      "22801-05-06-2026",
# #       "videos_in_folder": 2,
# #       "report": {
# #         "analysis_id":     "22801-05-06-2026",
# #         "train_detail_id": 22801,
# #         "processing_time": 108.124,
# #         "video_info":      [ ... ],   ← list with one entry per video
# #         "violations":      [ ... ]    ← all violations across ALL videos
# #       }
# #     }
# #   ]
# # }

# # Timestamp logic
# # ---------------
# # For each folder group, videos are processed in seq_no order.
# # A running time_offset and frame_offset accumulate as each video finishes.

# #   global_timestamp  = local_video_time  + time_offset
# #   global_frame      = local_frame_index + frame_offset

# # In the report:
# #   "timestamp"                  = global_timestamp  (HH:MM:SS into the full recording)
# #   "original_video_timestamp"   = "<db_filename> <local_time>"  (time within that file)
# # """

# # import json
# # import os
# # import tempfile
# # import traceback
# # import uuid
# # from typing import Any, Dict, List, Optional, Tuple

# # import cv2

# # from fastapi import FastAPI, HTTPException
# # from fastapi.middleware.cors import CORSMiddleware
# # from fastapi.responses import JSONResponse

# # from main import GadgetDetectionPipeline
# # from utils.violation_store import ViolationStore
# # from utils.db_s3_uploader import (
# #     download_video_from_s3,
# #     get_pending_videos,
# #     set_process_flag,
# # )

# # # ── App setup ──────────────────────────────────────────────────────────────────

# # app = FastAPI(
# #     title   = "Loco Pilot Distraction Detection API",
# #     version = "3.0.0",
# # )

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins     = ["*"],
# #     allow_credentials = True,
# #     allow_methods     = ["*"],
# #     allow_headers     = ["*"],
# # )


# # # ── Routes ─────────────────────────────────────────────────────────────────────

# # @app.get("/")
# # def root() -> dict:
# #     return {
# #         "status":  "success",
# #         "message": "Loco Pilot Distraction Detection API is running",
# #         "health":  "/health",
# #         "docs":    "/docs",
# #         "trigger": "POST /trigger  — scan DB, process all pending videos, return results",
# #     }


# # @app.get("/health", tags=["status"])
# # def health() -> dict:
# #     return {"status": "ok"}


# # # ── Main trigger ───────────────────────────────────────────────────────────────

# # @app.post("/trigger", tags=["batch"])
# # def trigger_batch() -> JSONResponse:
# #     """
# #     Pick the SINGLE oldest pending folder (by upload_timestamp ASC LIMIT 1),
# #     process all its videos in seq_no order, and return the result.

# #     Call /trigger again to process the next folder.
# #     """
# #     batch_id = uuid.uuid4().hex[:12]

# #     # get_pending_videos() now returns only the videos of the one oldest folder
# #     try:
# #         pending = get_pending_videos()
# #     except Exception as exc:
# #         raise HTTPException(
# #             status_code = 500,
# #             detail      = f"Failed to query pending videos from DB: {exc}",
# #         )

# #     if not pending:
# #         return JSONResponse(content={
# #             "batch_id":          batch_id,
# #             "total_videos":      0,
# #             "completed":         0,
# #             "failed":            0,
# #             "folders_processed": 0,
# #             "folders":           [],
# #             "message":           "No pending videos found (process_flag = 'N').",
# #         })

# #     # All rows belong to the same folder — take metadata from the first row
# #     folder_name     = pending[0]["folder_name"]
# #     train_detail_id = pending[0]["train_detail_id"]
# #     total_videos    = len(pending)

# #     print(f"[Batch:{batch_id}] Processing folder='{folder_name}'  "
# #           f"train={train_detail_id}  videos={total_videos}")

# #     report, error, completed, failed = _process_folder_group(
# #         train_detail_id = train_detail_id,
# #         folder_name     = folder_name,
# #         rows            = pending,
# #     )

# #     folder_entry: Dict[str, Any] = {
# #         "train_detail_id":  train_detail_id,
# #         "folder_name":      folder_name,
# #         "videos_in_folder": total_videos,
# #     }
# #     if report is not None:
# #         folder_entry["report"] = report
# #     if error:
# #         folder_entry["error"] = error

# #     return JSONResponse(content={
# #         "batch_id":          batch_id,
# #         "total_videos":      total_videos,
# #         "completed":         completed,
# #         "failed":            failed,
# #         "folders_processed": 1,
# #         "folders":           [folder_entry],
# #     })


# # # ── Per-folder-group processor ─────────────────────────────────────────────────

# # def _video_duration_seconds(path: str) -> float:
# #     """Read the duration of a video file using OpenCV."""
# #     cap = cv2.VideoCapture(path)
# #     if not cap.isOpened():
# #         return 0.0
# #     fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
# #     total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# #     cap.release()
# #     return total / fps if fps > 0 and total > 0 else 0.0


# # def _process_folder_group(
# #     train_detail_id: int,
# #     folder_name:     str,
# #     rows:            List[Dict[str, Any]],
# # ) -> Tuple[Optional[Dict[str, Any]], Optional[str], int, int]:
# #     """
# #     Process one folder group — all videos that share the same
# #     (train_detail_id, folder_name), ordered by seq_no.

# #     Key design
# #     ──────────
# #     • One ViolationStore is created for the whole folder and shared
# #       across every pipeline run.  This means violations from all videos
# #       accumulate in a single store.
# #     • time_offset and frame_offset grow after each video so timestamps
# #       are continuous across the whole recording session.
# #     • finalize() is called ONCE after all videos are done.

# #     Returns (report_dict, error_str, n_completed, n_failed).
# #     """
# #     n_videos = len(rows)
# #     print(
# #         f"\n[Batch] ── Folder: train={train_detail_id}  "
# #         f"folder='{folder_name}'  ({n_videos} video(s)) ──"
# #     )

# #     analysis_id = folder_name

# #     # Step 1 — mark all rows in-progress
# #     for row in rows:
# #         try:
# #             set_process_flag(row["id"], "I")
# #         except Exception as exc:
# #             print(f"[Batch] Could not mark row {row['id']} as 'I': {exc}")

# #     # Step 2 — download every video from S3 to a temp file
# #     # We also record the DB filename alongside each temp path.
# #     tmp_entries: List[Tuple[str, str]] = []   # [(tmp_path, db_filename), ...]
# #     try:
# #         for row in rows:
# #             suffix = os.path.splitext(row["filename"] or "video.mp4")[1] or ".mp4"
# #             tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
# #             os.close(tmp_fd)
# #             download_video_from_s3(row["s3_video_path"], tmp_path)
# #             tmp_entries.append((tmp_path, row["filename"] or ""))
# #     except Exception as exc:
# #         err = f"folder='{folder_name}' download failed: {exc}"
# #         print(f"[Batch] {err}")
# #         _cleanup_temps([p for p, _ in tmp_entries])
# #         return None, err, 0, n_videos

# #     # Step 3 — create ONE shared ViolationStore for the entire folder
# #     shared_vstore = ViolationStore(
# #         analysis_id     = analysis_id,
# #         train_detail_id = train_detail_id,
# #         # No video_info here — each pipeline run will call add_video_info()
# #     )

# #     # Step 4 — run the pipeline over each video in seq_no order,
# #     #           accumulating time_offset and frame_offset between videos.
# #     time_offset  = 0.0
# #     frame_offset = 0
# #     total_processing_time = 0.0

# #     try:
# #         for idx, (row, (tmp_path, db_filename)) in enumerate(zip(rows, tmp_entries)):
# #             print(
# #                 f"[Batch]   [{idx + 1}/{n_videos}]  "
# #                 f"{db_filename}  (seq={row['seq_no']})  "
# #                 f"time_offset={time_offset:.2f}s  frame_offset={frame_offset}"
# #             )

# #             import time as _time
# #             t0 = _time.time()

# #             pipeline = GadgetDetectionPipeline(
# #                 source          = tmp_path,
# #                 analysis_id     = analysis_id,
# #                 train_detail_id = train_detail_id,
# #                 save            = False,
# #                 display         = False,
# #                 shared_vstore   = shared_vstore,
# #                 time_offset     = time_offset,
# #                 frame_offset    = frame_offset,
# #                 source_filename = db_filename,
# #             )
# #             pipeline.run()   # returns "" in batch mode; vstore is NOT finalized here

# #             video_duration = _video_duration_seconds(tmp_path)
# #             video_frames   = _get_frame_count(tmp_path)

# #             total_processing_time += _time.time() - t0

# #             # Advance offsets for the next video
# #             time_offset  += video_duration
# #             frame_offset += video_frames

# #         # Step 5 — finalize the shared store ONCE with total processing time
# #         report_path = shared_vstore.finalize(processing_time=round(total_processing_time, 3))

# #         # Step 6 — mark all rows done
# #         for row in rows:
# #             try:
# #                 set_process_flag(row["id"], "Y")
# #             except Exception as exc:
# #                 print(f"[Batch] Could not mark row {row['id']} as 'Y': {exc}")

# #         # Step 7 — read and return the report
# #         if report_path and os.path.isfile(report_path):
# #             with open(report_path, encoding="utf-8") as f:
# #                 return json.load(f), None, n_videos, 0
# #         else:
# #             err = f"folder='{folder_name}' no report file after finalize"
# #             print(f"[Batch] {err}")
# #             return None, err, 0, n_videos

# #     except Exception as exc:
# #         # flags stay at 'I' — intentional so operator can inspect
# #         err = (
# #             f"folder='{folder_name}' pipeline error: {exc}\n"
# #             + traceback.format_exc()
# #         )
# #         print(f"[Batch] {err}")
# #         return None, err, 0, n_videos

# #     finally:
# #         _cleanup_temps([p for p, _ in tmp_entries])


# # def _get_frame_count(path: str) -> int:
# #     cap = cv2.VideoCapture(path)
# #     if not cap.isOpened():
# #         return 0
# #     count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# #     cap.release()
# #     return count


# # # ── Temp file cleanup ──────────────────────────────────────────────────────────

# # def _cleanup_temps(paths: List[str]) -> None:
# #     for p in paths:
# #         try:
# #             if os.path.isfile(p):
# #                 os.remove(p)
# #         except OSError:
# #             pass



# # # # from __future__ import annotations

# # # # """
# # # # api.py — Loco Pilot Distraction Detection  (DB-driven batch runner)
# # # # ====================================================================

# # # # Videos are NO LONGER uploaded through this API.
# # # # Instead, this service polls the database for video_files rows where
# # # # process_flag = 'N', downloads each video from S3, runs the detection
# # # # pipeline, and marks the row done (process_flag = 'Y').

# # # # Flag lifecycle
# # # # --------------
# # # #   N  →  pending (set by the frontend / ingestion service when uploading)
# # # #   I  →  in-progress (set here, immediately before the pipeline starts)
# # # #   Y  →  done (set here, only on successful pipeline completion)

# # # # If the pipeline crashes the flag stays at 'I' so operators can see
# # # # exactly which video failed — it is NOT silently reset to 'N'.

# # # # Endpoints
# # # # ---------
# # # #   GET  /              — health / welcome
# # # #   GET  /health        — {"status": "ok"}
# # # #   POST /trigger       — kick off a DB-scan + batch run immediately
# # # #                         (returns job_id; poll /status/<job_id>)
# # # #   GET  /status/<id>   — queued | processing | done | failed
# # # #   GET  /result/<id>   — final JSON report (consumed once; deleted from memory)
# # # # """

# # # # import json
# # # # import os
# # # # import tempfile
# # # # import traceback
# # # # import uuid
# # # # from concurrent.futures import ThreadPoolExecutor
# # # # from itertools import groupby
# # # # from typing import Any, Dict, List, Optional

# # # # from fastapi import BackgroundTasks, FastAPI, HTTPException
# # # # from fastapi.middleware.cors import CORSMiddleware
# # # # from fastapi.responses import JSONResponse

# # # # from main import GadgetDetectionPipeline
# # # # from utils.db_s3_uploader import (
# # # #     download_video_from_s3,
# # # #     get_pending_videos,
# # # #     set_process_flag,
# # # # )

# # # # # ── App setup ──────────────────────────────────────────────────────────────────

# # # # app = FastAPI(
# # # #     title   = "Loco Pilot Distraction Detection API",
# # # #     version = "3.0.0",
# # # # )

# # # # app.add_middleware(
# # # #     CORSMiddleware,
# # # #     allow_origins     = ["*"],
# # # #     allow_credentials = True,
# # # #     allow_methods     = ["*"],
# # # #     allow_headers     = ["*"],
# # # # )

# # # # # In-memory job registry  {job_id: {"status": str, "result": dict|None, "error": str|None}}
# # # # _jobs:    Dict[str, Dict[str, Any]] = {}
# # # # _executor = ThreadPoolExecutor(max_workers=1)  # one batch at a time


# # # # # ── Routes ─────────────────────────────────────────────────────────────────────

# # # # @app.get("/")
# # # # def root() -> dict:
# # # #     return {
# # # #         "status":  "success",
# # # #         "message": "Loco Pilot Distraction Detection API is running",
# # # #         "health":  "/health",
# # # #         "docs":    "/docs",
# # # #         "trigger": "POST /trigger  — start a DB-scan + batch run",
# # # #     }


# # # # @app.get("/health", tags=["status"])
# # # # def health() -> dict:
# # # #     return {"status": "ok"}


# # # # # ── Batch trigger ──────────────────────────────────────────────────────────────

# # # # @app.post("/trigger", tags=["batch"])
# # # # async def trigger_batch(background_tasks: BackgroundTasks) -> JSONResponse:
# # # #     """
# # # #     Scan the DB for all pending videos (process_flag='N') and process them
# # # #     as a batch.  Returns immediately with a job_id; poll /status/<job_id>.
# # # #     """
# # # #     job_id = str(uuid.uuid4())
# # # #     _jobs[job_id] = {"status": "queued", "result": None, "error": None}
# # # #     _executor.submit(_run_batch, job_id)
# # # #     return JSONResponse(
# # # #         status_code = 202,
# # # #         content = {
# # # #             "job_id":  job_id,
# # # #             "status":  "queued",
# # # #             "message": f"Batch job accepted. Poll GET /status/{job_id} for progress.",
# # # #         },
# # # #     )


# # # # @app.get("/status/{job_id}", tags=["batch"])
# # # # def job_status(job_id: str) -> JSONResponse:
# # # #     job = _jobs.get(job_id)
# # # #     if job is None:
# # # #         raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
# # # #     resp: Dict[str, Any] = {"job_id": job_id, "status": job["status"]}
# # # #     if job["status"] == "failed":
# # # #         resp["error"] = job["error"]
# # # #     return JSONResponse(content=resp)


# # # # @app.get("/result/{job_id}", tags=["batch"])
# # # # def job_result(job_id: str) -> JSONResponse:
# # # #     job = _jobs.get(job_id)
# # # #     if job is None:
# # # #         raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
# # # #     if job["status"] == "failed":
# # # #         raise HTTPException(status_code=500, detail=job["error"])
# # # #     if job["status"] in ("queued", "processing"):
# # # #         raise HTTPException(
# # # #             status_code=409,
# # # #             detail=f"Job is still '{job['status']}'. Try again later.",
# # # #         )
# # # #     result = _jobs.pop(job_id)["result"]
# # # #     return JSONResponse(content=result)


# # # # # ── Batch worker ───────────────────────────────────────────────────────────────

# # # # def _run_batch(job_id: str) -> None:
# # # #     """
# # # #     Entry point executed in the thread pool.

# # # #     1. Query DB for all rows with process_flag = 'N'.
# # # #     2. Group them by (train_detail_id, folder_name) — each group is one
# # # #        logical analysis (the videos in a folder are one continuous recording).
# # # #     3. For every group:
# # # #          a. Mark every row as 'I' (in-progress).
# # # #          b. Download each video from S3 to a temp file.
# # # #          c. Run GadgetDetectionPipeline over every video in sequence.
# # # #          d. On success: mark every row as 'Y' and collect the report.
# # # #          e. On failure: leave rows at 'I' (flag stays for operator inspection).
# # # #     4. Collect per-group reports and write them to the job registry.
# # # #     """
# # # #     _jobs[job_id]["status"] = "processing"
# # # #     all_reports: List[Dict[str, Any]] = []

# # # #     try:
# # # #         pending = get_pending_videos()

# # # #         if not pending:
# # # #             print("[Batch] No pending videos found.")
# # # #             _jobs[job_id]["status"] = "done"
# # # #             _jobs[job_id]["result"] = {"message": "No pending videos.", "reports": []}
# # # #             return

# # # #         # Group by (train_detail_id, folder_name) — same order as the DB query
# # # #         def _group_key(row: Dict[str, Any]):
# # # #             return (row["train_detail_id"], row["folder_name"])

# # # #         groups = [
# # # #             (key, list(rows))
# # # #             for key, rows in groupby(pending, key=_group_key)
# # # #         ]
# # # #         print(f"[Batch] {len(pending)} video(s) across {len(groups)} folder group(s).")

# # # #         for (train_detail_id, folder_name), rows in groups:
# # # #             report = _process_folder_group(
# # # #                 train_detail_id = train_detail_id,
# # # #                 folder_name     = folder_name,
# # # #                 rows            = rows,
# # # #             )
# # # #             if report:
# # # #                 all_reports.append(report)

# # # #         _jobs[job_id]["status"] = "done"
# # # #         _jobs[job_id]["result"] = {
# # # #             "message": f"Processed {len(all_reports)} folder group(s).",
# # # #             "reports": all_reports,
# # # #         }

# # # #     except Exception as exc:
# # # #         _jobs[job_id]["status"] = "failed"
# # # #         _jobs[job_id]["error"]  = f"{exc}\n{traceback.format_exc()}"
# # # #         print(f"[Batch] Fatal error: {exc}")


# # # # def _process_folder_group(
# # # #     train_detail_id: int,
# # # #     folder_name:     str,
# # # #     rows:            List[Dict[str, Any]],
# # # # ) -> Optional[Dict[str, Any]]:
# # # #     """
# # # #     Process one folder group (all videos that belong to a single analysis).

# # # #     Rows are already ordered by seq_no (guaranteed by the DB query).
# # # #     Each video is processed in sequence so frame offsets accumulate
# # # #     correctly across files.

# # # #     Returns the JSON report dict on success, None on failure.
# # # #     """
# # # #     print(
# # # #         f"\n[Batch] ── Folder group: train={train_detail_id}  "
# # # #         f"folder='{folder_name}'  ({len(rows)} video(s)) ──"
# # # #     )

# # # #     # Use folder_name as the analysis_id (unique per recording session)
# # # #     analysis_id = folder_name

# # # #     # ── Step 1: mark all rows as in-progress ─────────────────────────────────
# # # #     for row in rows:
# # # #         try:
# # # #             set_process_flag(row["id"], "I")
# # # #         except Exception as exc:
# # # #             print(f"[Batch] Could not mark row {row['id']} as 'I': {exc}")
# # # #             # Non-fatal — continue; the row stays 'N' which is safer than crashing

# # # #     # ── Step 2: download videos to temp files ─────────────────────────────────
# # # #     tmp_paths: List[str] = []
# # # #     try:
# # # #         for row in rows:
# # # #             suffix    = os.path.splitext(row["filename"] or "video.mp4")[1] or ".mp4"
# # # #             tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
# # # #             os.close(tmp_fd)
# # # #             download_video_from_s3(row["s3_video_path"], tmp_path)
# # # #             tmp_paths.append(tmp_path)
# # # #     except Exception as exc:
# # # #         print(f"[Batch] Download failed for folder '{folder_name}': {exc}")
# # # #         _cleanup_temps(tmp_paths)
# # # #         return None

# # # #     # ── Step 3: run the pipeline over each video in sequence ──────────────────
# # # #     report_path: str = ""
# # # #     try:
# # # #         # The pipeline processes one video at a time but shares the same
# # # #         # ViolationStore (via analysis_id = folder_name) so violations from
# # # #         # every video land in the same report.
# # # #         for idx, (row, tmp_path) in enumerate(zip(rows, tmp_paths)):
# # # #             print(
# # # #                 f"[Batch]   [{idx+1}/{len(rows)}] Processing  "
# # # #                 f"{row['filename']}  (seq={row['seq_no']})"
# # # #             )
# # # #             pipeline = GadgetDetectionPipeline(
# # # #                 source          = tmp_path,
# # # #                 analysis_id     = analysis_id,
# # # #                 train_detail_id = train_detail_id,
# # # #                 save            = False,
# # # #                 display         = False,
# # # #             )
# # # #             # run() returns the path to analysis_report.json
# # # #             report_path = pipeline.run()

# # # #         # ── Step 4: mark all rows as done ────────────────────────────────────
# # # #         for row in rows:
# # # #             try:
# # # #                 set_process_flag(row["id"], "Y")
# # # #             except Exception as exc:
# # # #                 print(f"[Batch] Could not mark row {row['id']} as 'Y': {exc}")

# # # #         # ── Step 5: read and return the report ───────────────────────────────
# # # #         if report_path and os.path.isfile(report_path):
# # # #             with open(report_path, encoding="utf-8") as f:
# # # #                 return json.load(f)
# # # #         else:
# # # #             print(f"[Batch] Report file missing for folder '{folder_name}'")
# # # #             return None

# # # #     except Exception as exc:
# # # #         # Leave flags at 'I' so the operator can see which group failed
# # # #         print(
# # # #             f"[Batch] Pipeline error for folder '{folder_name}': {exc}\n"
# # # #             + traceback.format_exc()
# # # #         )
# # # #         return None

# # # #     finally:
# # # #         _cleanup_temps(tmp_paths)


# # # # def _cleanup_temps(paths: List[str]) -> None:
# # # #     for p in paths:
# # # #         try:
# # # #             if os.path.isfile(p):
# # # #                 os.remove(p)
# # # #         except OSError:
# # # #             pass



# # # from __future__ import annotations

# # # """
# # # api.py — Loco Pilot Distraction Detection  (DB-driven batch runner)
# # # ====================================================================

# # # Videos are NOT uploaded through this API.
# # # This service polls the database for video_files rows where
# # # process_flag = 'N', downloads each video from S3, runs the detection
# # # pipeline synchronously, and returns the final result directly.

# # # Flag lifecycle
# # # --------------
# # #   N  →  pending   (set by the frontend when a video is uploaded to S3)
# # #   I  →  in-progress (set here before the pipeline starts)
# # #   Y  →  done      (set here only on successful pipeline completion)

# # # If the pipeline crashes the flag stays at 'I' so operators can see
# # # exactly which video failed.

# # # Endpoints
# # # ---------
# # #   GET  /         — health / welcome
# # #   GET  /health   — {"status": "ok"}
# # #   POST /trigger  — scan DB → download from S3 → run pipeline → return result
# # # """

# # # import json
# # # import os
# # # import tempfile
# # # import traceback
# # # from itertools import groupby
# # # from typing import Any, Dict, List, Optional

# # # from fastapi import FastAPI, HTTPException
# # # from fastapi.middleware.cors import CORSMiddleware
# # # from fastapi.responses import JSONResponse

# # # from main import GadgetDetectionPipeline
# # # from utils.db_s3_uploader import (
# # #     download_video_from_s3,
# # #     get_pending_videos,
# # #     set_process_flag,
# # # )

# # # # ── App setup ──────────────────────────────────────────────────────────────────

# # # app = FastAPI(
# # #     title   = "Loco Pilot Distraction Detection API",
# # #     version = "3.0.0",
# # # )

# # # app.add_middleware(
# # #     CORSMiddleware,
# # #     allow_origins     = ["*"],
# # #     allow_credentials = True,
# # #     allow_methods     = ["*"],
# # #     allow_headers     = ["*"],
# # # )


# # # # ── Routes ─────────────────────────────────────────────────────────────────────

# # # @app.get("/")
# # # def root() -> dict:
# # #     return {
# # #         "status":  "success",
# # #         "message": "Loco Pilot Distraction Detection API is running",
# # #         "health":  "/health",
# # #         "docs":    "/docs",
# # #         "trigger": "POST /trigger  — scan DB, process all pending videos, return results",
# # #     }


# # # @app.get("/health", tags=["status"])
# # # def health() -> dict:
# # #     return {"status": "ok"}


# # # # ── Main trigger ───────────────────────────────────────────────────────────────

# # # @app.post("/trigger", tags=["batch"])
# # # def trigger_batch() -> JSONResponse:
# # #     """
# # #     Scan DB for all rows with process_flag = 'N'.
# # #     Group by (train_detail_id, folder_name) — each group is one logical
# # #     analysis (a folder of sequential videos from one recording session).
# # #     Process every group in sequence, then return all reports together.

# # #     Flag lifecycle per video row:
# # #       N  →  I  (before pipeline starts)
# # #       I  →  Y  (after pipeline succeeds)
# # #       stays I   (if pipeline fails — visible to operators)
# # #     """
# # #     try:
# # #         pending = get_pending_videos()
# # #     except Exception as exc:
# # #         raise HTTPException(
# # #             status_code=500,
# # #             detail=f"Failed to query pending videos from DB: {exc}",
# # #         )

# # #     if not pending:
# # #         return JSONResponse(content={
# # #             "status":  "ok",
# # #             "message": "No pending videos found (process_flag = 'N').",
# # #             "reports": [],
# # #         })

# # #     def _group_key(row: Dict[str, Any]):
# # #         return (row["train_detail_id"], row["folder_name"])

# # #     groups = [
# # #         (key, list(rows))
# # #         for key, rows in groupby(pending, key=_group_key)
# # #     ]
# # #     print(f"[Batch] {len(pending)} video(s) across {len(groups)} folder group(s).")

# # #     all_reports: List[Dict[str, Any]] = []
# # #     errors:      List[str]            = []

# # #     for (train_detail_id, folder_name), rows in groups:
# # #         report, error = _process_folder_group(
# # #             train_detail_id = train_detail_id,
# # #             folder_name     = folder_name,
# # #             rows            = rows,
# # #         )
# # #         if report is not None:
# # #             all_reports.append(report)
# # #         if error:
# # #             errors.append(error)

# # #     return JSONResponse(content={
# # #         "status":         "ok" if not errors else "partial",
# # #         "groups_total":   len(groups),
# # #         "groups_success": len(all_reports),
# # #         "groups_failed":  len(errors),
# # #         "errors":         errors,
# # #         "reports":        all_reports,
# # #     })


# # # # ── Per-folder-group processor ─────────────────────────────────────────────────

# # # def _process_folder_group(
# # #     train_detail_id: int,
# # #     folder_name:     str,
# # #     rows:            List[Dict[str, Any]],
# # # ) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
# # #     """
# # #     Process one folder group — all videos that share the same
# # #     (train_detail_id, folder_name), ordered by seq_no.

# # #     Returns (report_dict, None) on success, (None, error_str) on failure.
# # #     """
# # #     print(
# # #         f"\n[Batch] ── Folder: train={train_detail_id}  "
# # #         f"folder='{folder_name}'  ({len(rows)} video(s)) ──"
# # #     )

# # #     # folder_name is unique per recording session → use as analysis_id
# # #     analysis_id = folder_name

# # #     # Step 1 — mark all rows in-progress
# # #     for row in rows:
# # #         try:
# # #             set_process_flag(row["id"], "I")
# # #         except Exception as exc:
# # #             print(f"[Batch] Could not mark row {row['id']} as 'I': {exc}")

# # #     # Step 2 — download every video from S3 to a temp file
# # #     tmp_paths: List[str] = []
# # #     try:
# # #         for row in rows:
# # #             suffix = os.path.splitext(row["filename"] or "video.mp4")[1] or ".mp4"
# # #             tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
# # #             os.close(tmp_fd)
# # #             download_video_from_s3(row["s3_video_path"], tmp_path)
# # #             tmp_paths.append(tmp_path)
# # #     except Exception as exc:
# # #         err = f"folder='{folder_name}' download failed: {exc}"
# # #         print(f"[Batch] {err}")
# # #         _cleanup_temps(tmp_paths)
# # #         return None, err

# # #     # Step 3 — run the pipeline over each video in seq_no order
# # #     report_path = ""
# # #     try:
# # #         for idx, (row, tmp_path) in enumerate(zip(rows, tmp_paths)):
# # #             print(
# # #                 f"[Batch]   [{idx + 1}/{len(rows)}]  "
# # #                 f"{row['filename']}  (seq={row['seq_no']})"
# # #             )
# # #             pipeline = GadgetDetectionPipeline(
# # #                 source          = tmp_path,
# # #                 analysis_id     = analysis_id,
# # #                 train_detail_id = train_detail_id,
# # #                 save            = False,
# # #                 display         = False,
# # #             )
# # #             report_path = pipeline.run()

# # #         # Step 4 — mark all rows done
# # #         for row in rows:
# # #             try:
# # #                 set_process_flag(row["id"], "Y")
# # #             except Exception as exc:
# # #                 print(f"[Batch] Could not mark row {row['id']} as 'Y': {exc}")

# # #         # Step 5 — read and return the report
# # #         if report_path and os.path.isfile(report_path):
# # #             with open(report_path, encoding="utf-8") as f:
# # #                 return json.load(f), None
# # #         else:
# # #             err = f"folder='{folder_name}' pipeline returned no report file"
# # #             print(f"[Batch] {err}")
# # #             return None, err

# # #     except Exception as exc:
# # #         # flags stay at 'I' — intentional, so operator can inspect
# # #         err = (
# # #             f"folder='{folder_name}' pipeline error: {exc}\n"
# # #             + traceback.format_exc()
# # #         )
# # #         print(f"[Batch] {err}")
# # #         return None, err

# # #     finally:
# # #         _cleanup_temps(tmp_paths)


# # # # ── Temp file cleanup ──────────────────────────────────────────────────────────

# # # def _cleanup_temps(paths: List[str]) -> None:
# # #     for p in paths:
# # #         try:
# # #             if os.path.isfile(p):
# # #                 os.remove(p)
# # #         except OSError:
# # #             pass


# # from __future__ import annotations

# # """
# # api.py — Loco Pilot Distraction Detection  (DB-driven batch runner)
# # ====================================================================

# # Videos are NOT uploaded through this API.
# # This service polls the database for video_files rows where
# # process_flag = 'N', downloads each video from S3, runs the detection
# # pipeline synchronously, and returns the final result directly.

# # Flag lifecycle
# # --------------
# #   N  →  pending   (set by the frontend when a video is uploaded to S3)
# #   I  →  in-progress (set here before the pipeline starts)
# #   Y  →  done      (set here only on successful pipeline completion)

# # If the pipeline crashes the flag stays at 'I' so operators can see
# # exactly which video failed.

# # Endpoints
# # ---------
# #   GET  /         — health / welcome
# #   GET  /health   — {"status": "ok"}
# #   POST /trigger  — scan DB → download from S3 → run pipeline → return result
# # """

# # import json
# # import os
# # import tempfile
# # import traceback
# # import uuid
# # from itertools import groupby
# # from typing import Any, Dict, List, Optional, Tuple

# # from fastapi import FastAPI, HTTPException
# # from fastapi.middleware.cors import CORSMiddleware
# # from fastapi.responses import JSONResponse

# # from main import GadgetDetectionPipeline
# # from utils.db_s3_uploader import (
# #     download_video_from_s3,
# #     get_pending_videos,
# #     set_process_flag,
# # )

# # # ── App setup ──────────────────────────────────────────────────────────────────

# # app = FastAPI(
# #     title   = "Loco Pilot Distraction Detection API",
# #     version = "3.0.0",
# # )

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins     = ["*"],
# #     allow_credentials = True,
# #     allow_methods     = ["*"],
# #     allow_headers     = ["*"],
# # )


# # # ── Routes ─────────────────────────────────────────────────────────────────────

# # @app.get("/")
# # def root() -> dict:
# #     return {
# #         "status":  "success",
# #         "message": "Loco Pilot Distraction Detection API is running",
# #         "health":  "/health",
# #         "docs":    "/docs",
# #         "trigger": "POST /trigger  — scan DB, process all pending videos, return results",
# #     }


# # @app.get("/health", tags=["status"])
# # def health() -> dict:
# #     return {"status": "ok"}


# # # ── Main trigger ───────────────────────────────────────────────────────────────

# # @app.post("/trigger", tags=["batch"])
# # def trigger_batch() -> JSONResponse:
# #     """
# #     Scan DB for all rows with process_flag = 'N'.
# #     Group by (train_detail_id, folder_name) — each group is one logical
# #     analysis (a folder of sequential videos from one recording session).
# #     Process every group in sequence, then return all results in the
# #     target batch envelope format.

# #     Response shape:
# #     {
# #       "batch_id":          "<hex>",
# #       "total_videos":      N,
# #       "completed":         N,
# #       "failed":            N,
# #       "folders_processed": N,
# #       "folders": [
# #         {
# #           "train_detail_id":   22803,
# #           "folder_name":       "22803-05-06-2026",
# #           "videos_in_folder":  8,
# #           "report": { ... }   ← full analysis_report.json content
# #         },
# #         ...
# #       ]
# #     }
# #     """
# #     try:
# #         pending = get_pending_videos()
# #     except Exception as exc:
# #         raise HTTPException(
# #             status_code = 500,
# #             detail      = f"Failed to query pending videos from DB: {exc}",
# #         )

# #     if not pending:
# #         return JSONResponse(content={
# #             "batch_id":          uuid.uuid4().hex[:12],
# #             "total_videos":      0,
# #             "completed":         0,
# #             "failed":            0,
# #             "folders_processed": 0,
# #             "folders":           [],
# #             "message":           "No pending videos found (process_flag = 'N').",
# #         })

# #     batch_id = uuid.uuid4().hex[:12]

# #     def _group_key(row: Dict[str, Any]):
# #         return (row["train_detail_id"], row["folder_name"])

# #     groups = [
# #         (key, list(grp_rows))
# #         for key, grp_rows in groupby(pending, key=_group_key)
# #     ]
# #     print(f"[Batch:{batch_id}] {len(pending)} video(s) across {len(groups)} folder group(s).")

# #     total_videos = len(pending)
# #     completed    = 0
# #     failed       = 0
# #     folders_out: List[Dict[str, Any]] = []

# #     for (train_detail_id, folder_name), rows in groups:
# #         report, error, n_completed, n_failed = _process_folder_group(
# #             train_detail_id = train_detail_id,
# #             folder_name     = folder_name,
# #             rows            = rows,
# #         )
# #         completed += n_completed
# #         failed    += n_failed

# #         folder_entry: Dict[str, Any] = {
# #             "train_detail_id":  train_detail_id,
# #             "folder_name":      folder_name,
# #             "videos_in_folder": len(rows),
# #         }
# #         if report is not None:
# #             folder_entry["report"] = report
# #         if error:
# #             folder_entry["error"] = error

# #         folders_out.append(folder_entry)

# #     return JSONResponse(content={
# #         "batch_id":          batch_id,
# #         "total_videos":      total_videos,
# #         "completed":         completed,
# #         "failed":            failed,
# #         "folders_processed": len(folders_out),
# #         "folders":           folders_out,
# #     })


# # # ── Per-folder-group processor ─────────────────────────────────────────────────

# # def _process_folder_group(
# #     train_detail_id: int,
# #     folder_name:     str,
# #     rows:            List[Dict[str, Any]],
# # ) -> Tuple[Optional[Dict[str, Any]], Optional[str], int, int]:
# #     """
# #     Process one folder group — all videos that share the same
# #     (train_detail_id, folder_name), ordered by seq_no.

# #     Returns (report_dict, error_str, n_completed, n_failed).
# #     """
# #     n_videos = len(rows)
# #     print(
# #         f"\n[Batch] ── Folder: train={train_detail_id}  "
# #         f"folder='{folder_name}'  ({n_videos} video(s)) ──"
# #     )

# #     # folder_name is unique per recording session → use as analysis_id
# #     analysis_id = folder_name

# #     # Step 1 — mark all rows in-progress
# #     for row in rows:
# #         try:
# #             set_process_flag(row["id"], "I")
# #         except Exception as exc:
# #             print(f"[Batch] Could not mark row {row['id']} as 'I': {exc}")

# #     # Step 2 — download every video from S3 to a temp file
# #     tmp_paths: List[str] = []
# #     try:
# #         for row in rows:
# #             suffix = os.path.splitext(row["filename"] or "video.mp4")[1] or ".mp4"
# #             tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
# #             os.close(tmp_fd)
# #             download_video_from_s3(row["s3_video_path"], tmp_path)
# #             tmp_paths.append(tmp_path)
# #     except Exception as exc:
# #         err = f"folder='{folder_name}' download failed: {exc}"
# #         print(f"[Batch] {err}")
# #         _cleanup_temps(tmp_paths)
# #         return None, err, 0, n_videos

# #     # Step 3 — run the pipeline over each video in seq_no order
# #     report_path = ""
# #     try:
# #         for idx, (row, tmp_path) in enumerate(zip(rows, tmp_paths)):
# #             print(
# #                 f"[Batch]   [{idx + 1}/{n_videos}]  "
# #                 f"{row['filename']}  (seq={row['seq_no']})"
# #             )
# #             pipeline = GadgetDetectionPipeline(
# #                 source          = tmp_path,
# #                 analysis_id     = analysis_id,
# #                 train_detail_id = train_detail_id,
# #                 save            = False,
# #                 display         = False,
# #             )
# #             report_path = pipeline.run()

# #         # Step 4 — mark all rows done
# #         for row in rows:
# #             try:
# #                 set_process_flag(row["id"], "Y")
# #             except Exception as exc:
# #                 print(f"[Batch] Could not mark row {row['id']} as 'Y': {exc}")

# #         # Step 5 — read and return the report
# #         if report_path and os.path.isfile(report_path):
# #             with open(report_path, encoding="utf-8") as f:
# #                 return json.load(f), None, n_videos, 0
# #         else:
# #             err = f"folder='{folder_name}' pipeline returned no report file"
# #             print(f"[Batch] {err}")
# #             return None, err, 0, n_videos

# #     except Exception as exc:
# #         # flags stay at 'I' — intentional, so operator can inspect
# #         err = (
# #             f"folder='{folder_name}' pipeline error: {exc}\n"
# #             + traceback.format_exc()
# #         )
# #         print(f"[Batch] {err}")
# #         return None, err, 0, n_videos

# #     finally:
# #         _cleanup_temps(tmp_paths)


# # # ── Temp file cleanup ──────────────────────────────────────────────────────────

# # def _cleanup_temps(paths: List[str]) -> None:
# #     for p in paths:
# #         try:
# #             if os.path.isfile(p):
# #                 os.remove(p)
# #         except OSError:
# #             pass

# from __future__ import annotations

# """
# api.py — Loco Pilot Distraction Detection  (DB-driven batch runner)
# ====================================================================

# Videos are NOT uploaded through this API.
# This service polls the database for video_files rows where
# process_flag = 'N', downloads each video from S3, runs the detection
# pipeline synchronously, and returns the final result directly.

# Flag lifecycle
# --------------
#   N  →  pending   (set by the frontend when a video is uploaded to S3)
#   I  →  in-progress (set here before the pipeline starts)
#   Y  →  done      (set here only on successful pipeline completion)

# If the pipeline crashes the flag stays at 'I' so operators can see
# exactly which video failed.

# Endpoints
# ---------
#   GET  /         — health / welcome
#   GET  /health   — {"status": "ok"}
#   POST /trigger  — scan DB → download from S3 → run pipeline → return result

# Response shape
# --------------
# {
#   "batch_id":          "444e4af5ff5f",
#   "total_videos":      2,
#   "completed":         2,
#   "failed":            0,
#   "folders_processed": 1,
#   "folders": [
#     {
#       "train_detail_id":  22801,
#       "folder_name":      "22801-05-06-2026",
#       "videos_in_folder": 2,
#       "report": {
#         "analysis_id":     "22801-05-06-2026",
#         "train_detail_id": 22801,
#         "processing_time": 108.124,
#         "video_info":      [ ... ],   ← list with one entry per video
#         "violations":      [ ... ]    ← all violations across ALL videos
#       }
#     }
#   ]
# }

# Timestamp logic
# ---------------
# For each folder group, videos are processed in seq_no order.
# A running time_offset and frame_offset accumulate as each video finishes.

#   global_timestamp  = local_video_time  + time_offset
#   global_frame      = local_frame_index + frame_offset

# In the report:
#   "timestamp"                  = global_timestamp  (HH:MM:SS into the full recording)
#   "original_video_timestamp"   = "<db_filename> <local_time>"  (time within that file)
# """

# import json
# import os
# import tempfile
# import traceback
# import uuid
# from typing import Any, Dict, List, Optional, Tuple

# import cv2

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse

# from main import GadgetDetectionPipeline
# from utils.violation_store import ViolationStore
# from utils.db_s3_uploader import (
#     download_video_from_s3,
#     get_pending_videos,
#     set_process_flag,
# )

# # ── App setup ──────────────────────────────────────────────────────────────────

# app = FastAPI(
#     title   = "Loco Pilot Distraction Detection API",
#     version = "3.0.0",
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins     = ["*"],
#     allow_credentials = True,
#     allow_methods     = ["*"],
#     allow_headers     = ["*"],
# )


# # ── Routes ─────────────────────────────────────────────────────────────────────

# @app.get("/")
# def root() -> dict:
#     return {
#         "status":  "success",
#         "message": "Loco Pilot Distraction Detection API is running",
#         "health":  "/health",
#         "docs":    "/docs",
#         "trigger": "POST /trigger  — scan DB, process all pending videos, return results",
#     }


# @app.get("/health", tags=["status"])
# def health() -> dict:
#     return {"status": "ok"}


# # ── Main trigger ───────────────────────────────────────────────────────────────

# @app.post("/trigger", tags=["batch"])
# def trigger_batch() -> JSONResponse:
#     """
#     Process ALL pending folders one at a time, in upload_timestamp ASC order.

#     Flow per iteration
#     ──────────────────
#     1. get_pending_videos()  →  returns videos of the ONE oldest pending folder
#                                 (ORDER BY upload_timestamp ASC LIMIT 1 on folder,
#                                  then all its videos by seq_no)
#     2. Process that folder completely  (N → I → pipeline → Y)
#     3. Loop again — because those rows are now 'Y', the next call to
#        get_pending_videos() naturally returns the NEXT oldest folder
#     4. Repeat until get_pending_videos() returns []  →  all done

#     Result: one POST /trigger processes every pending folder,
#     strictly one folder at a time, in upload order.
#     """
#     batch_id     = uuid.uuid4().hex[:12]
#     total_videos = 0
#     completed    = 0
#     failed       = 0
#     folders_out: List[Dict[str, Any]] = []
#     folder_index = 0

#     print(f"[Batch:{batch_id}] Starting — processing all pending folders one by one (upload_timestamp ASC).")

#     while True:
#         # ── Fetch the next oldest pending folder ─────────────────────────
#         try:
#             pending = get_pending_videos()   # always returns 1 folder or []
#         except Exception as exc:
#             raise HTTPException(
#                 status_code = 500,
#                 detail      = f"DB query failed (folder #{folder_index + 1}): {exc}",
#             )

#         if not pending:
#             break   # no more pending folders

#         folder_index    += 1
#         folder_name      = pending[0]["folder_name"]
#         train_detail_id  = pending[0]["train_detail_id"]
#         n_videos         = len(pending)
#         total_videos    += n_videos

#         print(f"[Batch:{batch_id}] ── Folder {folder_index}: '{folder_name}'  "
#               f"train={train_detail_id}  videos={n_videos} ──")

#         # ── Process this folder completely ───────────────────────────────
#         report, error, n_ok, n_fail = _process_folder_group(
#             train_detail_id = train_detail_id,
#             folder_name     = folder_name,
#             rows            = pending,
#         )
#         completed += n_ok
#         failed    += n_fail

#         folder_entry: Dict[str, Any] = {
#             "train_detail_id":  train_detail_id,
#             "folder_name":      folder_name,
#             "videos_in_folder": n_videos,
#         }
#         if report is not None:
#             folder_entry["report"] = report
#         if error:
#             folder_entry["error"] = error

#         folders_out.append(folder_entry)
#         print(f"[Batch:{batch_id}] Folder '{folder_name}' done — checking for next pending folder...")

#     # ── All done ─────────────────────────────────────────────────────────
#     if not folders_out:
#         return JSONResponse(content={
#             "batch_id":          batch_id,
#             "total_videos":      0,
#             "completed":         0,
#             "failed":            0,
#             "folders_processed": 0,
#             "folders":           [],
#             "message":           "No pending videos found (process_flag = 'N').",
#         })

#     print(f"[Batch:{batch_id}] All folders done.  "
#           f"folders={len(folders_out)}  videos={total_videos}  "
#           f"completed={completed}  failed={failed}")

#     return JSONResponse(content={
#         "batch_id":          batch_id,
#         "total_videos":      total_videos,
#         "completed":         completed,
#         "failed":            failed,
#         "folders_processed": len(folders_out),
#         "folders":           folders_out,
#     })


# # ── Per-folder-group processor ─────────────────────────────────────────────────

# def _video_duration_seconds(path: str) -> float:
#     """Read the duration of a video file using OpenCV."""
#     cap = cv2.VideoCapture(path)
#     if not cap.isOpened():
#         return 0.0
#     fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
#     total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#     cap.release()
#     return total / fps if fps > 0 and total > 0 else 0.0


# def _process_folder_group(
#     train_detail_id: int,
#     folder_name:     str,
#     rows:            List[Dict[str, Any]],
# ) -> Tuple[Optional[Dict[str, Any]], Optional[str], int, int]:
#     """
#     Process one folder group — all videos that share the same
#     (train_detail_id, folder_name), ordered by seq_no.

#     Key design
#     ──────────
#     • One ViolationStore is created for the whole folder and shared
#       across every pipeline run.  This means violations from all videos
#       accumulate in a single store.
#     • time_offset and frame_offset grow after each video so timestamps
#       are continuous across the whole recording session.
#     • finalize() is called ONCE after all videos are done.

#     Returns (report_dict, error_str, n_completed, n_failed).
#     """
#     n_videos = len(rows)
#     print(
#         f"\n[Batch] ── Folder: train={train_detail_id}  "
#         f"folder='{folder_name}'  ({n_videos} video(s)) ──"
#     )

#     analysis_id = folder_name

#     # Step 1 — mark all rows in-progress
#     for row in rows:
#         try:
#             set_process_flag(row["id"], "I")
#         except Exception as exc:
#             print(f"[Batch] Could not mark row {row['id']} as 'I': {exc}")

#     # Step 2 — download every video from S3 to a temp file
#     # We also record the DB filename alongside each temp path.
#     tmp_entries: List[Tuple[str, str]] = []   # [(tmp_path, db_filename), ...]
#     try:
#         for row in rows:
#             suffix = os.path.splitext(row["filename"] or "video.mp4")[1] or ".mp4"
#             tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
#             os.close(tmp_fd)
#             download_video_from_s3(row["s3_video_path"], tmp_path)
#             tmp_entries.append((tmp_path, row["filename"] or ""))
#     except Exception as exc:
#         err = f"folder='{folder_name}' download failed: {exc}"
#         print(f"[Batch] {err}")
#         _cleanup_temps([p for p, _ in tmp_entries])
#         return None, err, 0, n_videos

#     # Step 3 — create ONE shared ViolationStore for the entire folder
#     shared_vstore = ViolationStore(
#         analysis_id     = analysis_id,
#         train_detail_id = train_detail_id,
#         # No video_info here — each pipeline run will call add_video_info()
#     )

#     # Step 4 — run the pipeline over each video in seq_no order,
#     #           accumulating time_offset and frame_offset between videos.
#     time_offset  = 0.0
#     frame_offset = 0
#     total_processing_time = 0.0

#     try:
#         for idx, (row, (tmp_path, db_filename)) in enumerate(zip(rows, tmp_entries)):
#             print(
#                 f"[Batch]   [{idx + 1}/{n_videos}]  "
#                 f"{db_filename}  (seq={row['seq_no']})  "
#                 f"time_offset={time_offset:.2f}s  frame_offset={frame_offset}"
#             )

#             import time as _time
#             t0 = _time.time()

#             pipeline = GadgetDetectionPipeline(
#                 source          = tmp_path,
#                 analysis_id     = analysis_id,
#                 train_detail_id = train_detail_id,
#                 save            = False,
#                 display         = False,
#                 shared_vstore   = shared_vstore,
#                 time_offset     = time_offset,
#                 frame_offset    = frame_offset,
#                 source_filename = db_filename,
#             )
#             pipeline.run()   # returns "" in batch mode; vstore is NOT finalized here

#             video_duration = _video_duration_seconds(tmp_path)
#             video_frames   = _get_frame_count(tmp_path)

#             total_processing_time += _time.time() - t0

#             # Advance offsets for the next video
#             time_offset  += video_duration
#             frame_offset += video_frames

#         # Step 5 — finalize the shared store ONCE with total processing time
#         report_path = shared_vstore.finalize(processing_time=round(total_processing_time, 3))

#         # Step 6 — mark all rows done
#         for row in rows:
#             try:
#                 set_process_flag(row["id"], "Y")
#             except Exception as exc:
#                 print(f"[Batch] Could not mark row {row['id']} as 'Y': {exc}")

#         # Step 7 — read and return the report
#         if report_path and os.path.isfile(report_path):
#             with open(report_path, encoding="utf-8") as f:
#                 return json.load(f), None, n_videos, 0
#         else:
#             err = f"folder='{folder_name}' no report file after finalize"
#             print(f"[Batch] {err}")
#             return None, err, 0, n_videos

#     except Exception as exc:
#         # flags stay at 'I' — intentional so operator can inspect
#         err = (
#             f"folder='{folder_name}' pipeline error: {exc}\n"
#             + traceback.format_exc()
#         )
#         print(f"[Batch] {err}")
#         return None, err, 0, n_videos

#     finally:
#         _cleanup_temps([p for p, _ in tmp_entries])


# def _get_frame_count(path: str) -> int:
#     cap = cv2.VideoCapture(path)
#     if not cap.isOpened():
#         return 0
#     count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     cap.release()
#     return count


# # ── Temp file cleanup ──────────────────────────────────────────────────────────

# def _cleanup_temps(paths: List[str]) -> None:
#     for p in paths:
#         try:
#             if os.path.isfile(p):
#                 os.remove(p)
#         except OSError:
#             pass


# # # # # from __future__ import annotations

# # # # # """
# # # # # api.py — Loco Pilot Distraction Detection  (DB-driven batch runner)
# # # # # ====================================================================

# # # # # Videos are NO LONGER uploaded through this API.
# # # # # Instead, this service polls the database for video_files rows where
# # # # # process_flag = 'N', downloads each video from S3, runs the detection
# # # # # pipeline, and marks the row done (process_flag = 'Y').

# # # # # Flag lifecycle
# # # # # --------------
# # # # #   N  →  pending (set by the frontend / ingestion service when uploading)
# # # # #   I  →  in-progress (set here, immediately before the pipeline starts)
# # # # #   Y  →  done (set here, only on successful pipeline completion)

# # # # # If the pipeline crashes the flag stays at 'I' so operators can see
# # # # # exactly which video failed — it is NOT silently reset to 'N'.

# # # # # Endpoints
# # # # # ---------
# # # # #   GET  /              — health / welcome
# # # # #   GET  /health        — {"status": "ok"}
# # # # #   POST /trigger       — kick off a DB-scan + batch run immediately
# # # # #                         (returns job_id; poll /status/<job_id>)
# # # # #   GET  /status/<id>   — queued | processing | done | failed
# # # # #   GET  /result/<id>   — final JSON report (consumed once; deleted from memory)
# # # # # """

# # # # # import json
# # # # # import os
# # # # # import tempfile
# # # # # import traceback
# # # # # import uuid
# # # # # from concurrent.futures import ThreadPoolExecutor
# # # # # from itertools import groupby
# # # # # from typing import Any, Dict, List, Optional

# # # # # from fastapi import BackgroundTasks, FastAPI, HTTPException
# # # # # from fastapi.middleware.cors import CORSMiddleware
# # # # # from fastapi.responses import JSONResponse

# # # # # from main import GadgetDetectionPipeline
# # # # # from utils.db_s3_uploader import (
# # # # #     download_video_from_s3,
# # # # #     get_pending_videos,
# # # # #     set_process_flag,
# # # # # )

# # # # # # ── App setup ──────────────────────────────────────────────────────────────────

# # # # # app = FastAPI(
# # # # #     title   = "Loco Pilot Distraction Detection API",
# # # # #     version = "3.0.0",
# # # # # )

# # # # # app.add_middleware(
# # # # #     CORSMiddleware,
# # # # #     allow_origins     = ["*"],
# # # # #     allow_credentials = True,
# # # # #     allow_methods     = ["*"],
# # # # #     allow_headers     = ["*"],
# # # # # )

# # # # # # In-memory job registry  {job_id: {"status": str, "result": dict|None, "error": str|None}}
# # # # # _jobs:    Dict[str, Dict[str, Any]] = {}
# # # # # _executor = ThreadPoolExecutor(max_workers=1)  # one batch at a time


# # # # # # ── Routes ─────────────────────────────────────────────────────────────────────

# # # # # @app.get("/")
# # # # # def root() -> dict:
# # # # #     return {
# # # # #         "status":  "success",
# # # # #         "message": "Loco Pilot Distraction Detection API is running",
# # # # #         "health":  "/health",
# # # # #         "docs":    "/docs",
# # # # #         "trigger": "POST /trigger  — start a DB-scan + batch run",
# # # # #     }


# # # # # @app.get("/health", tags=["status"])
# # # # # def health() -> dict:
# # # # #     return {"status": "ok"}


# # # # # # ── Batch trigger ──────────────────────────────────────────────────────────────

# # # # # @app.post("/trigger", tags=["batch"])
# # # # # async def trigger_batch(background_tasks: BackgroundTasks) -> JSONResponse:
# # # # #     """
# # # # #     Scan the DB for all pending videos (process_flag='N') and process them
# # # # #     as a batch.  Returns immediately with a job_id; poll /status/<job_id>.
# # # # #     """
# # # # #     job_id = str(uuid.uuid4())
# # # # #     _jobs[job_id] = {"status": "queued", "result": None, "error": None}
# # # # #     _executor.submit(_run_batch, job_id)
# # # # #     return JSONResponse(
# # # # #         status_code = 202,
# # # # #         content = {
# # # # #             "job_id":  job_id,
# # # # #             "status":  "queued",
# # # # #             "message": f"Batch job accepted. Poll GET /status/{job_id} for progress.",
# # # # #         },
# # # # #     )


# # # # # @app.get("/status/{job_id}", tags=["batch"])
# # # # # def job_status(job_id: str) -> JSONResponse:
# # # # #     job = _jobs.get(job_id)
# # # # #     if job is None:
# # # # #         raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
# # # # #     resp: Dict[str, Any] = {"job_id": job_id, "status": job["status"]}
# # # # #     if job["status"] == "failed":
# # # # #         resp["error"] = job["error"]
# # # # #     return JSONResponse(content=resp)


# # # # # @app.get("/result/{job_id}", tags=["batch"])
# # # # # def job_result(job_id: str) -> JSONResponse:
# # # # #     job = _jobs.get(job_id)
# # # # #     if job is None:
# # # # #         raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
# # # # #     if job["status"] == "failed":
# # # # #         raise HTTPException(status_code=500, detail=job["error"])
# # # # #     if job["status"] in ("queued", "processing"):
# # # # #         raise HTTPException(
# # # # #             status_code=409,
# # # # #             detail=f"Job is still '{job['status']}'. Try again later.",
# # # # #         )
# # # # #     result = _jobs.pop(job_id)["result"]
# # # # #     return JSONResponse(content=result)


# # # # # # ── Batch worker ───────────────────────────────────────────────────────────────

# # # # # def _run_batch(job_id: str) -> None:
# # # # #     """
# # # # #     Entry point executed in the thread pool.

# # # # #     1. Query DB for all rows with process_flag = 'N'.
# # # # #     2. Group them by (train_detail_id, folder_name) — each group is one
# # # # #        logical analysis (the videos in a folder are one continuous recording).
# # # # #     3. For every group:
# # # # #          a. Mark every row as 'I' (in-progress).
# # # # #          b. Download each video from S3 to a temp file.
# # # # #          c. Run GadgetDetectionPipeline over every video in sequence.
# # # # #          d. On success: mark every row as 'Y' and collect the report.
# # # # #          e. On failure: leave rows at 'I' (flag stays for operator inspection).
# # # # #     4. Collect per-group reports and write them to the job registry.
# # # # #     """
# # # # #     _jobs[job_id]["status"] = "processing"
# # # # #     all_reports: List[Dict[str, Any]] = []

# # # # #     try:
# # # # #         pending = get_pending_videos()

# # # # #         if not pending:
# # # # #             print("[Batch] No pending videos found.")
# # # # #             _jobs[job_id]["status"] = "done"
# # # # #             _jobs[job_id]["result"] = {"message": "No pending videos.", "reports": []}
# # # # #             return

# # # # #         # Group by (train_detail_id, folder_name) — same order as the DB query
# # # # #         def _group_key(row: Dict[str, Any]):
# # # # #             return (row["train_detail_id"], row["folder_name"])

# # # # #         groups = [
# # # # #             (key, list(rows))
# # # # #             for key, rows in groupby(pending, key=_group_key)
# # # # #         ]
# # # # #         print(f"[Batch] {len(pending)} video(s) across {len(groups)} folder group(s).")

# # # # #         for (train_detail_id, folder_name), rows in groups:
# # # # #             report = _process_folder_group(
# # # # #                 train_detail_id = train_detail_id,
# # # # #                 folder_name     = folder_name,
# # # # #                 rows            = rows,
# # # # #             )
# # # # #             if report:
# # # # #                 all_reports.append(report)

# # # # #         _jobs[job_id]["status"] = "done"
# # # # #         _jobs[job_id]["result"] = {
# # # # #             "message": f"Processed {len(all_reports)} folder group(s).",
# # # # #             "reports": all_reports,
# # # # #         }

# # # # #     except Exception as exc:
# # # # #         _jobs[job_id]["status"] = "failed"
# # # # #         _jobs[job_id]["error"]  = f"{exc}\n{traceback.format_exc()}"
# # # # #         print(f"[Batch] Fatal error: {exc}")


# # # # # def _process_folder_group(
# # # # #     train_detail_id: int,
# # # # #     folder_name:     str,
# # # # #     rows:            List[Dict[str, Any]],
# # # # # ) -> Optional[Dict[str, Any]]:
# # # # #     """
# # # # #     Process one folder group (all videos that belong to a single analysis).

# # # # #     Rows are already ordered by seq_no (guaranteed by the DB query).
# # # # #     Each video is processed in sequence so frame offsets accumulate
# # # # #     correctly across files.

# # # # #     Returns the JSON report dict on success, None on failure.
# # # # #     """
# # # # #     print(
# # # # #         f"\n[Batch] ── Folder group: train={train_detail_id}  "
# # # # #         f"folder='{folder_name}'  ({len(rows)} video(s)) ──"
# # # # #     )

# # # # #     # Use folder_name as the analysis_id (unique per recording session)
# # # # #     analysis_id = folder_name

# # # # #     # ── Step 1: mark all rows as in-progress ─────────────────────────────────
# # # # #     for row in rows:
# # # # #         try:
# # # # #             set_process_flag(row["id"], "I")
# # # # #         except Exception as exc:
# # # # #             print(f"[Batch] Could not mark row {row['id']} as 'I': {exc}")
# # # # #             # Non-fatal — continue; the row stays 'N' which is safer than crashing

# # # # #     # ── Step 2: download videos to temp files ─────────────────────────────────
# # # # #     tmp_paths: List[str] = []
# # # # #     try:
# # # # #         for row in rows:
# # # # #             suffix    = os.path.splitext(row["filename"] or "video.mp4")[1] or ".mp4"
# # # # #             tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
# # # # #             os.close(tmp_fd)
# # # # #             download_video_from_s3(row["s3_video_path"], tmp_path)
# # # # #             tmp_paths.append(tmp_path)
# # # # #     except Exception as exc:
# # # # #         print(f"[Batch] Download failed for folder '{folder_name}': {exc}")
# # # # #         _cleanup_temps(tmp_paths)
# # # # #         return None

# # # # #     # ── Step 3: run the pipeline over each video in sequence ──────────────────
# # # # #     report_path: str = ""
# # # # #     try:
# # # # #         # The pipeline processes one video at a time but shares the same
# # # # #         # ViolationStore (via analysis_id = folder_name) so violations from
# # # # #         # every video land in the same report.
# # # # #         for idx, (row, tmp_path) in enumerate(zip(rows, tmp_paths)):
# # # # #             print(
# # # # #                 f"[Batch]   [{idx+1}/{len(rows)}] Processing  "
# # # # #                 f"{row['filename']}  (seq={row['seq_no']})"
# # # # #             )
# # # # #             pipeline = GadgetDetectionPipeline(
# # # # #                 source          = tmp_path,
# # # # #                 analysis_id     = analysis_id,
# # # # #                 train_detail_id = train_detail_id,
# # # # #                 save            = False,
# # # # #                 display         = False,
# # # # #             )
# # # # #             # run() returns the path to analysis_report.json
# # # # #             report_path = pipeline.run()

# # # # #         # ── Step 4: mark all rows as done ────────────────────────────────────
# # # # #         for row in rows:
# # # # #             try:
# # # # #                 set_process_flag(row["id"], "Y")
# # # # #             except Exception as exc:
# # # # #                 print(f"[Batch] Could not mark row {row['id']} as 'Y': {exc}")

# # # # #         # ── Step 5: read and return the report ───────────────────────────────
# # # # #         if report_path and os.path.isfile(report_path):
# # # # #             with open(report_path, encoding="utf-8") as f:
# # # # #                 return json.load(f)
# # # # #         else:
# # # # #             print(f"[Batch] Report file missing for folder '{folder_name}'")
# # # # #             return None

# # # # #     except Exception as exc:
# # # # #         # Leave flags at 'I' so the operator can see which group failed
# # # # #         print(
# # # # #             f"[Batch] Pipeline error for folder '{folder_name}': {exc}\n"
# # # # #             + traceback.format_exc()
# # # # #         )
# # # # #         return None

# # # # #     finally:
# # # # #         _cleanup_temps(tmp_paths)


# # # # # def _cleanup_temps(paths: List[str]) -> None:
# # # # #     for p in paths:
# # # # #         try:
# # # # #             if os.path.isfile(p):
# # # # #                 os.remove(p)
# # # # #         except OSError:
# # # # #             pass



# # # # from __future__ import annotations

# # # # """
# # # # api.py — Loco Pilot Distraction Detection  (DB-driven batch runner)
# # # # ====================================================================

# # # # Videos are NOT uploaded through this API.
# # # # This service polls the database for video_files rows where
# # # # process_flag = 'N', downloads each video from S3, runs the detection
# # # # pipeline synchronously, and returns the final result directly.

# # # # Flag lifecycle
# # # # --------------
# # # #   N  →  pending   (set by the frontend when a video is uploaded to S3)
# # # #   I  →  in-progress (set here before the pipeline starts)
# # # #   Y  →  done      (set here only on successful pipeline completion)

# # # # If the pipeline crashes the flag stays at 'I' so operators can see
# # # # exactly which video failed.

# # # # Endpoints
# # # # ---------
# # # #   GET  /         — health / welcome
# # # #   GET  /health   — {"status": "ok"}
# # # #   POST /trigger  — scan DB → download from S3 → run pipeline → return result
# # # # """

# # # # import json
# # # # import os
# # # # import tempfile
# # # # import traceback
# # # # from itertools import groupby
# # # # from typing import Any, Dict, List, Optional

# # # # from fastapi import FastAPI, HTTPException
# # # # from fastapi.middleware.cors import CORSMiddleware
# # # # from fastapi.responses import JSONResponse

# # # # from main import GadgetDetectionPipeline
# # # # from utils.db_s3_uploader import (
# # # #     download_video_from_s3,
# # # #     get_pending_videos,
# # # #     set_process_flag,
# # # # )

# # # # # ── App setup ──────────────────────────────────────────────────────────────────

# # # # app = FastAPI(
# # # #     title   = "Loco Pilot Distraction Detection API",
# # # #     version = "3.0.0",
# # # # )

# # # # app.add_middleware(
# # # #     CORSMiddleware,
# # # #     allow_origins     = ["*"],
# # # #     allow_credentials = True,
# # # #     allow_methods     = ["*"],
# # # #     allow_headers     = ["*"],
# # # # )


# # # # # ── Routes ─────────────────────────────────────────────────────────────────────

# # # # @app.get("/")
# # # # def root() -> dict:
# # # #     return {
# # # #         "status":  "success",
# # # #         "message": "Loco Pilot Distraction Detection API is running",
# # # #         "health":  "/health",
# # # #         "docs":    "/docs",
# # # #         "trigger": "POST /trigger  — scan DB, process all pending videos, return results",
# # # #     }


# # # # @app.get("/health", tags=["status"])
# # # # def health() -> dict:
# # # #     return {"status": "ok"}


# # # # # ── Main trigger ───────────────────────────────────────────────────────────────

# # # # @app.post("/trigger", tags=["batch"])
# # # # def trigger_batch() -> JSONResponse:
# # # #     """
# # # #     Scan DB for all rows with process_flag = 'N'.
# # # #     Group by (train_detail_id, folder_name) — each group is one logical
# # # #     analysis (a folder of sequential videos from one recording session).
# # # #     Process every group in sequence, then return all reports together.

# # # #     Flag lifecycle per video row:
# # # #       N  →  I  (before pipeline starts)
# # # #       I  →  Y  (after pipeline succeeds)
# # # #       stays I   (if pipeline fails — visible to operators)
# # # #     """
# # # #     try:
# # # #         pending = get_pending_videos()
# # # #     except Exception as exc:
# # # #         raise HTTPException(
# # # #             status_code=500,
# # # #             detail=f"Failed to query pending videos from DB: {exc}",
# # # #         )

# # # #     if not pending:
# # # #         return JSONResponse(content={
# # # #             "status":  "ok",
# # # #             "message": "No pending videos found (process_flag = 'N').",
# # # #             "reports": [],
# # # #         })

# # # #     def _group_key(row: Dict[str, Any]):
# # # #         return (row["train_detail_id"], row["folder_name"])

# # # #     groups = [
# # # #         (key, list(rows))
# # # #         for key, rows in groupby(pending, key=_group_key)
# # # #     ]
# # # #     print(f"[Batch] {len(pending)} video(s) across {len(groups)} folder group(s).")

# # # #     all_reports: List[Dict[str, Any]] = []
# # # #     errors:      List[str]            = []

# # # #     for (train_detail_id, folder_name), rows in groups:
# # # #         report, error = _process_folder_group(
# # # #             train_detail_id = train_detail_id,
# # # #             folder_name     = folder_name,
# # # #             rows            = rows,
# # # #         )
# # # #         if report is not None:
# # # #             all_reports.append(report)
# # # #         if error:
# # # #             errors.append(error)

# # # #     return JSONResponse(content={
# # # #         "status":         "ok" if not errors else "partial",
# # # #         "groups_total":   len(groups),
# # # #         "groups_success": len(all_reports),
# # # #         "groups_failed":  len(errors),
# # # #         "errors":         errors,
# # # #         "reports":        all_reports,
# # # #     })


# # # # # ── Per-folder-group processor ─────────────────────────────────────────────────

# # # # def _process_folder_group(
# # # #     train_detail_id: int,
# # # #     folder_name:     str,
# # # #     rows:            List[Dict[str, Any]],
# # # # ) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
# # # #     """
# # # #     Process one folder group — all videos that share the same
# # # #     (train_detail_id, folder_name), ordered by seq_no.

# # # #     Returns (report_dict, None) on success, (None, error_str) on failure.
# # # #     """
# # # #     print(
# # # #         f"\n[Batch] ── Folder: train={train_detail_id}  "
# # # #         f"folder='{folder_name}'  ({len(rows)} video(s)) ──"
# # # #     )

# # # #     # folder_name is unique per recording session → use as analysis_id
# # # #     analysis_id = folder_name

# # # #     # Step 1 — mark all rows in-progress
# # # #     for row in rows:
# # # #         try:
# # # #             set_process_flag(row["id"], "I")
# # # #         except Exception as exc:
# # # #             print(f"[Batch] Could not mark row {row['id']} as 'I': {exc}")

# # # #     # Step 2 — download every video from S3 to a temp file
# # # #     tmp_paths: List[str] = []
# # # #     try:
# # # #         for row in rows:
# # # #             suffix = os.path.splitext(row["filename"] or "video.mp4")[1] or ".mp4"
# # # #             tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
# # # #             os.close(tmp_fd)
# # # #             download_video_from_s3(row["s3_video_path"], tmp_path)
# # # #             tmp_paths.append(tmp_path)
# # # #     except Exception as exc:
# # # #         err = f"folder='{folder_name}' download failed: {exc}"
# # # #         print(f"[Batch] {err}")
# # # #         _cleanup_temps(tmp_paths)
# # # #         return None, err

# # # #     # Step 3 — run the pipeline over each video in seq_no order
# # # #     report_path = ""
# # # #     try:
# # # #         for idx, (row, tmp_path) in enumerate(zip(rows, tmp_paths)):
# # # #             print(
# # # #                 f"[Batch]   [{idx + 1}/{len(rows)}]  "
# # # #                 f"{row['filename']}  (seq={row['seq_no']})"
# # # #             )
# # # #             pipeline = GadgetDetectionPipeline(
# # # #                 source          = tmp_path,
# # # #                 analysis_id     = analysis_id,
# # # #                 train_detail_id = train_detail_id,
# # # #                 save            = False,
# # # #                 display         = False,
# # # #             )
# # # #             report_path = pipeline.run()

# # # #         # Step 4 — mark all rows done
# # # #         for row in rows:
# # # #             try:
# # # #                 set_process_flag(row["id"], "Y")
# # # #             except Exception as exc:
# # # #                 print(f"[Batch] Could not mark row {row['id']} as 'Y': {exc}")

# # # #         # Step 5 — read and return the report
# # # #         if report_path and os.path.isfile(report_path):
# # # #             with open(report_path, encoding="utf-8") as f:
# # # #                 return json.load(f), None
# # # #         else:
# # # #             err = f"folder='{folder_name}' pipeline returned no report file"
# # # #             print(f"[Batch] {err}")
# # # #             return None, err

# # # #     except Exception as exc:
# # # #         # flags stay at 'I' — intentional, so operator can inspect
# # # #         err = (
# # # #             f"folder='{folder_name}' pipeline error: {exc}\n"
# # # #             + traceback.format_exc()
# # # #         )
# # # #         print(f"[Batch] {err}")
# # # #         return None, err

# # # #     finally:
# # # #         _cleanup_temps(tmp_paths)


# # # # # ── Temp file cleanup ──────────────────────────────────────────────────────────

# # # # def _cleanup_temps(paths: List[str]) -> None:
# # # #     for p in paths:
# # # #         try:
# # # #             if os.path.isfile(p):
# # # #                 os.remove(p)
# # # #         except OSError:
# # # #             pass


# # # from __future__ import annotations

# # # """
# # # api.py — Loco Pilot Distraction Detection  (DB-driven batch runner)
# # # ====================================================================

# # # Videos are NOT uploaded through this API.
# # # This service polls the database for video_files rows where
# # # process_flag = 'N', downloads each video from S3, runs the detection
# # # pipeline synchronously, and returns the final result directly.

# # # Flag lifecycle
# # # --------------
# # #   N  →  pending   (set by the frontend when a video is uploaded to S3)
# # #   I  →  in-progress (set here before the pipeline starts)
# # #   Y  →  done      (set here only on successful pipeline completion)

# # # If the pipeline crashes the flag stays at 'I' so operators can see
# # # exactly which video failed.

# # # Endpoints
# # # ---------
# # #   GET  /         — health / welcome
# # #   GET  /health   — {"status": "ok"}
# # #   POST /trigger  — scan DB → download from S3 → run pipeline → return result
# # # """

# # # import json
# # # import os
# # # import tempfile
# # # import traceback
# # # import uuid
# # # from itertools import groupby
# # # from typing import Any, Dict, List, Optional, Tuple

# # # from fastapi import FastAPI, HTTPException
# # # from fastapi.middleware.cors import CORSMiddleware
# # # from fastapi.responses import JSONResponse

# # # from main import GadgetDetectionPipeline
# # # from utils.db_s3_uploader import (
# # #     download_video_from_s3,
# # #     get_pending_videos,
# # #     set_process_flag,
# # # )

# # # # ── App setup ──────────────────────────────────────────────────────────────────

# # # app = FastAPI(
# # #     title   = "Loco Pilot Distraction Detection API",
# # #     version = "3.0.0",
# # # )

# # # app.add_middleware(
# # #     CORSMiddleware,
# # #     allow_origins     = ["*"],
# # #     allow_credentials = True,
# # #     allow_methods     = ["*"],
# # #     allow_headers     = ["*"],
# # # )


# # # # ── Routes ─────────────────────────────────────────────────────────────────────

# # # @app.get("/")
# # # def root() -> dict:
# # #     return {
# # #         "status":  "success",
# # #         "message": "Loco Pilot Distraction Detection API is running",
# # #         "health":  "/health",
# # #         "docs":    "/docs",
# # #         "trigger": "POST /trigger  — scan DB, process all pending videos, return results",
# # #     }


# # # @app.get("/health", tags=["status"])
# # # def health() -> dict:
# # #     return {"status": "ok"}


# # # # ── Main trigger ───────────────────────────────────────────────────────────────

# # # @app.post("/trigger", tags=["batch"])
# # # def trigger_batch() -> JSONResponse:
# # #     """
# # #     Scan DB for all rows with process_flag = 'N'.
# # #     Group by (train_detail_id, folder_name) — each group is one logical
# # #     analysis (a folder of sequential videos from one recording session).
# # #     Process every group in sequence, then return all results in the
# # #     target batch envelope format.

# # #     Response shape:
# # #     {
# # #       "batch_id":          "<hex>",
# # #       "total_videos":      N,
# # #       "completed":         N,
# # #       "failed":            N,
# # #       "folders_processed": N,
# # #       "folders": [
# # #         {
# # #           "train_detail_id":   22803,
# # #           "folder_name":       "22803-05-06-2026",
# # #           "videos_in_folder":  8,
# # #           "report": { ... }   ← full analysis_report.json content
# # #         },
# # #         ...
# # #       ]
# # #     }
# # #     """
# # #     try:
# # #         pending = get_pending_videos()
# # #     except Exception as exc:
# # #         raise HTTPException(
# # #             status_code = 500,
# # #             detail      = f"Failed to query pending videos from DB: {exc}",
# # #         )

# # #     if not pending:
# # #         return JSONResponse(content={
# # #             "batch_id":          uuid.uuid4().hex[:12],
# # #             "total_videos":      0,
# # #             "completed":         0,
# # #             "failed":            0,
# # #             "folders_processed": 0,
# # #             "folders":           [],
# # #             "message":           "No pending videos found (process_flag = 'N').",
# # #         })

# # #     batch_id = uuid.uuid4().hex[:12]

# # #     def _group_key(row: Dict[str, Any]):
# # #         return (row["train_detail_id"], row["folder_name"])

# # #     groups = [
# # #         (key, list(grp_rows))
# # #         for key, grp_rows in groupby(pending, key=_group_key)
# # #     ]
# # #     print(f"[Batch:{batch_id}] {len(pending)} video(s) across {len(groups)} folder group(s).")

# # #     total_videos = len(pending)
# # #     completed    = 0
# # #     failed       = 0
# # #     folders_out: List[Dict[str, Any]] = []

# # #     for (train_detail_id, folder_name), rows in groups:
# # #         report, error, n_completed, n_failed = _process_folder_group(
# # #             train_detail_id = train_detail_id,
# # #             folder_name     = folder_name,
# # #             rows            = rows,
# # #         )
# # #         completed += n_completed
# # #         failed    += n_failed

# # #         folder_entry: Dict[str, Any] = {
# # #             "train_detail_id":  train_detail_id,
# # #             "folder_name":      folder_name,
# # #             "videos_in_folder": len(rows),
# # #         }
# # #         if report is not None:
# # #             folder_entry["report"] = report
# # #         if error:
# # #             folder_entry["error"] = error

# # #         folders_out.append(folder_entry)

# # #     return JSONResponse(content={
# # #         "batch_id":          batch_id,
# # #         "total_videos":      total_videos,
# # #         "completed":         completed,
# # #         "failed":            failed,
# # #         "folders_processed": len(folders_out),
# # #         "folders":           folders_out,
# # #     })


# # # # ── Per-folder-group processor ─────────────────────────────────────────────────

# # # def _process_folder_group(
# # #     train_detail_id: int,
# # #     folder_name:     str,
# # #     rows:            List[Dict[str, Any]],
# # # ) -> Tuple[Optional[Dict[str, Any]], Optional[str], int, int]:
# # #     """
# # #     Process one folder group — all videos that share the same
# # #     (train_detail_id, folder_name), ordered by seq_no.

# # #     Returns (report_dict, error_str, n_completed, n_failed).
# # #     """
# # #     n_videos = len(rows)
# # #     print(
# # #         f"\n[Batch] ── Folder: train={train_detail_id}  "
# # #         f"folder='{folder_name}'  ({n_videos} video(s)) ──"
# # #     )

# # #     # folder_name is unique per recording session → use as analysis_id
# # #     analysis_id = folder_name

# # #     # Step 1 — mark all rows in-progress
# # #     for row in rows:
# # #         try:
# # #             set_process_flag(row["id"], "I")
# # #         except Exception as exc:
# # #             print(f"[Batch] Could not mark row {row['id']} as 'I': {exc}")

# # #     # Step 2 — download every video from S3 to a temp file
# # #     tmp_paths: List[str] = []
# # #     try:
# # #         for row in rows:
# # #             suffix = os.path.splitext(row["filename"] or "video.mp4")[1] or ".mp4"
# # #             tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
# # #             os.close(tmp_fd)
# # #             download_video_from_s3(row["s3_video_path"], tmp_path)
# # #             tmp_paths.append(tmp_path)
# # #     except Exception as exc:
# # #         err = f"folder='{folder_name}' download failed: {exc}"
# # #         print(f"[Batch] {err}")
# # #         _cleanup_temps(tmp_paths)
# # #         return None, err, 0, n_videos

# # #     # Step 3 — run the pipeline over each video in seq_no order
# # #     report_path = ""
# # #     try:
# # #         for idx, (row, tmp_path) in enumerate(zip(rows, tmp_paths)):
# # #             print(
# # #                 f"[Batch]   [{idx + 1}/{n_videos}]  "
# # #                 f"{row['filename']}  (seq={row['seq_no']})"
# # #             )
# # #             pipeline = GadgetDetectionPipeline(
# # #                 source          = tmp_path,
# # #                 analysis_id     = analysis_id,
# # #                 train_detail_id = train_detail_id,
# # #                 save            = False,
# # #                 display         = False,
# # #             )
# # #             report_path = pipeline.run()

# # #         # Step 4 — mark all rows done
# # #         for row in rows:
# # #             try:
# # #                 set_process_flag(row["id"], "Y")
# # #             except Exception as exc:
# # #                 print(f"[Batch] Could not mark row {row['id']} as 'Y': {exc}")

# # #         # Step 5 — read and return the report
# # #         if report_path and os.path.isfile(report_path):
# # #             with open(report_path, encoding="utf-8") as f:
# # #                 return json.load(f), None, n_videos, 0
# # #         else:
# # #             err = f"folder='{folder_name}' pipeline returned no report file"
# # #             print(f"[Batch] {err}")
# # #             return None, err, 0, n_videos

# # #     except Exception as exc:
# # #         # flags stay at 'I' — intentional, so operator can inspect
# # #         err = (
# # #             f"folder='{folder_name}' pipeline error: {exc}\n"
# # #             + traceback.format_exc()
# # #         )
# # #         print(f"[Batch] {err}")
# # #         return None, err, 0, n_videos

# # #     finally:
# # #         _cleanup_temps(tmp_paths)


# # # # ── Temp file cleanup ──────────────────────────────────────────────────────────

# # # def _cleanup_temps(paths: List[str]) -> None:
# # #     for p in paths:
# # #         try:
# # #             if os.path.isfile(p):
# # #                 os.remove(p)
# # #         except OSError:
# # #             pass

# # from __future__ import annotations

# # """
# # api.py — Loco Pilot Distraction Detection  (DB-driven batch runner)
# # ====================================================================

# # Videos are NOT uploaded through this API.
# # This service polls the database for video_files rows where
# # process_flag = 'N', downloads each video from S3, runs the detection
# # pipeline synchronously, and returns the final result directly.

# # Flag lifecycle
# # --------------
# #   N  →  pending   (set by the frontend when a video is uploaded to S3)
# #   I  →  in-progress (set here before the pipeline starts)
# #   Y  →  done      (set here only on successful pipeline completion)

# # If the pipeline crashes the flag stays at 'I' so operators can see
# # exactly which video failed.

# # Endpoints
# # ---------
# #   GET  /         — health / welcome
# #   GET  /health   — {"status": "ok"}
# #   POST /trigger  — scan DB → download from S3 → run pipeline → return result

# # Response shape
# # --------------
# # {
# #   "batch_id":          "444e4af5ff5f",
# #   "total_videos":      2,
# #   "completed":         2,
# #   "failed":            0,
# #   "folders_processed": 1,
# #   "folders": [
# #     {
# #       "train_detail_id":  22801,
# #       "folder_name":      "22801-05-06-2026",
# #       "videos_in_folder": 2,
# #       "report": {
# #         "analysis_id":     "22801-05-06-2026",
# #         "train_detail_id": 22801,
# #         "processing_time": 108.124,
# #         "video_info":      [ ... ],   ← list with one entry per video
# #         "violations":      [ ... ]    ← all violations across ALL videos
# #       }
# #     }
# #   ]
# # }

# # Timestamp logic
# # ---------------
# # For each folder group, videos are processed in seq_no order.
# # A running time_offset and frame_offset accumulate as each video finishes.

# #   global_timestamp  = local_video_time  + time_offset
# #   global_frame      = local_frame_index + frame_offset

# # In the report:
# #   "timestamp"                  = global_timestamp  (HH:MM:SS into the full recording)
# #   "original_video_timestamp"   = "<db_filename> <local_time>"  (time within that file)
# # """

# # import json
# # import os
# # import tempfile
# # import traceback
# # import uuid
# # from itertools import groupby
# # from typing import Any, Dict, List, Optional, Tuple

# # import cv2

# # from fastapi import FastAPI, HTTPException
# # from fastapi.middleware.cors import CORSMiddleware
# # from fastapi.responses import JSONResponse

# # from main import GadgetDetectionPipeline
# # from utils.violation_store import ViolationStore
# # from utils.db_s3_uploader import (
# #     download_video_from_s3,
# #     get_pending_videos,
# #     set_process_flag,
# # )

# # # ── App setup ──────────────────────────────────────────────────────────────────

# # app = FastAPI(
# #     title   = "Loco Pilot Distraction Detection API",
# #     version = "3.0.0",
# # )

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins     = ["*"],
# #     allow_credentials = True,
# #     allow_methods     = ["*"],
# #     allow_headers     = ["*"],
# # )


# # # ── Routes ─────────────────────────────────────────────────────────────────────

# # @app.get("/")
# # def root() -> dict:
# #     return {
# #         "status":  "success",
# #         "message": "Loco Pilot Distraction Detection API is running",
# #         "health":  "/health",
# #         "docs":    "/docs",
# #         "trigger": "POST /trigger  — scan DB, process all pending videos, return results",
# #     }


# # @app.get("/health", tags=["status"])
# # def health() -> dict:
# #     return {"status": "ok"}


# # # ── Main trigger ───────────────────────────────────────────────────────────────

# # @app.post("/trigger", tags=["batch"])
# # def trigger_batch() -> JSONResponse:
# #     """
# #     Scan DB for all rows with process_flag = 'N'.
# #     Group by (train_detail_id, folder_name).
# #     Process every group in sequence, return the batch envelope.
# #     """
# #     try:
# #         pending = get_pending_videos()
# #     except Exception as exc:
# #         raise HTTPException(
# #             status_code = 500,
# #             detail      = f"Failed to query pending videos from DB: {exc}",
# #         )

# #     batch_id = uuid.uuid4().hex[:12]

# #     if not pending:
# #         return JSONResponse(content={
# #             "batch_id":          batch_id,
# #             "total_videos":      0,
# #             "completed":         0,
# #             "failed":            0,
# #             "folders_processed": 0,
# #             "folders":           [],
# #             "message":           "No pending videos found (process_flag = 'N').",
# #         })

# #     def _group_key(row: Dict[str, Any]):
# #         return (row["train_detail_id"], row["folder_name"])

# #     groups = [
# #         (key, list(grp_rows))
# #         for key, grp_rows in groupby(pending, key=_group_key)
# #     ]
# #     print(f"[Batch:{batch_id}] {len(pending)} video(s) across {len(groups)} folder group(s).")

# #     total_videos = len(pending)
# #     completed    = 0
# #     failed       = 0
# #     folders_out: List[Dict[str, Any]] = []

# #     for (train_detail_id, folder_name), rows in groups:
# #         report, error, n_ok, n_fail = _process_folder_group(
# #             train_detail_id = train_detail_id,
# #             folder_name     = folder_name,
# #             rows            = rows,
# #         )
# #         completed += n_ok
# #         failed    += n_fail

# #         folder_entry: Dict[str, Any] = {
# #             "train_detail_id":  train_detail_id,
# #             "folder_name":      folder_name,
# #             "videos_in_folder": len(rows),
# #         }
# #         if report is not None:
# #             folder_entry["report"] = report
# #         if error:
# #             folder_entry["error"] = error

# #         folders_out.append(folder_entry)

# #     return JSONResponse(content={
# #         "batch_id":          batch_id,
# #         "total_videos":      total_videos,
# #         "completed":         completed,
# #         "failed":            failed,
# #         "folders_processed": len(folders_out),
# #         "folders":           folders_out,
# #     })


# # # ── Per-folder-group processor ─────────────────────────────────────────────────

# # def _video_duration_seconds(path: str) -> float:
# #     """Read the duration of a video file using OpenCV."""
# #     cap = cv2.VideoCapture(path)
# #     if not cap.isOpened():
# #         return 0.0
# #     fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
# #     total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# #     cap.release()
# #     return total / fps if fps > 0 and total > 0 else 0.0


# # def _process_folder_group(
# #     train_detail_id: int,
# #     folder_name:     str,
# #     rows:            List[Dict[str, Any]],
# # ) -> Tuple[Optional[Dict[str, Any]], Optional[str], int, int]:
# #     """
# #     Process one folder group — all videos that share the same
# #     (train_detail_id, folder_name), ordered by seq_no.

# #     Key design
# #     ──────────
# #     • One ViolationStore is created for the whole folder and shared
# #       across every pipeline run.  This means violations from all videos
# #       accumulate in a single store.
# #     • time_offset and frame_offset grow after each video so timestamps
# #       are continuous across the whole recording session.
# #     • finalize() is called ONCE after all videos are done.

# #     Returns (report_dict, error_str, n_completed, n_failed).
# #     """
# #     n_videos = len(rows)
# #     print(
# #         f"\n[Batch] ── Folder: train={train_detail_id}  "
# #         f"folder='{folder_name}'  ({n_videos} video(s)) ──"
# #     )

# #     analysis_id = folder_name

# #     # Step 1 — mark all rows in-progress
# #     for row in rows:
# #         try:
# #             set_process_flag(row["id"], "I")
# #         except Exception as exc:
# #             print(f"[Batch] Could not mark row {row['id']} as 'I': {exc}")

# #     # Step 2 — download every video from S3 to a temp file
# #     # We also record the DB filename alongside each temp path.
# #     tmp_entries: List[Tuple[str, str]] = []   # [(tmp_path, db_filename), ...]
# #     try:
# #         for row in rows:
# #             suffix = os.path.splitext(row["filename"] or "video.mp4")[1] or ".mp4"
# #             tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
# #             os.close(tmp_fd)
# #             download_video_from_s3(row["s3_video_path"], tmp_path)
# #             tmp_entries.append((tmp_path, row["filename"] or ""))
# #     except Exception as exc:
# #         err = f"folder='{folder_name}' download failed: {exc}"
# #         print(f"[Batch] {err}")
# #         _cleanup_temps([p for p, _ in tmp_entries])
# #         return None, err, 0, n_videos

# #     # Step 3 — create ONE shared ViolationStore for the entire folder
# #     shared_vstore = ViolationStore(
# #         analysis_id     = analysis_id,
# #         train_detail_id = train_detail_id,
# #         # No video_info here — each pipeline run will call add_video_info()
# #     )

# #     # Step 4 — run the pipeline over each video in seq_no order,
# #     #           accumulating time_offset and frame_offset between videos.
# #     time_offset  = 0.0
# #     frame_offset = 0
# #     total_processing_time = 0.0

# #     try:
# #         for idx, (row, (tmp_path, db_filename)) in enumerate(zip(rows, tmp_entries)):
# #             print(
# #                 f"[Batch]   [{idx + 1}/{n_videos}]  "
# #                 f"{db_filename}  (seq={row['seq_no']})  "
# #                 f"time_offset={time_offset:.2f}s  frame_offset={frame_offset}"
# #             )

# #             import time as _time
# #             t0 = _time.time()

# #             pipeline = GadgetDetectionPipeline(
# #                 source          = tmp_path,
# #                 analysis_id     = analysis_id,
# #                 train_detail_id = train_detail_id,
# #                 save            = False,
# #                 display         = False,
# #                 shared_vstore   = shared_vstore,
# #                 time_offset     = time_offset,
# #                 frame_offset    = frame_offset,
# #                 source_filename = db_filename,
# #             )
# #             pipeline.run()   # returns "" in batch mode; vstore is NOT finalized here

# #             video_duration = _video_duration_seconds(tmp_path)
# #             video_frames   = _get_frame_count(tmp_path)

# #             total_processing_time += _time.time() - t0

# #             # Advance offsets for the next video
# #             time_offset  += video_duration
# #             frame_offset += video_frames

# #         # Step 5 — finalize the shared store ONCE with total processing time
# #         report_path = shared_vstore.finalize(processing_time=round(total_processing_time, 3))

# #         # Step 6 — mark all rows done
# #         for row in rows:
# #             try:
# #                 set_process_flag(row["id"], "Y")
# #             except Exception as exc:
# #                 print(f"[Batch] Could not mark row {row['id']} as 'Y': {exc}")

# #         # Step 7 — read and return the report
# #         if report_path and os.path.isfile(report_path):
# #             with open(report_path, encoding="utf-8") as f:
# #                 return json.load(f), None, n_videos, 0
# #         else:
# #             err = f"folder='{folder_name}' no report file after finalize"
# #             print(f"[Batch] {err}")
# #             return None, err, 0, n_videos

# #     except Exception as exc:
# #         # flags stay at 'I' — intentional so operator can inspect
# #         err = (
# #             f"folder='{folder_name}' pipeline error: {exc}\n"
# #             + traceback.format_exc()
# #         )
# #         print(f"[Batch] {err}")
# #         return None, err, 0, n_videos

# #     finally:
# #         _cleanup_temps([p for p, _ in tmp_entries])


# # def _get_frame_count(path: str) -> int:
# #     cap = cv2.VideoCapture(path)
# #     if not cap.isOpened():
# #         return 0
# #     count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# #     cap.release()
# #     return count


# # # ── Temp file cleanup ──────────────────────────────────────────────────────────

# # def _cleanup_temps(paths: List[str]) -> None:
# #     for p in paths:
# #         try:
# #             if os.path.isfile(p):
# #                 os.remove(p)
# #         except OSError:
# #             pass

# # # # from __future__ import annotations

# # # # """
# # # # api.py — Loco Pilot Distraction Detection  (DB-driven batch runner)
# # # # ====================================================================

# # # # Videos are NO LONGER uploaded through this API.
# # # # Instead, this service polls the database for video_files rows where
# # # # process_flag = 'N', downloads each video from S3, runs the detection
# # # # pipeline, and marks the row done (process_flag = 'Y').

# # # # Flag lifecycle
# # # # --------------
# # # #   N  →  pending (set by the frontend / ingestion service when uploading)
# # # #   I  →  in-progress (set here, immediately before the pipeline starts)
# # # #   Y  →  done (set here, only on successful pipeline completion)

# # # # If the pipeline crashes the flag stays at 'I' so operators can see
# # # # exactly which video failed — it is NOT silently reset to 'N'.

# # # # Endpoints
# # # # ---------
# # # #   GET  /              — health / welcome
# # # #   GET  /health        — {"status": "ok"}
# # # #   POST /trigger       — kick off a DB-scan + batch run immediately
# # # #                         (returns job_id; poll /status/<job_id>)
# # # #   GET  /status/<id>   — queued | processing | done | failed
# # # #   GET  /result/<id>   — final JSON report (consumed once; deleted from memory)
# # # # """

# # # # import json
# # # # import os
# # # # import tempfile
# # # # import traceback
# # # # import uuid
# # # # from concurrent.futures import ThreadPoolExecutor
# # # # from itertools import groupby
# # # # from typing import Any, Dict, List, Optional

# # # # from fastapi import BackgroundTasks, FastAPI, HTTPException
# # # # from fastapi.middleware.cors import CORSMiddleware
# # # # from fastapi.responses import JSONResponse

# # # # from main import GadgetDetectionPipeline
# # # # from utils.db_s3_uploader import (
# # # #     download_video_from_s3,
# # # #     get_pending_videos,
# # # #     set_process_flag,
# # # # )

# # # # # ── App setup ──────────────────────────────────────────────────────────────────

# # # # app = FastAPI(
# # # #     title   = "Loco Pilot Distraction Detection API",
# # # #     version = "3.0.0",
# # # # )

# # # # app.add_middleware(
# # # #     CORSMiddleware,
# # # #     allow_origins     = ["*"],
# # # #     allow_credentials = True,
# # # #     allow_methods     = ["*"],
# # # #     allow_headers     = ["*"],
# # # # )

# # # # # In-memory job registry  {job_id: {"status": str, "result": dict|None, "error": str|None}}
# # # # _jobs:    Dict[str, Dict[str, Any]] = {}
# # # # _executor = ThreadPoolExecutor(max_workers=1)  # one batch at a time


# # # # # ── Routes ─────────────────────────────────────────────────────────────────────

# # # # @app.get("/")
# # # # def root() -> dict:
# # # #     return {
# # # #         "status":  "success",
# # # #         "message": "Loco Pilot Distraction Detection API is running",
# # # #         "health":  "/health",
# # # #         "docs":    "/docs",
# # # #         "trigger": "POST /trigger  — start a DB-scan + batch run",
# # # #     }


# # # # @app.get("/health", tags=["status"])
# # # # def health() -> dict:
# # # #     return {"status": "ok"}


# # # # # ── Batch trigger ──────────────────────────────────────────────────────────────

# # # # @app.post("/trigger", tags=["batch"])
# # # # async def trigger_batch(background_tasks: BackgroundTasks) -> JSONResponse:
# # # #     """
# # # #     Scan the DB for all pending videos (process_flag='N') and process them
# # # #     as a batch.  Returns immediately with a job_id; poll /status/<job_id>.
# # # #     """
# # # #     job_id = str(uuid.uuid4())
# # # #     _jobs[job_id] = {"status": "queued", "result": None, "error": None}
# # # #     _executor.submit(_run_batch, job_id)
# # # #     return JSONResponse(
# # # #         status_code = 202,
# # # #         content = {
# # # #             "job_id":  job_id,
# # # #             "status":  "queued",
# # # #             "message": f"Batch job accepted. Poll GET /status/{job_id} for progress.",
# # # #         },
# # # #     )


# # # # @app.get("/status/{job_id}", tags=["batch"])
# # # # def job_status(job_id: str) -> JSONResponse:
# # # #     job = _jobs.get(job_id)
# # # #     if job is None:
# # # #         raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
# # # #     resp: Dict[str, Any] = {"job_id": job_id, "status": job["status"]}
# # # #     if job["status"] == "failed":
# # # #         resp["error"] = job["error"]
# # # #     return JSONResponse(content=resp)


# # # # @app.get("/result/{job_id}", tags=["batch"])
# # # # def job_result(job_id: str) -> JSONResponse:
# # # #     job = _jobs.get(job_id)
# # # #     if job is None:
# # # #         raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
# # # #     if job["status"] == "failed":
# # # #         raise HTTPException(status_code=500, detail=job["error"])
# # # #     if job["status"] in ("queued", "processing"):
# # # #         raise HTTPException(
# # # #             status_code=409,
# # # #             detail=f"Job is still '{job['status']}'. Try again later.",
# # # #         )
# # # #     result = _jobs.pop(job_id)["result"]
# # # #     return JSONResponse(content=result)


# # # # # ── Batch worker ───────────────────────────────────────────────────────────────

# # # # def _run_batch(job_id: str) -> None:
# # # #     """
# # # #     Entry point executed in the thread pool.

# # # #     1. Query DB for all rows with process_flag = 'N'.
# # # #     2. Group them by (train_detail_id, folder_name) — each group is one
# # # #        logical analysis (the videos in a folder are one continuous recording).
# # # #     3. For every group:
# # # #          a. Mark every row as 'I' (in-progress).
# # # #          b. Download each video from S3 to a temp file.
# # # #          c. Run GadgetDetectionPipeline over every video in sequence.
# # # #          d. On success: mark every row as 'Y' and collect the report.
# # # #          e. On failure: leave rows at 'I' (flag stays for operator inspection).
# # # #     4. Collect per-group reports and write them to the job registry.
# # # #     """
# # # #     _jobs[job_id]["status"] = "processing"
# # # #     all_reports: List[Dict[str, Any]] = []

# # # #     try:
# # # #         pending = get_pending_videos()

# # # #         if not pending:
# # # #             print("[Batch] No pending videos found.")
# # # #             _jobs[job_id]["status"] = "done"
# # # #             _jobs[job_id]["result"] = {"message": "No pending videos.", "reports": []}
# # # #             return

# # # #         # Group by (train_detail_id, folder_name) — same order as the DB query
# # # #         def _group_key(row: Dict[str, Any]):
# # # #             return (row["train_detail_id"], row["folder_name"])

# # # #         groups = [
# # # #             (key, list(rows))
# # # #             for key, rows in groupby(pending, key=_group_key)
# # # #         ]
# # # #         print(f"[Batch] {len(pending)} video(s) across {len(groups)} folder group(s).")

# # # #         for (train_detail_id, folder_name), rows in groups:
# # # #             report = _process_folder_group(
# # # #                 train_detail_id = train_detail_id,
# # # #                 folder_name     = folder_name,
# # # #                 rows            = rows,
# # # #             )
# # # #             if report:
# # # #                 all_reports.append(report)

# # # #         _jobs[job_id]["status"] = "done"
# # # #         _jobs[job_id]["result"] = {
# # # #             "message": f"Processed {len(all_reports)} folder group(s).",
# # # #             "reports": all_reports,
# # # #         }

# # # #     except Exception as exc:
# # # #         _jobs[job_id]["status"] = "failed"
# # # #         _jobs[job_id]["error"]  = f"{exc}\n{traceback.format_exc()}"
# # # #         print(f"[Batch] Fatal error: {exc}")


# # # # def _process_folder_group(
# # # #     train_detail_id: int,
# # # #     folder_name:     str,
# # # #     rows:            List[Dict[str, Any]],
# # # # ) -> Optional[Dict[str, Any]]:
# # # #     """
# # # #     Process one folder group (all videos that belong to a single analysis).

# # # #     Rows are already ordered by seq_no (guaranteed by the DB query).
# # # #     Each video is processed in sequence so frame offsets accumulate
# # # #     correctly across files.

# # # #     Returns the JSON report dict on success, None on failure.
# # # #     """
# # # #     print(
# # # #         f"\n[Batch] ── Folder group: train={train_detail_id}  "
# # # #         f"folder='{folder_name}'  ({len(rows)} video(s)) ──"
# # # #     )

# # # #     # Use folder_name as the analysis_id (unique per recording session)
# # # #     analysis_id = folder_name

# # # #     # ── Step 1: mark all rows as in-progress ─────────────────────────────────
# # # #     for row in rows:
# # # #         try:
# # # #             set_process_flag(row["id"], "I")
# # # #         except Exception as exc:
# # # #             print(f"[Batch] Could not mark row {row['id']} as 'I': {exc}")
# # # #             # Non-fatal — continue; the row stays 'N' which is safer than crashing

# # # #     # ── Step 2: download videos to temp files ─────────────────────────────────
# # # #     tmp_paths: List[str] = []
# # # #     try:
# # # #         for row in rows:
# # # #             suffix    = os.path.splitext(row["filename"] or "video.mp4")[1] or ".mp4"
# # # #             tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
# # # #             os.close(tmp_fd)
# # # #             download_video_from_s3(row["s3_video_path"], tmp_path)
# # # #             tmp_paths.append(tmp_path)
# # # #     except Exception as exc:
# # # #         print(f"[Batch] Download failed for folder '{folder_name}': {exc}")
# # # #         _cleanup_temps(tmp_paths)
# # # #         return None

# # # #     # ── Step 3: run the pipeline over each video in sequence ──────────────────
# # # #     report_path: str = ""
# # # #     try:
# # # #         # The pipeline processes one video at a time but shares the same
# # # #         # ViolationStore (via analysis_id = folder_name) so violations from
# # # #         # every video land in the same report.
# # # #         for idx, (row, tmp_path) in enumerate(zip(rows, tmp_paths)):
# # # #             print(
# # # #                 f"[Batch]   [{idx+1}/{len(rows)}] Processing  "
# # # #                 f"{row['filename']}  (seq={row['seq_no']})"
# # # #             )
# # # #             pipeline = GadgetDetectionPipeline(
# # # #                 source          = tmp_path,
# # # #                 analysis_id     = analysis_id,
# # # #                 train_detail_id = train_detail_id,
# # # #                 save            = False,
# # # #                 display         = False,
# # # #             )
# # # #             # run() returns the path to analysis_report.json
# # # #             report_path = pipeline.run()

# # # #         # ── Step 4: mark all rows as done ────────────────────────────────────
# # # #         for row in rows:
# # # #             try:
# # # #                 set_process_flag(row["id"], "Y")
# # # #             except Exception as exc:
# # # #                 print(f"[Batch] Could not mark row {row['id']} as 'Y': {exc}")

# # # #         # ── Step 5: read and return the report ───────────────────────────────
# # # #         if report_path and os.path.isfile(report_path):
# # # #             with open(report_path, encoding="utf-8") as f:
# # # #                 return json.load(f)
# # # #         else:
# # # #             print(f"[Batch] Report file missing for folder '{folder_name}'")
# # # #             return None

# # # #     except Exception as exc:
# # # #         # Leave flags at 'I' so the operator can see which group failed
# # # #         print(
# # # #             f"[Batch] Pipeline error for folder '{folder_name}': {exc}\n"
# # # #             + traceback.format_exc()
# # # #         )
# # # #         return None

# # # #     finally:
# # # #         _cleanup_temps(tmp_paths)


# # # # def _cleanup_temps(paths: List[str]) -> None:
# # # #     for p in paths:
# # # #         try:
# # # #             if os.path.isfile(p):
# # # #                 os.remove(p)
# # # #         except OSError:
# # # #             pass



# # # from __future__ import annotations

# # # """
# # # api.py — Loco Pilot Distraction Detection  (DB-driven batch runner)
# # # ====================================================================

# # # Videos are NOT uploaded through this API.
# # # This service polls the database for video_files rows where
# # # process_flag = 'N', downloads each video from S3, runs the detection
# # # pipeline synchronously, and returns the final result directly.

# # # Flag lifecycle
# # # --------------
# # #   N  →  pending   (set by the frontend when a video is uploaded to S3)
# # #   I  →  in-progress (set here before the pipeline starts)
# # #   Y  →  done      (set here only on successful pipeline completion)

# # # If the pipeline crashes the flag stays at 'I' so operators can see
# # # exactly which video failed.

# # # Endpoints
# # # ---------
# # #   GET  /         — health / welcome
# # #   GET  /health   — {"status": "ok"}
# # #   POST /trigger  — scan DB → download from S3 → run pipeline → return result
# # # """

# # # import json
# # # import os
# # # import tempfile
# # # import traceback
# # # from itertools import groupby
# # # from typing import Any, Dict, List, Optional

# # # from fastapi import FastAPI, HTTPException
# # # from fastapi.middleware.cors import CORSMiddleware
# # # from fastapi.responses import JSONResponse

# # # from main import GadgetDetectionPipeline
# # # from utils.db_s3_uploader import (
# # #     download_video_from_s3,
# # #     get_pending_videos,
# # #     set_process_flag,
# # # )

# # # # ── App setup ──────────────────────────────────────────────────────────────────

# # # app = FastAPI(
# # #     title   = "Loco Pilot Distraction Detection API",
# # #     version = "3.0.0",
# # # )

# # # app.add_middleware(
# # #     CORSMiddleware,
# # #     allow_origins     = ["*"],
# # #     allow_credentials = True,
# # #     allow_methods     = ["*"],
# # #     allow_headers     = ["*"],
# # # )


# # # # ── Routes ─────────────────────────────────────────────────────────────────────

# # # @app.get("/")
# # # def root() -> dict:
# # #     return {
# # #         "status":  "success",
# # #         "message": "Loco Pilot Distraction Detection API is running",
# # #         "health":  "/health",
# # #         "docs":    "/docs",
# # #         "trigger": "POST /trigger  — scan DB, process all pending videos, return results",
# # #     }


# # # @app.get("/health", tags=["status"])
# # # def health() -> dict:
# # #     return {"status": "ok"}


# # # # ── Main trigger ───────────────────────────────────────────────────────────────

# # # @app.post("/trigger", tags=["batch"])
# # # def trigger_batch() -> JSONResponse:
# # #     """
# # #     Scan DB for all rows with process_flag = 'N'.
# # #     Group by (train_detail_id, folder_name) — each group is one logical
# # #     analysis (a folder of sequential videos from one recording session).
# # #     Process every group in sequence, then return all reports together.

# # #     Flag lifecycle per video row:
# # #       N  →  I  (before pipeline starts)
# # #       I  →  Y  (after pipeline succeeds)
# # #       stays I   (if pipeline fails — visible to operators)
# # #     """
# # #     try:
# # #         pending = get_pending_videos()
# # #     except Exception as exc:
# # #         raise HTTPException(
# # #             status_code=500,
# # #             detail=f"Failed to query pending videos from DB: {exc}",
# # #         )

# # #     if not pending:
# # #         return JSONResponse(content={
# # #             "status":  "ok",
# # #             "message": "No pending videos found (process_flag = 'N').",
# # #             "reports": [],
# # #         })

# # #     def _group_key(row: Dict[str, Any]):
# # #         return (row["train_detail_id"], row["folder_name"])

# # #     groups = [
# # #         (key, list(rows))
# # #         for key, rows in groupby(pending, key=_group_key)
# # #     ]
# # #     print(f"[Batch] {len(pending)} video(s) across {len(groups)} folder group(s).")

# # #     all_reports: List[Dict[str, Any]] = []
# # #     errors:      List[str]            = []

# # #     for (train_detail_id, folder_name), rows in groups:
# # #         report, error = _process_folder_group(
# # #             train_detail_id = train_detail_id,
# # #             folder_name     = folder_name,
# # #             rows            = rows,
# # #         )
# # #         if report is not None:
# # #             all_reports.append(report)
# # #         if error:
# # #             errors.append(error)

# # #     return JSONResponse(content={
# # #         "status":         "ok" if not errors else "partial",
# # #         "groups_total":   len(groups),
# # #         "groups_success": len(all_reports),
# # #         "groups_failed":  len(errors),
# # #         "errors":         errors,
# # #         "reports":        all_reports,
# # #     })


# # # # ── Per-folder-group processor ─────────────────────────────────────────────────

# # # def _process_folder_group(
# # #     train_detail_id: int,
# # #     folder_name:     str,
# # #     rows:            List[Dict[str, Any]],
# # # ) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
# # #     """
# # #     Process one folder group — all videos that share the same
# # #     (train_detail_id, folder_name), ordered by seq_no.

# # #     Returns (report_dict, None) on success, (None, error_str) on failure.
# # #     """
# # #     print(
# # #         f"\n[Batch] ── Folder: train={train_detail_id}  "
# # #         f"folder='{folder_name}'  ({len(rows)} video(s)) ──"
# # #     )

# # #     # folder_name is unique per recording session → use as analysis_id
# # #     analysis_id = folder_name

# # #     # Step 1 — mark all rows in-progress
# # #     for row in rows:
# # #         try:
# # #             set_process_flag(row["id"], "I")
# # #         except Exception as exc:
# # #             print(f"[Batch] Could not mark row {row['id']} as 'I': {exc}")

# # #     # Step 2 — download every video from S3 to a temp file
# # #     tmp_paths: List[str] = []
# # #     try:
# # #         for row in rows:
# # #             suffix = os.path.splitext(row["filename"] or "video.mp4")[1] or ".mp4"
# # #             tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
# # #             os.close(tmp_fd)
# # #             download_video_from_s3(row["s3_video_path"], tmp_path)
# # #             tmp_paths.append(tmp_path)
# # #     except Exception as exc:
# # #         err = f"folder='{folder_name}' download failed: {exc}"
# # #         print(f"[Batch] {err}")
# # #         _cleanup_temps(tmp_paths)
# # #         return None, err

# # #     # Step 3 — run the pipeline over each video in seq_no order
# # #     report_path = ""
# # #     try:
# # #         for idx, (row, tmp_path) in enumerate(zip(rows, tmp_paths)):
# # #             print(
# # #                 f"[Batch]   [{idx + 1}/{len(rows)}]  "
# # #                 f"{row['filename']}  (seq={row['seq_no']})"
# # #             )
# # #             pipeline = GadgetDetectionPipeline(
# # #                 source          = tmp_path,
# # #                 analysis_id     = analysis_id,
# # #                 train_detail_id = train_detail_id,
# # #                 save            = False,
# # #                 display         = False,
# # #             )
# # #             report_path = pipeline.run()

# # #         # Step 4 — mark all rows done
# # #         for row in rows:
# # #             try:
# # #                 set_process_flag(row["id"], "Y")
# # #             except Exception as exc:
# # #                 print(f"[Batch] Could not mark row {row['id']} as 'Y': {exc}")

# # #         # Step 5 — read and return the report
# # #         if report_path and os.path.isfile(report_path):
# # #             with open(report_path, encoding="utf-8") as f:
# # #                 return json.load(f), None
# # #         else:
# # #             err = f"folder='{folder_name}' pipeline returned no report file"
# # #             print(f"[Batch] {err}")
# # #             return None, err

# # #     except Exception as exc:
# # #         # flags stay at 'I' — intentional, so operator can inspect
# # #         err = (
# # #             f"folder='{folder_name}' pipeline error: {exc}\n"
# # #             + traceback.format_exc()
# # #         )
# # #         print(f"[Batch] {err}")
# # #         return None, err

# # #     finally:
# # #         _cleanup_temps(tmp_paths)


# # # # ── Temp file cleanup ──────────────────────────────────────────────────────────

# # # def _cleanup_temps(paths: List[str]) -> None:
# # #     for p in paths:
# # #         try:
# # #             if os.path.isfile(p):
# # #                 os.remove(p)
# # #         except OSError:
# # #             pass


# # from __future__ import annotations

# # """
# # api.py — Loco Pilot Distraction Detection  (DB-driven batch runner)
# # ====================================================================

# # Videos are NOT uploaded through this API.
# # This service polls the database for video_files rows where
# # process_flag = 'N', downloads each video from S3, runs the detection
# # pipeline synchronously, and returns the final result directly.

# # Flag lifecycle
# # --------------
# #   N  →  pending   (set by the frontend when a video is uploaded to S3)
# #   I  →  in-progress (set here before the pipeline starts)
# #   Y  →  done      (set here only on successful pipeline completion)

# # If the pipeline crashes the flag stays at 'I' so operators can see
# # exactly which video failed.

# # Endpoints
# # ---------
# #   GET  /         — health / welcome
# #   GET  /health   — {"status": "ok"}
# #   POST /trigger  — scan DB → download from S3 → run pipeline → return result
# # """

# # import json
# # import os
# # import tempfile
# # import traceback
# # import uuid
# # from itertools import groupby
# # from typing import Any, Dict, List, Optional, Tuple

# # from fastapi import FastAPI, HTTPException
# # from fastapi.middleware.cors import CORSMiddleware
# # from fastapi.responses import JSONResponse

# # from main import GadgetDetectionPipeline
# # from utils.db_s3_uploader import (
# #     download_video_from_s3,
# #     get_pending_videos,
# #     set_process_flag,
# # )

# # # ── App setup ──────────────────────────────────────────────────────────────────

# # app = FastAPI(
# #     title   = "Loco Pilot Distraction Detection API",
# #     version = "3.0.0",
# # )

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins     = ["*"],
# #     allow_credentials = True,
# #     allow_methods     = ["*"],
# #     allow_headers     = ["*"],
# # )


# # # ── Routes ─────────────────────────────────────────────────────────────────────

# # @app.get("/")
# # def root() -> dict:
# #     return {
# #         "status":  "success",
# #         "message": "Loco Pilot Distraction Detection API is running",
# #         "health":  "/health",
# #         "docs":    "/docs",
# #         "trigger": "POST /trigger  — scan DB, process all pending videos, return results",
# #     }


# # @app.get("/health", tags=["status"])
# # def health() -> dict:
# #     return {"status": "ok"}


# # # ── Main trigger ───────────────────────────────────────────────────────────────

# # @app.post("/trigger", tags=["batch"])
# # def trigger_batch() -> JSONResponse:
# #     """
# #     Scan DB for all rows with process_flag = 'N'.
# #     Group by (train_detail_id, folder_name) — each group is one logical
# #     analysis (a folder of sequential videos from one recording session).
# #     Process every group in sequence, then return all results in the
# #     target batch envelope format.

# #     Response shape:
# #     {
# #       "batch_id":          "<hex>",
# #       "total_videos":      N,
# #       "completed":         N,
# #       "failed":            N,
# #       "folders_processed": N,
# #       "folders": [
# #         {
# #           "train_detail_id":   22803,
# #           "folder_name":       "22803-05-06-2026",
# #           "videos_in_folder":  8,
# #           "report": { ... }   ← full analysis_report.json content
# #         },
# #         ...
# #       ]
# #     }
# #     """
# #     try:
# #         pending = get_pending_videos()
# #     except Exception as exc:
# #         raise HTTPException(
# #             status_code = 500,
# #             detail      = f"Failed to query pending videos from DB: {exc}",
# #         )

# #     if not pending:
# #         return JSONResponse(content={
# #             "batch_id":          uuid.uuid4().hex[:12],
# #             "total_videos":      0,
# #             "completed":         0,
# #             "failed":            0,
# #             "folders_processed": 0,
# #             "folders":           [],
# #             "message":           "No pending videos found (process_flag = 'N').",
# #         })

# #     batch_id = uuid.uuid4().hex[:12]

# #     def _group_key(row: Dict[str, Any]):
# #         return (row["train_detail_id"], row["folder_name"])

# #     groups = [
# #         (key, list(grp_rows))
# #         for key, grp_rows in groupby(pending, key=_group_key)
# #     ]
# #     print(f"[Batch:{batch_id}] {len(pending)} video(s) across {len(groups)} folder group(s).")

# #     total_videos = len(pending)
# #     completed    = 0
# #     failed       = 0
# #     folders_out: List[Dict[str, Any]] = []

# #     for (train_detail_id, folder_name), rows in groups:
# #         report, error, n_completed, n_failed = _process_folder_group(
# #             train_detail_id = train_detail_id,
# #             folder_name     = folder_name,
# #             rows            = rows,
# #         )
# #         completed += n_completed
# #         failed    += n_failed

# #         folder_entry: Dict[str, Any] = {
# #             "train_detail_id":  train_detail_id,
# #             "folder_name":      folder_name,
# #             "videos_in_folder": len(rows),
# #         }
# #         if report is not None:
# #             folder_entry["report"] = report
# #         if error:
# #             folder_entry["error"] = error

# #         folders_out.append(folder_entry)

# #     return JSONResponse(content={
# #         "batch_id":          batch_id,
# #         "total_videos":      total_videos,
# #         "completed":         completed,
# #         "failed":            failed,
# #         "folders_processed": len(folders_out),
# #         "folders":           folders_out,
# #     })


# # # ── Per-folder-group processor ─────────────────────────────────────────────────

# # def _process_folder_group(
# #     train_detail_id: int,
# #     folder_name:     str,
# #     rows:            List[Dict[str, Any]],
# # ) -> Tuple[Optional[Dict[str, Any]], Optional[str], int, int]:
# #     """
# #     Process one folder group — all videos that share the same
# #     (train_detail_id, folder_name), ordered by seq_no.

# #     Returns (report_dict, error_str, n_completed, n_failed).
# #     """
# #     n_videos = len(rows)
# #     print(
# #         f"\n[Batch] ── Folder: train={train_detail_id}  "
# #         f"folder='{folder_name}'  ({n_videos} video(s)) ──"
# #     )

# #     # folder_name is unique per recording session → use as analysis_id
# #     analysis_id = folder_name

# #     # Step 1 — mark all rows in-progress
# #     for row in rows:
# #         try:
# #             set_process_flag(row["id"], "I")
# #         except Exception as exc:
# #             print(f"[Batch] Could not mark row {row['id']} as 'I': {exc}")

# #     # Step 2 — download every video from S3 to a temp file
# #     tmp_paths: List[str] = []
# #     try:
# #         for row in rows:
# #             suffix = os.path.splitext(row["filename"] or "video.mp4")[1] or ".mp4"
# #             tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
# #             os.close(tmp_fd)
# #             download_video_from_s3(row["s3_video_path"], tmp_path)
# #             tmp_paths.append(tmp_path)
# #     except Exception as exc:
# #         err = f"folder='{folder_name}' download failed: {exc}"
# #         print(f"[Batch] {err}")
# #         _cleanup_temps(tmp_paths)
# #         return None, err, 0, n_videos

# #     # Step 3 — run the pipeline over each video in seq_no order
# #     report_path = ""
# #     try:
# #         for idx, (row, tmp_path) in enumerate(zip(rows, tmp_paths)):
# #             print(
# #                 f"[Batch]   [{idx + 1}/{n_videos}]  "
# #                 f"{row['filename']}  (seq={row['seq_no']})"
# #             )
# #             pipeline = GadgetDetectionPipeline(
# #                 source          = tmp_path,
# #                 analysis_id     = analysis_id,
# #                 train_detail_id = train_detail_id,
# #                 save            = False,
# #                 display         = False,
# #             )
# #             report_path = pipeline.run()

# #         # Step 4 — mark all rows done
# #         for row in rows:
# #             try:
# #                 set_process_flag(row["id"], "Y")
# #             except Exception as exc:
# #                 print(f"[Batch] Could not mark row {row['id']} as 'Y': {exc}")

# #         # Step 5 — read and return the report
# #         if report_path and os.path.isfile(report_path):
# #             with open(report_path, encoding="utf-8") as f:
# #                 return json.load(f), None, n_videos, 0
# #         else:
# #             err = f"folder='{folder_name}' pipeline returned no report file"
# #             print(f"[Batch] {err}")
# #             return None, err, 0, n_videos

# #     except Exception as exc:
# #         # flags stay at 'I' — intentional, so operator can inspect
# #         err = (
# #             f"folder='{folder_name}' pipeline error: {exc}\n"
# #             + traceback.format_exc()
# #         )
# #         print(f"[Batch] {err}")
# #         return None, err, 0, n_videos

# #     finally:
# #         _cleanup_temps(tmp_paths)


# # # ── Temp file cleanup ──────────────────────────────────────────────────────────

# # def _cleanup_temps(paths: List[str]) -> None:
# #     for p in paths:
# #         try:
# #             if os.path.isfile(p):
# #                 os.remove(p)
# #         except OSError:
# #             pass

# from __future__ import annotations

# """
# api.py — Loco Pilot Distraction Detection  (DB-driven batch runner)
# ====================================================================

# Videos are NOT uploaded through this API.
# This service polls the database for video_files rows where
# process_flag = 'N', downloads each video from S3, runs the detection
# pipeline synchronously, and returns the final result directly.

# Flag lifecycle
# --------------
#   N  →  pending   (set by the frontend when a video is uploaded to S3)
#   I  →  in-progress (set here before the pipeline starts)
#   Y  →  done      (set here only on successful pipeline completion)

# If the pipeline crashes the flag stays at 'I' so operators can see
# exactly which video failed.

# Endpoints
# ---------
#   GET  /         — health / welcome
#   GET  /health   — {"status": "ok"}
#   POST /trigger  — scan DB → download from S3 → run pipeline → return result

# Response shape
# --------------
# {
#   "batch_id":          "444e4af5ff5f",
#   "total_videos":      2,
#   "completed":         2,
#   "failed":            0,
#   "folders_processed": 1,
#   "folders": [
#     {
#       "train_detail_id":  22801,
#       "folder_name":      "22801-05-06-2026",
#       "videos_in_folder": 2,
#       "report": {
#         "analysis_id":     "22801-05-06-2026",
#         "train_detail_id": 22801,
#         "processing_time": 108.124,
#         "video_info":      [ ... ],   ← list with one entry per video
#         "violations":      [ ... ]    ← all violations across ALL videos
#       }
#     }
#   ]
# }

# Timestamp logic
# ---------------
# For each folder group, videos are processed in seq_no order.
# A running time_offset and frame_offset accumulate as each video finishes.

#   global_timestamp  = local_video_time  + time_offset
#   global_frame      = local_frame_index + frame_offset

# In the report:
#   "timestamp"                  = global_timestamp  (HH:MM:SS into the full recording)
#   "original_video_timestamp"   = "<db_filename> <local_time>"  (time within that file)
# """

# import json
# import os
# import tempfile
# import traceback
# import uuid
# from typing import Any, Dict, List, Optional, Tuple

# import cv2

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse

# from main import GadgetDetectionPipeline
# from utils.violation_store import ViolationStore
# from utils.db_s3_uploader import (
#     download_video_from_s3,
#     get_pending_videos,
#     set_process_flag,
# )

# # ── App setup ──────────────────────────────────────────────────────────────────

# app = FastAPI(
#     title   = "Loco Pilot Distraction Detection API",
#     version = "3.0.0",
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins     = ["*"],
#     allow_credentials = True,
#     allow_methods     = ["*"],
#     allow_headers     = ["*"],
# )


# # ── Routes ─────────────────────────────────────────────────────────────────────

# @app.get("/")
# def root() -> dict:
#     return {
#         "status":  "success",
#         "message": "Loco Pilot Distraction Detection API is running",
#         "health":  "/health",
#         "docs":    "/docs",
#         "trigger": "POST /trigger  — scan DB, process all pending videos, return results",
#     }


# @app.get("/health", tags=["status"])
# def health() -> dict:
#     return {"status": "ok"}


# # ── Main trigger ───────────────────────────────────────────────────────────────

# @app.post("/trigger", tags=["batch"])
# def trigger_batch() -> JSONResponse:
#     """
#     Pick the SINGLE oldest pending folder (by upload_timestamp ASC LIMIT 1),
#     process all its videos in seq_no order, and return the result.

#     Call /trigger again to process the next folder.
#     """
#     batch_id = uuid.uuid4().hex[:12]

#     # get_pending_videos() now returns only the videos of the one oldest folder
#     try:
#         pending = get_pending_videos()
#     except Exception as exc:
#         raise HTTPException(
#             status_code = 500,
#             detail      = f"Failed to query pending videos from DB: {exc}",
#         )

#     if not pending:
#         return JSONResponse(content={
#             "batch_id":          batch_id,
#             "total_videos":      0,
#             "completed":         0,
#             "failed":            0,
#             "folders_processed": 0,
#             "folders":           [],
#             "message":           "No pending videos found (process_flag = 'N').",
#         })

#     # All rows belong to the same folder — take metadata from the first row
#     folder_name     = pending[0]["folder_name"]
#     train_detail_id = pending[0]["train_detail_id"]
#     total_videos    = len(pending)

#     print(f"[Batch:{batch_id}] Processing folder='{folder_name}'  "
#           f"train={train_detail_id}  videos={total_videos}")

#     report, error, completed, failed = _process_folder_group(
#         train_detail_id = train_detail_id,
#         folder_name     = folder_name,
#         rows            = pending,
#     )

#     folder_entry: Dict[str, Any] = {
#         "train_detail_id":  train_detail_id,
#         "folder_name":      folder_name,
#         "videos_in_folder": total_videos,
#     }
#     if report is not None:
#         folder_entry["report"] = report
#     if error:
#         folder_entry["error"] = error

#     return JSONResponse(content={
#         "batch_id":          batch_id,
#         "total_videos":      total_videos,
#         "completed":         completed,
#         "failed":            failed,
#         "folders_processed": 1,
#         "folders":           [folder_entry],
#     })


# # ── Per-folder-group processor ─────────────────────────────────────────────────

# def _video_duration_seconds(path: str) -> float:
#     """Read the duration of a video file using OpenCV."""
#     cap = cv2.VideoCapture(path)
#     if not cap.isOpened():
#         return 0.0
#     fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
#     total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#     cap.release()
#     return total / fps if fps > 0 and total > 0 else 0.0


# def _process_folder_group(
#     train_detail_id: int,
#     folder_name:     str,
#     rows:            List[Dict[str, Any]],
# ) -> Tuple[Optional[Dict[str, Any]], Optional[str], int, int]:
#     """
#     Process one folder group — all videos that share the same
#     (train_detail_id, folder_name), ordered by seq_no.

#     Key design
#     ──────────
#     • One ViolationStore is created for the whole folder and shared
#       across every pipeline run.  This means violations from all videos
#       accumulate in a single store.
#     • time_offset and frame_offset grow after each video so timestamps
#       are continuous across the whole recording session.
#     • finalize() is called ONCE after all videos are done.

#     Returns (report_dict, error_str, n_completed, n_failed).
#     """
#     n_videos = len(rows)
#     print(
#         f"\n[Batch] ── Folder: train={train_detail_id}  "
#         f"folder='{folder_name}'  ({n_videos} video(s)) ──"
#     )

#     analysis_id = folder_name

#     # Step 1 — mark all rows in-progress
#     for row in rows:
#         try:
#             set_process_flag(row["id"], "I")
#         except Exception as exc:
#             print(f"[Batch] Could not mark row {row['id']} as 'I': {exc}")

#     # Step 2 — download every video from S3 to a temp file
#     # We also record the DB filename alongside each temp path.
#     tmp_entries: List[Tuple[str, str]] = []   # [(tmp_path, db_filename), ...]
#     try:
#         for row in rows:
#             suffix = os.path.splitext(row["filename"] or "video.mp4")[1] or ".mp4"
#             tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
#             os.close(tmp_fd)
#             download_video_from_s3(row["s3_video_path"], tmp_path)
#             tmp_entries.append((tmp_path, row["filename"] or ""))
#     except Exception as exc:
#         err = f"folder='{folder_name}' download failed: {exc}"
#         print(f"[Batch] {err}")
#         _cleanup_temps([p for p, _ in tmp_entries])
#         return None, err, 0, n_videos

#     # Step 3 — create ONE shared ViolationStore for the entire folder
#     shared_vstore = ViolationStore(
#         analysis_id     = analysis_id,
#         train_detail_id = train_detail_id,
#         # No video_info here — each pipeline run will call add_video_info()
#     )

#     # Step 4 — run the pipeline over each video in seq_no order,
#     #           accumulating time_offset and frame_offset between videos.
#     time_offset  = 0.0
#     frame_offset = 0
#     total_processing_time = 0.0

#     try:
#         for idx, (row, (tmp_path, db_filename)) in enumerate(zip(rows, tmp_entries)):
#             print(
#                 f"[Batch]   [{idx + 1}/{n_videos}]  "
#                 f"{db_filename}  (seq={row['seq_no']})  "
#                 f"time_offset={time_offset:.2f}s  frame_offset={frame_offset}"
#             )

#             import time as _time
#             t0 = _time.time()

#             pipeline = GadgetDetectionPipeline(
#                 source          = tmp_path,
#                 analysis_id     = analysis_id,
#                 train_detail_id = train_detail_id,
#                 save            = False,
#                 display         = False,
#                 shared_vstore   = shared_vstore,
#                 time_offset     = time_offset,
#                 frame_offset    = frame_offset,
#                 source_filename = db_filename,
#             )
#             pipeline.run()   # returns "" in batch mode; vstore is NOT finalized here

#             video_duration = _video_duration_seconds(tmp_path)
#             video_frames   = _get_frame_count(tmp_path)

#             total_processing_time += _time.time() - t0

#             # Advance offsets for the next video
#             time_offset  += video_duration
#             frame_offset += video_frames

#         # Step 5 — finalize the shared store ONCE with total processing time
#         report_path = shared_vstore.finalize(processing_time=round(total_processing_time, 3))

#         # Step 6 — mark all rows done
#         for row in rows:
#             try:
#                 set_process_flag(row["id"], "Y")
#             except Exception as exc:
#                 print(f"[Batch] Could not mark row {row['id']} as 'Y': {exc}")

#         # Step 7 — read and return the report
#         if report_path and os.path.isfile(report_path):
#             with open(report_path, encoding="utf-8") as f:
#                 return json.load(f), None, n_videos, 0
#         else:
#             err = f"folder='{folder_name}' no report file after finalize"
#             print(f"[Batch] {err}")
#             return None, err, 0, n_videos

#     except Exception as exc:
#         # flags stay at 'I' — intentional so operator can inspect
#         err = (
#             f"folder='{folder_name}' pipeline error: {exc}\n"
#             + traceback.format_exc()
#         )
#         print(f"[Batch] {err}")
#         return None, err, 0, n_videos

#     finally:
#         _cleanup_temps([p for p, _ in tmp_entries])


# def _get_frame_count(path: str) -> int:
#     cap = cv2.VideoCapture(path)
#     if not cap.isOpened():
#         return 0
#     count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     cap.release()
#     return count


# # ── Temp file cleanup ──────────────────────────────────────────────────────────

# def _cleanup_temps(paths: List[str]) -> None:
#     for p in paths:
#         try:
#             if os.path.isfile(p):
#                 os.remove(p)
#         except OSError:
#             pass



# # # from __future__ import annotations

# # # """
# # # api.py — Loco Pilot Distraction Detection  (DB-driven batch runner)
# # # ====================================================================

# # # Videos are NO LONGER uploaded through this API.
# # # Instead, this service polls the database for video_files rows where
# # # process_flag = 'N', downloads each video from S3, runs the detection
# # # pipeline, and marks the row done (process_flag = 'Y').

# # # Flag lifecycle
# # # --------------
# # #   N  →  pending (set by the frontend / ingestion service when uploading)
# # #   I  →  in-progress (set here, immediately before the pipeline starts)
# # #   Y  →  done (set here, only on successful pipeline completion)

# # # If the pipeline crashes the flag stays at 'I' so operators can see
# # # exactly which video failed — it is NOT silently reset to 'N'.

# # # Endpoints
# # # ---------
# # #   GET  /              — health / welcome
# # #   GET  /health        — {"status": "ok"}
# # #   POST /trigger       — kick off a DB-scan + batch run immediately
# # #                         (returns job_id; poll /status/<job_id>)
# # #   GET  /status/<id>   — queued | processing | done | failed
# # #   GET  /result/<id>   — final JSON report (consumed once; deleted from memory)
# # # """

# # # import json
# # # import os
# # # import tempfile
# # # import traceback
# # # import uuid
# # # from concurrent.futures import ThreadPoolExecutor
# # # from itertools import groupby
# # # from typing import Any, Dict, List, Optional

# # # from fastapi import BackgroundTasks, FastAPI, HTTPException
# # # from fastapi.middleware.cors import CORSMiddleware
# # # from fastapi.responses import JSONResponse

# # # from main import GadgetDetectionPipeline
# # # from utils.db_s3_uploader import (
# # #     download_video_from_s3,
# # #     get_pending_videos,
# # #     set_process_flag,
# # # )

# # # # ── App setup ──────────────────────────────────────────────────────────────────

# # # app = FastAPI(
# # #     title   = "Loco Pilot Distraction Detection API",
# # #     version = "3.0.0",
# # # )

# # # app.add_middleware(
# # #     CORSMiddleware,
# # #     allow_origins     = ["*"],
# # #     allow_credentials = True,
# # #     allow_methods     = ["*"],
# # #     allow_headers     = ["*"],
# # # )

# # # # In-memory job registry  {job_id: {"status": str, "result": dict|None, "error": str|None}}
# # # _jobs:    Dict[str, Dict[str, Any]] = {}
# # # _executor = ThreadPoolExecutor(max_workers=1)  # one batch at a time


# # # # ── Routes ─────────────────────────────────────────────────────────────────────

# # # @app.get("/")
# # # def root() -> dict:
# # #     return {
# # #         "status":  "success",
# # #         "message": "Loco Pilot Distraction Detection API is running",
# # #         "health":  "/health",
# # #         "docs":    "/docs",
# # #         "trigger": "POST /trigger  — start a DB-scan + batch run",
# # #     }


# # # @app.get("/health", tags=["status"])
# # # def health() -> dict:
# # #     return {"status": "ok"}


# # # # ── Batch trigger ──────────────────────────────────────────────────────────────

# # # @app.post("/trigger", tags=["batch"])
# # # async def trigger_batch(background_tasks: BackgroundTasks) -> JSONResponse:
# # #     """
# # #     Scan the DB for all pending videos (process_flag='N') and process them
# # #     as a batch.  Returns immediately with a job_id; poll /status/<job_id>.
# # #     """
# # #     job_id = str(uuid.uuid4())
# # #     _jobs[job_id] = {"status": "queued", "result": None, "error": None}
# # #     _executor.submit(_run_batch, job_id)
# # #     return JSONResponse(
# # #         status_code = 202,
# # #         content = {
# # #             "job_id":  job_id,
# # #             "status":  "queued",
# # #             "message": f"Batch job accepted. Poll GET /status/{job_id} for progress.",
# # #         },
# # #     )


# # # @app.get("/status/{job_id}", tags=["batch"])
# # # def job_status(job_id: str) -> JSONResponse:
# # #     job = _jobs.get(job_id)
# # #     if job is None:
# # #         raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
# # #     resp: Dict[str, Any] = {"job_id": job_id, "status": job["status"]}
# # #     if job["status"] == "failed":
# # #         resp["error"] = job["error"]
# # #     return JSONResponse(content=resp)


# # # @app.get("/result/{job_id}", tags=["batch"])
# # # def job_result(job_id: str) -> JSONResponse:
# # #     job = _jobs.get(job_id)
# # #     if job is None:
# # #         raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
# # #     if job["status"] == "failed":
# # #         raise HTTPException(status_code=500, detail=job["error"])
# # #     if job["status"] in ("queued", "processing"):
# # #         raise HTTPException(
# # #             status_code=409,
# # #             detail=f"Job is still '{job['status']}'. Try again later.",
# # #         )
# # #     result = _jobs.pop(job_id)["result"]
# # #     return JSONResponse(content=result)


# # # # ── Batch worker ───────────────────────────────────────────────────────────────

# # # def _run_batch(job_id: str) -> None:
# # #     """
# # #     Entry point executed in the thread pool.

# # #     1. Query DB for all rows with process_flag = 'N'.
# # #     2. Group them by (train_detail_id, folder_name) — each group is one
# # #        logical analysis (the videos in a folder are one continuous recording).
# # #     3. For every group:
# # #          a. Mark every row as 'I' (in-progress).
# # #          b. Download each video from S3 to a temp file.
# # #          c. Run GadgetDetectionPipeline over every video in sequence.
# # #          d. On success: mark every row as 'Y' and collect the report.
# # #          e. On failure: leave rows at 'I' (flag stays for operator inspection).
# # #     4. Collect per-group reports and write them to the job registry.
# # #     """
# # #     _jobs[job_id]["status"] = "processing"
# # #     all_reports: List[Dict[str, Any]] = []

# # #     try:
# # #         pending = get_pending_videos()

# # #         if not pending:
# # #             print("[Batch] No pending videos found.")
# # #             _jobs[job_id]["status"] = "done"
# # #             _jobs[job_id]["result"] = {"message": "No pending videos.", "reports": []}
# # #             return

# # #         # Group by (train_detail_id, folder_name) — same order as the DB query
# # #         def _group_key(row: Dict[str, Any]):
# # #             return (row["train_detail_id"], row["folder_name"])

# # #         groups = [
# # #             (key, list(rows))
# # #             for key, rows in groupby(pending, key=_group_key)
# # #         ]
# # #         print(f"[Batch] {len(pending)} video(s) across {len(groups)} folder group(s).")

# # #         for (train_detail_id, folder_name), rows in groups:
# # #             report = _process_folder_group(
# # #                 train_detail_id = train_detail_id,
# # #                 folder_name     = folder_name,
# # #                 rows            = rows,
# # #             )
# # #             if report:
# # #                 all_reports.append(report)

# # #         _jobs[job_id]["status"] = "done"
# # #         _jobs[job_id]["result"] = {
# # #             "message": f"Processed {len(all_reports)} folder group(s).",
# # #             "reports": all_reports,
# # #         }

# # #     except Exception as exc:
# # #         _jobs[job_id]["status"] = "failed"
# # #         _jobs[job_id]["error"]  = f"{exc}\n{traceback.format_exc()}"
# # #         print(f"[Batch] Fatal error: {exc}")


# # # def _process_folder_group(
# # #     train_detail_id: int,
# # #     folder_name:     str,
# # #     rows:            List[Dict[str, Any]],
# # # ) -> Optional[Dict[str, Any]]:
# # #     """
# # #     Process one folder group (all videos that belong to a single analysis).

# # #     Rows are already ordered by seq_no (guaranteed by the DB query).
# # #     Each video is processed in sequence so frame offsets accumulate
# # #     correctly across files.

# # #     Returns the JSON report dict on success, None on failure.
# # #     """
# # #     print(
# # #         f"\n[Batch] ── Folder group: train={train_detail_id}  "
# # #         f"folder='{folder_name}'  ({len(rows)} video(s)) ──"
# # #     )

# # #     # Use folder_name as the analysis_id (unique per recording session)
# # #     analysis_id = folder_name

# # #     # ── Step 1: mark all rows as in-progress ─────────────────────────────────
# # #     for row in rows:
# # #         try:
# # #             set_process_flag(row["id"], "I")
# # #         except Exception as exc:
# # #             print(f"[Batch] Could not mark row {row['id']} as 'I': {exc}")
# # #             # Non-fatal — continue; the row stays 'N' which is safer than crashing

# # #     # ── Step 2: download videos to temp files ─────────────────────────────────
# # #     tmp_paths: List[str] = []
# # #     try:
# # #         for row in rows:
# # #             suffix    = os.path.splitext(row["filename"] or "video.mp4")[1] or ".mp4"
# # #             tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
# # #             os.close(tmp_fd)
# # #             download_video_from_s3(row["s3_video_path"], tmp_path)
# # #             tmp_paths.append(tmp_path)
# # #     except Exception as exc:
# # #         print(f"[Batch] Download failed for folder '{folder_name}': {exc}")
# # #         _cleanup_temps(tmp_paths)
# # #         return None

# # #     # ── Step 3: run the pipeline over each video in sequence ──────────────────
# # #     report_path: str = ""
# # #     try:
# # #         # The pipeline processes one video at a time but shares the same
# # #         # ViolationStore (via analysis_id = folder_name) so violations from
# # #         # every video land in the same report.
# # #         for idx, (row, tmp_path) in enumerate(zip(rows, tmp_paths)):
# # #             print(
# # #                 f"[Batch]   [{idx+1}/{len(rows)}] Processing  "
# # #                 f"{row['filename']}  (seq={row['seq_no']})"
# # #             )
# # #             pipeline = GadgetDetectionPipeline(
# # #                 source          = tmp_path,
# # #                 analysis_id     = analysis_id,
# # #                 train_detail_id = train_detail_id,
# # #                 save            = False,
# # #                 display         = False,
# # #             )
# # #             # run() returns the path to analysis_report.json
# # #             report_path = pipeline.run()

# # #         # ── Step 4: mark all rows as done ────────────────────────────────────
# # #         for row in rows:
# # #             try:
# # #                 set_process_flag(row["id"], "Y")
# # #             except Exception as exc:
# # #                 print(f"[Batch] Could not mark row {row['id']} as 'Y': {exc}")

# # #         # ── Step 5: read and return the report ───────────────────────────────
# # #         if report_path and os.path.isfile(report_path):
# # #             with open(report_path, encoding="utf-8") as f:
# # #                 return json.load(f)
# # #         else:
# # #             print(f"[Batch] Report file missing for folder '{folder_name}'")
# # #             return None

# # #     except Exception as exc:
# # #         # Leave flags at 'I' so the operator can see which group failed
# # #         print(
# # #             f"[Batch] Pipeline error for folder '{folder_name}': {exc}\n"
# # #             + traceback.format_exc()
# # #         )
# # #         return None

# # #     finally:
# # #         _cleanup_temps(tmp_paths)


# # # def _cleanup_temps(paths: List[str]) -> None:
# # #     for p in paths:
# # #         try:
# # #             if os.path.isfile(p):
# # #                 os.remove(p)
# # #         except OSError:
# # #             pass



# # from __future__ import annotations

# # """
# # api.py — Loco Pilot Distraction Detection  (DB-driven batch runner)
# # ====================================================================

# # Videos are NOT uploaded through this API.
# # This service polls the database for video_files rows where
# # process_flag = 'N', downloads each video from S3, runs the detection
# # pipeline synchronously, and returns the final result directly.

# # Flag lifecycle
# # --------------
# #   N  →  pending   (set by the frontend when a video is uploaded to S3)
# #   I  →  in-progress (set here before the pipeline starts)
# #   Y  →  done      (set here only on successful pipeline completion)

# # If the pipeline crashes the flag stays at 'I' so operators can see
# # exactly which video failed.

# # Endpoints
# # ---------
# #   GET  /         — health / welcome
# #   GET  /health   — {"status": "ok"}
# #   POST /trigger  — scan DB → download from S3 → run pipeline → return result
# # """

# # import json
# # import os
# # import tempfile
# # import traceback
# # from itertools import groupby
# # from typing import Any, Dict, List, Optional

# # from fastapi import FastAPI, HTTPException
# # from fastapi.middleware.cors import CORSMiddleware
# # from fastapi.responses import JSONResponse

# # from main import GadgetDetectionPipeline
# # from utils.db_s3_uploader import (
# #     download_video_from_s3,
# #     get_pending_videos,
# #     set_process_flag,
# # )

# # # ── App setup ──────────────────────────────────────────────────────────────────

# # app = FastAPI(
# #     title   = "Loco Pilot Distraction Detection API",
# #     version = "3.0.0",
# # )

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins     = ["*"],
# #     allow_credentials = True,
# #     allow_methods     = ["*"],
# #     allow_headers     = ["*"],
# # )


# # # ── Routes ─────────────────────────────────────────────────────────────────────

# # @app.get("/")
# # def root() -> dict:
# #     return {
# #         "status":  "success",
# #         "message": "Loco Pilot Distraction Detection API is running",
# #         "health":  "/health",
# #         "docs":    "/docs",
# #         "trigger": "POST /trigger  — scan DB, process all pending videos, return results",
# #     }


# # @app.get("/health", tags=["status"])
# # def health() -> dict:
# #     return {"status": "ok"}


# # # ── Main trigger ───────────────────────────────────────────────────────────────

# # @app.post("/trigger", tags=["batch"])
# # def trigger_batch() -> JSONResponse:
# #     """
# #     Scan DB for all rows with process_flag = 'N'.
# #     Group by (train_detail_id, folder_name) — each group is one logical
# #     analysis (a folder of sequential videos from one recording session).
# #     Process every group in sequence, then return all reports together.

# #     Flag lifecycle per video row:
# #       N  →  I  (before pipeline starts)
# #       I  →  Y  (after pipeline succeeds)
# #       stays I   (if pipeline fails — visible to operators)
# #     """
# #     try:
# #         pending = get_pending_videos()
# #     except Exception as exc:
# #         raise HTTPException(
# #             status_code=500,
# #             detail=f"Failed to query pending videos from DB: {exc}",
# #         )

# #     if not pending:
# #         return JSONResponse(content={
# #             "status":  "ok",
# #             "message": "No pending videos found (process_flag = 'N').",
# #             "reports": [],
# #         })

# #     def _group_key(row: Dict[str, Any]):
# #         return (row["train_detail_id"], row["folder_name"])

# #     groups = [
# #         (key, list(rows))
# #         for key, rows in groupby(pending, key=_group_key)
# #     ]
# #     print(f"[Batch] {len(pending)} video(s) across {len(groups)} folder group(s).")

# #     all_reports: List[Dict[str, Any]] = []
# #     errors:      List[str]            = []

# #     for (train_detail_id, folder_name), rows in groups:
# #         report, error = _process_folder_group(
# #             train_detail_id = train_detail_id,
# #             folder_name     = folder_name,
# #             rows            = rows,
# #         )
# #         if report is not None:
# #             all_reports.append(report)
# #         if error:
# #             errors.append(error)

# #     return JSONResponse(content={
# #         "status":         "ok" if not errors else "partial",
# #         "groups_total":   len(groups),
# #         "groups_success": len(all_reports),
# #         "groups_failed":  len(errors),
# #         "errors":         errors,
# #         "reports":        all_reports,
# #     })


# # # ── Per-folder-group processor ─────────────────────────────────────────────────

# # def _process_folder_group(
# #     train_detail_id: int,
# #     folder_name:     str,
# #     rows:            List[Dict[str, Any]],
# # ) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
# #     """
# #     Process one folder group — all videos that share the same
# #     (train_detail_id, folder_name), ordered by seq_no.

# #     Returns (report_dict, None) on success, (None, error_str) on failure.
# #     """
# #     print(
# #         f"\n[Batch] ── Folder: train={train_detail_id}  "
# #         f"folder='{folder_name}'  ({len(rows)} video(s)) ──"
# #     )

# #     # folder_name is unique per recording session → use as analysis_id
# #     analysis_id = folder_name

# #     # Step 1 — mark all rows in-progress
# #     for row in rows:
# #         try:
# #             set_process_flag(row["id"], "I")
# #         except Exception as exc:
# #             print(f"[Batch] Could not mark row {row['id']} as 'I': {exc}")

# #     # Step 2 — download every video from S3 to a temp file
# #     tmp_paths: List[str] = []
# #     try:
# #         for row in rows:
# #             suffix = os.path.splitext(row["filename"] or "video.mp4")[1] or ".mp4"
# #             tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
# #             os.close(tmp_fd)
# #             download_video_from_s3(row["s3_video_path"], tmp_path)
# #             tmp_paths.append(tmp_path)
# #     except Exception as exc:
# #         err = f"folder='{folder_name}' download failed: {exc}"
# #         print(f"[Batch] {err}")
# #         _cleanup_temps(tmp_paths)
# #         return None, err

# #     # Step 3 — run the pipeline over each video in seq_no order
# #     report_path = ""
# #     try:
# #         for idx, (row, tmp_path) in enumerate(zip(rows, tmp_paths)):
# #             print(
# #                 f"[Batch]   [{idx + 1}/{len(rows)}]  "
# #                 f"{row['filename']}  (seq={row['seq_no']})"
# #             )
# #             pipeline = GadgetDetectionPipeline(
# #                 source          = tmp_path,
# #                 analysis_id     = analysis_id,
# #                 train_detail_id = train_detail_id,
# #                 save            = False,
# #                 display         = False,
# #             )
# #             report_path = pipeline.run()

# #         # Step 4 — mark all rows done
# #         for row in rows:
# #             try:
# #                 set_process_flag(row["id"], "Y")
# #             except Exception as exc:
# #                 print(f"[Batch] Could not mark row {row['id']} as 'Y': {exc}")

# #         # Step 5 — read and return the report
# #         if report_path and os.path.isfile(report_path):
# #             with open(report_path, encoding="utf-8") as f:
# #                 return json.load(f), None
# #         else:
# #             err = f"folder='{folder_name}' pipeline returned no report file"
# #             print(f"[Batch] {err}")
# #             return None, err

# #     except Exception as exc:
# #         # flags stay at 'I' — intentional, so operator can inspect
# #         err = (
# #             f"folder='{folder_name}' pipeline error: {exc}\n"
# #             + traceback.format_exc()
# #         )
# #         print(f"[Batch] {err}")
# #         return None, err

# #     finally:
# #         _cleanup_temps(tmp_paths)


# # # ── Temp file cleanup ──────────────────────────────────────────────────────────

# # def _cleanup_temps(paths: List[str]) -> None:
# #     for p in paths:
# #         try:
# #             if os.path.isfile(p):
# #                 os.remove(p)
# #         except OSError:
# #             pass


# from __future__ import annotations

# """
# api.py — Loco Pilot Distraction Detection  (DB-driven batch runner)
# ====================================================================

# Videos are NOT uploaded through this API.
# This service polls the database for video_files rows where
# process_flag = 'N', downloads each video from S3, runs the detection
# pipeline synchronously, and returns the final result directly.

# Flag lifecycle
# --------------
#   N  →  pending   (set by the frontend when a video is uploaded to S3)
#   I  →  in-progress (set here before the pipeline starts)
#   Y  →  done      (set here only on successful pipeline completion)

# If the pipeline crashes the flag stays at 'I' so operators can see
# exactly which video failed.

# Endpoints
# ---------
#   GET  /         — health / welcome
#   GET  /health   — {"status": "ok"}
#   POST /trigger  — scan DB → download from S3 → run pipeline → return result
# """

# import json
# import os
# import tempfile
# import traceback
# import uuid
# from itertools import groupby
# from typing import Any, Dict, List, Optional, Tuple

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse

# from main import GadgetDetectionPipeline
# from utils.db_s3_uploader import (
#     download_video_from_s3,
#     get_pending_videos,
#     set_process_flag,
# )

# # ── App setup ──────────────────────────────────────────────────────────────────

# app = FastAPI(
#     title   = "Loco Pilot Distraction Detection API",
#     version = "3.0.0",
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins     = ["*"],
#     allow_credentials = True,
#     allow_methods     = ["*"],
#     allow_headers     = ["*"],
# )


# # ── Routes ─────────────────────────────────────────────────────────────────────

# @app.get("/")
# def root() -> dict:
#     return {
#         "status":  "success",
#         "message": "Loco Pilot Distraction Detection API is running",
#         "health":  "/health",
#         "docs":    "/docs",
#         "trigger": "POST /trigger  — scan DB, process all pending videos, return results",
#     }


# @app.get("/health", tags=["status"])
# def health() -> dict:
#     return {"status": "ok"}


# # ── Main trigger ───────────────────────────────────────────────────────────────

# @app.post("/trigger", tags=["batch"])
# def trigger_batch() -> JSONResponse:
#     """
#     Scan DB for all rows with process_flag = 'N'.
#     Group by (train_detail_id, folder_name) — each group is one logical
#     analysis (a folder of sequential videos from one recording session).
#     Process every group in sequence, then return all results in the
#     target batch envelope format.

#     Response shape:
#     {
#       "batch_id":          "<hex>",
#       "total_videos":      N,
#       "completed":         N,
#       "failed":            N,
#       "folders_processed": N,
#       "folders": [
#         {
#           "train_detail_id":   22803,
#           "folder_name":       "22803-05-06-2026",
#           "videos_in_folder":  8,
#           "report": { ... }   ← full analysis_report.json content
#         },
#         ...
#       ]
#     }
#     """
#     try:
#         pending = get_pending_videos()
#     except Exception as exc:
#         raise HTTPException(
#             status_code = 500,
#             detail      = f"Failed to query pending videos from DB: {exc}",
#         )

#     if not pending:
#         return JSONResponse(content={
#             "batch_id":          uuid.uuid4().hex[:12],
#             "total_videos":      0,
#             "completed":         0,
#             "failed":            0,
#             "folders_processed": 0,
#             "folders":           [],
#             "message":           "No pending videos found (process_flag = 'N').",
#         })

#     batch_id = uuid.uuid4().hex[:12]

#     def _group_key(row: Dict[str, Any]):
#         return (row["train_detail_id"], row["folder_name"])

#     groups = [
#         (key, list(grp_rows))
#         for key, grp_rows in groupby(pending, key=_group_key)
#     ]
#     print(f"[Batch:{batch_id}] {len(pending)} video(s) across {len(groups)} folder group(s).")

#     total_videos = len(pending)
#     completed    = 0
#     failed       = 0
#     folders_out: List[Dict[str, Any]] = []

#     for (train_detail_id, folder_name), rows in groups:
#         report, error, n_completed, n_failed = _process_folder_group(
#             train_detail_id = train_detail_id,
#             folder_name     = folder_name,
#             rows            = rows,
#         )
#         completed += n_completed
#         failed    += n_failed

#         folder_entry: Dict[str, Any] = {
#             "train_detail_id":  train_detail_id,
#             "folder_name":      folder_name,
#             "videos_in_folder": len(rows),
#         }
#         if report is not None:
#             folder_entry["report"] = report
#         if error:
#             folder_entry["error"] = error

#         folders_out.append(folder_entry)

#     return JSONResponse(content={
#         "batch_id":          batch_id,
#         "total_videos":      total_videos,
#         "completed":         completed,
#         "failed":            failed,
#         "folders_processed": len(folders_out),
#         "folders":           folders_out,
#     })


# # ── Per-folder-group processor ─────────────────────────────────────────────────

# def _process_folder_group(
#     train_detail_id: int,
#     folder_name:     str,
#     rows:            List[Dict[str, Any]],
# ) -> Tuple[Optional[Dict[str, Any]], Optional[str], int, int]:
#     """
#     Process one folder group — all videos that share the same
#     (train_detail_id, folder_name), ordered by seq_no.

#     Returns (report_dict, error_str, n_completed, n_failed).
#     """
#     n_videos = len(rows)
#     print(
#         f"\n[Batch] ── Folder: train={train_detail_id}  "
#         f"folder='{folder_name}'  ({n_videos} video(s)) ──"
#     )

#     # folder_name is unique per recording session → use as analysis_id
#     analysis_id = folder_name

#     # Step 1 — mark all rows in-progress
#     for row in rows:
#         try:
#             set_process_flag(row["id"], "I")
#         except Exception as exc:
#             print(f"[Batch] Could not mark row {row['id']} as 'I': {exc}")

#     # Step 2 — download every video from S3 to a temp file
#     tmp_paths: List[str] = []
#     try:
#         for row in rows:
#             suffix = os.path.splitext(row["filename"] or "video.mp4")[1] or ".mp4"
#             tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
#             os.close(tmp_fd)
#             download_video_from_s3(row["s3_video_path"], tmp_path)
#             tmp_paths.append(tmp_path)
#     except Exception as exc:
#         err = f"folder='{folder_name}' download failed: {exc}"
#         print(f"[Batch] {err}")
#         _cleanup_temps(tmp_paths)
#         return None, err, 0, n_videos

#     # Step 3 — run the pipeline over each video in seq_no order
#     report_path = ""
#     try:
#         for idx, (row, tmp_path) in enumerate(zip(rows, tmp_paths)):
#             print(
#                 f"[Batch]   [{idx + 1}/{n_videos}]  "
#                 f"{row['filename']}  (seq={row['seq_no']})"
#             )
#             pipeline = GadgetDetectionPipeline(
#                 source          = tmp_path,
#                 analysis_id     = analysis_id,
#                 train_detail_id = train_detail_id,
#                 save            = False,
#                 display         = False,
#             )
#             report_path = pipeline.run()

#         # Step 4 — mark all rows done
#         for row in rows:
#             try:
#                 set_process_flag(row["id"], "Y")
#             except Exception as exc:
#                 print(f"[Batch] Could not mark row {row['id']} as 'Y': {exc}")

#         # Step 5 — read and return the report
#         if report_path and os.path.isfile(report_path):
#             with open(report_path, encoding="utf-8") as f:
#                 return json.load(f), None, n_videos, 0
#         else:
#             err = f"folder='{folder_name}' pipeline returned no report file"
#             print(f"[Batch] {err}")
#             return None, err, 0, n_videos

#     except Exception as exc:
#         # flags stay at 'I' — intentional, so operator can inspect
#         err = (
#             f"folder='{folder_name}' pipeline error: {exc}\n"
#             + traceback.format_exc()
#         )
#         print(f"[Batch] {err}")
#         return None, err, 0, n_videos

#     finally:
#         _cleanup_temps(tmp_paths)


# # ── Temp file cleanup ──────────────────────────────────────────────────────────

# def _cleanup_temps(paths: List[str]) -> None:
#     for p in paths:
#         try:
#             if os.path.isfile(p):
#                 os.remove(p)
#         except OSError:
#             pass

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

from main import GadgetDetectionPipeline
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

    # Step 1 — mark first video as 'I' (in-progress), rest as 'Q' (queued).
    #           Operators can now see exactly which video is active vs waiting.
    for idx, row in enumerate(rows):
        flag = "I" if idx == 0 else "Q"
        try:
            set_process_flag(row["id"], flag)
        except Exception as exc:
            print(f"[Batch] Could not mark row {row['id']} as '{flag}': {exc}")

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
            # Promote Q → I for every video after the first
            # (first video was already marked 'I' in Step 1)
            if idx > 0:
                try:
                    set_process_flag(row["id"], "I")
                except Exception as exc:
                    print(f"[Batch] Could not promote row {row['id']} Q→I: {exc}")

            print(
                f"[Batch]   [{idx + 1}/{n_videos}]  "
                f"{db_filename}  (seq={row['seq_no']})  "
                f"time_offset={time_offset:.2f}s  frame_offset={frame_offset}"
            )

            import time as _time
            t0 = _time.time()

            pipeline = GadgetDetectionPipeline(
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

            # Mark this video done immediately — don't wait for the whole folder
            try:
                set_process_flag(row["id"], "Y")
            except Exception as exc:
                print(f"[Batch] Could not mark row {row['id']} as 'Y': {exc}")

            # Advance offsets for the next video
            time_offset  += video_duration
            frame_offset += video_frames

        # Step 5 — finalize the shared store ONCE with total processing time
        report_path = shared_vstore.finalize(processing_time=round(total_processing_time, 3))

        # Step 6 — all rows already marked 'Y' individually in the loop above

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