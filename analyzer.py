# # """
# # analyzer.py
# # ───────────
# # Bridges the existing GadgetDetectionPipeline / ViolationStore to the new
# # Journey-based workflow.

# # Responsibilities
# # ────────────────
# # 1. Receive a list of (VideoJob, local_tmp_path) pairs.
# # 2. Run the existing ML pipeline over each video in sequence_no order,
# #    using the shared ViolationStore / time_offset / frame_offset pattern
# #    already present in the legacy batch runner (api.py).
# # 3. After all videos finish, call the ViolationStore internal steps
# #    (dedup → merge → extract frames) without triggering the legacy
# #    S3/DB upload path (i.e. we do NOT call finalize()).
# # 4. Upload every violation frame to S3 under
# #       journeys/<journeyId>/frames/<filename>.jpg
# #    and replace the local path with the returned S3 key.
# # 5. Return a list of VideoResult objects ready for the completion callback.

# # What this file does NOT do
# # ──────────────────────────
# # • Does NOT modify any detector, ML model, or ViolationStore internals.
# # • Does NOT call finalize() — that would trigger the old S3/DB path.
# # • Does NOT call add_video_info() — GadgetDetectionPipeline.run() already
# #   calls it internally when shared_vstore is provided.
# # • Does NOT call any callback — that is consumer.py's responsibility.
# # • Does NOT acknowledge the RabbitMQ message.
# # """

# # from __future__ import annotations

# # import os
# # import time
# # from typing import Dict, List, Tuple

# # import cv2

# # from models import VideoJob, VideoResult, ViolationResult
# # from s3_service import upload_frame, upload_frame_from_path

# # # Import the existing pipeline and store — unchanged
# # from main import GadgetDetectionPipeline
# # from utils.violation_store import ViolationStore

# # # OUTPUTS_ROOT mirrors the constant inside violation_store.py
# # OUTPUTS_ROOT = "outputs"


# # # ── Helpers ──────────────────────────────────────────────────────────────────

# # def _video_meta(path: str) -> Tuple[float, float, int, float]:
# #     """Returns (duration_seconds, fps, total_frames, size_mb)."""
# #     cap = cv2.VideoCapture(path)
# #     if not cap.isOpened():
# #         return 0.0, 25.0, 0, 0.0
# #     fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
# #     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# #     cap.release()
# #     duration = total / fps if fps > 0 and total > 0 else 0.0
# #     size_mb  = round(os.path.getsize(path) / 1_000_000, 2) if os.path.isfile(path) else 0.0
# #     return duration, fps, total, size_mb


# # def _fmt_duration(seconds: float) -> str:
# #     t  = int(seconds)
# #     hh = t // 3600
# #     mm = (t % 3600) // 60
# #     ss = t % 60
# #     return f"{hh}:{mm:02d}:{ss:02d}"


# # def _event_type_to_violation_type(event_type: str) -> str:
# #     """Map internal event_type strings → Spring Boot ViolationType enum values."""
# #     return {
# #         "phone_use":       "PHONE_USAGE",
# #         "seat_absence":    "SEAT_ABSENCE",
# #         "drowsy":          "DROWSINESS",
# #         "sleeping":        "DROWSINESS",
# #         "sleeping_absent": "DROWSINESS",
# #     }.get(event_type.lower(), event_type.upper())


# # def _severity_for_risk(risk_score: int) -> str:
# #     if risk_score >= 80:
# #         return "HIGH"
# #     if risk_score >= 50:
# #         return "MEDIUM"
# #     return "LOW"


# # def _local_frame_path(vstore: ViolationStore, relative_path: str) -> str:
# #     """
# #     Convert the relative frame path stored by ViolationStore._save_frame()
# #     back to an absolute local disk path.

# #     _save_frame() returns:  "<analysis_id>/frames/<filename>"
# #     The actual disk path is: "outputs/<analysis_id>/frames/<filename>"
# #     """
# #     filename = os.path.basename(relative_path)
# #     return os.path.join(OUTPUTS_ROOT, vstore.analysis_id, "frames", filename)


# # # ── Main entry point ──────────────────────────────────────────────────────────

# # def analyze_journey(
# #     job_id:      str,
# #     journey_id:  int,
# #     video_jobs:  List[VideoJob],
# #     tmp_paths:   Dict[int, str],          # video_id → local tmp file path
# #     progress_cb  = None,                  # optional callable(progress_pct: int, message: str)
# # ) -> Tuple[List[VideoResult], float]:
# #     """
# #     Run the full detection pipeline over all videos in a journey.

# #     Parameters
# #     ──────────
# #     job_id      : RabbitMQ job ID (used for logging only).
# #     journey_id  : Journey ID — used for S3 key prefix and ViolationStore ID.
# #     video_jobs  : List of VideoJob objects (will be sorted by sequence_no).
# #     tmp_paths   : Mapping of video_id → absolute local temp file path.
# #     progress_cb : Optional callable(pct: int, msg: str) for progress updates.

# #     Returns
# #     ───────
# #     (video_results, total_wall_clock_seconds)
# #     """
# #     # Sort by sequence_no — critical for correct time/frame offset continuity
# #     ordered  = sorted(video_jobs, key=lambda v: v.sequence_no)
# #     n_videos = len(ordered)

# #     # One shared ViolationStore for the whole journey.
# #     # GadgetDetectionPipeline.run() calls add_video_info() itself when
# #     # shared_vstore is provided — do NOT call it here.
# #     analysis_id   = f"journey_{journey_id}"
# #     shared_vstore = ViolationStore(
# #         analysis_id     = analysis_id,
# #         train_detail_id = journey_id,
# #     )

# #     time_offset  = 0.0
# #     frame_offset = 0
# #     wall_start   = time.time()

# #     # Per-video metadata cache (keyed by video_id)
# #     meta_by_id: Dict[int, dict] = {}

# #     # ── Step 4: Run the AI pipeline over each video ───────────────────────────
# #     for idx, vj in enumerate(ordered):
# #         tmp_path    = tmp_paths[vj.video_id]
# #         db_filename = os.path.basename(vj.s3_key)

# #         duration, fps, total_frames, size_mb = _video_meta(tmp_path)
# #         meta_by_id[vj.video_id] = {
# #             "duration_seconds":   duration,
# #             "duration_formatted": _fmt_duration(duration),
# #             "fps":                fps,
# #             "size_mb":            size_mb,
# #             "total_frames":       total_frames,
# #         }

# #         print(
# #             f"[Analyzer:{job_id}]  [{idx + 1}/{n_videos}]  "
# #             f"video_id={vj.video_id}  seq={vj.sequence_no}  "
# #             f"file={db_filename}  "
# #             f"time_offset={time_offset:.2f}s  frame_offset={frame_offset}"
# #         )

# #         # GadgetDetectionPipeline.run() calls shared_vstore.add_video_info()
# #         # internally when shared_vstore is not None — do not duplicate that call.
# #         pipeline = GadgetDetectionPipeline(
# #             source          = tmp_path,
# #             analysis_id     = analysis_id,
# #             train_detail_id = journey_id,
# #             save            = False,
# #             display         = False,
# #             shared_vstore   = shared_vstore,
# #             time_offset     = time_offset,
# #             frame_offset    = frame_offset,
# #             source_filename = db_filename,
# #         )
# #         pipeline.run()   # violations accumulate into shared_vstore; store NOT finalized here

# #         # Advance offsets for the next video
# #         time_offset  += duration
# #         frame_offset += total_frames

# #         # Optionally report per-video progress (10 %–90 % band)
# #         if progress_cb:
# #             pct = 10 + int(((idx + 1) / n_videos) * 80)
# #             try:
# #                 progress_cb(pct, f"Analyzed video {idx + 1} of {n_videos}")
# #             except Exception as exc:
# #                 print(f"[Analyzer:{job_id}]  progress_cb error (non-fatal): {exc}")

# #     total_wall_seconds = time.time() - wall_start

# #     # ── Step 5a: Dedup + merge (same steps as ViolationStore.finalize()) ──────
# #     shared_vstore._deduplicate_by_frame()
# #     shared_vstore._merge_by_time_window()

# #     # ── Step 5b: Extract / save frames to local disk ──────────────────────────
# #     # extract_violation_frames() needs the source video file to do a second-pass
# #     # re-read for violations whose annotated_frame was not captured in memory.
# #     # We run it for each temp video file that is still on disk.
# #     for vj in ordered:
# #         tmp_path = tmp_paths[vj.video_id]
# #         if os.path.isfile(tmp_path):
# #             shared_vstore.extract_violation_frames(tmp_path)

# #     # ── Step 5c: Upload frames to S3 and replace local paths with S3 keys ────
# #     # After extract_violation_frames():
# #     #   v.frame_path  = "<analysis_id>/frames/<filename>"   (relative)
# #     #   v.annotated_frame = None  (freed by extract_violation_frames)
# #     #
# #     # We reconstruct the absolute disk path, upload it, and replace
# #     # v.frame_path with the returned S3 key.
# #     for v in shared_vstore._violations:
# #         if v.frame_path:
# #             local_path = _local_frame_path(shared_vstore, v.frame_path)
# #             if os.path.isfile(local_path):
# #                 try:
# #                     s3_key     = upload_frame_from_path(local_path, journey_id)
# #                     v.frame_path = s3_key
# #                 except Exception as exc:
# #                     print(f"[Analyzer:{job_id}]  Frame upload failed ({local_path}): {exc}")
# #             else:
# #                 print(f"[Analyzer:{job_id}]  Frame file not found on disk: {local_path}")
# #         elif v.annotated_frame is not None:
# #             # Safety fallback: frame still in memory (extract step failed for it)
# #             filename = _frame_filename(v)
# #             try:
# #                 s3_key        = upload_frame(v.annotated_frame, journey_id, filename)
# #                 v.frame_path      = s3_key
# #                 v.annotated_frame = None
# #             except Exception as exc:
# #                 print(f"[Analyzer:{job_id}]  In-memory frame upload failed ({filename}): {exc}")

# #     # ── Steps 6–7: Build VideoResult / ViolationResult objects ───────────────
# #     # Map each violation to the video it came from via source_filename.
# #     violations_by_filename: Dict[str, list] = {}
# #     for v in shared_vstore._violations:
# #         violations_by_filename.setdefault(v.source_filename, []).append(v)

# #     video_results: List[VideoResult] = []
# #     for vj in ordered:
# #         meta        = meta_by_id[vj.video_id]
# #         db_filename = os.path.basename(vj.s3_key)
# #         raw_viols   = violations_by_filename.get(db_filename, [])

# #         violation_results = []
# #         for v in raw_viols:
# #             t = int(round(v.timestamp))
# #             violation_results.append(
# #                 ViolationResult(
# #                     violation_type           = _event_type_to_violation_type(v.type),
# #                     severity                 = _severity_for_risk(v.risk_score),
# #                     # ViolationStore stores confidence as 0.0–1.0; API expects 0–100
# #                     confidence               = round(v.confidence * 100, 2),
# #                     risk_score               = v.risk_score,
# #                     timestamp                = v.time_str,
# #                     timestamp_seconds        = t,
# #                     original_video_timestamp = f"{v.source_filename} {v.local_time_str}",
# #                     frame_paths              = [v.frame_path] if v.frame_path else [],
# #                 )
# #             )

# #         video_results.append(
# #             VideoResult(
# #                 video_id           = vj.video_id,
# #                 video_name         = db_filename,
# #                 sequence_no        = vj.sequence_no,
# #                 duration_seconds   = meta["duration_seconds"],
# #                 duration_formatted = meta["duration_formatted"],
# #                 fps                = meta["fps"],
# #                 size_mb            = meta["size_mb"],
# #                 violations         = violation_results,
# #             )
# #         )

# #     return video_results, total_wall_seconds


# # # ── Private helpers ───────────────────────────────────────────────────────────

# # def _frame_filename(v) -> str:
# #     """Derive a JPEG filename from a _Violation object (matches _save_frame logic)."""
# #     distraction   = "_".join(sorted(v.events))
# #     filename_time = v.time_str.replace(":", "-")
# #     return f"{distraction}_{filename_time}.jpg"

# """
# analyzer.py
# ───────────
# Bridges the existing GadgetDetectionPipeline / ViolationStore to the new
# Journey-based workflow.

# Responsibilities
# ────────────────
# 1. Receive a list of (VideoJob, local_tmp_path) pairs.
# 2. Run the existing ML pipeline over each video in sequence_no order,
#    using the shared ViolationStore / time_offset / frame_offset pattern.
# 3. After all videos finish, call the ViolationStore internal steps
#    (dedup → merge → extract frames) without triggering the legacy
#    S3/DB upload path (i.e. we do NOT call finalize()).
# 4. Upload every violation frame to S3 under
#       <folderName>/frames/<filename>.jpg
#    and replace the local path with the returned S3 key.
# 5. Return a list of VideoResult objects ready for the completion callback.

# Changes from previous version
# ──────────────────────────────
# • ViolationResult.timestamp_seconds  — now a float (seconds) instead of
#   the HH:MM:SS string. Spring Boot derives the display string itself.
# • ViolationResult.original_video_timestamp — now a float (local-video seconds)
#   instead of "<filename> HH:MM:SS".
# • ViolationResult.risk_score — now a float (API spec uses float).
# • VideoResult.original_s3_key — new field; populated from VideoJob.s3_key.
# • upload_frame_from_path() is called with the journey's folderName so that
#   frames land under <folderName>/frames/… rather than journeys/<id>/frames/….
# • _severity_for_risk() adds the new "CRITICAL" tier (>= 95) from the docs.
# """

# from __future__ import annotations

# import os
# import time
# from typing import Dict, List, Tuple

# import cv2

# from models import VideoJob, VideoResult, ViolationResult
# from s3_service import upload_frame, upload_frame_from_path

# # Import the existing pipeline and store — unchanged
# from main import GadgetDetectionPipeline
# from utils.violation_store import ViolationStore

# # OUTPUTS_ROOT mirrors the constant inside violation_store.py
# OUTPUTS_ROOT = "outputs"


# # ── Helpers ──────────────────────────────────────────────────────────────────

# def _video_meta(path: str) -> Tuple[float, float, int, float]:
#     """Returns (duration_seconds, fps, total_frames, size_mb)."""
#     cap = cv2.VideoCapture(path)
#     if not cap.isOpened():
#         return 0.0, 25.0, 0, 0.0
#     fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
#     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     cap.release()
#     duration = total / fps if fps > 0 and total > 0 else 0.0
#     size_mb  = round(os.path.getsize(path) / 1_000_000, 2) if os.path.isfile(path) else 0.0
#     return duration, fps, total, size_mb


# def _event_type_to_violation_type(event_type: str) -> str:
#     """
#     Map internal event_type strings → CVVRS API ViolationType enum values.

#     CVVRS types: PHONE_USAGE | DROWSY | DISTRACTION | EATING |
#                  EYES_CLOSED | POSTURE_ALERT | HAND_TO_EAR | SEAT_ABSENCE
#     """
#     return {
#         "phone_use":       "PHONE_USAGE",
#         "seat_absence":    "SEAT_ABSENCE",
#         "drowsy":          "DROWSY",
#         "sleeping":        "DROWSY",
#         "sleeping_absent": "DROWSY",
#         "distraction":     "DISTRACTION",
#         "eating":          "EATING",
#         "eyes_closed":     "EYES_CLOSED",
#         "posture_alert":   "POSTURE_ALERT",
#         "hand_to_ear":     "HAND_TO_EAR",
#     }.get(event_type.lower(), event_type.upper())


# def _severity_for_risk(risk_score: float) -> str:
#     """
#     Map risk score → severity string.

#     Tiers (aligned with CVVRS severity reference):
#         >= 95  → CRITICAL
#         >= 80  → HIGH
#         >= 50  → MEDIUM
#         <  50  → LOW
#     """
#     if risk_score >= 95:
#         return "CRITICAL"
#     if risk_score >= 80:
#         return "HIGH"
#     if risk_score >= 50:
#         return "MEDIUM"
#     return "LOW"


# def _local_frame_path(vstore: ViolationStore, relative_path: str) -> str:
#     """
#     Convert the relative frame path stored by ViolationStore._save_frame()
#     back to an absolute local disk path.

#     _save_frame() returns:  "<analysis_id>/frames/<filename>"
#     The actual disk path is: "outputs/<analysis_id>/frames/<filename>"
#     """
#     filename = os.path.basename(relative_path)
#     return os.path.join(OUTPUTS_ROOT, vstore.analysis_id, "frames", filename)


# # ── Main entry point ──────────────────────────────────────────────────────────

# def analyze_journey(
#     job_id:      str,
#     journey_id:  int,
#     folder_name: str,                  # NEW — e.g. "journeys/1/2026-06-10/JRN-..."
#     video_jobs:  List[VideoJob],
#     tmp_paths:   Dict[int, str],       # video_id → local tmp file path
#     progress_cb  = None,               # optional callable(progress_pct: int, message: str, current_video: int)
# ) -> Tuple[List[VideoResult], float]:
#     """
#     Run the full detection pipeline over all videos in a journey.

#     Parameters
#     ──────────
#     job_id      : RabbitMQ job ID (used for logging only).
#     journey_id  : Journey ID.
#     folder_name : Journey folder prefix used for S3 frame key construction,
#                   e.g. "journeys/1/2026-06-10/JRN-20260610-1-ABC123".
#     video_jobs  : List of VideoJob objects (will be sorted by sequence_no).
#     tmp_paths   : Mapping of video_id → absolute local temp file path.
#     progress_cb : Optional callable(pct, msg, current_video_idx) for updates.

#     Returns
#     ───────
#     (video_results, total_wall_clock_seconds)
#     """
#     # Sort by sequence_no — critical for correct time/frame offset continuity
#     ordered  = sorted(video_jobs, key=lambda v: v.sequence_no)
#     n_videos = len(ordered)

#     # One shared ViolationStore for the whole journey.
#     analysis_id   = f"journey_{journey_id}"
#     shared_vstore = ViolationStore(
#         analysis_id     = analysis_id,
#         train_detail_id = journey_id,
#     )

#     time_offset  = 0.0
#     frame_offset = 0
#     wall_start   = time.time()

#     # Per-video metadata cache (keyed by video_id)
#     meta_by_id: Dict[int, dict] = {}

#     # ── Run the AI pipeline over each video ──────────────────────────────────
#     for idx, vj in enumerate(ordered):
#         tmp_path    = tmp_paths[vj.video_id]
#         db_filename = os.path.basename(vj.s3_key)

#         duration, fps, total_frames, size_mb = _video_meta(tmp_path)
#         meta_by_id[vj.video_id] = {
#             "duration_seconds": duration,
#             "fps":              fps,
#             "size_mb":          size_mb,
#             "total_frames":     total_frames,
#         }

#         print(
#             f"[Analyzer:{job_id}]  [{idx + 1}/{n_videos}]  "
#             f"video_id={vj.video_id}  seq={vj.sequence_no}  "
#             f"file={db_filename}  "
#             f"time_offset={time_offset:.2f}s  frame_offset={frame_offset}"
#         )

#         pipeline = GadgetDetectionPipeline(
#             source          = tmp_path,
#             analysis_id     = analysis_id,
#             train_detail_id = journey_id,
#             save            = False,
#             display         = False,
#             shared_vstore   = shared_vstore,
#             time_offset     = time_offset,
#             frame_offset    = frame_offset,
#             source_filename = db_filename,
#         )
#         pipeline.run()

#         # Advance offsets for the next video
#         time_offset  += duration
#         frame_offset += total_frames

#         # Optionally report per-video progress (10 %–90 % band)
#         if progress_cb:
#             pct = 10 + int(((idx + 1) / n_videos) * 80)
#             try:
#                 progress_cb(pct, f"Analyzed video {idx + 1} of {n_videos}", idx + 1)
#             except Exception as exc:
#                 print(f"[Analyzer:{job_id}]  progress_cb error (non-fatal): {exc}")

#     total_wall_seconds = time.time() - wall_start

#     # ── Dedup + merge ────────────────────────────────────────────────────────
#     shared_vstore._deduplicate_by_frame()
#     shared_vstore._merge_by_time_window()

#     # ── Extract / save frames to local disk ──────────────────────────────────
#     for vj in ordered:
#         tmp_path = tmp_paths[vj.video_id]
#         if os.path.isfile(tmp_path):
#             shared_vstore.extract_violation_frames(tmp_path)

#     # ── Upload frames to S3 and replace local paths with S3 keys ─────────────
#     # Frame S3 key: <folderName>/frames/<filename>.jpg
#     for v in shared_vstore._violations:
#         if v.frame_path:
#             local_path = _local_frame_path(shared_vstore, v.frame_path)
#             if os.path.isfile(local_path):
#                 try:
#                     s3_key     = upload_frame_from_path(local_path, journey_id,
#                                                         folder_name=folder_name)
#                     v.frame_path = s3_key
#                 except Exception as exc:
#                     print(f"[Analyzer:{job_id}]  Frame upload failed ({local_path}): {exc}")
#             else:
#                 print(f"[Analyzer:{job_id}]  Frame file not found on disk: {local_path}")
#         elif v.annotated_frame is not None:
#             filename = _frame_filename(v)
#             try:
#                 s3_key            = upload_frame(v.annotated_frame, journey_id,
#                                                  filename, folder_name=folder_name)
#                 v.frame_path      = s3_key
#                 v.annotated_frame = None
#             except Exception as exc:
#                 print(f"[Analyzer:{job_id}]  In-memory frame upload failed ({filename}): {exc}")

#     # ── Build VideoResult / ViolationResult objects ───────────────────────────
#     # Map each violation to the video it came from via source_filename.
#     violations_by_filename: Dict[str, list] = {}
#     for v in shared_vstore._violations:
#         violations_by_filename.setdefault(v.source_filename, []).append(v)

#     video_results: List[VideoResult] = []
#     for vj in ordered:
#         meta        = meta_by_id[vj.video_id]
#         db_filename = os.path.basename(vj.s3_key)
#         raw_viols   = violations_by_filename.get(db_filename, [])

#         violation_results = []
#         for v in raw_viols:
#             # timestamp_seconds: float seconds from journey start (global)
#             journey_ts = float(v.timestamp) if hasattr(v, "timestamp") else 0.0

#             # original_video_timestamp: float seconds within the source video (local)
#             # v.local_time_str is "HH:MM:SS" — convert to seconds
#             local_ts = _hms_to_seconds(getattr(v, "local_time_str", "0:00:00"))

#             violation_results.append(
#                 ViolationResult(
#                     violation_type           = _event_type_to_violation_type(v.type),
#                     severity                 = _severity_for_risk(v.risk_score),
#                     confidence               = round(v.confidence * 100, 2),
#                     risk_score               = float(v.risk_score),
#                     timestamp_seconds        = journey_ts,
#                     original_video_timestamp = local_ts,
#                     frame_paths              = [v.frame_path] if v.frame_path else [],
#                 )
#             )

#         video_results.append(
#             VideoResult(
#                 video_id         = vj.video_id,
#                 video_name       = db_filename,
#                 sequence_no      = vj.sequence_no,
#                 duration_seconds = meta["duration_seconds"],
#                 original_s3_key  = vj.s3_key,   # NEW — required by API
#                 violations       = violation_results,
#             )
#         )

#     return video_results, total_wall_seconds


# # ── Private helpers ───────────────────────────────────────────────────────────

# def _frame_filename(v) -> str:
#     """Derive a JPEG filename from a _Violation object (matches _save_frame logic)."""
#     distraction   = "_".join(sorted(v.events))
#     filename_time = v.time_str.replace(":", "-")
#     return f"{distraction}_{filename_time}.jpg"


# def _hms_to_seconds(hms: str) -> float:
#     """
#     Convert "H:MM:SS" or "HH:MM:SS" to float seconds.
#     Returns 0.0 on any parse error.
#     """
#     try:
#         parts = hms.strip().split(":")
#         if len(parts) == 3:
#             return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
#         if len(parts) == 2:
#             return int(parts[0]) * 60 + float(parts[1])
#         return float(parts[0])
#     except Exception:
#         return 0.0


"""

analyzer.py

───────────

Bridges the existing GadgetDetectionPipeline / ViolationStore to the new

Journey-based workflow.
 
Responsibilities

────────────────

1. Receive a list of (VideoJob, local_tmp_path) pairs.

2. Run the existing ML pipeline over each video in sequence_no order,

   using the shared ViolationStore / time_offset / frame_offset pattern.

3. After all videos finish, call the ViolationStore internal steps

   (dedup → merge → extract frames) without triggering the legacy

   S3/DB upload path (i.e. we do NOT call finalize()).

4. Upload every violation frame to S3 under
<folderName>/frames/<filename>.jpg

   and replace the local path with the returned S3 key.

5. Return a list of VideoResult objects ready for the completion callback.
 
Fixes from previous version

─────────────────────────────

• Restored _fmt_duration() helper which was accidentally removed.

• VideoResult now populated with durationFormatted, fps, sizeMb — these were

  collected in meta_by_id but never passed to VideoResult constructor.

• meta_by_id now stores duration_formatted key.

"""
 
from __future__ import annotations
 
import os

import time

from typing import Dict, List, Tuple
 
import cv2
 
from models import VideoJob, VideoResult, ViolationResult

from s3_service import upload_frame, upload_frame_from_path
 
# Import the existing pipeline and store — unchanged

from main import GadgetDetectionPipeline

from utils.violation_store import ViolationStore
 
# OUTPUTS_ROOT mirrors the constant inside violation_store.py

OUTPUTS_ROOT = "outputs"
 
 
# ── Helpers ──────────────────────────────────────────────────────────────────
 
def _video_meta(path: str) -> Tuple[float, float, int, float]:

    """Returns (duration_seconds, fps, total_frames, size_mb)."""

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():

        return 0.0, 25.0, 0, 0.0

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    duration = total / fps if fps > 0 and total > 0 else 0.0

    size_mb  = round(os.path.getsize(path) / 1_000_000, 2) if os.path.isfile(path) else 0.0

    return duration, fps, total, size_mb
 
 
def _fmt_duration(seconds: float) -> str:

    """FIX: restored — converts float seconds to 'H:MM:SS' string."""

    t  = int(seconds)

    hh = t // 3600

    mm = (t % 3600) // 60

    ss = t % 60

    return f"{hh}:{mm:02d}:{ss:02d}"
 
 
def _event_type_to_violation_type(event_type: str) -> str:

    """

    Map internal event_type strings → CVVRS API ViolationType enum values.
 
    CVVRS types: PHONE_USAGE | DROWSY | DISTRACTION | EATING |

                 EYES_CLOSED | POSTURE_ALERT | HAND_TO_EAR | SEAT_ABSENCE

    """

    return {

        "phone_use":       "PHONE_USAGE",

        "seat_absence":    "SEAT_ABSENCE",

        "drowsy":          "DROWSY",

        "sleeping":        "DROWSY",

        "sleeping_absent": "DROWSY",

        "distraction":     "DISTRACTION",

        "eating":          "EATING",

        "eyes_closed":     "EYES_CLOSED",

        "posture_alert":   "POSTURE_ALERT",

        "hand_to_ear":     "HAND_TO_EAR",

    }.get(event_type.lower(), event_type.upper())
 
 
def _severity_for_risk(risk_score: float) -> str:

    """

    Map risk score → severity string.
 
    Tiers (aligned with CVVRS severity reference):
>= 95  → CRITICAL
>= 80  → HIGH
>= 50  → MEDIUM
<  50  → LOW

    """

    if risk_score >= 95:

        return "CRITICAL"

    if risk_score >= 80:

        return "HIGH"

    if risk_score >= 50:

        return "MEDIUM"

    return "LOW"
 
 
def _local_frame_path(vstore: ViolationStore, relative_path: str) -> str:

    """

    Convert the relative frame path stored by ViolationStore._save_frame()

    back to an absolute local disk path.
 
    _save_frame() returns:  "<analysis_id>/frames/<filename>"

    The actual disk path is: "outputs/<analysis_id>/frames/<filename>"

    """

    filename = os.path.basename(relative_path)

    return os.path.join(OUTPUTS_ROOT, vstore.analysis_id, "frames", filename)
 
 
def _hms_to_seconds(hms: str) -> float:

    """

    Convert "H:MM:SS" or "HH:MM:SS" to float seconds.

    Returns 0.0 on any parse error.

    """

    try:

        parts = hms.strip().split(":")

        if len(parts) == 3:

            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])

        if len(parts) == 2:

            return int(parts[0]) * 60 + float(parts[1])

        return float(parts[0])

    except Exception:

        return 0.0
 
 
# ── Main entry point ──────────────────────────────────────────────────────────
 
def analyze_journey(

    job_id:      str,

    journey_id:  int,

    folder_name: str,                  # e.g. "journeys/1/2026-06-10/JRN-..."

    video_jobs:  List[VideoJob],

    tmp_paths:   Dict[int, str],       # video_id → local tmp file path

    progress_cb  = None,               # optional callable(pct, msg, current_video)

) -> Tuple[List[VideoResult], float]:

    """

    Run the full detection pipeline over all videos in a journey.
 
    Parameters

    ──────────

    job_id      : RabbitMQ job ID (used for logging only).

    journey_id  : Journey ID.

    folder_name : Journey folder prefix used for S3 frame key construction,

                  e.g. "journeys/1/2026-06-10/JRN-20260610-1-ABC123".

    video_jobs  : List of VideoJob objects (will be sorted by sequence_no).

    tmp_paths   : Mapping of video_id → absolute local temp file path.

    progress_cb : Optional callable(pct, msg, current_video_idx) for updates.
 
    Returns

    ───────

    (video_results, total_wall_clock_seconds)

    """

    # Sort by sequence_no — critical for correct time/frame offset continuity

    ordered  = sorted(video_jobs, key=lambda v: v.sequence_no)

    n_videos = len(ordered)
 
    # One shared ViolationStore for the whole journey.

    analysis_id   = f"journey_{journey_id}"

    shared_vstore = ViolationStore(

        analysis_id     = analysis_id,

        train_detail_id = journey_id,

    )
 
    time_offset  = 0.0

    frame_offset = 0

    wall_start   = time.time()
 
    # Per-video metadata cache (keyed by video_id)

    meta_by_id: Dict[int, dict] = {}
 
    # ── Run the AI pipeline over each video ──────────────────────────────────

    for idx, vj in enumerate(ordered):

        tmp_path    = tmp_paths[vj.video_id]

        db_filename = os.path.basename(vj.s3_key)
 
        duration, fps, total_frames, size_mb = _video_meta(tmp_path)
 
        # FIX: store duration_formatted so VideoResult can use it

        meta_by_id[vj.video_id] = {

            "duration_seconds":   duration,

            "duration_formatted": _fmt_duration(duration),   # FIX

            "fps":                fps,                        # FIX

            "size_mb":            size_mb,                    # FIX

            "total_frames":       total_frames,

        }
 
        print(

            f"[Analyzer:{job_id}]  [{idx + 1}/{n_videos}]  "

            f"video_id={vj.video_id}  seq={vj.sequence_no}  "

            f"file={db_filename}  "

            f"time_offset={time_offset:.2f}s  frame_offset={frame_offset}"

        )
 
        pipeline = GadgetDetectionPipeline(

            source          = tmp_path,

            analysis_id     = analysis_id,

            train_detail_id = journey_id,

            save            = False,

            display         = False,

            shared_vstore   = shared_vstore,

            time_offset     = time_offset,

            frame_offset    = frame_offset,

            source_filename = db_filename,

        )

        pipeline.run()
 
        # Advance offsets for the next video

        time_offset  += duration

        frame_offset += total_frames
 
        # Report per-video progress (10%–90% band)

        if progress_cb:

            pct = 10 + int(((idx + 1) / n_videos) * 80)

            try:

                progress_cb(pct, f"Analyzed video {idx + 1} of {n_videos}", idx + 1)

            except Exception as exc:

                print(f"[Analyzer:{job_id}]  progress_cb error (non-fatal): {exc}")
 
    total_wall_seconds = time.time() - wall_start
 
    # ── Dedup + merge ────────────────────────────────────────────────────────

    shared_vstore._deduplicate_by_frame()

    shared_vstore._merge_by_time_window()
 
    # ── Extract / save frames to local disk ──────────────────────────────────

    for vj in ordered:

        tmp_path = tmp_paths[vj.video_id]

        if os.path.isfile(tmp_path):

            shared_vstore.extract_violation_frames(tmp_path)
 
    # ── Upload frames to S3 and replace local paths with S3 keys ─────────────

    # Frame S3 key: <folderName>/frames/<filename>.jpg

    for v in shared_vstore._violations:

        if v.frame_path:

            local_path = _local_frame_path(shared_vstore, v.frame_path)

            if os.path.isfile(local_path):

                try:

                    s3_key       = upload_frame_from_path(local_path, journey_id,

                                                          folder_name=folder_name)

                    v.frame_path = s3_key

                except Exception as exc:

                    print(f"[Analyzer:{job_id}]  Frame upload failed ({local_path}): {exc}")

            else:

                print(f"[Analyzer:{job_id}]  Frame file not found on disk: {local_path}")

        elif v.annotated_frame is not None:

            filename = _frame_filename(v)

            try:

                s3_key            = upload_frame(v.annotated_frame, journey_id,

                                                 filename, folder_name=folder_name)

                v.frame_path      = s3_key

                v.annotated_frame = None

            except Exception as exc:

                print(f"[Analyzer:{job_id}]  In-memory frame upload failed ({filename}): {exc}")
 
    # ── Build VideoResult / ViolationResult objects ───────────────────────────

    violations_by_filename: Dict[str, list] = {}

    for v in shared_vstore._violations:

        violations_by_filename.setdefault(v.source_filename, []).append(v)
 
    video_results: List[VideoResult] = []

    for vj in ordered:

        meta        = meta_by_id[vj.video_id]

        db_filename = os.path.basename(vj.s3_key)

        raw_viols   = violations_by_filename.get(db_filename, [])
 
        violation_results = []

        for v in raw_viols:

            # timestamp_seconds: float seconds from journey start (global)

            journey_ts = float(v.timestamp) if hasattr(v, "timestamp") else 0.0
 
            # original_video_timestamp: float seconds within the source video (local)

            local_ts = _hms_to_seconds(getattr(v, "local_time_str", "0:00:00"))
 
            violation_results.append(

                ViolationResult(

                    violation_type           = _event_type_to_violation_type(v.type),

                    severity                 = _severity_for_risk(v.risk_score),

                    confidence               = round(v.confidence * 100, 2),

                    risk_score               = float(v.risk_score),

                    timestamp_seconds        = journey_ts,

                    original_video_timestamp = local_ts,

                    frame_paths              = [v.frame_path] if v.frame_path else [],

                )

            )
 
        # FIX: pass durationFormatted, fps, sizeMb to VideoResult

        video_results.append(

            VideoResult(

                video_id           = vj.video_id,

                video_name         = db_filename,

                sequence_no        = vj.sequence_no,

                duration_seconds   = meta["duration_seconds"],

                duration_formatted = meta["duration_formatted"],   # FIX

                fps                = meta["fps"],                   # FIX

                size_mb            = meta["size_mb"],               # FIX

                original_s3_key    = vj.s3_key,

                violations         = violation_results,

            )

        )
 
    return video_results, total_wall_seconds
 
 
# ── Private helpers ───────────────────────────────────────────────────────────
 
def _frame_filename(v) -> str:

    """Derive a JPEG filename from a _Violation object (matches _save_frame logic)."""

    distraction   = "_".join(sorted(v.events))

    filename_time = v.time_str.replace(":", "-")

    return f"{distraction}_{filename_time}.jpg"
 