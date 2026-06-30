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


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    """Converts float seconds to 'H:MM:SS' string."""
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
    rabbit_connection = None,
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

    # Track which videos were skipped due to a per-video error so we can
    # still include them in the VideoResult list with zero violations.
    failed_video_ids: set = set()

    # ── Run the AI pipeline over each video ──────────────────────────────────
    for idx, vj in enumerate(ordered):
        tmp_path    = tmp_paths[vj.video_id]
        db_filename = os.path.basename(vj.s3_key)

        duration, fps, total_frames, size_mb = _video_meta(tmp_path)

        meta_by_id[vj.video_id] = {
            "duration_seconds":   duration,
            "duration_formatted": _fmt_duration(duration),
            "fps":                fps,
            "size_mb":            size_mb,
            "total_frames":       total_frames,
        }

        print(
            f"[Analyzer:{job_id}]  [{idx + 1}/{n_videos}]  "
            f"video_id={vj.video_id}  seq={vj.sequence_no}  "
            f"file={db_filename}  "
            f"time_offset={time_offset:.2f}s  frame_offset={frame_offset}"
        )

        pipeline = GadgetDetectionPipeline(
            source           = tmp_path,
            analysis_id      = analysis_id,
            train_detail_id  = journey_id,
            save             = False,
            display          = False,
            shared_vstore    = shared_vstore,
            time_offset      = time_offset,
            frame_offset     = frame_offset,
            source_filename  = db_filename,
        )

        # ── FIX — Issue #2: per-video error isolation ─────────────────────────
        # Wrap pipeline.run() so a failure for one video is logged and skipped,
        # allowing the remaining videos in the journey to be processed normally.
        #
        # We catch BaseException (not just Exception) because
        # GadgetDetectionPipeline.run() calls sys.exit(1) when it cannot open
        # a video file — sys.exit raises SystemExit which is a BaseException
        # subclass and would otherwise escape an "except Exception" guard.
        #
        # KeyboardInterrupt is explicitly re-raised so Ctrl-C still works.
        # SystemExit is also re-raised so a genuine fatal signal propagates to
        # the consumer's outer handler (nack + DLQ) instead of being swallowed.
        try:
            pipeline.run()
        except KeyboardInterrupt:
            raise
        except SystemExit:
            raise
        except BaseException as video_exc:
            print(
                f"[Analyzer:{job_id}]  WARNING — video_id={vj.video_id} "
                f"seq={vj.sequence_no} FAILED (skipping): {video_exc!r}"
            )
            failed_video_ids.add(vj.video_id)
            # Fall through: advance offsets and continue with the next video.

        # Advance offsets for the next video regardless of success/failure.
        # Using the metadata we already captured keeps offsets consistent for
        # subsequent videos even when this one's pipeline run was skipped.
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

    # ── Dedup + merge ─────────────────────────────────────────────────────────
    shared_vstore._deduplicate_by_frame()
    shared_vstore._merge_by_time_window()

    # ── Extract frames + upload to S3 ───────────────────────────────────────
    #
    # BUG FIX — Wrong frames in violation evidence
    # ─────────────────────────────────────────────
    # The old code called:
    #   for vj in ordered:
    #       shared_vstore.extract_violation_frames(tmp_path)
    #
    # extract_violation_frames() iterates ALL violations in the shared store
    # and seeks to v.frame_index (a GLOBAL frame number across all videos).
    # When called with video_2's path, it seeks to e.g. frame 4500 in video_2
    # — but frame 4500 is from video_1 (frame_index was global, not local).
    # Result: every violation gets a frame from the WRONG video.
    #
    # Fix: build a per-filename path map, then for each violation use its
    # source_filename to open the correct video and seek using local_time_str
    # (seconds within that specific video file) — which is correct regardless
    # of how many videos precede it in the journey.

    # Map db_filename → local tmp path
    path_by_filename: Dict[str, str] = {
        os.path.basename(vj.s3_key): tmp_paths[vj.video_id]
        for vj in ordered
    }

    for v in shared_vstore._violations:
        # ── Case 1: annotated frame already in memory — upload directly ──────
        if v.annotated_frame is not None:
            filename = _frame_filename(v)
            try:
                s3_key            = upload_frame(v.annotated_frame, journey_id,
                                                 filename, folder_name=folder_name)
                v.frame_path      = s3_key
                v.annotated_frame = None
            except Exception as exc:
                print(f"[Analyzer:{job_id}]  In-memory frame upload failed ({filename}): {exc}")
            continue

        # ── Case 2: no frame yet — extract from the correct source video ─────
        if v.frame_path:
            # Already extracted to disk by a previous extract_violation_frames()
            # call (e.g. standalone mode). Upload it.
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
            continue

        # ── Case 3: no frame at all — seek into the correct source video ─────
        src_file = getattr(v, "source_filename", "")
        tmp_path = path_by_filename.get(src_file, "")
        if not tmp_path or not os.path.isfile(tmp_path):
            print(f"[Analyzer:{job_id}]  Cannot find source video for violation "
                  f"(source_filename={src_file!r}) — skipping frame extraction")
            continue

        # Use local_time_str (seconds within this video) to seek correctly.
        # v.local_time_str is set by record_violation(local_video_time=...) and
        # stored as an "H:MM:SS" string. Convert back to seconds for cv2 seek.
        local_secs = _hms_to_seconds(getattr(v, "local_time_str", "0:00:00"))

        cap = cv2.VideoCapture(tmp_path)
        frame_img = None
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_MSEC, local_secs * 1000.0)
            ret, frame_img = cap.read()
            if not ret:
                # Fallback: try seeking by fraction of duration
                fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                local_frame = int(local_secs * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(local_frame, max(0, total - 1)))
                ret, frame_img = cap.read()
        cap.release()

        if frame_img is None:
            print(f"[Analyzer:{job_id}]  Frame seek failed for {src_file} "
                  f"@ local_time={local_secs:.2f}s — skipping")
            continue

        filename = _frame_filename(v)
        try:
            s3_key       = upload_frame(frame_img, journey_id,
                                        filename, folder_name=folder_name)
            v.frame_path = s3_key
        except Exception as exc:
            print(f"[Analyzer:{job_id}]  Extracted frame upload failed ({filename}): {exc}")

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
                    timestamp_seconds        = _fmt_duration(journey_ts),
                    original_video_timestamp = _fmt_duration(local_ts),
                    frame_paths              = [v.frame_path] if v.frame_path else [],
                )
            )

        video_results.append(
            VideoResult(
                video_id           = vj.video_id,
                video_name         = db_filename,
                sequence_no        = vj.sequence_no,
                duration_seconds   = meta["duration_seconds"],
                duration_formatted = meta["duration_formatted"],
                fps                = meta["fps"],
                size_mb            = meta["size_mb"],
                original_s3_key    = vj.s3_key,
                violations         = violation_results,
            )
        )

    if failed_video_ids:
        print(
            f"[Analyzer:{job_id}]  Journey analysis complete with "
            f"{len(failed_video_ids)} skipped video(s): {sorted(failed_video_ids)}"
        )

    return video_results, total_wall_seconds


# ── Private helpers ───────────────────────────────────────────────────────────

def _frame_filename(v) -> str:
    """Derive a JPEG filename from a _Violation object (matches _save_frame logic)."""
    distraction   = "_".join(sorted(v.events))
    filename_time = v.time_str.replace(":", "-")
    return f"{distraction}_{filename_time}.jpg"