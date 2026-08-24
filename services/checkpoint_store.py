"""
checkpoint_store.py
────────────────────
Durable, on-disk per-video checkpointing for a journey.

Why this exists
────────────────
Before this module, a video's "is it done" bookkeeping lived only in
in-memory `multiprocessing.Queue` objects (events_q / result_q, see
journey_runner.py + consumer.py). That is enough to survive a single GPU
*worker* process crashing (the parent process is still alive and still
holds whatever it already drained off the queue), but it is NOT durable
against the parent/consumer process itself being restarted (deploy,
OOM-kill at the OS level, host reboot) — every in-flight journey's
already-completed videos would have to be reconstructed from nothing.

This module adds a small, dependency-free JSON checkpoint file per job,
written atomically (write-temp-then-rename) after every video state
transition. It is intentionally simple — local disk, one JSON file per
job_id — so it has no new infrastructure dependency. If/when this needs
to survive the LOCAL DISK also disappearing (e.g. multi-host worker
pool with ephemeral local storage), the same `save_checkpoint`/
`load_checkpoints` calls can be pointed at S3 or a database without any
caller-side changes — every call site already goes through this module.

Video lifecycle (Fix 2)
────────────────────────
    PENDING → PROCESSING → FINALIZING → COMPLETED
                  │
                  └──────────────────→ FAILED / INTERRUPTED

  PENDING     — video is part of the journey but hasn't started yet.
  PROCESSING  — pipeline.run() (inference) is in progress for this video.
  FINALIZING  — inference finished; evidence-frame upload / S3 persistence
                / result construction is in progress. A video must reach
                COMPLETED only *after* this step, never before — see Fix 5.
  COMPLETED   — inference + evidence persistence + result build are all
                durable. Safe to never re-process, even after a crash.
  FAILED      — this video failed for a video-specific reason (bad file,
                decode error, etc.) and will not be retried further.
  INTERRUPTED — processing started but was cut short by a worker crash,
                OOM, or watchdog-detected stall. May be retried from
                scratch (frame-level resume is intentionally out of scope
                — see Fix 7) or ultimately marked FAILED once retries are
                exhausted.

Each checkpoint record contains at minimum the fields called out in
Fix 3: journey_id, job_id, video_id, sequence_no, status,
processed_frame_count, total_frames, violations, evidence_frame_paths,
processing_time, last_processed_timestamp.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from typing import Dict, List, Optional

CHECKPOINT_ROOT = os.environ.get("CHECKPOINT_ROOT", os.path.join("outputs", "checkpoints"))

# ── Video states (Fix 2) ────────────────────────────────────────────────────
PENDING     = "PENDING"
PROCESSING  = "PROCESSING"
FINALIZING  = "FINALIZING"
COMPLETED   = "COMPLETED"
FAILED      = "FAILED"
INTERRUPTED = "INTERRUPTED"

_VALID_STATES = {PENDING, PROCESSING, FINALIZING, COMPLETED, FAILED, INTERRUPTED}

# One lock per process is enough — every writer for a given job_id also
# runs inside the same process (the GPU worker handling that journey), so
# this only needs to protect against the rare case of the read-and-modify
# happening from more than one thread inside that same process.
_lock = threading.Lock()


def _path_for(job_id: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(job_id))
    return os.path.join(CHECKPOINT_ROOT, f"{safe}.json")


def _atomic_write(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".ckpt_", dir=os.path.dirname(path) or ".")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)  # atomic on POSIX and Windows
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


def load_checkpoints(job_id: str) -> Dict[str, dict]:
    """Returns {video_id_str: record} for this job, or {} if none exist."""
    path = _path_for(job_id)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def init_journey(job_id: str, journey_id: int, video_ids: List[int],
                  sequence_by_id: Optional[Dict[int, int]] = None) -> None:
    """
    Called once, before the first video of a journey starts, so every
    video has a PENDING record from the very beginning (Fix 12 also
    benefits: total/pending/completed/failed can always be derived from
    this file, never from a separately-maintained counter).
    """
    sequence_by_id = sequence_by_id or {}
    with _lock:
        data = load_checkpoints(job_id)
        for vid in video_ids:
            key = str(vid)
            if key not in data:
                data[key] = {
                    "journeyId":             journey_id,
                    "jobId":                 job_id,
                    "videoId":               vid,
                    "sequenceNo":            sequence_by_id.get(vid),
                    "status":                PENDING,
                    "processedFrameCount":   0,
                    "totalFrames":           0,
                    "violations":            [],
                    "evidenceFramePaths":    [],
                    "processingTime":        None,
                    "lastProcessedTimestamp": time.time(),
                }
        _atomic_write(_path_for(job_id), data)


def save_checkpoint(
    job_id: str,
    journey_id: int,
    video_id: int,
    sequence_no: Optional[int],
    status: str,
    *,
    processed_frame_count: int = 0,
    total_frames: int = 0,
    violations: Optional[List[dict]] = None,
    evidence_frame_paths: Optional[List[str]] = None,
    processing_time: Optional[float] = None,
) -> None:
    """
    Durable checkpoint write for one video's state transition (Fix 3).
    Safe to call frequently — each call is a full atomic rewrite of the
    (small, per-job, not per-journey-wide-history) checkpoint file.
    """
    if status not in _VALID_STATES:
        raise ValueError(f"Invalid checkpoint status: {status!r}")
    with _lock:
        data = load_checkpoints(job_id)
        key = str(video_id)
        record = data.get(key, {})
        record.update({
            "journeyId":              journey_id,
            "jobId":                  job_id,
            "videoId":                video_id,
            "sequenceNo":             sequence_no if sequence_no is not None else record.get("sequenceNo"),
            "status":                 status,
            "processedFrameCount":    processed_frame_count,
            "totalFrames":            total_frames,
            "lastProcessedTimestamp": time.time(),
        })
        if violations is not None:
            record["violations"] = violations
        if evidence_frame_paths is not None:
            record["evidenceFramePaths"] = evidence_frame_paths
        if processing_time is not None:
            record["processingTime"] = processing_time
        data[key] = record
        _atomic_write(_path_for(job_id), data)


def summarize(job_id: str) -> Dict[str, int]:
    """
    Derives journey counters PURELY from the durable per-video records —
    never from an independently incremented counter (Fix 12: "Do not
    maintain independent counters that can become inconsistent").
    """
    data = load_checkpoints(job_id)
    counts = {s: 0 for s in _VALID_STATES}
    for rec in data.values():
        counts[rec.get("status", PENDING)] = counts.get(rec.get("status", PENDING), 0) + 1
    counts["TOTAL"] = len(data)
    return counts


def clear(job_id: str) -> None:
    """Best-effort cleanup once a journey's completion callback has been
    delivered successfully — the checkpoint file is only needed while
    recovery is possible."""
    try:
        path = _path_for(job_id)
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass