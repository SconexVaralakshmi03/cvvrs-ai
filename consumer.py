"""
consumer.py
───────────
RabbitMQ consumer for the Journey-based analysis workflow.

Processing flow
───────────────
 1. Consume message from 'dev.dev.dev.analysis.jobs'.
 2. Idempotency check — if job already completed, ACK and skip.
 3. [KEEPALIVE ON] Start keepalive thread — covers download + analysis.
 4. Download all videos from S3.
 5. Send PROCESSING progress (10%).
 6. Analyse videos via the existing AI pipeline (analyzer.py).
 7. VideoResult / ViolationResult objects built inside analyzer.py.
 8. Calculate overall processing time (wall-clock ms).
 9. Send completion callback.
10. [KEEPALIVE OFF] Stop keepalive thread.
11. Acknowledge RabbitMQ message.

On any unhandled exception:
    • Log the full traceback.
    • Send a failure callback to Spring Boot.
    • Nack the message (no requeue → dead-letter exchange).

═══════════════════════════════════════════════════════════════════
ROOT CAUSE ANALYSIS (from 8-video test run)
═══════════════════════════════════════════════════════════════════

What happened
─────────────
1. 8 videos downloaded sequentially over 7 min 53 s.
   During this entire window the main thread is inside boto3
   network I/O.  pika's I/O loop receives ZERO CPU.  No heartbeat
   frames are sent to the broker.

2. The broker's NAT device/firewall kills idle TCP connections
   after ~60-300 s of silence.  By the time the download finishes
   and analysis starts, the TCP socket is already dead.

3. pipeline.run() runs fine (it doesn't need pika).  All 8 videos
   process, 57 frames upload, completion callback succeeds.

4. basic_ack() → ChannelWrongStateError: Channel is closed.
   send_failed() is called → Spring Boot marks job FAILED.

5. On reconnect, broker redelivers (redelivered=True).
   Idempotency check → HTTP 500 (not implemented) → job re-runs.
   The same 8 videos are downloaded and processed AGAIN.

Root cause
──────────
The keepalive was started AFTER the downloads — but the downloads
were what killed the connection.  The keepalive must cover the
ENTIRE job duration: download phase + analysis phase.

Fix
───
_AnalysisKeepalive is now entered BEFORE the download loop.
It covers: download → analysis → frame upload → callback.
The thread is stopped only AFTER completion callback returns,
just before basic_ack().

Thread-safety contract
──────────────────────
• During download: main thread is in boto3/network I/O (no pika).
  Keepalive thread holds pika_lock and calls process_data_events.
• During analysis: main thread is in pipeline.run() (no pika).
  Keepalive thread holds pika_lock and calls process_data_events.
• During ACK/NACK: keepalive is stopped (thread.join() completed).
  Main thread holds pika_lock and calls basic_ack/basic_nack.
These windows never overlap → no concurrent pika access → safe.

For 40 videos × 15 min
──────────────────────
• Processing speed from log: ~7.4× realtime → 15-min video ≈ 122 s
• 40 videos total analysis: ~81 min
• Keepalive fires every 15 s → 4 heartbeats/min throughout
• Broker heartbeat interval: 60 s → pika sends every 30 s at idle
• Total job duration: downloads + ~81 min analysis
• Connection will stay alive for the entire duration ✅

═══════════════════════════════════════════════════════════════════
FATAL-INTERRUPTION HANDLING (partial-journey recovery)
═══════════════════════════════════════════════════════════════════
See `_finalize_on_interruption()` and the restructured exception
handling in `_handle_job()` below for the policy that governs what
happens when a fatal interruption (RabbitMQ connection loss, Ctrl+C,
OpenCV/native crash, OOM, unexpected exception, etc.) hits a journey
that is already in progress:

  • Zero videos succeeded so far  → existing FAILED-callback path
    (send_failed + NACK) is unchanged.
  • One or more videos already succeeded → we do NOT call the failed
    callback. Instead we wait up to RECOVERY_WAIT_SECONDS (2 min,
    used as a bounded retry budget for the completion callback),
    mark every still-unprocessed video as failed, send the
    COMPLETED_WITH_ERRORS completion callback (which already carries
    both the successful VideoResults and the failed/unprocessed
    video details — this is the same payload shape used by the
    existing subprocess-crash path), and finalize (ACK) the message
    so the worker moves on to the next queued journey. Resource
    cleanup (_cleanup(tmp_paths), resource_manager.cleanup_after_failure)
    still runs unconditionally on this path.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import queue as _queue
import tempfile
import threading
import time
import traceback
from datetime import datetime
from functools import partial
from typing import Dict

import pika
from dotenv import load_dotenv

from pipeline.analyzer import analyze_journey
from pipeline.worker_pool import worker_pool, GPU_WORKERS
from services.callback_client import (
    send_completed,
    send_failed,
    send_video_failed,
    send_progress,
    set_base_url,
    check_job_completed,
    compute_journey_status,
    try_start_job,
    finish_job,
)
from schemas.models import AnalysisJobMessage, CompletionPayload, VideoResult, ViolationResult
from services.s3_service import download_video, upload_text_log, upload_json_result
from logging_utils.journey_log import build_journey_log_text
from services.resource_manager import resource_manager, memory_monitor

# ── Config / credentials ──────────────────────────────────────────────────────
_ENV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "config", "credentials.env",
)
load_dotenv(_ENV_PATH)

RABBITMQ_URL       = os.environ.get("RABBITMQ_URL",              "amqp://guest:guest@localhost:5672/")
QUEUE_NAME         = os.environ.get("ANALYSIS_QUEUE",             "dev.analysis.jobs")
EXCHANGE_NAME      = os.environ.get("ANALYSIS_EXCHANGE",          "dev.analysis.exchange")
ROUTING_KEY        = os.environ.get("ANALYSIS_ROUTING",           "dev.analysis.jobs.created")
# Defaults to GPU_WORKERS: with N persistent GPU workers we want the broker
# to have up to N unacked messages in flight so all N workers can be busy
# at once (see worker_pool.py). Override explicitly via RABBITMQ_PREFETCH
# if you ever want the broker-side buffer to differ from the worker count.
PREFETCH_COUNT     = int(os.environ.get("RABBITMQ_PREFETCH",      str(GPU_WORKERS)))
RECONNECT_DELAY    = int(os.environ.get("RECONNECT_DELAY_SECONDS", "5"))

# AMQP heartbeat interval negotiated with the broker (seconds).
# pika sends a frame every interval/2 automatically while start_consuming()
# is idle.  60 s keeps most NAT devices happy.
HEARTBEAT_INTERVAL = int(os.environ.get("RABBITMQ_HEARTBEAT",    "60"))

# How often the keepalive thread pokes pika during blocking work.
# Must be < HEARTBEAT_INTERVAL / 2 (i.e. < 30 s).
KEEPALIVE_INTERVAL = int(os.environ.get("RABBITMQ_KEEPALIVE",    "15"))

# How long (seconds) we are willing to keep retrying the COMPLETED_WITH_ERRORS
# completion callback after a fatal interruption that struck a journey which
# already had at least one successfully-processed video. This is a bounded
# "wait for recovery" window, not a re-run of any video — the videos that
# already succeeded are never reprocessed.
RECOVERY_WAIT_SECONDS  = int(os.environ.get("RECOVERY_WAIT_SECONDS",  "120"))
RECOVERY_RETRY_SECONDS = int(os.environ.get("RECOVERY_RETRY_SECONDS", "5"))

# ── Subprocess isolation config ────────────────────────────────────────────
# Each journey's analysis runs in its own child process (see journey_runner.py)
# so a native OpenCV crash (OOM / access violation / segfault) kills only
# that child — never the consumer process itself, never RabbitMQ's
# connection, never other in-flight jobs.
#
# ═══════════════════════════════════════════════════════════════════════════
# Fix 1 — PROGRESS-BASED WATCHDOG (replaces the old fixed/scaled journey
# timeout as the PRIMARY mechanism for detecting a hung video).
# ═══════════════════════════════════════════════════════════════════════════
#
# The old design computed a single wall-clock budget for the WHOLE journey
# (base + per_video_budget × video_count) and killed the entire worker the
# moment total elapsed time exceeded it — even if every video was actually
# still making healthy progress. A journey with one genuinely slow (but
# healthy) video, or more videos than the constant the budget was calibrated
# for, could get killed while doing perfectly good work.
#
# WATCHDOG_STUCK_SECONDS is now the primary signal: a video is only ever
# considered stuck when it has produced NO measurable progress (no frame
# advance, no heartbeat — see main.py's progress_cb / analyzer.py's
# video_progress_cb) for this many consecutive seconds. A video that keeps
# advancing — 20 minutes, 1 hour, 2 hours — is left alone indefinitely.
WATCHDOG_STUCK_SECONDS = float(os.environ.get("WATCHDOG_STUCK_SECONDS", "600"))  # 10 min

# How many times a video that's detected as stuck is retried FROM SCRATCH
# (Fix 7 — frame-level resume is out of scope; a full re-run of just that
# video is the first implementation) before it is given up on and marked
# FAILED so the rest of the journey can continue.
WATCHDOG_MAX_VIDEO_RETRIES = int(os.environ.get("WATCHDOG_MAX_VIDEO_RETRIES", "1"))

# ABSOLUTE_JOURNEY_CEILING_SECONDS is an OPT-IN, generous last-resort safety
# valve only — 0 (the default) disables it entirely. Per the fix
# requirement, wall-clock elapsed time must never be the PRIMARY reason a
# healthy, still-progressing journey gets killed; this exists only to catch
# truly pathological situations (e.g. the watchdog/progress-reporting path
# itself is broken) an operator may want an outer bound on. Left at 0
# unless explicitly configured.
ABSOLUTE_JOURNEY_CEILING_SECONDS = float(os.environ.get("ABSOLUTE_JOURNEY_CEILING_SECONDS", "0"))

# How often the parent polls the child's events_q for per-video progress
# while waiting for it to finish (also doubles as the keepalive cadence
# check — process_data_events still only happens on the separate keepalive
# thread, this loop just drains progress events).
EVENTS_POLL_INTERVAL_SECONDS = float(os.environ.get("EVENTS_POLL_INTERVAL_SECONDS", "1.0"))

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger("consumer")

import logging_utils.job_logging as job_logging
job_logging.install()


def _upload_job_console_log(job_id: str, folder_name: str) -> None:
    """
    Close out this journey's captured logs and upload them to S3 next to
    the existing structured <jobId>.txt report — as TWO SEPARATE files,
    never combined:

      <jobId>_console.txt     — terminal/pipeline output only (download,
                                 worker-dispatch, callback, pipeline
                                 banners/progress) — NO violation lines.
      <jobId>_violations.txt  — violation-event lines ONLY, e.g.
                                 "[00:00:21] One of the pilots is using a
                                 mobile phone  [CRITICAL]" — nothing else.

    Best-effort — never raises, since losing either diagnostic log must
    never fail the journey itself.
    """
    try:
        job_logging.finish_job(job_id)  # flush + close the consumer-side handle
        terminal_text, violations_text = job_logging.read_and_clear(job_id)

        if terminal_text.strip():
            upload_text_log(
                text        = terminal_text,
                folder_name = folder_name,
                filename    = f"{job_id}_console.txt",
            )
            log.info("[Job %s]  Console (terminal) log uploaded to S3 (%d bytes).",
                     job_id, len(terminal_text))

        if violations_text.strip():
            upload_text_log(
                text        = violations_text,
                folder_name = folder_name,
                filename    = f"{job_id}_violations.txt",
            )
            log.info("[Job %s]  Violations log uploaded to S3 (%d bytes).",
                     job_id, len(violations_text))
    except Exception as exc:
        log.warning("[Job %s]  Console/violations log upload failed (non-fatal): %s",
                    job_id, exc)


# ── Job-duration keepalive ────────────────────────────────────────────────────

class _JobKeepalive:
    """
    HISTORICAL NOTE / WHY THIS IS NOW A NO-OP:

    This used to run a background thread calling connection.
    process_data_events() every KEEPALIVE_INTERVAL seconds, because the
    OLD architecture had _on_message() call _handle_job() DIRECTLY on the
    single pika I/O thread — so that thread was fully blocked for the
    entire job (download + analysis + callbacks), unable to pump pika's
    own I/O loop or answer heartbeats, hence needing a separate thread to
    do it instead. This class's own original docstring said so
    explicitly: "During the job body... the main thread makes NO pika
    calls, so the keepalive thread runs uncontested."

    That assumption is FALSE now. Since the worker-pool refactor,
    _on_message() spawns a thread and returns immediately (see
    _on_message below), so the main thread stays inside pika's own
    channel.start_consuming() loop continuously — which ALREADY pumps
    I/O and answers heartbeats by itself, for every in-flight journey,
    with no help needed.

    Worse: running this thread's process_data_events() concurrently
    with the main thread's own continuous internal pika activity — now
    genuinely concurrent, not "uncontested" — is exactly what was
    producing the repeated
        IndexError: pop from an empty deque
    connection crashes: two threads touching the same BlockingConnection's
    internal transport buffers at once. pika_lock only ever serialized
    THIS thread against ACK/NACK calls, never against the main thread's
    own internal socket I/O inside start_consuming(), which was never
    lock-protected at all.

    Kept as a no-op (rather than deleted) so the `with _JobKeepalive(...):`
    call sites don't need to change. See _ack_and_flush()/_nack() below
    for the actual fix for the cross-thread ACK/NACK problem this class
    used to (unsafely) work around: connection.add_callback_threadsafe().
    """

    def __init__(self, connection: pika.BlockingConnection,
                 pika_lock: threading.Lock, job_id: str):
        self._job_id = job_id

    def __enter__(self):
        log.debug(
            "[Job %s]  Keepalive no-op (start_consuming() on the main "
            "thread already pumps I/O/heartbeats continuously now).",
            self._job_id,
        )
        return self

    def __exit__(self, *_):
        pass


# ── ACK / NACK helpers ────────────────────────────────────────────────────────
#
# Both now use connection.add_callback_threadsafe() — pika's own documented
# mechanism for safely calling into a BlockingConnection from a thread OTHER
# than the one running start_consuming(). It thread-safely wakes the
# connection's I/O loop and runs the given callback ON that thread, so the
# actual basic_ack()/basic_nack() call never races with the main thread's
# own internal socket I/O the way a direct cross-thread call (even under a
# manual lock) could. pika_lock is kept only to serialize our own logging/
# bookkeeping around the schedule call, not as the actual safety mechanism.

def _ack_and_flush(channel, connection, pika_lock: threading.Lock,
                   delivery_tag: int, job_id: str) -> None:
    """ACK the message via the connection's own thread (thread-safe)."""
    # Record completion BEFORE ACKing — if ACK fails the message is requeued
    # and the local cache will catch the redelivery on the next attempt.
    from services.callback_client import mark_job_completed
    mark_job_completed(job_id)

    def _do_ack():
        try:
            channel.basic_ack(delivery_tag=delivery_tag)
            log.info("[Job %s]  Message acknowledged.", job_id)
        except Exception as exc:
            log.warning("[Job %s]  ACK failed (connection gone?): %s",
                        job_id, exc)

    try:
        connection.add_callback_threadsafe(_do_ack)
    except Exception as exc:
        # Connection is already gone — nothing we can do; the message
        # will be redelivered and the local idempotency cache (already
        # updated above via mark_job_completed) will catch it.
        log.warning(
            "[Job %s]  Could not schedule ACK (connection gone?): %s",
            job_id, exc,
        )


def _nack(channel, pika_lock: threading.Lock,
          delivery_tag: int, job_id: str) -> None:
    """NACK the message via the connection's own thread (thread-safe)."""
    def _do_nack():
        try:
            channel.basic_nack(delivery_tag=delivery_tag, requeue=False)
            log.warning("[Job %s]  Message nacked (no requeue).", job_id)
        except Exception as exc:
            log.warning("[Job %s]  NACK failed (connection gone?): %s",
                        job_id, exc)

    try:
        channel.connection.add_callback_threadsafe(_do_nack)
    except Exception as exc:
        log.warning(
            "[Job %s]  Could not schedule NACK (connection gone?): %s",
            job_id, exc,
        )


def _violation_timestamp_sort_key(v: dict) -> float:
    """Journey-global timestamp for a raw violation dict, used only to
    order the list before it becomes the payload. Defensive about the
    value's shape (plain float from analyzer.py at this stage, but be
    tolerant of an already-formatted 'H:MM:SS' string just in case)."""
    ts = v.get("timestamp", 0.0)
    if isinstance(ts, str):
        parts = ts.split(":")
        try:
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            return float(ts)
        except Exception:
            return 0.0
    try:
        return float(ts or 0.0)
    except Exception:
        return 0.0


def _video_result_from_dict(vr: dict) -> VideoResult:
    """Builds a VideoResult from the dict shape used both by the
    subprocess's clean-completion payload (result_q "result" type) and by
    the per-video crash-recovery snapshot (analyzer.py's
    _build_partial_video_result, carried on video_done events)."""
    # FIX — raw violations arrive in per-detector emission order (e.g. the
    # hand-raise detector's pass appends its events, then the RSL-hand-brake
    # detector's pass appends its own, etc.), NOT in chronological order
    # within the video. journey_log.py's v_sorted() already re-sorts for the
    # plain-text log, but that sort never touched this list — which is the
    # one that becomes the completion payload sent to Spring Boot and
    # rendered into the PDF report. Sort here, once, so every downstream
    # consumer (payload + PDF) sees violations in ascending timestamp order.
    raw_violations = sorted(vr.get("violations", []), key=_violation_timestamp_sort_key)
    return VideoResult(
        video_id           = vr["videoId"] if isinstance(vr.get("videoId"), int) else int(vr["videoId"]),
        video_name         = vr["videoName"],
        sequence_no        = vr["sequenceNo"],
        duration_seconds   = vr["durationSeconds"],
        duration_formatted = vr.get("durationFormatted", ""),
        fps                = vr.get("fps", 0.0),
        size_mb            = vr.get("sizeMb", 0.0),
        original_s3_key    = vr["originalS3Key"],
       violations         = [
            ViolationResult(
                violation_type           = v["violationType"],
                severity                 = v["severity"],
                confidence               = v["confidence"],
                risk_score               = v["riskScore"],
                timestamp_seconds        = v["timestamp"],
                original_video_timestamp = v["originalVideoTimestamp"],
                # FIX — these two were never read from the dict, so they
                # silently fell back to the dataclass defaults (0.0 / None)
                # every time a result crossed the subprocess boundary
                # (worker process -> main consumer process), even though
                # analyzer.py had already computed them correctly. This is
                # what was turning a real true_duration into `null` in the
                # payload sent to the Java API.
                duration_seconds         = v.get("durationSeconds", 0.0),
                trigger_duration_seconds = v.get("triggerDurationSeconds"),
                frame_paths              = v.get("framePaths", []),
                # FIX — status/role were never read here either, so they
                # always fell back to the dataclass defaults ("TRUE" / None)
                # on every result that crossed the subprocess boundary, even
                # when the LLM verification step had already computed the
                # real verdict (confirmed/rejected + LP/ALP/BOTH/None) in the
                # child process. This is what was turning a correct
                # status=FALSE / role=LP in the worker logs into
                # status=true / role=null in the payload sent to Java.
                # `status` crosses as a real JSON bool (see
                # ViolationResult.to_dict()) — convert back to the "TRUE"/
                # "FALSE" string the dataclass field stores internally.
                status                    = "TRUE" if v.get("status", True) else "FALSE",
                role                      = v.get("role"),
            )
            for v in raw_violations
        ],
    )


# ── Subprocess-supervised journey execution ────────────────────────────────
#
# Replaces a direct in-process call to analyze_journey() with a supervised
# child process. This is the fix for the failure mode where a native
# OpenCV OOM / access violation / segfault kills the ENTIRE worker process
# (RabbitMQ connection thread included) partway through a journey, leaving
# the remaining videos in that journey permanently unprocessed and the
# job stuck — because no Python `except` clause can run after a native
# crash kills the interpreter.
#
# With this in place: the child process is the one that dies. The parent
# (this function) detects the death via process.exitcode after join(),
# and — using the per-video "done" events the child streamed before it
# died — marks every video that never reported completion as failed too,
# then returns a failed_videos dict covering ALL of them (the one that
# actually crashed AND any after it that never got a chance to run).
def _run_journey_batch(
    job_id:       str,
    journey_id:   int,
    folder_name:  str,
    video_jobs:   list,          # only the videos THIS batch should attempt
    tmp_paths:    Dict[int, str],
    progress_cb,
    reported_ids: set,           # SHARED across retries — never double-report
    n_videos_total: int,         # for progress_cb's percentage (whole journey)
    n_already_done: int,         # videos completed in EARLIER batches
    initial_time_offset:  float, # Fix 7 — keeps timestamps continuous on retry
    initial_frame_offset: int,
):
    """
    Runs ONE submission to the GPU worker pool — either the initial attempt
    at a journey, or (Fix 7) a retry covering only the videos that hadn't
    yet completed after a previous batch's video was found stuck. Returns:

        (outcome, video_results, batch_wall_seconds, failed_videos,
         detail_by_id, stuck_video_id)

    outcome is one of:
      "done"     — the worker completed this batch cleanly (or with a
                   caught journey-level error). stuck_video_id is None.
      "stuck"    — Fix 1/6/7: the watchdog determined the CURRENT video
                   (stuck_video_id) has made no measurable progress for
                   WATCHDOG_STUCK_SECONDS. Its worker has already been
                   discarded. Videos in this batch that completed BEFORE
                   the stuck one are included in video_results/failed_videos
                   as normal; the stuck video and everything after it in
                   this batch are NOT included — the caller decides whether
                   to retry (from the stuck video onward) or give up.
      "abnormal" — worker crash / native OOM / parent-side interruption.
                   Every not-yet-completed video in THIS batch is marked
                   failed/not-processed (old crash-recovery behavior,
                   unchanged), stuck_video_id is None.
    """
    handle   = worker_pool.submit(job_id, journey_id, folder_name,
                                   video_jobs, tmp_paths,
                                   initial_time_offset=initial_time_offset,
                                   initial_frame_offset=initial_frame_offset)
    events_q = handle.events_q
    result_q = handle.result_q
    log.info("[Job %s]  Journey batch (%d video(s)) dispatched to GPU worker  "
             "pid=%s%s", job_id, len(video_jobs), handle.pid,
             "  [retry/continuation]" if initial_time_offset else "")

    ordered    = sorted(video_jobs, key=lambda v: v.sequence_no)
    n_videos   = len(ordered)
    completed_ids: Dict[int, bool] = {}      # video_id -> ok
    detail_by_id:  Dict[int, dict] = {}       # video_id -> {"errorType","reason"}
    # Best-effort per-video VideoResult snapshots (dict shape — see
    # analyzer.py's _build_partial_video_result), captured the moment each
    # video succeeds. If the child process then dies abnormally (native
    # crash/OOM/segfault) before it can send the final result_q payload,
    # these snapshots let us PRESERVE the already-succeeded videos' real
    # data instead of discarding it and marking them failed too.
    partial_results: Dict[int, dict] = {}
    batch_start_time = time.time()

    # ── Fix 1: progress-based watchdog state ──────────────────────────────
    # last_progress_at resets to "now" whenever EITHER a per-frame progress
    # event OR a video_done event arrives for the video currently at the
    # front of `ordered` that hasn't completed yet — i.e. whenever there is
    # any evidence the current video is still moving forward.
    last_progress_at = time.time()

    def _current_video_id():
        """The video this batch is presumed to be working on right now —
        the first one in sequence order that hasn't reported done yet.
        Sequential-per-worker processing (analyzer.py processes one video
        at a time) makes this reliable without needing to trust event
        ordering across a multiprocessing.Queue."""
        for vj in ordered:
            if vj.video_id not in completed_ids:
                return vj.video_id
        return None

    def _report_failure_immediately(vid: int, error_message: str,
                                      error_type: str | None,
                                      stack_trace: str | None,
                                      reason: str | None) -> None:
        """Calls the failed endpoint for one video, exactly once (across
        this batch AND any earlier/later retry batches — reported_ids is
        shared with the caller)."""
        if vid in reported_ids:
            return
        reported_ids.add(vid)
        try:
            send_video_failed(
                job_id        = job_id,
                journey_id    = journey_id,
                video_id      = vid,
                error_type    = error_type or "PROCESSING_ERROR",
                error_message = error_message,
                stack_trace   = stack_trace or "",
                reason        = reason,
            )
        except Exception as exc:
            log.error("[Job %s]  send_video_failed failed for video=%d: %s",
                       job_id, vid, exc)

    def _handle_event(ev: dict) -> None:
        """Shared per-event handling, used by every drain site (the
        polling loop, the post-exit final drain, AND the interrupt-
        triggered drain below) so a parent-side interruption never skips
        this bookkeeping the way it used to."""
        nonlocal last_progress_at
        ev_type = ev.get("type")

        if ev_type == "progress":
            # Fix 1/6: real evidence this video is still moving forward.
            last_progress_at = ev.get("last_progress_time") or time.time()
            return

        if ev_type != "video_done":
            return

        last_progress_at = time.time()  # finishing a video is progress too
        vid = ev["video_id"]
        ok  = ev.get("ok", False)
        completed_ids[vid] = ok
        if ok:
            snapshot = ev.get("video_result")
            if snapshot:
                partial_results[vid] = snapshot
        if not ok:
            err = ev.get("error") or "Unknown per-video failure"
            detail_by_id[vid] = {
                "errorType": ev.get("error_type") or "PROCESSING_ERROR",
                "reason":    ev.get("reason"),
            }
            # ── Phase 1: report THIS video as FAILED right now ─────
            _report_failure_immediately(
                vid, err, ev.get("error_type"), ev.get("stack_trace"),
                ev.get("reason"),
            )

    def _drain_events_once() -> None:
        """Non-blocking drain of whatever is currently sitting on
        events_q. Used for the interrupt-recovery final drain below."""
        while True:
            try:
                ev = events_q.get_nowait()
            except _queue.Empty:
                break
            _handle_event(ev)

    def _finalize_abnormal(label: str, crash_reason: str):
        """Shared finalization for a crashed child, a genuinely STUCK
        video giving up permanently, or a parent-side interruption
        (Ctrl+C, KeyboardInterrupt, RabbitMQ connection loss, any
        unexpected exception) while the parent was waiting on the child.
        Preserves real VideoResult data for any video whose success
        snapshot already arrived on events_q, and marks only the
        genuinely-unfinished videos as failed/not-processed — never
        discards already-completed work."""
        try:
            resource_manager.emergency_cleanup(reason=label)
        except Exception:
            log.error("[Job %s]  emergency_cleanup() failed:\n%s",
                      job_id, traceback.format_exc())

        failed_videos: Dict[int, str] = {}
        video_results = []
        for vj in ordered:
            if vj.video_id in completed_ids and completed_ids[vj.video_id] and \
                    vj.video_id in partial_results:
                # This video finished successfully AND we have its real
                # result snapshot (analyzer.py's _build_partial_video_result,
                # carried on the video_done event) — preserve it as a
                # genuine success rather than discarding it. NOT added to
                # failed_videos / not reported to /failed; the journey-level
                # outcome is decided by the caller based on succeeded vs.
                # failed counts and will resolve to COMPLETED_WITH_ERRORS
                # (never FAILED) as long as at least one video succeeded.
                try:
                    video_results.append(_video_result_from_dict(partial_results[vj.video_id]))
                    continue
                except Exception as exc:
                    log.error(
                        "[Job %s]  Failed to rebuild VideoResult from "
                        "recovery snapshot for video_id=%d (%s) — falling "
                        "back to marking it failed.", job_id, vj.video_id, exc,
                    )
                    # fall through to the failure-marking branches below

            if vj.video_id in reported_ids:
                # Already individually reported above (either a normal
                # per-video failure, or already synthesized in a prior pass).
                continue
            if vj.video_id in completed_ids and completed_ids[vj.video_id]:
                # Finished successfully, but no result snapshot is
                # available — we cannot fabricate violation data, so we
                # still must surface it truthfully: mark it failed too,
                # with a distinct reason, so Spring Boot doesn't silently
                # lose a video's results.
                msg = (
                    "Processed successfully but worker was interrupted "
                    f"before journey results could be finalized ({label}); "
                    "re-run required."
                )
                failed_videos[vj.video_id] = msg
                detail_by_id[vj.video_id] = {
                    "errorType": "RESOURCE_EXHAUSTION", "reason": None,
                }
                _report_failure_immediately(vj.video_id, msg, "RESOURCE_EXHAUSTION",
                                              None, None)
            else:
                # Never started, or started but the interruption happened
                # mid-video — report it as NOT_PROCESSED, immediately.
                reason_msg = "Not Processed - Worker Resource Exhaustion"
                failed_videos[vj.video_id] = crash_reason
                detail_by_id[vj.video_id] = {
                    "errorType": "NOT_PROCESSED", "reason": reason_msg,
                }
                _report_failure_immediately(vj.video_id, crash_reason,
                                              "NOT_PROCESSED", None, reason_msg)

        return "abnormal", video_results, time.time() - batch_start_time, failed_videos, detail_by_id, None

    # ── Drain events_q while waiting for the child to finish ──────────────
    #
    # Any interruption here is caught, a final drain collects whatever the
    # child already finished, the child is terminated cleanly, and we
    # finalize through the SAME "abnormal exit" logic used for a crashed
    # child — so already-completed videos are never lost and no process is
    # left orphaned.
    result_payload = None
    try:
        while True:
            # Pull any progress/video_done events that arrived since the
            # last poll.
            drained_any = False
            while True:
                try:
                    ev = events_q.get_nowait()
                except _queue.Empty:
                    break
                drained_any = True
                _handle_event(ev)
                if ev.get("type") == "video_done" and progress_cb:
                    done_so_far = n_already_done + len(completed_ids)
                    pct = 10 + int((done_so_far / max(n_videos_total, 1)) * 80)
                    progress_cb(pct, f"Analyzed video {done_so_far} of {n_videos_total}",
                                done_so_far)

            # ── "Did this batch finish?" signal ──────────────────────────
            # A persistent worker never exits between journeys (that's the
            # whole point — see worker_pool.py), so we can no longer use
            # process.exitcode to detect completion the way the old
            # one-shot-subprocess code did. The batch is done, cleanly or
            # with a caught error, the moment its result lands on result_q.
            try:
                result_payload = result_q.get_nowait()
                break
            except _queue.Empty:
                pass

            if not handle.is_alive():
                log.error(
                    "[Job %s]  GPU worker (pid=%s) is gone — treating as a "
                    "crash.", job_id, handle.pid,
                )
                handle.discard()
                return _finalize_abnormal(
                    label="GPU worker crashed",
                    crash_reason="GPU worker crashed — video was not processed.",
                )

            # ── Fix 1/6/7: progress-based watchdog ───────────────────────
            # Only active while there is a "current video" still in flight
            # in THIS batch — once every video in the batch has reported
            # done, the worker is in journey-level finalization (dedup,
            # LLM verification pass, journey text log) which is not a
            # per-video stall this watchdog is meant to catch.
            stuck_vid = _current_video_id()
            if stuck_vid is not None and \
                    (time.time() - last_progress_at) > WATCHDOG_STUCK_SECONDS:
                log.error(
                    "[Job %s]  WATCHDOG: video_id=%d has made no progress "
                    "for %.0fs (limit=%.0fs) — treating as stuck. Killing "
                    "its GPU worker (pid=%s); the REST of the journey is "
                    "NOT affected.", job_id, stuck_vid,
                    time.time() - last_progress_at, WATCHDOG_STUCK_SECONDS,
                    handle.pid,
                )
                handle.discard()  # kills only this worker; pool replaces it
                # Videos in `ordered` BEFORE the stuck one already completed
                # (successfully OR with an ordinary per-video failure) in
                # this batch — return them as real results / real failures.
                # The stuck video itself and anything after it are simply
                # omitted here; the caller (retry loop) decides to retry
                # from the stuck video onward or give up on it.
                video_results = [
                    _video_result_from_dict(partial_results[vid_])
                    for vid_ in completed_ids
                    if completed_ids[vid_] and vid_ in partial_results
                ]
                pre_stuck_failed = {
                    vid_: (detail_by_id.get(vid_, {}) or {}).get("reason")
                          or "Video processing failed — see worker log for details"
                    for vid_ in completed_ids
                    if not completed_ids[vid_]
                }
                return ("stuck", video_results, time.time() - batch_start_time,
                        pre_stuck_failed, detail_by_id, stuck_vid)

            if ABSOLUTE_JOURNEY_CEILING_SECONDS and \
                    (time.time() - batch_start_time) > ABSOLUTE_JOURNEY_CEILING_SECONDS:
                log.error(
                    "[Job %s]  Batch exceeded the opt-in absolute ceiling "
                    "(%.0fs) — killing and replacing its GPU worker "
                    "(pid=%s).", job_id, ABSOLUTE_JOURNEY_CEILING_SECONDS,
                    handle.pid,
                )
                handle.discard()
                return _finalize_abnormal(
                    label="absolute journey ceiling exceeded",
                    crash_reason=(
                        "Worker exceeded the configured absolute journey "
                        "ceiling — video was not processed."
                    ),
                )

            if not drained_any:
                time.sleep(EVENTS_POLL_INTERVAL_SECONDS)
    except BaseException as exc:
        # BaseException (not just Exception) so KeyboardInterrupt/SystemExit
        # are caught here too, per the "any interruption while the parent is
        # waiting" requirement.
        log.error(
            "[Job %s]  Parent interrupted while waiting on GPU worker "
            "(%s: %s) — draining already-completed video results and "
            "discarding the worker (it will be replaced) instead of "
            "losing them.", job_id, type(exc).__name__, exc,
        )
        _drain_events_once()
        handle.discard()
        log.info(
            "[Job %s]  GPU worker discarded after parent interruption "
            "(pool replaces it automatically — no orphaned process left "
            "running).", job_id,
        )
        return _finalize_abnormal(
            label=f"parent interrupted: {type(exc).__name__}: {exc}",
            crash_reason=(
                f"Worker interrupted ({type(exc).__name__}) — "
                "video was not processed."
            ),
        )

    # ── Clean outcome: the worker computed a full result ───────────────────
    if result_payload is not None:
        # The worker process itself is healthy — return it to the pool so
        # it can pick up the next queued journey (model stays loaded).
        handle.mark_finished()

        if result_payload.get("type") == "result":
            video_results = [_video_result_from_dict(vr) for vr in result_payload["video_results"]]
            return ("done", video_results, result_payload["wall_seconds"],
                     result_payload["failed_videos"], detail_by_id, None)

        if result_payload.get("type") == "error":
            # Worker returned cleanly (from its own point of view) but
            # analyze_journey raised above its own per-video isolation —
            # treat every not-yet-completed video in THIS batch as failed,
            # and report every single one immediately.
            log.error("[Job %s]  GPU worker reported an error: %s",
                       job_id, result_payload.get("message"))
            failed_videos = {}
            for vj in ordered:
                if vj.video_id in completed_ids and completed_ids[vj.video_id]:
                    continue
                msg = result_payload.get("message", "Unknown error")
                failed_videos[vj.video_id] = msg
                detail_by_id[vj.video_id] = {"errorType": "PROCESSING_ERROR", "reason": None}
                _report_failure_immediately(
                    vj.video_id, msg, "PROCESSING_ERROR",
                    result_payload.get("traceback"), None,
                )
            return ("done", [], time.time() - batch_start_time, failed_videos, detail_by_id, None)

    # ── Abnormal outcome: crash, OOM-kill, segfault, etc. ───────────────────
    # By the time we get here the worker has already been discarded (via
    # handle.discard() above) and the pool has spawned its replacement.
    log.error(
        "[Job %s]  GPU worker died/was killed abnormally  "
        "videos_completed=%d/%d (this batch)", job_id, len(completed_ids), n_videos,
    )
    return _finalize_abnormal(
        label="GPU worker crashed",
        crash_reason="GPU worker crashed — video was not processed.",
    )


def _run_journey_supervised(
    job_id:      str,
    journey_id:  int,
    folder_name: str,
    video_jobs:  list,
    tmp_paths:   Dict[int, str],
    progress_cb,
):
    """
    Returns (video_results, wall_seconds, failed_videos, video_error_details)
    — failed_videos/video_error_details have the same shape analyze_journey()
    produces, plus this function ALSO calls send_video_failed() immediately
    for every video as its failure becomes known (Phase 1 requirement: "no
    failed video should wait until journey completion to be reported") —
    runs the actual work in an isolated child process so a native crash
    can't take down the consumer.

    Fix 1/7 — this is now a thin RETRY LOOP around `_run_journey_batch()`.
    The old implementation killed the ENTIRE journey (every video, even
    ones that hadn't started yet, and re-attempted nothing) the moment a
    single fixed wall-clock timeout was exceeded. Now: a progress-based
    watchdog inside `_run_journey_batch()` detects only the ONE video that
    is actually stuck, that video's worker is discarded, and the video is
    retried FROM SCRATCH (frame-level resume is out of scope — see Fix 7)
    in a fresh batch covering just the stuck video onward. Videos before
    it are never re-run; videos after it simply haven't started yet. Only
    after WATCHDOG_MAX_VIDEO_RETRIES failed retries of the SAME video is
    it finally given up on and marked FAILED, and the journey continues
    with whatever comes after it.
    """
    ordered        = sorted(video_jobs, key=lambda v: v.sequence_no)
    n_videos_total = len(ordered)
    outer_start    = time.time()

    all_video_results: list            = []
    all_failed_videos: Dict[int, str]  = {}
    all_detail_by_id:  Dict[int, dict] = {}
    reported_ids:      set             = set()   # shared across every retry batch
    retry_count_by_video: Dict[int, int] = {}

    remaining = list(ordered)
    time_offset_seed  = 0.0
    frame_offset_seed = 0

    while remaining:
        outcome, video_results, _batch_wall, failed_videos, detail_by_id, stuck_vid = \
            _run_journey_batch(
                job_id, journey_id, folder_name, remaining, tmp_paths,
                progress_cb, reported_ids, n_videos_total,
                len(all_video_results),
                time_offset_seed, frame_offset_seed,
            )

        all_video_results.extend(video_results)
        all_failed_videos.update(failed_videos)
        all_detail_by_id.update(detail_by_id)

        if outcome == "done":
            break

        if outcome == "abnormal":
            # Old crash-recovery behavior, scoped to whatever was still
            # `remaining` — everything before this batch (earlier retries)
            # is already safely accumulated above.
            break

        # outcome == "stuck"
        retry_count_by_video[stuck_vid] = retry_count_by_video.get(stuck_vid, 0) + 1
        stuck_idx = next(i for i, vj in enumerate(remaining) if vj.video_id == stuck_vid)

        # Recompute continuity seeds from everything genuinely completed so
        # far (across ALL batches so far), so journey-global violation
        # timestamps stay continuous across the retry.
        time_offset_seed  = sum(vr.duration_seconds for vr in all_video_results)
        frame_offset_seed = int(time_offset_seed * 25.0)  # 25fps fallback estimate — cosmetic only, never used for timestamps

        if retry_count_by_video[stuck_vid] <= WATCHDOG_MAX_VIDEO_RETRIES:
            log.warning(
                "[Job %s]  Retrying stuck video_id=%d from scratch "
                "(attempt %d/%d) — journey continues, no other video is "
                "affected.", job_id, stuck_vid,
                retry_count_by_video[stuck_vid], WATCHDOG_MAX_VIDEO_RETRIES,
            )
            remaining = remaining[stuck_idx:]  # stuck video + everything after it (never started)
            continue

        # Retries exhausted — give up on this ONE video, mark it FAILED,
        # and continue the journey with whatever comes after it (Fix 7:
        # "if a video gets stuck, do not kill the entire journey").
        msg = (
            f"Video exceeded the watchdog stall limit "
            f"({WATCHDOG_STUCK_SECONDS:.0f}s with no progress) after "
            f"{WATCHDOG_MAX_VIDEO_RETRIES} retry(ies) — marked FAILED; "
            "remaining videos in the journey continue."
        )
        log.error("[Job %s]  Giving up on video_id=%d: %s", job_id, stuck_vid, msg)
        all_failed_videos[stuck_vid] = msg
        all_detail_by_id[stuck_vid]  = {"errorType": "STALLED", "reason": msg}
        try:
            send_video_failed(
                job_id        = job_id,
                journey_id    = journey_id,
                video_id      = stuck_vid,
                error_type    = "STALLED",
                error_message = msg,
                stack_trace   = "",
                reason        = msg,
            )
        except Exception as exc:
            log.error("[Job %s]  send_video_failed failed for stalled "
                       "video=%d: %s", job_id, stuck_vid, exc)
        remaining = remaining[stuck_idx + 1:]  # skip the failed video, keep going

    return all_video_results, time.time() - outer_start, all_failed_videos, all_detail_by_id


# ── Fatal-interruption finalization (partial-journey recovery) ────────────
#
# Called from _handle_job's outer exception handler when a fatal
# interruption (unexpected exception, KeyboardInterrupt, etc.) strikes
# AFTER at least one video has already been processed successfully by
# `_run_journey_supervised`. Per spec we must NOT call the failed
# callback in this case — instead we mark every still-unprocessed video
# as failed, wait up to RECOVERY_WAIT_SECONDS while retrying the
# completion callback, and let the caller ACK the message either way so
# the worker can move on to the next queued journey.
def _send_completed_with_recovery(completion_dict: dict, job_id: str) -> bool:
    """
    Sends the COMPLETED_WITH_ERRORS completion callback, retrying for up
    to RECOVERY_WAIT_SECONDS if the first attempt fails (e.g. the same
    transient condition that interrupted the journey — connection loss,
    broker hiccup — may still be resolving). Never re-processes any
    video; only retries delivering the already-final payload.

    Returns True if the callback was eventually delivered, False if the
    recovery window was exhausted without success (the journey's partial
    results are still preserved locally via the uploaded text log either
    way — see build_journey_log_text()/upload_text_log() in _handle_job).
    """
    deadline = time.time() + RECOVERY_WAIT_SECONDS
    attempt  = 0
    while True:
        attempt += 1
        try:
            send_completed(completion_dict)
            log.info(
                "[Job %s]  COMPLETED_WITH_ERRORS callback delivered on "
                "attempt %d after fatal interruption.", job_id, attempt,
            )
            return True
        except Exception as exc:
            remaining = deadline - time.time()
            if remaining <= 0:
                log.error(
                    "[Job %s]  Could not deliver COMPLETED_WITH_ERRORS "
                    "callback within the %ds recovery window (%d attempts): "
                    "%s — giving up; partial results remain preserved in "
                    "the uploaded text log.", job_id, RECOVERY_WAIT_SECONDS,
                    attempt, exc,
                )
                return False
            wait_s = min(RECOVERY_RETRY_SECONDS, remaining)
            log.warning(
                "[Job %s]  COMPLETED_WITH_ERRORS callback attempt %d "
                "failed (%s) — retrying in %.0fs (recovery window "
                "remaining %.0fs).", job_id, attempt, exc, wait_s, remaining,
            )
            time.sleep(wait_s)


def _finalize_partial_journey_on_interruption(
    *,
    job_id: str,
    journey_id: int,
    train_detail_id: int,
    folder_name: str,
    msg_videos: list,
    video_results: list,
    failed_videos: Dict[int, str],
    video_error_details: Dict[int, dict],
    wall_seconds: float,
    job_started_at: datetime,
    interruption_reason: str,
) -> None:
    """
    Builds and sends the COMPLETED_WITH_ERRORS payload after a fatal
    interruption that hit a journey with at least one already-succeeded
    video. Marks every video that is neither in video_results nor already
    in failed_videos as failed (reason: fatal interruption), uploads the
    text log (so results are preserved even if the callback ultimately
    can't be delivered), and attempts the completion callback with a
    bounded recovery window.
    """
    succeeded_ids = {vr.video_id for vr in video_results}
    accounted_ids = succeeded_ids | set(failed_videos.keys())

    for vj in msg_videos:
        if vj.video_id in accounted_ids:
            continue
        msg = (
            f"Not Processed - Fatal Interruption ({interruption_reason}) — "
            "journey was interrupted before this video could be processed."
        )
        failed_videos[vj.video_id] = msg
        video_error_details[vj.video_id] = {
            "errorType": "NOT_PROCESSED",
            "reason":    "Not Processed - Fatal Interruption",
        }
        try:
            send_video_failed(
                job_id        = job_id,
                journey_id    = journey_id,
                video_id      = vj.video_id,
                error_type    = "NOT_PROCESSED",
                error_message = msg,
                stack_trace   = "",
                reason        = "Not Processed - Fatal Interruption",
            )
        except Exception as exc:
            log.error("[Job %s]  send_video_failed failed for video=%d "
                      "during interruption finalization: %s",
                      job_id, vj.video_id, exc)

    failed_ids = set(failed_videos.keys())
    journey_status = compute_journey_status(
        total_videos        = len(msg_videos),
        succeeded_video_ids = succeeded_ids,
        failed_video_ids    = failed_ids,
    )
    # By construction at least one video succeeded, so this should always
    # resolve to COMPLETED_WITH_ERRORS (or COMPLETED, if the interruption
    # struck after the very last video already succeeded) — never FAILED.
    if journey_status == "FAILED":
        journey_status = "COMPLETED_WITH_ERRORS"

    log.warning(
        "[Job %s]  Finalizing partial journey after fatal interruption "
        "(%s).  succeeded=%d  failed=%d  status=%s",
        job_id, interruption_reason, len(succeeded_ids), len(failed_ids),
        journey_status,
    )

    try:
        log_text = build_journey_log_text(
            job_id              = job_id,
            journey_id          = journey_id,
            video_results       = video_results,
            total_wall_seconds  = wall_seconds,
            started_at          = job_started_at,
            failed_videos       = failed_videos,
            video_error_details = video_error_details,
            journey_status      = journey_status,
        )
        upload_text_log(
            text        = log_text,
            folder_name = folder_name,
            filename    = f"{job_id}.txt",
        )
        log.info("[Job %s]  Text log uploaded to S3 (partial-journey path).",
                 job_id)
    except Exception as log_exc:
        log.warning("[Job %s]  Text log build/upload failed (non-fatal): %s",
                    job_id, log_exc)
    _upload_job_console_log(job_id, folder_name)

    processing_time_ms = int(wall_seconds * 1000)
    completion = CompletionPayload(
        job_id          = job_id,
        journey_id      = journey_id,
        train_detail_id = train_detail_id,
        folder_name     = folder_name,
        processing_time = processing_time_ms,
        video_results   = video_results,
    )
    completion_dict = completion.to_dict()
    completion_dict["journeyStatus"] = journey_status

    try:
        json_key = upload_json_result(
            payload     = completion_dict,
            folder_name = folder_name,
            filename    = f"{job_id}_result.json",
        )
        log.info("[Job %s]  Completion JSON result uploaded to S3 (partial-journey "
                 "path): %s", job_id, json_key)
    except Exception as exc:
        log.warning("[Job %s]  Completion JSON result upload failed (non-fatal, "
                    "callback will still be sent): %s", job_id, exc)

    # Notify the frontend of progress/status before the final callback too,
    # best-effort — mirrors the normal-path "Sending results to backend"
    # progress update so the UI doesn't appear stuck at whatever percentage
    # it was at when the interruption happened.
    try:
        send_progress(
            job_id, journey_id, 95,
            f"Recovering from interruption — status {journey_status}",
            current_video=len(msg_videos),
        )
    except Exception as exc:
        log.warning("[Job %s]  progress callback failed (non-fatal): %s",
                    job_id, exc)

    _send_completed_with_recovery(completion_dict, job_id)


# ── Job handler ───────────────────────────────────────────────────────────────

def _handle_job(
    msg: AnalysisJobMessage,
    channel,
    method,
    connection,
    pika_lock: threading.Lock,
) -> None:
    """
    Full processing flow for one AnalysisJobMessage.
    ACKs on success, NACKs (no requeue) on failure.
    """
    job_id          = msg.job_id
    journey_id      = msg.journey_id
    train_detail_id = msg.train_detail_id
    folder_name     = msg.folder_name
    tmp_paths: Dict[int, str] = {}
    job_started_at  = datetime.now()

    # Tag this thread so every log.*() call it makes from here on — in
    # consumer.py, callback_client.py, resource_manager.py, worker_pool.py —
    # is also captured into logs/jobs/<job_id>.log (see job_logging.py).
    job_logging.start_job(job_id)

    log.info(
        "[Job %s]  journey=%d  trainDetail=%d  videos=%d  folder=%s  redelivered=%s",
        job_id, journey_id, train_detail_id, len(msg.videos), folder_name,
        method.redelivered,
    )

    if msg.callback_base_url:
        log.info("[Job %s]  callbackBaseUrl=%s", job_id, msg.callback_base_url)
        set_base_url(msg.callback_base_url)

    # ── Idempotency guard ─────────────────────────────────────────────────────
    try:
        if check_job_completed(job_id):
            log.warning(
                "[Job %s]  Already completed (redelivered=%s) — ACKing and skipping.",
                job_id, method.redelivered,
            )
            _ack_and_flush(channel, connection, pika_lock,
                           method.delivery_tag, job_id)
            return
    except Exception as idem_err:
        log.warning(
            "[Job %s]  Idempotency check failed (%s) — proceeding.",
            job_id, idem_err,
        )

    # ── In-progress guard ──────────────────────────────────────────────────────
    # Catches a redelivery that arrives WHILE the original delivery is still
    # being processed on another thread (e.g. RabbitMQ connection drop +
    # reconnect redelivers an unacked message before the original job thread
    # has finished). check_job_completed() above only catches a job that has
    # ALREADY finished — this catches the "still running right now" case,
    # which the old single-threaded design could never encounter but the
    # worker-pool concurrency model makes possible.
    if not try_start_job(job_id):
        log.warning(
            "[Job %s]  Already being processed by another thread right now "
            "(redelivered=%s) — ACKing this duplicate delivery WITHOUT "
            "reprocessing. The in-flight thread owns this journey and will "
            "send the real completion/failure callback.",
            job_id, method.redelivered,
        )
        _ack_and_flush(channel, connection, pika_lock,
                       method.delivery_tag, job_id)
        return

    # ── Start keepalive — covers download + analysis + callback ───────────────
    # FIX: keepalive starts HERE, before downloads, because the download phase
    # can take several minutes and will starve pika's heartbeat just as badly
    # as the analysis phase.  The thread is stopped after send_completed()
    # returns, just before basic_ack().
    with _JobKeepalive(connection, pika_lock, job_id):
        job_failed = False

        # Mutable state visible to the outer except block, so a fatal
        # interruption at ANY point after _run_journey_supervised has
        # returned at least one successful video can be finalized as
        # COMPLETED_WITH_ERRORS instead of FAILED.
        video_results:       list            = []
        failed_videos:       Dict[int, str]  = {}
        video_error_details: Dict[int, dict] = {}
        wall_seconds:        float           = 0.0

        try:
            # ── Step 2: Download all videos from S3 ──────────────────────────
            # DIAGNOSTIC: explicit phase-start timestamp (job-scoped) so this
            # journey's download WINDOW can be directly overlaid against any
            # other concurrently-running journey's window in the logs — the
            # existing "Downloaded video_id=..." lines only show completion,
            # never when the phase (or each video) actually started.
            download_phase_start = time.time()
            log.info(
                "[Job %s]  [DOWNLOAD-DIAG] Download phase STARTED  "
                "wall_clock=%s  video_count=%d",
                job_id, time.strftime("%H:%M:%S", time.localtime(download_phase_start)),
                len(msg.videos),
            )
            send_progress(job_id, journey_id, 5, "Downloading videos")

            for vj in sorted(msg.videos, key=lambda v: v.sequence_no):
                suffix   = os.path.splitext(vj.s3_key)[1] or ".mp4"
                fd, path = tempfile.mkstemp(suffix=suffix)
                os.close(fd)
                video_dl_start = time.time()
                log.info(
                    "[Job %s]  [DOWNLOAD-DIAG] video_id=%d seq=%d STARTED  "
                    "wall_clock=%s",
                    job_id, vj.video_id, vj.sequence_no,
                    time.strftime("%H:%M:%S", time.localtime(video_dl_start)),
                )
                download_video(vj.s3_key, path)
                tmp_paths[vj.video_id] = path
                log.info(
                    "[Job %s]  Downloaded video_id=%d  seq=%d  → %s  "
                    "(took %.1fs)",
                    job_id, vj.video_id, vj.sequence_no, path,
                    time.time() - video_dl_start,
                )

            log.info(
                "[Job %s]  [DOWNLOAD-DIAG] Download phase COMPLETED  "
                "total_duration_s=%.1f  video_count=%d  avg_s_per_video=%.1f",
                job_id, time.time() - download_phase_start, len(msg.videos),
                (time.time() - download_phase_start) / max(len(msg.videos), 1),
            )

            # ── Step 3: Progress after download ──────────────────────────────
            try:
                send_progress(
                job_id, journey_id, 10,
                "Downloads complete — starting analysis",
                current_video=1,
            )
            except Exception as exc:
                log.warning("[Job %s]  progress(10) callback failed (non-fatal): %s",
                            job_id, exc)

            # ── Steps 4–7: Analyse + upload frames + build results ────────────
            def _progress(pct: int, message: str, current_video: int = 1) -> None:
                try:
                    send_progress(job_id, journey_id, pct, message,
                                  current_video=current_video)
                except Exception as exc:
                    log.warning(
                        "[Job %s]  progress callback failed (non-fatal): %s",
                        job_id, exc,
                    )

            # NOTE: _run_journey_supervised already isolates native crashes,
            # OOM, and per-journey timeouts inside the child process and
            # NEVER lets those abort this call — it always returns a
            # (video_results, wall_seconds, failed_videos, video_error_details)
            # tuple for those cases (see its "Abnormal exit" branch above).
            # Assigning straight into the outer-scope variables means that
            # if a fatal interruption happens in any step AFTER this call
            # (text log build, callback send, etc.), the outer except block
            # below still has the correct partial results to work with.
            (video_results, wall_seconds,
             failed_videos, video_error_details) = _run_journey_supervised(
                job_id      = job_id,
                journey_id  = journey_id,
                folder_name = folder_name,
                video_jobs  = msg.videos,
                tmp_paths   = tmp_paths,
                progress_cb = _progress,
            )

            # ── Step 8: Calculate processing time (ms) ────────────────────────
            processing_time_ms = int(wall_seconds * 1000)

            succeeded_ids = {vr.video_id for vr in video_results
                              if vr.video_id not in failed_videos}
            failed_ids    = set(failed_videos.keys())
            journey_status = compute_journey_status(
                total_videos         = len(msg.videos),
                succeeded_video_ids  = succeeded_ids,
                failed_video_ids     = failed_ids,
            )

            log.info(
                "[Job %s]  Analysis complete.  videos=%d  violations=%d  time=%dms  "
                "failed=%d  status=%s",
                job_id,
                len(video_results),
                sum(len(vr.violations) for vr in video_results),
                processing_time_ms,
                len(failed_videos),
                journey_status,
            )

            # ── Step 8b: Build and upload the .txt analysis log ───────────────
            # Built for EVERY outcome now (COMPLETED, COMPLETED_WITH_ERRORS,
            # and FAILED-with-partial-results) — not just full success — so
            # the log always reflects which videos actually failed and why.
            # Written to the same dynamic journey folder used for frames:
            # <folderName>/<jobId>.txt
            try:
                log_text = build_journey_log_text(
                    job_id              = job_id,
                    journey_id          = journey_id,
                    video_results       = video_results,
                    total_wall_seconds  = wall_seconds,
                    started_at          = job_started_at,
                    failed_videos       = failed_videos,
                    video_error_details = video_error_details,
                    journey_status      = journey_status,
                )
                upload_text_log(
                    text        = log_text,
                    folder_name = folder_name,
                    filename    = f"{job_id}.txt",
                )
                log.info("[Job %s]  Text log uploaded to S3.", job_id)
            except Exception as log_exc:
                log.warning(
                    "[Job %s]  Text log build/upload failed (non-fatal): %s",
                    job_id, log_exc,
                )
            _upload_job_console_log(job_id, folder_name)

            # ── Step 9: Send completion OR failed callback ─────────────────────
            # Per-video failures were ALREADY reported individually and
            # immediately inside _run_journey_supervised (Phase 1 requirement
            # — no video waits until journey end to be reported). This step
            # decides the JOURNEY-level outcome:
            #
            #   COMPLETED              → send_completed with all video_results.
            #   COMPLETED_WITH_ERRORS  → still send_completed (the journey DID
            #                            finish — some videos succeeded and
            #                            their results are real); per-video
            #                            failures are already on record via
            #                            the immediate /failed calls above.
            #   FAILED                 → every video failed (or none ran) —
            #                            send one journey-level /failed call
            #                            summarizing it, no /completed call.
            if journey_status == "FAILED":
                error_message = "; ".join(
                    f"video_id={vid}: {msg_}" for vid, msg_ in failed_videos.items()
                ) or "Journey failed before any video could be processed."
                log.error(
                    "[Job %s]  Journey FAILED — all %d video(s) failed: %s",
                    job_id, len(failed_videos), error_message,
                )
                send_failed(job_id, journey_id, error_message)
                log.info("[Job %s]  Journey-level failed callback sent.", job_id)
            else:
                send_progress(
                    job_id, journey_id, 95,
                    "Sending results to backend",
                    current_video=len(msg.videos),
                )
                completion = CompletionPayload(
                    job_id          = job_id,
                    journey_id      = journey_id,
                    train_detail_id = train_detail_id,
                    folder_name     = folder_name,
                    processing_time = processing_time_ms,
                    video_results   = video_results,
                )
                completion_dict = completion.to_dict()
                completion_dict["journeyStatus"] = journey_status

                try:
                    json_key = upload_json_result(
                        payload     = completion_dict,
                        folder_name = folder_name,
                        filename    = f"{job_id}_result.json",
                    )
                    log.info("[Job %s]  Completion JSON result uploaded to S3: %s",
                             job_id, json_key)
                except Exception as exc:
                    log.warning("[Job %s]  Completion JSON result upload failed "
                                "(non-fatal, callback will still be sent): %s",
                                job_id, exc)

                send_completed(completion_dict)
                log.info(
                    "[Job %s]  Completion callback sent.  status=%s  failedVideos=%d",
                    job_id, journey_status, len(failed_videos),
                )

        except BaseException as exc:
            # BaseException (not just Exception) so that fatal interruptions
            # such as KeyboardInterrupt (Ctrl+C) and SystemExit are handled
            # by the same partial-results-preserving logic as any other
            # unexpected error, instead of skipping cleanup/finalization.
            err_detail = traceback.format_exc()
            log.error("[Job %s]  FATAL INTERRUPTION:\n%s", job_id, err_detail)

            # ── Resource lifecycle (Phase 6): release resources on any
            # journey-level failure that isn't already covered by the
            # subprocess-crash path (e.g. S3 download failure, an exception
            # raised while building/uploading the text log, or a callback
            # failure that propagated up instead of being caught locally).
            try:
                resource_manager.cleanup_after_failure(
                    job_id=job_id, reason=f"{type(exc).__name__}: {exc}",
                )
            except Exception:
                log.error("[Job %s]  cleanup_after_failure() failed:\n%s",
                          job_id, traceback.format_exc())

            if video_results:
                # ── At least one video already succeeded: do NOT call the
                # failed callback. Preserve the successful results, mark the
                # rest as failed, and finalize as COMPLETED_WITH_ERRORS
                # (with a bounded recovery/retry window for the callback).
                job_failed = False
                try:
                    _finalize_partial_journey_on_interruption(
                        job_id               = job_id,
                        journey_id           = journey_id,
                        train_detail_id      = train_detail_id,
                        folder_name          = folder_name,
                        msg_videos           = msg.videos,
                        video_results        = video_results,
                        failed_videos        = failed_videos,
                        video_error_details  = video_error_details,
                        wall_seconds         = wall_seconds,
                        job_started_at       = job_started_at,
                        interruption_reason  = f"{type(exc).__name__}: {exc}",
                    )
                except Exception as finalize_exc:
                    # Even the recovery path itself failed unexpectedly —
                    # this is the only case where we still fall back to the
                    # failed callback, since we genuinely could not finalize
                    # the partial results any other way.
                    log.error(
                        "[Job %s]  Partial-journey finalization itself "
                        "failed: %s — falling back to failed callback.",
                        job_id, finalize_exc,
                    )
                    job_failed = True
                    try:
                        send_failed(job_id, journey_id,
                                    f"{exc}\n\n{err_detail}\n\n"
                                    f"(partial-finalization also failed: {finalize_exc})")
                    except Exception as sf_exc:
                        log.warning("[Job %s]  send_failed itself failed: %s",
                                    job_id, sf_exc)
            else:
                # ── Zero videos succeeded — existing FAILED-callback path. ──
                job_failed = True
                try:
                    send_failed(job_id, journey_id, f"{exc}\n\n{err_detail}")
                except Exception as sf_exc:
                    log.warning("[Job %s]  send_failed itself failed: %s",
                                job_id, sf_exc)
        finally:
            # Always release temp files / resources, on every path: clean
            # success, FAILED, COMPLETED_WITH_ERRORS, or a still-failing
            # finalization attempt.
            _cleanup(tmp_paths)

    # ── Keepalive stopped here — safe to call pika ────────────────────────────
    # Release the in-progress claim before the terminal ACK/NACK either way,
    # so a later genuine redelivery (e.g. after this worker instance itself
    # dies before reaching this point) is never permanently blocked.
    finish_job(job_id)
    if job_failed:
        _nack(channel, pika_lock, method.delivery_tag, job_id)
    else:
        # ── Step 10: ACK ─────────────────────────────────────────────────────
        # Also reached for the COMPLETED_WITH_ERRORS-after-interruption path:
        # the message is finalized here so the worker moves on to the next
        # queued journey, exactly as it does after a normal completion.
        _ack_and_flush(channel, connection, pika_lock,
                       method.delivery_tag, job_id)


def _cleanup(tmp_paths: Dict[int, str]) -> None:
    for path in tmp_paths.values():
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError as exc:
            log.warning("Could not remove temp file %s: %s", path, exc)


# ── RabbitMQ callback ─────────────────────────────────────────────────────────

def _on_message(channel, method, properties, body,
                connection, pika_lock: threading.Lock) -> None:
    log.info(
        "[RabbitMQ] Message received  delivery_tag=%s  redelivered=%s  body=%s",
        method.delivery_tag,
        method.redelivered,
        body.decode("utf-8", errors="replace")[:500],
    )
    if method.redelivered:
        log.warning(
            "[RabbitMQ] ⚠ REDELIVERED MESSAGE (delivery_tag=%s) — this typically means "
            "the previous processing attempt was interrupted before ACK. Possible causes: "
            "(1) RabbitMQ consumer_timeout fired (broker default 30 min) while the job "
            "was still running — fix: raise consumer_timeout on the broker to ≥12 hours; "
            "(2) worker process was killed/crashed mid-job. "
            "The local idempotency cache will prevent re-processing if this worker "
            "already completed the job in this session.",
            method.delivery_tag,
        )
    try:
        data = json.loads(body)
        msg  = AnalysisJobMessage.from_dict(data)
    except Exception as exc:
        log.error("Failed to parse RabbitMQ message: %s\nBody: %s",
                  exc, body[:500])
        with pika_lock:
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        return

    # ── Run this journey on its own thread ──────────────────────────────────
    # _handle_job() blocks for the entire journey (downloads + analysis +
    # callbacks), which used to mean pika's I/O loop — and therefore the
    # NEXT message — was stuck waiting behind it. With GPU_WORKERS persistent
    # workers now able to process journeys in parallel, we hand each journey
    # off to its own thread so _on_message can return immediately and let
    # pika deliver the next message (up to PREFETCH_COUNT, which defaults to
    # GPU_WORKERS). Concurrency is naturally bounded twice over: the broker
    # won't have more than PREFETCH_COUNT unacked messages in flight, and
    # worker_pool.submit() inside _handle_job blocks any thread beyond that
    # until a GPU worker is actually free. ACK/NACK and the per-job keepalive
    # thread already coordinate through the shared pika_lock, so multiple
    # of these job-handler threads running at once is safe the same way one
    # job-thread + one keepalive-thread running concurrently always was.
    threading.Thread(
        target = _handle_job,
        args   = (msg, channel, method, connection, pika_lock),
        name   = f"Job-{msg.job_id}",
        daemon = True,
    ).start()


# ── Consumer connect + consume ────────────────────────────────────────────────

def _connect_and_consume() -> None:
    log.info("[Consumer]  Connecting to RabbitMQ: %s  queue=%s",
             RABBITMQ_URL, QUEUE_NAME)
    params = pika.URLParameters(RABBITMQ_URL)
    params.heartbeat                  = HEARTBEAT_INTERVAL   # 60 s
    params.blocked_connection_timeout = 300
    # IMPORTANT: Do NOT set socket_timeout here. The default (no timeout) is
    # correct for a long-running consumer — a fixed socket_timeout would
    # drop the TCP connection after that many idle seconds, which for a
    # multi-hour analysis job (20 videos × ~400s = ~2.3 hours) would close
    # the socket long before the job finishes. Heartbeats (HEARTBEAT_INTERVAL)
    # and the keepalive thread (process_data_events every KEEPALIVE_INTERVAL
    # seconds) are sufficient to keep NAT devices and the broker happy.
    connection = pika.BlockingConnection(params)
    channel    = connection.channel()

    # One lock per connection — shared by _handle_job and _JobKeepalive.
    pika_lock = threading.Lock()

    channel.exchange_declare(
        exchange      = EXCHANGE_NAME,
        exchange_type = "direct",
        durable       = True,
    )

    # Declare the queue passively (it must already exist on the broker) but
    # log a clear warning if consumer_timeout needs to be raised on the broker.
    # RabbitMQ 3.8.15+ enforces a consumer_timeout (default 30 minutes) that
    # cancels any consumer that hasn't ACKed a message within that window —
    # even if the TCP connection and heartbeats are healthy. For journeys
    # longer than 30 min this causes the broker to cancel the consumer,
    # requeue the message, and the worker picks it up again as a redelivery.
    #
    # FIX ON THE BROKER (run once as rabbitmq admin):
    #   rabbitmqctl set_policy consumer-timeout \
    #     ".*" '{"consumer-timeout": 43200000}' \
    #     --apply-to queues
    # That sets a 12-hour consumer timeout (43200000 ms) on all queues.
    # Alternatively set consumer_timeout = false in rabbitmq.conf to disable it.
    #
    # The local idempotency cache in callback_client.py (_completed_jobs /
    # mark_job_completed) is the WORKER-SIDE guard: if the broker does
    # redeliver despite the above, the worker catches it and ACKs immediately
    # without re-processing.
    channel.queue_declare(queue=QUEUE_NAME, passive=True)
    channel.queue_bind(
        queue       = QUEUE_NAME,
        exchange    = EXCHANGE_NAME,
        routing_key = ROUTING_KEY,
    )

    channel.basic_qos(prefetch_count=PREFETCH_COUNT)
    channel.basic_consume(
        queue               = QUEUE_NAME,
        on_message_callback = partial(
            _on_message, connection=connection, pika_lock=pika_lock
        ),
    )

    log.info(
        "[Consumer]  Waiting for messages on queue '%s' "
        "(exchange='%s', routing='%s')  heartbeat=%ds  keepalive=%ds...",
        QUEUE_NAME, EXCHANGE_NAME, ROUTING_KEY,
        HEARTBEAT_INTERVAL, KEEPALIVE_INTERVAL,
    )
    channel.start_consuming()


# ── Consumer startup with reconnect retry ─────────────────────────────────────

def start() -> None:
    # ── Phase 2: startup recovery — MUST complete before RabbitMQ connects ───
    # Runs the full sequence: resource cleanup → GPU cleanup → temp file
    # cleanup → stale workspace cleanup → zombie-process detection. No
    # consumer.basic_consume() call happens until this returns.
    log.info("[Consumer]  Running startup resource cleanup...")
    try:
        resource_manager.initialize_service()
    except Exception:
        log.error("[Consumer]  initialize_service() raised — continuing "
                  "startup anyway (cleanup failures must never block the "
                  "worker from starting):\n%s", traceback.format_exc())

    # ── Phase 8: background RAM/GPU/thread monitor ───────────────────────────
    # Runs continuously for the lifetime of the process, independent of
    # per-journey cleanup, so slow leaks across many journeys are caught
    # too (not just spikes within a single journey).
    try:
        memory_monitor.start()
    except Exception:
        log.error("[Consumer]  memory_monitor.start() failed (non-fatal):\n%s",
                  traceback.format_exc())

    # ── GPU worker pool startup ───────────────────────────────────────────────
    # Spawns GPU_WORKERS persistent worker processes ONCE, here, before the
    # RabbitMQ consumer starts pulling messages. Each worker's YOLO/TensorRT/
    # CUDA context loads lazily on its first video and then stays resident
    # for the lifetime of the worker — see worker_pool.py / journey_runner.
    # run_worker_loop(). This replaces the old "spawn a fresh subprocess per
    # journey" behavior.
    log.info("[Consumer]  Starting GPU worker pool (GPU_WORKERS=%d)...", GPU_WORKERS)
    worker_pool.start()

    while True:
        try:
            _connect_and_consume()
        except KeyboardInterrupt:
            log.info("[Consumer]  Interrupted by user — shutting down.")
            break
        except Exception as exc:
            log.error(
                "[Consumer]  Connection lost: %s — reconnecting in %ds...",
                exc, RECONNECT_DELAY,
            )
            time.sleep(RECONNECT_DELAY)

    # ── Graceful shutdown ──────────────────────────────────────────────────────
    try:
        memory_monitor.stop()
    except Exception:
        pass
    try:
        worker_pool.shutdown()
    except Exception:
        log.error("[Consumer]  worker_pool.shutdown() failed:\n%s",
                  traceback.format_exc())
    try:
        resource_manager.cleanup_on_shutdown()
    except Exception:
        log.error("[Consumer]  cleanup_on_shutdown() failed:\n%s",
                  traceback.format_exc())


if __name__ == "__main__":
    start()