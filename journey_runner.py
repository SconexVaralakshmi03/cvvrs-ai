"""
journey_runner.py
──────────────────
Runs inside an isolated CHILD PROCESS (multiprocessing.Process), spawned by
consumer.py for every journey. This is the fix for native OpenCV crashes
(cv2 OutOfMemoryError, cv2.pyd access violation, segfault) that previously
killed the ENTIRE worker — including RabbitMQ's connection thread and every
other in-flight job — because a native crash terminates the OS process and
no Python `except`, not even `except BaseException`, can run after that.

Why a subprocess fixes this
────────────────────────────
A Python try/except can only catch things the interpreter is still alive
to raise. A genuine native allocation failure or access violation kills
the process before any Python exception object even exists. The only way
to "catch" that is from OUTSIDE the process: the parent (consumer.py)
spawns this file's `run_journey_in_subprocess()` target as a child, and
watches `process.exitcode` after `process.join()`. If the child died
abnormally, the parent treats it as a crash and is responsible for
deciding which videos were already completed vs. not.

How the parent finds out which videos already finished
────────────────────────────────────────────────────────
`analyze_journey()` in analyzer.py already processes videos one at a time
in a for-loop. We pass it an optional `video_done_cb` (see the small patch
to analyzer.py) that fires immediately after each video — success OR
per-video-caught-failure — completes. Here, that callback pushes a small
picklable progress event onto `events_q` (a multiprocessing.Queue shared
with the parent). The parent drains this queue continuously, so even if
the child is killed by the OS one frame into video 3, the parent already
knows videos 1 and 2 finished (or failed) and only needs to mark video 3
(and anything after it) as failed — exactly matching "if a video can't be
processed due to OOM, treat the remaining videos as failed too."

The final result (video_results / wall_seconds / failed_videos) is sent
back over `result_q` only on a clean return. If the child dies, `result_q`
never receives anything — that absence IS the crash signal the parent
checks for.

CVVRS reliability fixes applied in this file:
  • Fix 1/6/7 — a `_video_progress_cb` is now wired into
    `analyze_journey()` (see analyzer.py) alongside the existing
    `_video_done_cb`, pushing lightweight `{"type": "progress", ...}`
    events onto the same `events_q` the parent (consumer.py) already
    polls. This gives the parent's watchdog real per-frame progress
    (current video, current frame, processed frames, last progress time)
    instead of only "a video finished" events — the basis for a
    progress-based watchdog instead of a fixed wall-clock journey
    timeout.
  • Fix 7 — `_process_one_journey_impl` / `_process_one_journey` /
    `run_journey_in_subprocess` / `run_worker_loop` now accept optional
    `initial_time_offset` / `initial_frame_offset`, forwarded straight
    through to `analyze_journey()`, so consumer.py can resubmit just the
    REMAINING videos of a journey (after a stuck video was detected and
    its worker discarded) with journey-global timestamps still continuous.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import traceback
from typing import Dict, List

from analyzer import analyze_journey
from models import VideoJob


# Event dicts pushed onto events_q by video_done_cb, e.g.:
#   {"type": "video_done", "video_id": 12, "ok": True,  "error": None,
#    "error_type": None, "stack_trace": None, "reason": None}
#   {"type": "video_done", "video_id": 13, "ok": False, "error": "decode error",
#    "error_type": "DECODE_ERROR", "stack_trace": "...", "reason": None}
#   {"type": "video_done", "video_id": 14, "ok": False,
#    "error": "Not Processed - Worker Resource Exhaustion",
#    "error_type": "NOT_PROCESSED", "stack_trace": None,
#    "reason": "Not Processed - Worker Resource Exhaustion"}
#
# Fix 1/6 — progress events, pushed frequently WHILE a video is being
# processed (not just when it finishes):
#   {"type": "progress", "video_id": 14, "current_frame": 35200,
#    "processed_frames": 11733, "total_frames": 55000,
#    "last_progress_time": 1737000000.123}
#
# Final outcome pushed onto result_q (at most one item, only on clean exit):
#   {"type": "result", "video_results": [...], "wall_seconds": 87.3,
#    "failed_videos": {13: "decode error"}}
#   {"type": "error",  "traceback": "...", "message": "..."}


def _process_one_journey(
    job_id: str,
    journey_id: int,
    folder_name: str,
    video_jobs: List[VideoJob],
    tmp_paths: Dict[int, str],
    events_q: "mp.Queue",
    result_q: "mp.Queue",
    initial_time_offset: float = 0.0,
    initial_frame_offset: int = 0,
) -> None:
    """
    Thin wrapper around _process_one_journey_impl() that captures every
    print()/stdout line this journey produces (the detection pipeline's own
    output — banners, per-video progress, violation lines — is almost
    entirely bare print(), not the logging module) into
    logs/jobs/<job_id>.log, alongside the consumer-side log lines for the
    same job_id (see job_logging.py). Safe here specifically because a
    persistent GPU worker only ever runs one journey at a time.
    """
    import job_logging
    with job_logging.JobStdoutTee(job_id):
        _process_one_journey_impl(
            job_id, journey_id, folder_name, video_jobs, tmp_paths,
            events_q, result_q,
            initial_time_offset=initial_time_offset,
            initial_frame_offset=initial_frame_offset,
        )


def _process_one_journey_impl(
    job_id: str,
    journey_id: int,
    folder_name: str,
    video_jobs: List[VideoJob],
    tmp_paths: Dict[int, str],
    events_q: "mp.Queue",
    result_q: "mp.Queue",
    initial_time_offset: float = 0.0,
    initial_frame_offset: int = 0,
) -> None:
    """
    Runs exactly ONE journey's analysis and reports back over
    events_q/result_q — the same wire format regardless of whether this is
    called from a one-shot subprocess (run_journey_in_subprocess, kept for
    backward compatibility) or from inside a long-lived GPU worker process's
    main loop (run_worker_loop, used by the persistent worker-pool
    architecture — see worker_pool.py).

    IMPORTANT: this function does NOT load or release the YOLO model. The
    model is a per-process lazy singleton owned by gadget_detector.py
    (_get_model() / release_model()) and analyzer.py's per-video pipeline
    never tears it down between videos or journeys. In the persistent
    worker-pool architecture that means the model loads once — on the
    first video of the first journey this worker process ever handles —
    and then stays resident in this process's memory/CUDA context for
    every journey the worker processes afterwards, for as long as the
    worker lives. Only the per-journey temporary resources (VideoCapture,
    frames, numpy arrays, temp CUDA tensors) are released below, via
    resource_manager's cleanup hooks — never the model itself.
    """
    # ── Resource lifecycle (Phase 3): clean slate before this journey ────────
    # This is a brand-new `spawn`'d interpreter, so there is no Python-level
    # model object here yet. But the GPU itself may still be holding VRAM
    # reserved by a PREVIOUS journey's process if the driver hasn't finished
    # tearing down that process's CUDA context yet (this can lag process
    # exit, especially under back-to-back journeys). cleanup_before_journey()
    # is a no-op if the device is already clean, and prevents this journey's
    # model load from fighting over memory that should have been freed.
    try:
        from resource_manager import resource_manager
        resource_manager.cleanup_before_journey(job_id=job_id)
    except Exception:
        pass  # resource_manager unavailable — not fatal, proceed anyway

    # ── Fix 1/6: per-video progress heartbeat ─────────────────────────────
    # Pushed frequently WHILE a video is processing (see analyzer.py's
    # video_progress_cb wiring into main.py's per-frame progress_cb).
    # Best-effort / non-blocking — never allowed to disrupt the frame loop.
    def _video_progress_cb(video_id: int, current_frame: int,
                            processed_frames: int, total_frames: int,
                            last_progress_time: float) -> None:
        try:
            events_q.put(
                {
                    "type":               "progress",
                    "video_id":           video_id,
                    "current_frame":      current_frame,
                    "processed_frames":   processed_frames,
                    "total_frames":       total_frames,
                    "last_progress_time": last_progress_time,
                },
                block=False,
            )
        except Exception:
            pass

    def _video_done_cb(video_id: int, ok: bool, error: str | None,
                        error_type: str | None = None,
                        stack_trace: str | None = None,
                        reason: str | None = None,
                        video_result: dict | None = None) -> None:
        try:
            events_q.put(
                {
                    "type":         "video_done",
                    "video_id":     video_id,
                    "ok":           ok,
                    "error":        error,
                    "error_type":   error_type,
                    "stack_trace":  stack_trace,
                    "reason":       reason,
                    # Best-effort per-video result snapshot (see analyzer.py's
                    # _build_partial_video_result) — only present when ok=True.
                    # Lets the parent preserve real violation data for videos
                    # that finished before a native crash kills this child,
                    # instead of having to mark them failed too.
                    "video_result": video_result,
                },
                block=False,
            )
        except Exception:
            # Never let a full/broken events queue take down the child —
            # the parent's primary crash signal is exitcode + result_q
            # absence, the events queue is best-effort progress detail.
            pass
        # ── Resource lifecycle (Phase 3): per-video GPU cleanup ──────────────
        # Fired after every video, success or failure, so VRAM fragmentation
        # doesn't accumulate across the videos within a single journey.
        try:
            from resource_manager import resource_manager
            resource_manager.cleanup_after_video(job_id=job_id, video_id=video_id)
        except Exception:
            pass

    try:
        video_results, wall_seconds, failed_videos = analyze_journey(
            job_id       = job_id,
            journey_id   = journey_id,
            folder_name  = folder_name,
            video_jobs   = video_jobs,
            tmp_paths    = tmp_paths,
            progress_cb  = None,   # percent-complete callbacks stay in the parent
            video_done_cb = _video_done_cb,
            video_progress_cb = _video_progress_cb,
            initial_time_offset  = initial_time_offset,
            initial_frame_offset = initial_frame_offset,
        )
        result_q.put({
            "type":          "result",
            "video_results": [vr.to_dict() for vr in video_results],
            "wall_seconds":  wall_seconds,
            "failed_videos": failed_videos,
        })
    except BaseException as exc:  # noqa: BLE001 - last line of defense in the child
        # Covers ordinary Python exceptions raised above analyzer.py's own
        # per-video isolation (e.g. a bug in the dedup/frame-upload stage
        # that runs after the per-video loop). True native crashes (OOM,
        # access violation, segfault) will NOT reach this except — they
        # kill the process directly, which is exactly the case the parent
        # detects via process.exitcode / join() instead.
        try:
            from resource_manager import resource_manager
            resource_manager.cleanup_after_failure(
                job_id=job_id, reason=f"{type(exc).__name__}: {exc}",
            )
        except Exception:
            pass
        try:
            result_q.put({
                "type":      "error",
                "message":   str(exc),
                "traceback": traceback.format_exc(),
            })
        except Exception:
            pass
    finally:
        # ── Resource lifecycle (Phase 3/4): release TEMPORARY resources ──────
        # cleanup_after_journey() flushes the CUDA allocator cache / runs GC
        # and logs RSS/GPU deltas — it deliberately does NOT touch the YOLO
        # model singleton (see gadget_detector.py's release_model() docstring
        # and resource_manager.cleanup_after_journey()'s docstring). That is
        # exactly what makes this function safe to call from inside a
        # persistent worker's loop (run_worker_loop below): every journey
        # gets a clean VRAM slate, but the model stays loaded across
        # journeys for the lifetime of the worker process. Runs on every
        # path: clean success, caught exception above, AND — to the extent
        # Python is still alive to run a finally at all — anything else. A
        # genuine native crash (OOM/access violation) skips this entirely;
        # that case is handled by worker_pool.py's crash detection instead
        # (the dead worker process is replaced with a fresh one).
        try:
            from resource_manager import resource_manager
            resource_manager.cleanup_after_journey(job_id=job_id)
        except Exception:
            pass


def run_journey_in_subprocess(
    job_id: str,
    journey_id: int,
    folder_name: str,
    video_jobs: List[VideoJob],
    tmp_paths: Dict[int, str],
    events_q: "mp.Queue",
    result_q: "mp.Queue",
    initial_time_offset: float = 0.0,
    initial_frame_offset: int = 0,
) -> None:
    """
    Back-compat ONE-SHOT entry point: target function for a throwaway
    multiprocessing.Process that processes exactly one journey and then
    exits. Kept for any caller (tests, CLI tooling) that still wants the
    old "fresh process per journey" behavior. The live worker-pool
    architecture (consumer.py + worker_pool.py) uses run_worker_loop()
    below instead, so the model doesn't get reloaded every journey.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")
    _process_one_journey(
        job_id, journey_id, folder_name, video_jobs, tmp_paths,
        events_q, result_q,
        initial_time_offset=initial_time_offset,
        initial_frame_offset=initial_frame_offset,
    )


def run_worker_loop(
    worker_id: int,
    job_queue: "mp.Queue",
    events_q: "mp.Queue",
    result_q: "mp.Queue",
) -> None:
    """
    Target function for a PERSISTENT GPU worker process (multiprocessing
    .Process, spawned once at service startup by worker_pool.py and kept
    alive for the lifetime of the service).

    Loads nothing eagerly. The YOLO model (gadget_detector._get_model())
    lazy-loads on the first video of the first journey this worker
    processes, and then stays resident — model weights, CUDA context, the
    works — in this process's memory for every subsequent journey handed
    to it, for as long as this worker lives. That is the entire point of
    the persistent worker-pool architecture: journeys arrive one after
    another on job_queue, but "Load YOLO / Load TensorRT / Initialize
    CUDA" only ever happens once per worker, not once per journey.

    job_queue protocol
    ─────────────
    Each item is either:
      • a 5-tuple (job_id, journey_id, folder_name, video_jobs, tmp_paths)
        — process this journey, then loop back and wait for the next one.
      • a 7-tuple (job_id, journey_id, folder_name, video_jobs, tmp_paths,
        initial_time_offset, initial_frame_offset) — Fix 7: same, but for
        a RETRY submission covering only the videos remaining after a
        stuck video was detected elsewhere in this journey; the offsets
        keep journey-global violation timestamps continuous.
      • None — shutdown sentinel; exit the loop and let the process end.

    events_q / result_q are the SAME pair of queues for every journey this
    worker ever processes (created once by worker_pool.py alongside
    job_queue) — the caller distinguishes one journey's events from
    another's by only ever having one journey in flight per worker at a
    time (worker_pool.py enforces this: a worker is only handed a new job
    once the previous one's result has been consumed).
    """
    # Keep OpenCV/MKL/etc. from oversubscribing CPU when several worker
    # processes run concurrently on the same host.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

    print(f"[GPUWorker-{worker_id}]  Started  pid={os.getpid()}  "
          f"— waiting for journeys (model loads lazily on first video)...")

    while True:
        job = job_queue.get()  # blocks until a journey arrives or shutdown
        if job is None:
            print(f"[GPUWorker-{worker_id}]  Shutdown signal received — exiting.")
            break

        if len(job) == 7:
            (job_id, journey_id, folder_name, video_jobs, tmp_paths,
             initial_time_offset, initial_frame_offset) = job
        else:
            job_id, journey_id, folder_name, video_jobs, tmp_paths = job
            initial_time_offset, initial_frame_offset = 0.0, 0
        print(f"[GPUWorker-{worker_id}]  Picked up journey job={job_id} "
              f"({len(video_jobs)} video(s)"
              f"{', retry/continuation' if initial_time_offset else ''}).")
        try:
            _process_one_journey(
                job_id, journey_id, folder_name, video_jobs, tmp_paths,
                events_q, result_q,
                initial_time_offset=initial_time_offset,
                initial_frame_offset=initial_frame_offset,
            )
        except BaseException as exc:  # noqa: BLE001 - last line of defense
            # _process_one_journey already catches everything it can and
            # reports via result_q itself; this only fires for something
            # unexpected in the loop plumbing around it. A true native
            # crash (OOM/access violation/segfault) will NOT reach this
            # except — it kills the process directly, which worker_pool.py
            # detects via is_alive() and handles by spawning a replacement.
            try:
                result_q.put({
                    "type":      "error",
                    "message":   str(exc),
                    "traceback": traceback.format_exc(),
                })
            except Exception:
                pass
        # ── Loop back — model/CUDA context stay loaded ────────────────────────
        # Only this journey's temporary resources were released above
        # (resource_manager.cleanup_after_journey(), inside
        # _process_one_journey's finally block). The worker is now idle
        # and ready for the next journey worker_pool.py assigns it.
        print(f"[GPUWorker-{worker_id}]  Finished journey job={job_id} "
              f"— waiting for next journey.")