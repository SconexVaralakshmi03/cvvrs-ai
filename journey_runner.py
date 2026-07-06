# # """
# # journey_runner.py
# # ──────────────────
# # Runs inside an isolated CHILD PROCESS (multiprocessing.Process), spawned by
# # consumer.py for every journey. This is the fix for native OpenCV crashes
# # (cv2 OutOfMemoryError, cv2.pyd access violation, segfault) that previously
# # killed the ENTIRE worker — including RabbitMQ's connection thread and every
# # other in-flight job — because a native crash terminates the OS process and
# # no Python `except`, not even `except BaseException`, can run after that.

# # Why a subprocess fixes this
# # ────────────────────────────
# # A Python try/except can only catch things the interpreter is still alive
# # to raise. A genuine native allocation failure or access violation kills
# # the process before any Python exception object even exists. The only way
# # to "catch" that is from OUTSIDE the process: the parent (consumer.py)
# # spawns this file's `run_journey_in_subprocess()` target as a child, and
# # watches `process.exitcode` after `process.join()`. If the child died
# # abnormally, the parent treats it as a crash and is responsible for
# # deciding which videos were already completed vs. not.

# # How the parent finds out which videos already finished
# # ────────────────────────────────────────────────────────
# # `analyze_journey()` in analyzer.py already processes videos one at a time
# # in a for-loop. We pass it an optional `video_done_cb` (see the small patch
# # to analyzer.py) that fires immediately after each video — success OR
# # per-video-caught-failure — completes. Here, that callback pushes a small
# # picklable progress event onto `events_q` (a multiprocessing.Queue shared
# # with the parent). The parent drains this queue continuously, so even if
# # the child is killed by the OS one frame into video 3, the parent already
# # knows videos 1 and 2 finished (or failed) and only needs to mark video 3
# # (and anything after it) as failed — exactly matching "if a video can't be
# # processed due to OOM, treat the remaining videos as failed too."

# # The final result (video_results / wall_seconds / failed_videos) is sent
# # back over `result_q` only on a clean return. If the child dies, `result_q`
# # never receives anything — that absence IS the crash signal the parent
# # checks for.
# # """

# # from __future__ import annotations

# # import multiprocessing as mp
# # import os
# # import traceback
# # from typing import Dict, List

# # from analyzer import analyze_journey
# # from models import VideoJob


# # # Event dicts pushed onto events_q by video_done_cb, e.g.:
# # #   {"type": "video_done", "video_id": 12, "ok": True,  "error": None,
# # #    "error_type": None, "stack_trace": None, "reason": None}
# # #   {"type": "video_done", "video_id": 13, "ok": False, "error": "decode error",
# # #    "error_type": "DECODE_ERROR", "stack_trace": "...", "reason": None}
# # #   {"type": "video_done", "video_id": 14, "ok": False,
# # #    "error": "Not Processed - Worker Resource Exhaustion",
# # #    "error_type": "NOT_PROCESSED", "stack_trace": None,
# # #    "reason": "Not Processed - Worker Resource Exhaustion"}
# # #
# # # Final outcome pushed onto result_q (at most one item, only on clean exit):
# # #   {"type": "result", "video_results": [...], "wall_seconds": 87.3,
# # #    "failed_videos": {13: "decode error"}}
# # #   {"type": "error",  "traceback": "...", "message": "..."}


# # def run_journey_in_subprocess(
# #     job_id: str,
# #     journey_id: int,
# #     folder_name: str,
# #     video_jobs: List[VideoJob],
# #     tmp_paths: Dict[int, str],
# #     events_q: "mp.Queue",
# #     result_q: "mp.Queue",
# # ) -> None:
# #     """
# #     Target function for multiprocessing.Process. Runs entirely in the
# #     child. Must only communicate back to the parent via events_q/result_q
# #     (no shared memory, no return value — multiprocessing.Process ignores
# #     return values).
# #     """
# #     # Keep OpenCV/MKL/etc. from oversubscribing CPU when several of these
# #     # child processes run concurrently on the same host (multi-user load).
# #     os.environ.setdefault("OMP_NUM_THREADS", "1")
# #     os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

# #     def _video_done_cb(video_id: int, ok: bool, error: str | None,
# #                         error_type: str | None = None,
# #                         stack_trace: str | None = None,
# #                         reason: str | None = None) -> None:
# #         try:
# #             events_q.put(
# #                 {
# #                     "type":        "video_done",
# #                     "video_id":    video_id,
# #                     "ok":          ok,
# #                     "error":       error,
# #                     "error_type":  error_type,
# #                     "stack_trace": stack_trace,
# #                     "reason":      reason,
# #                 },
# #                 block=False,
# #             )
# #         except Exception:
# #             # Never let a full/broken events queue take down the child —
# #             # the parent's primary crash signal is exitcode + result_q
# #             # absence, the events queue is best-effort progress detail.
# #             pass

# #     try:
# #         video_results, wall_seconds, failed_videos = analyze_journey(
# #             job_id       = job_id,
# #             journey_id   = journey_id,
# #             folder_name  = folder_name,
# #             video_jobs   = video_jobs,
# #             tmp_paths    = tmp_paths,
# #             progress_cb  = None,   # progress callbacks stay in the parent
# #             video_done_cb = _video_done_cb,
# #         )
# #         result_q.put({
# #             "type":          "result",
# #             "video_results": [vr.to_dict() for vr in video_results],
# #             "wall_seconds":  wall_seconds,
# #             "failed_videos": failed_videos,
# #         })
# #     except BaseException as exc:  # noqa: BLE001 - last line of defense in the child
# #         # Covers ordinary Python exceptions raised above analyzer.py's own
# #         # per-video isolation (e.g. a bug in the dedup/frame-upload stage
# #         # that runs after the per-video loop). True native crashes (OOM,
# #         # access violation, segfault) will NOT reach this except — they
# #         # kill the process directly, which is exactly the case the parent
# #         # detects via process.exitcode / join() instead.
# #         try:
# #             result_q.put({
# #                 "type":      "error",
# #                 "message":   str(exc),
# #                 "traceback": traceback.format_exc(),
# #             })
# #         except Exception:
# #             pass


# """
# journey_runner.py
# ──────────────────
# Runs inside an isolated CHILD PROCESS (multiprocessing.Process), spawned by
# consumer.py for every journey. This is the fix for native OpenCV crashes
# (cv2 OutOfMemoryError, cv2.pyd access violation, segfault) that previously
# killed the ENTIRE worker — including RabbitMQ's connection thread and every
# other in-flight job — because a native crash terminates the OS process and
# no Python `except`, not even `except BaseException`, can run after that.

# Why a subprocess fixes this
# ────────────────────────────
# A Python try/except can only catch things the interpreter is still alive
# to raise. A genuine native allocation failure or access violation kills
# the process before any Python exception object even exists. The only way
# to "catch" that is from OUTSIDE the process: the parent (consumer.py)
# spawns this file's `run_journey_in_subprocess()` target as a child, and
# watches `process.exitcode` after `process.join()`. If the child died
# abnormally, the parent treats it as a crash and is responsible for
# deciding which videos were already completed vs. not.

# How the parent finds out which videos already finished
# ────────────────────────────────────────────────────────
# `analyze_journey()` in analyzer.py already processes videos one at a time
# in a for-loop. We pass it an optional `video_done_cb` (see the small patch
# to analyzer.py) that fires immediately after each video — success OR
# per-video-caught-failure — completes. Here, that callback pushes a small
# picklable progress event onto `events_q` (a multiprocessing.Queue shared
# with the parent). The parent drains this queue continuously, so even if
# the child is killed by the OS one frame into video 3, the parent already
# knows videos 1 and 2 finished (or failed) and only needs to mark video 3
# (and anything after it) as failed — exactly matching "if a video can't be
# processed due to OOM, treat the remaining videos as failed too."

# The final result (video_results / wall_seconds / failed_videos) is sent
# back over `result_q` only on a clean return. If the child dies, `result_q`
# never receives anything — that absence IS the crash signal the parent
# checks for.
# """

# from __future__ import annotations

# import multiprocessing as mp
# import os
# import traceback
# from typing import Dict, List

# from analyzer import analyze_journey
# from models import VideoJob


# # Event dicts pushed onto events_q by video_done_cb, e.g.:
# #   {"type": "video_done", "video_id": 12, "ok": True,  "error": None,
# #    "error_type": None, "stack_trace": None, "reason": None}
# #   {"type": "video_done", "video_id": 13, "ok": False, "error": "decode error",
# #    "error_type": "DECODE_ERROR", "stack_trace": "...", "reason": None}
# #   {"type": "video_done", "video_id": 14, "ok": False,
# #    "error": "Not Processed - Worker Resource Exhaustion",
# #    "error_type": "NOT_PROCESSED", "stack_trace": None,
# #    "reason": "Not Processed - Worker Resource Exhaustion"}
# #
# # Final outcome pushed onto result_q (at most one item, only on clean exit):
# #   {"type": "result", "video_results": [...], "wall_seconds": 87.3,
# #    "failed_videos": {13: "decode error"}}
# #   {"type": "error",  "traceback": "...", "message": "..."}


# def run_journey_in_subprocess(
#     job_id: str,
#     journey_id: int,
#     folder_name: str,
#     video_jobs: List[VideoJob],
#     tmp_paths: Dict[int, str],
#     events_q: "mp.Queue",
#     result_q: "mp.Queue",
# ) -> None:
#     """
#     Target function for multiprocessing.Process. Runs entirely in the
#     child. Must only communicate back to the parent via events_q/result_q
#     (no shared memory, no return value — multiprocessing.Process ignores
#     return values).
#     """
#     # Keep OpenCV/MKL/etc. from oversubscribing CPU when several of these
#     # child processes run concurrently on the same host (multi-user load).
#     os.environ.setdefault("OMP_NUM_THREADS", "1")
#     os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

#     # ── Resource lifecycle (Phase 3): clean slate before this journey ────────
#     # This is a brand-new `spawn`'d interpreter, so there is no Python-level
#     # model object here yet. But the GPU itself may still be holding VRAM
#     # reserved by a PREVIOUS journey's process if the driver hasn't finished
#     # tearing down that process's CUDA context yet (this can lag process
#     # exit, especially under back-to-back journeys). cleanup_before_journey()
#     # is a no-op if the device is already clean, and prevents this journey's
#     # model load from fighting over memory that should have been freed.
#     try:
#         from resource_manager import resource_manager
#         resource_manager.cleanup_before_journey(job_id=job_id)
#     except Exception:
#         pass  # resource_manager unavailable — not fatal, proceed anyway

#     def _video_done_cb(video_id: int, ok: bool, error: str | None,
#                         error_type: str | None = None,
#                         stack_trace: str | None = None,
#                         reason: str | None = None) -> None:
#         try:
#             events_q.put(
#                 {
#                     "type":        "video_done",
#                     "video_id":    video_id,
#                     "ok":          ok,
#                     "error":       error,
#                     "error_type":  error_type,
#                     "stack_trace": stack_trace,
#                     "reason":      reason,
#                 },
#                 block=False,
#             )
#         except Exception:
#             # Never let a full/broken events queue take down the child —
#             # the parent's primary crash signal is exitcode + result_q
#             # absence, the events queue is best-effort progress detail.
#             pass
#         # ── Resource lifecycle (Phase 3): per-video GPU cleanup ──────────────
#         # Fired after every video, success or failure, so VRAM fragmentation
#         # doesn't accumulate across the videos within a single journey.
#         try:
#             from resource_manager import resource_manager
#             resource_manager.cleanup_after_video(job_id=job_id, video_id=video_id)
#         except Exception:
#             pass

#     try:
#         video_results, wall_seconds, failed_videos = analyze_journey(
#             job_id       = job_id,
#             journey_id   = journey_id,
#             folder_name  = folder_name,
#             video_jobs   = video_jobs,
#             tmp_paths    = tmp_paths,
#             progress_cb  = None,   # progress callbacks stay in the parent
#             video_done_cb = _video_done_cb,
#         )
#         result_q.put({
#             "type":          "result",
#             "video_results": [vr.to_dict() for vr in video_results],
#             "wall_seconds":  wall_seconds,
#             "failed_videos": failed_videos,
#         })
#     except BaseException as exc:  # noqa: BLE001 - last line of defense in the child
#         # Covers ordinary Python exceptions raised above analyzer.py's own
#         # per-video isolation (e.g. a bug in the dedup/frame-upload stage
#         # that runs after the per-video loop). True native crashes (OOM,
#         # access violation, segfault) will NOT reach this except — they
#         # kill the process directly, which is exactly the case the parent
#         # detects via process.exitcode / join() instead.
#         try:
#             from resource_manager import resource_manager
#             resource_manager.cleanup_after_failure(
#                 job_id=job_id, reason=f"{type(exc).__name__}: {exc}",
#             )
#         except Exception:
#             pass
#         try:
#             result_q.put({
#                 "type":      "error",
#                 "message":   str(exc),
#                 "traceback": traceback.format_exc(),
#             })
#         except Exception:
#             pass
#     finally:
#         # ── Resource lifecycle (Phase 3/4): release model + VRAM before exit ─
#         # This process is about to die anyway (each journey gets a fresh
#         # `spawn`'d process), but doing this explicitly — rather than
#         # relying solely on OS/driver cleanup timing after process exit —
#         # means the NEXT journey's child process is far less likely to
#         # start while this one's CUDA context is still being torn down.
#         # Runs on every path: clean success, caught exception above, AND
#         # — to the extent Python is still alive to run a finally at all —
#         # anything else. A genuine native crash (OOM/access violation)
#         # skips this entirely; that case is handled by the parent's
#         # exitcode-based emergency_cleanup() in consumer.py instead.
#         try:
#             from resource_manager import resource_manager
#             resource_manager.cleanup_after_journey(job_id=job_id)
#         except Exception:
#             pass


# """
# journey_runner.py
# ──────────────────
# Runs inside an isolated CHILD PROCESS (multiprocessing.Process), spawned by
# consumer.py for every journey. This is the fix for native OpenCV crashes
# (cv2 OutOfMemoryError, cv2.pyd access violation, segfault) that previously
# killed the ENTIRE worker — including RabbitMQ's connection thread and every
# other in-flight job — because a native crash terminates the OS process and
# no Python `except`, not even `except BaseException`, can run after that.

# Why a subprocess fixes this
# ────────────────────────────
# A Python try/except can only catch things the interpreter is still alive
# to raise. A genuine native allocation failure or access violation kills
# the process before any Python exception object even exists. The only way
# to "catch" that is from OUTSIDE the process: the parent (consumer.py)
# spawns this file's `run_journey_in_subprocess()` target as a child, and
# watches `process.exitcode` after `process.join()`. If the child died
# abnormally, the parent treats it as a crash and is responsible for
# deciding which videos were already completed vs. not.

# How the parent finds out which videos already finished
# ────────────────────────────────────────────────────────
# `analyze_journey()` in analyzer.py already processes videos one at a time
# in a for-loop. We pass it an optional `video_done_cb` (see the small patch
# to analyzer.py) that fires immediately after each video — success OR
# per-video-caught-failure — completes. Here, that callback pushes a small
# picklable progress event onto `events_q` (a multiprocessing.Queue shared
# with the parent). The parent drains this queue continuously, so even if
# the child is killed by the OS one frame into video 3, the parent already
# knows videos 1 and 2 finished (or failed) and only needs to mark video 3
# (and anything after it) as failed — exactly matching "if a video can't be
# processed due to OOM, treat the remaining videos as failed too."

# The final result (video_results / wall_seconds / failed_videos) is sent
# back over `result_q` only on a clean return. If the child dies, `result_q`
# never receives anything — that absence IS the crash signal the parent
# checks for.
# """

# from __future__ import annotations

# import multiprocessing as mp
# import os
# import traceback
# from typing import Dict, List

# from analyzer import analyze_journey
# from models import VideoJob


# # Event dicts pushed onto events_q by video_done_cb, e.g.:
# #   {"type": "video_done", "video_id": 12, "ok": True,  "error": None,
# #    "error_type": None, "stack_trace": None, "reason": None}
# #   {"type": "video_done", "video_id": 13, "ok": False, "error": "decode error",
# #    "error_type": "DECODE_ERROR", "stack_trace": "...", "reason": None}
# #   {"type": "video_done", "video_id": 14, "ok": False,
# #    "error": "Not Processed - Worker Resource Exhaustion",
# #    "error_type": "NOT_PROCESSED", "stack_trace": None,
# #    "reason": "Not Processed - Worker Resource Exhaustion"}
# #
# # Final outcome pushed onto result_q (at most one item, only on clean exit):
# #   {"type": "result", "video_results": [...], "wall_seconds": 87.3,
# #    "failed_videos": {13: "decode error"}}
# #   {"type": "error",  "traceback": "...", "message": "..."}


# def run_journey_in_subprocess(
#     job_id: str,
#     journey_id: int,
#     folder_name: str,
#     video_jobs: List[VideoJob],
#     tmp_paths: Dict[int, str],
#     events_q: "mp.Queue",
#     result_q: "mp.Queue",
# ) -> None:
#     """
#     Target function for multiprocessing.Process. Runs entirely in the
#     child. Must only communicate back to the parent via events_q/result_q
#     (no shared memory, no return value — multiprocessing.Process ignores
#     return values).
#     """
#     # Keep OpenCV/MKL/etc. from oversubscribing CPU when several of these
#     # child processes run concurrently on the same host (multi-user load).
#     os.environ.setdefault("OMP_NUM_THREADS", "1")
#     os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

#     def _video_done_cb(video_id: int, ok: bool, error: str | None,
#                         error_type: str | None = None,
#                         stack_trace: str | None = None,
#                         reason: str | None = None) -> None:
#         try:
#             events_q.put(
#                 {
#                     "type":        "video_done",
#                     "video_id":    video_id,
#                     "ok":          ok,
#                     "error":       error,
#                     "error_type":  error_type,
#                     "stack_trace": stack_trace,
#                     "reason":      reason,
#                 },
#                 block=False,
#             )
#         except Exception:
#             # Never let a full/broken events queue take down the child —
#             # the parent's primary crash signal is exitcode + result_q
#             # absence, the events queue is best-effort progress detail.
#             pass

#     try:
#         video_results, wall_seconds, failed_videos = analyze_journey(
#             job_id       = job_id,
#             journey_id   = journey_id,
#             folder_name  = folder_name,
#             video_jobs   = video_jobs,
#             tmp_paths    = tmp_paths,
#             progress_cb  = None,   # progress callbacks stay in the parent
#             video_done_cb = _video_done_cb,
#         )
#         result_q.put({
#             "type":          "result",
#             "video_results": [vr.to_dict() for vr in video_results],
#             "wall_seconds":  wall_seconds,
#             "failed_videos": failed_videos,
#         })
#     except BaseException as exc:  # noqa: BLE001 - last line of defense in the child
#         # Covers ordinary Python exceptions raised above analyzer.py's own
#         # per-video isolation (e.g. a bug in the dedup/frame-upload stage
#         # that runs after the per-video loop). True native crashes (OOM,
#         # access violation, segfault) will NOT reach this except — they
#         # kill the process directly, which is exactly the case the parent
#         # detects via process.exitcode / join() instead.
#         try:
#             result_q.put({
#                 "type":      "error",
#                 "message":   str(exc),
#                 "traceback": traceback.format_exc(),
#             })
#         except Exception:
#             pass


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
# Final outcome pushed onto result_q (at most one item, only on clean exit):
#   {"type": "result", "video_results": [...], "wall_seconds": 87.3,
#    "failed_videos": {13: "decode error"}}
#   {"type": "error",  "traceback": "...", "message": "..."}


def run_journey_in_subprocess(
    job_id: str,
    journey_id: int,
    folder_name: str,
    video_jobs: List[VideoJob],
    tmp_paths: Dict[int, str],
    events_q: "mp.Queue",
    result_q: "mp.Queue",
) -> None:
    """
    Target function for multiprocessing.Process. Runs entirely in the
    child. Must only communicate back to the parent via events_q/result_q
    (no shared memory, no return value — multiprocessing.Process ignores
    return values).
    """
    # Keep OpenCV/MKL/etc. from oversubscribing CPU when several of these
    # child processes run concurrently on the same host (multi-user load).
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

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
            progress_cb  = None,   # progress callbacks stay in the parent
            video_done_cb = _video_done_cb,
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
        # ── Resource lifecycle (Phase 3/4): release model + VRAM before exit ─
        # This process is about to die anyway (each journey gets a fresh
        # `spawn`'d process), but doing this explicitly — rather than
        # relying solely on OS/driver cleanup timing after process exit —
        # means the NEXT journey's child process is far less likely to
        # start while this one's CUDA context is still being torn down.
        # Runs on every path: clean success, caught exception above, AND
        # — to the extent Python is still alive to run a finally at all —
        # anything else. A genuine native crash (OOM/access violation)
        # skips this entirely; that case is handled by the parent's
        # exitcode-based emergency_cleanup() in consumer.py instead.
        try:
            from resource_manager import resource_manager
            resource_manager.cleanup_after_journey(job_id=job_id)
        except Exception:
            pass