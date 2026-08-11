# # # # # """
# # # # # journey_runner.py
# # # # # ──────────────────
# # # # # Runs inside an isolated CHILD PROCESS (multiprocessing.Process), spawned by
# # # # # consumer.py for every journey. This is the fix for native OpenCV crashes
# # # # # (cv2 OutOfMemoryError, cv2.pyd access violation, segfault) that previously
# # # # # killed the ENTIRE worker — including RabbitMQ's connection thread and every
# # # # # other in-flight job — because a native crash terminates the OS process and
# # # # # no Python `except`, not even `except BaseException`, can run after that.

# # # # # Why a subprocess fixes this
# # # # # ────────────────────────────
# # # # # A Python try/except can only catch things the interpreter is still alive
# # # # # to raise. A genuine native allocation failure or access violation kills
# # # # # the process before any Python exception object even exists. The only way
# # # # # to "catch" that is from OUTSIDE the process: the parent (consumer.py)
# # # # # spawns this file's `run_journey_in_subprocess()` target as a child, and
# # # # # watches `process.exitcode` after `process.join()`. If the child died
# # # # # abnormally, the parent treats it as a crash and is responsible for
# # # # # deciding which videos were already completed vs. not.

# # # # # How the parent finds out which videos already finished
# # # # # ────────────────────────────────────────────────────────
# # # # # `analyze_journey()` in analyzer.py already processes videos one at a time
# # # # # in a for-loop. We pass it an optional `video_done_cb` (see the small patch
# # # # # to analyzer.py) that fires immediately after each video — success OR
# # # # # per-video-caught-failure — completes. Here, that callback pushes a small
# # # # # picklable progress event onto `events_q` (a multiprocessing.Queue shared
# # # # # with the parent). The parent drains this queue continuously, so even if
# # # # # the child is killed by the OS one frame into video 3, the parent already
# # # # # knows videos 1 and 2 finished (or failed) and only needs to mark video 3
# # # # # (and anything after it) as failed — exactly matching "if a video can't be
# # # # # processed due to OOM, treat the remaining videos as failed too."

# # # # # The final result (video_results / wall_seconds / failed_videos) is sent
# # # # # back over `result_q` only on a clean return. If the child dies, `result_q`
# # # # # never receives anything — that absence IS the crash signal the parent
# # # # # checks for.
# # # # # """

# # # # # from __future__ import annotations

# # # # # import multiprocessing as mp
# # # # # import os
# # # # # import traceback
# # # # # from typing import Dict, List

# # # # # from analyzer import analyze_journey
# # # # # from models import VideoJob


# # # # # # Event dicts pushed onto events_q by video_done_cb, e.g.:
# # # # # #   {"type": "video_done", "video_id": 12, "ok": True,  "error": None,
# # # # # #    "error_type": None, "stack_trace": None, "reason": None}
# # # # # #   {"type": "video_done", "video_id": 13, "ok": False, "error": "decode error",
# # # # # #    "error_type": "DECODE_ERROR", "stack_trace": "...", "reason": None}
# # # # # #   {"type": "video_done", "video_id": 14, "ok": False,
# # # # # #    "error": "Not Processed - Worker Resource Exhaustion",
# # # # # #    "error_type": "NOT_PROCESSED", "stack_trace": None,
# # # # # #    "reason": "Not Processed - Worker Resource Exhaustion"}
# # # # # #
# # # # # # Final outcome pushed onto result_q (at most one item, only on clean exit):
# # # # # #   {"type": "result", "video_results": [...], "wall_seconds": 87.3,
# # # # # #    "failed_videos": {13: "decode error"}}
# # # # # #   {"type": "error",  "traceback": "...", "message": "..."}


# # # # # def run_journey_in_subprocess(
# # # # #     job_id: str,
# # # # #     journey_id: int,
# # # # #     folder_name: str,
# # # # #     video_jobs: List[VideoJob],
# # # # #     tmp_paths: Dict[int, str],
# # # # #     events_q: "mp.Queue",
# # # # #     result_q: "mp.Queue",
# # # # # ) -> None:
# # # # #     """
# # # # #     Target function for multiprocessing.Process. Runs entirely in the
# # # # #     child. Must only communicate back to the parent via events_q/result_q
# # # # #     (no shared memory, no return value — multiprocessing.Process ignores
# # # # #     return values).
# # # # #     """
# # # # #     # Keep OpenCV/MKL/etc. from oversubscribing CPU when several of these
# # # # #     # child processes run concurrently on the same host (multi-user load).
# # # # #     os.environ.setdefault("OMP_NUM_THREADS", "1")
# # # # #     os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

# # # # #     def _video_done_cb(video_id: int, ok: bool, error: str | None,
# # # # #                         error_type: str | None = None,
# # # # #                         stack_trace: str | None = None,
# # # # #                         reason: str | None = None) -> None:
# # # # #         try:
# # # # #             events_q.put(
# # # # #                 {
# # # # #                     "type":        "video_done",
# # # # #                     "video_id":    video_id,
# # # # #                     "ok":          ok,
# # # # #                     "error":       error,
# # # # #                     "error_type":  error_type,
# # # # #                     "stack_trace": stack_trace,
# # # # #                     "reason":      reason,
# # # # #                 },
# # # # #                 block=False,
# # # # #             )
# # # # #         except Exception:
# # # # #             # Never let a full/broken events queue take down the child —
# # # # #             # the parent's primary crash signal is exitcode + result_q
# # # # #             # absence, the events queue is best-effort progress detail.
# # # # #             pass

# # # # #     try:
# # # # #         video_results, wall_seconds, failed_videos = analyze_journey(
# # # # #             job_id       = job_id,
# # # # #             journey_id   = journey_id,
# # # # #             folder_name  = folder_name,
# # # # #             video_jobs   = video_jobs,
# # # # #             tmp_paths    = tmp_paths,
# # # # #             progress_cb  = None,   # progress callbacks stay in the parent
# # # # #             video_done_cb = _video_done_cb,
# # # # #         )
# # # # #         result_q.put({
# # # # #             "type":          "result",
# # # # #             "video_results": [vr.to_dict() for vr in video_results],
# # # # #             "wall_seconds":  wall_seconds,
# # # # #             "failed_videos": failed_videos,
# # # # #         })
# # # # #     except BaseException as exc:  # noqa: BLE001 - last line of defense in the child
# # # # #         # Covers ordinary Python exceptions raised above analyzer.py's own
# # # # #         # per-video isolation (e.g. a bug in the dedup/frame-upload stage
# # # # #         # that runs after the per-video loop). True native crashes (OOM,
# # # # #         # access violation, segfault) will NOT reach this except — they
# # # # #         # kill the process directly, which is exactly the case the parent
# # # # #         # detects via process.exitcode / join() instead.
# # # # #         try:
# # # # #             result_q.put({
# # # # #                 "type":      "error",
# # # # #                 "message":   str(exc),
# # # # #                 "traceback": traceback.format_exc(),
# # # # #             })
# # # # #         except Exception:
# # # # #             pass


# # # # """
# # # # journey_runner.py
# # # # ──────────────────
# # # # Runs inside an isolated CHILD PROCESS (multiprocessing.Process), spawned by
# # # # consumer.py for every journey. This is the fix for native OpenCV crashes
# # # # (cv2 OutOfMemoryError, cv2.pyd access violation, segfault) that previously
# # # # killed the ENTIRE worker — including RabbitMQ's connection thread and every
# # # # other in-flight job — because a native crash terminates the OS process and
# # # # no Python `except`, not even `except BaseException`, can run after that.

# # # # Why a subprocess fixes this
# # # # ────────────────────────────
# # # # A Python try/except can only catch things the interpreter is still alive
# # # # to raise. A genuine native allocation failure or access violation kills
# # # # the process before any Python exception object even exists. The only way
# # # # to "catch" that is from OUTSIDE the process: the parent (consumer.py)
# # # # spawns this file's `run_journey_in_subprocess()` target as a child, and
# # # # watches `process.exitcode` after `process.join()`. If the child died
# # # # abnormally, the parent treats it as a crash and is responsible for
# # # # deciding which videos were already completed vs. not.

# # # # How the parent finds out which videos already finished
# # # # ────────────────────────────────────────────────────────
# # # # `analyze_journey()` in analyzer.py already processes videos one at a time
# # # # in a for-loop. We pass it an optional `video_done_cb` (see the small patch
# # # # to analyzer.py) that fires immediately after each video — success OR
# # # # per-video-caught-failure — completes. Here, that callback pushes a small
# # # # picklable progress event onto `events_q` (a multiprocessing.Queue shared
# # # # with the parent). The parent drains this queue continuously, so even if
# # # # the child is killed by the OS one frame into video 3, the parent already
# # # # knows videos 1 and 2 finished (or failed) and only needs to mark video 3
# # # # (and anything after it) as failed — exactly matching "if a video can't be
# # # # processed due to OOM, treat the remaining videos as failed too."

# # # # The final result (video_results / wall_seconds / failed_videos) is sent
# # # # back over `result_q` only on a clean return. If the child dies, `result_q`
# # # # never receives anything — that absence IS the crash signal the parent
# # # # checks for.
# # # # """

# # # # from __future__ import annotations

# # # # import multiprocessing as mp
# # # # import os
# # # # import traceback
# # # # from typing import Dict, List

# # # # from analyzer import analyze_journey
# # # # from models import VideoJob


# # # # # Event dicts pushed onto events_q by video_done_cb, e.g.:
# # # # #   {"type": "video_done", "video_id": 12, "ok": True,  "error": None,
# # # # #    "error_type": None, "stack_trace": None, "reason": None}
# # # # #   {"type": "video_done", "video_id": 13, "ok": False, "error": "decode error",
# # # # #    "error_type": "DECODE_ERROR", "stack_trace": "...", "reason": None}
# # # # #   {"type": "video_done", "video_id": 14, "ok": False,
# # # # #    "error": "Not Processed - Worker Resource Exhaustion",
# # # # #    "error_type": "NOT_PROCESSED", "stack_trace": None,
# # # # #    "reason": "Not Processed - Worker Resource Exhaustion"}
# # # # #
# # # # # Final outcome pushed onto result_q (at most one item, only on clean exit):
# # # # #   {"type": "result", "video_results": [...], "wall_seconds": 87.3,
# # # # #    "failed_videos": {13: "decode error"}}
# # # # #   {"type": "error",  "traceback": "...", "message": "..."}


# # # # def run_journey_in_subprocess(
# # # #     job_id: str,
# # # #     journey_id: int,
# # # #     folder_name: str,
# # # #     video_jobs: List[VideoJob],
# # # #     tmp_paths: Dict[int, str],
# # # #     events_q: "mp.Queue",
# # # #     result_q: "mp.Queue",
# # # # ) -> None:
# # # #     """
# # # #     Target function for multiprocessing.Process. Runs entirely in the
# # # #     child. Must only communicate back to the parent via events_q/result_q
# # # #     (no shared memory, no return value — multiprocessing.Process ignores
# # # #     return values).
# # # #     """
# # # #     # Keep OpenCV/MKL/etc. from oversubscribing CPU when several of these
# # # #     # child processes run concurrently on the same host (multi-user load).
# # # #     os.environ.setdefault("OMP_NUM_THREADS", "1")
# # # #     os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

# # # #     # ── Resource lifecycle (Phase 3): clean slate before this journey ────────
# # # #     # This is a brand-new `spawn`'d interpreter, so there is no Python-level
# # # #     # model object here yet. But the GPU itself may still be holding VRAM
# # # #     # reserved by a PREVIOUS journey's process if the driver hasn't finished
# # # #     # tearing down that process's CUDA context yet (this can lag process
# # # #     # exit, especially under back-to-back journeys). cleanup_before_journey()
# # # #     # is a no-op if the device is already clean, and prevents this journey's
# # # #     # model load from fighting over memory that should have been freed.
# # # #     try:
# # # #         from resource_manager import resource_manager
# # # #         resource_manager.cleanup_before_journey(job_id=job_id)
# # # #     except Exception:
# # # #         pass  # resource_manager unavailable — not fatal, proceed anyway

# # # #     def _video_done_cb(video_id: int, ok: bool, error: str | None,
# # # #                         error_type: str | None = None,
# # # #                         stack_trace: str | None = None,
# # # #                         reason: str | None = None) -> None:
# # # #         try:
# # # #             events_q.put(
# # # #                 {
# # # #                     "type":        "video_done",
# # # #                     "video_id":    video_id,
# # # #                     "ok":          ok,
# # # #                     "error":       error,
# # # #                     "error_type":  error_type,
# # # #                     "stack_trace": stack_trace,
# # # #                     "reason":      reason,
# # # #                 },
# # # #                 block=False,
# # # #             )
# # # #         except Exception:
# # # #             # Never let a full/broken events queue take down the child —
# # # #             # the parent's primary crash signal is exitcode + result_q
# # # #             # absence, the events queue is best-effort progress detail.
# # # #             pass
# # # #         # ── Resource lifecycle (Phase 3): per-video GPU cleanup ──────────────
# # # #         # Fired after every video, success or failure, so VRAM fragmentation
# # # #         # doesn't accumulate across the videos within a single journey.
# # # #         try:
# # # #             from resource_manager import resource_manager
# # # #             resource_manager.cleanup_after_video(job_id=job_id, video_id=video_id)
# # # #         except Exception:
# # # #             pass

# # # #     try:
# # # #         video_results, wall_seconds, failed_videos = analyze_journey(
# # # #             job_id       = job_id,
# # # #             journey_id   = journey_id,
# # # #             folder_name  = folder_name,
# # # #             video_jobs   = video_jobs,
# # # #             tmp_paths    = tmp_paths,
# # # #             progress_cb  = None,   # progress callbacks stay in the parent
# # # #             video_done_cb = _video_done_cb,
# # # #         )
# # # #         result_q.put({
# # # #             "type":          "result",
# # # #             "video_results": [vr.to_dict() for vr in video_results],
# # # #             "wall_seconds":  wall_seconds,
# # # #             "failed_videos": failed_videos,
# # # #         })
# # # #     except BaseException as exc:  # noqa: BLE001 - last line of defense in the child
# # # #         # Covers ordinary Python exceptions raised above analyzer.py's own
# # # #         # per-video isolation (e.g. a bug in the dedup/frame-upload stage
# # # #         # that runs after the per-video loop). True native crashes (OOM,
# # # #         # access violation, segfault) will NOT reach this except — they
# # # #         # kill the process directly, which is exactly the case the parent
# # # #         # detects via process.exitcode / join() instead.
# # # #         try:
# # # #             from resource_manager import resource_manager
# # # #             resource_manager.cleanup_after_failure(
# # # #                 job_id=job_id, reason=f"{type(exc).__name__}: {exc}",
# # # #             )
# # # #         except Exception:
# # # #             pass
# # # #         try:
# # # #             result_q.put({
# # # #                 "type":      "error",
# # # #                 "message":   str(exc),
# # # #                 "traceback": traceback.format_exc(),
# # # #             })
# # # #         except Exception:
# # # #             pass
# # # #     finally:
# # # #         # ── Resource lifecycle (Phase 3/4): release model + VRAM before exit ─
# # # #         # This process is about to die anyway (each journey gets a fresh
# # # #         # `spawn`'d process), but doing this explicitly — rather than
# # # #         # relying solely on OS/driver cleanup timing after process exit —
# # # #         # means the NEXT journey's child process is far less likely to
# # # #         # start while this one's CUDA context is still being torn down.
# # # #         # Runs on every path: clean success, caught exception above, AND
# # # #         # — to the extent Python is still alive to run a finally at all —
# # # #         # anything else. A genuine native crash (OOM/access violation)
# # # #         # skips this entirely; that case is handled by the parent's
# # # #         # exitcode-based emergency_cleanup() in consumer.py instead.
# # # #         try:
# # # #             from resource_manager import resource_manager
# # # #             resource_manager.cleanup_after_journey(job_id=job_id)
# # # #         except Exception:
# # # #             pass


# # # # """
# # # # journey_runner.py
# # # # ──────────────────
# # # # Runs inside an isolated CHILD PROCESS (multiprocessing.Process), spawned by
# # # # consumer.py for every journey. This is the fix for native OpenCV crashes
# # # # (cv2 OutOfMemoryError, cv2.pyd access violation, segfault) that previously
# # # # killed the ENTIRE worker — including RabbitMQ's connection thread and every
# # # # other in-flight job — because a native crash terminates the OS process and
# # # # no Python `except`, not even `except BaseException`, can run after that.

# # # # Why a subprocess fixes this
# # # # ────────────────────────────
# # # # A Python try/except can only catch things the interpreter is still alive
# # # # to raise. A genuine native allocation failure or access violation kills
# # # # the process before any Python exception object even exists. The only way
# # # # to "catch" that is from OUTSIDE the process: the parent (consumer.py)
# # # # spawns this file's `run_journey_in_subprocess()` target as a child, and
# # # # watches `process.exitcode` after `process.join()`. If the child died
# # # # abnormally, the parent treats it as a crash and is responsible for
# # # # deciding which videos were already completed vs. not.

# # # # How the parent finds out which videos already finished
# # # # ────────────────────────────────────────────────────────
# # # # `analyze_journey()` in analyzer.py already processes videos one at a time
# # # # in a for-loop. We pass it an optional `video_done_cb` (see the small patch
# # # # to analyzer.py) that fires immediately after each video — success OR
# # # # per-video-caught-failure — completes. Here, that callback pushes a small
# # # # picklable progress event onto `events_q` (a multiprocessing.Queue shared
# # # # with the parent). The parent drains this queue continuously, so even if
# # # # the child is killed by the OS one frame into video 3, the parent already
# # # # knows videos 1 and 2 finished (or failed) and only needs to mark video 3
# # # # (and anything after it) as failed — exactly matching "if a video can't be
# # # # processed due to OOM, treat the remaining videos as failed too."

# # # # The final result (video_results / wall_seconds / failed_videos) is sent
# # # # back over `result_q` only on a clean return. If the child dies, `result_q`
# # # # never receives anything — that absence IS the crash signal the parent
# # # # checks for.
# # # # """

# # # # from __future__ import annotations

# # # # import multiprocessing as mp
# # # # import os
# # # # import traceback
# # # # from typing import Dict, List

# # # # from analyzer import analyze_journey
# # # # from models import VideoJob


# # # # # Event dicts pushed onto events_q by video_done_cb, e.g.:
# # # # #   {"type": "video_done", "video_id": 12, "ok": True,  "error": None,
# # # # #    "error_type": None, "stack_trace": None, "reason": None}
# # # # #   {"type": "video_done", "video_id": 13, "ok": False, "error": "decode error",
# # # # #    "error_type": "DECODE_ERROR", "stack_trace": "...", "reason": None}
# # # # #   {"type": "video_done", "video_id": 14, "ok": False,
# # # # #    "error": "Not Processed - Worker Resource Exhaustion",
# # # # #    "error_type": "NOT_PROCESSED", "stack_trace": None,
# # # # #    "reason": "Not Processed - Worker Resource Exhaustion"}
# # # # #
# # # # # Final outcome pushed onto result_q (at most one item, only on clean exit):
# # # # #   {"type": "result", "video_results": [...], "wall_seconds": 87.3,
# # # # #    "failed_videos": {13: "decode error"}}
# # # # #   {"type": "error",  "traceback": "...", "message": "..."}


# # # # def run_journey_in_subprocess(
# # # #     job_id: str,
# # # #     journey_id: int,
# # # #     folder_name: str,
# # # #     video_jobs: List[VideoJob],
# # # #     tmp_paths: Dict[int, str],
# # # #     events_q: "mp.Queue",
# # # #     result_q: "mp.Queue",
# # # # ) -> None:
# # # #     """
# # # #     Target function for multiprocessing.Process. Runs entirely in the
# # # #     child. Must only communicate back to the parent via events_q/result_q
# # # #     (no shared memory, no return value — multiprocessing.Process ignores
# # # #     return values).
# # # #     """
# # # #     # Keep OpenCV/MKL/etc. from oversubscribing CPU when several of these
# # # #     # child processes run concurrently on the same host (multi-user load).
# # # #     os.environ.setdefault("OMP_NUM_THREADS", "1")
# # # #     os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

# # # #     def _video_done_cb(video_id: int, ok: bool, error: str | None,
# # # #                         error_type: str | None = None,
# # # #                         stack_trace: str | None = None,
# # # #                         reason: str | None = None) -> None:
# # # #         try:
# # # #             events_q.put(
# # # #                 {
# # # #                     "type":        "video_done",
# # # #                     "video_id":    video_id,
# # # #                     "ok":          ok,
# # # #                     "error":       error,
# # # #                     "error_type":  error_type,
# # # #                     "stack_trace": stack_trace,
# # # #                     "reason":      reason,
# # # #                 },
# # # #                 block=False,
# # # #             )
# # # #         except Exception:
# # # #             # Never let a full/broken events queue take down the child —
# # # #             # the parent's primary crash signal is exitcode + result_q
# # # #             # absence, the events queue is best-effort progress detail.
# # # #             pass

# # # #     try:
# # # #         video_results, wall_seconds, failed_videos = analyze_journey(
# # # #             job_id       = job_id,
# # # #             journey_id   = journey_id,
# # # #             folder_name  = folder_name,
# # # #             video_jobs   = video_jobs,
# # # #             tmp_paths    = tmp_paths,
# # # #             progress_cb  = None,   # progress callbacks stay in the parent
# # # #             video_done_cb = _video_done_cb,
# # # #         )
# # # #         result_q.put({
# # # #             "type":          "result",
# # # #             "video_results": [vr.to_dict() for vr in video_results],
# # # #             "wall_seconds":  wall_seconds,
# # # #             "failed_videos": failed_videos,
# # # #         })
# # # #     except BaseException as exc:  # noqa: BLE001 - last line of defense in the child
# # # #         # Covers ordinary Python exceptions raised above analyzer.py's own
# # # #         # per-video isolation (e.g. a bug in the dedup/frame-upload stage
# # # #         # that runs after the per-video loop). True native crashes (OOM,
# # # #         # access violation, segfault) will NOT reach this except — they
# # # #         # kill the process directly, which is exactly the case the parent
# # # #         # detects via process.exitcode / join() instead.
# # # #         try:
# # # #             result_q.put({
# # # #                 "type":      "error",
# # # #                 "message":   str(exc),
# # # #                 "traceback": traceback.format_exc(),
# # # #             })
# # # #         except Exception:
# # # #             pass


# # # """
# # # journey_runner.py
# # # ──────────────────
# # # Runs inside an isolated CHILD PROCESS (multiprocessing.Process), spawned by
# # # consumer.py for every journey. This is the fix for native OpenCV crashes
# # # (cv2 OutOfMemoryError, cv2.pyd access violation, segfault) that previously
# # # killed the ENTIRE worker — including RabbitMQ's connection thread and every
# # # other in-flight job — because a native crash terminates the OS process and
# # # no Python `except`, not even `except BaseException`, can run after that.

# # # Why a subprocess fixes this
# # # ────────────────────────────
# # # A Python try/except can only catch things the interpreter is still alive
# # # to raise. A genuine native allocation failure or access violation kills
# # # the process before any Python exception object even exists. The only way
# # # to "catch" that is from OUTSIDE the process: the parent (consumer.py)
# # # spawns this file's `run_journey_in_subprocess()` target as a child, and
# # # watches `process.exitcode` after `process.join()`. If the child died
# # # abnormally, the parent treats it as a crash and is responsible for
# # # deciding which videos were already completed vs. not.

# # # How the parent finds out which videos already finished
# # # ────────────────────────────────────────────────────────
# # # `analyze_journey()` in analyzer.py already processes videos one at a time
# # # in a for-loop. We pass it an optional `video_done_cb` (see the small patch
# # # to analyzer.py) that fires immediately after each video — success OR
# # # per-video-caught-failure — completes. Here, that callback pushes a small
# # # picklable progress event onto `events_q` (a multiprocessing.Queue shared
# # # with the parent). The parent drains this queue continuously, so even if
# # # the child is killed by the OS one frame into video 3, the parent already
# # # knows videos 1 and 2 finished (or failed) and only needs to mark video 3
# # # (and anything after it) as failed — exactly matching "if a video can't be
# # # processed due to OOM, treat the remaining videos as failed too."

# # # The final result (video_results / wall_seconds / failed_videos) is sent
# # # back over `result_q` only on a clean return. If the child dies, `result_q`
# # # never receives anything — that absence IS the crash signal the parent
# # # checks for.
# # # """

# # # from __future__ import annotations

# # # import multiprocessing as mp
# # # import os
# # # import traceback
# # # from typing import Dict, List

# # # from analyzer import analyze_journey
# # # from models import VideoJob


# # # # Event dicts pushed onto events_q by video_done_cb, e.g.:
# # # #   {"type": "video_done", "video_id": 12, "ok": True,  "error": None,
# # # #    "error_type": None, "stack_trace": None, "reason": None}
# # # #   {"type": "video_done", "video_id": 13, "ok": False, "error": "decode error",
# # # #    "error_type": "DECODE_ERROR", "stack_trace": "...", "reason": None}
# # # #   {"type": "video_done", "video_id": 14, "ok": False,
# # # #    "error": "Not Processed - Worker Resource Exhaustion",
# # # #    "error_type": "NOT_PROCESSED", "stack_trace": None,
# # # #    "reason": "Not Processed - Worker Resource Exhaustion"}
# # # #
# # # # Final outcome pushed onto result_q (at most one item, only on clean exit):
# # # #   {"type": "result", "video_results": [...], "wall_seconds": 87.3,
# # # #    "failed_videos": {13: "decode error"}}
# # # #   {"type": "error",  "traceback": "...", "message": "..."}


# # # def run_journey_in_subprocess(
# # #     job_id: str,
# # #     journey_id: int,
# # #     folder_name: str,
# # #     video_jobs: List[VideoJob],
# # #     tmp_paths: Dict[int, str],
# # #     events_q: "mp.Queue",
# # #     result_q: "mp.Queue",
# # # ) -> None:
# # #     """
# # #     Target function for multiprocessing.Process. Runs entirely in the
# # #     child. Must only communicate back to the parent via events_q/result_q
# # #     (no shared memory, no return value — multiprocessing.Process ignores
# # #     return values).
# # #     """
# # #     # Keep OpenCV/MKL/etc. from oversubscribing CPU when several of these
# # #     # child processes run concurrently on the same host (multi-user load).
# # #     os.environ.setdefault("OMP_NUM_THREADS", "1")
# # #     os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

# # #     # ── Resource lifecycle (Phase 3): clean slate before this journey ────────
# # #     # This is a brand-new `spawn`'d interpreter, so there is no Python-level
# # #     # model object here yet. But the GPU itself may still be holding VRAM
# # #     # reserved by a PREVIOUS journey's process if the driver hasn't finished
# # #     # tearing down that process's CUDA context yet (this can lag process
# # #     # exit, especially under back-to-back journeys). cleanup_before_journey()
# # #     # is a no-op if the device is already clean, and prevents this journey's
# # #     # model load from fighting over memory that should have been freed.
# # #     try:
# # #         from resource_manager import resource_manager
# # #         resource_manager.cleanup_before_journey(job_id=job_id)
# # #     except Exception:
# # #         pass  # resource_manager unavailable — not fatal, proceed anyway

# # #     def _video_done_cb(video_id: int, ok: bool, error: str | None,
# # #                         error_type: str | None = None,
# # #                         stack_trace: str | None = None,
# # #                         reason: str | None = None,
# # #                         video_result: dict | None = None) -> None:
# # #         try:
# # #             events_q.put(
# # #                 {
# # #                     "type":         "video_done",
# # #                     "video_id":     video_id,
# # #                     "ok":           ok,
# # #                     "error":        error,
# # #                     "error_type":   error_type,
# # #                     "stack_trace":  stack_trace,
# # #                     "reason":       reason,
# # #                     # Best-effort per-video result snapshot (see analyzer.py's
# # #                     # _build_partial_video_result) — only present when ok=True.
# # #                     # Lets the parent preserve real violation data for videos
# # #                     # that finished before a native crash kills this child,
# # #                     # instead of having to mark them failed too.
# # #                     "video_result": video_result,
# # #                 },
# # #                 block=False,
# # #             )
# # #         except Exception:
# # #             # Never let a full/broken events queue take down the child —
# # #             # the parent's primary crash signal is exitcode + result_q
# # #             # absence, the events queue is best-effort progress detail.
# # #             pass
# # #         # ── Resource lifecycle (Phase 3): per-video GPU cleanup ──────────────
# # #         # Fired after every video, success or failure, so VRAM fragmentation
# # #         # doesn't accumulate across the videos within a single journey.
# # #         try:
# # #             from resource_manager import resource_manager
# # #             resource_manager.cleanup_after_video(job_id=job_id, video_id=video_id)
# # #         except Exception:
# # #             pass

# # #     try:
# # #         video_results, wall_seconds, failed_videos = analyze_journey(
# # #             job_id       = job_id,
# # #             journey_id   = journey_id,
# # #             folder_name  = folder_name,
# # #             video_jobs   = video_jobs,
# # #             tmp_paths    = tmp_paths,
# # #             progress_cb  = None,   # progress callbacks stay in the parent
# # #             video_done_cb = _video_done_cb,
# # #         )
# # #         result_q.put({
# # #             "type":          "result",
# # #             "video_results": [vr.to_dict() for vr in video_results],
# # #             "wall_seconds":  wall_seconds,
# # #             "failed_videos": failed_videos,
# # #         })
# # #     except BaseException as exc:  # noqa: BLE001 - last line of defense in the child
# # #         # Covers ordinary Python exceptions raised above analyzer.py's own
# # #         # per-video isolation (e.g. a bug in the dedup/frame-upload stage
# # #         # that runs after the per-video loop). True native crashes (OOM,
# # #         # access violation, segfault) will NOT reach this except — they
# # #         # kill the process directly, which is exactly the case the parent
# # #         # detects via process.exitcode / join() instead.
# # #         try:
# # #             from resource_manager import resource_manager
# # #             resource_manager.cleanup_after_failure(
# # #                 job_id=job_id, reason=f"{type(exc).__name__}: {exc}",
# # #             )
# # #         except Exception:
# # #             pass
# # #         try:
# # #             result_q.put({
# # #                 "type":      "error",
# # #                 "message":   str(exc),
# # #                 "traceback": traceback.format_exc(),
# # #             })
# # #         except Exception:
# # #             pass
# # #     finally:
# # #         # ── Resource lifecycle (Phase 3/4): release model + VRAM before exit ─
# # #         # This process is about to die anyway (each journey gets a fresh
# # #         # `spawn`'d process), but doing this explicitly — rather than
# # #         # relying solely on OS/driver cleanup timing after process exit —
# # #         # means the NEXT journey's child process is far less likely to
# # #         # start while this one's CUDA context is still being torn down.
# # #         # Runs on every path: clean success, caught exception above, AND
# # #         # — to the extent Python is still alive to run a finally at all —
# # #         # anything else. A genuine native crash (OOM/access violation)
# # #         # skips this entirely; that case is handled by the parent's
# # #         # exitcode-based emergency_cleanup() in consumer.py instead.
# # #         try:
# # #             from resource_manager import resource_manager
# # #             resource_manager.cleanup_after_journey(job_id=job_id)
# # #         except Exception:
# # #             pass


# # # # """
# # # # journey_runner.py
# # # # ──────────────────
# # # # Runs inside an isolated CHILD PROCESS (multiprocessing.Process), spawned by
# # # # consumer.py for every journey. This is the fix for native OpenCV crashes
# # # # (cv2 OutOfMemoryError, cv2.pyd access violation, segfault) that previously
# # # # killed the ENTIRE worker — including RabbitMQ's connection thread and every
# # # # other in-flight job — because a native crash terminates the OS process and
# # # # no Python `except`, not even `except BaseException`, can run after that.

# # # # Why a subprocess fixes this
# # # # ────────────────────────────
# # # # A Python try/except can only catch things the interpreter is still alive
# # # # to raise. A genuine native allocation failure or access violation kills
# # # # the process before any Python exception object even exists. The only way
# # # # to "catch" that is from OUTSIDE the process: the parent (consumer.py)
# # # # spawns this file's `run_journey_in_subprocess()` target as a child, and
# # # # watches `process.exitcode` after `process.join()`. If the child died
# # # # abnormally, the parent treats it as a crash and is responsible for
# # # # deciding which videos were already completed vs. not.

# # # # How the parent finds out which videos already finished
# # # # ────────────────────────────────────────────────────────
# # # # `analyze_journey()` in analyzer.py already processes videos one at a time
# # # # in a for-loop. We pass it an optional `video_done_cb` (see the small patch
# # # # to analyzer.py) that fires immediately after each video — success OR
# # # # per-video-caught-failure — completes. Here, that callback pushes a small
# # # # picklable progress event onto `events_q` (a multiprocessing.Queue shared
# # # # with the parent). The parent drains this queue continuously, so even if
# # # # the child is killed by the OS one frame into video 3, the parent already
# # # # knows videos 1 and 2 finished (or failed) and only needs to mark video 3
# # # # (and anything after it) as failed — exactly matching "if a video can't be
# # # # processed due to OOM, treat the remaining videos as failed too."

# # # # The final result (video_results / wall_seconds / failed_videos) is sent
# # # # back over `result_q` only on a clean return. If the child dies, `result_q`
# # # # never receives anything — that absence IS the crash signal the parent
# # # # checks for.
# # # # """

# # # # from __future__ import annotations

# # # # import multiprocessing as mp
# # # # import os
# # # # import traceback
# # # # from typing import Dict, List

# # # # from analyzer import analyze_journey
# # # # from models import VideoJob


# # # # # Event dicts pushed onto events_q by video_done_cb, e.g.:
# # # # #   {"type": "video_done", "video_id": 12, "ok": True,  "error": None,
# # # # #    "error_type": None, "stack_trace": None, "reason": None}
# # # # #   {"type": "video_done", "video_id": 13, "ok": False, "error": "decode error",
# # # # #    "error_type": "DECODE_ERROR", "stack_trace": "...", "reason": None}
# # # # #   {"type": "video_done", "video_id": 14, "ok": False,
# # # # #    "error": "Not Processed - Worker Resource Exhaustion",
# # # # #    "error_type": "NOT_PROCESSED", "stack_trace": None,
# # # # #    "reason": "Not Processed - Worker Resource Exhaustion"}
# # # # #
# # # # # Final outcome pushed onto result_q (at most one item, only on clean exit):
# # # # #   {"type": "result", "video_results": [...], "wall_seconds": 87.3,
# # # # #    "failed_videos": {13: "decode error"}}
# # # # #   {"type": "error",  "traceback": "...", "message": "..."}


# # # # def run_journey_in_subprocess(
# # # #     job_id: str,
# # # #     journey_id: int,
# # # #     folder_name: str,
# # # #     video_jobs: List[VideoJob],
# # # #     tmp_paths: Dict[int, str],
# # # #     events_q: "mp.Queue",
# # # #     result_q: "mp.Queue",
# # # # ) -> None:
# # # #     """
# # # #     Target function for multiprocessing.Process. Runs entirely in the
# # # #     child. Must only communicate back to the parent via events_q/result_q
# # # #     (no shared memory, no return value — multiprocessing.Process ignores
# # # #     return values).
# # # #     """
# # # #     # Keep OpenCV/MKL/etc. from oversubscribing CPU when several of these
# # # #     # child processes run concurrently on the same host (multi-user load).
# # # #     os.environ.setdefault("OMP_NUM_THREADS", "1")
# # # #     os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

# # # #     def _video_done_cb(video_id: int, ok: bool, error: str | None,
# # # #                         error_type: str | None = None,
# # # #                         stack_trace: str | None = None,
# # # #                         reason: str | None = None) -> None:
# # # #         try:
# # # #             events_q.put(
# # # #                 {
# # # #                     "type":        "video_done",
# # # #                     "video_id":    video_id,
# # # #                     "ok":          ok,
# # # #                     "error":       error,
# # # #                     "error_type":  error_type,
# # # #                     "stack_trace": stack_trace,
# # # #                     "reason":      reason,
# # # #                 },
# # # #                 block=False,
# # # #             )
# # # #         except Exception:
# # # #             # Never let a full/broken events queue take down the child —
# # # #             # the parent's primary crash signal is exitcode + result_q
# # # #             # absence, the events queue is best-effort progress detail.
# # # #             pass

# # # #     try:
# # # #         video_results, wall_seconds, failed_videos = analyze_journey(
# # # #             job_id       = job_id,
# # # #             journey_id   = journey_id,
# # # #             folder_name  = folder_name,
# # # #             video_jobs   = video_jobs,
# # # #             tmp_paths    = tmp_paths,
# # # #             progress_cb  = None,   # progress callbacks stay in the parent
# # # #             video_done_cb = _video_done_cb,
# # # #         )
# # # #         result_q.put({
# # # #             "type":          "result",
# # # #             "video_results": [vr.to_dict() for vr in video_results],
# # # #             "wall_seconds":  wall_seconds,
# # # #             "failed_videos": failed_videos,
# # # #         })
# # # #     except BaseException as exc:  # noqa: BLE001 - last line of defense in the child
# # # #         # Covers ordinary Python exceptions raised above analyzer.py's own
# # # #         # per-video isolation (e.g. a bug in the dedup/frame-upload stage
# # # #         # that runs after the per-video loop). True native crashes (OOM,
# # # #         # access violation, segfault) will NOT reach this except — they
# # # #         # kill the process directly, which is exactly the case the parent
# # # #         # detects via process.exitcode / join() instead.
# # # #         try:
# # # #             result_q.put({
# # # #                 "type":      "error",
# # # #                 "message":   str(exc),
# # # #                 "traceback": traceback.format_exc(),
# # # #             })
# # # #         except Exception:
# # # #             pass


# # # """
# # # journey_runner.py
# # # ──────────────────
# # # Runs inside an isolated CHILD PROCESS (multiprocessing.Process), spawned by
# # # consumer.py for every journey. This is the fix for native OpenCV crashes
# # # (cv2 OutOfMemoryError, cv2.pyd access violation, segfault) that previously
# # # killed the ENTIRE worker — including RabbitMQ's connection thread and every
# # # other in-flight job — because a native crash terminates the OS process and
# # # no Python `except`, not even `except BaseException`, can run after that.

# # # Why a subprocess fixes this
# # # ────────────────────────────
# # # A Python try/except can only catch things the interpreter is still alive
# # # to raise. A genuine native allocation failure or access violation kills
# # # the process before any Python exception object even exists. The only way
# # # to "catch" that is from OUTSIDE the process: the parent (consumer.py)
# # # spawns this file's `run_journey_in_subprocess()` target as a child, and
# # # watches `process.exitcode` after `process.join()`. If the child died
# # # abnormally, the parent treats it as a crash and is responsible for
# # # deciding which videos were already completed vs. not.

# # # How the parent finds out which videos already finished
# # # ────────────────────────────────────────────────────────
# # # `analyze_journey()` in analyzer.py already processes videos one at a time
# # # in a for-loop. We pass it an optional `video_done_cb` (see the small patch
# # # to analyzer.py) that fires immediately after each video — success OR
# # # per-video-caught-failure — completes. Here, that callback pushes a small
# # # picklable progress event onto `events_q` (a multiprocessing.Queue shared
# # # with the parent). The parent drains this queue continuously, so even if
# # # the child is killed by the OS one frame into video 3, the parent already
# # # knows videos 1 and 2 finished (or failed) and only needs to mark video 3
# # # (and anything after it) as failed — exactly matching "if a video can't be
# # # processed due to OOM, treat the remaining videos as failed too."

# # # The final result (video_results / wall_seconds / failed_videos) is sent
# # # back over `result_q` only on a clean return. If the child dies, `result_q`
# # # never receives anything — that absence IS the crash signal the parent
# # # checks for.
# # # """

# # # from __future__ import annotations

# # # import multiprocessing as mp
# # # import os
# # # import traceback
# # # from typing import Dict, List

# # # from analyzer import analyze_journey
# # # from models import VideoJob


# # # # Event dicts pushed onto events_q by video_done_cb, e.g.:
# # # #   {"type": "video_done", "video_id": 12, "ok": True,  "error": None,
# # # #    "error_type": None, "stack_trace": None, "reason": None}
# # # #   {"type": "video_done", "video_id": 13, "ok": False, "error": "decode error",
# # # #    "error_type": "DECODE_ERROR", "stack_trace": "...", "reason": None}
# # # #   {"type": "video_done", "video_id": 14, "ok": False,
# # # #    "error": "Not Processed - Worker Resource Exhaustion",
# # # #    "error_type": "NOT_PROCESSED", "stack_trace": None,
# # # #    "reason": "Not Processed - Worker Resource Exhaustion"}
# # # #
# # # # Final outcome pushed onto result_q (at most one item, only on clean exit):
# # # #   {"type": "result", "video_results": [...], "wall_seconds": 87.3,
# # # #    "failed_videos": {13: "decode error"}}
# # # #   {"type": "error",  "traceback": "...", "message": "..."}


# # # def run_journey_in_subprocess(
# # #     job_id: str,
# # #     journey_id: int,
# # #     folder_name: str,
# # #     video_jobs: List[VideoJob],
# # #     tmp_paths: Dict[int, str],
# # #     events_q: "mp.Queue",
# # #     result_q: "mp.Queue",
# # # ) -> None:
# # #     """
# # #     Target function for multiprocessing.Process. Runs entirely in the
# # #     child. Must only communicate back to the parent via events_q/result_q
# # #     (no shared memory, no return value — multiprocessing.Process ignores
# # #     return values).
# # #     """
# # #     # Keep OpenCV/MKL/etc. from oversubscribing CPU when several of these
# # #     # child processes run concurrently on the same host (multi-user load).
# # #     os.environ.setdefault("OMP_NUM_THREADS", "1")
# # #     os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

# # #     # ── Resource lifecycle (Phase 3): clean slate before this journey ────────
# # #     # This is a brand-new `spawn`'d interpreter, so there is no Python-level
# # #     # model object here yet. But the GPU itself may still be holding VRAM
# # #     # reserved by a PREVIOUS journey's process if the driver hasn't finished
# # #     # tearing down that process's CUDA context yet (this can lag process
# # #     # exit, especially under back-to-back journeys). cleanup_before_journey()
# # #     # is a no-op if the device is already clean, and prevents this journey's
# # #     # model load from fighting over memory that should have been freed.
# # #     try:
# # #         from resource_manager import resource_manager
# # #         resource_manager.cleanup_before_journey(job_id=job_id)
# # #     except Exception:
# # #         pass  # resource_manager unavailable — not fatal, proceed anyway

# # #     def _video_done_cb(video_id: int, ok: bool, error: str | None,
# # #                         error_type: str | None = None,
# # #                         stack_trace: str | None = None,
# # #                         reason: str | None = None) -> None:
# # #         try:
# # #             events_q.put(
# # #                 {
# # #                     "type":        "video_done",
# # #                     "video_id":    video_id,
# # #                     "ok":          ok,
# # #                     "error":       error,
# # #                     "error_type":  error_type,
# # #                     "stack_trace": stack_trace,
# # #                     "reason":      reason,
# # #                 },
# # #                 block=False,
# # #             )
# # #         except Exception:
# # #             # Never let a full/broken events queue take down the child —
# # #             # the parent's primary crash signal is exitcode + result_q
# # #             # absence, the events queue is best-effort progress detail.
# # #             pass
# # #         # ── Resource lifecycle (Phase 3): per-video GPU cleanup ──────────────
# # #         # Fired after every video, success or failure, so VRAM fragmentation
# # #         # doesn't accumulate across the videos within a single journey.
# # #         try:
# # #             from resource_manager import resource_manager
# # #             resource_manager.cleanup_after_video(job_id=job_id, video_id=video_id)
# # #         except Exception:
# # #             pass

# # #     try:
# # #         video_results, wall_seconds, failed_videos = analyze_journey(
# # #             job_id       = job_id,
# # #             journey_id   = journey_id,
# # #             folder_name  = folder_name,
# # #             video_jobs   = video_jobs,
# # #             tmp_paths    = tmp_paths,
# # #             progress_cb  = None,   # progress callbacks stay in the parent
# # #             video_done_cb = _video_done_cb,
# # #         )
# # #         result_q.put({
# # #             "type":          "result",
# # #             "video_results": [vr.to_dict() for vr in video_results],
# # #             "wall_seconds":  wall_seconds,
# # #             "failed_videos": failed_videos,
# # #         })
# # #     except BaseException as exc:  # noqa: BLE001 - last line of defense in the child
# # #         # Covers ordinary Python exceptions raised above analyzer.py's own
# # #         # per-video isolation (e.g. a bug in the dedup/frame-upload stage
# # #         # that runs after the per-video loop). True native crashes (OOM,
# # #         # access violation, segfault) will NOT reach this except — they
# # #         # kill the process directly, which is exactly the case the parent
# # #         # detects via process.exitcode / join() instead.
# # #         try:
# # #             from resource_manager import resource_manager
# # #             resource_manager.cleanup_after_failure(
# # #                 job_id=job_id, reason=f"{type(exc).__name__}: {exc}",
# # #             )
# # #         except Exception:
# # #             pass
# # #         try:
# # #             result_q.put({
# # #                 "type":      "error",
# # #                 "message":   str(exc),
# # #                 "traceback": traceback.format_exc(),
# # #             })
# # #         except Exception:
# # #             pass
# # #     finally:
# # #         # ── Resource lifecycle (Phase 3/4): release model + VRAM before exit ─
# # #         # This process is about to die anyway (each journey gets a fresh
# # #         # `spawn`'d process), but doing this explicitly — rather than
# # #         # relying solely on OS/driver cleanup timing after process exit —
# # #         # means the NEXT journey's child process is far less likely to
# # #         # start while this one's CUDA context is still being torn down.
# # #         # Runs on every path: clean success, caught exception above, AND
# # #         # — to the extent Python is still alive to run a finally at all —
# # #         # anything else. A genuine native crash (OOM/access violation)
# # #         # skips this entirely; that case is handled by the parent's
# # #         # exitcode-based emergency_cleanup() in consumer.py instead.
# # #         try:
# # #             from resource_manager import resource_manager
# # #             resource_manager.cleanup_after_journey(job_id=job_id)
# # #         except Exception:
# # #             pass


# # # """
# # # journey_runner.py
# # # ──────────────────
# # # Runs inside an isolated CHILD PROCESS (multiprocessing.Process), spawned by
# # # consumer.py for every journey. This is the fix for native OpenCV crashes
# # # (cv2 OutOfMemoryError, cv2.pyd access violation, segfault) that previously
# # # killed the ENTIRE worker — including RabbitMQ's connection thread and every
# # # other in-flight job — because a native crash terminates the OS process and
# # # no Python `except`, not even `except BaseException`, can run after that.

# # # Why a subprocess fixes this
# # # ────────────────────────────
# # # A Python try/except can only catch things the interpreter is still alive
# # # to raise. A genuine native allocation failure or access violation kills
# # # the process before any Python exception object even exists. The only way
# # # to "catch" that is from OUTSIDE the process: the parent (consumer.py)
# # # spawns this file's `run_journey_in_subprocess()` target as a child, and
# # # watches `process.exitcode` after `process.join()`. If the child died
# # # abnormally, the parent treats it as a crash and is responsible for
# # # deciding which videos were already completed vs. not.

# # # How the parent finds out which videos already finished
# # # ────────────────────────────────────────────────────────
# # # `analyze_journey()` in analyzer.py already processes videos one at a time
# # # in a for-loop. We pass it an optional `video_done_cb` (see the small patch
# # # to analyzer.py) that fires immediately after each video — success OR
# # # per-video-caught-failure — completes. Here, that callback pushes a small
# # # picklable progress event onto `events_q` (a multiprocessing.Queue shared
# # # with the parent). The parent drains this queue continuously, so even if
# # # the child is killed by the OS one frame into video 3, the parent already
# # # knows videos 1 and 2 finished (or failed) and only needs to mark video 3
# # # (and anything after it) as failed — exactly matching "if a video can't be
# # # processed due to OOM, treat the remaining videos as failed too."

# # # The final result (video_results / wall_seconds / failed_videos) is sent
# # # back over `result_q` only on a clean return. If the child dies, `result_q`
# # # never receives anything — that absence IS the crash signal the parent
# # # checks for.
# # # """

# # # from __future__ import annotations

# # # import multiprocessing as mp
# # # import os
# # # import traceback
# # # from typing import Dict, List

# # # from analyzer import analyze_journey
# # # from models import VideoJob


# # # # Event dicts pushed onto events_q by video_done_cb, e.g.:
# # # #   {"type": "video_done", "video_id": 12, "ok": True,  "error": None,
# # # #    "error_type": None, "stack_trace": None, "reason": None}
# # # #   {"type": "video_done", "video_id": 13, "ok": False, "error": "decode error",
# # # #    "error_type": "DECODE_ERROR", "stack_trace": "...", "reason": None}
# # # #   {"type": "video_done", "video_id": 14, "ok": False,
# # # #    "error": "Not Processed - Worker Resource Exhaustion",
# # # #    "error_type": "NOT_PROCESSED", "stack_trace": None,
# # # #    "reason": "Not Processed - Worker Resource Exhaustion"}
# # # #
# # # # Final outcome pushed onto result_q (at most one item, only on clean exit):
# # # #   {"type": "result", "video_results": [...], "wall_seconds": 87.3,
# # # #    "failed_videos": {13: "decode error"}}
# # # #   {"type": "error",  "traceback": "...", "message": "..."}


# # # def run_journey_in_subprocess(
# # #     job_id: str,
# # #     journey_id: int,
# # #     folder_name: str,
# # #     video_jobs: List[VideoJob],
# # #     tmp_paths: Dict[int, str],
# # #     events_q: "mp.Queue",
# # #     result_q: "mp.Queue",
# # # ) -> None:
# # #     """
# # #     Target function for multiprocessing.Process. Runs entirely in the
# # #     child. Must only communicate back to the parent via events_q/result_q
# # #     (no shared memory, no return value — multiprocessing.Process ignores
# # #     return values).
# # #     """
# # #     # Keep OpenCV/MKL/etc. from oversubscribing CPU when several of these
# # #     # child processes run concurrently on the same host (multi-user load).
# # #     os.environ.setdefault("OMP_NUM_THREADS", "1")
# # #     os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

# # #     def _video_done_cb(video_id: int, ok: bool, error: str | None,
# # #                         error_type: str | None = None,
# # #                         stack_trace: str | None = None,
# # #                         reason: str | None = None) -> None:
# # #         try:
# # #             events_q.put(
# # #                 {
# # #                     "type":        "video_done",
# # #                     "video_id":    video_id,
# # #                     "ok":          ok,
# # #                     "error":       error,
# # #                     "error_type":  error_type,
# # #                     "stack_trace": stack_trace,
# # #                     "reason":      reason,
# # #                 },
# # #                 block=False,
# # #             )
# # #         except Exception:
# # #             # Never let a full/broken events queue take down the child —
# # #             # the parent's primary crash signal is exitcode + result_q
# # #             # absence, the events queue is best-effort progress detail.
# # #             pass

# # #     try:
# # #         video_results, wall_seconds, failed_videos = analyze_journey(
# # #             job_id       = job_id,
# # #             journey_id   = journey_id,
# # #             folder_name  = folder_name,
# # #             video_jobs   = video_jobs,
# # #             tmp_paths    = tmp_paths,
# # #             progress_cb  = None,   # progress callbacks stay in the parent
# # #             video_done_cb = _video_done_cb,
# # #         )
# # #         result_q.put({
# # #             "type":          "result",
# # #             "video_results": [vr.to_dict() for vr in video_results],
# # #             "wall_seconds":  wall_seconds,
# # #             "failed_videos": failed_videos,
# # #         })
# # #     except BaseException as exc:  # noqa: BLE001 - last line of defense in the child
# # #         # Covers ordinary Python exceptions raised above analyzer.py's own
# # #         # per-video isolation (e.g. a bug in the dedup/frame-upload stage
# # #         # that runs after the per-video loop). True native crashes (OOM,
# # #         # access violation, segfault) will NOT reach this except — they
# # #         # kill the process directly, which is exactly the case the parent
# # #         # detects via process.exitcode / join() instead.
# # #         try:
# # #             result_q.put({
# # #                 "type":      "error",
# # #                 "message":   str(exc),
# # #                 "traceback": traceback.format_exc(),
# # #             })
# # #         except Exception:
# # #             pass


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


# # def _process_one_journey(
# #     job_id: str,
# #     journey_id: int,
# #     folder_name: str,
# #     video_jobs: List[VideoJob],
# #     tmp_paths: Dict[int, str],
# #     events_q: "mp.Queue",
# #     result_q: "mp.Queue",
# # ) -> None:
# #     """
# #     Runs exactly ONE journey's analysis and reports back over
# #     events_q/result_q — the same wire format regardless of whether this is
# #     called from a one-shot subprocess (run_journey_in_subprocess, kept for
# #     backward compatibility) or from inside a long-lived GPU worker process's
# #     main loop (run_worker_loop, used by the persistent worker-pool
# #     architecture — see worker_pool.py).

# #     IMPORTANT: this function does NOT load or release the YOLO model. The
# #     model is a per-process lazy singleton owned by gadget_detector.py
# #     (_get_model() / release_model()) and analyzer.py's per-video pipeline
# #     never tears it down between videos or journeys. In the persistent
# #     worker-pool architecture that means the model loads once — on the
# #     first video of the first journey this worker process ever handles —
# #     and then stays resident in this process's memory/CUDA context for
# #     every journey the worker processes afterwards, for as long as the
# #     worker lives. Only the per-journey temporary resources (VideoCapture,
# #     frames, numpy arrays, temp CUDA tensors) are released below, via
# #     resource_manager's cleanup hooks — never the model itself.
# #     """
# #     # ── Resource lifecycle (Phase 3): clean slate before this journey ────────
# #     # This is a brand-new `spawn`'d interpreter, so there is no Python-level
# #     # model object here yet. But the GPU itself may still be holding VRAM
# #     # reserved by a PREVIOUS journey's process if the driver hasn't finished
# #     # tearing down that process's CUDA context yet (this can lag process
# #     # exit, especially under back-to-back journeys). cleanup_before_journey()
# #     # is a no-op if the device is already clean, and prevents this journey's
# #     # model load from fighting over memory that should have been freed.
# #     try:
# #         from resource_manager import resource_manager
# #         resource_manager.cleanup_before_journey(job_id=job_id)
# #     except Exception:
# #         pass  # resource_manager unavailable — not fatal, proceed anyway

# #     def _video_done_cb(video_id: int, ok: bool, error: str | None,
# #                         error_type: str | None = None,
# #                         stack_trace: str | None = None,
# #                         reason: str | None = None,
# #                         video_result: dict | None = None) -> None:
# #         try:
# #             events_q.put(
# #                 {
# #                     "type":         "video_done",
# #                     "video_id":     video_id,
# #                     "ok":           ok,
# #                     "error":        error,
# #                     "error_type":   error_type,
# #                     "stack_trace":  stack_trace,
# #                     "reason":       reason,
# #                     # Best-effort per-video result snapshot (see analyzer.py's
# #                     # _build_partial_video_result) — only present when ok=True.
# #                     # Lets the parent preserve real violation data for videos
# #                     # that finished before a native crash kills this child,
# #                     # instead of having to mark them failed too.
# #                     "video_result": video_result,
# #                 },
# #                 block=False,
# #             )
# #         except Exception:
# #             # Never let a full/broken events queue take down the child —
# #             # the parent's primary crash signal is exitcode + result_q
# #             # absence, the events queue is best-effort progress detail.
# #             pass
# #         # ── Resource lifecycle (Phase 3): per-video GPU cleanup ──────────────
# #         # Fired after every video, success or failure, so VRAM fragmentation
# #         # doesn't accumulate across the videos within a single journey.
# #         try:
# #             from resource_manager import resource_manager
# #             resource_manager.cleanup_after_video(job_id=job_id, video_id=video_id)
# #         except Exception:
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
# #             from resource_manager import resource_manager
# #             resource_manager.cleanup_after_failure(
# #                 job_id=job_id, reason=f"{type(exc).__name__}: {exc}",
# #             )
# #         except Exception:
# #             pass
# #         try:
# #             result_q.put({
# #                 "type":      "error",
# #                 "message":   str(exc),
# #                 "traceback": traceback.format_exc(),
# #             })
# #         except Exception:
# #             pass
# #     finally:
# #         # ── Resource lifecycle (Phase 3/4): release TEMPORARY resources ──────
# #         # cleanup_after_journey() flushes the CUDA allocator cache / runs GC
# #         # and logs RSS/GPU deltas — it deliberately does NOT touch the YOLO
# #         # model singleton (see gadget_detector.py's release_model() docstring
# #         # and resource_manager.cleanup_after_journey()'s docstring). That is
# #         # exactly what makes this function safe to call from inside a
# #         # persistent worker's loop (run_worker_loop below): every journey
# #         # gets a clean VRAM slate, but the model stays loaded across
# #         # journeys for the lifetime of the worker process. Runs on every
# #         # path: clean success, caught exception above, AND — to the extent
# #         # Python is still alive to run a finally at all — anything else. A
# #         # genuine native crash (OOM/access violation) skips this entirely;
# #         # that case is handled by worker_pool.py's crash detection instead
# #         # (the dead worker process is replaced with a fresh one).
# #         try:
# #             from resource_manager import resource_manager
# #             resource_manager.cleanup_after_journey(job_id=job_id)
# #         except Exception:
# #             pass


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
# #     Back-compat ONE-SHOT entry point: target function for a throwaway
# #     multiprocessing.Process that processes exactly one journey and then
# #     exits. Kept for any caller (tests, CLI tooling) that still wants the
# #     old "fresh process per journey" behavior. The live worker-pool
# #     architecture (consumer.py + worker_pool.py) uses run_worker_loop()
# #     below instead, so the model doesn't get reloaded every journey.
# #     """
# #     os.environ.setdefault("OMP_NUM_THREADS", "1")
# #     os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")
# #     _process_one_journey(
# #         job_id, journey_id, folder_name, video_jobs, tmp_paths,
# #         events_q, result_q,
# #     )


# # def run_worker_loop(
# #     worker_id: int,
# #     job_queue: "mp.Queue",
# #     events_q: "mp.Queue",
# #     result_q: "mp.Queue",
# # ) -> None:
# #     """
# #     Target function for a PERSISTENT GPU worker process (multiprocessing
# #     .Process, spawned once at service startup by worker_pool.py and kept
# #     alive for the lifetime of the service).

# #     Loads nothing eagerly. The YOLO model (gadget_detector._get_model())
# #     lazy-loads on the first video of the first journey this worker
# #     processes, and then stays resident — model weights, CUDA context, the
# #     works — in this process's memory for every subsequent journey handed
# #     to it, for as long as this worker lives. That is the entire point of
# #     the persistent worker-pool architecture: journeys arrive one after
# #     another on job_queue, but "Load YOLO / Load TensorRT / Initialize
# #     CUDA" only ever happens once per worker, not once per journey.

# #     job_queue protocol
# #     ───────────────────
# #     Each item is either:
# #       • a 5-tuple (job_id, journey_id, folder_name, video_jobs, tmp_paths)
# #         — process this journey, then loop back and wait for the next one.
# #       • None — shutdown sentinel; exit the loop and let the process end.

# #     events_q / result_q are the SAME pair of queues for every journey this
# #     worker ever processes (created once by worker_pool.py alongside
# #     job_queue) — the caller distinguishes one journey's events from
# #     another's by only ever having one journey in flight per worker at a
# #     time (worker_pool.py enforces this: a worker is only handed a new job
# #     once the previous one's result has been consumed).
# #     """
# #     # Keep OpenCV/MKL/etc. from oversubscribing CPU when several worker
# #     # processes run concurrently on the same host.
# #     os.environ.setdefault("OMP_NUM_THREADS", "1")
# #     os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

# #     print(f"[GPUWorker-{worker_id}]  Started  pid={os.getpid()}  "
# #           f"— waiting for journeys (model loads lazily on first video)...")

# #     while True:
# #         job = job_queue.get()  # blocks until a journey arrives or shutdown
# #         if job is None:
# #             print(f"[GPUWorker-{worker_id}]  Shutdown signal received — exiting.")
# #             break

# #         job_id, journey_id, folder_name, video_jobs, tmp_paths = job
# #         print(f"[GPUWorker-{worker_id}]  Picked up journey job={job_id} "
# #               f"({len(video_jobs)} video(s)).")
# #         try:
# #             _process_one_journey(
# #                 job_id, journey_id, folder_name, video_jobs, tmp_paths,
# #                 events_q, result_q,
# #             )
# #         except BaseException as exc:  # noqa: BLE001 - last line of defense
# #             # _process_one_journey already catches everything it can and
# #             # reports via result_q itself; this only fires for something
# #             # unexpected in the loop plumbing around it. A true native
# #             # crash (OOM/access violation/segfault) will NOT reach this
# #             # except — it kills the process directly, which worker_pool.py
# #             # detects via is_alive() and handles by spawning a replacement.
# #             try:
# #                 result_q.put({
# #                     "type":      "error",
# #                     "message":   str(exc),
# #                     "traceback": traceback.format_exc(),
# #                 })
# #             except Exception:
# #                 pass
# #         # ── Loop back — model/CUDA context stay loaded ────────────────────
# #         # Only this journey's temporary resources were released above
# #         # (resource_manager.cleanup_after_journey(), inside
# #         # _process_one_journey's finally block). The worker is now idle
# #         # and ready for the next journey worker_pool.py assigns it.
# #         print(f"[GPUWorker-{worker_id}]  Finished journey job={job_id} "
# #               f"— waiting for next journey.")

# # # """
# # # journey_runner.py
# # # ──────────────────
# # # Runs inside an isolated CHILD PROCESS (multiprocessing.Process), spawned by
# # # consumer.py for every journey. This is the fix for native OpenCV crashes
# # # (cv2 OutOfMemoryError, cv2.pyd access violation, segfault) that previously
# # # killed the ENTIRE worker — including RabbitMQ's connection thread and every
# # # other in-flight job — because a native crash terminates the OS process and
# # # no Python `except`, not even `except BaseException`, can run after that.

# # # Why a subprocess fixes this
# # # ────────────────────────────
# # # A Python try/except can only catch things the interpreter is still alive
# # # to raise. A genuine native allocation failure or access violation kills
# # # the process before any Python exception object even exists. The only way
# # # to "catch" that is from OUTSIDE the process: the parent (consumer.py)
# # # spawns this file's `run_journey_in_subprocess()` target as a child, and
# # # watches `process.exitcode` after `process.join()`. If the child died
# # # abnormally, the parent treats it as a crash and is responsible for
# # # deciding which videos were already completed vs. not.

# # # How the parent finds out which videos already finished
# # # ────────────────────────────────────────────────────────
# # # `analyze_journey()` in analyzer.py already processes videos one at a time
# # # in a for-loop. We pass it an optional `video_done_cb` (see the small patch
# # # to analyzer.py) that fires immediately after each video — success OR
# # # per-video-caught-failure — completes. Here, that callback pushes a small
# # # picklable progress event onto `events_q` (a multiprocessing.Queue shared
# # # with the parent). The parent drains this queue continuously, so even if
# # # the child is killed by the OS one frame into video 3, the parent already
# # # knows videos 1 and 2 finished (or failed) and only needs to mark video 3
# # # (and anything after it) as failed — exactly matching "if a video can't be
# # # processed due to OOM, treat the remaining videos as failed too."

# # # The final result (video_results / wall_seconds / failed_videos) is sent
# # # back over `result_q` only on a clean return. If the child dies, `result_q`
# # # never receives anything — that absence IS the crash signal the parent
# # # checks for.
# # # """

# # # from __future__ import annotations

# # # import multiprocessing as mp
# # # import os
# # # import traceback
# # # from typing import Dict, List

# # # from analyzer import analyze_journey
# # # from models import VideoJob


# # # # Event dicts pushed onto events_q by video_done_cb, e.g.:
# # # #   {"type": "video_done", "video_id": 12, "ok": True,  "error": None,
# # # #    "error_type": None, "stack_trace": None, "reason": None}
# # # #   {"type": "video_done", "video_id": 13, "ok": False, "error": "decode error",
# # # #    "error_type": "DECODE_ERROR", "stack_trace": "...", "reason": None}
# # # #   {"type": "video_done", "video_id": 14, "ok": False,
# # # #    "error": "Not Processed - Worker Resource Exhaustion",
# # # #    "error_type": "NOT_PROCESSED", "stack_trace": None,
# # # #    "reason": "Not Processed - Worker Resource Exhaustion"}
# # # #
# # # # Final outcome pushed onto result_q (at most one item, only on clean exit):
# # # #   {"type": "result", "video_results": [...], "wall_seconds": 87.3,
# # # #    "failed_videos": {13: "decode error"}}
# # # #   {"type": "error",  "traceback": "...", "message": "..."}


# # # def run_journey_in_subprocess(
# # #     job_id: str,
# # #     journey_id: int,
# # #     folder_name: str,
# # #     video_jobs: List[VideoJob],
# # #     tmp_paths: Dict[int, str],
# # #     events_q: "mp.Queue",
# # #     result_q: "mp.Queue",
# # # ) -> None:
# # #     """
# # #     Target function for multiprocessing.Process. Runs entirely in the
# # #     child. Must only communicate back to the parent via events_q/result_q
# # #     (no shared memory, no return value — multiprocessing.Process ignores
# # #     return values).
# # #     """
# # #     # Keep OpenCV/MKL/etc. from oversubscribing CPU when several of these
# # #     # child processes run concurrently on the same host (multi-user load).
# # #     os.environ.setdefault("OMP_NUM_THREADS", "1")
# # #     os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

# # #     def _video_done_cb(video_id: int, ok: bool, error: str | None,
# # #                         error_type: str | None = None,
# # #                         stack_trace: str | None = None,
# # #                         reason: str | None = None) -> None:
# # #         try:
# # #             events_q.put(
# # #                 {
# # #                     "type":        "video_done",
# # #                     "video_id":    video_id,
# # #                     "ok":          ok,
# # #                     "error":       error,
# # #                     "error_type":  error_type,
# # #                     "stack_trace": stack_trace,
# # #                     "reason":      reason,
# # #                 },
# # #                 block=False,
# # #             )
# # #         except Exception:
# # #             # Never let a full/broken events queue take down the child —
# # #             # the parent's primary crash signal is exitcode + result_q
# # #             # absence, the events queue is best-effort progress detail.
# # #             pass

# # #     try:
# # #         video_results, wall_seconds, failed_videos = analyze_journey(
# # #             job_id       = job_id,
# # #             journey_id   = journey_id,
# # #             folder_name  = folder_name,
# # #             video_jobs   = video_jobs,
# # #             tmp_paths    = tmp_paths,
# # #             progress_cb  = None,   # progress callbacks stay in the parent
# # #             video_done_cb = _video_done_cb,
# # #         )
# # #         result_q.put({
# # #             "type":          "result",
# # #             "video_results": [vr.to_dict() for vr in video_results],
# # #             "wall_seconds":  wall_seconds,
# # #             "failed_videos": failed_videos,
# # #         })
# # #     except BaseException as exc:  # noqa: BLE001 - last line of defense in the child
# # #         # Covers ordinary Python exceptions raised above analyzer.py's own
# # #         # per-video isolation (e.g. a bug in the dedup/frame-upload stage
# # #         # that runs after the per-video loop). True native crashes (OOM,
# # #         # access violation, segfault) will NOT reach this except — they
# # #         # kill the process directly, which is exactly the case the parent
# # #         # detects via process.exitcode / join() instead.
# # #         try:
# # #             result_q.put({
# # #                 "type":      "error",
# # #                 "message":   str(exc),
# # #                 "traceback": traceback.format_exc(),
# # #             })
# # #         except Exception:
# # #             pass


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

# #     # ── Resource lifecycle (Phase 3): clean slate before this journey ────────
# #     # This is a brand-new `spawn`'d interpreter, so there is no Python-level
# #     # model object here yet. But the GPU itself may still be holding VRAM
# #     # reserved by a PREVIOUS journey's process if the driver hasn't finished
# #     # tearing down that process's CUDA context yet (this can lag process
# #     # exit, especially under back-to-back journeys). cleanup_before_journey()
# #     # is a no-op if the device is already clean, and prevents this journey's
# #     # model load from fighting over memory that should have been freed.
# #     try:
# #         from resource_manager import resource_manager
# #         resource_manager.cleanup_before_journey(job_id=job_id)
# #     except Exception:
# #         pass  # resource_manager unavailable — not fatal, proceed anyway

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
# #         # ── Resource lifecycle (Phase 3): per-video GPU cleanup ──────────────
# #         # Fired after every video, success or failure, so VRAM fragmentation
# #         # doesn't accumulate across the videos within a single journey.
# #         try:
# #             from resource_manager import resource_manager
# #             resource_manager.cleanup_after_video(job_id=job_id, video_id=video_id)
# #         except Exception:
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
# #             from resource_manager import resource_manager
# #             resource_manager.cleanup_after_failure(
# #                 job_id=job_id, reason=f"{type(exc).__name__}: {exc}",
# #             )
# #         except Exception:
# #             pass
# #         try:
# #             result_q.put({
# #                 "type":      "error",
# #                 "message":   str(exc),
# #                 "traceback": traceback.format_exc(),
# #             })
# #         except Exception:
# #             pass
# #     finally:
# #         # ── Resource lifecycle (Phase 3/4): release model + VRAM before exit ─
# #         # This process is about to die anyway (each journey gets a fresh
# #         # `spawn`'d process), but doing this explicitly — rather than
# #         # relying solely on OS/driver cleanup timing after process exit —
# #         # means the NEXT journey's child process is far less likely to
# #         # start while this one's CUDA context is still being torn down.
# #         # Runs on every path: clean success, caught exception above, AND
# #         # — to the extent Python is still alive to run a finally at all —
# #         # anything else. A genuine native crash (OOM/access violation)
# #         # skips this entirely; that case is handled by the parent's
# #         # exitcode-based emergency_cleanup() in consumer.py instead.
# #         try:
# #             from resource_manager import resource_manager
# #             resource_manager.cleanup_after_journey(job_id=job_id)
# #         except Exception:
# #             pass


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


# def _process_one_journey(
#     job_id: str,
#     journey_id: int,
#     folder_name: str,
#     video_jobs: List[VideoJob],
#     tmp_paths: Dict[int, str],
#     events_q: "mp.Queue",
#     result_q: "mp.Queue",
# ) -> None:
#     """
#     Runs exactly ONE journey's analysis and reports back over
#     events_q/result_q — the same wire format regardless of whether this is
#     called from a one-shot subprocess (run_journey_in_subprocess, kept for
#     backward compatibility) or from inside a long-lived GPU worker process's
#     main loop (run_worker_loop, used by the persistent worker-pool
#     architecture — see worker_pool.py).

#     IMPORTANT: this function does NOT load or release the YOLO model. The
#     model is a per-process lazy singleton owned by gadget_detector.py
#     (_get_model() / release_model()) and analyzer.py's per-video pipeline
#     never tears it down between videos or journeys. In the persistent
#     worker-pool architecture that means the model loads once — on the
#     first video of the first journey this worker process ever handles —
#     and then stays resident in this process's memory/CUDA context for
#     every journey the worker processes afterwards, for as long as the
#     worker lives. Only the per-journey temporary resources (VideoCapture,
#     frames, numpy arrays, temp CUDA tensors) are released below, via
#     resource_manager's cleanup hooks — never the model itself.
#     """
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
#                         reason: str | None = None,
#                         video_result: dict | None = None) -> None:
#         try:
#             events_q.put(
#                 {
#                     "type":         "video_done",
#                     "video_id":     video_id,
#                     "ok":           ok,
#                     "error":        error,
#                     "error_type":   error_type,
#                     "stack_trace":  stack_trace,
#                     "reason":       reason,
#                     # Best-effort per-video result snapshot (see analyzer.py's
#                     # _build_partial_video_result) — only present when ok=True.
#                     # Lets the parent preserve real violation data for videos
#                     # that finished before a native crash kills this child,
#                     # instead of having to mark them failed too.
#                     "video_result": video_result,
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
#         # ── Resource lifecycle (Phase 3/4): release TEMPORARY resources ──────
#         # cleanup_after_journey() flushes the CUDA allocator cache / runs GC
#         # and logs RSS/GPU deltas — it deliberately does NOT touch the YOLO
#         # model singleton (see gadget_detector.py's release_model() docstring
#         # and resource_manager.cleanup_after_journey()'s docstring). That is
#         # exactly what makes this function safe to call from inside a
#         # persistent worker's loop (run_worker_loop below): every journey
#         # gets a clean VRAM slate, but the model stays loaded across
#         # journeys for the lifetime of the worker process. Runs on every
#         # path: clean success, caught exception above, AND — to the extent
#         # Python is still alive to run a finally at all — anything else. A
#         # genuine native crash (OOM/access violation) skips this entirely;
#         # that case is handled by worker_pool.py's crash detection instead
#         # (the dead worker process is replaced with a fresh one).
#         try:
#             from resource_manager import resource_manager
#             resource_manager.cleanup_after_journey(job_id=job_id)
#         except Exception:
#             pass


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
#     Back-compat ONE-SHOT entry point: target function for a throwaway
#     multiprocessing.Process that processes exactly one journey and then
#     exits. Kept for any caller (tests, CLI tooling) that still wants the
#     old "fresh process per journey" behavior. The live worker-pool
#     architecture (consumer.py + worker_pool.py) uses run_worker_loop()
#     below instead, so the model doesn't get reloaded every journey.
#     """
#     os.environ.setdefault("OMP_NUM_THREADS", "1")
#     os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")
#     _process_one_journey(
#         job_id, journey_id, folder_name, video_jobs, tmp_paths,
#         events_q, result_q,
#     )


# def run_worker_loop(
#     worker_id: int,
#     job_queue: "mp.Queue",
#     events_q: "mp.Queue",
#     result_q: "mp.Queue",
# ) -> None:
#     """
#     Target function for a PERSISTENT GPU worker process (multiprocessing
#     .Process, spawned once at service startup by worker_pool.py and kept
#     alive for the lifetime of the service).

#     Loads nothing eagerly. The YOLO model (gadget_detector._get_model())
#     lazy-loads on the first video of the first journey this worker
#     processes, and then stays resident — model weights, CUDA context, the
#     works — in this process's memory for every subsequent journey handed
#     to it, for as long as this worker lives. That is the entire point of
#     the persistent worker-pool architecture: journeys arrive one after
#     another on job_queue, but "Load YOLO / Load TensorRT / Initialize
#     CUDA" only ever happens once per worker, not once per journey.

#     job_queue protocol
#     ───────────────────
#     Each item is either:
#       • a 5-tuple (job_id, journey_id, folder_name, video_jobs, tmp_paths)
#         — process this journey, then loop back and wait for the next one.
#       • None — shutdown sentinel; exit the loop and let the process end.

#     events_q / result_q are the SAME pair of queues for every journey this
#     worker ever processes (created once by worker_pool.py alongside
#     job_queue) — the caller distinguishes one journey's events from
#     another's by only ever having one journey in flight per worker at a
#     time (worker_pool.py enforces this: a worker is only handed a new job
#     once the previous one's result has been consumed).
#     """
#     # Keep OpenCV/MKL/etc. from oversubscribing CPU when several worker
#     # processes run concurrently on the same host.
#     os.environ.setdefault("OMP_NUM_THREADS", "1")
#     os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

#     print(f"[GPUWorker-{worker_id}]  Started  pid={os.getpid()}  "
#           f"— waiting for journeys (model loads lazily on first video)...")

#     while True:
#         job = job_queue.get()  # blocks until a journey arrives or shutdown
#         if job is None:
#             print(f"[GPUWorker-{worker_id}]  Shutdown signal received — exiting.")
#             break

#         job_id, journey_id, folder_name, video_jobs, tmp_paths = job
#         print(f"[GPUWorker-{worker_id}]  Picked up journey job={job_id} "
#               f"({len(video_jobs)} video(s)).")
#         try:
#             _process_one_journey(
#                 job_id, journey_id, folder_name, video_jobs, tmp_paths,
#                 events_q, result_q,
#             )
#         except BaseException as exc:  # noqa: BLE001 - last line of defense
#             # _process_one_journey already catches everything it can and
#             # reports via result_q itself; this only fires for something
#             # unexpected in the loop plumbing around it. A true native
#             # crash (OOM/access violation/segfault) will NOT reach this
#             # except — it kills the process directly, which worker_pool.py
#             # detects via is_alive() and handles by spawning a replacement.
#             try:
#                 result_q.put({
#                     "type":      "error",
#                     "message":   str(exc),
#                     "traceback": traceback.format_exc(),
#                 })
#             except Exception:
#                 pass
#         # ── Loop back — model/CUDA context stay loaded ────────────────────
#         # Only this journey's temporary resources were released above
#         # (resource_manager.cleanup_after_journey(), inside
#         # _process_one_journey's finally block). The worker is now idle
#         # and ready for the next journey worker_pool.py assigns it.
#         print(f"[GPUWorker-{worker_id}]  Finished journey job={job_id} "
#               f"— waiting for next journey.")




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


def _process_one_journey(
    job_id: str,
    journey_id: int,
    folder_name: str,
    video_jobs: List[VideoJob],
    tmp_paths: Dict[int, str],
    events_q: "mp.Queue",
    result_q: "mp.Queue",
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
        )


def _process_one_journey_impl(
    job_id: str,
    journey_id: int,
    folder_name: str,
    video_jobs: List[VideoJob],
    tmp_paths: Dict[int, str],
    events_q: "mp.Queue",
    result_q: "mp.Queue",
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
    )


def _worker_memory_line(job_id: str = "") -> str:
    """
    Best-effort 'RSS=...GB GPU=...GB' fragment for worker lifecycle log
    lines (see run_worker_loop below). Reuses the same RSS/GPU snapshot
    helpers analyzer.py and resource_manager.py already use elsewhere, so
    the numbers are directly comparable to the rest of the memory logging
    around a journey. Never raises — falls back to 'n/a' fields.
    """
    rss_str = "n/a"
    gpu_str = "n/a"
    try:
        from resource_manager import get_process_snapshot, get_gpu_memory_info
        snap = get_process_snapshot()
        if snap.rss_gb >= 0:
            rss_str = f"{snap.rss_gb:.3f}GB"
        gpu = get_gpu_memory_info()
        if gpu.available:
            gpu_str = f"{gpu.reserved_gb:.3f}GB"
    except Exception:
        pass
    return f"RSS={rss_str} GPU={gpu_str}"


def run_worker_loop(
    worker_id: int,
    job_queue: "mp.Queue",
    events_q: "mp.Queue",
    result_q: "mp.Queue",
) -> None:
    """
    Target function for a GPU worker process (multiprocessing.Process,
    spawned by worker_pool.py — once at service startup for the initial
    pool, and again by GPUWorkerPool.recycle_worker() every time a worker
    finishes a journey).

    ── Worker memory lifecycle ─────────────────────────────────────────
    This worker processes EXACTLY ONE journey and then returns, letting
    the process exit on its own. It does NOT loop back to wait for a
    second journey. That single-journey-per-process design is the whole
    point: process exit is the only mechanism that reliably guarantees
    the OS reclaims every native allocation the journey made — YOLO
    model runtime memory, PyTorch CPU/CUDA memory and the worker's CUDA
    context, OpenCV native buffers/VideoCapture/VideoWriter resources,
    MediaPipe native graph resources, NumPy frame arrays, the journey's
    ViolationStore/frame cache, temp journey objects, and any other
    native allocation belonging to this worker. gc.collect() /
    torch.cuda.empty_cache() alone (already run inside
    resource_manager.cleanup_after_journey(), see below) cannot
    guarantee that; only the process actually exiting can.

    worker_pool.py (the parent) detects this worker's exit, joins it, and
    immediately spawns a fresh replacement worker so the configured
    GPU_WORKERS pool size is always maintained. GPU_WORKERS is never
    reduced by this — see GPUWorkerPool.recycle_worker(). Only THIS
    worker is recycled; every other worker keeps processing its own
    journey completely undisturbed.

    The worker is recycled only after the ENTIRE journey finishes — every
    video in it has been attempted and the final journey result (success
    or already-marked-failed) has been safely handed to the parent over
    result_q — never mid-journey / after a single video.

    job_queue protocol
    ───────────────────
    The single item this worker ever reads is either:
      • a 5-tuple (job_id, journey_id, folder_name, video_jobs, tmp_paths)
        — process this one journey, then exit.
      • None — shutdown sentinel (no journey was ever assigned to this
        worker); exit immediately.
    """
    # Keep OpenCV/MKL/etc. from oversubscribing CPU when several worker
    # processes run concurrently on the same host.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

    pid = os.getpid()
    print(f"[WORKER START] worker_id={worker_id} pid={pid}")
    print(f"[GPUWorker-{worker_id}]  Started  pid={pid}  "
          f"— waiting for one journey (model loads lazily on first video)...")

    job = job_queue.get()  # blocks until a journey arrives or shutdown
    if job is None:
        print(f"[GPUWorker-{worker_id}]  Shutdown signal received before "
              f"any journey was assigned — exiting.")
        print(f"[WORKER EXIT] worker_id={worker_id} pid={pid} exit_code=0")
        return

    job_id, journey_id, folder_name, video_jobs, tmp_paths = job
    print(f"[GPUWorker-{worker_id}]  Picked up journey job={job_id} "
          f"({len(video_jobs)} video(s)).")
    print(f"[JOURNEY START] worker_id={worker_id} pid={pid} "
          f"journey_id={journey_id}\n{_worker_memory_line(job_id)}")

    recycle_reason = "JOURNEY_COMPLETE"
    try:
        _process_one_journey(
            job_id, journey_id, folder_name, video_jobs, tmp_paths,
            events_q, result_q,
        )
    except BaseException as exc:  # noqa: BLE001 - last line of defense
        # _process_one_journey already catches everything it can and
        # reports via result_q itself; this only fires for something
        # unexpected in the loop plumbing around it. A true native
        # crash (OOM/access violation/segfault) will NOT reach this
        # except — it kills the process directly, which worker_pool.py
        # detects via is_alive() and handles the same way (replacement
        # spawned) via its crash path instead.
        recycle_reason = "JOURNEY_FAILED"
        try:
            result_q.put({
                "type":      "error",
                "message":   str(exc),
                "traceback": traceback.format_exc(),
            })
        except Exception:
            pass

    print(f"[JOURNEY COMPLETE] worker_id={worker_id} pid={pid} "
          f"journey_id={journey_id}\n{_worker_memory_line(job_id)}")

    # ── Recycle this worker — process exit, not reuse ─────────────────────
    # Every journey-temporary resource has already been released above
    # (resource_manager.cleanup_after_journey(), run inside
    # _process_one_journey's finally block — CUDA cache flush + GC). What
    # remains resident (YOLO model weights, CUDA context, any lingering
    # native handles) is released only by this process actually exiting,
    # which is exactly what returning from run_worker_loop now does —
    # worker_pool.py's recycle_worker() is waiting on this process to
    # exit and will spawn this worker's replacement the moment it does.
    print(f"[GPUWorker-{worker_id}]  Finished journey job={job_id} — "
          f"recycling this worker (reason={recycle_reason}).")
    print(f"[WORKER RECYCLE] worker_id={worker_id} pid={pid} "
          f"reason={recycle_reason}")
    print(f"[WORKER EXIT] worker_id={worker_id} pid={pid} exit_code=0")