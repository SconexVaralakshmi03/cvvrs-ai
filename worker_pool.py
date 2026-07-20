# """
# worker_pool.py
# ────────────────
# Persistent GPU worker-process pool.

# This replaces the old architecture — a brand-new `multiprocessing.Process`
# spawned per journey, which reloaded YOLO/TensorRT/CUDA from scratch every
# time and then threw the whole process away — with a small, FIXED-SIZE pool
# of long-lived worker processes (GPU_WORKERS, configurable, default 2)
# created ONCE when the service starts and kept alive for its entire
# lifetime:

#     Application Starts
#             │
#             ▼
#     Create GPU Worker Pool (GPU_WORKERS processes)
#             │
#             ▼
#     Worker 1: Load YOLO once, Load TensorRT once, Init CUDA once
#     Worker 2: Load YOLO once, Load TensorRT once, Init CUDA once
#             │
#             ▼
#     Workers stay alive, waiting for journeys.

# Each worker's main loop lives in journey_runner.run_worker_loop(): pull a
# journey off its private job_queue, process it (module-level YOLO singleton
# in gadget_detector.py loads on first use and is never released between
# journeys), release only that journey's temporary resources, then loop back
# and wait for the next journey. The model is loaded once per worker, not
# once per journey.

# consumer.py hands each incoming journey to `worker_pool.submit(...)`,
# which blocks until a worker is free, dispatches the job to it, and
# returns a JourneyHandle that consumer.py polls for events/result exactly
# the way it used to poll the old one-shot subprocess object. If a worker
# dies (native crash / OOM / stuck-and-killed-on-timeout) it is detected and
# automatically replaced with a fresh one, so the pool always has exactly
# GPU_WORKERS live workers.

# This module intentionally knows nothing about RabbitMQ, S3, callbacks, or
# the shape of a "journey" beyond the 5-tuple journey_runner already expects
# — it is purely a process-lifecycle manager, so none of the existing
# message format / callback / detection / database / report-generation code
# had to change.
# """

# from __future__ import annotations

# import logging
# import multiprocessing as mp
# import os
# import threading
# from dataclasses import dataclass
# from typing import Dict, List, Optional

# from journey_runner import run_worker_loop

# log = logging.getLogger("worker_pool")

# # Number of persistent GPU worker processes. Configurable — this is the
# # "GPU_WORKERS = 2" knob from the requirements. Each one keeps its own
# # YOLO/TensorRT/CUDA context loaded for as long as the service runs.
# GPU_WORKERS = int(os.environ.get("GPU_WORKERS", "2"))

# # How long to wait for a worker to exit cleanly after a shutdown/replace
# # signal before forcing it.
# _WORKER_JOIN_TIMEOUT_SECONDS = float(os.environ.get("GPU_WORKER_JOIN_TIMEOUT_SECONDS", "15"))


# @dataclass
# class _Worker:
#     id:        int
#     process:   "mp.process.BaseProcess"
#     job_queue: "mp.Queue"
#     events_q:  "mp.Queue"
#     result_q:  "mp.Queue"


# class JourneyHandle:
#     """
#     Returned by GPUWorkerPool.submit() in place of the old throwaway
#     `multiprocessing.Process` object. Exposes the same events_q/result_q
#     attributes consumer.py's existing polling loop already knows how to
#     drain, plus the two outcomes that loop needs to distinguish:

#       • The journey finished (cleanly or with a caught error) — the
#         worker process itself is healthy. Call mark_finished() to return
#         it to the pool for the next journey.
#       • The journey's worker died or had to be killed (crash / OOM /
#         timeout) — call discard() instead. The pool replaces the worker
#         with a fresh one; it is never returned to the free list.
#     """

#     def __init__(self, worker: "_Worker", pool: "GPUWorkerPool"):
#         self._worker   = worker
#         self._pool     = pool
#         self._resolved = False
#         self.events_q  = worker.events_q
#         self.result_q  = worker.result_q

#     @property
#     def pid(self) -> Optional[int]:
#         return self._worker.process.pid

#     def is_alive(self) -> bool:
#         return self._worker.process.is_alive()

#     def mark_finished(self) -> None:
#         """The journey finished and the worker process is still healthy —
#         return it to the pool so it can pick up the next queued journey."""
#         if self._resolved:
#             return
#         self._resolved = True
#         self._pool.release(self._worker)

#     def discard(self) -> None:
#         """The worker crashed, hung, or had to be force-killed after a
#         timeout — never reuse it. The pool kills it if still alive and
#         spawns a replacement so the pool size stays constant."""
#         if self._resolved:
#             return
#         self._resolved = True
#         self._pool.replace_dead_worker(self._worker)


# class GPUWorkerPool:
#     """
#     Manages a fixed-size set of persistent GPU worker processes and hands
#     journeys out to whichever one is free, blocking the caller only when
#     every worker is already busy (i.e. GPU_WORKERS journeys are already
#     running in parallel).
#     """

#     def __init__(self, size: int = GPU_WORKERS):
#         self._size       = max(1, size)
#         self._ctx        = mp.get_context("spawn")
#         self._workers:   List[_Worker] = []
#         self._free_list:  List[_Worker] = []
#         self._lock       = threading.Lock()
#         self._not_empty  = threading.Condition(self._lock)
#         self._next_id    = 0
#         self._started    = False

#     # ── Startup / shutdown ──────────────────────────────────────────────

#     def start(self) -> None:
#         """Spawn all GPU_WORKERS persistent workers. Call once, at service
#         startup, before the RabbitMQ consumer begins consuming."""
#         if self._started:
#             return
#         log.info("[WorkerPool] Starting %d persistent GPU worker(s)...", self._size)
#         for _ in range(self._size):
#             self._spawn_worker()
#         self._started = True
#         log.info(
#             "[WorkerPool] All %d worker(s) started and waiting for journeys "
#             "(pids=%s).", self._size, [w.process.pid for w in self._workers],
#         )

#     def _spawn_worker(self) -> _Worker:
#         wid = self._next_id
#         self._next_id += 1
#         job_queue = self._ctx.Queue()
#         events_q  = self._ctx.Queue()
#         result_q  = self._ctx.Queue()
#         process = self._ctx.Process(
#             target = run_worker_loop,
#             args   = (wid, job_queue, events_q, result_q),
#             name   = f"GPUWorker-{wid}",
#             daemon = True,
#         )
#         process.start()
#         worker = _Worker(id=wid, process=process, job_queue=job_queue,
#                           events_q=events_q, result_q=result_q)
#         with self._not_empty:
#             self._workers.append(worker)
#             self._free_list.append(worker)
#             self._not_empty.notify()
#         log.info("[WorkerPool] Worker %d started  pid=%s", wid, process.pid)
#         return worker

#     def shutdown(self) -> None:
#         """Call once, when the service is shutting down gracefully. Sends
#         each worker a poison pill and waits for it to exit."""
#         log.info("[WorkerPool] Shutting down %d worker(s)...", len(self._workers))
#         for w in list(self._workers):
#             try:
#                 w.job_queue.put(None)
#             except Exception:
#                 pass
#         for w in list(self._workers):
#             try:
#                 w.process.join(timeout=_WORKER_JOIN_TIMEOUT_SECONDS)
#                 if w.process.is_alive():
#                     w.process.terminate()
#                     w.process.join(timeout=5)
#             except Exception:
#                 pass
#         with self._lock:
#             self._workers.clear()
#             self._free_list.clear()
#         log.info("[WorkerPool] Shutdown complete.")

#     # ── Acquire / release ────────────────────────────────────────────────

#     def acquire(self) -> _Worker:
#         """Blocks until a worker is free, then reserves it for the caller."""
#         with self._not_empty:
#             while not self._free_list:
#                 self._not_empty.wait()
#             return self._free_list.pop(0)

#     def release(self, worker: "_Worker") -> None:
#         """Return a healthy, idle worker to the free pool."""
#         if not worker.process.is_alive():
#             # Died between finishing its journey and being released —
#             # treat exactly like any other crash.
#             self.replace_dead_worker(worker)
#             return
#         with self._not_empty:
#             if worker in self._workers and worker not in self._free_list:
#                 self._free_list.append(worker)
#                 self._not_empty.notify()

#     def replace_dead_worker(self, worker: "_Worker") -> None:
#         """A worker crashed, hung, or was force-killed after a timeout —
#         remove it permanently and spawn a fresh replacement so the pool
#         always has exactly `size` live workers."""
#         log.warning(
#             "[WorkerPool] Worker %d (pid=%s) is being removed from the pool "
#             "— spawning a replacement.", worker.id, worker.process.pid,
#         )
#         try:
#             if worker.process.is_alive():
#                 worker.process.terminate()
#                 worker.process.join(timeout=10)
#                 if worker.process.is_alive():
#                     worker.process.kill()
#                     worker.process.join(timeout=5)
#         except Exception:
#             log.warning("[WorkerPool] Error terminating worker %d", worker.id,
#                         exc_info=True)

#         with self._lock:
#             if worker in self._workers:
#                 self._workers.remove(worker)
#             if worker in self._free_list:
#                 self._free_list.remove(worker)

#         if self._started:
#             self._spawn_worker()

#     # ── Submit a journey ─────────────────────────────────────────────────

#     def submit(self, job_id: str, journey_id: int, folder_name: str,
#                video_jobs: list, tmp_paths: Dict[int, str]) -> JourneyHandle:
#         """
#         Blocks until a worker is free (i.e. until fewer than GPU_WORKERS
#         journeys are currently in flight), hands the journey to it, and
#         returns a JourneyHandle the caller polls exactly like the old
#         subprocess object (.events_q / .result_q / .is_alive()).
#         """
#         worker = self.acquire()
#         try:
#             worker.job_queue.put(
#                 (job_id, journey_id, folder_name, video_jobs, tmp_paths)
#             )
#         except Exception:
#             # The worker's queue itself is broken (e.g. worker died between
#             # acquire() and put()) — treat as dead, replace it, and retry
#             # once on a fresh worker rather than silently losing the
#             # journey.
#             log.warning(
#                 "[WorkerPool] Failed to dispatch job=%s to worker %d — "
#                 "replacing worker and retrying once.", job_id, worker.id,
#                 exc_info=True,
#             )
#             self.replace_dead_worker(worker)
#             worker = self.acquire()
#             worker.job_queue.put(
#                 (job_id, journey_id, folder_name, video_jobs, tmp_paths)
#             )

#         log.info(
#             "[WorkerPool] Journey job=%s dispatched to worker %d (pid=%s).",
#             job_id, worker.id, worker.process.pid,
#         )
#         return JourneyHandle(worker, self)

#     # ── Diagnostics ──────────────────────────────────────────────────────

#     def status(self) -> dict:
#         with self._lock:
#             return {
#                 "size":      self._size,
#                 "total":     len(self._workers),
#                 "free":      len(self._free_list),
#                 "busy":      len(self._workers) - len(self._free_list),
#                 "pids":      [w.process.pid for w in self._workers],
#             }


# # Module-level singleton — consumer.py imports and uses this directly, the
# # same way it already does for `resource_manager` / `memory_monitor`.
# worker_pool = GPUWorkerPool()


"""
worker_pool.py
────────────────
Persistent GPU worker-process pool.

This replaces the old architecture — a brand-new `multiprocessing.Process`
spawned per journey, which reloaded YOLO/TensorRT/CUDA from scratch every
time and then threw the whole process away — with a small, FIXED-SIZE pool
of long-lived worker processes (GPU_WORKERS, configurable, default 2)
created ONCE when the service starts and kept alive for its entire
lifetime:

    Application Starts
            │
            ▼
    Create GPU Worker Pool (GPU_WORKERS processes)
            │
            ▼
    Worker 1: Load YOLO once, Load TensorRT once, Init CUDA once
    Worker 2: Load YOLO once, Load TensorRT once, Init CUDA once
            │
            ▼
    Workers stay alive, waiting for journeys.

Each worker's main loop lives in journey_runner.run_worker_loop(): pull a
journey off its private job_queue, process it (module-level YOLO singleton
in gadget_detector.py loads on first use and is never released between
journeys), release only that journey's temporary resources, then loop back
and wait for the next journey. The model is loaded once per worker, not
once per journey.

consumer.py hands each incoming journey to `worker_pool.submit(...)`,
which blocks until a worker is free, dispatches the job to it, and
returns a JourneyHandle that consumer.py polls for events/result exactly
the way it used to poll the old one-shot subprocess object. If a worker
dies (native crash / OOM / stuck-and-killed-on-timeout) it is detected and
automatically replaced with a fresh one, so the pool always has exactly
GPU_WORKERS live workers.

This module intentionally knows nothing about RabbitMQ, S3, callbacks, or
the shape of a "journey" beyond the 5-tuple journey_runner already expects
— it is purely a process-lifecycle manager, so none of the existing
message format / callback / detection / database / report-generation code
had to change.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

from journey_runner import run_worker_loop

log = logging.getLogger("worker_pool")

# Number of persistent GPU worker processes. Configurable — this is the
# "GPU_WORKERS = 2" knob from the requirements. Each one keeps its own
# YOLO/TensorRT/CUDA context loaded for as long as the service runs.
GPU_WORKERS = int(os.environ.get("GPU_WORKERS", "4"))

# How long to wait for a worker to exit cleanly after a shutdown/replace
# signal before forcing it.
_WORKER_JOIN_TIMEOUT_SECONDS = float(os.environ.get("GPU_WORKER_JOIN_TIMEOUT_SECONDS", "15"))


@dataclass
class _Worker:
    id:        int
    process:   "mp.process.BaseProcess"
    job_queue: "mp.Queue"
    events_q:  "mp.Queue"
    result_q:  "mp.Queue"


class JourneyHandle:
    """
    Returned by GPUWorkerPool.submit() in place of the old throwaway
    `multiprocessing.Process` object. Exposes the same events_q/result_q
    attributes consumer.py's existing polling loop already knows how to
    drain, plus the two outcomes that loop needs to distinguish:

      • The journey finished (cleanly or with a caught error) — the
        worker process itself is healthy. Call mark_finished() to return
        it to the pool for the next journey.
      • The journey's worker died or had to be killed (crash / OOM /
        timeout) — call discard() instead. The pool replaces the worker
        with a fresh one; it is never returned to the free list.
    """

    def __init__(self, worker: "_Worker", pool: "GPUWorkerPool"):
        self._worker   = worker
        self._pool     = pool
        self._resolved = False
        self.events_q  = worker.events_q
        self.result_q  = worker.result_q

    @property
    def pid(self) -> Optional[int]:
        return self._worker.process.pid

    def is_alive(self) -> bool:
        return self._worker.process.is_alive()

    def mark_finished(self) -> None:
        """The journey finished and the worker process is still healthy —
        return it to the pool so it can pick up the next queued journey."""
        if self._resolved:
            return
        self._resolved = True
        self._pool.release(self._worker)

    def discard(self) -> None:
        """The worker crashed, hung, or had to be force-killed after a
        timeout — never reuse it. The pool kills it if still alive and
        spawns a replacement so the pool size stays constant."""
        if self._resolved:
            return
        self._resolved = True
        self._pool.replace_dead_worker(self._worker)


class GPUWorkerPool:
    """
    Manages a fixed-size set of persistent GPU worker processes and hands
    journeys out to whichever one is free, blocking the caller only when
    every worker is already busy (i.e. GPU_WORKERS journeys are already
    running in parallel).
    """

    def __init__(self, size: int = GPU_WORKERS):
        self._size       = max(1, size)
        self._ctx        = mp.get_context("spawn")
        self._workers:   List[_Worker] = []
        self._free_list:  List[_Worker] = []
        self._lock       = threading.Lock()
        self._not_empty  = threading.Condition(self._lock)
        self._next_id    = 0
        self._started    = False

    # ── Startup / shutdown ──────────────────────────────────────────────

    def start(self) -> None:
        """Spawn all GPU_WORKERS persistent workers. Call once, at service
        startup, before the RabbitMQ consumer begins consuming."""
        if self._started:
            return
        log.info("[WorkerPool] Starting %d persistent GPU worker(s)...", self._size)
        for _ in range(self._size):
            self._spawn_worker()
        self._started = True
        log.info(
            "[WorkerPool] All %d worker(s) started and waiting for journeys "
            "(pids=%s).", self._size, [w.process.pid for w in self._workers],
        )

    def _spawn_worker(self) -> _Worker:
        wid = self._next_id
        self._next_id += 1
        job_queue = self._ctx.Queue()
        events_q  = self._ctx.Queue()
        result_q  = self._ctx.Queue()
        process = self._ctx.Process(
            target = run_worker_loop,
            args   = (wid, job_queue, events_q, result_q),
            name   = f"GPUWorker-{wid}",
            daemon = True,
        )
        process.start()
        worker = _Worker(id=wid, process=process, job_queue=job_queue,
                          events_q=events_q, result_q=result_q)
        with self._not_empty:
            self._workers.append(worker)
            self._free_list.append(worker)
            self._not_empty.notify()
        log.info("[WorkerPool] Worker %d started  pid=%s", wid, process.pid)
        return worker

    def shutdown(self) -> None:
        """Call once, when the service is shutting down gracefully. Sends
        each worker a poison pill and waits for it to exit."""
        log.info("[WorkerPool] Shutting down %d worker(s)...", len(self._workers))
        for w in list(self._workers):
            try:
                w.job_queue.put(None)
            except Exception:
                pass
        for w in list(self._workers):
            try:
                w.process.join(timeout=_WORKER_JOIN_TIMEOUT_SECONDS)
                if w.process.is_alive():
                    w.process.terminate()
                    w.process.join(timeout=5)
            except Exception:
                pass
        with self._lock:
            self._workers.clear()
            self._free_list.clear()
        log.info("[WorkerPool] Shutdown complete.")

    # ── Acquire / release ────────────────────────────────────────────────

    def acquire(self) -> _Worker:
        """Blocks until a worker is free, then reserves it for the caller."""
        with self._not_empty:
            while not self._free_list:
                self._not_empty.wait()
            return self._free_list.pop(0)

    def release(self, worker: "_Worker") -> None:
        """Return a healthy, idle worker to the free pool."""
        if not worker.process.is_alive():
            # Died between finishing its journey and being released —
            # treat exactly like any other crash.
            self.replace_dead_worker(worker)
            return
        with self._not_empty:
            if worker in self._workers and worker not in self._free_list:
                self._free_list.append(worker)
                self._not_empty.notify()

    def replace_dead_worker(self, worker: "_Worker") -> None:
        """A worker crashed, hung, or was force-killed after a timeout —
        remove it permanently and spawn a fresh replacement so the pool
        always has exactly `size` live workers."""
        log.warning(
            "[WorkerPool] Worker %d (pid=%s) is being removed from the pool "
            "— spawning a replacement.", worker.id, worker.process.pid,
        )
        try:
            if worker.process.is_alive():
                worker.process.terminate()
                worker.process.join(timeout=10)
                if worker.process.is_alive():
                    worker.process.kill()
                    worker.process.join(timeout=5)
        except Exception:
            log.warning("[WorkerPool] Error terminating worker %d", worker.id,
                        exc_info=True)

        with self._lock:
            if worker in self._workers:
                self._workers.remove(worker)
            if worker in self._free_list:
                self._free_list.remove(worker)

        if self._started:
            self._spawn_worker()

    # ── Submit a journey ─────────────────────────────────────────────────

    def submit(self, job_id: str, journey_id: int, folder_name: str,
               video_jobs: list, tmp_paths: Dict[int, str]) -> JourneyHandle:
        """
        Blocks until a worker is free (i.e. until fewer than GPU_WORKERS
        journeys are currently in flight), hands the journey to it, and
        returns a JourneyHandle the caller polls exactly like the old
        subprocess object (.events_q / .result_q / .is_alive()).
        """
        worker = self.acquire()
        try:
            worker.job_queue.put(
                (job_id, journey_id, folder_name, video_jobs, tmp_paths)
            )
        except Exception:
            # The worker's queue itself is broken (e.g. worker died between
            # acquire() and put()) — treat as dead, replace it, and retry
            # once on a fresh worker rather than silently losing the
            # journey.
            log.warning(
                "[WorkerPool] Failed to dispatch job=%s to worker %d — "
                "replacing worker and retrying once.", job_id, worker.id,
                exc_info=True,
            )
            self.replace_dead_worker(worker)
            worker = self.acquire()
            worker.job_queue.put(
                (job_id, journey_id, folder_name, video_jobs, tmp_paths)
            )

        log.info(
            "[WorkerPool] Journey job=%s dispatched to worker %d (pid=%s).",
            job_id, worker.id, worker.process.pid,
        )
        return JourneyHandle(worker, self)

    # ── Diagnostics ──────────────────────────────────────────────────────

    def status(self) -> dict:
        with self._lock:
            return {
                "size":      self._size,
                "total":     len(self._workers),
                "free":      len(self._free_list),
                "busy":      len(self._workers) - len(self._free_list),
                "pids":      [w.process.pid for w in self._workers],
            }


# Module-level singleton — consumer.py imports and uses this directly, the
# same way it already does for `resource_manager` / `memory_monitor`.
worker_pool = GPUWorkerPool()
