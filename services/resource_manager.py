"""
resource_manager.py
────────────────────
Centralized resource lifecycle management for the AI Video Analysis worker.

This module is the single place responsible for:
    • GPU (CUDA/torch) memory cleanup
    • Python garbage collection
    • Thread / process / file-handle accounting for diagnostics
    • Startup-time detection and removal of stale resources left behind by
      a previous crash or a Windows reboot
    • Periodic background memory/GPU/thread monitoring with leak warnings
    • Per-journey / per-video RAM/GPU/thread delta logging

Design notes
────────────
- Every public cleanup function is IDEMPOTENT — calling it twice in a row,
  or calling it when there is nothing to clean up, must never raise and
  must never change behavior. This matters because cleanup is invoked from
  multiple places (normal completion, exception handlers, crash-detection
  paths in the parent process) and we never want the cleanup call itself
  to become a new source of failure.
- torch is imported lazily and defensively everywhere. The worker must
  keep functioning (with GPU cleanup simply becoming a no-op) on a machine
  where torch/CUDA isn't available, rather than raising ImportError out of
  a cleanup path.
- This module performs NO changes to detection logic, model thresholds,
  violation generation, or business rules. It only manages the lifecycle
  of resources (memory, GPU, threads, files, processes) around that logic.
"""

from __future__ import annotations

import gc
import glob
import logging
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

log = logging.getLogger("resource_manager")

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    _PSUTIL_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Configuration (env-overridable, sensible defaults — no business logic here)
# ─────────────────────────────────────────────────────────────────────────────

# Memory monitor cadence (Phase 8).
MONITOR_INTERVAL_SECONDS = float(os.environ.get("RESOURCE_MONITOR_INTERVAL_SECONDS", "30"))

# Warn if process RSS grows by more than this many GB between two consecutive
# monitor ticks (a coarse leak signal, not a hard limit).
RSS_GROWTH_WARN_GB = float(os.environ.get("RESOURCE_RSS_GROWTH_WARN_GB", "0.5"))

# Warn if thread count grows by more than this between two consecutive ticks.
THREAD_GROWTH_WARN = int(os.environ.get("RESOURCE_THREAD_GROWTH_WARN", "10"))

# Abort a journey if CUDA reserved memory exceeds this fraction of total
# device memory (Phase 3 requirement #12 / "prevent CUDA OOM").
CUDA_ABORT_THRESHOLD = float(os.environ.get("RESOURCE_CUDA_ABORT_THRESHOLD", "0.85"))

# Where temp video files and stale workspace folders live. These must match
# what consumer.py / analyzer.py actually use, so startup cleanup looks in
# the right places.
TEMP_GLOB_PATTERNS = [
    os.path.join(tempfile.gettempdir(), "tmp*"),  # tempfile.mkstemp() default prefix
]
JOURNEY_WORKSPACE_ROOT = os.environ.get("JOURNEY_WORKSPACE_ROOT", "outputs")

# Files in JOURNEY_WORKSPACE_ROOT older than this are considered stale/orphaned
# from a crashed previous run (not from a journey currently in flight).
STALE_AGE_SECONDS = float(os.environ.get("RESOURCE_STALE_AGE_SECONDS", str(6 * 3600)))  # 6h


# ─────────────────────────────────────────────────────────────────────────────
# torch helpers — every call defensive, never raises, no-ops without CUDA
# ─────────────────────────────────────────────────────────────────────────────

def _torch():
    """Returns the torch module, or None if unavailable. Never raises."""
    try:
        import torch
        return torch
    except Exception:
        return None


def _cuda_available(torch_mod) -> bool:
    try:
        return bool(torch_mod is not None and torch_mod.cuda.is_available())
    except Exception:
        return False


@dataclass
class GpuMemoryInfo:
    available:       bool = False
    device:          str  = ""
    allocated_gb:     float = 0.0
    reserved_gb:      float = 0.0
    total_gb:         float = 0.0
    free_gb:          float = 0.0
    used_fraction:    float = 0.0   # reserved / total — what we threshold on


def get_gpu_memory_info(device_index: int = 0) -> GpuMemoryInfo:
    """
    Snapshot of current CUDA memory state. Always returns a valid
    GpuMemoryInfo — `available=False` (all-zero fields) if torch/CUDA
    isn't present, never raises.
    """
    torch = _torch()
    if not _cuda_available(torch):
        return GpuMemoryInfo(available=False)

    try:
        allocated = torch.cuda.memory_allocated(device_index) / (1024 ** 3)
        reserved  = torch.cuda.memory_reserved(device_index) / (1024 ** 3)
        free_b, total_b = torch.cuda.mem_get_info(device_index)
        free_gb  = free_b / (1024 ** 3)
        total_gb = total_b / (1024 ** 3)
        used_fraction = (total_gb - free_gb) / total_gb if total_gb > 0 else 0.0
        return GpuMemoryInfo(
            available     = True,
            device        = f"cuda:{device_index}",
            allocated_gb  = round(allocated, 3),
            reserved_gb   = round(reserved, 3),
            total_gb      = round(total_gb, 3),
            free_gb       = round(free_gb, 3),
            used_fraction = round(used_fraction, 4),
        )
    except Exception:
        log.warning("get_gpu_memory_info() failed", exc_info=True)
        return GpuMemoryInfo(available=False)


def gpu_cleanup() -> GpuMemoryInfo:
    """
    Phase 3 core cleanup sequence: gc.collect() → empty_cache() →
    ipc_collect() → synchronize(). Safe to call with no CUDA present (each
    step becomes a no-op) and safe to call repeatedly. Returns the GPU
    memory snapshot taken AFTER cleanup, for logging by the caller.
    """
    gc.collect()

    torch = _torch()
    if _cuda_available(torch):
        try:
            torch.cuda.empty_cache()
        except Exception:
            log.warning("torch.cuda.empty_cache() failed", exc_info=True)
        try:
            torch.cuda.ipc_collect()
        except Exception:
            log.warning("torch.cuda.ipc_collect() failed", exc_info=True)
        try:
            torch.cuda.synchronize()
        except Exception:
            log.warning("torch.cuda.synchronize() failed", exc_info=True)

    return get_gpu_memory_info()


def log_gpu_memory(label: str, job_id: str = "") -> GpuMemoryInfo:
    info = get_gpu_memory_info()
    if not info.available:
        log.info("[GPU %s] job=%s  CUDA not available", label, job_id)
        return info
    log.info(
        "[GPU %s] job=%s  device=%s  allocated=%.3fGB  reserved=%.3fGB  "
        "free=%.3fGB  total=%.3fGB  used=%.1f%%",
        label, job_id, info.device, info.allocated_gb, info.reserved_gb,
        info.free_gb, info.total_gb, info.used_fraction * 100,
    )
    return info


def is_gpu_over_threshold(threshold: float = CUDA_ABORT_THRESHOLD) -> bool:
    """
    True if CUDA is available AND current usage exceeds `threshold`.
    False (never aborts) when CUDA isn't available at all — this check is
    additive safety on top of CUDA, not a requirement for CUDA to exist.
    """
    info = get_gpu_memory_info()
    if not info.available:
        return False
    return info.used_fraction >= threshold


# ─────────────────────────────────────────────────────────────────────────────
# CPU / process diagnostics
# ─────────────────────────────────────────────────────────────────────────────

_PROCESS = psutil.Process(os.getpid()) if _PSUTIL_AVAILABLE else None


@dataclass
class ProcessSnapshot:
    rss_gb:        float = -1.0
    thread_count:  int   = 0
    open_handles:  int   = -1   # file descriptors / handles; -1 if unknown
    child_pids:    List[int] = field(default_factory=list)


def get_process_snapshot() -> ProcessSnapshot:
    """Best-effort process snapshot. Never raises; missing data is -1/empty."""
    snap = ProcessSnapshot(thread_count=threading.active_count())
    if _PROCESS is None:
        return snap
    try:
        snap.rss_gb = round(_PROCESS.memory_info().rss / (1024 ** 3), 4)
    except Exception:
        pass
    try:
        # num_handles() exists on Windows; num_fds() on POSIX.
        if hasattr(_PROCESS, "num_handles"):
            snap.open_handles = _PROCESS.num_handles()
        elif hasattr(_PROCESS, "num_fds"):
            snap.open_handles = _PROCESS.num_fds()
    except Exception:
        pass
    try:
        snap.child_pids = [c.pid for c in _PROCESS.children(recursive=True)]
    except Exception:
        pass
    return snap


def log_process_snapshot(label: str, job_id: str = "") -> ProcessSnapshot:
    snap = get_process_snapshot()
    rss_str = f"{snap.rss_gb:.3f}GB" if snap.rss_gb >= 0 else "n/a"
    handles_str = str(snap.open_handles) if snap.open_handles >= 0 else "n/a"
    log.info(
        "[PROC %s] job=%s  RSS=%s  threads=%d  handles=%s  children=%d",
        label, job_id, rss_str, snap.thread_count, handles_str, len(snap.child_pids),
    )
    return snap


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — central ResourceManager
# ─────────────────────────────────────────────────────────────────────────────

class ResourceManager:
    """
    Centralized entry point for every resource-lifecycle operation the
    worker needs. Stateless except for the bookkeeping needed to compute
    before/after deltas (Phase 9) — does not hold any GPU/model handles
    itself; those remain owned by gadget_detector.py / main.py, this class
    only triggers their cleanup hooks where applicable.

    All methods are idempotent and safe to call multiple times or in any
    order; none of them raise — failures are logged and swallowed, because
    a cleanup function failing must never block the worker from continuing
    to consume jobs.
    """

    def __init__(self) -> None:
        self._journey_start_snapshot: Optional[ProcessSnapshot] = None
        self._journey_start_gpu:      Optional[GpuMemoryInfo]   = None

    # ── Phase 2: startup ────────────────────────────────────────────────────

    def initialize_service(self) -> None:
        """
        Run once, at process start, BEFORE the RabbitMQ consumer connects.
        Order matches the required startup sequence:
          RESOURCE CLEANUP → GPU CLEANUP → TEMP FILE CLEANUP → PROCESS CLEANUP
        """
        log.info("[Startup] Resource cleanup beginning...")
        gc.collect()

        log.info("[Startup] GPU cleanup...")
        gpu_info = gpu_cleanup()
        if gpu_info.available:
            log.info(
                "[Startup] GPU state after cleanup: reserved=%.3fGB free=%.3fGB "
                "total=%.3fGB", gpu_info.reserved_gb, gpu_info.free_gb, gpu_info.total_gb,
            )
        else:
            log.info("[Startup] CUDA not available — skipping GPU memory report.")

        log.info("[Startup] Temp file cleanup...")
        removed = self._cleanup_stale_temp_files()
        log.info("[Startup] Removed %d stale temp file(s).", removed)

        log.info("[Startup] Stale journey workspace cleanup...")
        removed_dirs = self._cleanup_stale_journey_workspaces()
        log.info("[Startup] Removed %d stale journey workspace folder(s).", removed_dirs)

        log.info("[Startup] Process/zombie cleanup check...")
        zombies = self._detect_zombie_children()
        if zombies:
            log.warning("[Startup] Found %d zombie/orphan child process(es): %s",
                        len(zombies), zombies)
        else:
            log.info("[Startup] No zombie child processes detected.")

        log.info("[Startup] Resource cleanup complete — safe to start RabbitMQ consumer.")

    def _cleanup_stale_temp_files(self) -> int:
        """
        Remove leftover tempfile.mkstemp() files from a previous crashed
        run. Only removes files matching our own temp prefix pattern and
        older than STALE_AGE_SECONDS, so we never touch a file that could
        belong to a journey currently being downloaded by a still-running
        process (relevant if two workers briefly overlap during a restart).
        """
        removed = 0
        now = time.time()
        for pattern in TEMP_GLOB_PATTERNS:
            try:
                for path in glob.glob(pattern):
                    try:
                        if not os.path.isfile(path):
                            continue
                        if now - os.path.getmtime(path) < STALE_AGE_SECONDS:
                            continue
                        os.remove(path)
                        removed += 1
                    except OSError:
                        continue
            except Exception:
                log.warning("Stale temp file scan failed for pattern %s", pattern, exc_info=True)
        return removed

    def _cleanup_stale_journey_workspaces(self) -> int:
        """
        Remove incomplete/orphaned per-journey output folders (frames,
        partial logs) left behind by a journey that never finished because
        the worker crashed or the host rebooted mid-journey.
        """
        removed = 0
        if not os.path.isdir(JOURNEY_WORKSPACE_ROOT):
            return removed
        now = time.time()
        try:
            for name in os.listdir(JOURNEY_WORKSPACE_ROOT):
                full = os.path.join(JOURNEY_WORKSPACE_ROOT, name)
                try:
                    if not os.path.isdir(full):
                        continue
                    mtime = os.path.getmtime(full)
                    if now - mtime < STALE_AGE_SECONDS:
                        continue
                    shutil.rmtree(full, ignore_errors=True)
                    removed += 1
                except OSError:
                    continue
        except Exception:
            log.warning("Stale journey workspace scan failed", exc_info=True)
        return removed

    def _detect_zombie_children(self) -> List[int]:
        """
        Detect orphaned child processes left behind by a prior worker
        instance that died without reaping its multiprocessing.Process
        children (can happen after a kill -9 or Windows TerminateProcess
        on the parent). Detection only — does NOT kill anything
        automatically on startup, since a false positive here would kill a
        legitimate sibling worker; this is reported for operator visibility.
        """
        if _PROCESS is None:
            return []
        try:
            return [c.pid for c in _PROCESS.children(recursive=True) if c.is_running()]
        except Exception:
            return []

    # ── Phase 3 / 4: per-journey and per-video GPU + model cleanup ─────────

    def cleanup_before_journey(self, job_id: str = "") -> GpuMemoryInfo:
        """
        Call immediately before starting a new journey (in the process
        that is about to load/use the model — i.e. inside the child for
        the subprocess architecture, or in the parent if models are ever
        loaded there). Clears anything left over from the previous journey
        before this one allocates, and records the start-of-journey
        snapshot used later by cleanup_after_journey() to compute deltas.
        """
        gpu_info = gpu_cleanup()
        log_gpu_memory("JOURNEY START", job_id)
        self._journey_start_snapshot = log_process_snapshot("JOURNEY START", job_id)
        self._journey_start_gpu = gpu_info
        return gpu_info

    def cleanup_after_video(self, job_id: str = "", video_id: int = -1) -> GpuMemoryInfo:
        """
        Call after each individual video finishes (success or per-video
        failure) within a journey. Lighter-weight than the journey-level
        cleanup — primarily clears CUDA cache so the next video in the
        same journey starts from a known-clean state.
        """
        gpu_info = gpu_cleanup()
        log_gpu_memory(f"VIDEO END (video_id={video_id})", job_id)
        return gpu_info

    def cleanup_after_journey(self, job_id: str = "") -> GpuMemoryInfo:
        """
        Call once after a journey completes (successfully or with some
        videos failed, but the process itself did not crash).

        Model-handle cleanup is NOT done here — each GadgetDetectionPipeline
        instance releases its own YOLO/MediaPipe handles inside its
        _release_pipeline_resources() / run() finally block, which runs
        synchronously before analyze_journey()'s video loop moves on to
        the next video.  By the time this method is called, all per-video
        pipeline handles have already been closed.

        What we do here is the CUDA-cache / GC sweep that reclaims the
        residual reserved-but-unallocated VRAM that the CUDA allocator
        keeps as a pool after model tensors are freed.  This is what
        produces the post-journey "GPU reserved=0.162 GB" reading and the
        accompanying CUDA-leak warning — empty_cache() releases it back to
        the driver.

        NOTE: The previous implementation attempted
            from gadget_detector import release_model
        but 'gadget_detector' was never a top-level module — the detector
        lives at detector.gadget_detector and is instantiated per-pipeline,
        not as a global singleton.  That import always raised
        ModuleNotFoundError, which silenced the CUDA-cache flush that
        follows it (because the except swallowed the block), leaving the
        residual VRAM unreleased after every journey.
        """
        # Step 1: Python GC — break any reference cycles that might be
        # keeping tensor objects alive beyond their scope.
        gc.collect()

        # Step 2: CUDA-cache flush — returns the VRAM the allocator pools
        # back to the driver so the next journey starts from a clean slate.
        # gpu_cleanup() already does gc → empty_cache → ipc_collect →
        # synchronize in the correct order and is safe when CUDA is absent.
        gpu_info = gpu_cleanup()
        end_snapshot = log_process_snapshot("JOURNEY END", job_id)
        log_gpu_memory("JOURNEY END", job_id)
        self._log_journey_delta(job_id, end_snapshot, gpu_info)
        return gpu_info

    def cleanup_after_failure(self, job_id: str = "", reason: str = "") -> GpuMemoryInfo:
        """
        Call from any failure/exception handler — journey-level crash,
        per-video exception that escaped isolation, timeout-kill, etc.
        Identical effect to cleanup_after_journey() (full release + log),
        kept as a separate method per the spec so call sites can express
        intent clearly and so failure-path logging is distinguishable from
        the happy-path log line.
        """
        log.warning("[Cleanup] cleanup_after_failure() job=%s reason=%s", job_id, reason)
        return self.cleanup_after_journey(job_id=job_id)

    def cleanup_on_shutdown(self) -> None:
        """
        Call once when the worker process is shutting down gracefully
        (e.g. KeyboardInterrupt in consumer.start()). Best-effort final
        release of GPU memory; does not attempt to close RabbitMQ
        connections (that remains owned by consumer.py's own
        connection/channel objects).

        Per-pipeline model handles (YOLO, MediaPipe) are owned by
        GadgetDetectionPipeline instances and are released inside each
        pipeline's own run() finally block, so there is nothing to release
        here at the module level.  We just flush the CUDA allocator cache
        and run GC so the process exits without leaving reserved GPU memory.
        """
        log.info("[Shutdown] Releasing GPU memory and running final GC...")
        gc.collect()
        gpu_cleanup()
        log.info("[Shutdown] Cleanup complete.")

    def emergency_cleanup(self, reason: str = "") -> None:
        """
        Last-resort cleanup path — call when something has gone wrong badly
        enough that normal cleanup_after_failure() either cannot run or
        cannot be trusted (e.g. called from a watchdog after detecting a
        hung/killed child, or from a top-level except BaseException in the
        consumer loop). Deliberately does the absolute minimum that can
        still fail safely: GC + clear CUDA cache.

        Note: when this is called from the PARENT process after a child
        crash, the child's CUDA context and YOLO/MediaPipe handles died
        with the child process — the OS has already reclaimed them.  All
        the parent can usefully do is flush its own (typically empty) CUDA
        allocator cache, which is what gpu_cleanup() does.  Attempting to
        import and call release_model() from the parent is both unnecessary
        (the model never lives in the parent) and was previously broken
        (the module path 'gadget_detector' does not exist as a top-level
        import — the detector is at detector.gadget_detector and is
        instantiated per-pipeline, not as a global singleton).
        """
        log.error("[EMERGENCY CLEANUP] reason=%s", reason)
        try:
            gc.collect()
        except Exception:
            pass
        try:
            torch = _torch()
            if _cuda_available(torch):
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ── Phase 9: leak-detection delta logging ───────────────────────────────

    def _log_journey_delta(self, job_id: str, end_snapshot: ProcessSnapshot,
                            end_gpu: GpuMemoryInfo) -> None:
        if self._journey_start_snapshot is None:
            return
        start = self._journey_start_snapshot
        rss_delta = (
            round(end_snapshot.rss_gb - start.rss_gb, 4)
            if start.rss_gb >= 0 and end_snapshot.rss_gb >= 0 else None
        )
        thread_delta = end_snapshot.thread_count - start.thread_count
        gpu_delta = None
        if self._journey_start_gpu and self._journey_start_gpu.available and end_gpu.available:
            gpu_delta = round(end_gpu.reserved_gb - self._journey_start_gpu.reserved_gb, 4)

        log.info(
            "[LEAK CHECK] job=%s  RSS_delta=%s  thread_delta=%+d  GPU_reserved_delta=%s",
            job_id,
            f"{rss_delta:+.4f}GB" if rss_delta is not None else "n/a",
            thread_delta,
            f"{gpu_delta:+.4f}GB" if gpu_delta is not None else "n/a",
        )

        if rss_delta is not None and rss_delta > RSS_GROWTH_WARN_GB:
            log.warning(
                "[LEAK WARNING] job=%s  RSS grew by %.4fGB this journey "
                "(threshold=%.2fGB) — possible memory leak.",
                job_id, rss_delta, RSS_GROWTH_WARN_GB,
            )
        if thread_delta > THREAD_GROWTH_WARN:
            log.warning(
                "[LEAK WARNING] job=%s  thread count grew by %d this journey "
                "(threshold=%d) — possible thread leak.",
                job_id, thread_delta, THREAD_GROWTH_WARN,
            )
        if gpu_delta is not None and gpu_delta > 0.1:
            log.warning(
                "[LEAK WARNING] job=%s  GPU reserved memory grew by %.4fGB "
                "this journey and was not fully released — possible CUDA leak.",
                job_id, gpu_delta,
            )

        self._journey_start_snapshot = None
        self._journey_start_gpu = None


# Module-level singleton — the rest of the codebase imports and uses THIS
# instance rather than constructing its own, so all cleanup bookkeeping
# (e.g. journey-start snapshots for delta logging) is shared correctly.
resource_manager = ResourceManager()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 8 — background memory/GPU/thread monitor
# ─────────────────────────────────────────────────────────────────────────────

class MemoryMonitor:
    """
    Background daemon thread that periodically logs RAM/GPU/thread/process
    state and raises warnings when growth between ticks exceeds the
    configured thresholds. Independent of per-journey delta logging in
    ResourceManager — this runs continuously regardless of job activity,
    so it also catches leaks that build up slowly across many journeys
    rather than within a single one.
    """

    def __init__(self, interval_seconds: float = MONITOR_INTERVAL_SECONDS) -> None:
        self._interval = interval_seconds
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_rss_gb: Optional[float] = None
        self._last_thread_count: Optional[int] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return  # idempotent — already running
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="MemoryMonitor", daemon=True,
        )
        self._thread.start()
        log.info("[MemoryMonitor] started  interval=%.0fs", self._interval)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception:
                log.warning("[MemoryMonitor] tick failed", exc_info=True)
            self._stop_event.wait(self._interval)

    def _tick(self) -> None:
        proc = get_process_snapshot()
        gpu = get_gpu_memory_info()

        rss_str = f"{proc.rss_gb:.3f}GB" if proc.rss_gb >= 0 else "n/a"
        gpu_str = (
            f"reserved={gpu.reserved_gb:.3f}GB/{gpu.total_gb:.3f}GB "
            f"({gpu.used_fraction*100:.1f}%)"
            if gpu.available else "n/a"
        )

        # DIAGNOSTIC (root-cause investigation for uneven journey speed):
        # system-wide CPU% (all processes, all cores) and disk I/O counters,
        # sampled at the same cadence as everything else. interval=None
        # gives a non-blocking instantaneous reading based on the delta
        # since the LAST call — cheap, safe to call every tick.
        cpu_str  = "n/a"
        disk_str = "n/a"
        if _PSUTIL_AVAILABLE:
            try:
                cpu_pct = psutil.cpu_percent(interval=None)
                per_core = psutil.cpu_percent(interval=None, percpu=True)
                cpu_str = f"{cpu_pct:.0f}%  per_core={['%.0f'%c for c in per_core]}"
            except Exception:
                pass
            try:
                io = psutil.disk_io_counters()
                if io is not None:
                    disk_str = (
                        f"read={io.read_bytes/1024/1024:.0f}MB "
                        f"write={io.write_bytes/1024/1024:.0f}MB "
                        f"(cumulative)"
                    )
            except Exception:
                pass

        log.info(
            "[MemoryMonitor] RSS=%s  threads=%d  children=%d  handles=%s  "
            "GPU=%s  CPU=%s  DISK=%s",
            rss_str, proc.thread_count, len(proc.child_pids),
            proc.open_handles if proc.open_handles >= 0 else "n/a",
            gpu_str, cpu_str, disk_str,
        )

        if self._last_rss_gb is not None and proc.rss_gb >= 0:
            growth = proc.rss_gb - self._last_rss_gb
            if growth > RSS_GROWTH_WARN_GB:
                log.warning(
                    "[MemoryMonitor] RAM growth of %.3fGB since last check "
                    "(%.0fs ago) — possible leak.", growth, self._interval,
                )
        if self._last_thread_count is not None:
            growth = proc.thread_count - self._last_thread_count
            if growth > THREAD_GROWTH_WARN:
                log.warning(
                    "[MemoryMonitor] Thread count grew by %d since last check "
                    "— possible thread leak.", growth,
                )

        if proc.rss_gb >= 0:
            self._last_rss_gb = proc.rss_gb
        self._last_thread_count = proc.thread_count


# Module-level singleton, mirroring `resource_manager` above.
memory_monitor = MemoryMonitor()