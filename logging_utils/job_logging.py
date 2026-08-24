"""
job_logging.py
────────────────
Per-journey terminal log capture, kept STRICTLY SEPARATE from the
per-journey violations log — two different files, never mixed.

Every journey (job_id) gets:
  - a TERMINAL log (download/dispatch/callback lines + the pipeline's own
    banners/progress output) — everything EXCEPT violation events.
  - a VIOLATIONS log (only lines that look like a violation event, e.g.
    "[00:00:21] One of the pilots is using a mobile phone  [CRITICAL]").

This lets you pull up "the terminal log for journey X" or "the violations
for journey X" in isolation, instead of grepping either out of one big
interleaved multi-journey console.

Two independent capture mechanisms, because a journey's work spans two
different OS processes:

1. LOGGING-MODULE SIDE (consumer.py, callback_client.py, resource_manager.py,
   worker_pool.py — all in the single long-running consumer process, one
   thread per in-flight journey): a contextvars.ContextVar tags the current
   THREAD with its job_id, a logging.Filter attaches that tag to every
   LogRecord, and a shared logging.Handler routes tagged records to that
   job's TERMINAL file — in ADDITION to the normal console output, never
   replacing it. This side never produces violation lines, so it only
   ever writes to the terminal file. Safe under concurrency because
   contextvars are scoped per thread.

2. STDOUT/STDERR SIDE (journey_runner.py, inside each persistent GPU worker
   process): most of the detection pipeline's own output is bare print(),
   which the `logging` module never sees. Since each worker process only
   ever runs ONE journey at a time (see worker_pool.py's run_worker_loop),
   it's safe to temporarily redirect sys.stdout/sys.stderr for the
   duration of that journey — done via JobStdoutTee, which mirrors
   everything to the real console UNCHANGED while also classifying each
   complete line as violation-vs-terminal and routing it to exactly one
   of the two per-job files.

Both mechanisms compute their file paths purely from job_id (no IPC
needed, since both processes run on the same host), and consumer.py reads
each pair back separately and uploads them as two distinct S3 objects.
"""

from __future__ import annotations

import contextvars
import logging
import os
import re
import sys
import threading
from typing import Dict, Optional, TextIO

_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "jobs")

# Matches the pipeline's own violation-event print lines, e.g.:
#   [00:00:21] One of the pilots is using a mobile phone  [CRITICAL]
# Timestamp prefix "[HH:MM:SS]" + a "[SEVERITY]" suffix (CRITICAL/WARNING/
# INFO/etc). Everything else the worker prints (banners, progress,
# "Processing complete" summaries, per-video headers) does NOT match this
# and is treated as plain terminal/pipeline output instead.
_VIOLATION_LINE_RE = re.compile(r"^\[\d{2}:\d{2}:\d{2}\].*\[[A-Z]+\]\s*$")


def get_job_log_path(job_id: str, side: str) -> str:
    """
    logs/jobs/<job_id>.<side>.log

    side is one of:
      "consumer"   — this journey's consumer-process log.*() lines
                     (download, worker-dispatch, callback) — always
                     terminal/pipeline content, never violations.
      "worker"     — this journey's GPU-worker-process terminal output
                     (banners, progress, per-video headers) — everything
                     EXCEPT lines that look like a violation event.
      "violations" — ONLY this journey's violation-event lines (matched
                     by _VIOLATION_LINE_RE), kept completely separate so
                     the terminal log and the violations log never mix.

    Deliberately separate files per journey per side instead of one
    shared file multiple processes append to — both processes CAN compute
    the same path independently (no IPC needed), but concurrent
    multi-process append to one literal file is more fragile than it
    needs to be on Windows. Merged/labeled at read_and_clear() time.
    """
    os.makedirs(_LOG_DIR, exist_ok=True)
    safe = "".join(c for c in job_id if c.isalnum() or c in ("-", "_")) or "unknown"
    return os.path.join(_LOG_DIR, f"{safe}.{side}.log")


# ── logging-module side (consumer process) ────────────────────────────────────

current_job_id: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "current_job_id", default=None,
)


class _JobIdFilter(logging.Filter):
    """Attaches the calling THREAD's current job_id (if any) to every
    LogRecord that passes through, so _JobFileHandler knows where to route
    it. Reads contextvars.ContextVar, which is scoped per-thread — safe
    under the worker-pool's one-thread-per-journey concurrency model."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.job_id = current_job_id.get()
        return True


class _JobFileHandler(logging.Handler):
    """
    Routes every LogRecord that has a job_id (set via start_job() on the
    thread that emitted it) to logs/jobs/<job_id>.log, IN ADDITION to
    whatever the normal console/root handlers already do with it. Never
    suppresses or replaces normal logging output — only adds a per-job copy.
    """

    def __init__(self):
        super().__init__()
        self._files: Dict[str, TextIO] = {}
        self._lock = threading.Lock()
        self.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        ))

    def emit(self, record: logging.LogRecord) -> None:
        job_id = getattr(record, "job_id", None)
        if not job_id:
            return
        try:
            line = self.format(record)
        except Exception:
            return
        with self._lock:
            f = self._files.get(job_id)
            if f is None:
                try:
                    f = open(get_job_log_path(job_id, "consumer"), "a", encoding="utf-8")
                    self._files[job_id] = f
                except Exception:
                    return
            try:
                f.write(line + "\n")
                f.flush()
            except Exception:
                pass

    def close_job(self, job_id: str) -> None:
        """Call once a journey is fully done, so the file handle doesn't
        stay open for the rest of the worker's/consumer's lifetime across
        every journey it ever processes."""
        with self._lock:
            f = self._files.pop(job_id, None)
        if f is not None:
            try:
                f.close()
            except Exception:
                pass


_job_file_handler = _JobFileHandler()
_job_id_filter = _JobIdFilter()


def install(logger: Optional[logging.Logger] = None) -> None:
    """
    Attach per-job routing to a logger (defaults to the ROOT logger, so
    every module's logging.getLogger(...) calls are covered automatically —
    matches how consumer.py/callback_client.py/resource_manager.py/
    worker_pool.py already log). Call once at startup. Idempotent.
    """
    target = logger or logging.getLogger()
    if _job_file_handler not in target.handlers:
        target.addHandler(_job_file_handler)
    if _job_id_filter not in target.filters:
        target.addFilter(_job_id_filter)


def start_job(job_id: str) -> None:
    """Call at the very start of processing a journey, on whichever thread
    is doing that journey's work (consumer.py's _handle_job, right after
    job_id is known)."""
    current_job_id.set(job_id)


def finish_job(job_id: str) -> None:
    """Call once a journey is fully done (success, failure, or crash) to
    release its file handle. Distinct from callback_client.finish_job() —
    that one releases the in-progress dedup claim; this one releases a
    local file handle. Always call this one fully-qualified
    (job_logging.finish_job) to avoid confusing the two."""
    _job_file_handler.close_job(job_id)


def read_and_clear(job_id: str) -> "tuple[str, str]":
    """
    Read back everything captured for this journey so far, kept as TWO
    SEPARATE strings — never merged:

      terminal_text   = consumer-side lines + worker-side non-violation
                         lines (download, dispatch, callback, pipeline
                         banners/progress)
      violations_text = ONLY the worker-side violation-event lines

    Used by consumer.py to upload each to its own S3 file, the same way
    the existing structured <jobId>.txt log is uploaded. Deletes all
    local files afterward. finish_job() should be called first so the
    consumer-side file handle is closed and flushed; the worker-side
    files are already closed by the time this runs (JobStdoutTee closes
    them on __exit__, which happens before result_q ever receives this
    journey's result).
    """
    def _read_and_remove(path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            text = ""
        try:
            os.remove(path)
        except OSError:
            pass
        return text

    consumer_text   = _read_and_remove(get_job_log_path(job_id, "consumer"))
    worker_text     = _read_and_remove(get_job_log_path(job_id, "worker"))
    violations_text = _read_and_remove(get_job_log_path(job_id, "violations"))

    terminal_parts = []
    if consumer_text.strip():
        terminal_parts.append(
            f"{'='*70}\nCONSUMER-SIDE (download / worker-dispatch / callback)\n"
            f"{'='*70}\n{consumer_text}"
        )
    if worker_text.strip():
        terminal_parts.append(
            f"{'='*70}\nGPU-WORKER-SIDE (pipeline output, violations excluded)\n"
            f"{'='*70}\n{worker_text}"
        )
    terminal_text = "\n\n".join(terminal_parts)

    return terminal_text, violations_text


# ── stdout/stderr side (GPU worker process) ───────────────────────────────────

class _ClassifyingTeeStream:
    """
    Writes every write() to the real original stream unchanged (the live
    console still shows everything, interleaved, exactly as before) WHILE
    ALSO splitting complete lines between two per-job files based on
    whether each line looks like a violation-event line:

      - matches _VIOLATION_LINE_RE  → violations_file ONLY
      - everything else             → terminal_file ONLY

    Never both — that's the whole point (keep terminal output and the
    violations log from ever mixing).

    print()/write() calls don't guarantee whole-line chunks, so this
    buffers any trailing partial line across calls and only classifies
    once a '\\n' completes it.
    """

    def __init__(self, real_stream: TextIO, terminal_file: TextIO, violations_file: TextIO):
        self._real = real_stream
        self._terminal = terminal_file
        self._violations = violations_file
        self._buf = ""

    def write(self, data: str) -> int:
        try:
            self._real.write(data)
        except Exception:
            pass

        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            target = self._violations if _VIOLATION_LINE_RE.match(line.strip()) else self._terminal
            try:
                target.write(line + "\n")
            except Exception:
                pass
        return len(data)

    def flush(self) -> None:
        # Flush any leftover partial line (no trailing newline yet) as
        # terminal content — a violation line is always complete by the
        # time the pipeline prints it, so a dangling partial fragment is
        # never itself a violation line.
        if self._buf:
            try:
                self._terminal.write(self._buf)
            except Exception:
                pass
            self._buf = ""
        for s in (self._real, self._terminal, self._violations):
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        try:
            return self._real.isatty()
        except Exception:
            return False


class JobStdoutTee:
    """
    Context manager for the GPU worker process side: redirects
    sys.stdout/sys.stderr for the duration of the `with` block so every
    bare print() (the detection pipeline's output is almost entirely
    print(), not logging) is classified line-by-line and routed to
    exactly one of two per-job files — logs/jobs/<job_id>.worker.log
    (terminal/pipeline output) or logs/jobs/<job_id>.violations.log
    (violation events only) — while the real console still shows
    everything, unchanged, exactly as before.

    SAFE ONLY because each persistent GPU worker process handles exactly
    ONE journey at a time (see worker_pool.py's run_worker_loop) — if two
    journeys could ever run concurrently on the SAME process, this
    reassignment of the process-global sys.stdout would corrupt both
    journeys' output. Do not reuse this pattern anywhere multi-threaded.
    """

    def __init__(self, job_id: str):
        self._job_id = job_id
        self._terminal_file: Optional[TextIO] = None
        self._violations_file: Optional[TextIO] = None
        self._orig_stdout: Optional[TextIO] = None
        self._orig_stderr: Optional[TextIO] = None

    def __enter__(self) -> "JobStdoutTee":
        try:
            self._terminal_file = open(get_job_log_path(self._job_id, "worker"), "a", encoding="utf-8")
            self._violations_file = open(get_job_log_path(self._job_id, "violations"), "a", encoding="utf-8")
        except Exception:
            self._terminal_file = None
            self._violations_file = None
            return self
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = _ClassifyingTeeStream(self._orig_stdout, self._terminal_file, self._violations_file)
        sys.stderr = _ClassifyingTeeStream(self._orig_stderr, self._terminal_file, self._violations_file)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Flush any trailing partial line before closing.
        try:
            sys.stdout.flush()
        except Exception:
            pass
        if self._orig_stdout is not None:
            sys.stdout = self._orig_stdout
        if self._orig_stderr is not None:
            sys.stderr = self._orig_stderr
        for f in (self._terminal_file, self._violations_file):
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass