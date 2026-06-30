from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import traceback
from functools import partial
from typing import Dict

import pika
from dotenv import load_dotenv

from analyzer import analyze_journey
from callback_client import (
    send_completed,
    send_failed,
    send_progress,
    set_base_url,
    check_job_completed,
)
from models import AnalysisJobMessage, CompletionPayload
from s3_service import download_video

# ── Config / credentials ──────────────────────────────────────────────────────
_ENV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "config", "credentials.env",
)
load_dotenv(_ENV_PATH)

RABBITMQ_URL       = os.environ.get("RABBITMQ_URL",              "amqp://guest:guest@localhost:5672/")
QUEUE_NAME         = os.environ.get("ANALYSIS_QUEUE",             "analysis.jobs")
EXCHANGE_NAME      = os.environ.get("ANALYSIS_EXCHANGE",          "analysis.exchange")
ROUTING_KEY        = os.environ.get("ANALYSIS_ROUTING",           "analysis.job.created")
PREFETCH_COUNT     = int(os.environ.get("RABBITMQ_PREFETCH",      "1"))
RECONNECT_DELAY    = int(os.environ.get("RECONNECT_DELAY_SECONDS", "5"))

# AMQP heartbeat interval negotiated with the broker (seconds).
# pika sends a frame every interval/2 automatically while start_consuming()
# is idle.  60 s keeps most NAT devices happy.
HEARTBEAT_INTERVAL = int(os.environ.get("RABBITMQ_HEARTBEAT",    "60"))

# How often the keepalive thread pokes pika during blocking work.
# Must be < HEARTBEAT_INTERVAL / 2 (i.e. < 30 s).
KEEPALIVE_INTERVAL = int(os.environ.get("RABBITMQ_KEEPALIVE",    "15"))

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger("consumer")


# ── Job-duration keepalive ────────────────────────────────────────────────────

class _JobKeepalive:
    """
    Keeps the pika connection alive for the ENTIRE duration of a job —
    covering both the download phase and the analysis phase.

    Usage (context manager):

        with _JobKeepalive(connection, pika_lock, job_id):
            # download videos   (main thread: boto3, no pika)
            # analyze_journey() (main thread: OpenCV, no pika)
            # send_completed()  (main thread: HTTP, no pika)
        # keepalive stopped here — thread.join() has returned
        _ack_and_flush(...)   # now safe to call pika

    Thread-safety
    ─────────────
    A threading.Lock (pika_lock) is shared between this thread and
    the main thread.  Only one of them holds the lock at any time:
      • Keepalive: lock → process_data_events(time_limit=1) → release
      • Main (ACK): lock → basic_ack → process_data_events → release
    During the job body (download + analysis + HTTP callbacks) the
    main thread makes NO pika calls, so the keepalive thread runs
    uncontested.  After __exit__ (thread.join()) the keepalive is
    fully done before the main thread calls basic_ack.
    """

    def __init__(self, connection: pika.BlockingConnection,
                 pika_lock: threading.Lock, job_id: str):
        self._conn   = connection
        self._lock   = pika_lock
        self._job_id = job_id
        self._stop   = threading.Event()
        self._thread = threading.Thread(
            target = self._loop,
            name   = f"Keepalive-{job_id}",
            daemon = True,
        )

    def __enter__(self):
        self._thread.start()
        log.info(
            "[Job %s]  Keepalive thread started (interval=%ds, heartbeat=%ds).",
            self._job_id, KEEPALIVE_INTERVAL, HEARTBEAT_INTERVAL,
        )
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=KEEPALIVE_INTERVAL + 5)
        if self._thread.is_alive():
            log.warning("[Job %s]  Keepalive thread did not stop cleanly.", self._job_id)
        else:
            log.info("[Job %s]  Keepalive thread stopped.", self._job_id)

    def _loop(self):
        while not self._stop.wait(timeout=KEEPALIVE_INTERVAL):
            try:
                with self._lock:
                    self._conn.process_data_events(time_limit=1)
                log.debug("[Job %s]  [Keepalive] heartbeat sent.", self._job_id)
            except Exception as exc:
                log.warning(
                    "[Job %s]  [Keepalive] process_data_events failed: %s — "
                    "connection may be lost.", self._job_id, exc,
                )
                break


# ── ACK / NACK helpers ────────────────────────────────────────────────────────

def _ack_and_flush(channel, connection, pika_lock: threading.Lock,
                   delivery_tag: int, job_id: str) -> None:
    """ACK the message. Called only after keepalive thread has stopped."""
    with pika_lock:
        channel.basic_ack(delivery_tag=delivery_tag)
        try:
            connection.process_data_events(time_limit=0)
        except Exception as flush_err:
            log.warning("[Job %s]  ACK flush warning (non-fatal): %s",
                        job_id, flush_err)
    log.info("[Job %s]  Message acknowledged and ACK flushed.", job_id)


def _nack(channel, pika_lock: threading.Lock,
          delivery_tag: int, job_id: str) -> None:
    """NACK the message. Called only after keepalive thread has stopped."""
    with pika_lock:
        try:
            channel.basic_nack(delivery_tag=delivery_tag, requeue=False)
        except Exception as exc:
            log.warning("[Job %s]  NACK failed (connection gone?): %s", job_id, exc)
    log.warning("[Job %s]  Message nacked (no requeue).", job_id)


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

    # ── Start keepalive — covers download + analysis + callback ───────────────
    # FIX: keepalive starts HERE, before downloads, because the download phase
    # can take several minutes and will starve pika's heartbeat just as badly
    # as the analysis phase.  The thread is stopped after send_completed()
    # returns, just before basic_ack().
    with _JobKeepalive(connection, pika_lock, job_id):
        job_failed = False
        job_exc    = None

        try:
            # ── Step 2: Download all videos from S3 ──────────────────────────
            log.info("[Job %s]  Downloading %d video(s)...",
                     job_id, len(msg.videos))
            send_progress(job_id, journey_id, 5, "Downloading videos")

            for vj in sorted(msg.videos, key=lambda v: v.sequence_no):
                if os.path.isfile(vj.s3_key):
                    tmp_paths[vj.video_id] = vj.s3_key
                    log.info(
                        "[Job %s]  Using local video_id=%d  seq=%d  → %s",
                        job_id, vj.video_id, vj.sequence_no, vj.s3_key,
                    )
                else:
                    suffix   = os.path.splitext(vj.s3_key)[1] or ".mp4"
                    fd, path = tempfile.mkstemp(suffix=suffix)
                    os.close(fd)
                    download_video(vj.s3_key, path)
                    tmp_paths[vj.video_id] = path
                    log.info(
                        "[Job %s]  Downloaded video_id=%d  seq=%d  → %s",
                        job_id, vj.video_id, vj.sequence_no, path,
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

            video_results, wall_seconds = analyze_journey(
                job_id            = job_id,
                journey_id        = journey_id,
                folder_name       = folder_name,
                video_jobs        = msg.videos,
                tmp_paths         = tmp_paths,
                progress_cb       = _progress,
                rabbit_connection = connection,
            )

            # ── Step 8: Calculate processing time (ms) ────────────────────────
            processing_time_ms = int(wall_seconds * 1000)
            log.info(
                "[Job %s]  Analysis complete.  videos=%d  violations=%d  time=%dms",
                job_id,
                len(video_results),
                sum(len(vr.violations) for vr in video_results),
                processing_time_ms,
            )

            # ── Step 9: Send completion callback ──────────────────────────────
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
            send_completed(completion.to_dict())
            log.info("[Job %s]  Completion callback sent.", job_id)

        except Exception as exc:
            job_failed = True
            job_exc    = exc
            err_detail = traceback.format_exc()
            log.error("[Job %s]  FAILED:\n%s", job_id, err_detail)
            try:
                send_failed(job_id, journey_id, f"{exc}\n\n{err_detail}")
            except Exception as sf_exc:
                log.warning("[Job %s]  send_failed itself failed: %s",
                            job_id, sf_exc)
        finally:
            _cleanup(tmp_paths)

    # ── Keepalive stopped here — safe to call pika ────────────────────────────
    if job_failed:
        _nack(channel, pika_lock, method.delivery_tag, job_id)
    else:
        # ── Step 10: ACK ─────────────────────────────────────────────────────
        _ack_and_flush(channel, connection, pika_lock,
                       method.delivery_tag, job_id)


def _cleanup(tmp_paths: Dict[int, str]) -> None:
    temp_dir = tempfile.gettempdir()
    for path in tmp_paths.values():
        try:
            # ONLY delete the file if it is actually inside the system's temp directory!
            if os.path.isfile(path) and path.startswith(temp_dir):
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
    try:
        data = json.loads(body)
        msg  = AnalysisJobMessage.from_dict(data)
    except Exception as exc:
        log.error("Failed to parse RabbitMQ message: %s\nBody: %s",
                  exc, body[:500])
        with pika_lock:
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        return

    _handle_job(msg, channel, method, connection, pika_lock)


# ── Consumer connect + consume ────────────────────────────────────────────────

def _connect_and_consume() -> None:
    log.info("[Consumer]  Connecting to RabbitMQ: %s  queue=%s",
             RABBITMQ_URL, QUEUE_NAME)
    params = pika.URLParameters(RABBITMQ_URL)
    params.heartbeat                  = HEARTBEAT_INTERVAL   # 60 s
    params.blocked_connection_timeout = 300
    params.socket_timeout             = 300
    connection = pika.BlockingConnection(params)
    channel    = connection.channel()

    # One lock per connection — shared by _handle_job and _JobKeepalive.
    pika_lock = threading.Lock()

    channel.exchange_declare(
        exchange      = EXCHANGE_NAME,
        exchange_type = "direct",
        durable       = True,
    )
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


if __name__ == "__main__":
    start()