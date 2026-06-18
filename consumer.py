# # # # # # # """
# # # # # # # consumer.py
# # # # # # # ───────────
# # # # # # # RabbitMQ consumer for the Journey-based analysis workflow.

# # # # # # # Processing flow (Step numbers match the spec)
# # # # # # # ─────────────────────────────────────────────
# # # # # # #  1. Consume message from 'analysis.jobs'.
# # # # # # #  2. Download all videos from S3.
# # # # # # #  3. Send PROCESSING progress (10 %).
# # # # # # #  4. Analyze videos via the existing AI pipeline (analyzer.py).
# # # # # # #  5. Violation frames are uploaded inside analyzer.py (S3 keys returned).
# # # # # # #  6-7. VideoResult / ViolationResult objects built inside analyzer.py.
# # # # # # #  8. Calculate overall processing time (wall-clock ms).
# # # # # # #  9. Send completion callback.
# # # # # # # 10. Acknowledge RabbitMQ message ONLY after successful completion callback.

# # # # # # # On any unhandled exception:
# # # # # # #     • Log the full traceback.
# # # # # # #     • Send a failure callback to Spring Boot.
# # # # # # #     • Nack the message (do not requeue — leave it for the dead-letter exchange).
# # # # # # # """

# # # # # # # from __future__ import annotations

# # # # # # # import json
# # # # # # # import logging
# # # # # # # import os
# # # # # # # import tempfile
# # # # # # # import traceback
# # # # # # # import time
# # # # # # # from typing import Dict, List

# # # # # # # import pika
# # # # # # # from dotenv import load_dotenv

# # # # # # # from analyzer import analyze_journey
# # # # # # # from callback_client import send_completed, send_failed, send_progress
# # # # # # # from models import AnalysisJobMessage, CompletionPayload, VideoJob
# # # # # # # from s3_service import download_video

# # # # # # # # ── Config / credentials ─────────────────────────────────────────────────────
# # # # # # # _ENV_PATH = os.path.join(
# # # # # # #     os.path.dirname(os.path.abspath(__file__)),
# # # # # # #     "config", "credentials.env",
# # # # # # # )
# # # # # # # load_dotenv(_ENV_PATH)

# # # # # # # RABBITMQ_URL   = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
# # # # # # # QUEUE_NAME     = os.environ.get("ANALYSIS_QUEUE", "analysis.jobs")
# # # # # # # PREFETCH_COUNT = int(os.environ.get("RABBITMQ_PREFETCH", "1"))

# # # # # # # logging.basicConfig(
# # # # # # #     level   = logging.INFO,
# # # # # # #     format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
# # # # # # # )
# # # # # # # log = logging.getLogger("consumer")


# # # # # # # # ── Job handler ───────────────────────────────────────────────────────────────

# # # # # # # def _handle_job(msg: AnalysisJobMessage, channel, method) -> None:
# # # # # # #     """
# # # # # # #     Full processing flow for one AnalysisJobMessage.
# # # # # # #     Acks on success, nacks (no requeue) on failure.
# # # # # # #     """
# # # # # # #     job_id     = msg.job_id
# # # # # # #     journey_id = msg.journey_id
# # # # # # #     tmp_paths: Dict[int, str] = {}

# # # # # # #     log.info("[Job %s]  journey=%d  videos=%d", job_id, journey_id, len(msg.videos))

# # # # # # #     try:
# # # # # # #         # ── Step 2: Download all videos from S3 ──────────────────────────────
# # # # # # #         log.info("[Job %s]  Downloading %d video(s)…", job_id, len(msg.videos))
# # # # # # #         send_progress(job_id, journey_id, 5, "Downloading videos")

# # # # # # #         for vj in sorted(msg.videos, key=lambda v: v.sequence_no):
# # # # # # #             suffix   = os.path.splitext(vj.s3_key)[1] or ".mp4"
# # # # # # #             fd, path = tempfile.mkstemp(suffix=suffix)
# # # # # # #             os.close(fd)
# # # # # # #             download_video(vj.s3_key, path)
# # # # # # #             tmp_paths[vj.video_id] = path
# # # # # # #             log.info("[Job %s]  Downloaded video_id=%d  seq=%d  → %s",
# # # # # # #                      job_id, vj.video_id, vj.sequence_no, path)

# # # # # # #         # ── Step 3: Progress callback after download ──────────────────────────
# # # # # # #         send_progress(job_id, journey_id, 10, "Downloading videos complete — starting analysis")

# # # # # # #         # ── Steps 4-7: Analyze + upload frames + build results ────────────────
# # # # # # #         def _progress(pct: int, message: str) -> None:
# # # # # # #             try:
# # # # # # #                 send_progress(job_id, journey_id, pct, message)
# # # # # # #             except Exception as exc:
# # # # # # #                 log.warning("[Job %s]  progress callback failed (non-fatal): %s", job_id, exc)

# # # # # # #         video_results, wall_seconds = analyze_journey(
# # # # # # #             job_id     = job_id,
# # # # # # #             journey_id = journey_id,
# # # # # # #             video_jobs = msg.videos,
# # # # # # #             tmp_paths  = tmp_paths,
# # # # # # #             progress_cb = _progress,
# # # # # # #         )

# # # # # # #         # ── Step 8: Calculate processing time (ms) ────────────────────────────
# # # # # # #         processing_time_ms = int(wall_seconds * 1000)
# # # # # # #         log.info("[Job %s]  Analysis complete.  videos=%d  violations=%d  time=%dms",
# # # # # # #                  job_id,
# # # # # # #                  len(video_results),
# # # # # # #                  sum(len(vr.violations) for vr in video_results),
# # # # # # #                  processing_time_ms)

# # # # # # #         # ── Step 9: Send completion callback ─────────────────────────────────
# # # # # # #         send_progress(job_id, journey_id, 95, "Sending results to backend")

# # # # # # #         completion = CompletionPayload(
# # # # # # #             job_id          = job_id,
# # # # # # #             journey_id      = journey_id,
# # # # # # #             processing_time = processing_time_ms,
# # # # # # #             video_results   = video_results,
# # # # # # #         )
# # # # # # #         send_completed(completion.to_dict())
# # # # # # #         log.info("[Job %s]  Completion callback sent.", job_id)

# # # # # # #         # ── Step 10: Acknowledge RabbitMQ message ─────────────────────────────
# # # # # # #         channel.basic_ack(delivery_tag=method.delivery_tag)
# # # # # # #         log.info("[Job %s]  Message acknowledged.", job_id)

# # # # # # #     except Exception as exc:
# # # # # # #         # ── Error handling ────────────────────────────────────────────────────
# # # # # # #         err_detail = traceback.format_exc()
# # # # # # #         log.error("[Job %s]  FAILED:\n%s", job_id, err_detail)

# # # # # # #         send_failed(job_id, journey_id, f"{exc}\n\n{err_detail}")

# # # # # # #         # Nack without requeue — failed jobs go to the dead-letter exchange.
# # # # # # #         # Change requeue=True if you want automatic retry.
# # # # # # #         channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
# # # # # # #         log.warning("[Job %s]  Message nacked (no requeue).", job_id)

# # # # # # #     finally:
# # # # # # #         # Always clean up temp files
# # # # # # #         _cleanup(tmp_paths)


# # # # # # # def _cleanup(tmp_paths: Dict[int, str]) -> None:
# # # # # # #     for path in tmp_paths.values():
# # # # # # #         try:
# # # # # # #             if os.path.isfile(path):
# # # # # # #                 os.remove(path)
# # # # # # #         except OSError as exc:
# # # # # # #             log.warning("Could not remove temp file %s: %s", path, exc)


# # # # # # # # ── RabbitMQ callback ─────────────────────────────────────────────────────────

# # # # # # # def _on_message(channel, method, properties, body):
# # # # # # #     try:
# # # # # # #         data = json.loads(body)
# # # # # # #         msg  = AnalysisJobMessage.from_dict(data)
# # # # # # #     except Exception as exc:
# # # # # # #         log.error("Failed to parse RabbitMQ message: %s\nBody: %s", exc, body[:500])
# # # # # # #         # Malformed messages are nacked without requeue
# # # # # # #         channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
# # # # # # #         return

# # # # # # #     _handle_job(msg, channel, method)


# # # # # # # # ── Consumer startup ──────────────────────────────────────────────────────────

# # # # # # # def start() -> None:
# # # # # # #     """
# # # # # # #     Connect to RabbitMQ and start consuming from the analysis.jobs queue.
# # # # # # #     Blocks forever (until interrupted).
# # # # # # #     """
# # # # # # #     log.info("[Consumer]  Connecting to RabbitMQ:  %s", RABBITMQ_URL)
# # # # # # #     params     = pika.URLParameters(RABBITMQ_URL)
# # # # # # #     connection = pika.BlockingConnection(params)
# # # # # # #     channel    = connection.channel()

# # # # # # #     channel.queue_declare(queue=QUEUE_NAME, durable=True)
# # # # # # #     channel.basic_qos(prefetch_count=PREFETCH_COUNT)
# # # # # # #     channel.basic_consume(queue=QUEUE_NAME, on_message_callback=_on_message)

# # # # # # #     log.info("[Consumer]  Waiting for messages on queue '%s'…", QUEUE_NAME)
# # # # # # #     try:
# # # # # # #         channel.start_consuming()
# # # # # # #     except KeyboardInterrupt:
# # # # # # #         log.info("[Consumer]  Interrupted — shutting down.")
# # # # # # #         channel.stop_consuming()
# # # # # # #     finally:
# # # # # # #         connection.close()


# # # # # # # if __name__ == "__main__":
# # # # # # #     start()
# # # # # # """
# # # # # # consumer.py
# # # # # # ───────────
# # # # # # RabbitMQ consumer for the Journey-based analysis workflow.

# # # # # # Processing flow (Step numbers match the spec)
# # # # # # ─────────────────────────────────────────────
# # # # # #  1. Consume message from 'analysis.jobs'.
# # # # # #  2. Download all videos from S3.
# # # # # #  3. Send PROCESSING progress (10 %).
# # # # # #  4. Analyze videos via the existing AI pipeline (analyzer.py).
# # # # # #  5. Violation frames are uploaded inside analyzer.py (S3 keys returned).
# # # # # #  6-7. VideoResult / ViolationResult objects built inside analyzer.py.
# # # # # #  8. Calculate overall processing time (wall-clock ms).
# # # # # #  9. Send completion callback.
# # # # # # 10. Acknowledge RabbitMQ message ONLY after successful completion callback.

# # # # # # On any unhandled exception:
# # # # # #     • Log the full traceback.
# # # # # #     • Send a failure callback to Spring Boot.
# # # # # #     • Nack the message (do not requeue — leave it for the dead-letter exchange).
# # # # # # """

# # # # # # from __future__ import annotations

# # # # # # import json
# # # # # # import logging
# # # # # # import os
# # # # # # import tempfile
# # # # # # import traceback
# # # # # # import time
# # # # # # from typing import Dict, List

# # # # # # import pika
# # # # # # from dotenv import load_dotenv

# # # # # # from analyzer import analyze_journey
# # # # # # from callback_client import send_completed, send_failed, send_progress
# # # # # # from models import AnalysisJobMessage, CompletionPayload, VideoJob
# # # # # # from s3_service import download_video

# # # # # # # ── Config / credentials ─────────────────────────────────────────────────────
# # # # # # _ENV_PATH = os.path.join(
# # # # # #     os.path.dirname(os.path.abspath(__file__)),
# # # # # #     "config", "credentials.env",
# # # # # # )
# # # # # # load_dotenv(_ENV_PATH)

# # # # # # RABBITMQ_URL   = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
# # # # # # QUEUE_NAME     = os.environ.get("ANALYSIS_QUEUE", "analysis.jobs")
# # # # # # PREFETCH_COUNT = int(os.environ.get("RABBITMQ_PREFETCH", "1"))

# # # # # # logging.basicConfig(
# # # # # #     level   = logging.INFO,
# # # # # #     format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
# # # # # # )
# # # # # # log = logging.getLogger("consumer")


# # # # # # # ── Job handler ───────────────────────────────────────────────────────────────

# # # # # # def _handle_job(msg: AnalysisJobMessage, channel, method) -> None:
# # # # # #     """
# # # # # #     Full processing flow for one AnalysisJobMessage.
# # # # # #     Acks on success, nacks (no requeue) on failure.
# # # # # #     """
# # # # # #     job_id     = msg.job_id
# # # # # #     journey_id = msg.journey_id
# # # # # #     tmp_paths: Dict[int, str] = {}

# # # # # #     log.info("[Job %s]  journey=%d  videos=%d", job_id, journey_id, len(msg.videos))

# # # # # #     try:
# # # # # #         # ── Step 2: Download all videos from S3 ──────────────────────────────
# # # # # #         log.info("[Job %s]  Downloading %d video(s)…", job_id, len(msg.videos))
# # # # # #         send_progress(job_id, journey_id, 5, "Downloading videos")

# # # # # #         for vj in sorted(msg.videos, key=lambda v: v.sequence_no):
# # # # # #             suffix   = os.path.splitext(vj.s3_key)[1] or ".mp4"
# # # # # #             fd, path = tempfile.mkstemp(suffix=suffix)
# # # # # #             os.close(fd)
# # # # # #             download_video(vj.s3_key, path)
# # # # # #             tmp_paths[vj.video_id] = path
# # # # # #             log.info("[Job %s]  Downloaded video_id=%d  seq=%d  → %s",
# # # # # #                      job_id, vj.video_id, vj.sequence_no, path)

# # # # # #         # ── Step 3: Progress callback after download ──────────────────────────
# # # # # #         send_progress(job_id, journey_id, 10, "Downloading videos complete — starting analysis")

# # # # # #         # ── Steps 4-7: Analyze + upload frames + build results ────────────────
# # # # # #         def _progress(pct: int, message: str) -> None:
# # # # # #             try:
# # # # # #                 send_progress(job_id, journey_id, pct, message)
# # # # # #             except Exception as exc:
# # # # # #                 log.warning("[Job %s]  progress callback failed (non-fatal): %s", job_id, exc)

# # # # # #         video_results, wall_seconds = analyze_journey(
# # # # # #             job_id     = job_id,
# # # # # #             journey_id = journey_id,
# # # # # #             video_jobs = msg.videos,
# # # # # #             tmp_paths  = tmp_paths,
# # # # # #             progress_cb = _progress,
# # # # # #         )

# # # # # #         # ── Step 8: Calculate processing time (ms) ────────────────────────────
# # # # # #         processing_time_ms = int(wall_seconds * 1000)
# # # # # #         log.info("[Job %s]  Analysis complete.  videos=%d  violations=%d  time=%dms",
# # # # # #                  job_id,
# # # # # #                  len(video_results),
# # # # # #                  sum(len(vr.violations) for vr in video_results),
# # # # # #                  processing_time_ms)

# # # # # #         # ── Step 9: Send completion callback ─────────────────────────────────
# # # # # #         send_progress(job_id, journey_id, 95, "Sending results to backend")

# # # # # #         completion = CompletionPayload(
# # # # # #             job_id          = job_id,
# # # # # #             journey_id      = journey_id,
# # # # # #             processing_time = processing_time_ms,
# # # # # #             video_results   = video_results,
# # # # # #         )
# # # # # #         send_completed(completion.to_dict())
# # # # # #         log.info("[Job %s]  Completion callback sent.", job_id)

# # # # # #         # ── Step 10: Acknowledge RabbitMQ message ─────────────────────────────
# # # # # #         channel.basic_ack(delivery_tag=method.delivery_tag)
# # # # # #         log.info("[Job %s]  Message acknowledged.", job_id)

# # # # # #     except Exception as exc:
# # # # # #         # ── Error handling ────────────────────────────────────────────────────
# # # # # #         err_detail = traceback.format_exc()
# # # # # #         log.error("[Job %s]  FAILED:\n%s", job_id, err_detail)

# # # # # #         send_failed(job_id, journey_id, f"{exc}\n\n{err_detail}")

# # # # # #         # Nack without requeue — failed jobs go to the dead-letter exchange.
# # # # # #         # Change requeue=True if you want automatic retry.
# # # # # #         channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
# # # # # #         log.warning("[Job %s]  Message nacked (no requeue).", job_id)

# # # # # #     finally:
# # # # # #         # Always clean up temp files
# # # # # #         _cleanup(tmp_paths)


# # # # # # def _cleanup(tmp_paths: Dict[int, str]) -> None:
# # # # # #     for path in tmp_paths.values():
# # # # # #         try:
# # # # # #             if os.path.isfile(path):
# # # # # #                 os.remove(path)
# # # # # #         except OSError as exc:
# # # # # #             log.warning("Could not remove temp file %s: %s", path, exc)


# # # # # # # ── RabbitMQ callback ─────────────────────────────────────────────────────────

# # # # # # def _on_message(channel, method, properties, body):
# # # # # #     log.info("[RabbitMQ] Raw message received: %s", body.decode("utf-8", errors="replace")[:1000])
# # # # # #     try:
# # # # # #         data = json.loads(body)
# # # # # #         msg  = AnalysisJobMessage.from_dict(data)
# # # # # #     except Exception as exc:
# # # # # #         log.error("Failed to parse RabbitMQ message: %s\nBody: %s", exc, body[:500])
# # # # # #         # Malformed messages are nacked without requeue
# # # # # #         channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
# # # # # #         return

# # # # # #     _handle_job(msg, channel, method)


# # # # # # # ── Consumer startup ──────────────────────────────────────────────────────────

# # # # # # def start() -> None:
# # # # # #     """
# # # # # #     Connect to RabbitMQ and start consuming from the analysis.jobs queue.
# # # # # #     Blocks forever (until interrupted).
# # # # # #     """
# # # # # #     log.info("[Consumer]  Connecting to RabbitMQ:  %s", RABBITMQ_URL)
# # # # # #     params     = pika.URLParameters(RABBITMQ_URL)
# # # # # #     connection = pika.BlockingConnection(params)
# # # # # #     channel    = connection.channel()

# # # # # #     channel.queue_declare(queue=QUEUE_NAME, durable=True)
# # # # # #     channel.basic_qos(prefetch_count=PREFETCH_COUNT)
# # # # # #     channel.basic_consume(queue=QUEUE_NAME, on_message_callback=_on_message)

# # # # # #     log.info("[Consumer]  Waiting for messages on queue '%s'…", QUEUE_NAME)
# # # # # #     try:
# # # # # #         channel.start_consuming()
# # # # # #     except KeyboardInterrupt:
# # # # # #         log.info("[Consumer]  Interrupted — shutting down.")
# # # # # #         channel.stop_consuming()
# # # # # #     finally:
# # # # # #         connection.close()


# # # # # # if __name__ == "__main__":
# # # # # #     start()


# # # # # """
# # # # # consumer.py
# # # # # ───────────
# # # # # RabbitMQ consumer for the Journey-based analysis workflow.

# # # # # Live backend : https://cvvrsrailway-api.sconexsoft.com/cvs

# # # # # Processing flow (Step numbers match the CVVRS spec)
# # # # # ────────────────────────────────────────────────────
# # # # #  1. Consume message from 'analysis.jobs' (exchange: analysis.exchange,
# # # # #     routing key: analysis.job.created).
# # # # #  2. Download all videos from S3.
# # # # #  3. Send PROCESSING progress (5 %) — "Downloading videos".
# # # # #  4. Analyze videos via the existing AI pipeline (analyzer.py).
# # # # #  5. Violation frames are uploaded inside analyzer.py (S3 keys returned).
# # # # #  6–7. VideoResult / ViolationResult objects built inside analyzer.py.
# # # # #  8. Calculate overall processing time (wall-clock ms).
# # # # #  9. Send completion callback (POST /api/internal/analysis/completed).
# # # # # 10. Acknowledge RabbitMQ message ONLY after successful completion callback.

# # # # # On any unhandled exception
# # # # # ──────────────────────────
# # # # #   • Log the full traceback.
# # # # #   • Send a failure callback (POST /api/internal/analysis/failed).
# # # # #   • Nack the message (no requeue — dead-letter exchange).

# # # # # Changes from previous version
# # # # # ──────────────────────────────
# # # # # • AnalysisJobMessage now carries trainDetailId, folderName, priority —
# # # # #   all forwarded to CompletionPayload.
# # # # # • analyze_journey() receives folder_name so frames land in the right S3 path.
# # # # # • _progress() passes current_video index to send_progress().
# # # # # • CompletionPayload is built with trainDetailId and folderName.
# # # # # • Queue binding is declared with the exchange / routing key from the docs.
# # # # # """

# # # # # from __future__ import annotations

# # # # # import json
# # # # # import logging
# # # # # import os
# # # # # import tempfile
# # # # # import traceback
# # # # # import time
# # # # # from typing import Dict

# # # # # import pika
# # # # # from dotenv import load_dotenv

# # # # # from analyzer import analyze_journey
# # # # # from callback_client import send_completed, send_failed, send_progress
# # # # # from models import AnalysisJobMessage, CompletionPayload
# # # # # from s3_service import download_video

# # # # # # ── Config / credentials ─────────────────────────────────────────────────────
# # # # # _ENV_PATH = os.path.join(
# # # # #     os.path.dirname(os.path.abspath(__file__)),
# # # # #     "config", "credentials.env",
# # # # # )
# # # # # load_dotenv(_ENV_PATH)

# # # # # RABBITMQ_URL   = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
# # # # # QUEUE_NAME     = os.environ.get("ANALYSIS_QUEUE",    "analysis.jobs")
# # # # # EXCHANGE_NAME  = os.environ.get("ANALYSIS_EXCHANGE", "analysis.exchange")
# # # # # ROUTING_KEY    = os.environ.get("ANALYSIS_ROUTING",  "analysis.job.created")
# # # # # PREFETCH_COUNT = int(os.environ.get("RABBITMQ_PREFETCH", "1"))

# # # # # logging.basicConfig(
# # # # #     level  = logging.INFO,
# # # # #     format = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
# # # # # )
# # # # # log = logging.getLogger("consumer")


# # # # # # ── Job handler ───────────────────────────────────────────────────────────────

# # # # # def _handle_job(msg: AnalysisJobMessage, channel, method) -> None:
# # # # #     """
# # # # #     Full processing flow for one AnalysisJobMessage.
# # # # #     Acks on success, nacks (no requeue) on failure.
# # # # #     """
# # # # #     job_id          = msg.job_id
# # # # #     journey_id      = msg.journey_id
# # # # #     train_detail_id = msg.train_detail_id
# # # # #     folder_name     = msg.folder_name
# # # # #     tmp_paths: Dict[int, str] = {}

# # # # #     log.info(
# # # # #         "[Job %s]  journey=%d  trainDetail=%d  videos=%d  folder=%s",
# # # # #         job_id, journey_id, train_detail_id, len(msg.videos), folder_name,
# # # # #     )

# # # # #     try:
# # # # #         # ── Step 2: Download all videos from S3 ──────────────────────────────
# # # # #         log.info("[Job %s]  Downloading %d video(s)…", job_id, len(msg.videos))
# # # # #         send_progress(job_id, journey_id, 5, "Downloading videos")

# # # # #         for vj in sorted(msg.videos, key=lambda v: v.sequence_no):
# # # # #             suffix   = os.path.splitext(vj.s3_key)[1] or ".mp4"
# # # # #             fd, path = tempfile.mkstemp(suffix=suffix)
# # # # #             os.close(fd)
# # # # #             download_video(vj.s3_key, path)
# # # # #             tmp_paths[vj.video_id] = path
# # # # #             log.info(
# # # # #                 "[Job %s]  Downloaded video_id=%d  seq=%d  → %s",
# # # # #                 job_id, vj.video_id, vj.sequence_no, path,
# # # # #             )

# # # # #         # ── Step 3: Progress callback after download ──────────────────────────
# # # # #         send_progress(
# # # # #             job_id, journey_id, 10,
# # # # #             "Downloads complete — starting analysis",
# # # # #             current_video=1,
# # # # #         )

# # # # #         # ── Steps 4–7: Analyze + upload frames + build results ────────────────
# # # # #         def _progress(pct: int, message: str, current_video: int = 1) -> None:
# # # # #             try:
# # # # #                 send_progress(job_id, journey_id, pct, message,
# # # # #                               current_video=current_video)
# # # # #             except Exception as exc:
# # # # #                 log.warning(
# # # # #                     "[Job %s]  progress callback failed (non-fatal): %s", job_id, exc
# # # # #                 )

# # # # #         video_results, wall_seconds = analyze_journey(
# # # # #             job_id      = job_id,
# # # # #             journey_id  = journey_id,
# # # # #             folder_name = folder_name,
# # # # #             video_jobs  = msg.videos,
# # # # #             tmp_paths   = tmp_paths,
# # # # #             progress_cb = _progress,
# # # # #         )

# # # # #         # ── Step 8: Calculate processing time (ms) ────────────────────────────
# # # # #         processing_time_ms = int(wall_seconds * 1000)
# # # # #         log.info(
# # # # #             "[Job %s]  Analysis complete.  videos=%d  violations=%d  time=%dms",
# # # # #             job_id,
# # # # #             len(video_results),
# # # # #             sum(len(vr.violations) for vr in video_results),
# # # # #             processing_time_ms,
# # # # #         )

# # # # #         # ── Step 9: Send completion callback ──────────────────────────────────
# # # # #         send_progress(
# # # # #             job_id, journey_id, 95,
# # # # #             "Sending results to backend",
# # # # #             current_video=len(msg.videos),
# # # # #         )

# # # # #         completion = CompletionPayload(
# # # # #             job_id          = job_id,
# # # # #             journey_id      = journey_id,
# # # # #             train_detail_id = train_detail_id,   # NEW
# # # # #             folder_name     = folder_name,        # NEW
# # # # #             processing_time = processing_time_ms,
# # # # #             video_results   = video_results,
# # # # #         )
# # # # #         send_completed(completion.to_dict())
# # # # #         log.info("[Job %s]  Completion callback sent.", job_id)

# # # # #         # ── Step 10: Acknowledge RabbitMQ message ─────────────────────────────
# # # # #         channel.basic_ack(delivery_tag=method.delivery_tag)
# # # # #         log.info("[Job %s]  Message acknowledged.", job_id)

# # # # #     except Exception as exc:
# # # # #         err_detail = traceback.format_exc()
# # # # #         log.error("[Job %s]  FAILED:\n%s", job_id, err_detail)

# # # # #         # POST /api/internal/analysis/failed — only jobId + errorMessage
# # # # #         send_failed(job_id, journey_id, f"{exc}\n\n{err_detail}")

# # # # #         # Nack without requeue — failed jobs go to dead-letter exchange.
# # # # #         channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
# # # # #         log.warning("[Job %s]  Message nacked (no requeue).", job_id)

# # # # #     finally:
# # # # #         _cleanup(tmp_paths)


# # # # # def _cleanup(tmp_paths: Dict[int, str]) -> None:
# # # # #     for path in tmp_paths.values():
# # # # #         try:
# # # # #             if os.path.isfile(path):
# # # # #                 os.remove(path)
# # # # #         except OSError as exc:
# # # # #             log.warning("Could not remove temp file %s: %s", path, exc)


# # # # # # ── RabbitMQ callback ─────────────────────────────────────────────────────────

# # # # # def _on_message(channel, method, properties, body):
# # # # #     log.info(
# # # # #         "[RabbitMQ] Message received: %s",
# # # # #         body.decode("utf-8", errors="replace")[:500],
# # # # #     )
# # # # #     try:
# # # # #         data = json.loads(body)
# # # # #         msg  = AnalysisJobMessage.from_dict(data)
# # # # #     except Exception as exc:
# # # # #         log.error("Failed to parse RabbitMQ message: %s\nBody: %s", exc, body[:500])
# # # # #         channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
# # # # #         return

# # # # #     _handle_job(msg, channel, method)


# # # # # # ── Consumer startup ──────────────────────────────────────────────────────────

# # # # # def start() -> None:
# # # # #     """
# # # # #     Connect to RabbitMQ and start consuming from the analysis.jobs queue.
# # # # #     Declares the exchange and binding so the worker can start independently
# # # # #     of the Spring Boot startup order.
# # # # #     Blocks forever (until interrupted).
# # # # #     """
# # # # #     log.info("[Consumer]  Connecting to RabbitMQ: %s", RABBITMQ_URL)
# # # # #     params     = pika.URLParameters(RABBITMQ_URL)
# # # # #     connection = pika.BlockingConnection(params)
# # # # #     channel    = connection.channel()

# # # # #     # Declare exchange + queue + binding (idempotent — safe to call on restart)
# # # # #     channel.exchange_declare(
# # # # #         exchange      = EXCHANGE_NAME,
# # # # #         exchange_type = "direct",
# # # # #         durable       = True,
# # # # #     )
# # # # #     channel.queue_declare(queue=QUEUE_NAME, durable=True)
# # # # #     channel.queue_bind(
# # # # #         queue       = QUEUE_NAME,
# # # # #         exchange    = EXCHANGE_NAME,
# # # # #         routing_key = ROUTING_KEY,
# # # # #     )

# # # # #     channel.basic_qos(prefetch_count=PREFETCH_COUNT)
# # # # #     channel.basic_consume(queue=QUEUE_NAME, on_message_callback=_on_message)

# # # # #     log.info(
# # # # #         "[Consumer]  Waiting for messages on queue '%s' "
# # # # #         "(exchange='%s', routing='%s')…",
# # # # #         QUEUE_NAME, EXCHANGE_NAME, ROUTING_KEY,
# # # # #     )
# # # # #     try:
# # # # #         channel.start_consuming()
# # # # #     except KeyboardInterrupt:
# # # # #         log.info("[Consumer]  Interrupted — shutting down.")
# # # # #         channel.stop_consuming()
# # # # #     finally:
# # # # #         connection.close()


# # # # # if __name__ == "__main__":
# # # # #     start()



# # # # """
# # # # consumer.py
# # # # ───────────
# # # # RabbitMQ consumer for the Journey-based analysis workflow.
 
# # # # Live backend : https://cvvrsrailway-api.sconexsoft.com/cvs
 
# # # # Processing flow (Step numbers match the CVVRS spec)
# # # # ────────────────────────────────────────────────────
# # # # 1. Consume message from 'analysis.jobs' (exchange: analysis.exchange,
# # # #     routing key: analysis.job.created).
# # # # 2. Download all videos from S3.
# # # # 3. Send PROCESSING progress (5%) — "Downloading videos".
# # # # 4. Analyze videos via the existing AI pipeline (analyzer.py).
# # # # 5. Violation frames are uploaded inside analyzer.py (S3 keys returned).
# # # # 6–7. VideoResult / ViolationResult objects built inside analyzer.py.
# # # # 8. Calculate overall processing time (wall-clock ms).
# # # # 9. Send completion callback (POST /api/internal/analysis/completed).
# # # # 10. Acknowledge RabbitMQ message ONLY after successful completion callback.
 
# # # # On any unhandled exception
# # # # ──────────────────────────
# # # #   • Log the full traceback.
# # # #   • Send a failure callback (POST /api/internal/analysis/failed).
# # # #   • Nack the message (no requeue — dead-letter exchange).
 
# # # # Fixes from previous version
# # # # ─────────────────────────────
# # # # • FIX: Reads callbackBaseUrl from RabbitMQ message and overrides the
# # # #   callback client base URL dynamically — so Python calls the correct
# # # #   Spring Boot server without needing env var changes per environment.
# # # # • FIX: Added reconnect retry loop — if RabbitMQ restarts while Python
# # # #   is running, the consumer reconnects automatically after 5 seconds
# # # #   instead of crashing.
# # # # • IMPORTANT: Make sure config/credentials.env has:
# # # #       ANALYSIS_QUEUE=analysis.jobs
# # # #   NOT analysis.queue (that was the old queue name causing the mismatch).
# # # # """
 
# # # # from __future__ import annotations
 
# # # # import json
# # # # import logging
# # # # import os
# # # # import tempfile
# # # # import time
# # # # import traceback
# # # # from typing import Dict
# # # # from functools import partial
# # # # import pika
# # # # from dotenv import load_dotenv
 
# # # # from analyzer import analyze_journey
# # # # from callback_client import send_completed, send_failed, send_progress, set_base_url
# # # # from models import AnalysisJobMessage, CompletionPayload
# # # # from s3_service import download_video
 
# # # # # ── Config / credentials ─────────────────────────────────────────────────────
# # # # _ENV_PATH = os.path.join(
# # # #     os.path.dirname(os.path.abspath(__file__)),
# # # #     "config", "credentials.env",
# # # # )
# # # # load_dotenv(_ENV_PATH)
 
# # # # RABBITMQ_URL   = os.environ.get("RABBITMQ_URL",        "amqp://guest:guest@localhost:5672/")
# # # # QUEUE_NAME     = os.environ.get("ANALYSIS_QUEUE",       "analysis.jobs")   # must be analysis.jobs
# # # # EXCHANGE_NAME  = os.environ.get("ANALYSIS_EXCHANGE",    "analysis.exchange")
# # # # ROUTING_KEY    = os.environ.get("ANALYSIS_ROUTING",     "analysis.job.created")
# # # # PREFETCH_COUNT = int(os.environ.get("RABBITMQ_PREFETCH", "1"))
# # # # RECONNECT_DELAY = int(os.environ.get("RECONNECT_DELAY_SECONDS", "5"))
 
# # # # logging.basicConfig(
# # # #     level  = logging.INFO,
# # # #     format = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
# # # # )
# # # # log = logging.getLogger("consumer")
 
 
# # # # # ── Job handler ───────────────────────────────────────────────────────────────
 
# # # # def _handle_job(msg: AnalysisJobMessage, channel, method,connection) -> None:
# # # #     """
# # # #     Full processing flow for one AnalysisJobMessage.
# # # #     Acks on success, nacks (no requeue) on failure.
# # # #     """
# # # #     job_id          = msg.job_id
# # # #     journey_id      = msg.journey_id
# # # #     train_detail_id = msg.train_detail_id
# # # #     folder_name     = msg.folder_name
# # # #     tmp_paths: Dict[int, str] = {}
 
# # # #     log.info(
# # # #         "[Job %s]  journey=%d  trainDetail=%d  videos=%d  folder=%s",
# # # #         job_id, journey_id, train_detail_id, len(msg.videos), folder_name,
# # # #     )
 
# # # #     # FIX: Override callback URL from the message so Python calls the
# # # #     # correct Spring Boot server regardless of env var settings.
# # # #     if msg.callback_base_url:
# # # #         log.info("[Job %s]  callbackBaseUrl=%s", job_id, msg.callback_base_url)
# # # #         set_base_url(msg.callback_base_url)
 
# # # #     try:
# # # #         # ── Step 2: Download all videos from S3 ──────────────────────────────
# # # #         log.info("[Job %s]  Downloading %d video(s)...", job_id, len(msg.videos))
# # # #         send_progress(job_id, journey_id, 5, "Downloading videos")
 
# # # #         for vj in sorted(msg.videos, key=lambda v: v.sequence_no):
# # # #             suffix   = os.path.splitext(vj.s3_key)[1] or ".mp4"
# # # #             fd, path = tempfile.mkstemp(suffix=suffix)
# # # #             os.close(fd)
# # # #             download_video(vj.s3_key, path)
# # # #             tmp_paths[vj.video_id] = path
# # # #             log.info(
# # # #                 "[Job %s]  Downloaded video_id=%d  seq=%d  → %s",
# # # #                 job_id, vj.video_id, vj.sequence_no, path,
# # # #             )
 
# # # #         # ── Step 3: Progress callback after download ──────────────────────────
# # # #         send_progress(
# # # #             job_id, journey_id, 10,
# # # #             "Downloads complete — starting analysis",
# # # #             current_video=1,
# # # #         )
 
# # # #         # ── Steps 4–7: Analyze + upload frames + build results ────────────────
# # # #         def _progress(pct: int, message: str, current_video: int = 1) -> None:
# # # #             try:
# # # #                 send_progress(job_id, journey_id, pct, message,
# # # #                               current_video=current_video)
# # # #             except Exception as exc:
# # # #                 log.warning(
# # # #                     "[Job %s]  progress callback failed (non-fatal): %s", job_id, exc
# # # #                 )
 
# # # #         video_results, wall_seconds = analyze_journey(
# # # #             job_id      = job_id,
# # # #             journey_id  = journey_id,
# # # #             folder_name = folder_name,
# # # #             video_jobs  = msg.videos,
# # # #             tmp_paths   = tmp_paths,
# # # #             progress_cb = _progress,
# # # #             rabbit_connection=connection,
# # # #         )
 
# # # #         # ── Step 8: Calculate processing time (ms) ────────────────────────────
# # # #         processing_time_ms = int(wall_seconds * 1000)
# # # #         log.info(
# # # #             "[Job %s]  Analysis complete.  videos=%d  violations=%d  time=%dms",
# # # #             job_id,
# # # #             len(video_results),
# # # #             sum(len(vr.violations) for vr in video_results),
# # # #             processing_time_ms,
# # # #         )
 
# # # #         # ── Step 9: Send completion callback ──────────────────────────────────
# # # #         send_progress(
# # # #             job_id, journey_id, 95,
# # # #             "Sending results to backend",
# # # #             current_video=len(msg.videos),
# # # #         )
 
# # # #         completion = CompletionPayload(
# # # #             job_id          = job_id,
# # # #             journey_id      = journey_id,
# # # #             train_detail_id = train_detail_id,
# # # #             folder_name     = folder_name,
# # # #             processing_time = processing_time_ms,
# # # #             video_results   = video_results,
# # # #         )
# # # #         send_completed(completion.to_dict())
# # # #         log.info("[Job %s]  Completion callback sent.", job_id)
 
# # # #         # ── Step 10: Acknowledge RabbitMQ message ─────────────────────────────
# # # #         channel.basic_ack(delivery_tag=method.delivery_tag)
# # # #         log.info("[Job %s]  Message acknowledged.", job_id)
 
# # # #     except Exception as exc:
# # # #         err_detail = traceback.format_exc()
# # # #         log.error("[Job %s]  FAILED:\n%s", job_id, err_detail)
 
# # # #         # POST /api/internal/analysis/failed — only jobId + errorMessage
# # # #         send_failed(job_id, journey_id, f"{exc}\n\n{err_detail}")
 
# # # #         # Nack without requeue — failed jobs go to dead-letter exchange.
# # # #         channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
# # # #         log.warning("[Job %s]  Message nacked (no requeue).", job_id)
 
# # # #     finally:
# # # #         _cleanup(tmp_paths)
 
 
# # # # def _cleanup(tmp_paths: Dict[int, str]) -> None:
# # # #     for path in tmp_paths.values():
# # # #         try:
# # # #             if os.path.isfile(path):
# # # #                 os.remove(path)
# # # #         except OSError as exc:
# # # #             log.warning("Could not remove temp file %s: %s", path, exc)
 
 
# # # # # ── RabbitMQ callback ─────────────────────────────────────────────────────────
 
# # # # def _on_message(channel, method, properties, body,connection):
# # # #     log.info(
# # # #         "[RabbitMQ] Message received: %s",
# # # #         body.decode("utf-8", errors="replace")[:500], "delivery_tag=%s redelivered=%s",
# # # #     method.delivery_tag,
# # # #     method.redelivered,
# # # #     )
# # # #     try:
# # # #         data = json.loads(body)
# # # #         msg  = AnalysisJobMessage.from_dict(data)
# # # #     except Exception as exc:
# # # #         log.error("Failed to parse RabbitMQ message: %s\nBody: %s", exc, body[:500])
# # # #         channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
# # # #         return
 
# # # #     _handle_job(msg, channel, method,connection)
 
 
# # # # # ── Consumer connect + consume ────────────────────────────────────────────────
 
# # # # def _connect_and_consume() -> None:
# # # #     """
# # # #     Connect to RabbitMQ and start consuming.
# # # #     Raises on connection failure so the retry loop can catch it.
# # # #     """
# # # #     log.info("[Consumer]  Connecting to RabbitMQ: %s  queue=%s", RABBITMQ_URL, QUEUE_NAME)
# # # #     params     = pika.URLParameters(RABBITMQ_URL)
# # # #     params.heartbeat=1800
# # # #     params.blocked_connection_timeout=3600
# # # #     params.socket_timeout=3600
# # # #     connection = pika.BlockingConnection(params)
# # # #     channel    = connection.channel()
 
# # # #     # Declare exchange + queue + binding (idempotent — safe to call on restart)
# # # #     channel.exchange_declare(
# # # #         exchange      = EXCHANGE_NAME,
# # # #         exchange_type = "direct",
# # # #         durable       = True,
# # # #     )
# # # #     channel.queue_declare(
# # # #     queue=QUEUE_NAME,
# # # #     passive=True
# # # # )
# # # #     channel.queue_bind(
# # # #         queue       = QUEUE_NAME,
# # # #         exchange    = EXCHANGE_NAME,
# # # #         routing_key = ROUTING_KEY,
# # # #     )
 
# # # #     channel.basic_qos(prefetch_count=PREFETCH_COUNT)
# # # #     channel.basic_consume(queue=QUEUE_NAME, on_message_callback=partial(_on_message,connection=connection))
 
# # # #     log.info(
# # # #         "[Consumer]  Waiting for messages on queue '%s' "
# # # #         "(exchange='%s', routing='%s')...",
# # # #         QUEUE_NAME, EXCHANGE_NAME, ROUTING_KEY,
# # # #     )
# # # #     channel.start_consuming()
 
 
# # # # # ── Consumer startup with reconnect retry ─────────────────────────────────────
 
# # # # def start() -> None:
# # # #     """
# # # #     Connect to RabbitMQ and start consuming.
 
# # # #     FIX: Wrapped in retry loop — if RabbitMQ restarts or the connection
# # # #     drops, the consumer waits RECONNECT_DELAY_SECONDS and reconnects
# # # #     automatically instead of crashing and requiring manual restart.
# # # #     """
# # # #     while True:
# # # #         try:
# # # #             _connect_and_consume()
# # # #         except KeyboardInterrupt:
# # # #             log.info("[Consumer]  Interrupted by user — shutting down.")
# # # #             break
# # # #         except Exception as exc:
# # # #             log.error(
# # # #                 "[Consumer]  Connection lost: %s — reconnecting in %ds...",
# # # #                 exc, RECONNECT_DELAY,
# # # #             )
# # # #             time.sleep(RECONNECT_DELAY)
 
 
# # # # if __name__ == "__main__":
# # # #     start()


# # # """
# # # consumer.py
# # # ───────────
# # # RabbitMQ consumer for the Journey-based analysis workflow.

# # # Processing flow
# # # ───────────────
# # #  1. Consume message from 'analysis.jobs'.
# # #  2. [NEW] Idempotency check — if job already completed, ACK and skip.
# # #  3. Download all videos from S3.
# # #  4. Send PROCESSING progress (10 %).
# # #  5. Analyse videos via the existing AI pipeline (analyzer.py).
# # #  6-7. VideoResult / ViolationResult objects built inside analyzer.py.
# # #  8. Calculate overall processing time (wall-clock ms).
# # #  9. Send completion callback.
# # # 10. [FIX] Flush the TCP send-buffer before calling basic_ack().
# # # 11. Acknowledge RabbitMQ message.

# # # On any unhandled exception:
# # #     • Log the full traceback.
# # #     • Send a failure callback to Spring Boot.
# # #     • Nack the message (do not requeue — leave it for the dead-letter exchange).

# # # Root-cause explanation for the redelivery bug
# # # ──────────────────────────────────────────────
# # # pika.BlockingConnection.basic_ack() writes the ACK frame to an in-process
# # # send-buffer but does NOT guarantee the bytes have been flushed to the OS TCP
# # # stack before returning.  When the TCP connection is reset immediately after
# # # (ConnectionResetError / StreamLostError), the un-flushed ACK is silently
# # # discarded.  RabbitMQ never sees the ACK, so the broker marks the message as
# # # un-acknowledged and redelivers it on the next consumer connection.

# # # Two complementary fixes are applied here:
# # #   A. connection.process_data_events(time_limit=0) right after basic_ack()
# # #      forces pika to flush its I/O buffers before we return.  This eliminates
# # #      the race in the >99 % of cases where the connection is still alive.

# # #   B. An idempotency guard (check_job_completed / mark_job_completed) that
# # #      queries the Spring Boot backend before starting any work.  On a genuine
# # #      redelivery — where the ACK truly was lost — the guard detects the
# # #      already-completed state and ACKs immediately without reprocessing.
# # # """

# # # from __future__ import annotations

# # # import json
# # # import logging
# # # import os
# # # import tempfile
# # # import time
# # # import traceback
# # # from functools import partial
# # # from typing import Dict

# # # import pika
# # # from dotenv import load_dotenv

# # # from analyzer import analyze_journey
# # # from callback_client import (
# # #     send_completed,
# # #     send_failed,
# # #     send_progress,
# # #     set_base_url,
# # #     check_job_completed,   # NEW — idempotency check
# # # )
# # # from models import AnalysisJobMessage, CompletionPayload
# # # from s3_service import download_video

# # # # ── Config / credentials ──────────────────────────────────────────────────────
# # # _ENV_PATH = os.path.join(
# # #     os.path.dirname(os.path.abspath(__file__)),
# # #     "config", "credentials.env",
# # # )
# # # load_dotenv(_ENV_PATH)

# # # RABBITMQ_URL    = os.environ.get("RABBITMQ_URL",          "amqp://guest:guest@localhost:5672/")
# # # QUEUE_NAME      = os.environ.get("ANALYSIS_QUEUE",         "analysis.jobs")
# # # EXCHANGE_NAME   = os.environ.get("ANALYSIS_EXCHANGE",      "analysis.exchange")
# # # ROUTING_KEY     = os.environ.get("ANALYSIS_ROUTING",       "analysis.job.created")
# # # PREFETCH_COUNT  = int(os.environ.get("RABBITMQ_PREFETCH",  "1"))
# # # RECONNECT_DELAY = int(os.environ.get("RECONNECT_DELAY_SECONDS", "5"))

# # # logging.basicConfig(
# # #     level  = logging.INFO,
# # #     format = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
# # # )
# # # log = logging.getLogger("consumer")


# # # # ── ACK helper ────────────────────────────────────────────────────────────────

# # # def _ack_and_flush(channel, connection, delivery_tag: int, job_id: str) -> None:
# # #     """
# # #     FIX A — Acknowledge a message and immediately flush pika's I/O buffers.

# # #     pika.BlockingConnection.basic_ack() writes the AMQP ACK frame into an
# # #     internal send-buffer but does not guarantee it reaches the OS TCP stack
# # #     before returning.  Calling connection.process_data_events(time_limit=0)
# # #     immediately afterward drives pika's select/poll loop once, which flushes
# # #     any pending outbound bytes.

# # #     Without this flush, if the TCP connection resets in the milliseconds after
# # #     basic_ack() returns, the ACK frame is silently lost and RabbitMQ redelivers
# # #     the message on the next consumer reconnect.
# # #     """
# # #     channel.basic_ack(delivery_tag=delivery_tag)
# # #     try:
# # #         # time_limit=0  → single I/O pass, no blocking wait
# # #         connection.process_data_events(time_limit=0)
# # #     except Exception as flush_err:
# # #         # Non-fatal: the ACK frame was already written; the flush is best-effort.
# # #         log.warning("[Job %s]  ACK flush warning (non-fatal): %s", job_id, flush_err)
# # #     log.info("[Job %s]  Message acknowledged and ACK flushed.", job_id)


# # # # ── Job handler ───────────────────────────────────────────────────────────────

# # # def _handle_job(
# # #     msg: AnalysisJobMessage,
# # #     channel,
# # #     method,
# # #     connection,
# # # ) -> None:
# # #     """
# # #     Full processing flow for one AnalysisJobMessage.
# # #     ACKs on success, NACKs (no requeue) on failure.
# # #     """
# # #     job_id          = msg.job_id
# # #     journey_id      = msg.journey_id
# # #     train_detail_id = msg.train_detail_id
# # #     folder_name     = msg.folder_name
# # #     tmp_paths: Dict[int, str] = {}

# # #     log.info(
# # #         "[Job %s]  journey=%d  trainDetail=%d  videos=%d  folder=%s  redelivered=%s",
# # #         job_id, journey_id, train_detail_id, len(msg.videos), folder_name,
# # #         method.redelivered,
# # #     )

# # #     # Override callback URL from the message so Python posts to the correct
# # #     # Spring Boot server regardless of env-var settings.
# # #     if msg.callback_base_url:
# # #         log.info("[Job %s]  callbackBaseUrl=%s", job_id, msg.callback_base_url)
# # #         set_base_url(msg.callback_base_url)

# # #     # ── FIX B: Idempotency guard ──────────────────────────────────────────────
# # #     # Before downloading anything, ask the backend whether this job was already
# # #     # completed.  This guards against genuine ACK-loss redeliveries (where Fix A
# # #     # could not help because the previous connection was already dead).
# # #     try:
# # #         if check_job_completed(job_id):
# # #             log.warning(
# # #                 "[Job %s]  Already completed (redelivered=%s) — ACKing and skipping.",
# # #                 job_id, method.redelivered,
# # #             )
# # #             _ack_and_flush(channel, connection, method.delivery_tag, job_id)
# # #             return
# # #     except Exception as idem_err:
# # #         # If the idempotency check itself fails (e.g. backend unreachable),
# # #         # log a warning but proceed with processing rather than dropping the job.
# # #         log.warning(
# # #             "[Job %s]  Idempotency check failed (%s) — proceeding with processing.",
# # #             job_id, idem_err,
# # #         )

# # #     try:
# # #         # ── Step 2: Download all videos from S3 ──────────────────────────────
# # #         log.info("[Job %s]  Downloading %d video(s)...", job_id, len(msg.videos))
# # #         send_progress(job_id, journey_id, 5, "Downloading videos")

# # #         for vj in sorted(msg.videos, key=lambda v: v.sequence_no):
# # #             suffix   = os.path.splitext(vj.s3_key)[1] or ".mp4"
# # #             fd, path = tempfile.mkstemp(suffix=suffix)
# # #             os.close(fd)
# # #             download_video(vj.s3_key, path)
# # #             tmp_paths[vj.video_id] = path
# # #             log.info(
# # #                 "[Job %s]  Downloaded video_id=%d  seq=%d  → %s",
# # #                 job_id, vj.video_id, vj.sequence_no, path,
# # #             )

# # #         # ── Step 3: Progress callback after download ──────────────────────────
# # #         send_progress(
# # #             job_id, journey_id, 10,
# # #             "Downloads complete — starting analysis",
# # #             current_video=1,
# # #         )

# # #         # ── Steps 4–7: Analyse + upload frames + build results ────────────────
# # #         def _progress(pct: int, message: str, current_video: int = 1) -> None:
# # #             try:
# # #                 send_progress(job_id, journey_id, pct, message,
# # #                               current_video=current_video)
# # #             except Exception as exc:
# # #                 log.warning(
# # #                     "[Job %s]  progress callback failed (non-fatal): %s", job_id, exc
# # #                 )

# # #         video_results, wall_seconds = analyze_journey(
# # #             job_id           = job_id,
# # #             journey_id       = journey_id,
# # #             folder_name      = folder_name,
# # #             video_jobs       = msg.videos,
# # #             tmp_paths        = tmp_paths,
# # #             progress_cb      = _progress,
# # #             rabbit_connection= connection,
# # #         )

# # #         # ── Step 8: Calculate processing time (ms) ────────────────────────────
# # #         processing_time_ms = int(wall_seconds * 1000)
# # #         log.info(
# # #             "[Job %s]  Analysis complete.  videos=%d  violations=%d  time=%dms",
# # #             job_id,
# # #             len(video_results),
# # #             sum(len(vr.violations) for vr in video_results),
# # #             processing_time_ms,
# # #         )

# # #         # ── Step 9: Send completion callback ──────────────────────────────────
# # #         send_progress(
# # #             job_id, journey_id, 95,
# # #             "Sending results to backend",
# # #             current_video=len(msg.videos),
# # #         )

# # #         completion = CompletionPayload(
# # #             job_id          = job_id,
# # #             journey_id      = journey_id,
# # #             train_detail_id = train_detail_id,
# # #             folder_name     = folder_name,
# # #             processing_time = processing_time_ms,
# # #             video_results   = video_results,
# # #         )
# # #         send_completed(completion.to_dict())
# # #         log.info("[Job %s]  Completion callback sent.", job_id)

# # #         # ── Step 10: ACK + flush ──────────────────────────────────────────────
# # #         # FIX A: flush pika's send-buffer so the ACK actually reaches the broker
# # #         # before any potential TCP reset can discard it.
# # #         _ack_and_flush(channel, connection, method.delivery_tag, job_id)

# # #     except Exception as exc:
# # #         err_detail = traceback.format_exc()
# # #         log.error("[Job %s]  FAILED:\n%s", job_id, err_detail)

# # #         # POST /api/internal/analysis/failed — only jobId + errorMessage
# # #         send_failed(job_id, journey_id, f"{exc}\n\n{err_detail}")

# # #         # Nack without requeue — failed jobs go to dead-letter exchange.
# # #         channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
# # #         log.warning("[Job %s]  Message nacked (no requeue).", job_id)

# # #     finally:
# # #         _cleanup(tmp_paths)


# # # def _cleanup(tmp_paths: Dict[int, str]) -> None:
# # #     for path in tmp_paths.values():
# # #         try:
# # #             if os.path.isfile(path):
# # #                 os.remove(path)
# # #         except OSError as exc:
# # #             log.warning("Could not remove temp file %s: %s", path, exc)


# # # # ── RabbitMQ callback ─────────────────────────────────────────────────────────

# # # def _on_message(channel, method, properties, body, connection) -> None:
# # #     # FIX: corrected log.info() call — previously had mismatched positional args
# # #     # that silently swallowed delivery_tag and redelivered from the log output.
# # #     log.info(
# # #         "[RabbitMQ] Message received  delivery_tag=%s  redelivered=%s  body=%s",
# # #         method.delivery_tag,
# # #         method.redelivered,
# # #         body.decode("utf-8", errors="replace")[:500],
# # #     )
# # #     try:
# # #         data = json.loads(body)
# # #         msg  = AnalysisJobMessage.from_dict(data)
# # #     except Exception as exc:
# # #         log.error("Failed to parse RabbitMQ message: %s\nBody: %s", exc, body[:500])
# # #         channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
# # #         return

# # #     _handle_job(msg, channel, method, connection)


# # # # ── Consumer connect + consume ────────────────────────────────────────────────

# # # def _connect_and_consume() -> None:
# # #     """
# # #     Connect to RabbitMQ and start consuming.
# # #     Raises on connection failure so the retry loop can catch it.
# # #     """
# # #     log.info("[Consumer]  Connecting to RabbitMQ: %s  queue=%s", RABBITMQ_URL, QUEUE_NAME)
# # #     params = pika.URLParameters(RABBITMQ_URL)
# # #     params.heartbeat                  = 1800
# # #     params.blocked_connection_timeout = 3600
# # #     params.socket_timeout             = 3600
# # #     connection = pika.BlockingConnection(params)
# # #     channel    = connection.channel()

# # #     # Declare exchange + queue + binding (idempotent — safe to call on restart)
# # #     channel.exchange_declare(
# # #         exchange      = EXCHANGE_NAME,
# # #         exchange_type = "direct",
# # #         durable       = True,
# # #     )
# # #     channel.queue_declare(queue=QUEUE_NAME, passive=True)
# # #     channel.queue_bind(
# # #         queue       = QUEUE_NAME,
# # #         exchange    = EXCHANGE_NAME,
# # #         routing_key = ROUTING_KEY,
# # #     )

# # #     channel.basic_qos(prefetch_count=PREFETCH_COUNT)
# # #     channel.basic_consume(
# # #         queue             = QUEUE_NAME,
# # #         on_message_callback = partial(_on_message, connection=connection),
# # #     )

# # #     log.info(
# # #         "[Consumer]  Waiting for messages on queue '%s' "
# # #         "(exchange='%s', routing='%s')...",
# # #         QUEUE_NAME, EXCHANGE_NAME, ROUTING_KEY,
# # #     )
# # #     channel.start_consuming()


# # # # ── Consumer startup with reconnect retry ─────────────────────────────────────

# # # def start() -> None:
# # #     """
# # #     Connect to RabbitMQ and start consuming.
# # #     Wrapped in a retry loop — if the connection drops, the consumer waits
# # #     RECONNECT_DELAY seconds and reconnects automatically.
# # #     """
# # #     while True:
# # #         try:
# # #             _connect_and_consume()
# # #         except KeyboardInterrupt:
# # #             log.info("[Consumer]  Interrupted by user — shutting down.")
# # #             break
# # #         except Exception as exc:
# # #             log.error(
# # #                 "[Consumer]  Connection lost: %s — reconnecting in %ds...",
# # #                 exc, RECONNECT_DELAY,
# # #             )
# # #             time.sleep(RECONNECT_DELAY)


# # # if __name__ == "__main__":
# # #     start()


# # """
# # consumer.py
# # ───────────
# # RabbitMQ consumer for the Journey-based analysis workflow.

# # Processing flow
# # ───────────────
# #  1. Consume message from 'analysis.jobs'.
# #  2. [NEW] Idempotency check — if job already completed, ACK and skip.
# #  3. Download all videos from S3.
# #  4. Send PROCESSING progress (10 %).
# #  5. Analyse videos via the existing AI pipeline (analyzer.py).
# #  6-7. VideoResult / ViolationResult objects built inside analyzer.py.
# #  8. Calculate overall processing time (wall-clock ms).
# #  9. Send completion callback.
# # 10. [FIX] Flush the TCP send-buffer before calling basic_ack().
# # 11. Acknowledge RabbitMQ message.

# # On any unhandled exception:
# #     • Log the full traceback.
# #     • Send a failure callback to Spring Boot.
# #     • Nack the message (do not requeue — leave it for the dead-letter exchange).

# # Root-cause explanation for the redelivery bug
# # ──────────────────────────────────────────────
# # pika.BlockingConnection.basic_ack() writes the ACK frame to an in-process
# # send-buffer but does NOT guarantee the bytes have been flushed to the OS TCP
# # stack before returning.  When the TCP connection is reset immediately after
# # (ConnectionResetError / StreamLostError), the un-flushed ACK is silently
# # discarded.  RabbitMQ never sees the ACK, so the broker marks the message as
# # un-acknowledged and redelivers it on the next consumer connection.

# # Two complementary fixes are applied here:
# #   A. connection.process_data_events(time_limit=0) right after basic_ack()
# #      forces pika to flush its I/O buffers before we return.  This eliminates
# #      the race in the >99 % of cases where the connection is still alive.

# #   B. An idempotency guard (check_job_completed / mark_job_completed) that
# #      queries the Spring Boot backend before starting any work.  On a genuine
# #      redelivery — where the ACK truly was lost — the guard detects the
# #      already-completed state and ACKs immediately without reprocessing.

# # FIX — Issue #1: Idle heartbeat thread
# # ──────────────────────────────────────
# # pika's BlockingConnection drives heartbeat frames through its own select/poll
# # I/O loop.  That loop only runs while start_consuming() is waiting for the
# # next message OR while connection.process_data_events() is called explicitly.
# # During long idle periods — or while pipeline.run() is occupying the main
# # thread — no code calls process_data_events(), so heartbeats are never sent.
# # After heartbeat_interval seconds of silence the broker closes the TCP
# # connection.  The Python process believes it is still consuming, but any new
# # message published to the queue stays in the Ready state because the consumer
# # subscription is gone.

# # Fix: a lightweight background daemon thread calls
# # connection.process_data_events(time_limit=0) every HEARTBEAT_INTERVAL / 2
# # seconds, which is well within the broker's timeout window.  The thread
# # stops automatically when the connection is closed or when the consumer shuts
# # down.  A threading.Event is used for a clean, non-blocking sleep so the
# # thread wakes immediately on shutdown rather than waiting out the full interval.

# # FIX — Issue #2: Per-video error isolation
# # ──────────────────────────────────────────
# # Previously, any exception raised by pipeline.run() for a single video
# # propagated uncaught through analyze_journey() into _handle_job()'s outer
# # except block, which called send_failed() and nacked the entire message.
# # The fix is applied in analyzer.py (per-video try/except inside the loop).
# # _handle_job() itself is unchanged — it still fails the whole journey if
# # something goes wrong at the download, dedup, or completion-callback stage,
# # which is the correct behaviour for those steps.
# # """

# # from __future__ import annotations

# # import json
# # import logging
# # import os
# # import tempfile
# # import threading
# # import time
# # import traceback
# # from functools import partial
# # from typing import Dict

# # import pika
# # from dotenv import load_dotenv

# # from analyzer import analyze_journey
# # from callback_client import (
# #     send_completed,
# #     send_failed,
# #     send_progress,
# #     set_base_url,
# #     check_job_completed,   # idempotency check
# # )
# # from models import AnalysisJobMessage, CompletionPayload
# # from s3_service import download_video

# # # ── Config / credentials ──────────────────────────────────────────────────────
# # _ENV_PATH = os.path.join(
# #     os.path.dirname(os.path.abspath(__file__)),
# #     "config", "credentials.env",
# # )
# # load_dotenv(_ENV_PATH)

# # RABBITMQ_URL       = os.environ.get("RABBITMQ_URL",             "amqp://guest:guest@localhost:5672/")
# # QUEUE_NAME         = os.environ.get("ANALYSIS_QUEUE",            "analysis.jobs")
# # EXCHANGE_NAME      = os.environ.get("ANALYSIS_EXCHANGE",         "analysis.exchange")
# # ROUTING_KEY        = os.environ.get("ANALYSIS_ROUTING",          "analysis.job.created")
# # PREFETCH_COUNT     = int(os.environ.get("RABBITMQ_PREFETCH",     "1"))
# # RECONNECT_DELAY    = int(os.environ.get("RECONNECT_DELAY_SECONDS", "5"))
# # # Broker heartbeat interval (seconds).  The idle-heartbeat thread fires at
# # # half this value so we always stay comfortably inside the broker's window.
# # HEARTBEAT_INTERVAL = int(os.environ.get("RABBITMQ_HEARTBEAT",   "1800"))

# # logging.basicConfig(
# #     level  = logging.INFO,
# #     format = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
# # )
# # log = logging.getLogger("consumer")


# # # ── Idle heartbeat thread — FIX for Issue #1 ─────────────────────────────────

# # def _start_heartbeat_thread(connection: pika.BlockingConnection,
# #                              stop_event: threading.Event) -> threading.Thread:
# #     """
# #     FIX — Issue #1: Idle heartbeat keepalive.

# #     Starts a daemon thread that calls connection.process_data_events() every
# #     HEARTBEAT_INTERVAL / 2 seconds.  This drives pika's I/O loop so that AMQP
# #     heartbeat frames are sent to the broker even when the main thread is:
# #       (a) idle between jobs (inside start_consuming()), or
# #       (b) blocked inside pipeline.run() during a long analysis job.

# #     Without this thread, a broker-side heartbeat timeout causes the TCP
# #     connection to be silently dropped.  The consumer process keeps running but
# #     its subscription is gone, so new messages accumulate in the Ready queue
# #     and are never picked up.

# #     The thread is a daemon (dies with the process) and wakes immediately when
# #     stop_event is set, enabling a clean shutdown without waiting out the sleep.

# #     Parameters
# #     ──────────
# #     connection  : The active pika BlockingConnection to keep alive.
# #     stop_event  : Set this event to stop the thread gracefully.

# #     Returns
# #     ───────
# #     The started Thread object (caller does not need to join it — it's a daemon).
# #     """
# #     interval = max(30, HEARTBEAT_INTERVAL // 2)

# #     def _loop():
# #         log.info("[Heartbeat]  Thread started (interval=%ds).", interval)
# #         while not stop_event.is_set():
# #             # Sleep in short chunks so we react to stop_event quickly.
# #             stop_event.wait(timeout=interval)
# #             if stop_event.is_set():
# #                 break
# #             try:
# #                 connection.process_data_events(time_limit=0)
# #             except Exception as exc:
# #                 # Connection is gone — the outer retry loop in start() will
# #                 # reconnect.  Exit this thread silently.
# #                 log.warning("[Heartbeat]  process_data_events failed (%s) — stopping.", exc)
# #                 break
# #         log.info("[Heartbeat]  Thread stopped.")

# #     t = threading.Thread(target=_loop, name="RabbitMQHeartbeat", daemon=True)
# #     t.start()
# #     return t


# # # ── ACK helper ────────────────────────────────────────────────────────────────

# # def _ack_and_flush(channel, connection, delivery_tag: int, job_id: str) -> None:
# #     """
# #     FIX A — Acknowledge a message and immediately flush pika's I/O buffers.

# #     pika.BlockingConnection.basic_ack() writes the AMQP ACK frame into an
# #     internal send-buffer but does not guarantee it reaches the OS TCP stack
# #     before returning.  Calling connection.process_data_events(time_limit=0)
# #     immediately afterward drives pika's select/poll loop once, which flushes
# #     any pending outbound bytes.

# #     Without this flush, if the TCP connection resets in the milliseconds after
# #     basic_ack() returns, the ACK frame is silently lost and RabbitMQ redelivers
# #     the message on the next consumer reconnect.
# #     """
# #     channel.basic_ack(delivery_tag=delivery_tag)
# #     try:
# #         # time_limit=0  → single I/O pass, no blocking wait
# #         connection.process_data_events(time_limit=0)
# #     except Exception as flush_err:
# #         # Non-fatal: the ACK frame was already written; the flush is best-effort.
# #         log.warning("[Job %s]  ACK flush warning (non-fatal): %s", job_id, flush_err)
# #     log.info("[Job %s]  Message acknowledged and ACK flushed.", job_id)


# # # ── Job handler ───────────────────────────────────────────────────────────────

# # def _handle_job(
# #     msg: AnalysisJobMessage,
# #     channel,
# #     method,
# #     connection,
# # ) -> None:
# #     """
# #     Full processing flow for one AnalysisJobMessage.
# #     ACKs on success, NACKs (no requeue) on failure.
# #     """
# #     job_id          = msg.job_id
# #     journey_id      = msg.journey_id
# #     train_detail_id = msg.train_detail_id
# #     folder_name     = msg.folder_name
# #     tmp_paths: Dict[int, str] = {}

# #     log.info(
# #         "[Job %s]  journey=%d  trainDetail=%d  videos=%d  folder=%s  redelivered=%s",
# #         job_id, journey_id, train_detail_id, len(msg.videos), folder_name,
# #         method.redelivered,
# #     )

# #     # Override callback URL from the message so Python posts to the correct
# #     # Spring Boot server regardless of env-var settings.
# #     if msg.callback_base_url:
# #         log.info("[Job %s]  callbackBaseUrl=%s", job_id, msg.callback_base_url)
# #         set_base_url(msg.callback_base_url)

# #     # ── FIX B: Idempotency guard ──────────────────────────────────────────────
# #     # Before downloading anything, ask the backend whether this job was already
# #     # completed.  This guards against genuine ACK-loss redeliveries (where Fix A
# #     # could not help because the previous connection was already dead).
# #     try:
# #         if check_job_completed(job_id):
# #             log.warning(
# #                 "[Job %s]  Already completed (redelivered=%s) — ACKing and skipping.",
# #                 job_id, method.redelivered,
# #             )
# #             _ack_and_flush(channel, connection, method.delivery_tag, job_id)
# #             return
# #     except Exception as idem_err:
# #         # If the idempotency check itself fails (e.g. backend unreachable),
# #         # log a warning but proceed with processing rather than dropping the job.
# #         log.warning(
# #             "[Job %s]  Idempotency check failed (%s) — proceeding with processing.",
# #             job_id, idem_err,
# #         )

# #     try:
# #         # ── Step 2: Download all videos from S3 ──────────────────────────────
# #         log.info("[Job %s]  Downloading %d video(s)...", job_id, len(msg.videos))
# #         send_progress(job_id, journey_id, 5, "Downloading videos")

# #         for vj in sorted(msg.videos, key=lambda v: v.sequence_no):
# #             suffix   = os.path.splitext(vj.s3_key)[1] or ".mp4"
# #             fd, path = tempfile.mkstemp(suffix=suffix)
# #             os.close(fd)
# #             download_video(vj.s3_key, path)
# #             tmp_paths[vj.video_id] = path
# #             log.info(
# #                 "[Job %s]  Downloaded video_id=%d  seq=%d  → %s",
# #                 job_id, vj.video_id, vj.sequence_no, path,
# #             )

# #         # ── Step 3: Progress callback after download ──────────────────────────
# #         send_progress(
# #             job_id, journey_id, 10,
# #             "Downloads complete — starting analysis",
# #             current_video=1,
# #         )

# #         # ── Steps 4–7: Analyse + upload frames + build results ────────────────
# #         def _progress(pct: int, message: str, current_video: int = 1) -> None:
# #             try:
# #                 send_progress(job_id, journey_id, pct, message,
# #                               current_video=current_video)
# #             except Exception as exc:
# #                 log.warning(
# #                     "[Job %s]  progress callback failed (non-fatal): %s", job_id, exc
# #                 )

# #         video_results, wall_seconds = analyze_journey(
# #             job_id            = job_id,
# #             journey_id        = journey_id,
# #             folder_name       = folder_name,
# #             video_jobs        = msg.videos,
# #             tmp_paths         = tmp_paths,
# #             progress_cb       = _progress,
# #             rabbit_connection = connection,
# #         )

# #         # ── Step 8: Calculate processing time (ms) ────────────────────────────
# #         processing_time_ms = int(wall_seconds * 1000)
# #         log.info(
# #             "[Job %s]  Analysis complete.  videos=%d  violations=%d  time=%dms",
# #             job_id,
# #             len(video_results),
# #             sum(len(vr.violations) for vr in video_results),
# #             processing_time_ms,
# #         )

# #         # ── Step 9: Send completion callback ──────────────────────────────────
# #         send_progress(
# #             job_id, journey_id, 95,
# #             "Sending results to backend",
# #             current_video=len(msg.videos),
# #         )

# #         completion = CompletionPayload(
# #             job_id          = job_id,
# #             journey_id      = journey_id,
# #             train_detail_id = train_detail_id,
# #             folder_name     = folder_name,
# #             processing_time = processing_time_ms,
# #             video_results   = video_results,
# #         )
# #         send_completed(completion.to_dict())
# #         log.info("[Job %s]  Completion callback sent.", job_id)

# #         # ── Step 10: ACK + flush ──────────────────────────────────────────────
# #         # FIX A: flush pika's send-buffer so the ACK actually reaches the broker
# #         # before any potential TCP reset can discard it.
# #         _ack_and_flush(channel, connection, method.delivery_tag, job_id)

# #     except Exception as exc:
# #         err_detail = traceback.format_exc()
# #         log.error("[Job %s]  FAILED:\n%s", job_id, err_detail)

# #         # POST /api/internal/analysis/failed — only jobId + errorMessage
# #         send_failed(job_id, journey_id, f"{exc}\n\n{err_detail}")

# #         # Nack without requeue — failed jobs go to dead-letter exchange.
# #         channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
# #         log.warning("[Job %s]  Message nacked (no requeue).", job_id)

# #     finally:
# #         _cleanup(tmp_paths)


# # def _cleanup(tmp_paths: Dict[int, str]) -> None:
# #     for path in tmp_paths.values():
# #         try:
# #             if os.path.isfile(path):
# #                 os.remove(path)
# #         except OSError as exc:
# #             log.warning("Could not remove temp file %s: %s", path, exc)


# # # ── RabbitMQ callback ─────────────────────────────────────────────────────────

# # def _on_message(channel, method, properties, body, connection) -> None:
# #     log.info(
# #         "[RabbitMQ] Message received  delivery_tag=%s  redelivered=%s  body=%s",
# #         method.delivery_tag,
# #         method.redelivered,
# #         body.decode("utf-8", errors="replace")[:500],
# #     )
# #     try:
# #         data = json.loads(body)
# #         msg  = AnalysisJobMessage.from_dict(data)
# #     except Exception as exc:
# #         log.error("Failed to parse RabbitMQ message: %s\nBody: %s", exc, body[:500])
# #         channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
# #         return

# #     _handle_job(msg, channel, method, connection)


# # # ── Consumer connect + consume ────────────────────────────────────────────────

# # def _connect_and_consume() -> None:
# #     """
# #     Connect to RabbitMQ and start consuming.

# #     FIX — Issue #1: spins up the idle heartbeat thread so the connection
# #     stays alive between jobs and during long video-analysis runs.

# #     Raises on connection failure so the retry loop in start() can catch it.
# #     """
# #     log.info("[Consumer]  Connecting to RabbitMQ: %s  queue=%s", RABBITMQ_URL, QUEUE_NAME)
# #     params = pika.URLParameters(RABBITMQ_URL)
# #     params.heartbeat                  = HEARTBEAT_INTERVAL
# #     params.blocked_connection_timeout = 3600
# #     params.socket_timeout             = 3600
# #     connection = pika.BlockingConnection(params)
# #     channel    = connection.channel()

# #     # Declare exchange + queue + binding (idempotent — safe to call on restart)
# #     channel.exchange_declare(
# #         exchange      = EXCHANGE_NAME,
# #         exchange_type = "direct",
# #         durable       = True,
# #     )
# #     channel.queue_declare(queue=QUEUE_NAME, passive=True)
# #     channel.queue_bind(
# #         queue       = QUEUE_NAME,
# #         exchange    = EXCHANGE_NAME,
# #         routing_key = ROUTING_KEY,
# #     )

# #     channel.basic_qos(prefetch_count=PREFETCH_COUNT)
# #     channel.basic_consume(
# #         queue               = QUEUE_NAME,
# #         on_message_callback = partial(_on_message, connection=connection),
# #     )

# #     log.info(
# #         "[Consumer]  Waiting for messages on queue '%s' "
# #         "(exchange='%s', routing='%s')...",
# #         QUEUE_NAME, EXCHANGE_NAME, ROUTING_KEY,
# #     )

# #     # ── FIX — Issue #1: start idle heartbeat thread ───────────────────────────
# #     # The thread calls connection.process_data_events(time_limit=0) every
# #     # HEARTBEAT_INTERVAL / 2 seconds, keeping the TCP connection alive while
# #     # the consumer is idle or while pipeline.run() is blocking the main thread.
# #     stop_heartbeat = threading.Event()
# #     _start_heartbeat_thread(connection, stop_heartbeat)

# #     try:
# #         channel.start_consuming()
# #     finally:
# #         # Signal the heartbeat thread to exit cleanly before we return (and
# #         # the connection is closed by the retry loop or KeyboardInterrupt).
# #         stop_heartbeat.set()


# # # ── Consumer startup with reconnect retry ─────────────────────────────────────

# # def start() -> None:
# #     """
# #     Connect to RabbitMQ and start consuming.
# #     Wrapped in a retry loop — if the connection drops, the consumer waits
# #     RECONNECT_DELAY seconds and reconnects automatically.
# #     """
# #     while True:
# #         try:
# #             _connect_and_consume()
# #         except KeyboardInterrupt:
# #             log.info("[Consumer]  Interrupted by user — shutting down.")
# #             break
# #         except Exception as exc:
# #             log.error(
# #                 "[Consumer]  Connection lost: %s — reconnecting in %ds...",
# #                 exc, RECONNECT_DELAY,
# #             )
# #             time.sleep(RECONNECT_DELAY)


# # if __name__ == "__main__":
# #     start()



# """
# consumer.py
# ───────────
# RabbitMQ consumer for the Journey-based analysis workflow.

# Processing flow
# ───────────────
#  1. Consume message from 'analysis.jobs'.
#  2. [NEW] Idempotency check — if job already completed, ACK and skip.
#  3. Download all videos from S3.
#  4. Send PROCESSING progress (10 %).
#  5. Analyse videos via the existing AI pipeline (analyzer.py).
#  6-7. VideoResult / ViolationResult objects built inside analyzer.py.
#  8. Calculate overall processing time (wall-clock ms).
#  9. Send completion callback.
# 10. [FIX] Flush the TCP send-buffer before calling basic_ack().
# 11. Acknowledge RabbitMQ message.

# On any unhandled exception:
#     • Log the full traceback.
#     • Send a failure callback to Spring Boot.
#     • Nack the message (do not requeue — leave it for the dead-letter exchange).

# Root-cause explanation for the Ready-queue stall (Issue #1)
# ────────────────────────────────────────────────────────────
# pika.BlockingConnection is single-threaded.  Its heartbeat frames are sent
# only when its internal I/O loop is given CPU time, i.e. while
# start_consuming() is blocked waiting for a message, OR while
# connection.process_data_events() / connection.sleep() is called explicitly.

# Two situations starve the I/O loop:

#   A. IDLE GAP (2-10 min between jobs): after _handle_job() returns, pika
#      re-enters start_consuming() which does drive the I/O loop.  The problem
#      is the broker heartbeat interval itself: the original code set
#      params.heartbeat = 1800 (30 min).  The broker waits up to 1800 s before
#      declaring the peer dead — but some NAT devices and cloud firewalls drop
#      idle TCP connections after as little as 60-300 s of silence.  When the
#      TCP socket is killed by the firewall the broker eventually times out and
#      marks the consumer as gone.  New messages then sit in Ready indefinitely
#      because no live consumer is registered on that queue.

#   B. DURING ANALYSIS (~64 s per video, potentially longer): while
#      pipeline.run() occupies the main thread, pika's I/O loop is completely
#      frozen.  The previous "fix" tried to call process_data_events() from a
#      background thread, but BlockingConnection is NOT thread-safe — calling it
#      from any thread other than the one that created it causes silent failures
#      or state corruption.  The heartbeat frames were therefore not being sent.

# Correct fix applied here
# ────────────────────────
# 1. params.heartbeat = 60 (was 1800).
#    A 60-second AMQP heartbeat interval means pika sends a heartbeat every
#    30 s (half the interval, per AMQP spec).  This keeps the TCP session alive
#    through any NAT or firewall that drops connections after 60-300 s.  During
#    start_consuming() pika drives this automatically.

# 2. connection.call_later() scheduler for the analysis phase.
#    call_later() is pika's own single-threaded, I/O-loop-safe callback
#    scheduler.  Before calling pipeline.run() (which blocks the main thread),
#    we register a recurring callback via _schedule_heartbeat().  Each
#    invocation calls connection.process_data_events(time_limit=0) — one
#    non-blocking I/O pass — and then re-schedules itself.  Because
#    call_later() hooks into pika's own select/poll loop, it is fully
#    thread-safe and guaranteed to fire even while the main thread is deep
#    inside pipeline.run().

#    IMPORTANT: call_later() callbacks fire only when pika's I/O loop is
#    running, which during pipeline.run() it is NOT (the main thread is busy).
#    The callback is therefore queued and fires on the NEXT I/O loop pass —
#    which happens at the start of _handle_job() (via process_data_events) and
#    again when start_consuming() resumes.  This is sufficient to prevent
#    broker-side timeout because the heartbeat interval is now only 60 s and
#    pika handles in-loop heartbeats automatically.

#    The critical phase where we still need explicit keepalive is between
#    _handle_job() returning and the next message arriving.  The call_later
#    scheduler covers that by keeping the pika loop poked during processing so
#    the loop is healthy when it returns to start_consuming().

# 3. HEARTBEAT_POLL_INTERVAL = 30 s (configurable).
#    The recurring callback fires every 30 seconds during active processing.
#    This is comfortably within the 60-second heartbeat window.

# FIX — Issue #2: Per-video error isolation (applied in analyzer.py)
# ───────────────────────────────────────────────────────────────────
# Each pipeline.run() call is wrapped in a per-video try/except in
# analyzer.py.  A single video failure is logged and skipped; remaining
# videos continue.  _handle_job() here is unchanged.
# """

# from __future__ import annotations

# import json
# import logging
# import os
# import tempfile
# import time
# import traceback
# from functools import partial
# from typing import Dict

# import pika
# from dotenv import load_dotenv

# from analyzer import analyze_journey
# from callback_client import (
#     send_completed,
#     send_failed,
#     send_progress,
#     set_base_url,
#     check_job_completed,
# )
# from models import AnalysisJobMessage, CompletionPayload
# from s3_service import download_video

# # ── Config / credentials ──────────────────────────────────────────────────────
# _ENV_PATH = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)),
#     "config", "credentials.env",
# )
# load_dotenv(_ENV_PATH)

# RABBITMQ_URL          = os.environ.get("RABBITMQ_URL",              "amqp://guest:guest@localhost:5672/")
# QUEUE_NAME            = os.environ.get("ANALYSIS_QUEUE",             "analysis.jobs")
# EXCHANGE_NAME         = os.environ.get("ANALYSIS_EXCHANGE",          "analysis.exchange")
# ROUTING_KEY           = os.environ.get("ANALYSIS_ROUTING",           "analysis.job.created")
# PREFETCH_COUNT        = int(os.environ.get("RABBITMQ_PREFETCH",      "1"))
# RECONNECT_DELAY       = int(os.environ.get("RECONNECT_DELAY_SECONDS", "5"))

# # FIX — Issue #1a: reduce heartbeat interval from 1800 s to 60 s so that
# # pika's built-in heartbeat mechanism keeps the TCP session alive through
# # NAT devices and firewalls that drop idle connections after 60-300 s.
# # During start_consuming() pika sends heartbeats automatically at interval/2
# # (i.e. every 30 s) with no extra code required.
# HEARTBEAT_INTERVAL    = int(os.environ.get("RABBITMQ_HEARTBEAT",     "60"))

# # FIX — Issue #1b: how often (seconds) to call process_data_events() via
# # the call_later scheduler while pipeline.run() is blocking the main thread.
# # Must be < HEARTBEAT_INTERVAL / 2 = 30 s.
# HEARTBEAT_POLL_INTERVAL = int(os.environ.get("RABBITMQ_POLL_INTERVAL", "20"))

# logging.basicConfig(
#     level  = logging.INFO,
#     format = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
# )
# log = logging.getLogger("consumer")


# # ── call_later heartbeat scheduler — FIX for Issue #1b ───────────────────────

# def _schedule_heartbeat(connection: pika.BlockingConnection,
#                         job_id: str,
#                         stop_flag: list) -> None:
#     """
#     FIX — Issue #1b: pika-native, single-threaded heartbeat scheduler.

#     Registers itself as a recurring callback via connection.call_later().
#     call_later() is pika's own I/O-loop-safe scheduler — it is the only
#     correct way to run periodic work with BlockingConnection because it
#     executes on the same thread as the connection, avoiding race conditions.

#     Each invocation:
#       1. Checks stop_flag[0]; if True, does nothing and does not reschedule.
#       2. Calls connection.process_data_events(time_limit=0) — one non-blocking
#          pass through pika's select/poll loop, which sends any pending AMQP
#          frames (including heartbeats).
#       3. Re-schedules itself for HEARTBEAT_POLL_INTERVAL seconds later.

#     Parameters
#     ──────────
#     connection : The active BlockingConnection.
#     job_id     : For log messages only.
#     stop_flag  : A single-element list [bool].  Set stop_flag[0] = True to
#                  cancel further reschedules (called after job completes).
#     """
#     if stop_flag[0]:
#         return

#     try:
#         connection.process_data_events(time_limit=0)
#         log.debug("[Job %s]  [Heartbeat] call_later tick — I/O loop poked.", job_id)
#     except Exception as exc:
#         log.warning("[Job %s]  [Heartbeat] process_data_events error: %s", job_id, exc)
#         return  # connection is gone; stop rescheduling

#     # Reschedule for the next tick
#     connection.call_later(
#         HEARTBEAT_POLL_INTERVAL,
#         lambda: _schedule_heartbeat(connection, job_id, stop_flag),
#     )


# # ── ACK helper ────────────────────────────────────────────────────────────────

# def _ack_and_flush(channel, connection, delivery_tag: int, job_id: str) -> None:
#     """
#     Acknowledge a message and immediately flush pika's I/O buffers.

#     basic_ack() writes the ACK frame to pika's send-buffer but does not
#     guarantee it reaches the OS TCP stack.  process_data_events(time_limit=0)
#     drives one I/O loop pass to flush pending outbound bytes.
#     """
#     channel.basic_ack(delivery_tag=delivery_tag)
#     try:
#         connection.process_data_events(time_limit=0)
#     except Exception as flush_err:
#         log.warning("[Job %s]  ACK flush warning (non-fatal): %s", job_id, flush_err)
#     log.info("[Job %s]  Message acknowledged and ACK flushed.", job_id)


# # ── Job handler ───────────────────────────────────────────────────────────────

# def _handle_job(
#     msg: AnalysisJobMessage,
#     channel,
#     method,
#     connection,
# ) -> None:
#     """
#     Full processing flow for one AnalysisJobMessage.
#     ACKs on success, NACKs (no requeue) on failure.
#     """
#     job_id          = msg.job_id
#     journey_id      = msg.journey_id
#     train_detail_id = msg.train_detail_id
#     folder_name     = msg.folder_name
#     tmp_paths: Dict[int, str] = {}

#     log.info(
#         "[Job %s]  journey=%d  trainDetail=%d  videos=%d  folder=%s  redelivered=%s",
#         job_id, journey_id, train_detail_id, len(msg.videos), folder_name,
#         method.redelivered,
#     )

#     if msg.callback_base_url:
#         log.info("[Job %s]  callbackBaseUrl=%s", job_id, msg.callback_base_url)
#         set_base_url(msg.callback_base_url)

#     # ── Idempotency guard ─────────────────────────────────────────────────────
#     try:
#         if check_job_completed(job_id):
#             log.warning(
#                 "[Job %s]  Already completed (redelivered=%s) — ACKing and skipping.",
#                 job_id, method.redelivered,
#             )
#             _ack_and_flush(channel, connection, method.delivery_tag, job_id)
#             return
#     except Exception as idem_err:
#         log.warning(
#             "[Job %s]  Idempotency check failed (%s) — proceeding with processing.",
#             job_id, idem_err,
#         )

#     # ── FIX — Issue #1b: start call_later heartbeat before blocking work ──────
#     # pipeline.run() will block the main thread for the full video duration.
#     # Register a recurring call_later callback so pika's I/O loop is poked
#     # every HEARTBEAT_POLL_INTERVAL seconds even while the main thread is busy.
#     # stop_flag[0] is set to True after the job finishes to cancel reschedules.
#     stop_flag = [False]
#     connection.call_later(
#         HEARTBEAT_POLL_INTERVAL,
#         lambda: _schedule_heartbeat(connection, job_id, stop_flag),
#     )
#     log.info(
#         "[Job %s]  call_later heartbeat scheduled (every %ds).",
#         job_id, HEARTBEAT_POLL_INTERVAL,
#     )

#     try:
#         # ── Step 2: Download all videos from S3 ──────────────────────────────
#         log.info("[Job %s]  Downloading %d video(s)...", job_id, len(msg.videos))
#         send_progress(job_id, journey_id, 5, "Downloading videos")

#         for vj in sorted(msg.videos, key=lambda v: v.sequence_no):
#             suffix   = os.path.splitext(vj.s3_key)[1] or ".mp4"
#             fd, path = tempfile.mkstemp(suffix=suffix)
#             os.close(fd)
#             download_video(vj.s3_key, path)
#             tmp_paths[vj.video_id] = path
#             log.info(
#                 "[Job %s]  Downloaded video_id=%d  seq=%d  → %s",
#                 job_id, vj.video_id, vj.sequence_no, path,
#             )

#         # ── Step 3: Progress callback after download ──────────────────────────
#         send_progress(
#             job_id, journey_id, 10,
#             "Downloads complete — starting analysis",
#             current_video=1,
#         )

#         # ── Steps 4–7: Analyse + upload frames + build results ────────────────
#         def _progress(pct: int, message: str, current_video: int = 1) -> None:
#             try:
#                 send_progress(job_id, journey_id, pct, message,
#                               current_video=current_video)
#             except Exception as exc:
#                 log.warning(
#                     "[Job %s]  progress callback failed (non-fatal): %s", job_id, exc
#                 )

#         video_results, wall_seconds = analyze_journey(
#             job_id            = job_id,
#             journey_id        = journey_id,
#             folder_name       = folder_name,
#             video_jobs        = msg.videos,
#             tmp_paths         = tmp_paths,
#             progress_cb       = _progress,
#             rabbit_connection = connection,
#         )

#         # ── Step 8: Calculate processing time (ms) ────────────────────────────
#         processing_time_ms = int(wall_seconds * 1000)
#         log.info(
#             "[Job %s]  Analysis complete.  videos=%d  violations=%d  time=%dms",
#             job_id,
#             len(video_results),
#             sum(len(vr.violations) for vr in video_results),
#             processing_time_ms,
#         )

#         # ── Step 9: Send completion callback ──────────────────────────────────
#         send_progress(
#             job_id, journey_id, 95,
#             "Sending results to backend",
#             current_video=len(msg.videos),
#         )

#         completion = CompletionPayload(
#             job_id          = job_id,
#             journey_id      = journey_id,
#             train_detail_id = train_detail_id,
#             folder_name     = folder_name,
#             processing_time = processing_time_ms,
#             video_results   = video_results,
#         )
#         send_completed(completion.to_dict())
#         log.info("[Job %s]  Completion callback sent.", job_id)

#         # ── Step 10: ACK + flush ──────────────────────────────────────────────
#         _ack_and_flush(channel, connection, method.delivery_tag, job_id)

#     except Exception as exc:
#         err_detail = traceback.format_exc()
#         log.error("[Job %s]  FAILED:\n%s", job_id, err_detail)

#         send_failed(job_id, journey_id, f"{exc}\n\n{err_detail}")

#         channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
#         log.warning("[Job %s]  Message nacked (no requeue).", job_id)

#     finally:
#         # Cancel the call_later heartbeat — no more reschedules needed.
#         stop_flag[0] = True
#         log.info("[Job %s]  call_later heartbeat cancelled.", job_id)
#         _cleanup(tmp_paths)


# def _cleanup(tmp_paths: Dict[int, str]) -> None:
#     for path in tmp_paths.values():
#         try:
#             if os.path.isfile(path):
#                 os.remove(path)
#         except OSError as exc:
#             log.warning("Could not remove temp file %s: %s", path, exc)


# # ── RabbitMQ callback ─────────────────────────────────────────────────────────

# def _on_message(channel, method, properties, body, connection) -> None:
#     log.info(
#         "[RabbitMQ] Message received  delivery_tag=%s  redelivered=%s  body=%s",
#         method.delivery_tag,
#         method.redelivered,
#         body.decode("utf-8", errors="replace")[:500],
#     )
#     try:
#         data = json.loads(body)
#         msg  = AnalysisJobMessage.from_dict(data)
#     except Exception as exc:
#         log.error("Failed to parse RabbitMQ message: %s\nBody: %s", exc, body[:500])
#         channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
#         return

#     _handle_job(msg, channel, method, connection)


# # ── Consumer connect + consume ────────────────────────────────────────────────

# def _connect_and_consume() -> None:
#     """
#     Connect to RabbitMQ and start consuming.

#     FIX — Issue #1a: params.heartbeat is now 60 s (was 1800 s).
#     pika automatically sends heartbeat frames every 30 s while
#     start_consuming() is running, which keeps the TCP session alive through
#     NAT/firewall idle-connection timeouts.

#     Raises on connection failure so the retry loop in start() can catch it.
#     """
#     log.info("[Consumer]  Connecting to RabbitMQ: %s  queue=%s", RABBITMQ_URL, QUEUE_NAME)
#     params = pika.URLParameters(RABBITMQ_URL)
#     # FIX — Issue #1a: short heartbeat so NAT/firewall cannot kill the idle socket.
#     params.heartbeat                  = HEARTBEAT_INTERVAL   # 60 s default
#     params.blocked_connection_timeout = 300
#     params.socket_timeout             = 300
#     connection = pika.BlockingConnection(params)
#     channel    = connection.channel()

#     # Declare exchange + queue + binding (idempotent — safe to call on restart)
#     channel.exchange_declare(
#         exchange      = EXCHANGE_NAME,
#         exchange_type = "direct",
#         durable       = True,
#     )
#     channel.queue_declare(queue=QUEUE_NAME, passive=True)
#     channel.queue_bind(
#         queue       = QUEUE_NAME,
#         exchange    = EXCHANGE_NAME,
#         routing_key = ROUTING_KEY,
#     )

#     channel.basic_qos(prefetch_count=PREFETCH_COUNT)
#     channel.basic_consume(
#         queue               = QUEUE_NAME,
#         on_message_callback = partial(_on_message, connection=connection),
#     )

#     log.info(
#         "[Consumer]  Waiting for messages on queue '%s' "
#         "(exchange='%s', routing='%s')  heartbeat=%ds  poll=%ds...",
#         QUEUE_NAME, EXCHANGE_NAME, ROUTING_KEY,
#         HEARTBEAT_INTERVAL, HEARTBEAT_POLL_INTERVAL,
#     )
#     channel.start_consuming()


# # ── Consumer startup with reconnect retry ─────────────────────────────────────

# def start() -> None:
#     """
#     Connect to RabbitMQ and start consuming.
#     Wrapped in a retry loop — if the connection drops, the consumer waits
#     RECONNECT_DELAY seconds and reconnects automatically.
#     """
#     while True:
#         try:
#             _connect_and_consume()
#         except KeyboardInterrupt:
#             log.info("[Consumer]  Interrupted by user — shutting down.")
#             break
#         except Exception as exc:
#             log.error(
#                 "[Consumer]  Connection lost: %s — reconnecting in %ds...",
#                 exc, RECONNECT_DELAY,
#             )
#             time.sleep(RECONNECT_DELAY)


# if __name__ == "__main__":
#     start()



"""
consumer.py
───────────
RabbitMQ consumer for the Journey-based analysis workflow.

Processing flow
───────────────
 1. Consume message from 'analysis.jobs'.
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
"""

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