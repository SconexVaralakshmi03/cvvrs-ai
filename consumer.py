# # # """
# # # consumer.py
# # # ───────────
# # # RabbitMQ consumer for the Journey-based analysis workflow.

# # # Processing flow (Step numbers match the spec)
# # # ─────────────────────────────────────────────
# # #  1. Consume message from 'analysis.jobs'.
# # #  2. Download all videos from S3.
# # #  3. Send PROCESSING progress (10 %).
# # #  4. Analyze videos via the existing AI pipeline (analyzer.py).
# # #  5. Violation frames are uploaded inside analyzer.py (S3 keys returned).
# # #  6-7. VideoResult / ViolationResult objects built inside analyzer.py.
# # #  8. Calculate overall processing time (wall-clock ms).
# # #  9. Send completion callback.
# # # 10. Acknowledge RabbitMQ message ONLY after successful completion callback.

# # # On any unhandled exception:
# # #     • Log the full traceback.
# # #     • Send a failure callback to Spring Boot.
# # #     • Nack the message (do not requeue — leave it for the dead-letter exchange).
# # # """

# # # from __future__ import annotations

# # # import json
# # # import logging
# # # import os
# # # import tempfile
# # # import traceback
# # # import time
# # # from typing import Dict, List

# # # import pika
# # # from dotenv import load_dotenv

# # # from analyzer import analyze_journey
# # # from callback_client import send_completed, send_failed, send_progress
# # # from models import AnalysisJobMessage, CompletionPayload, VideoJob
# # # from s3_service import download_video

# # # # ── Config / credentials ─────────────────────────────────────────────────────
# # # _ENV_PATH = os.path.join(
# # #     os.path.dirname(os.path.abspath(__file__)),
# # #     "config", "credentials.env",
# # # )
# # # load_dotenv(_ENV_PATH)

# # # RABBITMQ_URL   = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
# # # QUEUE_NAME     = os.environ.get("ANALYSIS_QUEUE", "analysis.jobs")
# # # PREFETCH_COUNT = int(os.environ.get("RABBITMQ_PREFETCH", "1"))

# # # logging.basicConfig(
# # #     level   = logging.INFO,
# # #     format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
# # # )
# # # log = logging.getLogger("consumer")


# # # # ── Job handler ───────────────────────────────────────────────────────────────

# # # def _handle_job(msg: AnalysisJobMessage, channel, method) -> None:
# # #     """
# # #     Full processing flow for one AnalysisJobMessage.
# # #     Acks on success, nacks (no requeue) on failure.
# # #     """
# # #     job_id     = msg.job_id
# # #     journey_id = msg.journey_id
# # #     tmp_paths: Dict[int, str] = {}

# # #     log.info("[Job %s]  journey=%d  videos=%d", job_id, journey_id, len(msg.videos))

# # #     try:
# # #         # ── Step 2: Download all videos from S3 ──────────────────────────────
# # #         log.info("[Job %s]  Downloading %d video(s)…", job_id, len(msg.videos))
# # #         send_progress(job_id, journey_id, 5, "Downloading videos")

# # #         for vj in sorted(msg.videos, key=lambda v: v.sequence_no):
# # #             suffix   = os.path.splitext(vj.s3_key)[1] or ".mp4"
# # #             fd, path = tempfile.mkstemp(suffix=suffix)
# # #             os.close(fd)
# # #             download_video(vj.s3_key, path)
# # #             tmp_paths[vj.video_id] = path
# # #             log.info("[Job %s]  Downloaded video_id=%d  seq=%d  → %s",
# # #                      job_id, vj.video_id, vj.sequence_no, path)

# # #         # ── Step 3: Progress callback after download ──────────────────────────
# # #         send_progress(job_id, journey_id, 10, "Downloading videos complete — starting analysis")

# # #         # ── Steps 4-7: Analyze + upload frames + build results ────────────────
# # #         def _progress(pct: int, message: str) -> None:
# # #             try:
# # #                 send_progress(job_id, journey_id, pct, message)
# # #             except Exception as exc:
# # #                 log.warning("[Job %s]  progress callback failed (non-fatal): %s", job_id, exc)

# # #         video_results, wall_seconds = analyze_journey(
# # #             job_id     = job_id,
# # #             journey_id = journey_id,
# # #             video_jobs = msg.videos,
# # #             tmp_paths  = tmp_paths,
# # #             progress_cb = _progress,
# # #         )

# # #         # ── Step 8: Calculate processing time (ms) ────────────────────────────
# # #         processing_time_ms = int(wall_seconds * 1000)
# # #         log.info("[Job %s]  Analysis complete.  videos=%d  violations=%d  time=%dms",
# # #                  job_id,
# # #                  len(video_results),
# # #                  sum(len(vr.violations) for vr in video_results),
# # #                  processing_time_ms)

# # #         # ── Step 9: Send completion callback ─────────────────────────────────
# # #         send_progress(job_id, journey_id, 95, "Sending results to backend")

# # #         completion = CompletionPayload(
# # #             job_id          = job_id,
# # #             journey_id      = journey_id,
# # #             processing_time = processing_time_ms,
# # #             video_results   = video_results,
# # #         )
# # #         send_completed(completion.to_dict())
# # #         log.info("[Job %s]  Completion callback sent.", job_id)

# # #         # ── Step 10: Acknowledge RabbitMQ message ─────────────────────────────
# # #         channel.basic_ack(delivery_tag=method.delivery_tag)
# # #         log.info("[Job %s]  Message acknowledged.", job_id)

# # #     except Exception as exc:
# # #         # ── Error handling ────────────────────────────────────────────────────
# # #         err_detail = traceback.format_exc()
# # #         log.error("[Job %s]  FAILED:\n%s", job_id, err_detail)

# # #         send_failed(job_id, journey_id, f"{exc}\n\n{err_detail}")

# # #         # Nack without requeue — failed jobs go to the dead-letter exchange.
# # #         # Change requeue=True if you want automatic retry.
# # #         channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
# # #         log.warning("[Job %s]  Message nacked (no requeue).", job_id)

# # #     finally:
# # #         # Always clean up temp files
# # #         _cleanup(tmp_paths)


# # # def _cleanup(tmp_paths: Dict[int, str]) -> None:
# # #     for path in tmp_paths.values():
# # #         try:
# # #             if os.path.isfile(path):
# # #                 os.remove(path)
# # #         except OSError as exc:
# # #             log.warning("Could not remove temp file %s: %s", path, exc)


# # # # ── RabbitMQ callback ─────────────────────────────────────────────────────────

# # # def _on_message(channel, method, properties, body):
# # #     try:
# # #         data = json.loads(body)
# # #         msg  = AnalysisJobMessage.from_dict(data)
# # #     except Exception as exc:
# # #         log.error("Failed to parse RabbitMQ message: %s\nBody: %s", exc, body[:500])
# # #         # Malformed messages are nacked without requeue
# # #         channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
# # #         return

# # #     _handle_job(msg, channel, method)


# # # # ── Consumer startup ──────────────────────────────────────────────────────────

# # # def start() -> None:
# # #     """
# # #     Connect to RabbitMQ and start consuming from the analysis.jobs queue.
# # #     Blocks forever (until interrupted).
# # #     """
# # #     log.info("[Consumer]  Connecting to RabbitMQ:  %s", RABBITMQ_URL)
# # #     params     = pika.URLParameters(RABBITMQ_URL)
# # #     connection = pika.BlockingConnection(params)
# # #     channel    = connection.channel()

# # #     channel.queue_declare(queue=QUEUE_NAME, durable=True)
# # #     channel.basic_qos(prefetch_count=PREFETCH_COUNT)
# # #     channel.basic_consume(queue=QUEUE_NAME, on_message_callback=_on_message)

# # #     log.info("[Consumer]  Waiting for messages on queue '%s'…", QUEUE_NAME)
# # #     try:
# # #         channel.start_consuming()
# # #     except KeyboardInterrupt:
# # #         log.info("[Consumer]  Interrupted — shutting down.")
# # #         channel.stop_consuming()
# # #     finally:
# # #         connection.close()


# # # if __name__ == "__main__":
# # #     start()
# # """
# # consumer.py
# # ───────────
# # RabbitMQ consumer for the Journey-based analysis workflow.

# # Processing flow (Step numbers match the spec)
# # ─────────────────────────────────────────────
# #  1. Consume message from 'analysis.jobs'.
# #  2. Download all videos from S3.
# #  3. Send PROCESSING progress (10 %).
# #  4. Analyze videos via the existing AI pipeline (analyzer.py).
# #  5. Violation frames are uploaded inside analyzer.py (S3 keys returned).
# #  6-7. VideoResult / ViolationResult objects built inside analyzer.py.
# #  8. Calculate overall processing time (wall-clock ms).
# #  9. Send completion callback.
# # 10. Acknowledge RabbitMQ message ONLY after successful completion callback.

# # On any unhandled exception:
# #     • Log the full traceback.
# #     • Send a failure callback to Spring Boot.
# #     • Nack the message (do not requeue — leave it for the dead-letter exchange).
# # """

# # from __future__ import annotations

# # import json
# # import logging
# # import os
# # import tempfile
# # import traceback
# # import time
# # from typing import Dict, List

# # import pika
# # from dotenv import load_dotenv

# # from analyzer import analyze_journey
# # from callback_client import send_completed, send_failed, send_progress
# # from models import AnalysisJobMessage, CompletionPayload, VideoJob
# # from s3_service import download_video

# # # ── Config / credentials ─────────────────────────────────────────────────────
# # _ENV_PATH = os.path.join(
# #     os.path.dirname(os.path.abspath(__file__)),
# #     "config", "credentials.env",
# # )
# # load_dotenv(_ENV_PATH)

# # RABBITMQ_URL   = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
# # QUEUE_NAME     = os.environ.get("ANALYSIS_QUEUE", "analysis.jobs")
# # PREFETCH_COUNT = int(os.environ.get("RABBITMQ_PREFETCH", "1"))

# # logging.basicConfig(
# #     level   = logging.INFO,
# #     format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
# # )
# # log = logging.getLogger("consumer")


# # # ── Job handler ───────────────────────────────────────────────────────────────

# # def _handle_job(msg: AnalysisJobMessage, channel, method) -> None:
# #     """
# #     Full processing flow for one AnalysisJobMessage.
# #     Acks on success, nacks (no requeue) on failure.
# #     """
# #     job_id     = msg.job_id
# #     journey_id = msg.journey_id
# #     tmp_paths: Dict[int, str] = {}

# #     log.info("[Job %s]  journey=%d  videos=%d", job_id, journey_id, len(msg.videos))

# #     try:
# #         # ── Step 2: Download all videos from S3 ──────────────────────────────
# #         log.info("[Job %s]  Downloading %d video(s)…", job_id, len(msg.videos))
# #         send_progress(job_id, journey_id, 5, "Downloading videos")

# #         for vj in sorted(msg.videos, key=lambda v: v.sequence_no):
# #             suffix   = os.path.splitext(vj.s3_key)[1] or ".mp4"
# #             fd, path = tempfile.mkstemp(suffix=suffix)
# #             os.close(fd)
# #             download_video(vj.s3_key, path)
# #             tmp_paths[vj.video_id] = path
# #             log.info("[Job %s]  Downloaded video_id=%d  seq=%d  → %s",
# #                      job_id, vj.video_id, vj.sequence_no, path)

# #         # ── Step 3: Progress callback after download ──────────────────────────
# #         send_progress(job_id, journey_id, 10, "Downloading videos complete — starting analysis")

# #         # ── Steps 4-7: Analyze + upload frames + build results ────────────────
# #         def _progress(pct: int, message: str) -> None:
# #             try:
# #                 send_progress(job_id, journey_id, pct, message)
# #             except Exception as exc:
# #                 log.warning("[Job %s]  progress callback failed (non-fatal): %s", job_id, exc)

# #         video_results, wall_seconds = analyze_journey(
# #             job_id     = job_id,
# #             journey_id = journey_id,
# #             video_jobs = msg.videos,
# #             tmp_paths  = tmp_paths,
# #             progress_cb = _progress,
# #         )

# #         # ── Step 8: Calculate processing time (ms) ────────────────────────────
# #         processing_time_ms = int(wall_seconds * 1000)
# #         log.info("[Job %s]  Analysis complete.  videos=%d  violations=%d  time=%dms",
# #                  job_id,
# #                  len(video_results),
# #                  sum(len(vr.violations) for vr in video_results),
# #                  processing_time_ms)

# #         # ── Step 9: Send completion callback ─────────────────────────────────
# #         send_progress(job_id, journey_id, 95, "Sending results to backend")

# #         completion = CompletionPayload(
# #             job_id          = job_id,
# #             journey_id      = journey_id,
# #             processing_time = processing_time_ms,
# #             video_results   = video_results,
# #         )
# #         send_completed(completion.to_dict())
# #         log.info("[Job %s]  Completion callback sent.", job_id)

# #         # ── Step 10: Acknowledge RabbitMQ message ─────────────────────────────
# #         channel.basic_ack(delivery_tag=method.delivery_tag)
# #         log.info("[Job %s]  Message acknowledged.", job_id)

# #     except Exception as exc:
# #         # ── Error handling ────────────────────────────────────────────────────
# #         err_detail = traceback.format_exc()
# #         log.error("[Job %s]  FAILED:\n%s", job_id, err_detail)

# #         send_failed(job_id, journey_id, f"{exc}\n\n{err_detail}")

# #         # Nack without requeue — failed jobs go to the dead-letter exchange.
# #         # Change requeue=True if you want automatic retry.
# #         channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
# #         log.warning("[Job %s]  Message nacked (no requeue).", job_id)

# #     finally:
# #         # Always clean up temp files
# #         _cleanup(tmp_paths)


# # def _cleanup(tmp_paths: Dict[int, str]) -> None:
# #     for path in tmp_paths.values():
# #         try:
# #             if os.path.isfile(path):
# #                 os.remove(path)
# #         except OSError as exc:
# #             log.warning("Could not remove temp file %s: %s", path, exc)


# # # ── RabbitMQ callback ─────────────────────────────────────────────────────────

# # def _on_message(channel, method, properties, body):
# #     log.info("[RabbitMQ] Raw message received: %s", body.decode("utf-8", errors="replace")[:1000])
# #     try:
# #         data = json.loads(body)
# #         msg  = AnalysisJobMessage.from_dict(data)
# #     except Exception as exc:
# #         log.error("Failed to parse RabbitMQ message: %s\nBody: %s", exc, body[:500])
# #         # Malformed messages are nacked without requeue
# #         channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
# #         return

# #     _handle_job(msg, channel, method)


# # # ── Consumer startup ──────────────────────────────────────────────────────────

# # def start() -> None:
# #     """
# #     Connect to RabbitMQ and start consuming from the analysis.jobs queue.
# #     Blocks forever (until interrupted).
# #     """
# #     log.info("[Consumer]  Connecting to RabbitMQ:  %s", RABBITMQ_URL)
# #     params     = pika.URLParameters(RABBITMQ_URL)
# #     connection = pika.BlockingConnection(params)
# #     channel    = connection.channel()

# #     channel.queue_declare(queue=QUEUE_NAME, durable=True)
# #     channel.basic_qos(prefetch_count=PREFETCH_COUNT)
# #     channel.basic_consume(queue=QUEUE_NAME, on_message_callback=_on_message)

# #     log.info("[Consumer]  Waiting for messages on queue '%s'…", QUEUE_NAME)
# #     try:
# #         channel.start_consuming()
# #     except KeyboardInterrupt:
# #         log.info("[Consumer]  Interrupted — shutting down.")
# #         channel.stop_consuming()
# #     finally:
# #         connection.close()


# # if __name__ == "__main__":
# #     start()


# """
# consumer.py
# ───────────
# RabbitMQ consumer for the Journey-based analysis workflow.

# Live backend : https://cvvrsrailway-api.sconexsoft.com/cvs

# Processing flow (Step numbers match the CVVRS spec)
# ────────────────────────────────────────────────────
#  1. Consume message from 'analysis.jobs' (exchange: analysis.exchange,
#     routing key: analysis.job.created).
#  2. Download all videos from S3.
#  3. Send PROCESSING progress (5 %) — "Downloading videos".
#  4. Analyze videos via the existing AI pipeline (analyzer.py).
#  5. Violation frames are uploaded inside analyzer.py (S3 keys returned).
#  6–7. VideoResult / ViolationResult objects built inside analyzer.py.
#  8. Calculate overall processing time (wall-clock ms).
#  9. Send completion callback (POST /api/internal/analysis/completed).
# 10. Acknowledge RabbitMQ message ONLY after successful completion callback.

# On any unhandled exception
# ──────────────────────────
#   • Log the full traceback.
#   • Send a failure callback (POST /api/internal/analysis/failed).
#   • Nack the message (no requeue — dead-letter exchange).

# Changes from previous version
# ──────────────────────────────
# • AnalysisJobMessage now carries trainDetailId, folderName, priority —
#   all forwarded to CompletionPayload.
# • analyze_journey() receives folder_name so frames land in the right S3 path.
# • _progress() passes current_video index to send_progress().
# • CompletionPayload is built with trainDetailId and folderName.
# • Queue binding is declared with the exchange / routing key from the docs.
# """

# from __future__ import annotations

# import json
# import logging
# import os
# import tempfile
# import traceback
# import time
# from typing import Dict

# import pika
# from dotenv import load_dotenv

# from analyzer import analyze_journey
# from callback_client import send_completed, send_failed, send_progress
# from models import AnalysisJobMessage, CompletionPayload
# from s3_service import download_video

# # ── Config / credentials ─────────────────────────────────────────────────────
# _ENV_PATH = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)),
#     "config", "credentials.env",
# )
# load_dotenv(_ENV_PATH)

# RABBITMQ_URL   = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
# QUEUE_NAME     = os.environ.get("ANALYSIS_QUEUE",    "analysis.jobs")
# EXCHANGE_NAME  = os.environ.get("ANALYSIS_EXCHANGE", "analysis.exchange")
# ROUTING_KEY    = os.environ.get("ANALYSIS_ROUTING",  "analysis.job.created")
# PREFETCH_COUNT = int(os.environ.get("RABBITMQ_PREFETCH", "1"))

# logging.basicConfig(
#     level  = logging.INFO,
#     format = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
# )
# log = logging.getLogger("consumer")


# # ── Job handler ───────────────────────────────────────────────────────────────

# def _handle_job(msg: AnalysisJobMessage, channel, method) -> None:
#     """
#     Full processing flow for one AnalysisJobMessage.
#     Acks on success, nacks (no requeue) on failure.
#     """
#     job_id          = msg.job_id
#     journey_id      = msg.journey_id
#     train_detail_id = msg.train_detail_id
#     folder_name     = msg.folder_name
#     tmp_paths: Dict[int, str] = {}

#     log.info(
#         "[Job %s]  journey=%d  trainDetail=%d  videos=%d  folder=%s",
#         job_id, journey_id, train_detail_id, len(msg.videos), folder_name,
#     )

#     try:
#         # ── Step 2: Download all videos from S3 ──────────────────────────────
#         log.info("[Job %s]  Downloading %d video(s)…", job_id, len(msg.videos))
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

#         # ── Steps 4–7: Analyze + upload frames + build results ────────────────
#         def _progress(pct: int, message: str, current_video: int = 1) -> None:
#             try:
#                 send_progress(job_id, journey_id, pct, message,
#                               current_video=current_video)
#             except Exception as exc:
#                 log.warning(
#                     "[Job %s]  progress callback failed (non-fatal): %s", job_id, exc
#                 )

#         video_results, wall_seconds = analyze_journey(
#             job_id      = job_id,
#             journey_id  = journey_id,
#             folder_name = folder_name,
#             video_jobs  = msg.videos,
#             tmp_paths   = tmp_paths,
#             progress_cb = _progress,
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
#             train_detail_id = train_detail_id,   # NEW
#             folder_name     = folder_name,        # NEW
#             processing_time = processing_time_ms,
#             video_results   = video_results,
#         )
#         send_completed(completion.to_dict())
#         log.info("[Job %s]  Completion callback sent.", job_id)

#         # ── Step 10: Acknowledge RabbitMQ message ─────────────────────────────
#         channel.basic_ack(delivery_tag=method.delivery_tag)
#         log.info("[Job %s]  Message acknowledged.", job_id)

#     except Exception as exc:
#         err_detail = traceback.format_exc()
#         log.error("[Job %s]  FAILED:\n%s", job_id, err_detail)

#         # POST /api/internal/analysis/failed — only jobId + errorMessage
#         send_failed(job_id, journey_id, f"{exc}\n\n{err_detail}")

#         # Nack without requeue — failed jobs go to dead-letter exchange.
#         channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
#         log.warning("[Job %s]  Message nacked (no requeue).", job_id)

#     finally:
#         _cleanup(tmp_paths)


# def _cleanup(tmp_paths: Dict[int, str]) -> None:
#     for path in tmp_paths.values():
#         try:
#             if os.path.isfile(path):
#                 os.remove(path)
#         except OSError as exc:
#             log.warning("Could not remove temp file %s: %s", path, exc)


# # ── RabbitMQ callback ─────────────────────────────────────────────────────────

# def _on_message(channel, method, properties, body):
#     log.info(
#         "[RabbitMQ] Message received: %s",
#         body.decode("utf-8", errors="replace")[:500],
#     )
#     try:
#         data = json.loads(body)
#         msg  = AnalysisJobMessage.from_dict(data)
#     except Exception as exc:
#         log.error("Failed to parse RabbitMQ message: %s\nBody: %s", exc, body[:500])
#         channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
#         return

#     _handle_job(msg, channel, method)


# # ── Consumer startup ──────────────────────────────────────────────────────────

# def start() -> None:
#     """
#     Connect to RabbitMQ and start consuming from the analysis.jobs queue.
#     Declares the exchange and binding so the worker can start independently
#     of the Spring Boot startup order.
#     Blocks forever (until interrupted).
#     """
#     log.info("[Consumer]  Connecting to RabbitMQ: %s", RABBITMQ_URL)
#     params     = pika.URLParameters(RABBITMQ_URL)
#     connection = pika.BlockingConnection(params)
#     channel    = connection.channel()

#     # Declare exchange + queue + binding (idempotent — safe to call on restart)
#     channel.exchange_declare(
#         exchange      = EXCHANGE_NAME,
#         exchange_type = "direct",
#         durable       = True,
#     )
#     channel.queue_declare(queue=QUEUE_NAME, durable=True)
#     channel.queue_bind(
#         queue       = QUEUE_NAME,
#         exchange    = EXCHANGE_NAME,
#         routing_key = ROUTING_KEY,
#     )

#     channel.basic_qos(prefetch_count=PREFETCH_COUNT)
#     channel.basic_consume(queue=QUEUE_NAME, on_message_callback=_on_message)

#     log.info(
#         "[Consumer]  Waiting for messages on queue '%s' "
#         "(exchange='%s', routing='%s')…",
#         QUEUE_NAME, EXCHANGE_NAME, ROUTING_KEY,
#     )
#     try:
#         channel.start_consuming()
#     except KeyboardInterrupt:
#         log.info("[Consumer]  Interrupted — shutting down.")
#         channel.stop_consuming()
#     finally:
#         connection.close()


# if __name__ == "__main__":
#     start()



"""
consumer.py
───────────
RabbitMQ consumer for the Journey-based analysis workflow.
 
Live backend : https://cvvrsrailway-api.sconexsoft.com/cvs
 
Processing flow (Step numbers match the CVVRS spec)
────────────────────────────────────────────────────
1. Consume message from 'analysis.jobs' (exchange: analysis.exchange,
    routing key: analysis.job.created).
2. Download all videos from S3.
3. Send PROCESSING progress (5%) — "Downloading videos".
4. Analyze videos via the existing AI pipeline (analyzer.py).
5. Violation frames are uploaded inside analyzer.py (S3 keys returned).
6–7. VideoResult / ViolationResult objects built inside analyzer.py.
8. Calculate overall processing time (wall-clock ms).
9. Send completion callback (POST /api/internal/analysis/completed).
10. Acknowledge RabbitMQ message ONLY after successful completion callback.
 
On any unhandled exception
──────────────────────────
  • Log the full traceback.
  • Send a failure callback (POST /api/internal/analysis/failed).
  • Nack the message (no requeue — dead-letter exchange).
 
Fixes from previous version
─────────────────────────────
• FIX: Reads callbackBaseUrl from RabbitMQ message and overrides the
  callback client base URL dynamically — so Python calls the correct
  Spring Boot server without needing env var changes per environment.
• FIX: Added reconnect retry loop — if RabbitMQ restarts while Python
  is running, the consumer reconnects automatically after 5 seconds
  instead of crashing.
• IMPORTANT: Make sure config/credentials.env has:
      ANALYSIS_QUEUE=analysis.jobs
  NOT analysis.queue (that was the old queue name causing the mismatch).
"""
 
from __future__ import annotations
 
import json
import logging
import os
import tempfile
import time
import traceback
from typing import Dict
 
import pika
from dotenv import load_dotenv
 
from analyzer import analyze_journey
from callback_client import send_completed, send_failed, send_progress, set_base_url
from models import AnalysisJobMessage, CompletionPayload
from s3_service import download_video
 
# ── Config / credentials ─────────────────────────────────────────────────────
_ENV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "config", "credentials.env",
)
load_dotenv(_ENV_PATH)
 
RABBITMQ_URL   = os.environ.get("RABBITMQ_URL",        "amqp://guest:guest@localhost:5672/")
QUEUE_NAME     = os.environ.get("ANALYSIS_QUEUE",       "analysis.jobs")   # must be analysis.jobs
EXCHANGE_NAME  = os.environ.get("ANALYSIS_EXCHANGE",    "analysis.exchange")
ROUTING_KEY    = os.environ.get("ANALYSIS_ROUTING",     "analysis.job.created")
PREFETCH_COUNT = int(os.environ.get("RABBITMQ_PREFETCH", "1"))
RECONNECT_DELAY = int(os.environ.get("RECONNECT_DELAY_SECONDS", "5"))
 
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger("consumer")
 
 
# ── Job handler ───────────────────────────────────────────────────────────────
 
def _handle_job(msg: AnalysisJobMessage, channel, method) -> None:
    """
    Full processing flow for one AnalysisJobMessage.
    Acks on success, nacks (no requeue) on failure.
    """
    job_id          = msg.job_id
    journey_id      = msg.journey_id
    train_detail_id = msg.train_detail_id
    folder_name     = msg.folder_name
    tmp_paths: Dict[int, str] = {}
 
    log.info(
        "[Job %s]  journey=%d  trainDetail=%d  videos=%d  folder=%s",
        job_id, journey_id, train_detail_id, len(msg.videos), folder_name,
    )
 
    # FIX: Override callback URL from the message so Python calls the
    # correct Spring Boot server regardless of env var settings.
    if msg.callback_base_url:
        log.info("[Job %s]  callbackBaseUrl=%s", job_id, msg.callback_base_url)
        set_base_url(msg.callback_base_url)
 
    try:
        # ── Step 2: Download all videos from S3 ──────────────────────────────
        log.info("[Job %s]  Downloading %d video(s)...", job_id, len(msg.videos))
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
 
        # ── Step 3: Progress callback after download ──────────────────────────
        send_progress(
            job_id, journey_id, 10,
            "Downloads complete — starting analysis",
            current_video=1,
        )
 
        # ── Steps 4–7: Analyze + upload frames + build results ────────────────
        def _progress(pct: int, message: str, current_video: int = 1) -> None:
            try:
                send_progress(job_id, journey_id, pct, message,
                              current_video=current_video)
            except Exception as exc:
                log.warning(
                    "[Job %s]  progress callback failed (non-fatal): %s", job_id, exc
                )
 
        video_results, wall_seconds = analyze_journey(
            job_id      = job_id,
            journey_id  = journey_id,
            folder_name = folder_name,
            video_jobs  = msg.videos,
            tmp_paths   = tmp_paths,
            progress_cb = _progress,
        )
 
        # ── Step 8: Calculate processing time (ms) ────────────────────────────
        processing_time_ms = int(wall_seconds * 1000)
        log.info(
            "[Job %s]  Analysis complete.  videos=%d  violations=%d  time=%dms",
            job_id,
            len(video_results),
            sum(len(vr.violations) for vr in video_results),
            processing_time_ms,
        )
 
        # ── Step 9: Send completion callback ──────────────────────────────────
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
 
        # ── Step 10: Acknowledge RabbitMQ message ─────────────────────────────
        channel.basic_ack(delivery_tag=method.delivery_tag)
        log.info("[Job %s]  Message acknowledged.", job_id)
 
    except Exception as exc:
        err_detail = traceback.format_exc()
        log.error("[Job %s]  FAILED:\n%s", job_id, err_detail)
 
        # POST /api/internal/analysis/failed — only jobId + errorMessage
        send_failed(job_id, journey_id, f"{exc}\n\n{err_detail}")
 
        # Nack without requeue — failed jobs go to dead-letter exchange.
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        log.warning("[Job %s]  Message nacked (no requeue).", job_id)
 
    finally:
        _cleanup(tmp_paths)
 
 
def _cleanup(tmp_paths: Dict[int, str]) -> None:
    for path in tmp_paths.values():
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError as exc:
            log.warning("Could not remove temp file %s: %s", path, exc)
 
 
# ── RabbitMQ callback ─────────────────────────────────────────────────────────
 
def _on_message(channel, method, properties, body):
    log.info(
        "[RabbitMQ] Message received: %s",
        body.decode("utf-8", errors="replace")[:500],
    )
    try:
        data = json.loads(body)
        msg  = AnalysisJobMessage.from_dict(data)
    except Exception as exc:
        log.error("Failed to parse RabbitMQ message: %s\nBody: %s", exc, body[:500])
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        return
 
    _handle_job(msg, channel, method)
 
 
# ── Consumer connect + consume ────────────────────────────────────────────────
 
def _connect_and_consume() -> None:
    """
    Connect to RabbitMQ and start consuming.
    Raises on connection failure so the retry loop can catch it.
    """
    log.info("[Consumer]  Connecting to RabbitMQ: %s  queue=%s", RABBITMQ_URL, QUEUE_NAME)
    params     = pika.URLParameters(RABBITMQ_URL)
    connection = pika.BlockingConnection(params)
    channel    = connection.channel()
 
    # Declare exchange + queue + binding (idempotent — safe to call on restart)
    channel.exchange_declare(
        exchange      = EXCHANGE_NAME,
        exchange_type = "direct",
        durable       = True,
    )
    channel.queue_declare(
    queue=QUEUE_NAME,
    passive=True
)
    channel.queue_bind(
        queue       = QUEUE_NAME,
        exchange    = EXCHANGE_NAME,
        routing_key = ROUTING_KEY,
    )
 
    channel.basic_qos(prefetch_count=PREFETCH_COUNT)
    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=_on_message)
 
    log.info(
        "[Consumer]  Waiting for messages on queue '%s' "
        "(exchange='%s', routing='%s')...",
        QUEUE_NAME, EXCHANGE_NAME, ROUTING_KEY,
    )
    channel.start_consuming()
 
 
# ── Consumer startup with reconnect retry ─────────────────────────────────────
 
def start() -> None:
    """
    Connect to RabbitMQ and start consuming.
 
    FIX: Wrapped in retry loop — if RabbitMQ restarts or the connection
    drops, the consumer waits RECONNECT_DELAY_SECONDS and reconnects
    automatically instead of crashing and requiring manual restart.
    """
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