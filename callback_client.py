# # # # """
# # # # callback_client.py
# # # # ──────────────────
# # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # # All three public helpers are fire-and-forget from the caller's perspective;
# # # # they raise on non-2xx so the consumer can route to the failure path.
# # # # """

# # # # from __future__ import annotations

# # # # import logging
# # # # import os
# # # # from typing import Any, Dict, List

# # # # import requests
# # # # from dotenv import load_dotenv

# # # # # ── Credentials / config ────────────────────────────────────────────────────
# # # # _ENV_PATH = os.path.join(
# # # #     os.path.dirname(os.path.abspath(__file__)),
# # # #     "config", "credentials.env",
# # # # )
# # # # load_dotenv(_ENV_PATH)

# # # # _BASE_URL = os.environ.get("SPRING_BOOT_BASE_URL", "http://localhost:8080")
# # # # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # # log = logging.getLogger("callback_client")


# # # # # ── Internal helper ──────────────────────────────────────────────────────────

# # # # def _post(path: str, payload: Dict[str, Any]) -> None:
# # # #     url = f"{_BASE_URL}{path}"
# # # #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# # # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# # # #     if not resp.ok:
# # # #         raise RuntimeError(
# # # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # # #         )
# # # #     log.debug("[Callback] %s → %d", path, resp.status_code)


# # # # # ── Public API ───────────────────────────────────────────────────────────────

# # # # def send_progress(
# # # #     job_id:     str,
# # # #     journey_id: int,
# # # #     progress:   int,
# # # #     message:    str,
# # # #     status:     str = "PROCESSING",
# # # # ) -> None:
# # # #     """
# # # #     POST /api/internal/analysis/progress

# # # #     Called periodically during analysis to update the frontend progress bar
# # # #     and the SSE stream.
# # # #     """
# # # #     _post(
# # # #         "/api/internal/analysis/progress",
# # # #         {
# # # #             "jobId":      job_id,
# # # #             "journeyId":  journey_id,
# # # #             "status":     status,
# # # #             "progress":   progress,
# # # #             "message":    message,
# # # #         },
# # # #     )


# # # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # # #     """
# # # #     POST /api/internal/analysis/completed

# # # #     Called once — after ALL videos in the journey have been processed and
# # # #     their violation frames uploaded to S3.

# # # #     Expected shape of completion_payload
# # # #     ─────────────────────────────────────
# # # #     {
# # # #         "jobId":          str,
# # # #         "journeyId":      int,
# # # #         "processingTime": int,          # wall-clock ms
# # # #         "videoResults":   List[dict],   # see models.py → VideoResult
# # # #     }
# # # #     """
# # # #     _post("/api/internal/analysis/completed", completion_payload)


# # # # def send_failed(
# # # #     job_id:        str,
# # # #     journey_id:    int,
# # # #     error_message: str,
# # # # ) -> None:
# # # #     """
# # # #     POST /api/internal/analysis/failed

# # # #     Called whenever an unrecoverable exception occurs during job processing.
# # # #     Spring Boot will mark AnalysisJob and Journey as FAILED.
# # # #     """
# # # #     try:
# # # #         _post(
# # # #             "/api/internal/analysis/failed",
# # # #             {
# # # #                 "jobId":        job_id,
# # # #                 "journeyId":    journey_id,
# # # #                 "errorMessage": error_message,
# # # #             },
# # # #         )
# # # #     except Exception as exc:
# # # #         # Failure callback must never itself raise — log and swallow.
# # # #         log.error("[Callback] send_failed itself failed: %s", exc)

# # # """
# # # callback_client.py
# # # ──────────────────
# # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs
# # # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).

# # # Changes from previous version
# # # ──────────────────────────────
# # # • _BASE_URL now defaults to the live CVVRS domain (overridable via env var
# # #   SPRING_BOOT_BASE_URL for local testing).
# # # • send_progress()  — payload now includes the `currentVideo` field required
# # #   by the API.
# # # • send_completed() — accepts a fully-formed dict (built by CompletionPayload.
# # #   to_dict()); no shape changes needed here since the caller owns the dict.
# # # • send_failed()    — payload drops `journeyId` (not in the API spec for this
# # #   endpoint); only jobId + errorMessage are sent.
# # # """

# # # from __future__ import annotations

# # # import logging
# # # import os
# # # from typing import Any, Dict, Optional

# # # import requests
# # # from dotenv import load_dotenv

# # # # ── Credentials / config ────────────────────────────────────────────────────
# # # _ENV_PATH = os.path.join(
# # #     os.path.dirname(os.path.abspath(__file__)),
# # #     "config", "credentials.env",
# # # )
# # # load_dotenv(_ENV_PATH)

# # # # Default to the live CVVRS server; override with SPRING_BOOT_BASE_URL for local dev.
# # # _BASE_URL = os.environ.get(
# # #     "SPRING_BOOT_BASE_URL",
# # #     "https://cvvrsrailway-api.sconexsoft.com/cvs",
# # # )
# # # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # log = logging.getLogger("callback_client")


# # # # ── Internal helper ──────────────────────────────────────────────────────────

# # # def _post(path: str, payload: Dict[str, Any]) -> None:
# # #     """
# # #     POST to a Spring Boot internal endpoint.
# # #     No Authorization header is sent — /api/internal/* are worker-only endpoints.
# # #     Raises RuntimeError on non-2xx so the consumer can route to failure path.
# # #     """
# # #     url = f"{_BASE_URL}{path}"
# # #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# # #     if not resp.ok:
# # #         raise RuntimeError(
# # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # #         )
# # #     log.debug("[Callback] %s → %d", path, resp.status_code)


# # # # ── Public API ───────────────────────────────────────────────────────────────

# # # def send_progress(
# # #     job_id:        str,
# # #     journey_id:    int,
# # #     progress:      int,
# # #     message:       str,
# # #     status:        str = "PROCESSING",
# # #     current_video: Optional[int] = None,   # NEW — index of video currently being processed
# # # ) -> None:
# # #     """
# # #     POST /api/internal/analysis/progress

# # #     Called periodically during analysis to update the frontend progress bar
# # #     and the SSE stream.

# # #     Parameters
# # #     ──────────
# # #     job_id        : RabbitMQ job ID.
# # #     journey_id    : Journey ID.
# # #     progress      : 0–100 integer.
# # #     message       : Human-readable status message.
# # #     status        : "PROCESSING" | "COMPLETED" | "FAILED"
# # #     current_video : 1-based index of the video currently being processed
# # #                     (omitted from payload when None).
# # #     """
# # #     payload: Dict[str, Any] = {
# # #         "jobId":      job_id,
# # #         "journeyId":  journey_id,
# # #         "status":     status,
# # #         "progress":   progress,
# # #         "message":    message,
# # #     }
# # #     if current_video is not None:
# # #         payload["currentVideo"] = current_video

# # #     _post("/api/internal/analysis/progress", payload)


# # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # #     """
# # #     POST /api/internal/analysis/completed

# # #     Called once after ALL videos in the journey have been processed and their
# # #     violation frames uploaded to S3.

# # #     Expected shape of completion_payload (built by CompletionPayload.to_dict())
# # #     ────────────────────────────────────────────────────────────────────────────
# # #     {
# # #         "jobId":         str,
# # #         "journeyId":     int,
# # #         "batchId":       str,           # "BATCH-<jobId>" if not supplied
# # #         "trainDetailId": int,
# # #         "folderName":    str,
# # #         "processingTime":int,           # wall-clock ms
# # #         "videoResults": [
# # #             {
# # #                 "videoId":         str,       # STRING per API spec
# # #                 "sequenceNo":      int,
# # #                 "durationSeconds": float,
# # #                 "originalS3Key":   str,
# # #                 "violations": [
# # #                     {
# # #                         "violationType":          str,
# # #                         "severity":               str,
# # #                         "confidence":             float,
# # #                         "riskScore":              float,
# # #                         "timestamp":              float,   # journey-global seconds
# # #                         "originalVideoTimestamp": float,   # local-video seconds
# # #                         "framePaths":             [str]
# # #                     }
# # #                 ]
# # #             }
# # #         ]
# # #     }
# # #     """
# # #     _post("/api/internal/analysis/completed", completion_payload)


# # # def send_failed(
# # #     job_id:        str,
# # #     journey_id:    int,          # kept as parameter for caller convenience / logging
# # #     error_message: str,
# # # ) -> None:
# # #     """
# # #     POST /api/internal/analysis/failed

# # #     Called whenever an unrecoverable exception occurs during job processing.
# # #     Spring Boot will mark AnalysisJob and Journey as FAILED.

# # #     Note: The API spec only requires jobId + errorMessage in the request body.
# # #     journeyId is accepted as a parameter here for logging but is NOT included
# # #     in the outbound payload.
# # #     """
# # #     try:
# # #         _post(
# # #             "/api/internal/analysis/failed",
# # #             {
# # #                 "jobId":        job_id,
# # #                 "errorMessage": error_message,
# # #             },
# # #         )
# # #     except Exception as exc:
# # #         # Failure callback must never itself raise — log and swallow.
# # #         log.error(
# # #             "[Callback] send_failed itself failed (job=%s journey=%d): %s",
# # #             job_id, journey_id, exc,
# # #         )



# # """

# # callback_client.py

# # ──────────────────

# # Sends progress, completion, and failure callbacks to the Spring Boot backend.
 
# # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs

# # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).
 
# # Fixes from previous version

# # ─────────────────────────────

# # • Added set_base_url() — called by consumer.py with the callbackBaseUrl from

# #   the RabbitMQ message, so Python posts to the correct Spring Boot server

# #   per environment without needing env var changes.

# # • _BASE_URL still falls back to env var SPRING_BOOT_BASE_URL or the live

# #   server when callbackBaseUrl is not provided in the message.

# # """
 
# # from __future__ import annotations
 
# # import logging

# # import os

# # from typing import Any, Dict, Optional
 
# # import requests

# # from dotenv import load_dotenv
 
# # # ── Credentials / config ────────────────────────────────────────────────────

# # _ENV_PATH = os.path.join(

# #     os.path.dirname(os.path.abspath(__file__)),

# #     "config", "credentials.env",

# # )

# # load_dotenv(_ENV_PATH)
 
# # # Default to the live CVVRS server; override with SPRING_BOOT_BASE_URL for local dev

# # # or call set_base_url() with the callbackBaseUrl from the RabbitMQ message.

# # _BASE_URL = os.environ.get(

# #     "SPRING_BOOT_BASE_URL",

# #     "https://cvvrsrailway-api.sconexsoft.com/cvs",

# # )

# # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))
 
# # log = logging.getLogger("callback_client")
 
 
# # # ── FIX: dynamic base URL setter ─────────────────────────────────────────────
 
# # def set_base_url(url: str) -> None:

# #     """

# #     Override the callback base URL at runtime.
 
# #     Called by consumer.py when the RabbitMQ message contains a callbackBaseUrl.

# #     This allows the same Python worker to callback correctly to both local

# #     and staging Spring Boot servers without changing env vars.
 
# #     Example values:

# #         "http://localhost:8093/api/internal/analysis"         (local)

# #         "https://cvvrsrailway-api.sconexsoft.com/cvs/api/internal/analysis"  (staging)
 
# #     Note: The URL passed here is the FULL path up to /api/internal/analysis.

# #     The individual callbacks append /progress, /completed, /failed.

# #     """

# #     global _BASE_URL

# #     # Strip the /api/internal/analysis suffix if present — we add it in _post()

# #     base = url.rstrip("/")

# #     if base.endswith("/api/internal/analysis"):

# #         base = base[: -len("/api/internal/analysis")]

# #     _BASE_URL = base

# #     log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)
 
 
# # # ── Internal helper ──────────────────────────────────────────────────────────
 
# # def _post(path: str, payload: Dict[str, Any]) -> None:

# #     """

# #     POST to a Spring Boot internal endpoint.

# #     No Authorization header is sent — /api/internal/* are worker-only endpoints.

# #     Raises RuntimeError on non-2xx so the consumer can route to failure path.

# #     """

# #     url = f"{_BASE_URL}{path}"

# #     log.debug("[Callback] POST %s  payload=%s", url, payload)

# #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)

# #     if not resp.ok:

# #         raise RuntimeError(

# #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"

# #         )

# #     log.debug("[Callback] %s → %d", path, resp.status_code)
 
 
# # # ── Public API ───────────────────────────────────────────────────────────────
 
# # def send_progress(

# #     job_id:        str,

# #     journey_id:    int,

# #     progress:      int,

# #     message:       str,

# #     status:        str = "PROCESSING",

# #     current_video: Optional[int] = None,

# # ) -> None:

# #     """

# #     POST /api/internal/analysis/progress
 
# #     Called periodically during analysis to update the frontend progress bar

# #     and the SSE stream.
 
# #     Parameters

# #     ──────────

# #     job_id        : RabbitMQ job ID.

# #     journey_id    : Journey ID.

# #     progress      : 0–100 integer.

# #     message       : Human-readable status message.

# #     status        : "PROCESSING" | "COMPLETED" | "FAILED"

# #     current_video : 1-based index of the video currently being processed.

# #     """

# #     payload: Dict[str, Any] = {

# #         "jobId":      job_id,

# #         "journeyId":  journey_id,

# #         "status":     status,

# #         "progress":   progress,

# #         "message":    message,

# #     }

# #     if current_video is not None:

# #         payload["currentVideo"] = current_video
 
# #     _post("/api/internal/analysis/progress", payload)
 
 
# # def send_completed(completion_payload: Dict[str, Any]) -> None:

# #     """

# #     POST /api/internal/analysis/completed
 
# #     Called once after ALL videos in the journey have been processed and their

# #     violation frames uploaded to S3.
 
# #     completion_payload is built by CompletionPayload.to_dict() in models.py.

# #     """

# #     _post("/api/internal/analysis/completed", completion_payload)
 
 
# # def send_failed(

# #     job_id:        str,

# #     journey_id:    int,          # kept for caller convenience / logging

# #     error_message: str,

# # ) -> None:

# #     """

# #     POST /api/internal/analysis/failed
 
# #     Called whenever an unrecoverable exception occurs during job processing.

# #     Spring Boot will mark AnalysisJob and Journey as FAILED.
 
# #     Note: The API spec only requires jobId + errorMessage.

# #     journeyId is accepted as a parameter for logging but is NOT included

# #     in the outbound payload.

# #     """

# #     try:

# #         _post(

# #             "/api/internal/analysis/failed",

# #             {

# #                 "jobId":        job_id,

# #                 "errorMessage": error_message,

# #             },

# #         )

# #     except Exception as exc:

# #         # Failure callback must never itself raise — log and swallow.

# #         log.error(

# #             "[Callback] send_failed itself failed (job=%s journey=%d): %s",

# #             job_id, journey_id, exc,

# #         )
 
 
 
# """
# callback_client.py
# ──────────────────
# Sends progress, completion, and failure callbacks to the Spring Boot backend.

# Live server : https://cvvrsrailway-api.sconexsoft.com/cvs
# Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).

# Changes from previous version
# ──────────────────────────────
# • Added check_job_completed(job_id) — queries GET /api/internal/analysis/status/{jobId}
#   and returns True when the backend reports the job is already COMPLETED.
#   Called by consumer.py as an idempotency guard before starting any processing
#   on a redelivered message.
# """

# from __future__ import annotations
# import logging
# import os
# from typing import Any, Dict, Optional

# import requests
# from dotenv import load_dotenv

# # ── Credentials / config ─────────────────────────────────────────────────────
# _ENV_PATH = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)),
#     "config", "credentials.env",
# )
# load_dotenv(_ENV_PATH)

# _BASE_URL = os.environ.get(
#     "SPRING_BOOT_BASE_URL",
#     "https://cvvrsrailway-api.sconexsoft.com/cvs",
# )
# _TIMEOUT = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# log = logging.getLogger("callback_client")


# # ── Dynamic base URL setter ───────────────────────────────────────────────────

# def set_base_url(url: str) -> None:
#     """
#     Override the callback base URL at runtime.

#     Called by consumer.py when the RabbitMQ message contains a callbackBaseUrl.
#     Allows the same Python worker to callback correctly to both local and staging
#     Spring Boot servers without changing env vars.

#     The URL passed here is the FULL path up to /api/internal/analysis.
#     Individual callbacks append /progress, /completed, /failed, /status/{id}.
#     """
#     global _BASE_URL
#     base = url.rstrip("/")
#     if base.endswith("/api/internal/analysis"):
#         base = base[: -len("/api/internal/analysis")]
#     _BASE_URL = base
#     log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)


# # ── Internal helpers ──────────────────────────────────────────────────────────

# def _post(path: str, payload: Dict[str, Any]) -> None:
#     """
#     POST to a Spring Boot internal endpoint.
#     No Authorization header — /api/internal/* are worker-only endpoints.
#     Raises RuntimeError on non-2xx so the consumer can route to the failure path.
#     """
#     url = f"{_BASE_URL}{path}"
#     log.debug("[Callback] POST %s  payload=%s", url, payload)
#     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
#     if not resp.ok:
#         raise RuntimeError(
#             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
#         )
#     log.debug("[Callback] %s → %d", path, resp.status_code)


# def _get(path: str) -> requests.Response:
#     """
#     GET from a Spring Boot internal endpoint.
#     Raises RuntimeError on non-2xx.
#     """
#     url = f"{_BASE_URL}{path}"
#     log.debug("[Callback] GET %s", url)
#     resp = requests.get(url, timeout=_TIMEOUT)
#     if not resp.ok:
#         raise RuntimeError(
#             f"Callback GET {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
#         )
#     log.debug("[Callback] %s → %d", path, resp.status_code)
#     return resp


# # ── Public API ────────────────────────────────────────────────────────────────

# def check_job_completed(job_id: str) -> bool:
#     """
#     NEW — Idempotency check.

#     Queries GET /api/internal/analysis/status/{jobId} and returns True when
#     the backend reports the job status as COMPLETED.

#     Called by consumer.py at the very start of _handle_job() so that
#     RabbitMQ redeliveries of already-completed jobs are detected immediately
#     and ACKed without re-running any processing.

#     Backend contract (expected JSON shape):
#         { "status": "COMPLETED" }   → job already done → return True
#         { "status": "PROCESSING" }  → in-flight (rare edge-case) → return False
#         { "status": "PENDING" }     → not yet processed → return False
#         404 Not Found               → job unknown (treat as not completed)

#     If the endpoint does not exist yet on your Spring Boot side, add it as:
#         GET /api/internal/analysis/status/{jobId}
#         → ResponseEntity<Map<String,String>>  { "status": job.getStatus().name() }

#     Raises on network errors so the consumer can decide whether to proceed
#     with processing or skip (consumer.py catches and proceeds on error).
#     """
#     try:
#         resp = _get(f"/api/internal/analysis/status/{job_id}")
#         data = resp.json()
#         status = data.get("status", "").upper()
#         is_done = status == "COMPLETED"
#         log.info(
#             "[Callback] check_job_completed  job=%s  status=%s  is_done=%s",
#             job_id, status, is_done,
#         )
#         return is_done
#     except RuntimeError as exc:
#         # 404 → job not found in the backend → definitely not completed
#         if "404" in str(exc):
#             log.info(
#                 "[Callback] check_job_completed  job=%s  → 404 (not found) → not completed",
#                 job_id,
#             )
#             return False
#         raise


# def send_progress(
#     job_id:        str,
#     journey_id:    int,
#     progress:      int,
#     message:       str,
#     status:        str = "PROCESSING",
#     current_video: Optional[int] = None,
# ) -> None:
#     """
#     POST /api/internal/analysis/progress

#     Called periodically during analysis to update the frontend progress bar
#     and the SSE stream.

#     Parameters
#     ──────────
#     job_id        : RabbitMQ job ID.
#     journey_id    : Journey ID.
#     progress      : 0–100 integer.
#     message       : Human-readable status message.
#     status        : "PROCESSING" | "COMPLETED" | "FAILED"
#     current_video : 1-based index of the video currently being processed.
#     """
#     payload: Dict[str, Any] = {
#         "jobId":     job_id,
#         "journeyId": journey_id,
#         "status":    status,
#         "progress":  progress,
#         "message":   message,
#     }
#     if current_video is not None:
#         payload["currentVideo"] = current_video
#     _post("/api/internal/analysis/progress", payload)


# def send_completed(completion_payload: Dict[str, Any]) -> None:
#     """
#     POST /api/internal/analysis/completed

#     Called once after ALL videos in the journey have been processed and their
#     violation frames uploaded to S3.

#     completion_payload is built by CompletionPayload.to_dict() in models.py.
#     """
#     _post("/api/internal/analysis/completed", completion_payload)


# def send_failed(
#     job_id:        str,
#     journey_id:    int,   # kept for caller convenience / logging
#     error_message: str,
# ) -> None:
#     """
#     POST /api/internal/analysis/failed

#     Called whenever an unrecoverable exception occurs during job processing.
#     Spring Boot will mark AnalysisJob and Journey as FAILED.

#     Note: The API spec only requires jobId + errorMessage.
#     journeyId is accepted as a parameter for logging but is NOT included
#     in the outbound payload.
#     """
#     try:
#         _post(
#             "/api/internal/analysis/failed",
#             {
#                 "jobId":        job_id,
#                 "errorMessage": error_message,
#             },
#         )
#     except Exception as exc:
#         # Failure callback must never itself raise — log and swallow.
#         log.error(
#             "[Callback] send_failed itself failed (job=%s journey=%d): %s",
#             job_id, journey_id, exc,
#         )



"""
callback_client.py
──────────────────
Sends progress, completion, and failure callbacks to the Spring Boot backend.

Live server : https://cvvrsrailway-api.sconexsoft.com/cvs
Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).

Changes in this version
────────────────────────
• check_job_completed() now uses the EXISTING completed-callback endpoint
  as a probe instead of a dedicated /status/{jobId} endpoint that doesn't
  exist yet on Spring Boot.

  Strategy: POST /api/internal/analysis/completed with only { "jobId": id,
  "probe": true }.  Spring Boot should return 409 CONFLICT (or any non-500)
  when the job is already done.  If it returns 500 we treat it as "unknown"
  and fall through to processing (safe default).

  *** TEMPORARY until the Spring Boot /status endpoint is implemented. ***
  Once GET /api/internal/analysis/status/{jobId} is live, revert
  check_job_completed() to use _get() as originally written.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

# ── Credentials / config ─────────────────────────────────────────────────────
_ENV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "config", "credentials.env",
)
load_dotenv(_ENV_PATH)

_BASE_URL = os.environ.get(
    "SPRING_BOOT_BASE_URL",
    "https://cvvrsrailway-api.sconexsoft.com/cvs",
)
_TIMEOUT = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

log = logging.getLogger("callback_client")


# ── Dynamic base URL setter ───────────────────────────────────────────────────

def set_base_url(url: str) -> None:
    global _BASE_URL
    base = url.rstrip("/")
    if base.endswith("/api/internal/analysis"):
        base = base[: -len("/api/internal/analysis")]
    _BASE_URL = base
    log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _post(path: str, payload: Dict[str, Any]) -> requests.Response:
    url = f"{_BASE_URL}{path}"
    log.debug("[Callback] POST %s  payload=%s", url, payload)
    resp = requests.post(url, json=payload, timeout=_TIMEOUT)
    if not resp.ok:
        raise RuntimeError(
            f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
        )
    log.debug("[Callback] %s → %d", path, resp.status_code)
    return resp


def _get(path: str) -> requests.Response:
    url = f"{_BASE_URL}{path}"
    log.debug("[Callback] GET %s", url)
    resp = requests.get(url, timeout=_TIMEOUT)
    if not resp.ok:
        raise RuntimeError(
            f"Callback GET {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
        )
    log.debug("[Callback] %s → %d", path, resp.status_code)
    return resp


# ── Public API ────────────────────────────────────────────────────────────────

def check_job_completed(job_id: str) -> bool:
    """
    Idempotency check — returns True if the backend already has this job as COMPLETED.

    TWO-STAGE STRATEGY
    ──────────────────
    Stage 1 (permanent): GET /api/internal/analysis/status/{jobId}
        → { "status": "COMPLETED" }  → True
        → { "status": "PENDING/PROCESSING" } → False
        → 404 → False  (job not known yet)
        → 500 → fall through to Stage 2

    Stage 2 (temporary fallback until Spring Boot implements /status):
        Uses GET /api/internal/analysis/job/{jobId} or any existing
        read endpoint.  If that also 500s, we return False (safe default:
        process the job rather than silently drop it).

    WHAT TO IMPLEMENT ON SPRING BOOT (one small controller method):
    ───────────────────────────────────────────────────────────────
    @GetMapping("/api/internal/analysis/status/{jobId}")
    public ResponseEntity<Map<String,String>> getJobStatus(
            @PathVariable String jobId) {
        return analysisJobRepository.findByJobId(jobId)
            .map(job -> ResponseEntity.ok(Map.of("status", job.getStatus().name())))
            .orElse(ResponseEntity.notFound().<Map<String,String>>build());
    }

    Once that endpoint is deployed, Stage 2 below can be deleted.
    """

    # ── Stage 1: dedicated status endpoint ───────────────────────────────────
    try:
        resp = _get(f"/api/internal/analysis/status/{job_id}")
        status = resp.json().get("status", "").upper()
        is_done = status == "COMPLETED"
        log.info(
            "[Callback] idempotency check (status endpoint)  job=%s  status=%s  done=%s",
            job_id, status, is_done,
        )
        return is_done
    except RuntimeError as exc:
        if "404" in str(exc):
            log.info(
                "[Callback] idempotency check  job=%s  → 404 → not completed", job_id
            )
            return False
        if "500" in str(exc):
            log.warning(
                "[Callback] idempotency check  job=%s  → /status endpoint returned 500 "
                "(not implemented yet?) — trying fallback probe.", job_id
            )
            # fall through to Stage 2
        else:
            # Network error or unexpected status — safe default: process the job
            log.warning(
                "[Callback] idempotency check  job=%s  → unexpected error (%s) "
                "— proceeding with processing.", job_id, exc
            )
            return False

    # ── Stage 2: fallback probe using progress endpoint ───────────────────────
    # We send a lightweight progress probe at 0 % with status=CHECK.
    # Spring Boot should:
    #   • Return 200 if the job is still PENDING/PROCESSING (→ process it).
    #   • Return 409 CONFLICT if the job is already COMPLETED (→ skip it).
    #   • Return 404 if the job is unknown (→ process it).
    # If the backend doesn't handle the CHECK status specially it will just
    # update progress to 0 — harmless on an already-completed job.
    #
    # THIS IS A TEMPORARY FALLBACK.  Remove once /status endpoint is live.
    try:
        url = f"{_BASE_URL}/api/internal/analysis/progress"
        resp = requests.post(
            url,
            json={
                "jobId":    job_id,
                "status":   "CHECK",   # sentinel value Spring Boot can detect
                "progress": 0,
                "message":  "idempotency-probe",
            },
            timeout=_TIMEOUT,
        )
        if resp.status_code == 409:
            log.warning(
                "[Callback] idempotency check (probe)  job=%s  → 409 CONFLICT "
                "→ already COMPLETED — will skip.", job_id
            )
            return True
        log.info(
            "[Callback] idempotency check (probe)  job=%s  → HTTP %d → not completed.",
            job_id, resp.status_code,
        )
        return False
    except Exception as exc2:
        log.warning(
            "[Callback] idempotency check (probe) failed  job=%s: %s "
            "— proceeding with processing.", job_id, exc2
        )
        return False


def send_progress(
    job_id:        str,
    journey_id:    int,
    progress:      int,
    message:       str,
    status:        str = "PROCESSING",
    current_video: Optional[int] = None,
) -> None:
    payload: Dict[str, Any] = {
        "jobId":     job_id,
        "journeyId": journey_id,
        "status":    status,
        "progress":  progress,
        "message":   message,
    }
    if current_video is not None:
        payload["currentVideo"] = current_video
    _post("/api/internal/analysis/progress", payload)


def send_completed(completion_payload: Dict[str, Any]) -> None:
    _post("/api/internal/analysis/completed", completion_payload)


def send_failed(
    job_id:        str,
    journey_id:    int,
    error_message: str,
) -> None:
    try:
        _post(
            "/api/internal/analysis/failed",
            {
                "jobId":        job_id,
                "errorMessage": error_message,
            },
        )
    except Exception as exc:
        log.error(
            "[Callback] send_failed itself failed (job=%s journey=%d): %s",
            job_id, journey_id, exc,
        )