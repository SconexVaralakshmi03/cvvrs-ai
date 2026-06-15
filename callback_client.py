# # """
# # callback_client.py
# # ──────────────────
# # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # All three public helpers are fire-and-forget from the caller's perspective;
# # they raise on non-2xx so the consumer can route to the failure path.
# # """

# # from __future__ import annotations

# # import logging
# # import os
# # from typing import Any, Dict, List

# # import requests
# # from dotenv import load_dotenv

# # # ── Credentials / config ────────────────────────────────────────────────────
# # _ENV_PATH = os.path.join(
# #     os.path.dirname(os.path.abspath(__file__)),
# #     "config", "credentials.env",
# # )
# # load_dotenv(_ENV_PATH)

# # _BASE_URL = os.environ.get("SPRING_BOOT_BASE_URL", "http://localhost:8080")
# # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # log = logging.getLogger("callback_client")


# # # ── Internal helper ──────────────────────────────────────────────────────────

# # def _post(path: str, payload: Dict[str, Any]) -> None:
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
# #     job_id:     str,
# #     journey_id: int,
# #     progress:   int,
# #     message:    str,
# #     status:     str = "PROCESSING",
# # ) -> None:
# #     """
# #     POST /api/internal/analysis/progress

# #     Called periodically during analysis to update the frontend progress bar
# #     and the SSE stream.
# #     """
# #     _post(
# #         "/api/internal/analysis/progress",
# #         {
# #             "jobId":      job_id,
# #             "journeyId":  journey_id,
# #             "status":     status,
# #             "progress":   progress,
# #             "message":    message,
# #         },
# #     )


# # def send_completed(completion_payload: Dict[str, Any]) -> None:
# #     """
# #     POST /api/internal/analysis/completed

# #     Called once — after ALL videos in the journey have been processed and
# #     their violation frames uploaded to S3.

# #     Expected shape of completion_payload
# #     ─────────────────────────────────────
# #     {
# #         "jobId":          str,
# #         "journeyId":      int,
# #         "processingTime": int,          # wall-clock ms
# #         "videoResults":   List[dict],   # see models.py → VideoResult
# #     }
# #     """
# #     _post("/api/internal/analysis/completed", completion_payload)


# # def send_failed(
# #     job_id:        str,
# #     journey_id:    int,
# #     error_message: str,
# # ) -> None:
# #     """
# #     POST /api/internal/analysis/failed

# #     Called whenever an unrecoverable exception occurs during job processing.
# #     Spring Boot will mark AnalysisJob and Journey as FAILED.
# #     """
# #     try:
# #         _post(
# #             "/api/internal/analysis/failed",
# #             {
# #                 "jobId":        job_id,
# #                 "journeyId":    journey_id,
# #                 "errorMessage": error_message,
# #             },
# #         )
# #     except Exception as exc:
# #         # Failure callback must never itself raise — log and swallow.
# #         log.error("[Callback] send_failed itself failed: %s", exc)

# """
# callback_client.py
# ──────────────────
# Sends progress, completion, and failure callbacks to the Spring Boot backend.

# Live server : https://cvvrsrailway-api.sconexsoft.com/cvs
# Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).

# Changes from previous version
# ──────────────────────────────
# • _BASE_URL now defaults to the live CVVRS domain (overridable via env var
#   SPRING_BOOT_BASE_URL for local testing).
# • send_progress()  — payload now includes the `currentVideo` field required
#   by the API.
# • send_completed() — accepts a fully-formed dict (built by CompletionPayload.
#   to_dict()); no shape changes needed here since the caller owns the dict.
# • send_failed()    — payload drops `journeyId` (not in the API spec for this
#   endpoint); only jobId + errorMessage are sent.
# """

# from __future__ import annotations

# import logging
# import os
# from typing import Any, Dict, Optional

# import requests
# from dotenv import load_dotenv

# # ── Credentials / config ────────────────────────────────────────────────────
# _ENV_PATH = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)),
#     "config", "credentials.env",
# )
# load_dotenv(_ENV_PATH)

# # Default to the live CVVRS server; override with SPRING_BOOT_BASE_URL for local dev.
# _BASE_URL = os.environ.get(
#     "SPRING_BOOT_BASE_URL",
#     "https://cvvrsrailway-api.sconexsoft.com/cvs",
# )
# _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# log = logging.getLogger("callback_client")


# # ── Internal helper ──────────────────────────────────────────────────────────

# def _post(path: str, payload: Dict[str, Any]) -> None:
#     """
#     POST to a Spring Boot internal endpoint.
#     No Authorization header is sent — /api/internal/* are worker-only endpoints.
#     Raises RuntimeError on non-2xx so the consumer can route to failure path.
#     """
#     url = f"{_BASE_URL}{path}"
#     log.debug("[Callback] POST %s  payload=%s", url, payload)
#     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
#     if not resp.ok:
#         raise RuntimeError(
#             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
#         )
#     log.debug("[Callback] %s → %d", path, resp.status_code)


# # ── Public API ───────────────────────────────────────────────────────────────

# def send_progress(
#     job_id:        str,
#     journey_id:    int,
#     progress:      int,
#     message:       str,
#     status:        str = "PROCESSING",
#     current_video: Optional[int] = None,   # NEW — index of video currently being processed
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
#     current_video : 1-based index of the video currently being processed
#                     (omitted from payload when None).
#     """
#     payload: Dict[str, Any] = {
#         "jobId":      job_id,
#         "journeyId":  journey_id,
#         "status":     status,
#         "progress":   progress,
#         "message":    message,
#     }
#     if current_video is not None:
#         payload["currentVideo"] = current_video

#     _post("/api/internal/analysis/progress", payload)


# def send_completed(completion_payload: Dict[str, Any]) -> None:
#     """
#     POST /api/internal/analysis/completed

#     Called once after ALL videos in the journey have been processed and their
#     violation frames uploaded to S3.

#     Expected shape of completion_payload (built by CompletionPayload.to_dict())
#     ────────────────────────────────────────────────────────────────────────────
#     {
#         "jobId":         str,
#         "journeyId":     int,
#         "batchId":       str,           # "BATCH-<jobId>" if not supplied
#         "trainDetailId": int,
#         "folderName":    str,
#         "processingTime":int,           # wall-clock ms
#         "videoResults": [
#             {
#                 "videoId":         str,       # STRING per API spec
#                 "sequenceNo":      int,
#                 "durationSeconds": float,
#                 "originalS3Key":   str,
#                 "violations": [
#                     {
#                         "violationType":          str,
#                         "severity":               str,
#                         "confidence":             float,
#                         "riskScore":              float,
#                         "timestamp":              float,   # journey-global seconds
#                         "originalVideoTimestamp": float,   # local-video seconds
#                         "framePaths":             [str]
#                     }
#                 ]
#             }
#         ]
#     }
#     """
#     _post("/api/internal/analysis/completed", completion_payload)


# def send_failed(
#     job_id:        str,
#     journey_id:    int,          # kept as parameter for caller convenience / logging
#     error_message: str,
# ) -> None:
#     """
#     POST /api/internal/analysis/failed

#     Called whenever an unrecoverable exception occurs during job processing.
#     Spring Boot will mark AnalysisJob and Journey as FAILED.

#     Note: The API spec only requires jobId + errorMessage in the request body.
#     journeyId is accepted as a parameter here for logging but is NOT included
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
 
Fixes from previous version

─────────────────────────────

• Added set_base_url() — called by consumer.py with the callbackBaseUrl from

  the RabbitMQ message, so Python posts to the correct Spring Boot server

  per environment without needing env var changes.

• _BASE_URL still falls back to env var SPRING_BOOT_BASE_URL or the live

  server when callbackBaseUrl is not provided in the message.

"""
 
from __future__ import annotations
 
import logging

import os

from typing import Any, Dict, Optional
 
import requests

from dotenv import load_dotenv
 
# ── Credentials / config ────────────────────────────────────────────────────

_ENV_PATH = os.path.join(

    os.path.dirname(os.path.abspath(__file__)),

    "config", "credentials.env",

)

load_dotenv(_ENV_PATH)
 
# Default to the live CVVRS server; override with SPRING_BOOT_BASE_URL for local dev

# or call set_base_url() with the callbackBaseUrl from the RabbitMQ message.

_BASE_URL = os.environ.get(

    "SPRING_BOOT_BASE_URL",

    "https://cvvrsrailway-api.sconexsoft.com/cvs",

)

_TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))
 
log = logging.getLogger("callback_client")
 
 
# ── FIX: dynamic base URL setter ─────────────────────────────────────────────
 
def set_base_url(url: str) -> None:

    """

    Override the callback base URL at runtime.
 
    Called by consumer.py when the RabbitMQ message contains a callbackBaseUrl.

    This allows the same Python worker to callback correctly to both local

    and staging Spring Boot servers without changing env vars.
 
    Example values:

        "http://localhost:8093/api/internal/analysis"         (local)

        "https://cvvrsrailway-api.sconexsoft.com/cvs/api/internal/analysis"  (staging)
 
    Note: The URL passed here is the FULL path up to /api/internal/analysis.

    The individual callbacks append /progress, /completed, /failed.

    """

    global _BASE_URL

    # Strip the /api/internal/analysis suffix if present — we add it in _post()

    base = url.rstrip("/")

    if base.endswith("/api/internal/analysis"):

        base = base[: -len("/api/internal/analysis")]

    _BASE_URL = base

    log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)
 
 
# ── Internal helper ──────────────────────────────────────────────────────────
 
def _post(path: str, payload: Dict[str, Any]) -> None:

    """

    POST to a Spring Boot internal endpoint.

    No Authorization header is sent — /api/internal/* are worker-only endpoints.

    Raises RuntimeError on non-2xx so the consumer can route to failure path.

    """

    url = f"{_BASE_URL}{path}"

    log.debug("[Callback] POST %s  payload=%s", url, payload)

    resp = requests.post(url, json=payload, timeout=_TIMEOUT)

    if not resp.ok:

        raise RuntimeError(

            f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"

        )

    log.debug("[Callback] %s → %d", path, resp.status_code)
 
 
# ── Public API ───────────────────────────────────────────────────────────────
 
def send_progress(

    job_id:        str,

    journey_id:    int,

    progress:      int,

    message:       str,

    status:        str = "PROCESSING",

    current_video: Optional[int] = None,

) -> None:

    """

    POST /api/internal/analysis/progress
 
    Called periodically during analysis to update the frontend progress bar

    and the SSE stream.
 
    Parameters

    ──────────

    job_id        : RabbitMQ job ID.

    journey_id    : Journey ID.

    progress      : 0–100 integer.

    message       : Human-readable status message.

    status        : "PROCESSING" | "COMPLETED" | "FAILED"

    current_video : 1-based index of the video currently being processed.

    """

    payload: Dict[str, Any] = {

        "jobId":      job_id,

        "journeyId":  journey_id,

        "status":     status,

        "progress":   progress,

        "message":    message,

    }

    if current_video is not None:

        payload["currentVideo"] = current_video
 
    _post("/api/internal/analysis/progress", payload)
 
 
def send_completed(completion_payload: Dict[str, Any]) -> None:

    """

    POST /api/internal/analysis/completed
 
    Called once after ALL videos in the journey have been processed and their

    violation frames uploaded to S3.
 
    completion_payload is built by CompletionPayload.to_dict() in models.py.

    """

    _post("/api/internal/analysis/completed", completion_payload)
 
 
def send_failed(

    job_id:        str,

    journey_id:    int,          # kept for caller convenience / logging

    error_message: str,

) -> None:

    """

    POST /api/internal/analysis/failed
 
    Called whenever an unrecoverable exception occurs during job processing.

    Spring Boot will mark AnalysisJob and Journey as FAILED.
 
    Note: The API spec only requires jobId + errorMessage.

    journeyId is accepted as a parameter for logging but is NOT included

    in the outbound payload.

    """

    try:

        _post(

            "/api/internal/analysis/failed",

            {

                "jobId":        job_id,

                "errorMessage": error_message,

            },

        )

    except Exception as exc:

        # Failure callback must never itself raise — log and swallow.

        log.error(

            "[Callback] send_failed itself failed (job=%s journey=%d): %s",

            job_id, journey_id, exc,

        )
 