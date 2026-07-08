# # # # # # # # """
# # # # # # # # callback_client.py
# # # # # # # # ──────────────────
# # # # # # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # # # # # # All three public helpers are fire-and-forget from the caller's perspective;
# # # # # # # # they raise on non-2xx so the consumer can route to the failure path.
# # # # # # # # """

# # # # # # # # from __future__ import annotations

# # # # # # # # import logging
# # # # # # # # import os
# # # # # # # # from typing import Any, Dict, List

# # # # # # # # import requests
# # # # # # # # from dotenv import load_dotenv

# # # # # # # # # ── Credentials / config ────────────────────────────────────────────────────
# # # # # # # # _ENV_PATH = os.path.join(
# # # # # # # #     os.path.dirname(os.path.abspath(__file__)),
# # # # # # # #     "config", "credentials.env",
# # # # # # # # )
# # # # # # # # load_dotenv(_ENV_PATH)

# # # # # # # # _BASE_URL = os.environ.get("SPRING_BOOT_BASE_URL", "http://localhost:8080")
# # # # # # # # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # # # # # # log = logging.getLogger("callback_client")


# # # # # # # # # ── Internal helper ──────────────────────────────────────────────────────────

# # # # # # # # def _post(path: str, payload: Dict[str, Any]) -> None:
# # # # # # # #     url = f"{_BASE_URL}{path}"
# # # # # # # #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# # # # # # # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# # # # # # # #     if not resp.ok:
# # # # # # # #         raise RuntimeError(
# # # # # # # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # # # # # # #         )
# # # # # # # #     log.debug("[Callback] %s → %d", path, resp.status_code)


# # # # # # # # # ── Public API ───────────────────────────────────────────────────────────────

# # # # # # # # def send_progress(
# # # # # # # #     job_id:     str,
# # # # # # # #     journey_id: int,
# # # # # # # #     progress:   int,
# # # # # # # #     message:    str,
# # # # # # # #     status:     str = "PROCESSING",
# # # # # # # # ) -> None:
# # # # # # # #     """
# # # # # # # #     POST /api/internal/analysis/progress

# # # # # # # #     Called periodically during analysis to update the frontend progress bar
# # # # # # # #     and the SSE stream.
# # # # # # # #     """
# # # # # # # #     _post(
# # # # # # # #         "/api/internal/analysis/progress",
# # # # # # # #         {
# # # # # # # #             "jobId":      job_id,
# # # # # # # #             "journeyId":  journey_id,
# # # # # # # #             "status":     status,
# # # # # # # #             "progress":   progress,
# # # # # # # #             "message":    message,
# # # # # # # #         },
# # # # # # # #     )


# # # # # # # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # # # # # # #     """
# # # # # # # #     POST /api/internal/analysis/completed

# # # # # # # #     Called once — after ALL videos in the journey have been processed and
# # # # # # # #     their violation frames uploaded to S3.

# # # # # # # #     Expected shape of completion_payload
# # # # # # # #     ─────────────────────────────────────
# # # # # # # #     {
# # # # # # # #         "jobId":          str,
# # # # # # # #         "journeyId":      int,
# # # # # # # #         "processingTime": int,          # wall-clock ms
# # # # # # # #         "videoResults":   List[dict],   # see models.py → VideoResult
# # # # # # # #     }
# # # # # # # #     """
# # # # # # # #     _post("/api/internal/analysis/completed", completion_payload)


# # # # # # # # def send_failed(
# # # # # # # #     job_id:        str,
# # # # # # # #     journey_id:    int,
# # # # # # # #     error_message: str,
# # # # # # # # ) -> None:
# # # # # # # #     """
# # # # # # # #     POST /api/internal/analysis/failed

# # # # # # # #     Called whenever an unrecoverable exception occurs during job processing.
# # # # # # # #     Spring Boot will mark AnalysisJob and Journey as FAILED.
# # # # # # # #     """
# # # # # # # #     try:
# # # # # # # #         _post(
# # # # # # # #             "/api/internal/analysis/failed",
# # # # # # # #             {
# # # # # # # #                 "jobId":        job_id,
# # # # # # # #                 "journeyId":    journey_id,
# # # # # # # #                 "errorMessage": error_message,
# # # # # # # #             },
# # # # # # # #         )
# # # # # # # #     except Exception as exc:
# # # # # # # #         # Failure callback must never itself raise — log and swallow.
# # # # # # # #         log.error("[Callback] send_failed itself failed: %s", exc)

# # # # # # # """
# # # # # # # callback_client.py
# # # # # # # ──────────────────
# # # # # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # # # # # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs
# # # # # # # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).

# # # # # # # Changes from previous version
# # # # # # # ──────────────────────────────
# # # # # # # • _BASE_URL now defaults to the live CVVRS domain (overridable via env var
# # # # # # #   SPRING_BOOT_BASE_URL for local testing).
# # # # # # # • send_progress()  — payload now includes the `currentVideo` field required
# # # # # # #   by the API.
# # # # # # # • send_completed() — accepts a fully-formed dict (built by CompletionPayload.
# # # # # # #   to_dict()); no shape changes needed here since the caller owns the dict.
# # # # # # # • send_failed()    — payload drops `journeyId` (not in the API spec for this
# # # # # # #   endpoint); only jobId + errorMessage are sent.
# # # # # # # """

# # # # # # # from __future__ import annotations

# # # # # # # import logging
# # # # # # # import os
# # # # # # # from typing import Any, Dict, Optional

# # # # # # # import requests
# # # # # # # from dotenv import load_dotenv

# # # # # # # # ── Credentials / config ────────────────────────────────────────────────────
# # # # # # # _ENV_PATH = os.path.join(
# # # # # # #     os.path.dirname(os.path.abspath(__file__)),
# # # # # # #     "config", "credentials.env",
# # # # # # # )
# # # # # # # load_dotenv(_ENV_PATH)

# # # # # # # # Default to the live CVVRS server; override with SPRING_BOOT_BASE_URL for local dev.
# # # # # # # _BASE_URL = os.environ.get(
# # # # # # #     "SPRING_BOOT_BASE_URL",
# # # # # # #     "https://cvvrsrailway-api.sconexsoft.com/cvs",
# # # # # # # )
# # # # # # # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # # # # # log = logging.getLogger("callback_client")


# # # # # # # # ── Internal helper ──────────────────────────────────────────────────────────

# # # # # # # def _post(path: str, payload: Dict[str, Any]) -> None:
# # # # # # #     """
# # # # # # #     POST to a Spring Boot internal endpoint.
# # # # # # #     No Authorization header is sent — /api/internal/* are worker-only endpoints.
# # # # # # #     Raises RuntimeError on non-2xx so the consumer can route to failure path.
# # # # # # #     """
# # # # # # #     url = f"{_BASE_URL}{path}"
# # # # # # #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# # # # # # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# # # # # # #     if not resp.ok:
# # # # # # #         raise RuntimeError(
# # # # # # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # # # # # #         )
# # # # # # #     log.debug("[Callback] %s → %d", path, resp.status_code)


# # # # # # # # ── Public API ───────────────────────────────────────────────────────────────

# # # # # # # def send_progress(
# # # # # # #     job_id:        str,
# # # # # # #     journey_id:    int,
# # # # # # #     progress:      int,
# # # # # # #     message:       str,
# # # # # # #     status:        str = "PROCESSING",
# # # # # # #     current_video: Optional[int] = None,   # NEW — index of video currently being processed
# # # # # # # ) -> None:
# # # # # # #     """
# # # # # # #     POST /api/internal/analysis/progress

# # # # # # #     Called periodically during analysis to update the frontend progress bar
# # # # # # #     and the SSE stream.

# # # # # # #     Parameters
# # # # # # #     ──────────
# # # # # # #     job_id        : RabbitMQ job ID.
# # # # # # #     journey_id    : Journey ID.
# # # # # # #     progress      : 0–100 integer.
# # # # # # #     message       : Human-readable status message.
# # # # # # #     status        : "PROCESSING" | "COMPLETED" | "FAILED"
# # # # # # #     current_video : 1-based index of the video currently being processed
# # # # # # #                     (omitted from payload when None).
# # # # # # #     """
# # # # # # #     payload: Dict[str, Any] = {
# # # # # # #         "jobId":      job_id,
# # # # # # #         "journeyId":  journey_id,
# # # # # # #         "status":     status,
# # # # # # #         "progress":   progress,
# # # # # # #         "message":    message,
# # # # # # #     }
# # # # # # #     if current_video is not None:
# # # # # # #         payload["currentVideo"] = current_video

# # # # # # #     _post("/api/internal/analysis/progress", payload)


# # # # # # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # # # # # #     """
# # # # # # #     POST /api/internal/analysis/completed

# # # # # # #     Called once after ALL videos in the journey have been processed and their
# # # # # # #     violation frames uploaded to S3.

# # # # # # #     Expected shape of completion_payload (built by CompletionPayload.to_dict())
# # # # # # #     ────────────────────────────────────────────────────────────────────────────
# # # # # # #     {
# # # # # # #         "jobId":         str,
# # # # # # #         "journeyId":     int,
# # # # # # #         "batchId":       str,           # "BATCH-<jobId>" if not supplied
# # # # # # #         "trainDetailId": int,
# # # # # # #         "folderName":    str,
# # # # # # #         "processingTime":int,           # wall-clock ms
# # # # # # #         "videoResults": [
# # # # # # #             {
# # # # # # #                 "videoId":         str,       # STRING per API spec
# # # # # # #                 "sequenceNo":      int,
# # # # # # #                 "durationSeconds": float,
# # # # # # #                 "originalS3Key":   str,
# # # # # # #                 "violations": [
# # # # # # #                     {
# # # # # # #                         "violationType":          str,
# # # # # # #                         "severity":               str,
# # # # # # #                         "confidence":             float,
# # # # # # #                         "riskScore":              float,
# # # # # # #                         "timestamp":              float,   # journey-global seconds
# # # # # # #                         "originalVideoTimestamp": float,   # local-video seconds
# # # # # # #                         "framePaths":             [str]
# # # # # # #                     }
# # # # # # #                 ]
# # # # # # #             }
# # # # # # #         ]
# # # # # # #     }
# # # # # # #     """
# # # # # # #     _post("/api/internal/analysis/completed", completion_payload)


# # # # # # # def send_failed(
# # # # # # #     job_id:        str,
# # # # # # #     journey_id:    int,          # kept as parameter for caller convenience / logging
# # # # # # #     error_message: str,
# # # # # # # ) -> None:
# # # # # # #     """
# # # # # # #     POST /api/internal/analysis/failed

# # # # # # #     Called whenever an unrecoverable exception occurs during job processing.
# # # # # # #     Spring Boot will mark AnalysisJob and Journey as FAILED.

# # # # # # #     Note: The API spec only requires jobId + errorMessage in the request body.
# # # # # # #     journeyId is accepted as a parameter here for logging but is NOT included
# # # # # # #     in the outbound payload.
# # # # # # #     """
# # # # # # #     try:
# # # # # # #         _post(
# # # # # # #             "/api/internal/analysis/failed",
# # # # # # #             {
# # # # # # #                 "jobId":        job_id,
# # # # # # #                 "errorMessage": error_message,
# # # # # # #             },
# # # # # # #         )
# # # # # # #     except Exception as exc:
# # # # # # #         # Failure callback must never itself raise — log and swallow.
# # # # # # #         log.error(
# # # # # # #             "[Callback] send_failed itself failed (job=%s journey=%d): %s",
# # # # # # #             job_id, journey_id, exc,
# # # # # # #         )



# # # # # # """

# # # # # # callback_client.py

# # # # # # ──────────────────

# # # # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.
 
# # # # # # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs

# # # # # # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).
 
# # # # # # Fixes from previous version

# # # # # # ─────────────────────────────

# # # # # # • Added set_base_url() — called by consumer.py with the callbackBaseUrl from

# # # # # #   the RabbitMQ message, so Python posts to the correct Spring Boot server

# # # # # #   per environment without needing env var changes.

# # # # # # • _BASE_URL still falls back to env var SPRING_BOOT_BASE_URL or the live

# # # # # #   server when callbackBaseUrl is not provided in the message.

# # # # # # """
 
# # # # # # from __future__ import annotations
 
# # # # # # import logging

# # # # # # import os

# # # # # # from typing import Any, Dict, Optional
 
# # # # # # import requests

# # # # # # from dotenv import load_dotenv
 
# # # # # # # ── Credentials / config ────────────────────────────────────────────────────

# # # # # # _ENV_PATH = os.path.join(

# # # # # #     os.path.dirname(os.path.abspath(__file__)),

# # # # # #     "config", "credentials.env",

# # # # # # )

# # # # # # load_dotenv(_ENV_PATH)
 
# # # # # # # Default to the live CVVRS server; override with SPRING_BOOT_BASE_URL for local dev

# # # # # # # or call set_base_url() with the callbackBaseUrl from the RabbitMQ message.

# # # # # # _BASE_URL = os.environ.get(

# # # # # #     "SPRING_BOOT_BASE_URL",

# # # # # #     "https://cvvrsrailway-api.sconexsoft.com/cvs",

# # # # # # )

# # # # # # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))
 
# # # # # # log = logging.getLogger("callback_client")
 
 
# # # # # # # ── FIX: dynamic base URL setter ─────────────────────────────────────────────
 
# # # # # # def set_base_url(url: str) -> None:

# # # # # #     """

# # # # # #     Override the callback base URL at runtime.
 
# # # # # #     Called by consumer.py when the RabbitMQ message contains a callbackBaseUrl.

# # # # # #     This allows the same Python worker to callback correctly to both local

# # # # # #     and staging Spring Boot servers without changing env vars.
 
# # # # # #     Example values:

# # # # # #         "http://localhost:8093/api/internal/analysis"         (local)

# # # # # #         "https://cvvrsrailway-api.sconexsoft.com/cvs/api/internal/analysis"  (staging)
 
# # # # # #     Note: The URL passed here is the FULL path up to /api/internal/analysis.

# # # # # #     The individual callbacks append /progress, /completed, /failed.

# # # # # #     """

# # # # # #     global _BASE_URL

# # # # # #     # Strip the /api/internal/analysis suffix if present — we add it in _post()

# # # # # #     base = url.rstrip("/")

# # # # # #     if base.endswith("/api/internal/analysis"):

# # # # # #         base = base[: -len("/api/internal/analysis")]

# # # # # #     _BASE_URL = base

# # # # # #     log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)
 
 
# # # # # # # ── Internal helper ──────────────────────────────────────────────────────────
 
# # # # # # def _post(path: str, payload: Dict[str, Any]) -> None:

# # # # # #     """

# # # # # #     POST to a Spring Boot internal endpoint.

# # # # # #     No Authorization header is sent — /api/internal/* are worker-only endpoints.

# # # # # #     Raises RuntimeError on non-2xx so the consumer can route to failure path.

# # # # # #     """

# # # # # #     url = f"{_BASE_URL}{path}"

# # # # # #     log.debug("[Callback] POST %s  payload=%s", url, payload)

# # # # # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)

# # # # # #     if not resp.ok:

# # # # # #         raise RuntimeError(

# # # # # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"

# # # # # #         )

# # # # # #     log.debug("[Callback] %s → %d", path, resp.status_code)
 
 
# # # # # # # ── Public API ───────────────────────────────────────────────────────────────
 
# # # # # # def send_progress(

# # # # # #     job_id:        str,

# # # # # #     journey_id:    int,

# # # # # #     progress:      int,

# # # # # #     message:       str,

# # # # # #     status:        str = "PROCESSING",

# # # # # #     current_video: Optional[int] = None,

# # # # # # ) -> None:

# # # # # #     """

# # # # # #     POST /api/internal/analysis/progress
 
# # # # # #     Called periodically during analysis to update the frontend progress bar

# # # # # #     and the SSE stream.
 
# # # # # #     Parameters

# # # # # #     ──────────

# # # # # #     job_id        : RabbitMQ job ID.

# # # # # #     journey_id    : Journey ID.

# # # # # #     progress      : 0–100 integer.

# # # # # #     message       : Human-readable status message.

# # # # # #     status        : "PROCESSING" | "COMPLETED" | "FAILED"

# # # # # #     current_video : 1-based index of the video currently being processed.

# # # # # #     """

# # # # # #     payload: Dict[str, Any] = {

# # # # # #         "jobId":      job_id,

# # # # # #         "journeyId":  journey_id,

# # # # # #         "status":     status,

# # # # # #         "progress":   progress,

# # # # # #         "message":    message,

# # # # # #     }

# # # # # #     if current_video is not None:

# # # # # #         payload["currentVideo"] = current_video
 
# # # # # #     _post("/api/internal/analysis/progress", payload)
 
 
# # # # # # def send_completed(completion_payload: Dict[str, Any]) -> None:

# # # # # #     """

# # # # # #     POST /api/internal/analysis/completed
 
# # # # # #     Called once after ALL videos in the journey have been processed and their

# # # # # #     violation frames uploaded to S3.
 
# # # # # #     completion_payload is built by CompletionPayload.to_dict() in models.py.

# # # # # #     """

# # # # # #     _post("/api/internal/analysis/completed", completion_payload)
 
 
# # # # # # def send_failed(

# # # # # #     job_id:        str,

# # # # # #     journey_id:    int,          # kept for caller convenience / logging

# # # # # #     error_message: str,

# # # # # # ) -> None:

# # # # # #     """

# # # # # #     POST /api/internal/analysis/failed
 
# # # # # #     Called whenever an unrecoverable exception occurs during job processing.

# # # # # #     Spring Boot will mark AnalysisJob and Journey as FAILED.
 
# # # # # #     Note: The API spec only requires jobId + errorMessage.

# # # # # #     journeyId is accepted as a parameter for logging but is NOT included

# # # # # #     in the outbound payload.

# # # # # #     """

# # # # # #     try:

# # # # # #         _post(

# # # # # #             "/api/internal/analysis/failed",

# # # # # #             {

# # # # # #                 "jobId":        job_id,

# # # # # #                 "errorMessage": error_message,

# # # # # #             },

# # # # # #         )

# # # # # #     except Exception as exc:

# # # # # #         # Failure callback must never itself raise — log and swallow.

# # # # # #         log.error(

# # # # # #             "[Callback] send_failed itself failed (job=%s journey=%d): %s",

# # # # # #             job_id, journey_id, exc,

# # # # # #         )
 
 
 
# # # # # """
# # # # # callback_client.py
# # # # # ──────────────────
# # # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # # # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs
# # # # # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).

# # # # # Changes from previous version
# # # # # ──────────────────────────────
# # # # # • Added check_job_completed(job_id) — queries GET /api/internal/analysis/status/{jobId}
# # # # #   and returns True when the backend reports the job is already COMPLETED.
# # # # #   Called by consumer.py as an idempotency guard before starting any processing
# # # # #   on a redelivered message.
# # # # # """

# # # # # from __future__ import annotations
# # # # # import logging
# # # # # import os
# # # # # from typing import Any, Dict, Optional

# # # # # import requests
# # # # # from dotenv import load_dotenv

# # # # # # ── Credentials / config ─────────────────────────────────────────────────────
# # # # # _ENV_PATH = os.path.join(
# # # # #     os.path.dirname(os.path.abspath(__file__)),
# # # # #     "config", "credentials.env",
# # # # # )
# # # # # load_dotenv(_ENV_PATH)

# # # # # _BASE_URL = os.environ.get(
# # # # #     "SPRING_BOOT_BASE_URL",
# # # # #     "https://cvvrsrailway-api.sconexsoft.com/cvs",
# # # # # )
# # # # # _TIMEOUT = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # # # log = logging.getLogger("callback_client")


# # # # # # ── Dynamic base URL setter ───────────────────────────────────────────────────

# # # # # def set_base_url(url: str) -> None:
# # # # #     """
# # # # #     Override the callback base URL at runtime.

# # # # #     Called by consumer.py when the RabbitMQ message contains a callbackBaseUrl.
# # # # #     Allows the same Python worker to callback correctly to both local and staging
# # # # #     Spring Boot servers without changing env vars.

# # # # #     The URL passed here is the FULL path up to /api/internal/analysis.
# # # # #     Individual callbacks append /progress, /completed, /failed, /status/{id}.
# # # # #     """
# # # # #     global _BASE_URL
# # # # #     base = url.rstrip("/")
# # # # #     if base.endswith("/api/internal/analysis"):
# # # # #         base = base[: -len("/api/internal/analysis")]
# # # # #     _BASE_URL = base
# # # # #     log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)


# # # # # # ── Internal helpers ──────────────────────────────────────────────────────────

# # # # # def _post(path: str, payload: Dict[str, Any]) -> None:
# # # # #     """
# # # # #     POST to a Spring Boot internal endpoint.
# # # # #     No Authorization header — /api/internal/* are worker-only endpoints.
# # # # #     Raises RuntimeError on non-2xx so the consumer can route to the failure path.
# # # # #     """
# # # # #     url = f"{_BASE_URL}{path}"
# # # # #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# # # # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# # # # #     if not resp.ok:
# # # # #         raise RuntimeError(
# # # # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # # # #         )
# # # # #     log.debug("[Callback] %s → %d", path, resp.status_code)


# # # # # def _get(path: str) -> requests.Response:
# # # # #     """
# # # # #     GET from a Spring Boot internal endpoint.
# # # # #     Raises RuntimeError on non-2xx.
# # # # #     """
# # # # #     url = f"{_BASE_URL}{path}"
# # # # #     log.debug("[Callback] GET %s", url)
# # # # #     resp = requests.get(url, timeout=_TIMEOUT)
# # # # #     if not resp.ok:
# # # # #         raise RuntimeError(
# # # # #             f"Callback GET {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # # # #         )
# # # # #     log.debug("[Callback] %s → %d", path, resp.status_code)
# # # # #     return resp


# # # # # # ── Public API ────────────────────────────────────────────────────────────────

# # # # # def check_job_completed(job_id: str) -> bool:
# # # # #     """
# # # # #     NEW — Idempotency check.

# # # # #     Queries GET /api/internal/analysis/status/{jobId} and returns True when
# # # # #     the backend reports the job status as COMPLETED.

# # # # #     Called by consumer.py at the very start of _handle_job() so that
# # # # #     RabbitMQ redeliveries of already-completed jobs are detected immediately
# # # # #     and ACKed without re-running any processing.

# # # # #     Backend contract (expected JSON shape):
# # # # #         { "status": "COMPLETED" }   → job already done → return True
# # # # #         { "status": "PROCESSING" }  → in-flight (rare edge-case) → return False
# # # # #         { "status": "PENDING" }     → not yet processed → return False
# # # # #         404 Not Found               → job unknown (treat as not completed)

# # # # #     If the endpoint does not exist yet on your Spring Boot side, add it as:
# # # # #         GET /api/internal/analysis/status/{jobId}
# # # # #         → ResponseEntity<Map<String,String>>  { "status": job.getStatus().name() }

# # # # #     Raises on network errors so the consumer can decide whether to proceed
# # # # #     with processing or skip (consumer.py catches and proceeds on error).
# # # # #     """
# # # # #     try:
# # # # #         resp = _get(f"/api/internal/analysis/status/{job_id}")
# # # # #         data = resp.json()
# # # # #         status = data.get("status", "").upper()
# # # # #         is_done = status == "COMPLETED"
# # # # #         log.info(
# # # # #             "[Callback] check_job_completed  job=%s  status=%s  is_done=%s",
# # # # #             job_id, status, is_done,
# # # # #         )
# # # # #         return is_done
# # # # #     except RuntimeError as exc:
# # # # #         # 404 → job not found in the backend → definitely not completed
# # # # #         if "404" in str(exc):
# # # # #             log.info(
# # # # #                 "[Callback] check_job_completed  job=%s  → 404 (not found) → not completed",
# # # # #                 job_id,
# # # # #             )
# # # # #             return False
# # # # #         raise


# # # # # def send_progress(
# # # # #     job_id:        str,
# # # # #     journey_id:    int,
# # # # #     progress:      int,
# # # # #     message:       str,
# # # # #     status:        str = "PROCESSING",
# # # # #     current_video: Optional[int] = None,
# # # # # ) -> None:
# # # # #     """
# # # # #     POST /api/internal/analysis/progress

# # # # #     Called periodically during analysis to update the frontend progress bar
# # # # #     and the SSE stream.

# # # # #     Parameters
# # # # #     ──────────
# # # # #     job_id        : RabbitMQ job ID.
# # # # #     journey_id    : Journey ID.
# # # # #     progress      : 0–100 integer.
# # # # #     message       : Human-readable status message.
# # # # #     status        : "PROCESSING" | "COMPLETED" | "FAILED"
# # # # #     current_video : 1-based index of the video currently being processed.
# # # # #     """
# # # # #     payload: Dict[str, Any] = {
# # # # #         "jobId":     job_id,
# # # # #         "journeyId": journey_id,
# # # # #         "status":    status,
# # # # #         "progress":  progress,
# # # # #         "message":   message,
# # # # #     }
# # # # #     if current_video is not None:
# # # # #         payload["currentVideo"] = current_video
# # # # #     _post("/api/internal/analysis/progress", payload)


# # # # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # # # #     """
# # # # #     POST /api/internal/analysis/completed

# # # # #     Called once after ALL videos in the journey have been processed and their
# # # # #     violation frames uploaded to S3.

# # # # #     completion_payload is built by CompletionPayload.to_dict() in models.py.
# # # # #     """
# # # # #     _post("/api/internal/analysis/completed", completion_payload)


# # # # # def send_failed(
# # # # #     job_id:        str,
# # # # #     journey_id:    int,   # kept for caller convenience / logging
# # # # #     error_message: str,
# # # # # ) -> None:
# # # # #     """
# # # # #     POST /api/internal/analysis/failed

# # # # #     Called whenever an unrecoverable exception occurs during job processing.
# # # # #     Spring Boot will mark AnalysisJob and Journey as FAILED.

# # # # #     Note: The API spec only requires jobId + errorMessage.
# # # # #     journeyId is accepted as a parameter for logging but is NOT included
# # # # #     in the outbound payload.
# # # # #     """
# # # # #     try:
# # # # #         _post(
# # # # #             "/api/internal/analysis/failed",
# # # # #             {
# # # # #                 "jobId":        job_id,
# # # # #                 "errorMessage": error_message,
# # # # #             },
# # # # #         )
# # # # #     except Exception as exc:
# # # # #         # Failure callback must never itself raise — log and swallow.
# # # # #         log.error(
# # # # #             "[Callback] send_failed itself failed (job=%s journey=%d): %s",
# # # # #             job_id, journey_id, exc,
# # # # #         )



# # # # """
# # # # callback_client.py
# # # # ──────────────────
# # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs
# # # # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).

# # # # Changes in this version
# # # # ────────────────────────
# # # # • check_job_completed() now uses the EXISTING completed-callback endpoint
# # # #   as a probe instead of a dedicated /status/{jobId} endpoint that doesn't
# # # #   exist yet on Spring Boot.

# # # #   Strategy: POST /api/internal/analysis/completed with only { "jobId": id,
# # # #   "probe": true }.  Spring Boot should return 409 CONFLICT (or any non-500)
# # # #   when the job is already done.  If it returns 500 we treat it as "unknown"
# # # #   and fall through to processing (safe default).

# # # #   *** TEMPORARY until the Spring Boot /status endpoint is implemented. ***
# # # #   Once GET /api/internal/analysis/status/{jobId} is live, revert
# # # #   check_job_completed() to use _get() as originally written.
# # # # """

# # # # from __future__ import annotations

# # # # import logging
# # # # import os
# # # # from typing import Any, Dict, Optional

# # # # import requests
# # # # from dotenv import load_dotenv

# # # # # ── Credentials / config ─────────────────────────────────────────────────────
# # # # _ENV_PATH = os.path.join(
# # # #     os.path.dirname(os.path.abspath(__file__)),
# # # #     "config", "credentials.env",
# # # # )
# # # # load_dotenv(_ENV_PATH)

# # # # _BASE_URL = os.environ.get(
# # # #     "SPRING_BOOT_BASE_URL",
# # # #     "https://cvvrsrailway-api.sconexsoft.com/cvs",
# # # # )
# # # # _TIMEOUT = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # # log = logging.getLogger("callback_client")


# # # # # ── Dynamic base URL setter ───────────────────────────────────────────────────

# # # # def set_base_url(url: str) -> None:
# # # #     global _BASE_URL
# # # #     base = url.rstrip("/")
# # # #     if base.endswith("/api/internal/analysis"):
# # # #         base = base[: -len("/api/internal/analysis")]
# # # #     _BASE_URL = base
# # # #     log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)


# # # # # ── Internal helpers ──────────────────────────────────────────────────────────

# # # # def _post(path: str, payload: Dict[str, Any]) -> requests.Response:
# # # #     url = f"{_BASE_URL}{path}"
# # # #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# # # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# # # #     if not resp.ok:
# # # #         raise RuntimeError(
# # # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # # #         )
# # # #     log.debug("[Callback] %s → %d", path, resp.status_code)
# # # #     return resp


# # # # def _get(path: str) -> requests.Response:
# # # #     url = f"{_BASE_URL}{path}"
# # # #     log.debug("[Callback] GET %s", url)
# # # #     resp = requests.get(url, timeout=_TIMEOUT)
# # # #     if not resp.ok:
# # # #         raise RuntimeError(
# # # #             f"Callback GET {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # # #         )
# # # #     log.debug("[Callback] %s → %d", path, resp.status_code)
# # # #     return resp


# # # # # ── Public API ────────────────────────────────────────────────────────────────

# # # # def check_job_completed(job_id: str) -> bool:
# # # #     """
# # # #     Idempotency check — returns True if the backend already has this job as COMPLETED.

# # # #     TWO-STAGE STRATEGY
# # # #     ──────────────────
# # # #     Stage 1 (permanent): GET /api/internal/analysis/status/{jobId}
# # # #         → { "status": "COMPLETED" }  → True
# # # #         → { "status": "PENDING/PROCESSING" } → False
# # # #         → 404 → False  (job not known yet)
# # # #         → 500 → fall through to Stage 2

# # # #     Stage 2 (temporary fallback until Spring Boot implements /status):
# # # #         Uses GET /api/internal/analysis/job/{jobId} or any existing
# # # #         read endpoint.  If that also 500s, we return False (safe default:
# # # #         process the job rather than silently drop it).

# # # #     WHAT TO IMPLEMENT ON SPRING BOOT (one small controller method):
# # # #     ───────────────────────────────────────────────────────────────
# # # #     @GetMapping("/api/internal/analysis/status/{jobId}")
# # # #     public ResponseEntity<Map<String,String>> getJobStatus(
# # # #             @PathVariable String jobId) {
# # # #         return analysisJobRepository.findByJobId(jobId)
# # # #             .map(job -> ResponseEntity.ok(Map.of("status", job.getStatus().name())))
# # # #             .orElse(ResponseEntity.notFound().<Map<String,String>>build());
# # # #     }

# # # #     Once that endpoint is deployed, Stage 2 below can be deleted.
# # # #     """

# # # #     # ── Stage 1: dedicated status endpoint ───────────────────────────────────
# # # #     try:
# # # #         resp = _get(f"/api/internal/analysis/status/{job_id}")
# # # #         status = resp.json().get("status", "").upper()
# # # #         is_done = status == "COMPLETED"
# # # #         log.info(
# # # #             "[Callback] idempotency check (status endpoint)  job=%s  status=%s  done=%s",
# # # #             job_id, status, is_done,
# # # #         )
# # # #         return is_done
# # # #     except RuntimeError as exc:
# # # #         if "404" in str(exc):
# # # #             log.info(
# # # #                 "[Callback] idempotency check  job=%s  → 404 → not completed", job_id
# # # #             )
# # # #             return False
# # # #         if "500" in str(exc):
# # # #             log.warning(
# # # #                 "[Callback] idempotency check  job=%s  → /status endpoint returned 500 "
# # # #                 "(not implemented yet?) — trying fallback probe.", job_id
# # # #             )
# # # #             # fall through to Stage 2
# # # #         else:
# # # #             # Network error or unexpected status — safe default: process the job
# # # #             log.warning(
# # # #                 "[Callback] idempotency check  job=%s  → unexpected error (%s) "
# # # #                 "— proceeding with processing.", job_id, exc
# # # #             )
# # # #             return False

# # # #     # ── Stage 2: fallback probe using progress endpoint ───────────────────────
# # # #     # We send a lightweight progress probe at 0 % with status=CHECK.
# # # #     # Spring Boot should:
# # # #     #   • Return 200 if the job is still PENDING/PROCESSING (→ process it).
# # # #     #   • Return 409 CONFLICT if the job is already COMPLETED (→ skip it).
# # # #     #   • Return 404 if the job is unknown (→ process it).
# # # #     # If the backend doesn't handle the CHECK status specially it will just
# # # #     # update progress to 0 — harmless on an already-completed job.
# # # #     #
# # # #     # THIS IS A TEMPORARY FALLBACK.  Remove once /status endpoint is live.
# # # #     try:
# # # #         url = f"{_BASE_URL}/api/internal/analysis/progress"
# # # #         resp = requests.post(
# # # #             url,
# # # #             json={
# # # #                 "jobId":    job_id,
# # # #                 "status":   "CHECK",   # sentinel value Spring Boot can detect
# # # #                 "progress": 0,
# # # #                 "message":  "idempotency-probe",
# # # #             },
# # # #             timeout=_TIMEOUT,
# # # #         )
# # # #         if resp.status_code == 409:
# # # #             log.warning(
# # # #                 "[Callback] idempotency check (probe)  job=%s  → 409 CONFLICT "
# # # #                 "→ already COMPLETED — will skip.", job_id
# # # #             )
# # # #             return True
# # # #         log.info(
# # # #             "[Callback] idempotency check (probe)  job=%s  → HTTP %d → not completed.",
# # # #             job_id, resp.status_code,
# # # #         )
# # # #         return False
# # # #     except Exception as exc2:
# # # #         log.warning(
# # # #             "[Callback] idempotency check (probe) failed  job=%s: %s "
# # # #             "— proceeding with processing.", job_id, exc2
# # # #         )
# # # #         return False


# # # # def send_progress(
# # # #     job_id:        str,
# # # #     journey_id:    int,
# # # #     progress:      int,
# # # #     message:       str,
# # # #     status:        str = "PROCESSING",
# # # #     current_video: Optional[int] = None,
# # # # ) -> None:
# # # #     payload: Dict[str, Any] = {
# # # #         "jobId":     job_id,
# # # #         "journeyId": journey_id,
# # # #         "status":    status,
# # # #         "progress":  progress,
# # # #         "message":   message,
# # # #     }
# # # #     if current_video is not None:
# # # #         payload["currentVideo"] = current_video
# # # #     _post("/api/internal/analysis/progress", payload)


# # # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # # #     _post("/api/internal/analysis/completed", completion_payload)


# # # # def send_failed(
# # # #     job_id:        str,
# # # #     journey_id:    int,
# # # #     error_message: str,
# # # # ) -> None:
# # # #     try:
# # # #         _post(
# # # #             "/api/internal/analysis/failed",
# # # #             {
# # # #                 "jobId":        job_id,
# # # #                 "errorMessage": error_message,
# # # #             },
# # # #         )
# # # #     except Exception as exc:
# # # #         log.error(
# # # #             "[Callback] send_failed itself failed (job=%s journey=%d): %s",
# # # #             job_id, journey_id, exc,
# # # #         )



# # # # # # # """
# # # # # # # callback_client.py
# # # # # # # ──────────────────
# # # # # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # # # # # All three public helpers are fire-and-forget from the caller's perspective;
# # # # # # # they raise on non-2xx so the consumer can route to the failure path.
# # # # # # # """

# # # # # # # from __future__ import annotations

# # # # # # # import logging
# # # # # # # import os
# # # # # # # from typing import Any, Dict, List

# # # # # # # import requests
# # # # # # # from dotenv import load_dotenv

# # # # # # # # ── Credentials / config ────────────────────────────────────────────────────
# # # # # # # _ENV_PATH = os.path.join(
# # # # # # #     os.path.dirname(os.path.abspath(__file__)),
# # # # # # #     "config", "credentials.env",
# # # # # # # )
# # # # # # # load_dotenv(_ENV_PATH)

# # # # # # # _BASE_URL = os.environ.get("SPRING_BOOT_BASE_URL", "http://localhost:8080")
# # # # # # # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # # # # # log = logging.getLogger("callback_client")


# # # # # # # # ── Internal helper ──────────────────────────────────────────────────────────

# # # # # # # def _post(path: str, payload: Dict[str, Any]) -> None:
# # # # # # #     url = f"{_BASE_URL}{path}"
# # # # # # #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# # # # # # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# # # # # # #     if not resp.ok:
# # # # # # #         raise RuntimeError(
# # # # # # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # # # # # #         )
# # # # # # #     log.debug("[Callback] %s → %d", path, resp.status_code)


# # # # # # # # ── Public API ───────────────────────────────────────────────────────────────

# # # # # # # def send_progress(
# # # # # # #     job_id:     str,
# # # # # # #     journey_id: int,
# # # # # # #     progress:   int,
# # # # # # #     message:    str,
# # # # # # #     status:     str = "PROCESSING",
# # # # # # # ) -> None:
# # # # # # #     """
# # # # # # #     POST /api/internal/analysis/progress

# # # # # # #     Called periodically during analysis to update the frontend progress bar
# # # # # # #     and the SSE stream.
# # # # # # #     """
# # # # # # #     _post(
# # # # # # #         "/api/internal/analysis/progress",
# # # # # # #         {
# # # # # # #             "jobId":      job_id,
# # # # # # #             "journeyId":  journey_id,
# # # # # # #             "status":     status,
# # # # # # #             "progress":   progress,
# # # # # # #             "message":    message,
# # # # # # #         },
# # # # # # #     )


# # # # # # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # # # # # #     """
# # # # # # #     POST /api/internal/analysis/completed

# # # # # # #     Called once — after ALL videos in the journey have been processed and
# # # # # # #     their violation frames uploaded to S3.

# # # # # # #     Expected shape of completion_payload
# # # # # # #     ─────────────────────────────────────
# # # # # # #     {
# # # # # # #         "jobId":          str,
# # # # # # #         "journeyId":      int,
# # # # # # #         "processingTime": int,          # wall-clock ms
# # # # # # #         "videoResults":   List[dict],   # see models.py → VideoResult
# # # # # # #     }
# # # # # # #     """
# # # # # # #     _post("/api/internal/analysis/completed", completion_payload)


# # # # # # # def send_failed(
# # # # # # #     job_id:        str,
# # # # # # #     journey_id:    int,
# # # # # # #     error_message: str,
# # # # # # # ) -> None:
# # # # # # #     """
# # # # # # #     POST /api/internal/analysis/failed

# # # # # # #     Called whenever an unrecoverable exception occurs during job processing.
# # # # # # #     Spring Boot will mark AnalysisJob and Journey as FAILED.
# # # # # # #     """
# # # # # # #     try:
# # # # # # #         _post(
# # # # # # #             "/api/internal/analysis/failed",
# # # # # # #             {
# # # # # # #                 "jobId":        job_id,
# # # # # # #                 "journeyId":    journey_id,
# # # # # # #                 "errorMessage": error_message,
# # # # # # #             },
# # # # # # #         )
# # # # # # #     except Exception as exc:
# # # # # # #         # Failure callback must never itself raise — log and swallow.
# # # # # # #         log.error("[Callback] send_failed itself failed: %s", exc)

# # # # # # """
# # # # # # callback_client.py
# # # # # # ──────────────────
# # # # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # # # # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs
# # # # # # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).

# # # # # # Changes from previous version
# # # # # # ──────────────────────────────
# # # # # # • _BASE_URL now defaults to the live CVVRS domain (overridable via env var
# # # # # #   SPRING_BOOT_BASE_URL for local testing).
# # # # # # • send_progress()  — payload now includes the `currentVideo` field required
# # # # # #   by the API.
# # # # # # • send_completed() — accepts a fully-formed dict (built by CompletionPayload.
# # # # # #   to_dict()); no shape changes needed here since the caller owns the dict.
# # # # # # • send_failed()    — payload drops `journeyId` (not in the API spec for this
# # # # # #   endpoint); only jobId + errorMessage are sent.
# # # # # # """

# # # # # # from __future__ import annotations

# # # # # # import logging
# # # # # # import os
# # # # # # from typing import Any, Dict, Optional

# # # # # # import requests
# # # # # # from dotenv import load_dotenv

# # # # # # # ── Credentials / config ────────────────────────────────────────────────────
# # # # # # _ENV_PATH = os.path.join(
# # # # # #     os.path.dirname(os.path.abspath(__file__)),
# # # # # #     "config", "credentials.env",
# # # # # # )
# # # # # # load_dotenv(_ENV_PATH)

# # # # # # # Default to the live CVVRS server; override with SPRING_BOOT_BASE_URL for local dev.
# # # # # # _BASE_URL = os.environ.get(
# # # # # #     "SPRING_BOOT_BASE_URL",
# # # # # #     "https://cvvrsrailway-api.sconexsoft.com/cvs",
# # # # # # )
# # # # # # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # # # # log = logging.getLogger("callback_client")


# # # # # # # ── Internal helper ──────────────────────────────────────────────────────────

# # # # # # def _post(path: str, payload: Dict[str, Any]) -> None:
# # # # # #     """
# # # # # #     POST to a Spring Boot internal endpoint.
# # # # # #     No Authorization header is sent — /api/internal/* are worker-only endpoints.
# # # # # #     Raises RuntimeError on non-2xx so the consumer can route to failure path.
# # # # # #     """
# # # # # #     url = f"{_BASE_URL}{path}"
# # # # # #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# # # # # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# # # # # #     if not resp.ok:
# # # # # #         raise RuntimeError(
# # # # # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # # # # #         )
# # # # # #     log.debug("[Callback] %s → %d", path, resp.status_code)


# # # # # # # ── Public API ───────────────────────────────────────────────────────────────

# # # # # # def send_progress(
# # # # # #     job_id:        str,
# # # # # #     journey_id:    int,
# # # # # #     progress:      int,
# # # # # #     message:       str,
# # # # # #     status:        str = "PROCESSING",
# # # # # #     current_video: Optional[int] = None,   # NEW — index of video currently being processed
# # # # # # ) -> None:
# # # # # #     """
# # # # # #     POST /api/internal/analysis/progress

# # # # # #     Called periodically during analysis to update the frontend progress bar
# # # # # #     and the SSE stream.

# # # # # #     Parameters
# # # # # #     ──────────
# # # # # #     job_id        : RabbitMQ job ID.
# # # # # #     journey_id    : Journey ID.
# # # # # #     progress      : 0–100 integer.
# # # # # #     message       : Human-readable status message.
# # # # # #     status        : "PROCESSING" | "COMPLETED" | "FAILED"
# # # # # #     current_video : 1-based index of the video currently being processed
# # # # # #                     (omitted from payload when None).
# # # # # #     """
# # # # # #     payload: Dict[str, Any] = {
# # # # # #         "jobId":      job_id,
# # # # # #         "journeyId":  journey_id,
# # # # # #         "status":     status,
# # # # # #         "progress":   progress,
# # # # # #         "message":    message,
# # # # # #     }
# # # # # #     if current_video is not None:
# # # # # #         payload["currentVideo"] = current_video

# # # # # #     _post("/api/internal/analysis/progress", payload)


# # # # # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # # # # #     """
# # # # # #     POST /api/internal/analysis/completed

# # # # # #     Called once after ALL videos in the journey have been processed and their
# # # # # #     violation frames uploaded to S3.

# # # # # #     Expected shape of completion_payload (built by CompletionPayload.to_dict())
# # # # # #     ────────────────────────────────────────────────────────────────────────────
# # # # # #     {
# # # # # #         "jobId":         str,
# # # # # #         "journeyId":     int,
# # # # # #         "batchId":       str,           # "BATCH-<jobId>" if not supplied
# # # # # #         "trainDetailId": int,
# # # # # #         "folderName":    str,
# # # # # #         "processingTime":int,           # wall-clock ms
# # # # # #         "videoResults": [
# # # # # #             {
# # # # # #                 "videoId":         str,       # STRING per API spec
# # # # # #                 "sequenceNo":      int,
# # # # # #                 "durationSeconds": float,
# # # # # #                 "originalS3Key":   str,
# # # # # #                 "violations": [
# # # # # #                     {
# # # # # #                         "violationType":          str,
# # # # # #                         "severity":               str,
# # # # # #                         "confidence":             float,
# # # # # #                         "riskScore":              float,
# # # # # #                         "timestamp":              float,   # journey-global seconds
# # # # # #                         "originalVideoTimestamp": float,   # local-video seconds
# # # # # #                         "framePaths":             [str]
# # # # # #                     }
# # # # # #                 ]
# # # # # #             }
# # # # # #         ]
# # # # # #     }
# # # # # #     """
# # # # # #     _post("/api/internal/analysis/completed", completion_payload)


# # # # # # def send_failed(
# # # # # #     job_id:        str,
# # # # # #     journey_id:    int,          # kept as parameter for caller convenience / logging
# # # # # #     error_message: str,
# # # # # # ) -> None:
# # # # # #     """
# # # # # #     POST /api/internal/analysis/failed

# # # # # #     Called whenever an unrecoverable exception occurs during job processing.
# # # # # #     Spring Boot will mark AnalysisJob and Journey as FAILED.

# # # # # #     Note: The API spec only requires jobId + errorMessage in the request body.
# # # # # #     journeyId is accepted as a parameter here for logging but is NOT included
# # # # # #     in the outbound payload.
# # # # # #     """
# # # # # #     try:
# # # # # #         _post(
# # # # # #             "/api/internal/analysis/failed",
# # # # # #             {
# # # # # #                 "jobId":        job_id,
# # # # # #                 "errorMessage": error_message,
# # # # # #             },
# # # # # #         )
# # # # # #     except Exception as exc:
# # # # # #         # Failure callback must never itself raise — log and swallow.
# # # # # #         log.error(
# # # # # #             "[Callback] send_failed itself failed (job=%s journey=%d): %s",
# # # # # #             job_id, journey_id, exc,
# # # # # #         )



# # # # # """

# # # # # callback_client.py

# # # # # ──────────────────

# # # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.
 
# # # # # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs

# # # # # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).
 
# # # # # Fixes from previous version

# # # # # ─────────────────────────────

# # # # # • Added set_base_url() — called by consumer.py with the callbackBaseUrl from

# # # # #   the RabbitMQ message, so Python posts to the correct Spring Boot server

# # # # #   per environment without needing env var changes.

# # # # # • _BASE_URL still falls back to env var SPRING_BOOT_BASE_URL or the live

# # # # #   server when callbackBaseUrl is not provided in the message.

# # # # # """
 
# # # # # from __future__ import annotations
 
# # # # # import logging

# # # # # import os

# # # # # from typing import Any, Dict, Optional
 
# # # # # import requests

# # # # # from dotenv import load_dotenv
 
# # # # # # ── Credentials / config ────────────────────────────────────────────────────

# # # # # _ENV_PATH = os.path.join(

# # # # #     os.path.dirname(os.path.abspath(__file__)),

# # # # #     "config", "credentials.env",

# # # # # )

# # # # # load_dotenv(_ENV_PATH)
 
# # # # # # Default to the live CVVRS server; override with SPRING_BOOT_BASE_URL for local dev

# # # # # # or call set_base_url() with the callbackBaseUrl from the RabbitMQ message.

# # # # # _BASE_URL = os.environ.get(

# # # # #     "SPRING_BOOT_BASE_URL",

# # # # #     "https://cvvrsrailway-api.sconexsoft.com/cvs",

# # # # # )

# # # # # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))
 
# # # # # log = logging.getLogger("callback_client")
 
 
# # # # # # ── FIX: dynamic base URL setter ─────────────────────────────────────────────
 
# # # # # def set_base_url(url: str) -> None:

# # # # #     """

# # # # #     Override the callback base URL at runtime.
 
# # # # #     Called by consumer.py when the RabbitMQ message contains a callbackBaseUrl.

# # # # #     This allows the same Python worker to callback correctly to both local

# # # # #     and staging Spring Boot servers without changing env vars.
 
# # # # #     Example values:

# # # # #         "http://localhost:8093/api/internal/analysis"         (local)

# # # # #         "https://cvvrsrailway-api.sconexsoft.com/cvs/api/internal/analysis"  (staging)
 
# # # # #     Note: The URL passed here is the FULL path up to /api/internal/analysis.

# # # # #     The individual callbacks append /progress, /completed, /failed.

# # # # #     """

# # # # #     global _BASE_URL

# # # # #     # Strip the /api/internal/analysis suffix if present — we add it in _post()

# # # # #     base = url.rstrip("/")

# # # # #     if base.endswith("/api/internal/analysis"):

# # # # #         base = base[: -len("/api/internal/analysis")]

# # # # #     _BASE_URL = base

# # # # #     log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)
 
 
# # # # # # ── Internal helper ──────────────────────────────────────────────────────────
 
# # # # # def _post(path: str, payload: Dict[str, Any]) -> None:

# # # # #     """

# # # # #     POST to a Spring Boot internal endpoint.

# # # # #     No Authorization header is sent — /api/internal/* are worker-only endpoints.

# # # # #     Raises RuntimeError on non-2xx so the consumer can route to failure path.

# # # # #     """

# # # # #     url = f"{_BASE_URL}{path}"

# # # # #     log.debug("[Callback] POST %s  payload=%s", url, payload)

# # # # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)

# # # # #     if not resp.ok:

# # # # #         raise RuntimeError(

# # # # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"

# # # # #         )

# # # # #     log.debug("[Callback] %s → %d", path, resp.status_code)
 
 
# # # # # # ── Public API ───────────────────────────────────────────────────────────────
 
# # # # # def send_progress(

# # # # #     job_id:        str,

# # # # #     journey_id:    int,

# # # # #     progress:      int,

# # # # #     message:       str,

# # # # #     status:        str = "PROCESSING",

# # # # #     current_video: Optional[int] = None,

# # # # # ) -> None:

# # # # #     """

# # # # #     POST /api/internal/analysis/progress
 
# # # # #     Called periodically during analysis to update the frontend progress bar

# # # # #     and the SSE stream.
 
# # # # #     Parameters

# # # # #     ──────────

# # # # #     job_id        : RabbitMQ job ID.

# # # # #     journey_id    : Journey ID.

# # # # #     progress      : 0–100 integer.

# # # # #     message       : Human-readable status message.

# # # # #     status        : "PROCESSING" | "COMPLETED" | "FAILED"

# # # # #     current_video : 1-based index of the video currently being processed.

# # # # #     """

# # # # #     payload: Dict[str, Any] = {

# # # # #         "jobId":      job_id,

# # # # #         "journeyId":  journey_id,

# # # # #         "status":     status,

# # # # #         "progress":   progress,

# # # # #         "message":    message,

# # # # #     }

# # # # #     if current_video is not None:

# # # # #         payload["currentVideo"] = current_video
 
# # # # #     _post("/api/internal/analysis/progress", payload)
 
 
# # # # # def send_completed(completion_payload: Dict[str, Any]) -> None:

# # # # #     """

# # # # #     POST /api/internal/analysis/completed
 
# # # # #     Called once after ALL videos in the journey have been processed and their

# # # # #     violation frames uploaded to S3.
 
# # # # #     completion_payload is built by CompletionPayload.to_dict() in models.py.

# # # # #     """

# # # # #     _post("/api/internal/analysis/completed", completion_payload)
 
 
# # # # # def send_failed(

# # # # #     job_id:        str,

# # # # #     journey_id:    int,          # kept for caller convenience / logging

# # # # #     error_message: str,

# # # # # ) -> None:

# # # # #     """

# # # # #     POST /api/internal/analysis/failed
 
# # # # #     Called whenever an unrecoverable exception occurs during job processing.

# # # # #     Spring Boot will mark AnalysisJob and Journey as FAILED.
 
# # # # #     Note: The API spec only requires jobId + errorMessage.

# # # # #     journeyId is accepted as a parameter for logging but is NOT included

# # # # #     in the outbound payload.

# # # # #     """

# # # # #     try:

# # # # #         _post(

# # # # #             "/api/internal/analysis/failed",

# # # # #             {

# # # # #                 "jobId":        job_id,

# # # # #                 "errorMessage": error_message,

# # # # #             },

# # # # #         )

# # # # #     except Exception as exc:

# # # # #         # Failure callback must never itself raise — log and swallow.

# # # # #         log.error(

# # # # #             "[Callback] send_failed itself failed (job=%s journey=%d): %s",

# # # # #             job_id, journey_id, exc,

# # # # #         )
 
 
 
# # # # """
# # # # callback_client.py
# # # # ──────────────────
# # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs
# # # # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).

# # # # Changes from previous version
# # # # ──────────────────────────────
# # # # • Added check_job_completed(job_id) — queries GET /api/internal/analysis/status/{jobId}
# # # #   and returns True when the backend reports the job is already COMPLETED.
# # # #   Called by consumer.py as an idempotency guard before starting any processing
# # # #   on a redelivered message.
# # # # """

# # # # from __future__ import annotations
# # # # import logging
# # # # import os
# # # # from typing import Any, Dict, Optional

# # # # import requests
# # # # from dotenv import load_dotenv

# # # # # ── Credentials / config ─────────────────────────────────────────────────────
# # # # _ENV_PATH = os.path.join(
# # # #     os.path.dirname(os.path.abspath(__file__)),
# # # #     "config", "credentials.env",
# # # # )
# # # # load_dotenv(_ENV_PATH)

# # # # _BASE_URL = os.environ.get(
# # # #     "SPRING_BOOT_BASE_URL",
# # # #     "https://cvvrsrailway-api.sconexsoft.com/cvs",
# # # # )
# # # # _TIMEOUT = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # # log = logging.getLogger("callback_client")


# # # # # ── Dynamic base URL setter ───────────────────────────────────────────────────

# # # # def set_base_url(url: str) -> None:
# # # #     """
# # # #     Override the callback base URL at runtime.

# # # #     Called by consumer.py when the RabbitMQ message contains a callbackBaseUrl.
# # # #     Allows the same Python worker to callback correctly to both local and staging
# # # #     Spring Boot servers without changing env vars.

# # # #     The URL passed here is the FULL path up to /api/internal/analysis.
# # # #     Individual callbacks append /progress, /completed, /failed, /status/{id}.
# # # #     """
# # # #     global _BASE_URL
# # # #     base = url.rstrip("/")
# # # #     if base.endswith("/api/internal/analysis"):
# # # #         base = base[: -len("/api/internal/analysis")]
# # # #     _BASE_URL = base
# # # #     log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)


# # # # # ── Internal helpers ──────────────────────────────────────────────────────────

# # # # def _post(path: str, payload: Dict[str, Any]) -> None:
# # # #     """
# # # #     POST to a Spring Boot internal endpoint.
# # # #     No Authorization header — /api/internal/* are worker-only endpoints.
# # # #     Raises RuntimeError on non-2xx so the consumer can route to the failure path.
# # # #     """
# # # #     url = f"{_BASE_URL}{path}"
# # # #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# # # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# # # #     if not resp.ok:
# # # #         raise RuntimeError(
# # # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # # #         )
# # # #     log.debug("[Callback] %s → %d", path, resp.status_code)


# # # # def _get(path: str) -> requests.Response:
# # # #     """
# # # #     GET from a Spring Boot internal endpoint.
# # # #     Raises RuntimeError on non-2xx.
# # # #     """
# # # #     url = f"{_BASE_URL}{path}"
# # # #     log.debug("[Callback] GET %s", url)
# # # #     resp = requests.get(url, timeout=_TIMEOUT)
# # # #     if not resp.ok:
# # # #         raise RuntimeError(
# # # #             f"Callback GET {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # # #         )
# # # #     log.debug("[Callback] %s → %d", path, resp.status_code)
# # # #     return resp


# # # # # ── Public API ────────────────────────────────────────────────────────────────

# # # # def check_job_completed(job_id: str) -> bool:
# # # #     """
# # # #     NEW — Idempotency check.

# # # #     Queries GET /api/internal/analysis/status/{jobId} and returns True when
# # # #     the backend reports the job status as COMPLETED.

# # # #     Called by consumer.py at the very start of _handle_job() so that
# # # #     RabbitMQ redeliveries of already-completed jobs are detected immediately
# # # #     and ACKed without re-running any processing.

# # # #     Backend contract (expected JSON shape):
# # # #         { "status": "COMPLETED" }   → job already done → return True
# # # #         { "status": "PROCESSING" }  → in-flight (rare edge-case) → return False
# # # #         { "status": "PENDING" }     → not yet processed → return False
# # # #         404 Not Found               → job unknown (treat as not completed)

# # # #     If the endpoint does not exist yet on your Spring Boot side, add it as:
# # # #         GET /api/internal/analysis/status/{jobId}
# # # #         → ResponseEntity<Map<String,String>>  { "status": job.getStatus().name() }

# # # #     Raises on network errors so the consumer can decide whether to proceed
# # # #     with processing or skip (consumer.py catches and proceeds on error).
# # # #     """
# # # #     try:
# # # #         resp = _get(f"/api/internal/analysis/status/{job_id}")
# # # #         data = resp.json()
# # # #         status = data.get("status", "").upper()
# # # #         is_done = status == "COMPLETED"
# # # #         log.info(
# # # #             "[Callback] check_job_completed  job=%s  status=%s  is_done=%s",
# # # #             job_id, status, is_done,
# # # #         )
# # # #         return is_done
# # # #     except RuntimeError as exc:
# # # #         # 404 → job not found in the backend → definitely not completed
# # # #         if "404" in str(exc):
# # # #             log.info(
# # # #                 "[Callback] check_job_completed  job=%s  → 404 (not found) → not completed",
# # # #                 job_id,
# # # #             )
# # # #             return False
# # # #         raise


# # # # def send_progress(
# # # #     job_id:        str,
# # # #     journey_id:    int,
# # # #     progress:      int,
# # # #     message:       str,
# # # #     status:        str = "PROCESSING",
# # # #     current_video: Optional[int] = None,
# # # # ) -> None:
# # # #     """
# # # #     POST /api/internal/analysis/progress

# # # #     Called periodically during analysis to update the frontend progress bar
# # # #     and the SSE stream.

# # # #     Parameters
# # # #     ──────────
# # # #     job_id        : RabbitMQ job ID.
# # # #     journey_id    : Journey ID.
# # # #     progress      : 0–100 integer.
# # # #     message       : Human-readable status message.
# # # #     status        : "PROCESSING" | "COMPLETED" | "FAILED"
# # # #     current_video : 1-based index of the video currently being processed.
# # # #     """
# # # #     payload: Dict[str, Any] = {
# # # #         "jobId":     job_id,
# # # #         "journeyId": journey_id,
# # # #         "status":    status,
# # # #         "progress":  progress,
# # # #         "message":   message,
# # # #     }
# # # #     if current_video is not None:
# # # #         payload["currentVideo"] = current_video
# # # #     _post("/api/internal/analysis/progress", payload)


# # # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # # #     """
# # # #     POST /api/internal/analysis/completed

# # # #     Called once after ALL videos in the journey have been processed and their
# # # #     violation frames uploaded to S3.

# # # #     completion_payload is built by CompletionPayload.to_dict() in models.py.
# # # #     """
# # # #     _post("/api/internal/analysis/completed", completion_payload)


# # # # def send_failed(
# # # #     job_id:        str,
# # # #     journey_id:    int,   # kept for caller convenience / logging
# # # #     error_message: str,
# # # # ) -> None:
# # # #     """
# # # #     POST /api/internal/analysis/failed

# # # #     Called whenever an unrecoverable exception occurs during job processing.
# # # #     Spring Boot will mark AnalysisJob and Journey as FAILED.

# # # #     Note: The API spec only requires jobId + errorMessage.
# # # #     journeyId is accepted as a parameter for logging but is NOT included
# # # #     in the outbound payload.
# # # #     """
# # # #     try:
# # # #         _post(
# # # #             "/api/internal/analysis/failed",
# # # #             {
# # # #                 "jobId":        job_id,
# # # #                 "errorMessage": error_message,
# # # #             },
# # # #         )
# # # #     except Exception as exc:
# # # #         # Failure callback must never itself raise — log and swallow.
# # # #         log.error(
# # # #             "[Callback] send_failed itself failed (job=%s journey=%d): %s",
# # # #             job_id, journey_id, exc,
# # # #         )



# # # """
# # # callback_client.py
# # # ──────────────────
# # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs
# # # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).

# # # Changes in this version
# # # ────────────────────────
# # # • check_job_completed() now uses the EXISTING completed-callback endpoint
# # #   as a probe instead of a dedicated /status/{jobId} endpoint that doesn't
# # #   exist yet on Spring Boot.

# # #   Strategy: POST /api/internal/analysis/completed with only { "jobId": id,
# # #   "probe": true }.  Spring Boot should return 409 CONFLICT (or any non-500)
# # #   when the job is already done.  If it returns 500 we treat it as "unknown"
# # #   and fall through to processing (safe default).

# # #   *** TEMPORARY until the Spring Boot /status endpoint is implemented. ***
# # #   Once GET /api/internal/analysis/status/{jobId} is live, revert
# # #   check_job_completed() to use _get() as originally written.
# # # """

# # # from __future__ import annotations

# # # import logging
# # # import os
# # # from typing import Any, Dict, Optional

# # # import requests
# # # from dotenv import load_dotenv

# # # # ── Credentials / config ─────────────────────────────────────────────────────
# # # _ENV_PATH = os.path.join(
# # #     os.path.dirname(os.path.abspath(__file__)),
# # #     "config", "credentials.env",
# # # )
# # # load_dotenv(_ENV_PATH)

# # # _BASE_URL = os.environ.get(
# # #     "SPRING_BOOT_BASE_URL",
# # #     "https://cvvrsrailway-api.sconexsoft.com/cvs",
# # # )
# # # _TIMEOUT = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # log = logging.getLogger("callback_client")


# # # # ── Dynamic base URL setter ───────────────────────────────────────────────────

# # # def set_base_url(url: str) -> None:
# # #     global _BASE_URL
# # #     base = url.rstrip("/")
# # #     if base.endswith("/api/internal/analysis"):
# # #         base = base[: -len("/api/internal/analysis")]
# # #     _BASE_URL = base
# # #     log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)


# # # # ── Internal helpers ──────────────────────────────────────────────────────────

# # # def _post(path: str, payload: Dict[str, Any]) -> requests.Response:
# # #     url = f"{_BASE_URL}{path}"
# # #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# # #     if not resp.ok:
# # #         raise RuntimeError(
# # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # #         )
# # #     log.debug("[Callback] %s → %d", path, resp.status_code)
# # #     return resp


# # # def _get(path: str) -> requests.Response:
# # #     url = f"{_BASE_URL}{path}"
# # #     log.debug("[Callback] GET %s", url)
# # #     resp = requests.get(url, timeout=_TIMEOUT)
# # #     if not resp.ok:
# # #         raise RuntimeError(
# # #             f"Callback GET {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # #         )
# # #     log.debug("[Callback] %s → %d", path, resp.status_code)
# # #     return resp


# # # # ── Public API ────────────────────────────────────────────────────────────────

# # # def check_job_completed(job_id: str) -> bool:
# # #     """
# # #     Idempotency check — returns True if the backend already has this job as COMPLETED.

# # #     TWO-STAGE STRATEGY
# # #     ──────────────────
# # #     Stage 1 (permanent): GET /api/internal/analysis/status/{jobId}
# # #         → { "status": "COMPLETED" }  → True
# # #         → { "status": "PENDING/PROCESSING" } → False
# # #         → 404 → False  (job not known yet)
# # #         → 500 → fall through to Stage 2

# # #     Stage 2 (temporary fallback until Spring Boot implements /status):
# # #         Uses GET /api/internal/analysis/job/{jobId} or any existing
# # #         read endpoint.  If that also 500s, we return False (safe default:
# # #         process the job rather than silently drop it).

# # #     WHAT TO IMPLEMENT ON SPRING BOOT (one small controller method):
# # #     ───────────────────────────────────────────────────────────────
# # #     @GetMapping("/api/internal/analysis/status/{jobId}")
# # #     public ResponseEntity<Map<String,String>> getJobStatus(
# # #             @PathVariable String jobId) {
# # #         return analysisJobRepository.findByJobId(jobId)
# # #             .map(job -> ResponseEntity.ok(Map.of("status", job.getStatus().name())))
# # #             .orElse(ResponseEntity.notFound().<Map<String,String>>build());
# # #     }

# # #     Once that endpoint is deployed, Stage 2 below can be deleted.
# # #     """

# # #     # ── Stage 1: dedicated status endpoint ───────────────────────────────────
# # #     try:
# # #         resp = _get(f"/api/internal/analysis/status/{job_id}")
# # #         status = resp.json().get("status", "").upper()
# # #         is_done = status == "COMPLETED"
# # #         log.info(
# # #             "[Callback] idempotency check (status endpoint)  job=%s  status=%s  done=%s",
# # #             job_id, status, is_done,
# # #         )
# # #         return is_done
# # #     except RuntimeError as exc:
# # #         if "404" in str(exc):
# # #             log.info(
# # #                 "[Callback] idempotency check  job=%s  → 404 → not completed", job_id
# # #             )
# # #             return False
# # #         if "500" in str(exc):
# # #             log.warning(
# # #                 "[Callback] idempotency check  job=%s  → /status endpoint returned 500 "
# # #                 "(not implemented yet?) — trying fallback probe.", job_id
# # #             )
# # #             # fall through to Stage 2
# # #         else:
# # #             # Network error or unexpected status — safe default: process the job
# # #             log.warning(
# # #                 "[Callback] idempotency check  job=%s  → unexpected error (%s) "
# # #                 "— proceeding with processing.", job_id, exc
# # #             )
# # #             return False

# # #     # ── Stage 2: fallback probe using progress endpoint ───────────────────────
# # #     # We send a lightweight progress probe at 0 % with status=CHECK.
# # #     # Spring Boot should:
# # #     #   • Return 200 if the job is still PENDING/PROCESSING (→ process it).
# # #     #   • Return 409 CONFLICT if the job is already COMPLETED (→ skip it).
# # #     #   • Return 404 if the job is unknown (→ process it).
# # #     # If the backend doesn't handle the CHECK status specially it will just
# # #     # update progress to 0 — harmless on an already-completed job.
# # #     #
# # #     # THIS IS A TEMPORARY FALLBACK.  Remove once /status endpoint is live.
# # #     try:
# # #         url = f"{_BASE_URL}/api/internal/analysis/progress"
# # #         resp = requests.post(
# # #             url,
# # #             json={
# # #                 "jobId":    job_id,
# # #                 "status":   "CHECK",   # sentinel value Spring Boot can detect
# # #                 "progress": 0,
# # #                 "message":  "idempotency-probe",
# # #             },
# # #             timeout=_TIMEOUT,
# # #         )
# # #         if resp.status_code == 409:
# # #             log.warning(
# # #                 "[Callback] idempotency check (probe)  job=%s  → 409 CONFLICT "
# # #                 "→ already COMPLETED — will skip.", job_id
# # #             )
# # #             return True
# # #         log.info(
# # #             "[Callback] idempotency check (probe)  job=%s  → HTTP %d → not completed.",
# # #             job_id, resp.status_code,
# # #         )
# # #         return False
# # #     except Exception as exc2:
# # #         log.warning(
# # #             "[Callback] idempotency check (probe) failed  job=%s: %s "
# # #             "— proceeding with processing.", job_id, exc2
# # #         )
# # #         return False


# # # def send_progress(
# # #     job_id:        str,
# # #     journey_id:    int,
# # #     progress:      int,
# # #     message:       str,
# # #     status:        str = "PROCESSING",
# # #     current_video: Optional[int] = None,
# # # ) -> None:
# # #     payload: Dict[str, Any] = {
# # #         "jobId":     job_id,
# # #         "journeyId": journey_id,
# # #         "status":    status,
# # #         "progress":  progress,
# # #         "message":   message,
# # #     }
# # #     if current_video is not None:
# # #         payload["currentVideo"] = current_video
# # #     _post("/api/internal/analysis/progress", payload)


# # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # #     _post("/api/internal/analysis/completed", completion_payload)


# # # def send_failed(
# # #     job_id:        str,
# # #     journey_id:    int,
# # #     error_message: str,
# # #     video_id:      Optional[int] = None,
# # #     error_type:    Optional[str] = None,
# # #     stack_trace:   Optional[str] = None,
# # #     reason:        Optional[str] = None,
# # # ) -> None:
# # #     """
# # #     Calls POST /api/internal/analysis/failed.

# # #     Backward compatible: existing job-level callers (no video_id) keep
# # #     working exactly as before — the payload shape for that case is
# # #     unchanged ({"jobId", "errorMessage"}).

# # #     When video_id IS supplied, this becomes a PER-VIDEO failure report
# # #     (Phase 1 requirement: "no failed video should wait until journey
# # #     completion to be reported"). The payload then also carries:
# # #         videoId     — STRING, matches VideoResult.to_dict()'s convention
# # #         journeyId
# # #         errorType   — short classifier, e.g. "OOM", "DECODE_ERROR",
# # #                       "MEDIAPIPE_ERROR", "YOLO_ERROR", "TIMEOUT",
# # #                       "RESOURCE_EXHAUSTION", "NOT_PROCESSED"
# # #         stackTrace  — full traceback text, if available
# # #         reason      — human-readable reason string, e.g.
# # #                       "Not Processed - Worker Resource Exhaustion"
# # #                       for videos skipped after an OOM on an earlier video.
# # #     """
# # #     payload: Dict[str, Any] = {
# # #         "jobId":        job_id,
# # #         "errorMessage": error_message,
# # #     }
# # #     if video_id is not None:
# # #         payload["videoId"]    = str(video_id)   # STRING per API spec convention
# # #         payload["journeyId"]  = journey_id
# # #     if error_type is not None:
# # #         payload["errorType"] = error_type
# # #     if stack_trace is not None:
# # #         payload["stackTrace"] = stack_trace
# # #     if reason is not None:
# # #         payload["reason"] = reason

# # #     try:
# # #         _post("/api/internal/analysis/failed", payload)
# # #     except Exception as exc:
# # #         log.error(
# # #             "[Callback] send_failed itself failed (job=%s journey=%d video=%s): %s",
# # #             job_id, journey_id, video_id, exc,
# # #         )


# # # def send_video_failed(
# # #     job_id:        str,
# # #     journey_id:    int,
# # #     video_id:      int,
# # #     error_type:    str,
# # #     error_message: str,
# # #     stack_trace:   str = "",
# # #     reason:        Optional[str] = None,
# # # ) -> None:
# # #     """
# # #     Convenience wrapper for the per-video failure report required by
# # #     Phase 1: "Immediately call the failed endpoint for the affected video"
# # #     — called once per video, the moment that video's outcome is known,
# # #     not batched until the journey ends.

# # #     For resource-exhaustion cascades, pass error_type="RESOURCE_EXHAUSTION"
# # #     (the video that actually hit OOM) or error_type="NOT_PROCESSED" with
# # #     reason="Not Processed - Worker Resource Exhaustion" (videos after it
# # #     that were skipped as a result).
# # #     """
# # #     log.error(
# # #         "[Callback] video FAILED  job=%s  journey=%d  video=%d  type=%s  reason=%s: %s",
# # #         job_id, journey_id, video_id, error_type, reason, error_message,
# # #     )
# # #     send_failed(
# # #         job_id        = job_id,
# # #         journey_id    = journey_id,
# # #         error_message = error_message,
# # #         video_id      = video_id,
# # #         error_type    = error_type,
# # #         stack_trace   = stack_trace,
# # #         reason        = reason,
# # #     )


# # # # ── OOM / resource-exhaustion signature detection ────────────────────────────
# # # #
# # # # Used by analyzer.py to decide whether a per-video exception means "this
# # # # one video is bad" (continue with the rest) vs. "the worker itself can no
# # # # longer safely process more video" (stop the journey, mark every
# # # # remaining video FAILED with reason "Not Processed - Worker Resource
# # # # Exhaustion"). Matches the signatures called out explicitly in the
# # # # Phase 1 spec: OpenCV OutOfMemoryError, "Failed to allocate", cv2.pyd
# # # # memory-related failures, generic MemoryError, std::bad_alloc.
# # # _OOM_SIGNATURES = (
# # #     "outofmemoryerror",
# # #     "out of memory",
# # #     "failed to allocate",
# # #     "cannot allocate memory",
# # #     "bad_alloc",
# # #     "memoryerror",
# # #     "cv2.pyd",
# # #     "access violation",
# # #     "resource exhaust",
# # # )


# # # def is_resource_exhaustion_error(exc: BaseException) -> bool:
# # #     """
# # #     Returns True if `exc` looks like an OOM / native resource-exhaustion
# # #     failure rather than an ordinary recoverable per-video error.
# # #     """
# # #     text = f"{type(exc).__name__}: {exc}".lower()
# # #     if isinstance(exc, MemoryError):
# # #         return True
# # #     return any(sig in text for sig in _OOM_SIGNATURES)


# # # def classify_video_error(exc: BaseException) -> str:
# # #     """
# # #     Maps a caught per-video exception to a short errorType string for the
# # #     failed-endpoint payload. Best-effort classification by exception type
# # #     and message content — defaults to "PROCESSING_ERROR" when nothing
# # #     more specific matches.
# # #     """
# # #     if is_resource_exhaustion_error(exc):
# # #         return "RESOURCE_EXHAUSTION"
# # #     name = type(exc).__name__.lower()
# # #     text = str(exc).lower()
# # #     if "timeout" in name or "timeout" in text:
# # #         return "TIMEOUT"
# # #     if "mediapipe" in text:
# # #         return "MEDIAPIPE_ERROR"
# # #     if "yolo" in text or "ultralytics" in text:
# # #         return "YOLO_ERROR"
# # #     if "cv2" in text or "opencv" in text or name == "error":
# # #         return "OPENCV_ERROR"
# # #     if "cannot open" in text or "corrupt" in text or "decode" in text or "invalid" in text:
# # #         return "DECODE_ERROR"
# # #     return "PROCESSING_ERROR"


# # # def compute_journey_status(
# # #     total_videos: int,
# # #     succeeded_video_ids: "set[int]",
# # #     failed_video_ids: "set[int]",
# # # ) -> str:
# # #     """
# # #     Computes the journey-level terminal status per Phase 1 spec:
# # #         COMPLETED              — every video succeeded
# # #         COMPLETED_WITH_ERRORS  — at least one succeeded AND at least one failed
# # #         FAILED                 — every video failed (or none succeeded)

# # #     TIMED_OUT is NOT computed here — that's set explicitly by the watchdog
# # #     path in consumer.py, which knows it hit a timeout rather than inferring
# # #     it from video counts.
# # #     """
# # #     if total_videos == 0:
# # #         return "FAILED"
# # #     if not failed_video_ids:
# # #         return "COMPLETED"
# # #     if succeeded_video_ids:
# # #         return "COMPLETED_WITH_ERRORS"
# # #     return "FAILED"


# # # # # # # """
# # # # # # # callback_client.py
# # # # # # # ──────────────────
# # # # # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # # # # # All three public helpers are fire-and-forget from the caller's perspective;
# # # # # # # they raise on non-2xx so the consumer can route to the failure path.
# # # # # # # """

# # # # # # # from __future__ import annotations

# # # # # # # import logging
# # # # # # # import os
# # # # # # # from typing import Any, Dict, List

# # # # # # # import requests
# # # # # # # from dotenv import load_dotenv

# # # # # # # # ── Credentials / config ────────────────────────────────────────────────────
# # # # # # # _ENV_PATH = os.path.join(
# # # # # # #     os.path.dirname(os.path.abspath(__file__)),
# # # # # # #     "config", "credentials.env",
# # # # # # # )
# # # # # # # load_dotenv(_ENV_PATH)

# # # # # # # _BASE_URL = os.environ.get("SPRING_BOOT_BASE_URL", "http://localhost:8080")
# # # # # # # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # # # # # log = logging.getLogger("callback_client")


# # # # # # # # ── Internal helper ──────────────────────────────────────────────────────────

# # # # # # # def _post(path: str, payload: Dict[str, Any]) -> None:
# # # # # # #     url = f"{_BASE_URL}{path}"
# # # # # # #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# # # # # # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# # # # # # #     if not resp.ok:
# # # # # # #         raise RuntimeError(
# # # # # # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # # # # # #         )
# # # # # # #     log.debug("[Callback] %s → %d", path, resp.status_code)


# # # # # # # # ── Public API ───────────────────────────────────────────────────────────────

# # # # # # # def send_progress(
# # # # # # #     job_id:     str,
# # # # # # #     journey_id: int,
# # # # # # #     progress:   int,
# # # # # # #     message:    str,
# # # # # # #     status:     str = "PROCESSING",
# # # # # # # ) -> None:
# # # # # # #     """
# # # # # # #     POST /api/internal/analysis/progress

# # # # # # #     Called periodically during analysis to update the frontend progress bar
# # # # # # #     and the SSE stream.
# # # # # # #     """
# # # # # # #     _post(
# # # # # # #         "/api/internal/analysis/progress",
# # # # # # #         {
# # # # # # #             "jobId":      job_id,
# # # # # # #             "journeyId":  journey_id,
# # # # # # #             "status":     status,
# # # # # # #             "progress":   progress,
# # # # # # #             "message":    message,
# # # # # # #         },
# # # # # # #     )


# # # # # # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # # # # # #     """
# # # # # # #     POST /api/internal/analysis/completed

# # # # # # #     Called once — after ALL videos in the journey have been processed and
# # # # # # #     their violation frames uploaded to S3.

# # # # # # #     Expected shape of completion_payload
# # # # # # #     ─────────────────────────────────────
# # # # # # #     {
# # # # # # #         "jobId":          str,
# # # # # # #         "journeyId":      int,
# # # # # # #         "processingTime": int,          # wall-clock ms
# # # # # # #         "videoResults":   List[dict],   # see models.py → VideoResult
# # # # # # #     }
# # # # # # #     """
# # # # # # #     _post("/api/internal/analysis/completed", completion_payload)


# # # # # # # def send_failed(
# # # # # # #     job_id:        str,
# # # # # # #     journey_id:    int,
# # # # # # #     error_message: str,
# # # # # # # ) -> None:
# # # # # # #     """
# # # # # # #     POST /api/internal/analysis/failed

# # # # # # #     Called whenever an unrecoverable exception occurs during job processing.
# # # # # # #     Spring Boot will mark AnalysisJob and Journey as FAILED.
# # # # # # #     """
# # # # # # #     try:
# # # # # # #         _post(
# # # # # # #             "/api/internal/analysis/failed",
# # # # # # #             {
# # # # # # #                 "jobId":        job_id,
# # # # # # #                 "journeyId":    journey_id,
# # # # # # #                 "errorMessage": error_message,
# # # # # # #             },
# # # # # # #         )
# # # # # # #     except Exception as exc:
# # # # # # #         # Failure callback must never itself raise — log and swallow.
# # # # # # #         log.error("[Callback] send_failed itself failed: %s", exc)

# # # # # # """
# # # # # # callback_client.py
# # # # # # ──────────────────
# # # # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # # # # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs
# # # # # # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).

# # # # # # Changes from previous version
# # # # # # ──────────────────────────────
# # # # # # • _BASE_URL now defaults to the live CVVRS domain (overridable via env var
# # # # # #   SPRING_BOOT_BASE_URL for local testing).
# # # # # # • send_progress()  — payload now includes the `currentVideo` field required
# # # # # #   by the API.
# # # # # # • send_completed() — accepts a fully-formed dict (built by CompletionPayload.
# # # # # #   to_dict()); no shape changes needed here since the caller owns the dict.
# # # # # # • send_failed()    — payload drops `journeyId` (not in the API spec for this
# # # # # #   endpoint); only jobId + errorMessage are sent.
# # # # # # """

# # # # # # from __future__ import annotations

# # # # # # import logging
# # # # # # import os
# # # # # # from typing import Any, Dict, Optional

# # # # # # import requests
# # # # # # from dotenv import load_dotenv

# # # # # # # ── Credentials / config ────────────────────────────────────────────────────
# # # # # # _ENV_PATH = os.path.join(
# # # # # #     os.path.dirname(os.path.abspath(__file__)),
# # # # # #     "config", "credentials.env",
# # # # # # )
# # # # # # load_dotenv(_ENV_PATH)

# # # # # # # Default to the live CVVRS server; override with SPRING_BOOT_BASE_URL for local dev.
# # # # # # _BASE_URL = os.environ.get(
# # # # # #     "SPRING_BOOT_BASE_URL",
# # # # # #     "https://cvvrsrailway-api.sconexsoft.com/cvs",
# # # # # # )
# # # # # # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # # # # log = logging.getLogger("callback_client")


# # # # # # # ── Internal helper ──────────────────────────────────────────────────────────

# # # # # # def _post(path: str, payload: Dict[str, Any]) -> None:
# # # # # #     """
# # # # # #     POST to a Spring Boot internal endpoint.
# # # # # #     No Authorization header is sent — /api/internal/* are worker-only endpoints.
# # # # # #     Raises RuntimeError on non-2xx so the consumer can route to failure path.
# # # # # #     """
# # # # # #     url = f"{_BASE_URL}{path}"
# # # # # #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# # # # # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# # # # # #     if not resp.ok:
# # # # # #         raise RuntimeError(
# # # # # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # # # # #         )
# # # # # #     log.debug("[Callback] %s → %d", path, resp.status_code)


# # # # # # # ── Public API ───────────────────────────────────────────────────────────────

# # # # # # def send_progress(
# # # # # #     job_id:        str,
# # # # # #     journey_id:    int,
# # # # # #     progress:      int,
# # # # # #     message:       str,
# # # # # #     status:        str = "PROCESSING",
# # # # # #     current_video: Optional[int] = None,   # NEW — index of video currently being processed
# # # # # # ) -> None:
# # # # # #     """
# # # # # #     POST /api/internal/analysis/progress

# # # # # #     Called periodically during analysis to update the frontend progress bar
# # # # # #     and the SSE stream.

# # # # # #     Parameters
# # # # # #     ──────────
# # # # # #     job_id        : RabbitMQ job ID.
# # # # # #     journey_id    : Journey ID.
# # # # # #     progress      : 0–100 integer.
# # # # # #     message       : Human-readable status message.
# # # # # #     status        : "PROCESSING" | "COMPLETED" | "FAILED"
# # # # # #     current_video : 1-based index of the video currently being processed
# # # # # #                     (omitted from payload when None).
# # # # # #     """
# # # # # #     payload: Dict[str, Any] = {
# # # # # #         "jobId":      job_id,
# # # # # #         "journeyId":  journey_id,
# # # # # #         "status":     status,
# # # # # #         "progress":   progress,
# # # # # #         "message":    message,
# # # # # #     }
# # # # # #     if current_video is not None:
# # # # # #         payload["currentVideo"] = current_video

# # # # # #     _post("/api/internal/analysis/progress", payload)


# # # # # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # # # # #     """
# # # # # #     POST /api/internal/analysis/completed

# # # # # #     Called once after ALL videos in the journey have been processed and their
# # # # # #     violation frames uploaded to S3.

# # # # # #     Expected shape of completion_payload (built by CompletionPayload.to_dict())
# # # # # #     ────────────────────────────────────────────────────────────────────────────
# # # # # #     {
# # # # # #         "jobId":         str,
# # # # # #         "journeyId":     int,
# # # # # #         "batchId":       str,           # "BATCH-<jobId>" if not supplied
# # # # # #         "trainDetailId": int,
# # # # # #         "folderName":    str,
# # # # # #         "processingTime":int,           # wall-clock ms
# # # # # #         "videoResults": [
# # # # # #             {
# # # # # #                 "videoId":         str,       # STRING per API spec
# # # # # #                 "sequenceNo":      int,
# # # # # #                 "durationSeconds": float,
# # # # # #                 "originalS3Key":   str,
# # # # # #                 "violations": [
# # # # # #                     {
# # # # # #                         "violationType":          str,
# # # # # #                         "severity":               str,
# # # # # #                         "confidence":             float,
# # # # # #                         "riskScore":              float,
# # # # # #                         "timestamp":              float,   # journey-global seconds
# # # # # #                         "originalVideoTimestamp": float,   # local-video seconds
# # # # # #                         "framePaths":             [str]
# # # # # #                     }
# # # # # #                 ]
# # # # # #             }
# # # # # #         ]
# # # # # #     }
# # # # # #     """
# # # # # #     _post("/api/internal/analysis/completed", completion_payload)


# # # # # # def send_failed(
# # # # # #     job_id:        str,
# # # # # #     journey_id:    int,          # kept as parameter for caller convenience / logging
# # # # # #     error_message: str,
# # # # # # ) -> None:
# # # # # #     """
# # # # # #     POST /api/internal/analysis/failed

# # # # # #     Called whenever an unrecoverable exception occurs during job processing.
# # # # # #     Spring Boot will mark AnalysisJob and Journey as FAILED.

# # # # # #     Note: The API spec only requires jobId + errorMessage in the request body.
# # # # # #     journeyId is accepted as a parameter here for logging but is NOT included
# # # # # #     in the outbound payload.
# # # # # #     """
# # # # # #     try:
# # # # # #         _post(
# # # # # #             "/api/internal/analysis/failed",
# # # # # #             {
# # # # # #                 "jobId":        job_id,
# # # # # #                 "errorMessage": error_message,
# # # # # #             },
# # # # # #         )
# # # # # #     except Exception as exc:
# # # # # #         # Failure callback must never itself raise — log and swallow.
# # # # # #         log.error(
# # # # # #             "[Callback] send_failed itself failed (job=%s journey=%d): %s",
# # # # # #             job_id, journey_id, exc,
# # # # # #         )



# # # # # """

# # # # # callback_client.py

# # # # # ──────────────────

# # # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.
 
# # # # # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs

# # # # # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).
 
# # # # # Fixes from previous version

# # # # # ─────────────────────────────

# # # # # • Added set_base_url() — called by consumer.py with the callbackBaseUrl from

# # # # #   the RabbitMQ message, so Python posts to the correct Spring Boot server

# # # # #   per environment without needing env var changes.

# # # # # • _BASE_URL still falls back to env var SPRING_BOOT_BASE_URL or the live

# # # # #   server when callbackBaseUrl is not provided in the message.

# # # # # """
 
# # # # # from __future__ import annotations
 
# # # # # import logging

# # # # # import os

# # # # # from typing import Any, Dict, Optional
 
# # # # # import requests

# # # # # from dotenv import load_dotenv
 
# # # # # # ── Credentials / config ────────────────────────────────────────────────────

# # # # # _ENV_PATH = os.path.join(

# # # # #     os.path.dirname(os.path.abspath(__file__)),

# # # # #     "config", "credentials.env",

# # # # # )

# # # # # load_dotenv(_ENV_PATH)
 
# # # # # # Default to the live CVVRS server; override with SPRING_BOOT_BASE_URL for local dev

# # # # # # or call set_base_url() with the callbackBaseUrl from the RabbitMQ message.

# # # # # _BASE_URL = os.environ.get(

# # # # #     "SPRING_BOOT_BASE_URL",

# # # # #     "https://cvvrsrailway-api.sconexsoft.com/cvs",

# # # # # )

# # # # # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))
 
# # # # # log = logging.getLogger("callback_client")
 
 
# # # # # # ── FIX: dynamic base URL setter ─────────────────────────────────────────────
 
# # # # # def set_base_url(url: str) -> None:

# # # # #     """

# # # # #     Override the callback base URL at runtime.
 
# # # # #     Called by consumer.py when the RabbitMQ message contains a callbackBaseUrl.

# # # # #     This allows the same Python worker to callback correctly to both local

# # # # #     and staging Spring Boot servers without changing env vars.
 
# # # # #     Example values:

# # # # #         "http://localhost:8093/api/internal/analysis"         (local)

# # # # #         "https://cvvrsrailway-api.sconexsoft.com/cvs/api/internal/analysis"  (staging)
 
# # # # #     Note: The URL passed here is the FULL path up to /api/internal/analysis.

# # # # #     The individual callbacks append /progress, /completed, /failed.

# # # # #     """

# # # # #     global _BASE_URL

# # # # #     # Strip the /api/internal/analysis suffix if present — we add it in _post()

# # # # #     base = url.rstrip("/")

# # # # #     if base.endswith("/api/internal/analysis"):

# # # # #         base = base[: -len("/api/internal/analysis")]

# # # # #     _BASE_URL = base

# # # # #     log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)
 
 
# # # # # # ── Internal helper ──────────────────────────────────────────────────────────
 
# # # # # def _post(path: str, payload: Dict[str, Any]) -> None:

# # # # #     """

# # # # #     POST to a Spring Boot internal endpoint.

# # # # #     No Authorization header is sent — /api/internal/* are worker-only endpoints.

# # # # #     Raises RuntimeError on non-2xx so the consumer can route to failure path.

# # # # #     """

# # # # #     url = f"{_BASE_URL}{path}"

# # # # #     log.debug("[Callback] POST %s  payload=%s", url, payload)

# # # # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)

# # # # #     if not resp.ok:

# # # # #         raise RuntimeError(

# # # # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"

# # # # #         )

# # # # #     log.debug("[Callback] %s → %d", path, resp.status_code)
 
 
# # # # # # ── Public API ───────────────────────────────────────────────────────────────
 
# # # # # def send_progress(

# # # # #     job_id:        str,

# # # # #     journey_id:    int,

# # # # #     progress:      int,

# # # # #     message:       str,

# # # # #     status:        str = "PROCESSING",

# # # # #     current_video: Optional[int] = None,

# # # # # ) -> None:

# # # # #     """

# # # # #     POST /api/internal/analysis/progress
 
# # # # #     Called periodically during analysis to update the frontend progress bar

# # # # #     and the SSE stream.
 
# # # # #     Parameters

# # # # #     ──────────

# # # # #     job_id        : RabbitMQ job ID.

# # # # #     journey_id    : Journey ID.

# # # # #     progress      : 0–100 integer.

# # # # #     message       : Human-readable status message.

# # # # #     status        : "PROCESSING" | "COMPLETED" | "FAILED"

# # # # #     current_video : 1-based index of the video currently being processed.

# # # # #     """

# # # # #     payload: Dict[str, Any] = {

# # # # #         "jobId":      job_id,

# # # # #         "journeyId":  journey_id,

# # # # #         "status":     status,

# # # # #         "progress":   progress,

# # # # #         "message":    message,

# # # # #     }

# # # # #     if current_video is not None:

# # # # #         payload["currentVideo"] = current_video
 
# # # # #     _post("/api/internal/analysis/progress", payload)
 
 
# # # # # def send_completed(completion_payload: Dict[str, Any]) -> None:

# # # # #     """

# # # # #     POST /api/internal/analysis/completed
 
# # # # #     Called once after ALL videos in the journey have been processed and their

# # # # #     violation frames uploaded to S3.
 
# # # # #     completion_payload is built by CompletionPayload.to_dict() in models.py.

# # # # #     """

# # # # #     _post("/api/internal/analysis/completed", completion_payload)
 
 
# # # # # def send_failed(

# # # # #     job_id:        str,

# # # # #     journey_id:    int,          # kept for caller convenience / logging

# # # # #     error_message: str,

# # # # # ) -> None:

# # # # #     """

# # # # #     POST /api/internal/analysis/failed
 
# # # # #     Called whenever an unrecoverable exception occurs during job processing.

# # # # #     Spring Boot will mark AnalysisJob and Journey as FAILED.
 
# # # # #     Note: The API spec only requires jobId + errorMessage.

# # # # #     journeyId is accepted as a parameter for logging but is NOT included

# # # # #     in the outbound payload.

# # # # #     """

# # # # #     try:

# # # # #         _post(

# # # # #             "/api/internal/analysis/failed",

# # # # #             {

# # # # #                 "jobId":        job_id,

# # # # #                 "errorMessage": error_message,

# # # # #             },

# # # # #         )

# # # # #     except Exception as exc:

# # # # #         # Failure callback must never itself raise — log and swallow.

# # # # #         log.error(

# # # # #             "[Callback] send_failed itself failed (job=%s journey=%d): %s",

# # # # #             job_id, journey_id, exc,

# # # # #         )
 
 
 
# # # # """
# # # # callback_client.py
# # # # ──────────────────
# # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs
# # # # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).

# # # # Changes from previous version
# # # # ──────────────────────────────
# # # # • Added check_job_completed(job_id) — queries GET /api/internal/analysis/status/{jobId}
# # # #   and returns True when the backend reports the job is already COMPLETED.
# # # #   Called by consumer.py as an idempotency guard before starting any processing
# # # #   on a redelivered message.
# # # # """

# # # # from __future__ import annotations
# # # # import logging
# # # # import os
# # # # from typing import Any, Dict, Optional

# # # # import requests
# # # # from dotenv import load_dotenv

# # # # # ── Credentials / config ─────────────────────────────────────────────────────
# # # # _ENV_PATH = os.path.join(
# # # #     os.path.dirname(os.path.abspath(__file__)),
# # # #     "config", "credentials.env",
# # # # )
# # # # load_dotenv(_ENV_PATH)

# # # # _BASE_URL = os.environ.get(
# # # #     "SPRING_BOOT_BASE_URL",
# # # #     "https://cvvrsrailway-api.sconexsoft.com/cvs",
# # # # )
# # # # _TIMEOUT = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # # log = logging.getLogger("callback_client")


# # # # # ── Dynamic base URL setter ───────────────────────────────────────────────────

# # # # def set_base_url(url: str) -> None:
# # # #     """
# # # #     Override the callback base URL at runtime.

# # # #     Called by consumer.py when the RabbitMQ message contains a callbackBaseUrl.
# # # #     Allows the same Python worker to callback correctly to both local and staging
# # # #     Spring Boot servers without changing env vars.

# # # #     The URL passed here is the FULL path up to /api/internal/analysis.
# # # #     Individual callbacks append /progress, /completed, /failed, /status/{id}.
# # # #     """
# # # #     global _BASE_URL
# # # #     base = url.rstrip("/")
# # # #     if base.endswith("/api/internal/analysis"):
# # # #         base = base[: -len("/api/internal/analysis")]
# # # #     _BASE_URL = base
# # # #     log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)


# # # # # ── Internal helpers ──────────────────────────────────────────────────────────

# # # # def _post(path: str, payload: Dict[str, Any]) -> None:
# # # #     """
# # # #     POST to a Spring Boot internal endpoint.
# # # #     No Authorization header — /api/internal/* are worker-only endpoints.
# # # #     Raises RuntimeError on non-2xx so the consumer can route to the failure path.
# # # #     """
# # # #     url = f"{_BASE_URL}{path}"
# # # #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# # # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# # # #     if not resp.ok:
# # # #         raise RuntimeError(
# # # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # # #         )
# # # #     log.debug("[Callback] %s → %d", path, resp.status_code)


# # # # def _get(path: str) -> requests.Response:
# # # #     """
# # # #     GET from a Spring Boot internal endpoint.
# # # #     Raises RuntimeError on non-2xx.
# # # #     """
# # # #     url = f"{_BASE_URL}{path}"
# # # #     log.debug("[Callback] GET %s", url)
# # # #     resp = requests.get(url, timeout=_TIMEOUT)
# # # #     if not resp.ok:
# # # #         raise RuntimeError(
# # # #             f"Callback GET {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # # #         )
# # # #     log.debug("[Callback] %s → %d", path, resp.status_code)
# # # #     return resp


# # # # # ── Public API ────────────────────────────────────────────────────────────────

# # # # def check_job_completed(job_id: str) -> bool:
# # # #     """
# # # #     NEW — Idempotency check.

# # # #     Queries GET /api/internal/analysis/status/{jobId} and returns True when
# # # #     the backend reports the job status as COMPLETED.

# # # #     Called by consumer.py at the very start of _handle_job() so that
# # # #     RabbitMQ redeliveries of already-completed jobs are detected immediately
# # # #     and ACKed without re-running any processing.

# # # #     Backend contract (expected JSON shape):
# # # #         { "status": "COMPLETED" }   → job already done → return True
# # # #         { "status": "PROCESSING" }  → in-flight (rare edge-case) → return False
# # # #         { "status": "PENDING" }     → not yet processed → return False
# # # #         404 Not Found               → job unknown (treat as not completed)

# # # #     If the endpoint does not exist yet on your Spring Boot side, add it as:
# # # #         GET /api/internal/analysis/status/{jobId}
# # # #         → ResponseEntity<Map<String,String>>  { "status": job.getStatus().name() }

# # # #     Raises on network errors so the consumer can decide whether to proceed
# # # #     with processing or skip (consumer.py catches and proceeds on error).
# # # #     """
# # # #     try:
# # # #         resp = _get(f"/api/internal/analysis/status/{job_id}")
# # # #         data = resp.json()
# # # #         status = data.get("status", "").upper()
# # # #         is_done = status == "COMPLETED"
# # # #         log.info(
# # # #             "[Callback] check_job_completed  job=%s  status=%s  is_done=%s",
# # # #             job_id, status, is_done,
# # # #         )
# # # #         return is_done
# # # #     except RuntimeError as exc:
# # # #         # 404 → job not found in the backend → definitely not completed
# # # #         if "404" in str(exc):
# # # #             log.info(
# # # #                 "[Callback] check_job_completed  job=%s  → 404 (not found) → not completed",
# # # #                 job_id,
# # # #             )
# # # #             return False
# # # #         raise


# # # # def send_progress(
# # # #     job_id:        str,
# # # #     journey_id:    int,
# # # #     progress:      int,
# # # #     message:       str,
# # # #     status:        str = "PROCESSING",
# # # #     current_video: Optional[int] = None,
# # # # ) -> None:
# # # #     """
# # # #     POST /api/internal/analysis/progress

# # # #     Called periodically during analysis to update the frontend progress bar
# # # #     and the SSE stream.

# # # #     Parameters
# # # #     ──────────
# # # #     job_id        : RabbitMQ job ID.
# # # #     journey_id    : Journey ID.
# # # #     progress      : 0–100 integer.
# # # #     message       : Human-readable status message.
# # # #     status        : "PROCESSING" | "COMPLETED" | "FAILED"
# # # #     current_video : 1-based index of the video currently being processed.
# # # #     """
# # # #     payload: Dict[str, Any] = {
# # # #         "jobId":     job_id,
# # # #         "journeyId": journey_id,
# # # #         "status":    status,
# # # #         "progress":  progress,
# # # #         "message":   message,
# # # #     }
# # # #     if current_video is not None:
# # # #         payload["currentVideo"] = current_video
# # # #     _post("/api/internal/analysis/progress", payload)


# # # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # # #     """
# # # #     POST /api/internal/analysis/completed

# # # #     Called once after ALL videos in the journey have been processed and their
# # # #     violation frames uploaded to S3.

# # # #     completion_payload is built by CompletionPayload.to_dict() in models.py.
# # # #     """
# # # #     _post("/api/internal/analysis/completed", completion_payload)


# # # # def send_failed(
# # # #     job_id:        str,
# # # #     journey_id:    int,   # kept for caller convenience / logging
# # # #     error_message: str,
# # # # ) -> None:
# # # #     """
# # # #     POST /api/internal/analysis/failed

# # # #     Called whenever an unrecoverable exception occurs during job processing.
# # # #     Spring Boot will mark AnalysisJob and Journey as FAILED.

# # # #     Note: The API spec only requires jobId + errorMessage.
# # # #     journeyId is accepted as a parameter for logging but is NOT included
# # # #     in the outbound payload.
# # # #     """
# # # #     try:
# # # #         _post(
# # # #             "/api/internal/analysis/failed",
# # # #             {
# # # #                 "jobId":        job_id,
# # # #                 "errorMessage": error_message,
# # # #             },
# # # #         )
# # # #     except Exception as exc:
# # # #         # Failure callback must never itself raise — log and swallow.
# # # #         log.error(
# # # #             "[Callback] send_failed itself failed (job=%s journey=%d): %s",
# # # #             job_id, journey_id, exc,
# # # #         )



# # # """
# # # callback_client.py
# # # ──────────────────
# # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs
# # # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).

# # # Changes in this version
# # # ────────────────────────
# # # • check_job_completed() now uses the EXISTING completed-callback endpoint
# # #   as a probe instead of a dedicated /status/{jobId} endpoint that doesn't
# # #   exist yet on Spring Boot.

# # #   Strategy: POST /api/internal/analysis/completed with only { "jobId": id,
# # #   "probe": true }.  Spring Boot should return 409 CONFLICT (or any non-500)
# # #   when the job is already done.  If it returns 500 we treat it as "unknown"
# # #   and fall through to processing (safe default).

# # #   *** TEMPORARY until the Spring Boot /status endpoint is implemented. ***
# # #   Once GET /api/internal/analysis/status/{jobId} is live, revert
# # #   check_job_completed() to use _get() as originally written.
# # # """

# # # from __future__ import annotations

# # # import logging
# # # import os
# # # from typing import Any, Dict, Optional

# # # import requests
# # # from dotenv import load_dotenv

# # # # ── Credentials / config ─────────────────────────────────────────────────────
# # # _ENV_PATH = os.path.join(
# # #     os.path.dirname(os.path.abspath(__file__)),
# # #     "config", "credentials.env",
# # # )
# # # load_dotenv(_ENV_PATH)

# # # _BASE_URL = os.environ.get(
# # #     "SPRING_BOOT_BASE_URL",
# # #     "https://cvvrsrailway-api.sconexsoft.com/cvs",
# # # )
# # # _TIMEOUT = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # log = logging.getLogger("callback_client")


# # # # ── Dynamic base URL setter ───────────────────────────────────────────────────

# # # def set_base_url(url: str) -> None:
# # #     global _BASE_URL
# # #     base = url.rstrip("/")
# # #     if base.endswith("/api/internal/analysis"):
# # #         base = base[: -len("/api/internal/analysis")]
# # #     _BASE_URL = base
# # #     log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)


# # # # ── Internal helpers ──────────────────────────────────────────────────────────

# # # def _post(path: str, payload: Dict[str, Any]) -> requests.Response:
# # #     url = f"{_BASE_URL}{path}"
# # #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# # #     if not resp.ok:
# # #         raise RuntimeError(
# # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # #         )
# # #     log.debug("[Callback] %s → %d", path, resp.status_code)
# # #     return resp


# # # def _get(path: str) -> requests.Response:
# # #     url = f"{_BASE_URL}{path}"
# # #     log.debug("[Callback] GET %s", url)
# # #     resp = requests.get(url, timeout=_TIMEOUT)
# # #     if not resp.ok:
# # #         raise RuntimeError(
# # #             f"Callback GET {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # #         )
# # #     log.debug("[Callback] %s → %d", path, resp.status_code)
# # #     return resp


# # # # ── Public API ────────────────────────────────────────────────────────────────

# # # def check_job_completed(job_id: str) -> bool:
# # #     """
# # #     Idempotency check — returns True if the backend already has this job as COMPLETED.

# # #     TWO-STAGE STRATEGY
# # #     ──────────────────
# # #     Stage 1 (permanent): GET /api/internal/analysis/status/{jobId}
# # #         → { "status": "COMPLETED" }  → True
# # #         → { "status": "PENDING/PROCESSING" } → False
# # #         → 404 → False  (job not known yet)
# # #         → 500 → fall through to Stage 2

# # #     Stage 2 (temporary fallback until Spring Boot implements /status):
# # #         Uses GET /api/internal/analysis/job/{jobId} or any existing
# # #         read endpoint.  If that also 500s, we return False (safe default:
# # #         process the job rather than silently drop it).

# # #     WHAT TO IMPLEMENT ON SPRING BOOT (one small controller method):
# # #     ───────────────────────────────────────────────────────────────
# # #     @GetMapping("/api/internal/analysis/status/{jobId}")
# # #     public ResponseEntity<Map<String,String>> getJobStatus(
# # #             @PathVariable String jobId) {
# # #         return analysisJobRepository.findByJobId(jobId)
# # #             .map(job -> ResponseEntity.ok(Map.of("status", job.getStatus().name())))
# # #             .orElse(ResponseEntity.notFound().<Map<String,String>>build());
# # #     }

# # #     Once that endpoint is deployed, Stage 2 below can be deleted.
# # #     """

# # #     # ── Stage 1: dedicated status endpoint ───────────────────────────────────
# # #     try:
# # #         resp = _get(f"/api/internal/analysis/status/{job_id}")
# # #         status = resp.json().get("status", "").upper()
# # #         is_done = status == "COMPLETED"
# # #         log.info(
# # #             "[Callback] idempotency check (status endpoint)  job=%s  status=%s  done=%s",
# # #             job_id, status, is_done,
# # #         )
# # #         return is_done
# # #     except RuntimeError as exc:
# # #         if "404" in str(exc):
# # #             log.info(
# # #                 "[Callback] idempotency check  job=%s  → 404 → not completed", job_id
# # #             )
# # #             return False
# # #         if "500" in str(exc):
# # #             log.warning(
# # #                 "[Callback] idempotency check  job=%s  → /status endpoint returned 500 "
# # #                 "(not implemented yet?) — trying fallback probe.", job_id
# # #             )
# # #             # fall through to Stage 2
# # #         else:
# # #             # Network error or unexpected status — safe default: process the job
# # #             log.warning(
# # #                 "[Callback] idempotency check  job=%s  → unexpected error (%s) "
# # #                 "— proceeding with processing.", job_id, exc
# # #             )
# # #             return False

# # #     # ── Stage 2: fallback probe using progress endpoint ───────────────────────
# # #     # We send a lightweight progress probe at 0 % with status=CHECK.
# # #     # Spring Boot should:
# # #     #   • Return 200 if the job is still PENDING/PROCESSING (→ process it).
# # #     #   • Return 409 CONFLICT if the job is already COMPLETED (→ skip it).
# # #     #   • Return 404 if the job is unknown (→ process it).
# # #     # If the backend doesn't handle the CHECK status specially it will just
# # #     # update progress to 0 — harmless on an already-completed job.
# # #     #
# # #     # THIS IS A TEMPORARY FALLBACK.  Remove once /status endpoint is live.
# # #     try:
# # #         url = f"{_BASE_URL}/api/internal/analysis/progress"
# # #         resp = requests.post(
# # #             url,
# # #             json={
# # #                 "jobId":    job_id,
# # #                 "status":   "CHECK",   # sentinel value Spring Boot can detect
# # #                 "progress": 0,
# # #                 "message":  "idempotency-probe",
# # #             },
# # #             timeout=_TIMEOUT,
# # #         )
# # #         if resp.status_code == 409:
# # #             log.warning(
# # #                 "[Callback] idempotency check (probe)  job=%s  → 409 CONFLICT "
# # #                 "→ already COMPLETED — will skip.", job_id
# # #             )
# # #             return True
# # #         log.info(
# # #             "[Callback] idempotency check (probe)  job=%s  → HTTP %d → not completed.",
# # #             job_id, resp.status_code,
# # #         )
# # #         return False
# # #     except Exception as exc2:
# # #         log.warning(
# # #             "[Callback] idempotency check (probe) failed  job=%s: %s "
# # #             "— proceeding with processing.", job_id, exc2
# # #         )
# # #         return False


# # # def send_progress(
# # #     job_id:        str,
# # #     journey_id:    int,
# # #     progress:      int,
# # #     message:       str,
# # #     status:        str = "PROCESSING",
# # #     current_video: Optional[int] = None,
# # # ) -> None:
# # #     payload: Dict[str, Any] = {
# # #         "jobId":     job_id,
# # #         "journeyId": journey_id,
# # #         "status":    status,
# # #         "progress":  progress,
# # #         "message":   message,
# # #     }
# # #     if current_video is not None:
# # #         payload["currentVideo"] = current_video
# # #     _post("/api/internal/analysis/progress", payload)


# # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # #     _post("/api/internal/analysis/completed", completion_payload)


# # # def send_failed(
# # #     job_id:        str,
# # #     journey_id:    int,
# # #     error_message: str,
# # # ) -> None:
# # #     try:
# # #         _post(
# # #             "/api/internal/analysis/failed",
# # #             {
# # #                 "jobId":        job_id,
# # #                 "errorMessage": error_message,
# # #             },
# # #         )
# # #     except Exception as exc:
# # #         log.error(
# # #             "[Callback] send_failed itself failed (job=%s journey=%d): %s",
# # #             job_id, journey_id, exc,
# # #         )



# # # # # # """
# # # # # # callback_client.py
# # # # # # ──────────────────
# # # # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # # # # All three public helpers are fire-and-forget from the caller's perspective;
# # # # # # they raise on non-2xx so the consumer can route to the failure path.
# # # # # # """

# # # # # # from __future__ import annotations

# # # # # # import logging
# # # # # # import os
# # # # # # from typing import Any, Dict, List

# # # # # # import requests
# # # # # # from dotenv import load_dotenv

# # # # # # # ── Credentials / config ────────────────────────────────────────────────────
# # # # # # _ENV_PATH = os.path.join(
# # # # # #     os.path.dirname(os.path.abspath(__file__)),
# # # # # #     "config", "credentials.env",
# # # # # # )
# # # # # # load_dotenv(_ENV_PATH)

# # # # # # _BASE_URL = os.environ.get("SPRING_BOOT_BASE_URL", "http://localhost:8080")
# # # # # # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # # # # log = logging.getLogger("callback_client")


# # # # # # # ── Internal helper ──────────────────────────────────────────────────────────

# # # # # # def _post(path: str, payload: Dict[str, Any]) -> None:
# # # # # #     url = f"{_BASE_URL}{path}"
# # # # # #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# # # # # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# # # # # #     if not resp.ok:
# # # # # #         raise RuntimeError(
# # # # # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # # # # #         )
# # # # # #     log.debug("[Callback] %s → %d", path, resp.status_code)


# # # # # # # ── Public API ───────────────────────────────────────────────────────────────

# # # # # # def send_progress(
# # # # # #     job_id:     str,
# # # # # #     journey_id: int,
# # # # # #     progress:   int,
# # # # # #     message:    str,
# # # # # #     status:     str = "PROCESSING",
# # # # # # ) -> None:
# # # # # #     """
# # # # # #     POST /api/internal/analysis/progress

# # # # # #     Called periodically during analysis to update the frontend progress bar
# # # # # #     and the SSE stream.
# # # # # #     """
# # # # # #     _post(
# # # # # #         "/api/internal/analysis/progress",
# # # # # #         {
# # # # # #             "jobId":      job_id,
# # # # # #             "journeyId":  journey_id,
# # # # # #             "status":     status,
# # # # # #             "progress":   progress,
# # # # # #             "message":    message,
# # # # # #         },
# # # # # #     )


# # # # # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # # # # #     """
# # # # # #     POST /api/internal/analysis/completed

# # # # # #     Called once — after ALL videos in the journey have been processed and
# # # # # #     their violation frames uploaded to S3.

# # # # # #     Expected shape of completion_payload
# # # # # #     ─────────────────────────────────────
# # # # # #     {
# # # # # #         "jobId":          str,
# # # # # #         "journeyId":      int,
# # # # # #         "processingTime": int,          # wall-clock ms
# # # # # #         "videoResults":   List[dict],   # see models.py → VideoResult
# # # # # #     }
# # # # # #     """
# # # # # #     _post("/api/internal/analysis/completed", completion_payload)


# # # # # # def send_failed(
# # # # # #     job_id:        str,
# # # # # #     journey_id:    int,
# # # # # #     error_message: str,
# # # # # # ) -> None:
# # # # # #     """
# # # # # #     POST /api/internal/analysis/failed

# # # # # #     Called whenever an unrecoverable exception occurs during job processing.
# # # # # #     Spring Boot will mark AnalysisJob and Journey as FAILED.
# # # # # #     """
# # # # # #     try:
# # # # # #         _post(
# # # # # #             "/api/internal/analysis/failed",
# # # # # #             {
# # # # # #                 "jobId":        job_id,
# # # # # #                 "journeyId":    journey_id,
# # # # # #                 "errorMessage": error_message,
# # # # # #             },
# # # # # #         )
# # # # # #     except Exception as exc:
# # # # # #         # Failure callback must never itself raise — log and swallow.
# # # # # #         log.error("[Callback] send_failed itself failed: %s", exc)

# # # # # """
# # # # # callback_client.py
# # # # # ──────────────────
# # # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # # # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs
# # # # # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).

# # # # # Changes from previous version
# # # # # ──────────────────────────────
# # # # # • _BASE_URL now defaults to the live CVVRS domain (overridable via env var
# # # # #   SPRING_BOOT_BASE_URL for local testing).
# # # # # • send_progress()  — payload now includes the `currentVideo` field required
# # # # #   by the API.
# # # # # • send_completed() — accepts a fully-formed dict (built by CompletionPayload.
# # # # #   to_dict()); no shape changes needed here since the caller owns the dict.
# # # # # • send_failed()    — payload drops `journeyId` (not in the API spec for this
# # # # #   endpoint); only jobId + errorMessage are sent.
# # # # # """

# # # # # from __future__ import annotations

# # # # # import logging
# # # # # import os
# # # # # from typing import Any, Dict, Optional

# # # # # import requests
# # # # # from dotenv import load_dotenv

# # # # # # ── Credentials / config ────────────────────────────────────────────────────
# # # # # _ENV_PATH = os.path.join(
# # # # #     os.path.dirname(os.path.abspath(__file__)),
# # # # #     "config", "credentials.env",
# # # # # )
# # # # # load_dotenv(_ENV_PATH)

# # # # # # Default to the live CVVRS server; override with SPRING_BOOT_BASE_URL for local dev.
# # # # # _BASE_URL = os.environ.get(
# # # # #     "SPRING_BOOT_BASE_URL",
# # # # #     "https://cvvrsrailway-api.sconexsoft.com/cvs",
# # # # # )
# # # # # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # # # log = logging.getLogger("callback_client")


# # # # # # ── Internal helper ──────────────────────────────────────────────────────────

# # # # # def _post(path: str, payload: Dict[str, Any]) -> None:
# # # # #     """
# # # # #     POST to a Spring Boot internal endpoint.
# # # # #     No Authorization header is sent — /api/internal/* are worker-only endpoints.
# # # # #     Raises RuntimeError on non-2xx so the consumer can route to failure path.
# # # # #     """
# # # # #     url = f"{_BASE_URL}{path}"
# # # # #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# # # # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# # # # #     if not resp.ok:
# # # # #         raise RuntimeError(
# # # # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # # # #         )
# # # # #     log.debug("[Callback] %s → %d", path, resp.status_code)


# # # # # # ── Public API ───────────────────────────────────────────────────────────────

# # # # # def send_progress(
# # # # #     job_id:        str,
# # # # #     journey_id:    int,
# # # # #     progress:      int,
# # # # #     message:       str,
# # # # #     status:        str = "PROCESSING",
# # # # #     current_video: Optional[int] = None,   # NEW — index of video currently being processed
# # # # # ) -> None:
# # # # #     """
# # # # #     POST /api/internal/analysis/progress

# # # # #     Called periodically during analysis to update the frontend progress bar
# # # # #     and the SSE stream.

# # # # #     Parameters
# # # # #     ──────────
# # # # #     job_id        : RabbitMQ job ID.
# # # # #     journey_id    : Journey ID.
# # # # #     progress      : 0–100 integer.
# # # # #     message       : Human-readable status message.
# # # # #     status        : "PROCESSING" | "COMPLETED" | "FAILED"
# # # # #     current_video : 1-based index of the video currently being processed
# # # # #                     (omitted from payload when None).
# # # # #     """
# # # # #     payload: Dict[str, Any] = {
# # # # #         "jobId":      job_id,
# # # # #         "journeyId":  journey_id,
# # # # #         "status":     status,
# # # # #         "progress":   progress,
# # # # #         "message":    message,
# # # # #     }
# # # # #     if current_video is not None:
# # # # #         payload["currentVideo"] = current_video

# # # # #     _post("/api/internal/analysis/progress", payload)


# # # # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # # # #     """
# # # # #     POST /api/internal/analysis/completed

# # # # #     Called once after ALL videos in the journey have been processed and their
# # # # #     violation frames uploaded to S3.

# # # # #     Expected shape of completion_payload (built by CompletionPayload.to_dict())
# # # # #     ────────────────────────────────────────────────────────────────────────────
# # # # #     {
# # # # #         "jobId":         str,
# # # # #         "journeyId":     int,
# # # # #         "batchId":       str,           # "BATCH-<jobId>" if not supplied
# # # # #         "trainDetailId": int,
# # # # #         "folderName":    str,
# # # # #         "processingTime":int,           # wall-clock ms
# # # # #         "videoResults": [
# # # # #             {
# # # # #                 "videoId":         str,       # STRING per API spec
# # # # #                 "sequenceNo":      int,
# # # # #                 "durationSeconds": float,
# # # # #                 "originalS3Key":   str,
# # # # #                 "violations": [
# # # # #                     {
# # # # #                         "violationType":          str,
# # # # #                         "severity":               str,
# # # # #                         "confidence":             float,
# # # # #                         "riskScore":              float,
# # # # #                         "timestamp":              float,   # journey-global seconds
# # # # #                         "originalVideoTimestamp": float,   # local-video seconds
# # # # #                         "framePaths":             [str]
# # # # #                     }
# # # # #                 ]
# # # # #             }
# # # # #         ]
# # # # #     }
# # # # #     """
# # # # #     _post("/api/internal/analysis/completed", completion_payload)


# # # # # def send_failed(
# # # # #     job_id:        str,
# # # # #     journey_id:    int,          # kept as parameter for caller convenience / logging
# # # # #     error_message: str,
# # # # # ) -> None:
# # # # #     """
# # # # #     POST /api/internal/analysis/failed

# # # # #     Called whenever an unrecoverable exception occurs during job processing.
# # # # #     Spring Boot will mark AnalysisJob and Journey as FAILED.

# # # # #     Note: The API spec only requires jobId + errorMessage in the request body.
# # # # #     journeyId is accepted as a parameter here for logging but is NOT included
# # # # #     in the outbound payload.
# # # # #     """
# # # # #     try:
# # # # #         _post(
# # # # #             "/api/internal/analysis/failed",
# # # # #             {
# # # # #                 "jobId":        job_id,
# # # # #                 "errorMessage": error_message,
# # # # #             },
# # # # #         )
# # # # #     except Exception as exc:
# # # # #         # Failure callback must never itself raise — log and swallow.
# # # # #         log.error(
# # # # #             "[Callback] send_failed itself failed (job=%s journey=%d): %s",
# # # # #             job_id, journey_id, exc,
# # # # #         )



# # # # """

# # # # callback_client.py

# # # # ──────────────────

# # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.
 
# # # # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs

# # # # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).
 
# # # # Fixes from previous version

# # # # ─────────────────────────────

# # # # • Added set_base_url() — called by consumer.py with the callbackBaseUrl from

# # # #   the RabbitMQ message, so Python posts to the correct Spring Boot server

# # # #   per environment without needing env var changes.

# # # # • _BASE_URL still falls back to env var SPRING_BOOT_BASE_URL or the live

# # # #   server when callbackBaseUrl is not provided in the message.

# # # # """
 
# # # # from __future__ import annotations
 
# # # # import logging

# # # # import os

# # # # from typing import Any, Dict, Optional
 
# # # # import requests

# # # # from dotenv import load_dotenv
 
# # # # # ── Credentials / config ────────────────────────────────────────────────────

# # # # _ENV_PATH = os.path.join(

# # # #     os.path.dirname(os.path.abspath(__file__)),

# # # #     "config", "credentials.env",

# # # # )

# # # # load_dotenv(_ENV_PATH)
 
# # # # # Default to the live CVVRS server; override with SPRING_BOOT_BASE_URL for local dev

# # # # # or call set_base_url() with the callbackBaseUrl from the RabbitMQ message.

# # # # _BASE_URL = os.environ.get(

# # # #     "SPRING_BOOT_BASE_URL",

# # # #     "https://cvvrsrailway-api.sconexsoft.com/cvs",

# # # # )

# # # # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))
 
# # # # log = logging.getLogger("callback_client")
 
 
# # # # # ── FIX: dynamic base URL setter ─────────────────────────────────────────────
 
# # # # def set_base_url(url: str) -> None:

# # # #     """

# # # #     Override the callback base URL at runtime.
 
# # # #     Called by consumer.py when the RabbitMQ message contains a callbackBaseUrl.

# # # #     This allows the same Python worker to callback correctly to both local

# # # #     and staging Spring Boot servers without changing env vars.
 
# # # #     Example values:

# # # #         "http://localhost:8093/api/internal/analysis"         (local)

# # # #         "https://cvvrsrailway-api.sconexsoft.com/cvs/api/internal/analysis"  (staging)
 
# # # #     Note: The URL passed here is the FULL path up to /api/internal/analysis.

# # # #     The individual callbacks append /progress, /completed, /failed.

# # # #     """

# # # #     global _BASE_URL

# # # #     # Strip the /api/internal/analysis suffix if present — we add it in _post()

# # # #     base = url.rstrip("/")

# # # #     if base.endswith("/api/internal/analysis"):

# # # #         base = base[: -len("/api/internal/analysis")]

# # # #     _BASE_URL = base

# # # #     log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)
 
 
# # # # # ── Internal helper ──────────────────────────────────────────────────────────
 
# # # # def _post(path: str, payload: Dict[str, Any]) -> None:

# # # #     """

# # # #     POST to a Spring Boot internal endpoint.

# # # #     No Authorization header is sent — /api/internal/* are worker-only endpoints.

# # # #     Raises RuntimeError on non-2xx so the consumer can route to failure path.

# # # #     """

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

# # # #     job_id:        str,

# # # #     journey_id:    int,

# # # #     progress:      int,

# # # #     message:       str,

# # # #     status:        str = "PROCESSING",

# # # #     current_video: Optional[int] = None,

# # # # ) -> None:

# # # #     """

# # # #     POST /api/internal/analysis/progress
 
# # # #     Called periodically during analysis to update the frontend progress bar

# # # #     and the SSE stream.
 
# # # #     Parameters

# # # #     ──────────

# # # #     job_id        : RabbitMQ job ID.

# # # #     journey_id    : Journey ID.

# # # #     progress      : 0–100 integer.

# # # #     message       : Human-readable status message.

# # # #     status        : "PROCESSING" | "COMPLETED" | "FAILED"

# # # #     current_video : 1-based index of the video currently being processed.

# # # #     """

# # # #     payload: Dict[str, Any] = {

# # # #         "jobId":      job_id,

# # # #         "journeyId":  journey_id,

# # # #         "status":     status,

# # # #         "progress":   progress,

# # # #         "message":    message,

# # # #     }

# # # #     if current_video is not None:

# # # #         payload["currentVideo"] = current_video
 
# # # #     _post("/api/internal/analysis/progress", payload)
 
 
# # # # def send_completed(completion_payload: Dict[str, Any]) -> None:

# # # #     """

# # # #     POST /api/internal/analysis/completed
 
# # # #     Called once after ALL videos in the journey have been processed and their

# # # #     violation frames uploaded to S3.
 
# # # #     completion_payload is built by CompletionPayload.to_dict() in models.py.

# # # #     """

# # # #     _post("/api/internal/analysis/completed", completion_payload)
 
 
# # # # def send_failed(

# # # #     job_id:        str,

# # # #     journey_id:    int,          # kept for caller convenience / logging

# # # #     error_message: str,

# # # # ) -> None:

# # # #     """

# # # #     POST /api/internal/analysis/failed
 
# # # #     Called whenever an unrecoverable exception occurs during job processing.

# # # #     Spring Boot will mark AnalysisJob and Journey as FAILED.
 
# # # #     Note: The API spec only requires jobId + errorMessage.

# # # #     journeyId is accepted as a parameter for logging but is NOT included

# # # #     in the outbound payload.

# # # #     """

# # # #     try:

# # # #         _post(

# # # #             "/api/internal/analysis/failed",

# # # #             {

# # # #                 "jobId":        job_id,

# # # #                 "errorMessage": error_message,

# # # #             },

# # # #         )

# # # #     except Exception as exc:

# # # #         # Failure callback must never itself raise — log and swallow.

# # # #         log.error(

# # # #             "[Callback] send_failed itself failed (job=%s journey=%d): %s",

# # # #             job_id, journey_id, exc,

# # # #         )
 
 
 
# # # """
# # # callback_client.py
# # # ──────────────────
# # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs
# # # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).

# # # Changes from previous version
# # # ──────────────────────────────
# # # • Added check_job_completed(job_id) — queries GET /api/internal/analysis/status/{jobId}
# # #   and returns True when the backend reports the job is already COMPLETED.
# # #   Called by consumer.py as an idempotency guard before starting any processing
# # #   on a redelivered message.
# # # """

# # # from __future__ import annotations
# # # import logging
# # # import os
# # # from typing import Any, Dict, Optional

# # # import requests
# # # from dotenv import load_dotenv

# # # # ── Credentials / config ─────────────────────────────────────────────────────
# # # _ENV_PATH = os.path.join(
# # #     os.path.dirname(os.path.abspath(__file__)),
# # #     "config", "credentials.env",
# # # )
# # # load_dotenv(_ENV_PATH)

# # # _BASE_URL = os.environ.get(
# # #     "SPRING_BOOT_BASE_URL",
# # #     "https://cvvrsrailway-api.sconexsoft.com/cvs",
# # # )
# # # _TIMEOUT = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # log = logging.getLogger("callback_client")


# # # # ── Dynamic base URL setter ───────────────────────────────────────────────────

# # # def set_base_url(url: str) -> None:
# # #     """
# # #     Override the callback base URL at runtime.

# # #     Called by consumer.py when the RabbitMQ message contains a callbackBaseUrl.
# # #     Allows the same Python worker to callback correctly to both local and staging
# # #     Spring Boot servers without changing env vars.

# # #     The URL passed here is the FULL path up to /api/internal/analysis.
# # #     Individual callbacks append /progress, /completed, /failed, /status/{id}.
# # #     """
# # #     global _BASE_URL
# # #     base = url.rstrip("/")
# # #     if base.endswith("/api/internal/analysis"):
# # #         base = base[: -len("/api/internal/analysis")]
# # #     _BASE_URL = base
# # #     log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)


# # # # ── Internal helpers ──────────────────────────────────────────────────────────

# # # def _post(path: str, payload: Dict[str, Any]) -> None:
# # #     """
# # #     POST to a Spring Boot internal endpoint.
# # #     No Authorization header — /api/internal/* are worker-only endpoints.
# # #     Raises RuntimeError on non-2xx so the consumer can route to the failure path.
# # #     """
# # #     url = f"{_BASE_URL}{path}"
# # #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# # #     if not resp.ok:
# # #         raise RuntimeError(
# # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # #         )
# # #     log.debug("[Callback] %s → %d", path, resp.status_code)


# # # def _get(path: str) -> requests.Response:
# # #     """
# # #     GET from a Spring Boot internal endpoint.
# # #     Raises RuntimeError on non-2xx.
# # #     """
# # #     url = f"{_BASE_URL}{path}"
# # #     log.debug("[Callback] GET %s", url)
# # #     resp = requests.get(url, timeout=_TIMEOUT)
# # #     if not resp.ok:
# # #         raise RuntimeError(
# # #             f"Callback GET {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # #         )
# # #     log.debug("[Callback] %s → %d", path, resp.status_code)
# # #     return resp


# # # # ── Public API ────────────────────────────────────────────────────────────────

# # # def check_job_completed(job_id: str) -> bool:
# # #     """
# # #     NEW — Idempotency check.

# # #     Queries GET /api/internal/analysis/status/{jobId} and returns True when
# # #     the backend reports the job status as COMPLETED.

# # #     Called by consumer.py at the very start of _handle_job() so that
# # #     RabbitMQ redeliveries of already-completed jobs are detected immediately
# # #     and ACKed without re-running any processing.

# # #     Backend contract (expected JSON shape):
# # #         { "status": "COMPLETED" }   → job already done → return True
# # #         { "status": "PROCESSING" }  → in-flight (rare edge-case) → return False
# # #         { "status": "PENDING" }     → not yet processed → return False
# # #         404 Not Found               → job unknown (treat as not completed)

# # #     If the endpoint does not exist yet on your Spring Boot side, add it as:
# # #         GET /api/internal/analysis/status/{jobId}
# # #         → ResponseEntity<Map<String,String>>  { "status": job.getStatus().name() }

# # #     Raises on network errors so the consumer can decide whether to proceed
# # #     with processing or skip (consumer.py catches and proceeds on error).
# # #     """
# # #     try:
# # #         resp = _get(f"/api/internal/analysis/status/{job_id}")
# # #         data = resp.json()
# # #         status = data.get("status", "").upper()
# # #         is_done = status == "COMPLETED"
# # #         log.info(
# # #             "[Callback] check_job_completed  job=%s  status=%s  is_done=%s",
# # #             job_id, status, is_done,
# # #         )
# # #         return is_done
# # #     except RuntimeError as exc:
# # #         # 404 → job not found in the backend → definitely not completed
# # #         if "404" in str(exc):
# # #             log.info(
# # #                 "[Callback] check_job_completed  job=%s  → 404 (not found) → not completed",
# # #                 job_id,
# # #             )
# # #             return False
# # #         raise


# # # def send_progress(
# # #     job_id:        str,
# # #     journey_id:    int,
# # #     progress:      int,
# # #     message:       str,
# # #     status:        str = "PROCESSING",
# # #     current_video: Optional[int] = None,
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
# # #     current_video : 1-based index of the video currently being processed.
# # #     """
# # #     payload: Dict[str, Any] = {
# # #         "jobId":     job_id,
# # #         "journeyId": journey_id,
# # #         "status":    status,
# # #         "progress":  progress,
# # #         "message":   message,
# # #     }
# # #     if current_video is not None:
# # #         payload["currentVideo"] = current_video
# # #     _post("/api/internal/analysis/progress", payload)


# # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # #     """
# # #     POST /api/internal/analysis/completed

# # #     Called once after ALL videos in the journey have been processed and their
# # #     violation frames uploaded to S3.

# # #     completion_payload is built by CompletionPayload.to_dict() in models.py.
# # #     """
# # #     _post("/api/internal/analysis/completed", completion_payload)


# # # def send_failed(
# # #     job_id:        str,
# # #     journey_id:    int,   # kept for caller convenience / logging
# # #     error_message: str,
# # # ) -> None:
# # #     """
# # #     POST /api/internal/analysis/failed

# # #     Called whenever an unrecoverable exception occurs during job processing.
# # #     Spring Boot will mark AnalysisJob and Journey as FAILED.

# # #     Note: The API spec only requires jobId + errorMessage.
# # #     journeyId is accepted as a parameter for logging but is NOT included
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

# # Changes in this version
# # ────────────────────────
# # • check_job_completed() now uses the EXISTING completed-callback endpoint
# #   as a probe instead of a dedicated /status/{jobId} endpoint that doesn't
# #   exist yet on Spring Boot.

# #   Strategy: POST /api/internal/analysis/completed with only { "jobId": id,
# #   "probe": true }.  Spring Boot should return 409 CONFLICT (or any non-500)
# #   when the job is already done.  If it returns 500 we treat it as "unknown"
# #   and fall through to processing (safe default).

# #   *** TEMPORARY until the Spring Boot /status endpoint is implemented. ***
# #   Once GET /api/internal/analysis/status/{jobId} is live, revert
# #   check_job_completed() to use _get() as originally written.
# # """

# # from __future__ import annotations

# # import logging
# # import os
# # from typing import Any, Dict, Optional

# # import requests
# # from dotenv import load_dotenv

# # # ── Credentials / config ─────────────────────────────────────────────────────
# # _ENV_PATH = os.path.join(
# #     os.path.dirname(os.path.abspath(__file__)),
# #     "config", "credentials.env",
# # )
# # load_dotenv(_ENV_PATH)

# # _BASE_URL = os.environ.get(
# #     "SPRING_BOOT_BASE_URL",
# #     "https://cvvrsrailway-api.sconexsoft.com/cvs",
# # )
# # _TIMEOUT = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # log = logging.getLogger("callback_client")


# # # ── Dynamic base URL setter ───────────────────────────────────────────────────

# # def set_base_url(url: str) -> None:
# #     global _BASE_URL
# #     base = url.rstrip("/")
# #     if base.endswith("/api/internal/analysis"):
# #         base = base[: -len("/api/internal/analysis")]
# #     _BASE_URL = base
# #     log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)


# # # ── Internal helpers ──────────────────────────────────────────────────────────

# # def _post(path: str, payload: Dict[str, Any]) -> requests.Response:
# #     url = f"{_BASE_URL}{path}"
# #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# #     if not resp.ok:
# #         raise RuntimeError(
# #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# #         )
# #     log.debug("[Callback] %s → %d", path, resp.status_code)
# #     return resp


# # def _get(path: str) -> requests.Response:
# #     url = f"{_BASE_URL}{path}"
# #     log.debug("[Callback] GET %s", url)
# #     resp = requests.get(url, timeout=_TIMEOUT)
# #     if not resp.ok:
# #         raise RuntimeError(
# #             f"Callback GET {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# #         )
# #     log.debug("[Callback] %s → %d", path, resp.status_code)
# #     return resp


# # # ── Public API ────────────────────────────────────────────────────────────────

# # # ── Local idempotency cache ───────────────────────────────────────────────────
# # #
# # # When the worker finishes a job it records the job_id here so that if the
# # # same message is redelivered (RabbitMQ consumer-timeout cancels the consumer
# # # mid-journey, broker requeues, worker reconnects and picks it up again) the
# # # idempotency check catches it locally WITHOUT relying on the backend /status
# # # endpoint — which currently returns 500 for both probe strategies.
# # #
# # # The cache is process-lifetime only (lost on restart).  That is intentional:
# # # after a genuine worker crash we WANT to reprocess any unACKed job.  The
# # # repeat-job problem described in the bug report is caused by RabbitMQ's
# # # consumer_timeout firing while the worker is still alive and processing,
# # # so the in-memory cache is always present when the redelivery arrives.
# # #
# # # mark_job_completed() is called by consumer.py immediately before sending
# # # the ACK, so the job is in the cache for any subsequent redelivery.
# # _completed_jobs: set = set()


# # def mark_job_completed(job_id: str) -> None:
# #     """Record that this worker instance has successfully completed job_id."""
# #     _completed_jobs.add(job_id)
# #     log.info("[CallbackClient]  Marked job %s as locally completed.", job_id)


# # def check_job_completed(job_id: str) -> bool:
# #     """
# #     Idempotency check — returns True if the backend already has this job as COMPLETED.

# #     STAGE 0 (local cache — fastest, most reliable):
# #     ─────────────────────────────────────────────────
# #     If THIS worker process already completed and ACKed the job during the
# #     current run, it is in _completed_jobs and we return True immediately,
# #     bypassing both backend probes entirely.  This is the correct fix for the
# #     RabbitMQ consumer-timeout redelivery problem: the broker requeues the
# #     message and the worker's reconnect loop picks it up again, but the
# #     in-memory cache catches it before any work is repeated.

# #     TWO-STAGE STRATEGY (backend probes — for cross-process idempotency)
# #     ──────────────────
# #     Stage 1 (permanent): GET /api/internal/analysis/status/{jobId}
# #         → { "status": "COMPLETED" }  → True
# #         → { "status": "PENDING/PROCESSING" } → False
# #         → 404 → False  (job not known yet)
# #         → 500 → fall through to Stage 2

# #     Stage 2 (temporary fallback until Spring Boot implements /status):
# #         Uses GET /api/internal/analysis/job/{jobId} or any existing
# #         read endpoint.  If that also 500s, we return False (safe default:
# #         process the job rather than silently drop it).

# #     WHAT TO IMPLEMENT ON SPRING BOOT (one small controller method):
# #     ───────────────────────────────────────────────────────────────
# #     @GetMapping("/api/internal/analysis/status/{jobId}")
# #     public ResponseEntity<Map<String,String>> getJobStatus(
# #             @PathVariable String jobId) {
# #         return analysisJobRepository.findByJobId(jobId)
# #             .map(job -> ResponseEntity.ok(Map.of("status", job.getStatus().name())))
# #             .orElse(ResponseEntity.notFound().<Map<String,String>>build());
# #     }

# #     Once that endpoint is deployed, Stage 2 below can be deleted.
# #     """

# #     # ── Stage 0: local in-process cache (catches consumer-timeout redeliveries)
# #     if job_id in _completed_jobs:
# #         log.warning(
# #             "[Callback] idempotency check  job=%s  → found in local completed cache "
# #             "— this is a redelivery of an already-ACKed job (likely caused by "
# #             "RabbitMQ consumer_timeout). Skipping re-processing.", job_id,
# #         )
# #         return True

# #     # ── Stage 1: dedicated status endpoint ───────────────────────────────────
# #     try:
# #         resp = _get(f"/api/internal/analysis/status/{job_id}")
# #         status = resp.json().get("status", "").upper()
# #         is_done = status == "COMPLETED"
# #         log.info(
# #             "[Callback] idempotency check (status endpoint)  job=%s  status=%s  done=%s",
# #             job_id, status, is_done,
# #         )
# #         return is_done
# #     except RuntimeError as exc:
# #         if "404" in str(exc):
# #             log.info(
# #                 "[Callback] idempotency check  job=%s  → 404 → not completed", job_id
# #             )
# #             return False
# #         if "500" in str(exc):
# #             log.warning(
# #                 "[Callback] idempotency check  job=%s  → /status endpoint returned 500 "
# #                 "(not implemented yet?) — trying fallback probe.", job_id
# #             )
# #             # fall through to Stage 2
# #         else:
# #             # Network error or unexpected status — safe default: process the job
# #             log.warning(
# #                 "[Callback] idempotency check  job=%s  → unexpected error (%s) "
# #                 "— proceeding with processing.", job_id, exc
# #             )
# #             return False

# #     # ── Stage 2: fallback probe using progress endpoint ───────────────────────
# #     # We send a lightweight progress probe at 0 % with status=CHECK.
# #     # Spring Boot should:
# #     #   • Return 200 if the job is still PENDING/PROCESSING (→ process it).
# #     #   • Return 409 CONFLICT if the job is already COMPLETED (→ skip it).
# #     #   • Return 404 if the job is unknown (→ process it).
# #     # If the backend doesn't handle the CHECK status specially it will just
# #     # update progress to 0 — harmless on an already-completed job.
# #     #
# #     # THIS IS A TEMPORARY FALLBACK.  Remove once /status endpoint is live.
# #     try:
# #         url = f"{_BASE_URL}/api/internal/analysis/progress"
# #         resp = requests.post(
# #             url,
# #             json={
# #                 "jobId":    job_id,
# #                 "status":   "CHECK",   # sentinel value Spring Boot can detect
# #                 "progress": 0,
# #                 "message":  "idempotency-probe",
# #             },
# #             timeout=_TIMEOUT,
# #         )
# #         if resp.status_code == 409:
# #             log.warning(
# #                 "[Callback] idempotency check (probe)  job=%s  → 409 CONFLICT "
# #                 "→ already COMPLETED — will skip.", job_id
# #             )
# #             return True
# #         log.info(
# #             "[Callback] idempotency check (probe)  job=%s  → HTTP %d → not completed.",
# #             job_id, resp.status_code,
# #         )
# #         return False
# #     except Exception as exc2:
# #         log.warning(
# #             "[Callback] idempotency check (probe) failed  job=%s: %s "
# #             "— proceeding with processing.", job_id, exc2
# #         )
# #         return False


# # def send_progress(
# #     job_id:        str,
# #     journey_id:    int,
# #     progress:      int,
# #     message:       str,
# #     status:        str = "PROCESSING",
# #     current_video: Optional[int] = None,
# # ) -> None:
# #     payload: Dict[str, Any] = {
# #         "jobId":     job_id,
# #         "journeyId": journey_id,
# #         "status":    status,
# #         "progress":  progress,
# #         "message":   message,
# #     }
# #     if current_video is not None:
# #         payload["currentVideo"] = current_video
# #     _post("/api/internal/analysis/progress", payload)


# # def send_completed(completion_payload: Dict[str, Any]) -> None:
# #     _post("/api/internal/analysis/completed", completion_payload)


# # def send_failed(
# #     job_id:        str,
# #     journey_id:    int,
# #     error_message: str,
# #     video_id:      Optional[int] = None,
# #     error_type:    Optional[str] = None,
# #     stack_trace:   Optional[str] = None,
# #     reason:        Optional[str] = None,
# # ) -> None:
# #     """
# #     Calls POST /api/internal/analysis/failed.

# #     Backward compatible: existing job-level callers (no video_id) keep
# #     working exactly as before — the payload shape for that case is
# #     unchanged ({"jobId", "errorMessage"}).

# #     When video_id IS supplied, this becomes a PER-VIDEO failure report
# #     (Phase 1 requirement: "no failed video should wait until journey
# #     completion to be reported"). The payload then also carries:
# #         videoId     — STRING, matches VideoResult.to_dict()'s convention
# #         journeyId
# #         errorType   — short classifier, e.g. "OOM", "DECODE_ERROR",
# #                       "MEDIAPIPE_ERROR", "YOLO_ERROR", "TIMEOUT",
# #                       "RESOURCE_EXHAUSTION", "NOT_PROCESSED"
# #         stackTrace  — full traceback text, if available
# #         reason      — human-readable reason string, e.g.
# #                       "Not Processed - Worker Resource Exhaustion"
# #                       for videos skipped after an OOM on an earlier video.
# #     """
# #     payload: Dict[str, Any] = {
# #         "jobId":        job_id,
# #         "errorMessage": error_message,
# #     }
# #     if video_id is not None:
# #         payload["videoId"]    = str(video_id)   # STRING per API spec convention
# #         payload["journeyId"]  = journey_id
# #     if error_type is not None:
# #         payload["errorType"] = error_type
# #     if stack_trace is not None:
# #         payload["stackTrace"] = stack_trace
# #     if reason is not None:
# #         payload["reason"] = reason

# #     try:
# #         _post("/api/internal/analysis/failed", payload)
# #     except Exception as exc:
# #         log.error(
# #             "[Callback] send_failed itself failed (job=%s journey=%d video=%s): %s",
# #             job_id, journey_id, video_id, exc,
# #         )


# # def send_video_failed(
# #     job_id:        str,
# #     journey_id:    int,
# #     video_id:      int,
# #     error_type:    str,
# #     error_message: str,
# #     stack_trace:   str = "",
# #     reason:        Optional[str] = None,
# # ) -> None:
# #     """
# #     Convenience wrapper for the per-video failure report required by
# #     Phase 1: "Immediately call the failed endpoint for the affected video"
# #     — called once per video, the moment that video's outcome is known,
# #     not batched until the journey ends.

# #     For resource-exhaustion cascades, pass error_type="RESOURCE_EXHAUSTION"
# #     (the video that actually hit OOM) or error_type="NOT_PROCESSED" with
# #     reason="Not Processed - Worker Resource Exhaustion" (videos after it
# #     that were skipped as a result).
# #     """
# #     log.error(
# #         "[Callback] video FAILED  job=%s  journey=%d  video=%d  type=%s  reason=%s: %s",
# #         job_id, journey_id, video_id, error_type, reason, error_message,
# #     )
# #     send_failed(
# #         job_id        = job_id,
# #         journey_id    = journey_id,
# #         error_message = error_message,
# #         video_id      = video_id,
# #         error_type    = error_type,
# #         stack_trace   = stack_trace,
# #         reason        = reason,
# #     )


# # # ── OOM / resource-exhaustion signature detection ────────────────────────────
# # #
# # # Used by analyzer.py to decide whether a per-video exception means "this
# # # one video is bad" (continue with the rest) vs. "the worker itself can no
# # # longer safely process more video" (stop the journey, mark every
# # # remaining video FAILED with reason "Not Processed - Worker Resource
# # # Exhaustion"). Matches the signatures called out explicitly in the
# # # Phase 1 spec: OpenCV OutOfMemoryError, "Failed to allocate", cv2.pyd
# # # memory-related failures, generic MemoryError, std::bad_alloc.
# # _OOM_SIGNATURES = (
# #     "outofmemoryerror",
# #     "out of memory",
# #     "failed to allocate",
# #     "cannot allocate memory",
# #     "bad_alloc",
# #     "memoryerror",
# #     "cv2.pyd",
# #     "access violation",
# #     "resource exhaust",
# # )


# # def is_resource_exhaustion_error(exc: BaseException) -> bool:
# #     """
# #     Returns True if `exc` looks like an OOM / native resource-exhaustion
# #     failure rather than an ordinary recoverable per-video error.
# #     """
# #     text = f"{type(exc).__name__}: {exc}".lower()
# #     if isinstance(exc, MemoryError):
# #         return True
# #     return any(sig in text for sig in _OOM_SIGNATURES)


# # def classify_video_error(exc: BaseException) -> str:
# #     """
# #     Maps a caught per-video exception to a short errorType string for the
# #     failed-endpoint payload. Best-effort classification by exception type
# #     and message content — defaults to "PROCESSING_ERROR" when nothing
# #     more specific matches.
# #     """
# #     if is_resource_exhaustion_error(exc):
# #         return "RESOURCE_EXHAUSTION"
# #     name = type(exc).__name__.lower()
# #     text = str(exc).lower()
# #     if "timeout" in name or "timeout" in text:
# #         return "TIMEOUT"
# #     if "mediapipe" in text:
# #         return "MEDIAPIPE_ERROR"
# #     if "yolo" in text or "ultralytics" in text:
# #         return "YOLO_ERROR"
# #     if "cv2" in text or "opencv" in text or name == "error":
# #         return "OPENCV_ERROR"
# #     if "cannot open" in text or "corrupt" in text or "decode" in text or "invalid" in text:
# #         return "DECODE_ERROR"
# #     return "PROCESSING_ERROR"


# # def compute_journey_status(
# #     total_videos: int,
# #     succeeded_video_ids: "set[int]",
# #     failed_video_ids: "set[int]",
# # ) -> str:
# #     """
# #     Computes the journey-level terminal status per Phase 1 spec:
# #         COMPLETED              — every video succeeded
# #         COMPLETED_WITH_ERRORS  — at least one succeeded AND at least one failed
# #         FAILED                 — every video failed (or none succeeded)

# #     TIMED_OUT is NOT computed here — that's set explicitly by the watchdog
# #     path in consumer.py, which knows it hit a timeout rather than inferring
# #     it from video counts.
# #     """
# #     if total_videos == 0:
# #         return "FAILED"
# #     if not failed_video_ids:
# #         return "COMPLETED"
# #     if succeeded_video_ids:
# #         return "COMPLETED_WITH_ERRORS"
# #     return "FAILED"

# # # # # # # """
# # # # # # # callback_client.py
# # # # # # # ──────────────────
# # # # # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # # # # # All three public helpers are fire-and-forget from the caller's perspective;
# # # # # # # they raise on non-2xx so the consumer can route to the failure path.
# # # # # # # """

# # # # # # # from __future__ import annotations

# # # # # # # import logging
# # # # # # # import os
# # # # # # # from typing import Any, Dict, List

# # # # # # # import requests
# # # # # # # from dotenv import load_dotenv

# # # # # # # # ── Credentials / config ────────────────────────────────────────────────────
# # # # # # # _ENV_PATH = os.path.join(
# # # # # # #     os.path.dirname(os.path.abspath(__file__)),
# # # # # # #     "config", "credentials.env",
# # # # # # # )
# # # # # # # load_dotenv(_ENV_PATH)

# # # # # # # _BASE_URL = os.environ.get("SPRING_BOOT_BASE_URL", "http://localhost:8080")
# # # # # # # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # # # # # log = logging.getLogger("callback_client")


# # # # # # # # ── Internal helper ──────────────────────────────────────────────────────────

# # # # # # # def _post(path: str, payload: Dict[str, Any]) -> None:
# # # # # # #     url = f"{_BASE_URL}{path}"
# # # # # # #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# # # # # # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# # # # # # #     if not resp.ok:
# # # # # # #         raise RuntimeError(
# # # # # # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # # # # # #         )
# # # # # # #     log.debug("[Callback] %s → %d", path, resp.status_code)


# # # # # # # # ── Public API ───────────────────────────────────────────────────────────────

# # # # # # # def send_progress(
# # # # # # #     job_id:     str,
# # # # # # #     journey_id: int,
# # # # # # #     progress:   int,
# # # # # # #     message:    str,
# # # # # # #     status:     str = "PROCESSING",
# # # # # # # ) -> None:
# # # # # # #     """
# # # # # # #     POST /api/internal/analysis/progress

# # # # # # #     Called periodically during analysis to update the frontend progress bar
# # # # # # #     and the SSE stream.
# # # # # # #     """
# # # # # # #     _post(
# # # # # # #         "/api/internal/analysis/progress",
# # # # # # #         {
# # # # # # #             "jobId":      job_id,
# # # # # # #             "journeyId":  journey_id,
# # # # # # #             "status":     status,
# # # # # # #             "progress":   progress,
# # # # # # #             "message":    message,
# # # # # # #         },
# # # # # # #     )


# # # # # # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # # # # # #     """
# # # # # # #     POST /api/internal/analysis/completed

# # # # # # #     Called once — after ALL videos in the journey have been processed and
# # # # # # #     their violation frames uploaded to S3.

# # # # # # #     Expected shape of completion_payload
# # # # # # #     ─────────────────────────────────────
# # # # # # #     {
# # # # # # #         "jobId":          str,
# # # # # # #         "journeyId":      int,
# # # # # # #         "processingTime": int,          # wall-clock ms
# # # # # # #         "videoResults":   List[dict],   # see models.py → VideoResult
# # # # # # #     }
# # # # # # #     """
# # # # # # #     _post("/api/internal/analysis/completed", completion_payload)


# # # # # # # def send_failed(
# # # # # # #     job_id:        str,
# # # # # # #     journey_id:    int,
# # # # # # #     error_message: str,
# # # # # # # ) -> None:
# # # # # # #     """
# # # # # # #     POST /api/internal/analysis/failed

# # # # # # #     Called whenever an unrecoverable exception occurs during job processing.
# # # # # # #     Spring Boot will mark AnalysisJob and Journey as FAILED.
# # # # # # #     """
# # # # # # #     try:
# # # # # # #         _post(
# # # # # # #             "/api/internal/analysis/failed",
# # # # # # #             {
# # # # # # #                 "jobId":        job_id,
# # # # # # #                 "journeyId":    journey_id,
# # # # # # #                 "errorMessage": error_message,
# # # # # # #             },
# # # # # # #         )
# # # # # # #     except Exception as exc:
# # # # # # #         # Failure callback must never itself raise — log and swallow.
# # # # # # #         log.error("[Callback] send_failed itself failed: %s", exc)

# # # # # # """
# # # # # # callback_client.py
# # # # # # ──────────────────
# # # # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # # # # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs
# # # # # # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).

# # # # # # Changes from previous version
# # # # # # ──────────────────────────────
# # # # # # • _BASE_URL now defaults to the live CVVRS domain (overridable via env var
# # # # # #   SPRING_BOOT_BASE_URL for local testing).
# # # # # # • send_progress()  — payload now includes the `currentVideo` field required
# # # # # #   by the API.
# # # # # # • send_completed() — accepts a fully-formed dict (built by CompletionPayload.
# # # # # #   to_dict()); no shape changes needed here since the caller owns the dict.
# # # # # # • send_failed()    — payload drops `journeyId` (not in the API spec for this
# # # # # #   endpoint); only jobId + errorMessage are sent.
# # # # # # """

# # # # # # from __future__ import annotations

# # # # # # import logging
# # # # # # import os
# # # # # # from typing import Any, Dict, Optional

# # # # # # import requests
# # # # # # from dotenv import load_dotenv

# # # # # # # ── Credentials / config ────────────────────────────────────────────────────
# # # # # # _ENV_PATH = os.path.join(
# # # # # #     os.path.dirname(os.path.abspath(__file__)),
# # # # # #     "config", "credentials.env",
# # # # # # )
# # # # # # load_dotenv(_ENV_PATH)

# # # # # # # Default to the live CVVRS server; override with SPRING_BOOT_BASE_URL for local dev.
# # # # # # _BASE_URL = os.environ.get(
# # # # # #     "SPRING_BOOT_BASE_URL",
# # # # # #     "https://cvvrsrailway-api.sconexsoft.com/cvs",
# # # # # # )
# # # # # # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # # # # log = logging.getLogger("callback_client")


# # # # # # # ── Internal helper ──────────────────────────────────────────────────────────

# # # # # # def _post(path: str, payload: Dict[str, Any]) -> None:
# # # # # #     """
# # # # # #     POST to a Spring Boot internal endpoint.
# # # # # #     No Authorization header is sent — /api/internal/* are worker-only endpoints.
# # # # # #     Raises RuntimeError on non-2xx so the consumer can route to failure path.
# # # # # #     """
# # # # # #     url = f"{_BASE_URL}{path}"
# # # # # #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# # # # # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# # # # # #     if not resp.ok:
# # # # # #         raise RuntimeError(
# # # # # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # # # # #         )
# # # # # #     log.debug("[Callback] %s → %d", path, resp.status_code)


# # # # # # # ── Public API ───────────────────────────────────────────────────────────────

# # # # # # def send_progress(
# # # # # #     job_id:        str,
# # # # # #     journey_id:    int,
# # # # # #     progress:      int,
# # # # # #     message:       str,
# # # # # #     status:        str = "PROCESSING",
# # # # # #     current_video: Optional[int] = None,   # NEW — index of video currently being processed
# # # # # # ) -> None:
# # # # # #     """
# # # # # #     POST /api/internal/analysis/progress

# # # # # #     Called periodically during analysis to update the frontend progress bar
# # # # # #     and the SSE stream.

# # # # # #     Parameters
# # # # # #     ──────────
# # # # # #     job_id        : RabbitMQ job ID.
# # # # # #     journey_id    : Journey ID.
# # # # # #     progress      : 0–100 integer.
# # # # # #     message       : Human-readable status message.
# # # # # #     status        : "PROCESSING" | "COMPLETED" | "FAILED"
# # # # # #     current_video : 1-based index of the video currently being processed
# # # # # #                     (omitted from payload when None).
# # # # # #     """
# # # # # #     payload: Dict[str, Any] = {
# # # # # #         "jobId":      job_id,
# # # # # #         "journeyId":  journey_id,
# # # # # #         "status":     status,
# # # # # #         "progress":   progress,
# # # # # #         "message":    message,
# # # # # #     }
# # # # # #     if current_video is not None:
# # # # # #         payload["currentVideo"] = current_video

# # # # # #     _post("/api/internal/analysis/progress", payload)


# # # # # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # # # # #     """
# # # # # #     POST /api/internal/analysis/completed

# # # # # #     Called once after ALL videos in the journey have been processed and their
# # # # # #     violation frames uploaded to S3.

# # # # # #     Expected shape of completion_payload (built by CompletionPayload.to_dict())
# # # # # #     ────────────────────────────────────────────────────────────────────────────
# # # # # #     {
# # # # # #         "jobId":         str,
# # # # # #         "journeyId":     int,
# # # # # #         "batchId":       str,           # "BATCH-<jobId>" if not supplied
# # # # # #         "trainDetailId": int,
# # # # # #         "folderName":    str,
# # # # # #         "processingTime":int,           # wall-clock ms
# # # # # #         "videoResults": [
# # # # # #             {
# # # # # #                 "videoId":         str,       # STRING per API spec
# # # # # #                 "sequenceNo":      int,
# # # # # #                 "durationSeconds": float,
# # # # # #                 "originalS3Key":   str,
# # # # # #                 "violations": [
# # # # # #                     {
# # # # # #                         "violationType":          str,
# # # # # #                         "severity":               str,
# # # # # #                         "confidence":             float,
# # # # # #                         "riskScore":              float,
# # # # # #                         "timestamp":              float,   # journey-global seconds
# # # # # #                         "originalVideoTimestamp": float,   # local-video seconds
# # # # # #                         "framePaths":             [str]
# # # # # #                     }
# # # # # #                 ]
# # # # # #             }
# # # # # #         ]
# # # # # #     }
# # # # # #     """
# # # # # #     _post("/api/internal/analysis/completed", completion_payload)


# # # # # # def send_failed(
# # # # # #     job_id:        str,
# # # # # #     journey_id:    int,          # kept as parameter for caller convenience / logging
# # # # # #     error_message: str,
# # # # # # ) -> None:
# # # # # #     """
# # # # # #     POST /api/internal/analysis/failed

# # # # # #     Called whenever an unrecoverable exception occurs during job processing.
# # # # # #     Spring Boot will mark AnalysisJob and Journey as FAILED.

# # # # # #     Note: The API spec only requires jobId + errorMessage in the request body.
# # # # # #     journeyId is accepted as a parameter here for logging but is NOT included
# # # # # #     in the outbound payload.
# # # # # #     """
# # # # # #     try:
# # # # # #         _post(
# # # # # #             "/api/internal/analysis/failed",
# # # # # #             {
# # # # # #                 "jobId":        job_id,
# # # # # #                 "errorMessage": error_message,
# # # # # #             },
# # # # # #         )
# # # # # #     except Exception as exc:
# # # # # #         # Failure callback must never itself raise — log and swallow.
# # # # # #         log.error(
# # # # # #             "[Callback] send_failed itself failed (job=%s journey=%d): %s",
# # # # # #             job_id, journey_id, exc,
# # # # # #         )



# # # # # """

# # # # # callback_client.py

# # # # # ──────────────────

# # # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.
 
# # # # # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs

# # # # # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).
 
# # # # # Fixes from previous version

# # # # # ─────────────────────────────

# # # # # • Added set_base_url() — called by consumer.py with the callbackBaseUrl from

# # # # #   the RabbitMQ message, so Python posts to the correct Spring Boot server

# # # # #   per environment without needing env var changes.

# # # # # • _BASE_URL still falls back to env var SPRING_BOOT_BASE_URL or the live

# # # # #   server when callbackBaseUrl is not provided in the message.

# # # # # """
 
# # # # # from __future__ import annotations
 
# # # # # import logging

# # # # # import os

# # # # # from typing import Any, Dict, Optional
 
# # # # # import requests

# # # # # from dotenv import load_dotenv
 
# # # # # # ── Credentials / config ────────────────────────────────────────────────────

# # # # # _ENV_PATH = os.path.join(

# # # # #     os.path.dirname(os.path.abspath(__file__)),

# # # # #     "config", "credentials.env",

# # # # # )

# # # # # load_dotenv(_ENV_PATH)
 
# # # # # # Default to the live CVVRS server; override with SPRING_BOOT_BASE_URL for local dev

# # # # # # or call set_base_url() with the callbackBaseUrl from the RabbitMQ message.

# # # # # _BASE_URL = os.environ.get(

# # # # #     "SPRING_BOOT_BASE_URL",

# # # # #     "https://cvvrsrailway-api.sconexsoft.com/cvs",

# # # # # )

# # # # # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))
 
# # # # # log = logging.getLogger("callback_client")
 
 
# # # # # # ── FIX: dynamic base URL setter ─────────────────────────────────────────────
 
# # # # # def set_base_url(url: str) -> None:

# # # # #     """

# # # # #     Override the callback base URL at runtime.
 
# # # # #     Called by consumer.py when the RabbitMQ message contains a callbackBaseUrl.

# # # # #     This allows the same Python worker to callback correctly to both local

# # # # #     and staging Spring Boot servers without changing env vars.
 
# # # # #     Example values:

# # # # #         "http://localhost:8093/api/internal/analysis"         (local)

# # # # #         "https://cvvrsrailway-api.sconexsoft.com/cvs/api/internal/analysis"  (staging)
 
# # # # #     Note: The URL passed here is the FULL path up to /api/internal/analysis.

# # # # #     The individual callbacks append /progress, /completed, /failed.

# # # # #     """

# # # # #     global _BASE_URL

# # # # #     # Strip the /api/internal/analysis suffix if present — we add it in _post()

# # # # #     base = url.rstrip("/")

# # # # #     if base.endswith("/api/internal/analysis"):

# # # # #         base = base[: -len("/api/internal/analysis")]

# # # # #     _BASE_URL = base

# # # # #     log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)
 
 
# # # # # # ── Internal helper ──────────────────────────────────────────────────────────
 
# # # # # def _post(path: str, payload: Dict[str, Any]) -> None:

# # # # #     """

# # # # #     POST to a Spring Boot internal endpoint.

# # # # #     No Authorization header is sent — /api/internal/* are worker-only endpoints.

# # # # #     Raises RuntimeError on non-2xx so the consumer can route to failure path.

# # # # #     """

# # # # #     url = f"{_BASE_URL}{path}"

# # # # #     log.debug("[Callback] POST %s  payload=%s", url, payload)

# # # # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)

# # # # #     if not resp.ok:

# # # # #         raise RuntimeError(

# # # # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"

# # # # #         )

# # # # #     log.debug("[Callback] %s → %d", path, resp.status_code)
 
 
# # # # # # ── Public API ───────────────────────────────────────────────────────────────
 
# # # # # def send_progress(

# # # # #     job_id:        str,

# # # # #     journey_id:    int,

# # # # #     progress:      int,

# # # # #     message:       str,

# # # # #     status:        str = "PROCESSING",

# # # # #     current_video: Optional[int] = None,

# # # # # ) -> None:

# # # # #     """

# # # # #     POST /api/internal/analysis/progress
 
# # # # #     Called periodically during analysis to update the frontend progress bar

# # # # #     and the SSE stream.
 
# # # # #     Parameters

# # # # #     ──────────

# # # # #     job_id        : RabbitMQ job ID.

# # # # #     journey_id    : Journey ID.

# # # # #     progress      : 0–100 integer.

# # # # #     message       : Human-readable status message.

# # # # #     status        : "PROCESSING" | "COMPLETED" | "FAILED"

# # # # #     current_video : 1-based index of the video currently being processed.

# # # # #     """

# # # # #     payload: Dict[str, Any] = {

# # # # #         "jobId":      job_id,

# # # # #         "journeyId":  journey_id,

# # # # #         "status":     status,

# # # # #         "progress":   progress,

# # # # #         "message":    message,

# # # # #     }

# # # # #     if current_video is not None:

# # # # #         payload["currentVideo"] = current_video
 
# # # # #     _post("/api/internal/analysis/progress", payload)
 
 
# # # # # def send_completed(completion_payload: Dict[str, Any]) -> None:

# # # # #     """

# # # # #     POST /api/internal/analysis/completed
 
# # # # #     Called once after ALL videos in the journey have been processed and their

# # # # #     violation frames uploaded to S3.
 
# # # # #     completion_payload is built by CompletionPayload.to_dict() in models.py.

# # # # #     """

# # # # #     _post("/api/internal/analysis/completed", completion_payload)
 
 
# # # # # def send_failed(

# # # # #     job_id:        str,

# # # # #     journey_id:    int,          # kept for caller convenience / logging

# # # # #     error_message: str,

# # # # # ) -> None:

# # # # #     """

# # # # #     POST /api/internal/analysis/failed
 
# # # # #     Called whenever an unrecoverable exception occurs during job processing.

# # # # #     Spring Boot will mark AnalysisJob and Journey as FAILED.
 
# # # # #     Note: The API spec only requires jobId + errorMessage.

# # # # #     journeyId is accepted as a parameter for logging but is NOT included

# # # # #     in the outbound payload.

# # # # #     """

# # # # #     try:

# # # # #         _post(

# # # # #             "/api/internal/analysis/failed",

# # # # #             {

# # # # #                 "jobId":        job_id,

# # # # #                 "errorMessage": error_message,

# # # # #             },

# # # # #         )

# # # # #     except Exception as exc:

# # # # #         # Failure callback must never itself raise — log and swallow.

# # # # #         log.error(

# # # # #             "[Callback] send_failed itself failed (job=%s journey=%d): %s",

# # # # #             job_id, journey_id, exc,

# # # # #         )
 
 
 
# # # # """
# # # # callback_client.py
# # # # ──────────────────
# # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs
# # # # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).

# # # # Changes from previous version
# # # # ──────────────────────────────
# # # # • Added check_job_completed(job_id) — queries GET /api/internal/analysis/status/{jobId}
# # # #   and returns True when the backend reports the job is already COMPLETED.
# # # #   Called by consumer.py as an idempotency guard before starting any processing
# # # #   on a redelivered message.
# # # # """

# # # # from __future__ import annotations
# # # # import logging
# # # # import os
# # # # from typing import Any, Dict, Optional

# # # # import requests
# # # # from dotenv import load_dotenv

# # # # # ── Credentials / config ─────────────────────────────────────────────────────
# # # # _ENV_PATH = os.path.join(
# # # #     os.path.dirname(os.path.abspath(__file__)),
# # # #     "config", "credentials.env",
# # # # )
# # # # load_dotenv(_ENV_PATH)

# # # # _BASE_URL = os.environ.get(
# # # #     "SPRING_BOOT_BASE_URL",
# # # #     "https://cvvrsrailway-api.sconexsoft.com/cvs",
# # # # )
# # # # _TIMEOUT = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # # log = logging.getLogger("callback_client")


# # # # # ── Dynamic base URL setter ───────────────────────────────────────────────────

# # # # def set_base_url(url: str) -> None:
# # # #     """
# # # #     Override the callback base URL at runtime.

# # # #     Called by consumer.py when the RabbitMQ message contains a callbackBaseUrl.
# # # #     Allows the same Python worker to callback correctly to both local and staging
# # # #     Spring Boot servers without changing env vars.

# # # #     The URL passed here is the FULL path up to /api/internal/analysis.
# # # #     Individual callbacks append /progress, /completed, /failed, /status/{id}.
# # # #     """
# # # #     global _BASE_URL
# # # #     base = url.rstrip("/")
# # # #     if base.endswith("/api/internal/analysis"):
# # # #         base = base[: -len("/api/internal/analysis")]
# # # #     _BASE_URL = base
# # # #     log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)


# # # # # ── Internal helpers ──────────────────────────────────────────────────────────

# # # # def _post(path: str, payload: Dict[str, Any]) -> None:
# # # #     """
# # # #     POST to a Spring Boot internal endpoint.
# # # #     No Authorization header — /api/internal/* are worker-only endpoints.
# # # #     Raises RuntimeError on non-2xx so the consumer can route to the failure path.
# # # #     """
# # # #     url = f"{_BASE_URL}{path}"
# # # #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# # # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# # # #     if not resp.ok:
# # # #         raise RuntimeError(
# # # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # # #         )
# # # #     log.debug("[Callback] %s → %d", path, resp.status_code)


# # # # def _get(path: str) -> requests.Response:
# # # #     """
# # # #     GET from a Spring Boot internal endpoint.
# # # #     Raises RuntimeError on non-2xx.
# # # #     """
# # # #     url = f"{_BASE_URL}{path}"
# # # #     log.debug("[Callback] GET %s", url)
# # # #     resp = requests.get(url, timeout=_TIMEOUT)
# # # #     if not resp.ok:
# # # #         raise RuntimeError(
# # # #             f"Callback GET {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # # #         )
# # # #     log.debug("[Callback] %s → %d", path, resp.status_code)
# # # #     return resp


# # # # # ── Public API ────────────────────────────────────────────────────────────────

# # # # def check_job_completed(job_id: str) -> bool:
# # # #     """
# # # #     NEW — Idempotency check.

# # # #     Queries GET /api/internal/analysis/status/{jobId} and returns True when
# # # #     the backend reports the job status as COMPLETED.

# # # #     Called by consumer.py at the very start of _handle_job() so that
# # # #     RabbitMQ redeliveries of already-completed jobs are detected immediately
# # # #     and ACKed without re-running any processing.

# # # #     Backend contract (expected JSON shape):
# # # #         { "status": "COMPLETED" }   → job already done → return True
# # # #         { "status": "PROCESSING" }  → in-flight (rare edge-case) → return False
# # # #         { "status": "PENDING" }     → not yet processed → return False
# # # #         404 Not Found               → job unknown (treat as not completed)

# # # #     If the endpoint does not exist yet on your Spring Boot side, add it as:
# # # #         GET /api/internal/analysis/status/{jobId}
# # # #         → ResponseEntity<Map<String,String>>  { "status": job.getStatus().name() }

# # # #     Raises on network errors so the consumer can decide whether to proceed
# # # #     with processing or skip (consumer.py catches and proceeds on error).
# # # #     """
# # # #     try:
# # # #         resp = _get(f"/api/internal/analysis/status/{job_id}")
# # # #         data = resp.json()
# # # #         status = data.get("status", "").upper()
# # # #         is_done = status == "COMPLETED"
# # # #         log.info(
# # # #             "[Callback] check_job_completed  job=%s  status=%s  is_done=%s",
# # # #             job_id, status, is_done,
# # # #         )
# # # #         return is_done
# # # #     except RuntimeError as exc:
# # # #         # 404 → job not found in the backend → definitely not completed
# # # #         if "404" in str(exc):
# # # #             log.info(
# # # #                 "[Callback] check_job_completed  job=%s  → 404 (not found) → not completed",
# # # #                 job_id,
# # # #             )
# # # #             return False
# # # #         raise


# # # # def send_progress(
# # # #     job_id:        str,
# # # #     journey_id:    int,
# # # #     progress:      int,
# # # #     message:       str,
# # # #     status:        str = "PROCESSING",
# # # #     current_video: Optional[int] = None,
# # # # ) -> None:
# # # #     """
# # # #     POST /api/internal/analysis/progress

# # # #     Called periodically during analysis to update the frontend progress bar
# # # #     and the SSE stream.

# # # #     Parameters
# # # #     ──────────
# # # #     job_id        : RabbitMQ job ID.
# # # #     journey_id    : Journey ID.
# # # #     progress      : 0–100 integer.
# # # #     message       : Human-readable status message.
# # # #     status        : "PROCESSING" | "COMPLETED" | "FAILED"
# # # #     current_video : 1-based index of the video currently being processed.
# # # #     """
# # # #     payload: Dict[str, Any] = {
# # # #         "jobId":     job_id,
# # # #         "journeyId": journey_id,
# # # #         "status":    status,
# # # #         "progress":  progress,
# # # #         "message":   message,
# # # #     }
# # # #     if current_video is not None:
# # # #         payload["currentVideo"] = current_video
# # # #     _post("/api/internal/analysis/progress", payload)


# # # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # # #     """
# # # #     POST /api/internal/analysis/completed

# # # #     Called once after ALL videos in the journey have been processed and their
# # # #     violation frames uploaded to S3.

# # # #     completion_payload is built by CompletionPayload.to_dict() in models.py.
# # # #     """
# # # #     _post("/api/internal/analysis/completed", completion_payload)


# # # # def send_failed(
# # # #     job_id:        str,
# # # #     journey_id:    int,   # kept for caller convenience / logging
# # # #     error_message: str,
# # # # ) -> None:
# # # #     """
# # # #     POST /api/internal/analysis/failed

# # # #     Called whenever an unrecoverable exception occurs during job processing.
# # # #     Spring Boot will mark AnalysisJob and Journey as FAILED.

# # # #     Note: The API spec only requires jobId + errorMessage.
# # # #     journeyId is accepted as a parameter for logging but is NOT included
# # # #     in the outbound payload.
# # # #     """
# # # #     try:
# # # #         _post(
# # # #             "/api/internal/analysis/failed",
# # # #             {
# # # #                 "jobId":        job_id,
# # # #                 "errorMessage": error_message,
# # # #             },
# # # #         )
# # # #     except Exception as exc:
# # # #         # Failure callback must never itself raise — log and swallow.
# # # #         log.error(
# # # #             "[Callback] send_failed itself failed (job=%s journey=%d): %s",
# # # #             job_id, journey_id, exc,
# # # #         )



# # # """
# # # callback_client.py
# # # ──────────────────
# # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs
# # # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).

# # # Changes in this version
# # # ────────────────────────
# # # • check_job_completed() now uses the EXISTING completed-callback endpoint
# # #   as a probe instead of a dedicated /status/{jobId} endpoint that doesn't
# # #   exist yet on Spring Boot.

# # #   Strategy: POST /api/internal/analysis/completed with only { "jobId": id,
# # #   "probe": true }.  Spring Boot should return 409 CONFLICT (or any non-500)
# # #   when the job is already done.  If it returns 500 we treat it as "unknown"
# # #   and fall through to processing (safe default).

# # #   *** TEMPORARY until the Spring Boot /status endpoint is implemented. ***
# # #   Once GET /api/internal/analysis/status/{jobId} is live, revert
# # #   check_job_completed() to use _get() as originally written.
# # # """

# # # from __future__ import annotations

# # # import logging
# # # import os
# # # from typing import Any, Dict, Optional

# # # import requests
# # # from dotenv import load_dotenv

# # # # ── Credentials / config ─────────────────────────────────────────────────────
# # # _ENV_PATH = os.path.join(
# # #     os.path.dirname(os.path.abspath(__file__)),
# # #     "config", "credentials.env",
# # # )
# # # load_dotenv(_ENV_PATH)

# # # _BASE_URL = os.environ.get(
# # #     "SPRING_BOOT_BASE_URL",
# # #     "https://cvvrsrailway-api.sconexsoft.com/cvs",
# # # )
# # # _TIMEOUT = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # log = logging.getLogger("callback_client")


# # # # ── Dynamic base URL setter ───────────────────────────────────────────────────

# # # def set_base_url(url: str) -> None:
# # #     global _BASE_URL
# # #     base = url.rstrip("/")
# # #     if base.endswith("/api/internal/analysis"):
# # #         base = base[: -len("/api/internal/analysis")]
# # #     _BASE_URL = base
# # #     log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)


# # # # ── Internal helpers ──────────────────────────────────────────────────────────

# # # def _post(path: str, payload: Dict[str, Any]) -> requests.Response:
# # #     url = f"{_BASE_URL}{path}"
# # #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# # #     if not resp.ok:
# # #         raise RuntimeError(
# # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # #         )
# # #     log.debug("[Callback] %s → %d", path, resp.status_code)
# # #     return resp


# # # def _get(path: str) -> requests.Response:
# # #     url = f"{_BASE_URL}{path}"
# # #     log.debug("[Callback] GET %s", url)
# # #     resp = requests.get(url, timeout=_TIMEOUT)
# # #     if not resp.ok:
# # #         raise RuntimeError(
# # #             f"Callback GET {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # #         )
# # #     log.debug("[Callback] %s → %d", path, resp.status_code)
# # #     return resp


# # # # ── Public API ────────────────────────────────────────────────────────────────

# # # def check_job_completed(job_id: str) -> bool:
# # #     """
# # #     Idempotency check — returns True if the backend already has this job as COMPLETED.

# # #     TWO-STAGE STRATEGY
# # #     ──────────────────
# # #     Stage 1 (permanent): GET /api/internal/analysis/status/{jobId}
# # #         → { "status": "COMPLETED" }  → True
# # #         → { "status": "PENDING/PROCESSING" } → False
# # #         → 404 → False  (job not known yet)
# # #         → 500 → fall through to Stage 2

# # #     Stage 2 (temporary fallback until Spring Boot implements /status):
# # #         Uses GET /api/internal/analysis/job/{jobId} or any existing
# # #         read endpoint.  If that also 500s, we return False (safe default:
# # #         process the job rather than silently drop it).

# # #     WHAT TO IMPLEMENT ON SPRING BOOT (one small controller method):
# # #     ───────────────────────────────────────────────────────────────
# # #     @GetMapping("/api/internal/analysis/status/{jobId}")
# # #     public ResponseEntity<Map<String,String>> getJobStatus(
# # #             @PathVariable String jobId) {
# # #         return analysisJobRepository.findByJobId(jobId)
# # #             .map(job -> ResponseEntity.ok(Map.of("status", job.getStatus().name())))
# # #             .orElse(ResponseEntity.notFound().<Map<String,String>>build());
# # #     }

# # #     Once that endpoint is deployed, Stage 2 below can be deleted.
# # #     """

# # #     # ── Stage 1: dedicated status endpoint ───────────────────────────────────
# # #     try:
# # #         resp = _get(f"/api/internal/analysis/status/{job_id}")
# # #         status = resp.json().get("status", "").upper()
# # #         is_done = status == "COMPLETED"
# # #         log.info(
# # #             "[Callback] idempotency check (status endpoint)  job=%s  status=%s  done=%s",
# # #             job_id, status, is_done,
# # #         )
# # #         return is_done
# # #     except RuntimeError as exc:
# # #         if "404" in str(exc):
# # #             log.info(
# # #                 "[Callback] idempotency check  job=%s  → 404 → not completed", job_id
# # #             )
# # #             return False
# # #         if "500" in str(exc):
# # #             log.warning(
# # #                 "[Callback] idempotency check  job=%s  → /status endpoint returned 500 "
# # #                 "(not implemented yet?) — trying fallback probe.", job_id
# # #             )
# # #             # fall through to Stage 2
# # #         else:
# # #             # Network error or unexpected status — safe default: process the job
# # #             log.warning(
# # #                 "[Callback] idempotency check  job=%s  → unexpected error (%s) "
# # #                 "— proceeding with processing.", job_id, exc
# # #             )
# # #             return False

# # #     # ── Stage 2: fallback probe using progress endpoint ───────────────────────
# # #     # We send a lightweight progress probe at 0 % with status=CHECK.
# # #     # Spring Boot should:
# # #     #   • Return 200 if the job is still PENDING/PROCESSING (→ process it).
# # #     #   • Return 409 CONFLICT if the job is already COMPLETED (→ skip it).
# # #     #   • Return 404 if the job is unknown (→ process it).
# # #     # If the backend doesn't handle the CHECK status specially it will just
# # #     # update progress to 0 — harmless on an already-completed job.
# # #     #
# # #     # THIS IS A TEMPORARY FALLBACK.  Remove once /status endpoint is live.
# # #     try:
# # #         url = f"{_BASE_URL}/api/internal/analysis/progress"
# # #         resp = requests.post(
# # #             url,
# # #             json={
# # #                 "jobId":    job_id,
# # #                 "status":   "CHECK",   # sentinel value Spring Boot can detect
# # #                 "progress": 0,
# # #                 "message":  "idempotency-probe",
# # #             },
# # #             timeout=_TIMEOUT,
# # #         )
# # #         if resp.status_code == 409:
# # #             log.warning(
# # #                 "[Callback] idempotency check (probe)  job=%s  → 409 CONFLICT "
# # #                 "→ already COMPLETED — will skip.", job_id
# # #             )
# # #             return True
# # #         log.info(
# # #             "[Callback] idempotency check (probe)  job=%s  → HTTP %d → not completed.",
# # #             job_id, resp.status_code,
# # #         )
# # #         return False
# # #     except Exception as exc2:
# # #         log.warning(
# # #             "[Callback] idempotency check (probe) failed  job=%s: %s "
# # #             "— proceeding with processing.", job_id, exc2
# # #         )
# # #         return False


# # # def send_progress(
# # #     job_id:        str,
# # #     journey_id:    int,
# # #     progress:      int,
# # #     message:       str,
# # #     status:        str = "PROCESSING",
# # #     current_video: Optional[int] = None,
# # # ) -> None:
# # #     payload: Dict[str, Any] = {
# # #         "jobId":     job_id,
# # #         "journeyId": journey_id,
# # #         "status":    status,
# # #         "progress":  progress,
# # #         "message":   message,
# # #     }
# # #     if current_video is not None:
# # #         payload["currentVideo"] = current_video
# # #     _post("/api/internal/analysis/progress", payload)


# # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # #     _post("/api/internal/analysis/completed", completion_payload)


# # # def send_failed(
# # #     job_id:        str,
# # #     journey_id:    int,
# # #     error_message: str,
# # # ) -> None:
# # #     try:
# # #         _post(
# # #             "/api/internal/analysis/failed",
# # #             {
# # #                 "jobId":        job_id,
# # #                 "errorMessage": error_message,
# # #             },
# # #         )
# # #     except Exception as exc:
# # #         log.error(
# # #             "[Callback] send_failed itself failed (job=%s journey=%d): %s",
# # #             job_id, journey_id, exc,
# # #         )



# # # # # # """
# # # # # # callback_client.py
# # # # # # ──────────────────
# # # # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # # # # All three public helpers are fire-and-forget from the caller's perspective;
# # # # # # they raise on non-2xx so the consumer can route to the failure path.
# # # # # # """

# # # # # # from __future__ import annotations

# # # # # # import logging
# # # # # # import os
# # # # # # from typing import Any, Dict, List

# # # # # # import requests
# # # # # # from dotenv import load_dotenv

# # # # # # # ── Credentials / config ────────────────────────────────────────────────────
# # # # # # _ENV_PATH = os.path.join(
# # # # # #     os.path.dirname(os.path.abspath(__file__)),
# # # # # #     "config", "credentials.env",
# # # # # # )
# # # # # # load_dotenv(_ENV_PATH)

# # # # # # _BASE_URL = os.environ.get("SPRING_BOOT_BASE_URL", "http://localhost:8080")
# # # # # # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # # # # log = logging.getLogger("callback_client")


# # # # # # # ── Internal helper ──────────────────────────────────────────────────────────

# # # # # # def _post(path: str, payload: Dict[str, Any]) -> None:
# # # # # #     url = f"{_BASE_URL}{path}"
# # # # # #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# # # # # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# # # # # #     if not resp.ok:
# # # # # #         raise RuntimeError(
# # # # # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # # # # #         )
# # # # # #     log.debug("[Callback] %s → %d", path, resp.status_code)


# # # # # # # ── Public API ───────────────────────────────────────────────────────────────

# # # # # # def send_progress(
# # # # # #     job_id:     str,
# # # # # #     journey_id: int,
# # # # # #     progress:   int,
# # # # # #     message:    str,
# # # # # #     status:     str = "PROCESSING",
# # # # # # ) -> None:
# # # # # #     """
# # # # # #     POST /api/internal/analysis/progress

# # # # # #     Called periodically during analysis to update the frontend progress bar
# # # # # #     and the SSE stream.
# # # # # #     """
# # # # # #     _post(
# # # # # #         "/api/internal/analysis/progress",
# # # # # #         {
# # # # # #             "jobId":      job_id,
# # # # # #             "journeyId":  journey_id,
# # # # # #             "status":     status,
# # # # # #             "progress":   progress,
# # # # # #             "message":    message,
# # # # # #         },
# # # # # #     )


# # # # # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # # # # #     """
# # # # # #     POST /api/internal/analysis/completed

# # # # # #     Called once — after ALL videos in the journey have been processed and
# # # # # #     their violation frames uploaded to S3.

# # # # # #     Expected shape of completion_payload
# # # # # #     ─────────────────────────────────────
# # # # # #     {
# # # # # #         "jobId":          str,
# # # # # #         "journeyId":      int,
# # # # # #         "processingTime": int,          # wall-clock ms
# # # # # #         "videoResults":   List[dict],   # see models.py → VideoResult
# # # # # #     }
# # # # # #     """
# # # # # #     _post("/api/internal/analysis/completed", completion_payload)


# # # # # # def send_failed(
# # # # # #     job_id:        str,
# # # # # #     journey_id:    int,
# # # # # #     error_message: str,
# # # # # # ) -> None:
# # # # # #     """
# # # # # #     POST /api/internal/analysis/failed

# # # # # #     Called whenever an unrecoverable exception occurs during job processing.
# # # # # #     Spring Boot will mark AnalysisJob and Journey as FAILED.
# # # # # #     """
# # # # # #     try:
# # # # # #         _post(
# # # # # #             "/api/internal/analysis/failed",
# # # # # #             {
# # # # # #                 "jobId":        job_id,
# # # # # #                 "journeyId":    journey_id,
# # # # # #                 "errorMessage": error_message,
# # # # # #             },
# # # # # #         )
# # # # # #     except Exception as exc:
# # # # # #         # Failure callback must never itself raise — log and swallow.
# # # # # #         log.error("[Callback] send_failed itself failed: %s", exc)

# # # # # """
# # # # # callback_client.py
# # # # # ──────────────────
# # # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # # # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs
# # # # # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).

# # # # # Changes from previous version
# # # # # ──────────────────────────────
# # # # # • _BASE_URL now defaults to the live CVVRS domain (overridable via env var
# # # # #   SPRING_BOOT_BASE_URL for local testing).
# # # # # • send_progress()  — payload now includes the `currentVideo` field required
# # # # #   by the API.
# # # # # • send_completed() — accepts a fully-formed dict (built by CompletionPayload.
# # # # #   to_dict()); no shape changes needed here since the caller owns the dict.
# # # # # • send_failed()    — payload drops `journeyId` (not in the API spec for this
# # # # #   endpoint); only jobId + errorMessage are sent.
# # # # # """

# # # # # from __future__ import annotations

# # # # # import logging
# # # # # import os
# # # # # from typing import Any, Dict, Optional

# # # # # import requests
# # # # # from dotenv import load_dotenv

# # # # # # ── Credentials / config ────────────────────────────────────────────────────
# # # # # _ENV_PATH = os.path.join(
# # # # #     os.path.dirname(os.path.abspath(__file__)),
# # # # #     "config", "credentials.env",
# # # # # )
# # # # # load_dotenv(_ENV_PATH)

# # # # # # Default to the live CVVRS server; override with SPRING_BOOT_BASE_URL for local dev.
# # # # # _BASE_URL = os.environ.get(
# # # # #     "SPRING_BOOT_BASE_URL",
# # # # #     "https://cvvrsrailway-api.sconexsoft.com/cvs",
# # # # # )
# # # # # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # # # log = logging.getLogger("callback_client")


# # # # # # ── Internal helper ──────────────────────────────────────────────────────────

# # # # # def _post(path: str, payload: Dict[str, Any]) -> None:
# # # # #     """
# # # # #     POST to a Spring Boot internal endpoint.
# # # # #     No Authorization header is sent — /api/internal/* are worker-only endpoints.
# # # # #     Raises RuntimeError on non-2xx so the consumer can route to failure path.
# # # # #     """
# # # # #     url = f"{_BASE_URL}{path}"
# # # # #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# # # # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# # # # #     if not resp.ok:
# # # # #         raise RuntimeError(
# # # # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # # # #         )
# # # # #     log.debug("[Callback] %s → %d", path, resp.status_code)


# # # # # # ── Public API ───────────────────────────────────────────────────────────────

# # # # # def send_progress(
# # # # #     job_id:        str,
# # # # #     journey_id:    int,
# # # # #     progress:      int,
# # # # #     message:       str,
# # # # #     status:        str = "PROCESSING",
# # # # #     current_video: Optional[int] = None,   # NEW — index of video currently being processed
# # # # # ) -> None:
# # # # #     """
# # # # #     POST /api/internal/analysis/progress

# # # # #     Called periodically during analysis to update the frontend progress bar
# # # # #     and the SSE stream.

# # # # #     Parameters
# # # # #     ──────────
# # # # #     job_id        : RabbitMQ job ID.
# # # # #     journey_id    : Journey ID.
# # # # #     progress      : 0–100 integer.
# # # # #     message       : Human-readable status message.
# # # # #     status        : "PROCESSING" | "COMPLETED" | "FAILED"
# # # # #     current_video : 1-based index of the video currently being processed
# # # # #                     (omitted from payload when None).
# # # # #     """
# # # # #     payload: Dict[str, Any] = {
# # # # #         "jobId":      job_id,
# # # # #         "journeyId":  journey_id,
# # # # #         "status":     status,
# # # # #         "progress":   progress,
# # # # #         "message":    message,
# # # # #     }
# # # # #     if current_video is not None:
# # # # #         payload["currentVideo"] = current_video

# # # # #     _post("/api/internal/analysis/progress", payload)


# # # # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # # # #     """
# # # # #     POST /api/internal/analysis/completed

# # # # #     Called once after ALL videos in the journey have been processed and their
# # # # #     violation frames uploaded to S3.

# # # # #     Expected shape of completion_payload (built by CompletionPayload.to_dict())
# # # # #     ────────────────────────────────────────────────────────────────────────────
# # # # #     {
# # # # #         "jobId":         str,
# # # # #         "journeyId":     int,
# # # # #         "batchId":       str,           # "BATCH-<jobId>" if not supplied
# # # # #         "trainDetailId": int,
# # # # #         "folderName":    str,
# # # # #         "processingTime":int,           # wall-clock ms
# # # # #         "videoResults": [
# # # # #             {
# # # # #                 "videoId":         str,       # STRING per API spec
# # # # #                 "sequenceNo":      int,
# # # # #                 "durationSeconds": float,
# # # # #                 "originalS3Key":   str,
# # # # #                 "violations": [
# # # # #                     {
# # # # #                         "violationType":          str,
# # # # #                         "severity":               str,
# # # # #                         "confidence":             float,
# # # # #                         "riskScore":              float,
# # # # #                         "timestamp":              float,   # journey-global seconds
# # # # #                         "originalVideoTimestamp": float,   # local-video seconds
# # # # #                         "framePaths":             [str]
# # # # #                     }
# # # # #                 ]
# # # # #             }
# # # # #         ]
# # # # #     }
# # # # #     """
# # # # #     _post("/api/internal/analysis/completed", completion_payload)


# # # # # def send_failed(
# # # # #     job_id:        str,
# # # # #     journey_id:    int,          # kept as parameter for caller convenience / logging
# # # # #     error_message: str,
# # # # # ) -> None:
# # # # #     """
# # # # #     POST /api/internal/analysis/failed

# # # # #     Called whenever an unrecoverable exception occurs during job processing.
# # # # #     Spring Boot will mark AnalysisJob and Journey as FAILED.

# # # # #     Note: The API spec only requires jobId + errorMessage in the request body.
# # # # #     journeyId is accepted as a parameter here for logging but is NOT included
# # # # #     in the outbound payload.
# # # # #     """
# # # # #     try:
# # # # #         _post(
# # # # #             "/api/internal/analysis/failed",
# # # # #             {
# # # # #                 "jobId":        job_id,
# # # # #                 "errorMessage": error_message,
# # # # #             },
# # # # #         )
# # # # #     except Exception as exc:
# # # # #         # Failure callback must never itself raise — log and swallow.
# # # # #         log.error(
# # # # #             "[Callback] send_failed itself failed (job=%s journey=%d): %s",
# # # # #             job_id, journey_id, exc,
# # # # #         )



# # # # """

# # # # callback_client.py

# # # # ──────────────────

# # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.
 
# # # # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs

# # # # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).
 
# # # # Fixes from previous version

# # # # ─────────────────────────────

# # # # • Added set_base_url() — called by consumer.py with the callbackBaseUrl from

# # # #   the RabbitMQ message, so Python posts to the correct Spring Boot server

# # # #   per environment without needing env var changes.

# # # # • _BASE_URL still falls back to env var SPRING_BOOT_BASE_URL or the live

# # # #   server when callbackBaseUrl is not provided in the message.

# # # # """
 
# # # # from __future__ import annotations
 
# # # # import logging

# # # # import os

# # # # from typing import Any, Dict, Optional
 
# # # # import requests

# # # # from dotenv import load_dotenv
 
# # # # # ── Credentials / config ────────────────────────────────────────────────────

# # # # _ENV_PATH = os.path.join(

# # # #     os.path.dirname(os.path.abspath(__file__)),

# # # #     "config", "credentials.env",

# # # # )

# # # # load_dotenv(_ENV_PATH)
 
# # # # # Default to the live CVVRS server; override with SPRING_BOOT_BASE_URL for local dev

# # # # # or call set_base_url() with the callbackBaseUrl from the RabbitMQ message.

# # # # _BASE_URL = os.environ.get(

# # # #     "SPRING_BOOT_BASE_URL",

# # # #     "https://cvvrsrailway-api.sconexsoft.com/cvs",

# # # # )

# # # # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))
 
# # # # log = logging.getLogger("callback_client")
 
 
# # # # # ── FIX: dynamic base URL setter ─────────────────────────────────────────────
 
# # # # def set_base_url(url: str) -> None:

# # # #     """

# # # #     Override the callback base URL at runtime.
 
# # # #     Called by consumer.py when the RabbitMQ message contains a callbackBaseUrl.

# # # #     This allows the same Python worker to callback correctly to both local

# # # #     and staging Spring Boot servers without changing env vars.
 
# # # #     Example values:

# # # #         "http://localhost:8093/api/internal/analysis"         (local)

# # # #         "https://cvvrsrailway-api.sconexsoft.com/cvs/api/internal/analysis"  (staging)
 
# # # #     Note: The URL passed here is the FULL path up to /api/internal/analysis.

# # # #     The individual callbacks append /progress, /completed, /failed.

# # # #     """

# # # #     global _BASE_URL

# # # #     # Strip the /api/internal/analysis suffix if present — we add it in _post()

# # # #     base = url.rstrip("/")

# # # #     if base.endswith("/api/internal/analysis"):

# # # #         base = base[: -len("/api/internal/analysis")]

# # # #     _BASE_URL = base

# # # #     log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)
 
 
# # # # # ── Internal helper ──────────────────────────────────────────────────────────
 
# # # # def _post(path: str, payload: Dict[str, Any]) -> None:

# # # #     """

# # # #     POST to a Spring Boot internal endpoint.

# # # #     No Authorization header is sent — /api/internal/* are worker-only endpoints.

# # # #     Raises RuntimeError on non-2xx so the consumer can route to failure path.

# # # #     """

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

# # # #     job_id:        str,

# # # #     journey_id:    int,

# # # #     progress:      int,

# # # #     message:       str,

# # # #     status:        str = "PROCESSING",

# # # #     current_video: Optional[int] = None,

# # # # ) -> None:

# # # #     """

# # # #     POST /api/internal/analysis/progress
 
# # # #     Called periodically during analysis to update the frontend progress bar

# # # #     and the SSE stream.
 
# # # #     Parameters

# # # #     ──────────

# # # #     job_id        : RabbitMQ job ID.

# # # #     journey_id    : Journey ID.

# # # #     progress      : 0–100 integer.

# # # #     message       : Human-readable status message.

# # # #     status        : "PROCESSING" | "COMPLETED" | "FAILED"

# # # #     current_video : 1-based index of the video currently being processed.

# # # #     """

# # # #     payload: Dict[str, Any] = {

# # # #         "jobId":      job_id,

# # # #         "journeyId":  journey_id,

# # # #         "status":     status,

# # # #         "progress":   progress,

# # # #         "message":    message,

# # # #     }

# # # #     if current_video is not None:

# # # #         payload["currentVideo"] = current_video
 
# # # #     _post("/api/internal/analysis/progress", payload)
 
 
# # # # def send_completed(completion_payload: Dict[str, Any]) -> None:

# # # #     """

# # # #     POST /api/internal/analysis/completed
 
# # # #     Called once after ALL videos in the journey have been processed and their

# # # #     violation frames uploaded to S3.
 
# # # #     completion_payload is built by CompletionPayload.to_dict() in models.py.

# # # #     """

# # # #     _post("/api/internal/analysis/completed", completion_payload)
 
 
# # # # def send_failed(

# # # #     job_id:        str,

# # # #     journey_id:    int,          # kept for caller convenience / logging

# # # #     error_message: str,

# # # # ) -> None:

# # # #     """

# # # #     POST /api/internal/analysis/failed
 
# # # #     Called whenever an unrecoverable exception occurs during job processing.

# # # #     Spring Boot will mark AnalysisJob and Journey as FAILED.
 
# # # #     Note: The API spec only requires jobId + errorMessage.

# # # #     journeyId is accepted as a parameter for logging but is NOT included

# # # #     in the outbound payload.

# # # #     """

# # # #     try:

# # # #         _post(

# # # #             "/api/internal/analysis/failed",

# # # #             {

# # # #                 "jobId":        job_id,

# # # #                 "errorMessage": error_message,

# # # #             },

# # # #         )

# # # #     except Exception as exc:

# # # #         # Failure callback must never itself raise — log and swallow.

# # # #         log.error(

# # # #             "[Callback] send_failed itself failed (job=%s journey=%d): %s",

# # # #             job_id, journey_id, exc,

# # # #         )
 
 
 
# # # """
# # # callback_client.py
# # # ──────────────────
# # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs
# # # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).

# # # Changes from previous version
# # # ──────────────────────────────
# # # • Added check_job_completed(job_id) — queries GET /api/internal/analysis/status/{jobId}
# # #   and returns True when the backend reports the job is already COMPLETED.
# # #   Called by consumer.py as an idempotency guard before starting any processing
# # #   on a redelivered message.
# # # """

# # # from __future__ import annotations
# # # import logging
# # # import os
# # # from typing import Any, Dict, Optional

# # # import requests
# # # from dotenv import load_dotenv

# # # # ── Credentials / config ─────────────────────────────────────────────────────
# # # _ENV_PATH = os.path.join(
# # #     os.path.dirname(os.path.abspath(__file__)),
# # #     "config", "credentials.env",
# # # )
# # # load_dotenv(_ENV_PATH)

# # # _BASE_URL = os.environ.get(
# # #     "SPRING_BOOT_BASE_URL",
# # #     "https://cvvrsrailway-api.sconexsoft.com/cvs",
# # # )
# # # _TIMEOUT = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # log = logging.getLogger("callback_client")


# # # # ── Dynamic base URL setter ───────────────────────────────────────────────────

# # # def set_base_url(url: str) -> None:
# # #     """
# # #     Override the callback base URL at runtime.

# # #     Called by consumer.py when the RabbitMQ message contains a callbackBaseUrl.
# # #     Allows the same Python worker to callback correctly to both local and staging
# # #     Spring Boot servers without changing env vars.

# # #     The URL passed here is the FULL path up to /api/internal/analysis.
# # #     Individual callbacks append /progress, /completed, /failed, /status/{id}.
# # #     """
# # #     global _BASE_URL
# # #     base = url.rstrip("/")
# # #     if base.endswith("/api/internal/analysis"):
# # #         base = base[: -len("/api/internal/analysis")]
# # #     _BASE_URL = base
# # #     log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)


# # # # ── Internal helpers ──────────────────────────────────────────────────────────

# # # def _post(path: str, payload: Dict[str, Any]) -> None:
# # #     """
# # #     POST to a Spring Boot internal endpoint.
# # #     No Authorization header — /api/internal/* are worker-only endpoints.
# # #     Raises RuntimeError on non-2xx so the consumer can route to the failure path.
# # #     """
# # #     url = f"{_BASE_URL}{path}"
# # #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# # #     if not resp.ok:
# # #         raise RuntimeError(
# # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # #         )
# # #     log.debug("[Callback] %s → %d", path, resp.status_code)


# # # def _get(path: str) -> requests.Response:
# # #     """
# # #     GET from a Spring Boot internal endpoint.
# # #     Raises RuntimeError on non-2xx.
# # #     """
# # #     url = f"{_BASE_URL}{path}"
# # #     log.debug("[Callback] GET %s", url)
# # #     resp = requests.get(url, timeout=_TIMEOUT)
# # #     if not resp.ok:
# # #         raise RuntimeError(
# # #             f"Callback GET {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # #         )
# # #     log.debug("[Callback] %s → %d", path, resp.status_code)
# # #     return resp


# # # # ── Public API ────────────────────────────────────────────────────────────────

# # # def check_job_completed(job_id: str) -> bool:
# # #     """
# # #     NEW — Idempotency check.

# # #     Queries GET /api/internal/analysis/status/{jobId} and returns True when
# # #     the backend reports the job status as COMPLETED.

# # #     Called by consumer.py at the very start of _handle_job() so that
# # #     RabbitMQ redeliveries of already-completed jobs are detected immediately
# # #     and ACKed without re-running any processing.

# # #     Backend contract (expected JSON shape):
# # #         { "status": "COMPLETED" }   → job already done → return True
# # #         { "status": "PROCESSING" }  → in-flight (rare edge-case) → return False
# # #         { "status": "PENDING" }     → not yet processed → return False
# # #         404 Not Found               → job unknown (treat as not completed)

# # #     If the endpoint does not exist yet on your Spring Boot side, add it as:
# # #         GET /api/internal/analysis/status/{jobId}
# # #         → ResponseEntity<Map<String,String>>  { "status": job.getStatus().name() }

# # #     Raises on network errors so the consumer can decide whether to proceed
# # #     with processing or skip (consumer.py catches and proceeds on error).
# # #     """
# # #     try:
# # #         resp = _get(f"/api/internal/analysis/status/{job_id}")
# # #         data = resp.json()
# # #         status = data.get("status", "").upper()
# # #         is_done = status == "COMPLETED"
# # #         log.info(
# # #             "[Callback] check_job_completed  job=%s  status=%s  is_done=%s",
# # #             job_id, status, is_done,
# # #         )
# # #         return is_done
# # #     except RuntimeError as exc:
# # #         # 404 → job not found in the backend → definitely not completed
# # #         if "404" in str(exc):
# # #             log.info(
# # #                 "[Callback] check_job_completed  job=%s  → 404 (not found) → not completed",
# # #                 job_id,
# # #             )
# # #             return False
# # #         raise


# # # def send_progress(
# # #     job_id:        str,
# # #     journey_id:    int,
# # #     progress:      int,
# # #     message:       str,
# # #     status:        str = "PROCESSING",
# # #     current_video: Optional[int] = None,
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
# # #     current_video : 1-based index of the video currently being processed.
# # #     """
# # #     payload: Dict[str, Any] = {
# # #         "jobId":     job_id,
# # #         "journeyId": journey_id,
# # #         "status":    status,
# # #         "progress":  progress,
# # #         "message":   message,
# # #     }
# # #     if current_video is not None:
# # #         payload["currentVideo"] = current_video
# # #     _post("/api/internal/analysis/progress", payload)


# # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # #     """
# # #     POST /api/internal/analysis/completed

# # #     Called once after ALL videos in the journey have been processed and their
# # #     violation frames uploaded to S3.

# # #     completion_payload is built by CompletionPayload.to_dict() in models.py.
# # #     """
# # #     _post("/api/internal/analysis/completed", completion_payload)


# # # def send_failed(
# # #     job_id:        str,
# # #     journey_id:    int,   # kept for caller convenience / logging
# # #     error_message: str,
# # # ) -> None:
# # #     """
# # #     POST /api/internal/analysis/failed

# # #     Called whenever an unrecoverable exception occurs during job processing.
# # #     Spring Boot will mark AnalysisJob and Journey as FAILED.

# # #     Note: The API spec only requires jobId + errorMessage.
# # #     journeyId is accepted as a parameter for logging but is NOT included
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

# # Changes in this version
# # ────────────────────────
# # • check_job_completed() now uses the EXISTING completed-callback endpoint
# #   as a probe instead of a dedicated /status/{jobId} endpoint that doesn't
# #   exist yet on Spring Boot.

# #   Strategy: POST /api/internal/analysis/completed with only { "jobId": id,
# #   "probe": true }.  Spring Boot should return 409 CONFLICT (or any non-500)
# #   when the job is already done.  If it returns 500 we treat it as "unknown"
# #   and fall through to processing (safe default).

# #   *** TEMPORARY until the Spring Boot /status endpoint is implemented. ***
# #   Once GET /api/internal/analysis/status/{jobId} is live, revert
# #   check_job_completed() to use _get() as originally written.
# # """

# # from __future__ import annotations

# # import logging
# # import os
# # from typing import Any, Dict, Optional

# # import requests
# # from dotenv import load_dotenv

# # # ── Credentials / config ─────────────────────────────────────────────────────
# # _ENV_PATH = os.path.join(
# #     os.path.dirname(os.path.abspath(__file__)),
# #     "config", "credentials.env",
# # )
# # load_dotenv(_ENV_PATH)

# # _BASE_URL = os.environ.get(
# #     "SPRING_BOOT_BASE_URL",
# #     "https://cvvrsrailway-api.sconexsoft.com/cvs",
# # )
# # _TIMEOUT = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # log = logging.getLogger("callback_client")


# # # ── Dynamic base URL setter ───────────────────────────────────────────────────

# # def set_base_url(url: str) -> None:
# #     global _BASE_URL
# #     base = url.rstrip("/")
# #     if base.endswith("/api/internal/analysis"):
# #         base = base[: -len("/api/internal/analysis")]
# #     _BASE_URL = base
# #     log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)


# # # ── Internal helpers ──────────────────────────────────────────────────────────

# # def _post(path: str, payload: Dict[str, Any]) -> requests.Response:
# #     url = f"{_BASE_URL}{path}"
# #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# #     if not resp.ok:
# #         raise RuntimeError(
# #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# #         )
# #     log.debug("[Callback] %s → %d", path, resp.status_code)
# #     return resp


# # def _get(path: str) -> requests.Response:
# #     url = f"{_BASE_URL}{path}"
# #     log.debug("[Callback] GET %s", url)
# #     resp = requests.get(url, timeout=_TIMEOUT)
# #     if not resp.ok:
# #         raise RuntimeError(
# #             f"Callback GET {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# #         )
# #     log.debug("[Callback] %s → %d", path, resp.status_code)
# #     return resp


# # # ── Public API ────────────────────────────────────────────────────────────────

# # def check_job_completed(job_id: str) -> bool:
# #     """
# #     Idempotency check — returns True if the backend already has this job as COMPLETED.

# #     TWO-STAGE STRATEGY
# #     ──────────────────
# #     Stage 1 (permanent): GET /api/internal/analysis/status/{jobId}
# #         → { "status": "COMPLETED" }  → True
# #         → { "status": "PENDING/PROCESSING" } → False
# #         → 404 → False  (job not known yet)
# #         → 500 → fall through to Stage 2

# #     Stage 2 (temporary fallback until Spring Boot implements /status):
# #         Uses GET /api/internal/analysis/job/{jobId} or any existing
# #         read endpoint.  If that also 500s, we return False (safe default:
# #         process the job rather than silently drop it).

# #     WHAT TO IMPLEMENT ON SPRING BOOT (one small controller method):
# #     ───────────────────────────────────────────────────────────────
# #     @GetMapping("/api/internal/analysis/status/{jobId}")
# #     public ResponseEntity<Map<String,String>> getJobStatus(
# #             @PathVariable String jobId) {
# #         return analysisJobRepository.findByJobId(jobId)
# #             .map(job -> ResponseEntity.ok(Map.of("status", job.getStatus().name())))
# #             .orElse(ResponseEntity.notFound().<Map<String,String>>build());
# #     }

# #     Once that endpoint is deployed, Stage 2 below can be deleted.
# #     """

# #     # ── Stage 1: dedicated status endpoint ───────────────────────────────────
# #     try:
# #         resp = _get(f"/api/internal/analysis/status/{job_id}")
# #         status = resp.json().get("status", "").upper()
# #         is_done = status == "COMPLETED"
# #         log.info(
# #             "[Callback] idempotency check (status endpoint)  job=%s  status=%s  done=%s",
# #             job_id, status, is_done,
# #         )
# #         return is_done
# #     except RuntimeError as exc:
# #         if "404" in str(exc):
# #             log.info(
# #                 "[Callback] idempotency check  job=%s  → 404 → not completed", job_id
# #             )
# #             return False
# #         if "500" in str(exc):
# #             log.warning(
# #                 "[Callback] idempotency check  job=%s  → /status endpoint returned 500 "
# #                 "(not implemented yet?) — trying fallback probe.", job_id
# #             )
# #             # fall through to Stage 2
# #         else:
# #             # Network error or unexpected status — safe default: process the job
# #             log.warning(
# #                 "[Callback] idempotency check  job=%s  → unexpected error (%s) "
# #                 "— proceeding with processing.", job_id, exc
# #             )
# #             return False

# #     # ── Stage 2: fallback probe using progress endpoint ───────────────────────
# #     # We send a lightweight progress probe at 0 % with status=CHECK.
# #     # Spring Boot should:
# #     #   • Return 200 if the job is still PENDING/PROCESSING (→ process it).
# #     #   • Return 409 CONFLICT if the job is already COMPLETED (→ skip it).
# #     #   • Return 404 if the job is unknown (→ process it).
# #     # If the backend doesn't handle the CHECK status specially it will just
# #     # update progress to 0 — harmless on an already-completed job.
# #     #
# #     # THIS IS A TEMPORARY FALLBACK.  Remove once /status endpoint is live.
# #     try:
# #         url = f"{_BASE_URL}/api/internal/analysis/progress"
# #         resp = requests.post(
# #             url,
# #             json={
# #                 "jobId":    job_id,
# #                 "status":   "CHECK",   # sentinel value Spring Boot can detect
# #                 "progress": 0,
# #                 "message":  "idempotency-probe",
# #             },
# #             timeout=_TIMEOUT,
# #         )
# #         if resp.status_code == 409:
# #             log.warning(
# #                 "[Callback] idempotency check (probe)  job=%s  → 409 CONFLICT "
# #                 "→ already COMPLETED — will skip.", job_id
# #             )
# #             return True
# #         log.info(
# #             "[Callback] idempotency check (probe)  job=%s  → HTTP %d → not completed.",
# #             job_id, resp.status_code,
# #         )
# #         return False
# #     except Exception as exc2:
# #         log.warning(
# #             "[Callback] idempotency check (probe) failed  job=%s: %s "
# #             "— proceeding with processing.", job_id, exc2
# #         )
# #         return False


# # def send_progress(
# #     job_id:        str,
# #     journey_id:    int,
# #     progress:      int,
# #     message:       str,
# #     status:        str = "PROCESSING",
# #     current_video: Optional[int] = None,
# # ) -> None:
# #     payload: Dict[str, Any] = {
# #         "jobId":     job_id,
# #         "journeyId": journey_id,
# #         "status":    status,
# #         "progress":  progress,
# #         "message":   message,
# #     }
# #     if current_video is not None:
# #         payload["currentVideo"] = current_video
# #     _post("/api/internal/analysis/progress", payload)


# # def send_completed(completion_payload: Dict[str, Any]) -> None:
# #     _post("/api/internal/analysis/completed", completion_payload)


# # def send_failed(
# #     job_id:        str,
# #     journey_id:    int,
# #     error_message: str,
# #     video_id:      Optional[int] = None,
# #     error_type:    Optional[str] = None,
# #     stack_trace:   Optional[str] = None,
# #     reason:        Optional[str] = None,
# # ) -> None:
# #     """
# #     Calls POST /api/internal/analysis/failed.

# #     Backward compatible: existing job-level callers (no video_id) keep
# #     working exactly as before — the payload shape for that case is
# #     unchanged ({"jobId", "errorMessage"}).

# #     When video_id IS supplied, this becomes a PER-VIDEO failure report
# #     (Phase 1 requirement: "no failed video should wait until journey
# #     completion to be reported"). The payload then also carries:
# #         videoId     — STRING, matches VideoResult.to_dict()'s convention
# #         journeyId
# #         errorType   — short classifier, e.g. "OOM", "DECODE_ERROR",
# #                       "MEDIAPIPE_ERROR", "YOLO_ERROR", "TIMEOUT",
# #                       "RESOURCE_EXHAUSTION", "NOT_PROCESSED"
# #         stackTrace  — full traceback text, if available
# #         reason      — human-readable reason string, e.g.
# #                       "Not Processed - Worker Resource Exhaustion"
# #                       for videos skipped after an OOM on an earlier video.
# #     """
# #     payload: Dict[str, Any] = {
# #         "jobId":        job_id,
# #         "errorMessage": error_message,
# #     }
# #     if video_id is not None:
# #         payload["videoId"]    = str(video_id)   # STRING per API spec convention
# #         payload["journeyId"]  = journey_id
# #     if error_type is not None:
# #         payload["errorType"] = error_type
# #     if stack_trace is not None:
# #         payload["stackTrace"] = stack_trace
# #     if reason is not None:
# #         payload["reason"] = reason

# #     try:
# #         _post("/api/internal/analysis/failed", payload)
# #     except Exception as exc:
# #         log.error(
# #             "[Callback] send_failed itself failed (job=%s journey=%d video=%s): %s",
# #             job_id, journey_id, video_id, exc,
# #         )


# # def send_video_failed(
# #     job_id:        str,
# #     journey_id:    int,
# #     video_id:      int,
# #     error_type:    str,
# #     error_message: str,
# #     stack_trace:   str = "",
# #     reason:        Optional[str] = None,
# # ) -> None:
# #     """
# #     Convenience wrapper for the per-video failure report required by
# #     Phase 1: "Immediately call the failed endpoint for the affected video"
# #     — called once per video, the moment that video's outcome is known,
# #     not batched until the journey ends.

# #     For resource-exhaustion cascades, pass error_type="RESOURCE_EXHAUSTION"
# #     (the video that actually hit OOM) or error_type="NOT_PROCESSED" with
# #     reason="Not Processed - Worker Resource Exhaustion" (videos after it
# #     that were skipped as a result).
# #     """
# #     log.error(
# #         "[Callback] video FAILED  job=%s  journey=%d  video=%d  type=%s  reason=%s: %s",
# #         job_id, journey_id, video_id, error_type, reason, error_message,
# #     )
# #     send_failed(
# #         job_id        = job_id,
# #         journey_id    = journey_id,
# #         error_message = error_message,
# #         video_id      = video_id,
# #         error_type    = error_type,
# #         stack_trace   = stack_trace,
# #         reason        = reason,
# #     )


# # # ── OOM / resource-exhaustion signature detection ────────────────────────────
# # #
# # # Used by analyzer.py to decide whether a per-video exception means "this
# # # one video is bad" (continue with the rest) vs. "the worker itself can no
# # # longer safely process more video" (stop the journey, mark every
# # # remaining video FAILED with reason "Not Processed - Worker Resource
# # # Exhaustion"). Matches the signatures called out explicitly in the
# # # Phase 1 spec: OpenCV OutOfMemoryError, "Failed to allocate", cv2.pyd
# # # memory-related failures, generic MemoryError, std::bad_alloc.
# # _OOM_SIGNATURES = (
# #     "outofmemoryerror",
# #     "out of memory",
# #     "failed to allocate",
# #     "cannot allocate memory",
# #     "bad_alloc",
# #     "memoryerror",
# #     "cv2.pyd",
# #     "access violation",
# #     "resource exhaust",
# # )


# # def is_resource_exhaustion_error(exc: BaseException) -> bool:
# #     """
# #     Returns True if `exc` looks like an OOM / native resource-exhaustion
# #     failure rather than an ordinary recoverable per-video error.
# #     """
# #     text = f"{type(exc).__name__}: {exc}".lower()
# #     if isinstance(exc, MemoryError):
# #         return True
# #     return any(sig in text for sig in _OOM_SIGNATURES)


# # def classify_video_error(exc: BaseException) -> str:
# #     """
# #     Maps a caught per-video exception to a short errorType string for the
# #     failed-endpoint payload. Best-effort classification by exception type
# #     and message content — defaults to "PROCESSING_ERROR" when nothing
# #     more specific matches.
# #     """
# #     if is_resource_exhaustion_error(exc):
# #         return "RESOURCE_EXHAUSTION"
# #     name = type(exc).__name__.lower()
# #     text = str(exc).lower()
# #     if "timeout" in name or "timeout" in text:
# #         return "TIMEOUT"
# #     if "mediapipe" in text:
# #         return "MEDIAPIPE_ERROR"
# #     if "yolo" in text or "ultralytics" in text:
# #         return "YOLO_ERROR"
# #     if "cv2" in text or "opencv" in text or name == "error":
# #         return "OPENCV_ERROR"
# #     if "cannot open" in text or "corrupt" in text or "decode" in text or "invalid" in text:
# #         return "DECODE_ERROR"
# #     return "PROCESSING_ERROR"


# # def compute_journey_status(
# #     total_videos: int,
# #     succeeded_video_ids: "set[int]",
# #     failed_video_ids: "set[int]",
# # ) -> str:
# #     """
# #     Computes the journey-level terminal status per Phase 1 spec:
# #         COMPLETED              — every video succeeded
# #         COMPLETED_WITH_ERRORS  — at least one succeeded AND at least one failed
# #         FAILED                 — every video failed (or none succeeded)

# #     TIMED_OUT is NOT computed here — that's set explicitly by the watchdog
# #     path in consumer.py, which knows it hit a timeout rather than inferring
# #     it from video counts.
# #     """
# #     if total_videos == 0:
# #         return "FAILED"
# #     if not failed_video_ids:
# #         return "COMPLETED"
# #     if succeeded_video_ids:
# #         return "COMPLETED_WITH_ERRORS"
# #     return "FAILED"


# # # # # # """
# # # # # # callback_client.py
# # # # # # ──────────────────
# # # # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # # # # All three public helpers are fire-and-forget from the caller's perspective;
# # # # # # they raise on non-2xx so the consumer can route to the failure path.
# # # # # # """

# # # # # # from __future__ import annotations

# # # # # # import logging
# # # # # # import os
# # # # # # from typing import Any, Dict, List

# # # # # # import requests
# # # # # # from dotenv import load_dotenv

# # # # # # # ── Credentials / config ────────────────────────────────────────────────────
# # # # # # _ENV_PATH = os.path.join(
# # # # # #     os.path.dirname(os.path.abspath(__file__)),
# # # # # #     "config", "credentials.env",
# # # # # # )
# # # # # # load_dotenv(_ENV_PATH)

# # # # # # _BASE_URL = os.environ.get("SPRING_BOOT_BASE_URL", "http://localhost:8080")
# # # # # # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # # # # log = logging.getLogger("callback_client")


# # # # # # # ── Internal helper ──────────────────────────────────────────────────────────

# # # # # # def _post(path: str, payload: Dict[str, Any]) -> None:
# # # # # #     url = f"{_BASE_URL}{path}"
# # # # # #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# # # # # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# # # # # #     if not resp.ok:
# # # # # #         raise RuntimeError(
# # # # # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # # # # #         )
# # # # # #     log.debug("[Callback] %s → %d", path, resp.status_code)


# # # # # # # ── Public API ───────────────────────────────────────────────────────────────

# # # # # # def send_progress(
# # # # # #     job_id:     str,
# # # # # #     journey_id: int,
# # # # # #     progress:   int,
# # # # # #     message:    str,
# # # # # #     status:     str = "PROCESSING",
# # # # # # ) -> None:
# # # # # #     """
# # # # # #     POST /api/internal/analysis/progress

# # # # # #     Called periodically during analysis to update the frontend progress bar
# # # # # #     and the SSE stream.
# # # # # #     """
# # # # # #     _post(
# # # # # #         "/api/internal/analysis/progress",
# # # # # #         {
# # # # # #             "jobId":      job_id,
# # # # # #             "journeyId":  journey_id,
# # # # # #             "status":     status,
# # # # # #             "progress":   progress,
# # # # # #             "message":    message,
# # # # # #         },
# # # # # #     )


# # # # # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # # # # #     """
# # # # # #     POST /api/internal/analysis/completed

# # # # # #     Called once — after ALL videos in the journey have been processed and
# # # # # #     their violation frames uploaded to S3.

# # # # # #     Expected shape of completion_payload
# # # # # #     ─────────────────────────────────────
# # # # # #     {
# # # # # #         "jobId":          str,
# # # # # #         "journeyId":      int,
# # # # # #         "processingTime": int,          # wall-clock ms
# # # # # #         "videoResults":   List[dict],   # see models.py → VideoResult
# # # # # #     }
# # # # # #     """
# # # # # #     _post("/api/internal/analysis/completed", completion_payload)


# # # # # # def send_failed(
# # # # # #     job_id:        str,
# # # # # #     journey_id:    int,
# # # # # #     error_message: str,
# # # # # # ) -> None:
# # # # # #     """
# # # # # #     POST /api/internal/analysis/failed

# # # # # #     Called whenever an unrecoverable exception occurs during job processing.
# # # # # #     Spring Boot will mark AnalysisJob and Journey as FAILED.
# # # # # #     """
# # # # # #     try:
# # # # # #         _post(
# # # # # #             "/api/internal/analysis/failed",
# # # # # #             {
# # # # # #                 "jobId":        job_id,
# # # # # #                 "journeyId":    journey_id,
# # # # # #                 "errorMessage": error_message,
# # # # # #             },
# # # # # #         )
# # # # # #     except Exception as exc:
# # # # # #         # Failure callback must never itself raise — log and swallow.
# # # # # #         log.error("[Callback] send_failed itself failed: %s", exc)

# # # # # """
# # # # # callback_client.py
# # # # # ──────────────────
# # # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # # # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs
# # # # # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).

# # # # # Changes from previous version
# # # # # ──────────────────────────────
# # # # # • _BASE_URL now defaults to the live CVVRS domain (overridable via env var
# # # # #   SPRING_BOOT_BASE_URL for local testing).
# # # # # • send_progress()  — payload now includes the `currentVideo` field required
# # # # #   by the API.
# # # # # • send_completed() — accepts a fully-formed dict (built by CompletionPayload.
# # # # #   to_dict()); no shape changes needed here since the caller owns the dict.
# # # # # • send_failed()    — payload drops `journeyId` (not in the API spec for this
# # # # #   endpoint); only jobId + errorMessage are sent.
# # # # # """

# # # # # from __future__ import annotations

# # # # # import logging
# # # # # import os
# # # # # from typing import Any, Dict, Optional

# # # # # import requests
# # # # # from dotenv import load_dotenv

# # # # # # ── Credentials / config ────────────────────────────────────────────────────
# # # # # _ENV_PATH = os.path.join(
# # # # #     os.path.dirname(os.path.abspath(__file__)),
# # # # #     "config", "credentials.env",
# # # # # )
# # # # # load_dotenv(_ENV_PATH)

# # # # # # Default to the live CVVRS server; override with SPRING_BOOT_BASE_URL for local dev.
# # # # # _BASE_URL = os.environ.get(
# # # # #     "SPRING_BOOT_BASE_URL",
# # # # #     "https://cvvrsrailway-api.sconexsoft.com/cvs",
# # # # # )
# # # # # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # # # log = logging.getLogger("callback_client")


# # # # # # ── Internal helper ──────────────────────────────────────────────────────────

# # # # # def _post(path: str, payload: Dict[str, Any]) -> None:
# # # # #     """
# # # # #     POST to a Spring Boot internal endpoint.
# # # # #     No Authorization header is sent — /api/internal/* are worker-only endpoints.
# # # # #     Raises RuntimeError on non-2xx so the consumer can route to failure path.
# # # # #     """
# # # # #     url = f"{_BASE_URL}{path}"
# # # # #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# # # # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# # # # #     if not resp.ok:
# # # # #         raise RuntimeError(
# # # # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # # # #         )
# # # # #     log.debug("[Callback] %s → %d", path, resp.status_code)


# # # # # # ── Public API ───────────────────────────────────────────────────────────────

# # # # # def send_progress(
# # # # #     job_id:        str,
# # # # #     journey_id:    int,
# # # # #     progress:      int,
# # # # #     message:       str,
# # # # #     status:        str = "PROCESSING",
# # # # #     current_video: Optional[int] = None,   # NEW — index of video currently being processed
# # # # # ) -> None:
# # # # #     """
# # # # #     POST /api/internal/analysis/progress

# # # # #     Called periodically during analysis to update the frontend progress bar
# # # # #     and the SSE stream.

# # # # #     Parameters
# # # # #     ──────────
# # # # #     job_id        : RabbitMQ job ID.
# # # # #     journey_id    : Journey ID.
# # # # #     progress      : 0–100 integer.
# # # # #     message       : Human-readable status message.
# # # # #     status        : "PROCESSING" | "COMPLETED" | "FAILED"
# # # # #     current_video : 1-based index of the video currently being processed
# # # # #                     (omitted from payload when None).
# # # # #     """
# # # # #     payload: Dict[str, Any] = {
# # # # #         "jobId":      job_id,
# # # # #         "journeyId":  journey_id,
# # # # #         "status":     status,
# # # # #         "progress":   progress,
# # # # #         "message":    message,
# # # # #     }
# # # # #     if current_video is not None:
# # # # #         payload["currentVideo"] = current_video

# # # # #     _post("/api/internal/analysis/progress", payload)


# # # # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # # # #     """
# # # # #     POST /api/internal/analysis/completed

# # # # #     Called once after ALL videos in the journey have been processed and their
# # # # #     violation frames uploaded to S3.

# # # # #     Expected shape of completion_payload (built by CompletionPayload.to_dict())
# # # # #     ────────────────────────────────────────────────────────────────────────────
# # # # #     {
# # # # #         "jobId":         str,
# # # # #         "journeyId":     int,
# # # # #         "batchId":       str,           # "BATCH-<jobId>" if not supplied
# # # # #         "trainDetailId": int,
# # # # #         "folderName":    str,
# # # # #         "processingTime":int,           # wall-clock ms
# # # # #         "videoResults": [
# # # # #             {
# # # # #                 "videoId":         str,       # STRING per API spec
# # # # #                 "sequenceNo":      int,
# # # # #                 "durationSeconds": float,
# # # # #                 "originalS3Key":   str,
# # # # #                 "violations": [
# # # # #                     {
# # # # #                         "violationType":          str,
# # # # #                         "severity":               str,
# # # # #                         "confidence":             float,
# # # # #                         "riskScore":              float,
# # # # #                         "timestamp":              float,   # journey-global seconds
# # # # #                         "originalVideoTimestamp": float,   # local-video seconds
# # # # #                         "framePaths":             [str]
# # # # #                     }
# # # # #                 ]
# # # # #             }
# # # # #         ]
# # # # #     }
# # # # #     """
# # # # #     _post("/api/internal/analysis/completed", completion_payload)


# # # # # def send_failed(
# # # # #     job_id:        str,
# # # # #     journey_id:    int,          # kept as parameter for caller convenience / logging
# # # # #     error_message: str,
# # # # # ) -> None:
# # # # #     """
# # # # #     POST /api/internal/analysis/failed

# # # # #     Called whenever an unrecoverable exception occurs during job processing.
# # # # #     Spring Boot will mark AnalysisJob and Journey as FAILED.

# # # # #     Note: The API spec only requires jobId + errorMessage in the request body.
# # # # #     journeyId is accepted as a parameter here for logging but is NOT included
# # # # #     in the outbound payload.
# # # # #     """
# # # # #     try:
# # # # #         _post(
# # # # #             "/api/internal/analysis/failed",
# # # # #             {
# # # # #                 "jobId":        job_id,
# # # # #                 "errorMessage": error_message,
# # # # #             },
# # # # #         )
# # # # #     except Exception as exc:
# # # # #         # Failure callback must never itself raise — log and swallow.
# # # # #         log.error(
# # # # #             "[Callback] send_failed itself failed (job=%s journey=%d): %s",
# # # # #             job_id, journey_id, exc,
# # # # #         )



# # # # """

# # # # callback_client.py

# # # # ──────────────────

# # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.
 
# # # # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs

# # # # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).
 
# # # # Fixes from previous version

# # # # ─────────────────────────────

# # # # • Added set_base_url() — called by consumer.py with the callbackBaseUrl from

# # # #   the RabbitMQ message, so Python posts to the correct Spring Boot server

# # # #   per environment without needing env var changes.

# # # # • _BASE_URL still falls back to env var SPRING_BOOT_BASE_URL or the live

# # # #   server when callbackBaseUrl is not provided in the message.

# # # # """
 
# # # # from __future__ import annotations
 
# # # # import logging

# # # # import os

# # # # from typing import Any, Dict, Optional
 
# # # # import requests

# # # # from dotenv import load_dotenv
 
# # # # # ── Credentials / config ────────────────────────────────────────────────────

# # # # _ENV_PATH = os.path.join(

# # # #     os.path.dirname(os.path.abspath(__file__)),

# # # #     "config", "credentials.env",

# # # # )

# # # # load_dotenv(_ENV_PATH)
 
# # # # # Default to the live CVVRS server; override with SPRING_BOOT_BASE_URL for local dev

# # # # # or call set_base_url() with the callbackBaseUrl from the RabbitMQ message.

# # # # _BASE_URL = os.environ.get(

# # # #     "SPRING_BOOT_BASE_URL",

# # # #     "https://cvvrsrailway-api.sconexsoft.com/cvs",

# # # # )

# # # # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))
 
# # # # log = logging.getLogger("callback_client")
 
 
# # # # # ── FIX: dynamic base URL setter ─────────────────────────────────────────────
 
# # # # def set_base_url(url: str) -> None:

# # # #     """

# # # #     Override the callback base URL at runtime.
 
# # # #     Called by consumer.py when the RabbitMQ message contains a callbackBaseUrl.

# # # #     This allows the same Python worker to callback correctly to both local

# # # #     and staging Spring Boot servers without changing env vars.
 
# # # #     Example values:

# # # #         "http://localhost:8093/api/internal/analysis"         (local)

# # # #         "https://cvvrsrailway-api.sconexsoft.com/cvs/api/internal/analysis"  (staging)
 
# # # #     Note: The URL passed here is the FULL path up to /api/internal/analysis.

# # # #     The individual callbacks append /progress, /completed, /failed.

# # # #     """

# # # #     global _BASE_URL

# # # #     # Strip the /api/internal/analysis suffix if present — we add it in _post()

# # # #     base = url.rstrip("/")

# # # #     if base.endswith("/api/internal/analysis"):

# # # #         base = base[: -len("/api/internal/analysis")]

# # # #     _BASE_URL = base

# # # #     log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)
 
 
# # # # # ── Internal helper ──────────────────────────────────────────────────────────
 
# # # # def _post(path: str, payload: Dict[str, Any]) -> None:

# # # #     """

# # # #     POST to a Spring Boot internal endpoint.

# # # #     No Authorization header is sent — /api/internal/* are worker-only endpoints.

# # # #     Raises RuntimeError on non-2xx so the consumer can route to failure path.

# # # #     """

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

# # # #     job_id:        str,

# # # #     journey_id:    int,

# # # #     progress:      int,

# # # #     message:       str,

# # # #     status:        str = "PROCESSING",

# # # #     current_video: Optional[int] = None,

# # # # ) -> None:

# # # #     """

# # # #     POST /api/internal/analysis/progress
 
# # # #     Called periodically during analysis to update the frontend progress bar

# # # #     and the SSE stream.
 
# # # #     Parameters

# # # #     ──────────

# # # #     job_id        : RabbitMQ job ID.

# # # #     journey_id    : Journey ID.

# # # #     progress      : 0–100 integer.

# # # #     message       : Human-readable status message.

# # # #     status        : "PROCESSING" | "COMPLETED" | "FAILED"

# # # #     current_video : 1-based index of the video currently being processed.

# # # #     """

# # # #     payload: Dict[str, Any] = {

# # # #         "jobId":      job_id,

# # # #         "journeyId":  journey_id,

# # # #         "status":     status,

# # # #         "progress":   progress,

# # # #         "message":    message,

# # # #     }

# # # #     if current_video is not None:

# # # #         payload["currentVideo"] = current_video
 
# # # #     _post("/api/internal/analysis/progress", payload)
 
 
# # # # def send_completed(completion_payload: Dict[str, Any]) -> None:

# # # #     """

# # # #     POST /api/internal/analysis/completed
 
# # # #     Called once after ALL videos in the journey have been processed and their

# # # #     violation frames uploaded to S3.
 
# # # #     completion_payload is built by CompletionPayload.to_dict() in models.py.

# # # #     """

# # # #     _post("/api/internal/analysis/completed", completion_payload)
 
 
# # # # def send_failed(

# # # #     job_id:        str,

# # # #     journey_id:    int,          # kept for caller convenience / logging

# # # #     error_message: str,

# # # # ) -> None:

# # # #     """

# # # #     POST /api/internal/analysis/failed
 
# # # #     Called whenever an unrecoverable exception occurs during job processing.

# # # #     Spring Boot will mark AnalysisJob and Journey as FAILED.
 
# # # #     Note: The API spec only requires jobId + errorMessage.

# # # #     journeyId is accepted as a parameter for logging but is NOT included

# # # #     in the outbound payload.

# # # #     """

# # # #     try:

# # # #         _post(

# # # #             "/api/internal/analysis/failed",

# # # #             {

# # # #                 "jobId":        job_id,

# # # #                 "errorMessage": error_message,

# # # #             },

# # # #         )

# # # #     except Exception as exc:

# # # #         # Failure callback must never itself raise — log and swallow.

# # # #         log.error(

# # # #             "[Callback] send_failed itself failed (job=%s journey=%d): %s",

# # # #             job_id, journey_id, exc,

# # # #         )
 
 
 
# # # """
# # # callback_client.py
# # # ──────────────────
# # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs
# # # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).

# # # Changes from previous version
# # # ──────────────────────────────
# # # • Added check_job_completed(job_id) — queries GET /api/internal/analysis/status/{jobId}
# # #   and returns True when the backend reports the job is already COMPLETED.
# # #   Called by consumer.py as an idempotency guard before starting any processing
# # #   on a redelivered message.
# # # """

# # # from __future__ import annotations
# # # import logging
# # # import os
# # # from typing import Any, Dict, Optional

# # # import requests
# # # from dotenv import load_dotenv

# # # # ── Credentials / config ─────────────────────────────────────────────────────
# # # _ENV_PATH = os.path.join(
# # #     os.path.dirname(os.path.abspath(__file__)),
# # #     "config", "credentials.env",
# # # )
# # # load_dotenv(_ENV_PATH)

# # # _BASE_URL = os.environ.get(
# # #     "SPRING_BOOT_BASE_URL",
# # #     "https://cvvrsrailway-api.sconexsoft.com/cvs",
# # # )
# # # _TIMEOUT = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # log = logging.getLogger("callback_client")


# # # # ── Dynamic base URL setter ───────────────────────────────────────────────────

# # # def set_base_url(url: str) -> None:
# # #     """
# # #     Override the callback base URL at runtime.

# # #     Called by consumer.py when the RabbitMQ message contains a callbackBaseUrl.
# # #     Allows the same Python worker to callback correctly to both local and staging
# # #     Spring Boot servers without changing env vars.

# # #     The URL passed here is the FULL path up to /api/internal/analysis.
# # #     Individual callbacks append /progress, /completed, /failed, /status/{id}.
# # #     """
# # #     global _BASE_URL
# # #     base = url.rstrip("/")
# # #     if base.endswith("/api/internal/analysis"):
# # #         base = base[: -len("/api/internal/analysis")]
# # #     _BASE_URL = base
# # #     log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)


# # # # ── Internal helpers ──────────────────────────────────────────────────────────

# # # def _post(path: str, payload: Dict[str, Any]) -> None:
# # #     """
# # #     POST to a Spring Boot internal endpoint.
# # #     No Authorization header — /api/internal/* are worker-only endpoints.
# # #     Raises RuntimeError on non-2xx so the consumer can route to the failure path.
# # #     """
# # #     url = f"{_BASE_URL}{path}"
# # #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# # #     if not resp.ok:
# # #         raise RuntimeError(
# # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # #         )
# # #     log.debug("[Callback] %s → %d", path, resp.status_code)


# # # def _get(path: str) -> requests.Response:
# # #     """
# # #     GET from a Spring Boot internal endpoint.
# # #     Raises RuntimeError on non-2xx.
# # #     """
# # #     url = f"{_BASE_URL}{path}"
# # #     log.debug("[Callback] GET %s", url)
# # #     resp = requests.get(url, timeout=_TIMEOUT)
# # #     if not resp.ok:
# # #         raise RuntimeError(
# # #             f"Callback GET {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # #         )
# # #     log.debug("[Callback] %s → %d", path, resp.status_code)
# # #     return resp


# # # # ── Public API ────────────────────────────────────────────────────────────────

# # # def check_job_completed(job_id: str) -> bool:
# # #     """
# # #     NEW — Idempotency check.

# # #     Queries GET /api/internal/analysis/status/{jobId} and returns True when
# # #     the backend reports the job status as COMPLETED.

# # #     Called by consumer.py at the very start of _handle_job() so that
# # #     RabbitMQ redeliveries of already-completed jobs are detected immediately
# # #     and ACKed without re-running any processing.

# # #     Backend contract (expected JSON shape):
# # #         { "status": "COMPLETED" }   → job already done → return True
# # #         { "status": "PROCESSING" }  → in-flight (rare edge-case) → return False
# # #         { "status": "PENDING" }     → not yet processed → return False
# # #         404 Not Found               → job unknown (treat as not completed)

# # #     If the endpoint does not exist yet on your Spring Boot side, add it as:
# # #         GET /api/internal/analysis/status/{jobId}
# # #         → ResponseEntity<Map<String,String>>  { "status": job.getStatus().name() }

# # #     Raises on network errors so the consumer can decide whether to proceed
# # #     with processing or skip (consumer.py catches and proceeds on error).
# # #     """
# # #     try:
# # #         resp = _get(f"/api/internal/analysis/status/{job_id}")
# # #         data = resp.json()
# # #         status = data.get("status", "").upper()
# # #         is_done = status == "COMPLETED"
# # #         log.info(
# # #             "[Callback] check_job_completed  job=%s  status=%s  is_done=%s",
# # #             job_id, status, is_done,
# # #         )
# # #         return is_done
# # #     except RuntimeError as exc:
# # #         # 404 → job not found in the backend → definitely not completed
# # #         if "404" in str(exc):
# # #             log.info(
# # #                 "[Callback] check_job_completed  job=%s  → 404 (not found) → not completed",
# # #                 job_id,
# # #             )
# # #             return False
# # #         raise


# # # def send_progress(
# # #     job_id:        str,
# # #     journey_id:    int,
# # #     progress:      int,
# # #     message:       str,
# # #     status:        str = "PROCESSING",
# # #     current_video: Optional[int] = None,
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
# # #     current_video : 1-based index of the video currently being processed.
# # #     """
# # #     payload: Dict[str, Any] = {
# # #         "jobId":     job_id,
# # #         "journeyId": journey_id,
# # #         "status":    status,
# # #         "progress":  progress,
# # #         "message":   message,
# # #     }
# # #     if current_video is not None:
# # #         payload["currentVideo"] = current_video
# # #     _post("/api/internal/analysis/progress", payload)


# # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # #     """
# # #     POST /api/internal/analysis/completed

# # #     Called once after ALL videos in the journey have been processed and their
# # #     violation frames uploaded to S3.

# # #     completion_payload is built by CompletionPayload.to_dict() in models.py.
# # #     """
# # #     _post("/api/internal/analysis/completed", completion_payload)


# # # def send_failed(
# # #     job_id:        str,
# # #     journey_id:    int,   # kept for caller convenience / logging
# # #     error_message: str,
# # # ) -> None:
# # #     """
# # #     POST /api/internal/analysis/failed

# # #     Called whenever an unrecoverable exception occurs during job processing.
# # #     Spring Boot will mark AnalysisJob and Journey as FAILED.

# # #     Note: The API spec only requires jobId + errorMessage.
# # #     journeyId is accepted as a parameter for logging but is NOT included
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

# # Changes in this version
# # ────────────────────────
# # • check_job_completed() now uses the EXISTING completed-callback endpoint
# #   as a probe instead of a dedicated /status/{jobId} endpoint that doesn't
# #   exist yet on Spring Boot.

# #   Strategy: POST /api/internal/analysis/completed with only { "jobId": id,
# #   "probe": true }.  Spring Boot should return 409 CONFLICT (or any non-500)
# #   when the job is already done.  If it returns 500 we treat it as "unknown"
# #   and fall through to processing (safe default).

# #   *** TEMPORARY until the Spring Boot /status endpoint is implemented. ***
# #   Once GET /api/internal/analysis/status/{jobId} is live, revert
# #   check_job_completed() to use _get() as originally written.
# # """

# # from __future__ import annotations

# # import logging
# # import os
# # from typing import Any, Dict, Optional

# # import requests
# # from dotenv import load_dotenv

# # # ── Credentials / config ─────────────────────────────────────────────────────
# # _ENV_PATH = os.path.join(
# #     os.path.dirname(os.path.abspath(__file__)),
# #     "config", "credentials.env",
# # )
# # load_dotenv(_ENV_PATH)

# # _BASE_URL = os.environ.get(
# #     "SPRING_BOOT_BASE_URL",
# #     "https://cvvrsrailway-api.sconexsoft.com/cvs",
# # )
# # _TIMEOUT = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # log = logging.getLogger("callback_client")


# # # ── Dynamic base URL setter ───────────────────────────────────────────────────

# # def set_base_url(url: str) -> None:
# #     global _BASE_URL
# #     base = url.rstrip("/")
# #     if base.endswith("/api/internal/analysis"):
# #         base = base[: -len("/api/internal/analysis")]
# #     _BASE_URL = base
# #     log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)


# # # ── Internal helpers ──────────────────────────────────────────────────────────

# # def _post(path: str, payload: Dict[str, Any]) -> requests.Response:
# #     url = f"{_BASE_URL}{path}"
# #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# #     if not resp.ok:
# #         raise RuntimeError(
# #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# #         )
# #     log.debug("[Callback] %s → %d", path, resp.status_code)
# #     return resp


# # def _get(path: str) -> requests.Response:
# #     url = f"{_BASE_URL}{path}"
# #     log.debug("[Callback] GET %s", url)
# #     resp = requests.get(url, timeout=_TIMEOUT)
# #     if not resp.ok:
# #         raise RuntimeError(
# #             f"Callback GET {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# #         )
# #     log.debug("[Callback] %s → %d", path, resp.status_code)
# #     return resp


# # # ── Public API ────────────────────────────────────────────────────────────────

# # def check_job_completed(job_id: str) -> bool:
# #     """
# #     Idempotency check — returns True if the backend already has this job as COMPLETED.

# #     TWO-STAGE STRATEGY
# #     ──────────────────
# #     Stage 1 (permanent): GET /api/internal/analysis/status/{jobId}
# #         → { "status": "COMPLETED" }  → True
# #         → { "status": "PENDING/PROCESSING" } → False
# #         → 404 → False  (job not known yet)
# #         → 500 → fall through to Stage 2

# #     Stage 2 (temporary fallback until Spring Boot implements /status):
# #         Uses GET /api/internal/analysis/job/{jobId} or any existing
# #         read endpoint.  If that also 500s, we return False (safe default:
# #         process the job rather than silently drop it).

# #     WHAT TO IMPLEMENT ON SPRING BOOT (one small controller method):
# #     ───────────────────────────────────────────────────────────────
# #     @GetMapping("/api/internal/analysis/status/{jobId}")
# #     public ResponseEntity<Map<String,String>> getJobStatus(
# #             @PathVariable String jobId) {
# #         return analysisJobRepository.findByJobId(jobId)
# #             .map(job -> ResponseEntity.ok(Map.of("status", job.getStatus().name())))
# #             .orElse(ResponseEntity.notFound().<Map<String,String>>build());
# #     }

# #     Once that endpoint is deployed, Stage 2 below can be deleted.
# #     """

# #     # ── Stage 1: dedicated status endpoint ───────────────────────────────────
# #     try:
# #         resp = _get(f"/api/internal/analysis/status/{job_id}")
# #         status = resp.json().get("status", "").upper()
# #         is_done = status == "COMPLETED"
# #         log.info(
# #             "[Callback] idempotency check (status endpoint)  job=%s  status=%s  done=%s",
# #             job_id, status, is_done,
# #         )
# #         return is_done
# #     except RuntimeError as exc:
# #         if "404" in str(exc):
# #             log.info(
# #                 "[Callback] idempotency check  job=%s  → 404 → not completed", job_id
# #             )
# #             return False
# #         if "500" in str(exc):
# #             log.warning(
# #                 "[Callback] idempotency check  job=%s  → /status endpoint returned 500 "
# #                 "(not implemented yet?) — trying fallback probe.", job_id
# #             )
# #             # fall through to Stage 2
# #         else:
# #             # Network error or unexpected status — safe default: process the job
# #             log.warning(
# #                 "[Callback] idempotency check  job=%s  → unexpected error (%s) "
# #                 "— proceeding with processing.", job_id, exc
# #             )
# #             return False

# #     # ── Stage 2: fallback probe using progress endpoint ───────────────────────
# #     # We send a lightweight progress probe at 0 % with status=CHECK.
# #     # Spring Boot should:
# #     #   • Return 200 if the job is still PENDING/PROCESSING (→ process it).
# #     #   • Return 409 CONFLICT if the job is already COMPLETED (→ skip it).
# #     #   • Return 404 if the job is unknown (→ process it).
# #     # If the backend doesn't handle the CHECK status specially it will just
# #     # update progress to 0 — harmless on an already-completed job.
# #     #
# #     # THIS IS A TEMPORARY FALLBACK.  Remove once /status endpoint is live.
# #     try:
# #         url = f"{_BASE_URL}/api/internal/analysis/progress"
# #         resp = requests.post(
# #             url,
# #             json={
# #                 "jobId":    job_id,
# #                 "status":   "CHECK",   # sentinel value Spring Boot can detect
# #                 "progress": 0,
# #                 "message":  "idempotency-probe",
# #             },
# #             timeout=_TIMEOUT,
# #         )
# #         if resp.status_code == 409:
# #             log.warning(
# #                 "[Callback] idempotency check (probe)  job=%s  → 409 CONFLICT "
# #                 "→ already COMPLETED — will skip.", job_id
# #             )
# #             return True
# #         log.info(
# #             "[Callback] idempotency check (probe)  job=%s  → HTTP %d → not completed.",
# #             job_id, resp.status_code,
# #         )
# #         return False
# #     except Exception as exc2:
# #         log.warning(
# #             "[Callback] idempotency check (probe) failed  job=%s: %s "
# #             "— proceeding with processing.", job_id, exc2
# #         )
# #         return False


# # def send_progress(
# #     job_id:        str,
# #     journey_id:    int,
# #     progress:      int,
# #     message:       str,
# #     status:        str = "PROCESSING",
# #     current_video: Optional[int] = None,
# # ) -> None:
# #     payload: Dict[str, Any] = {
# #         "jobId":     job_id,
# #         "journeyId": journey_id,
# #         "status":    status,
# #         "progress":  progress,
# #         "message":   message,
# #     }
# #     if current_video is not None:
# #         payload["currentVideo"] = current_video
# #     _post("/api/internal/analysis/progress", payload)


# # def send_completed(completion_payload: Dict[str, Any]) -> None:
# #     _post("/api/internal/analysis/completed", completion_payload)


# # def send_failed(
# #     job_id:        str,
# #     journey_id:    int,
# #     error_message: str,
# # ) -> None:
# #     try:
# #         _post(
# #             "/api/internal/analysis/failed",
# #             {
# #                 "jobId":        job_id,
# #                 "errorMessage": error_message,
# #             },
# #         )
# #     except Exception as exc:
# #         log.error(
# #             "[Callback] send_failed itself failed (job=%s journey=%d): %s",
# #             job_id, journey_id, exc,
# #         )



# # # # # """
# # # # # callback_client.py
# # # # # ──────────────────
# # # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # # # All three public helpers are fire-and-forget from the caller's perspective;
# # # # # they raise on non-2xx so the consumer can route to the failure path.
# # # # # """

# # # # # from __future__ import annotations

# # # # # import logging
# # # # # import os
# # # # # from typing import Any, Dict, List

# # # # # import requests
# # # # # from dotenv import load_dotenv

# # # # # # ── Credentials / config ────────────────────────────────────────────────────
# # # # # _ENV_PATH = os.path.join(
# # # # #     os.path.dirname(os.path.abspath(__file__)),
# # # # #     "config", "credentials.env",
# # # # # )
# # # # # load_dotenv(_ENV_PATH)

# # # # # _BASE_URL = os.environ.get("SPRING_BOOT_BASE_URL", "http://localhost:8080")
# # # # # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # # # log = logging.getLogger("callback_client")


# # # # # # ── Internal helper ──────────────────────────────────────────────────────────

# # # # # def _post(path: str, payload: Dict[str, Any]) -> None:
# # # # #     url = f"{_BASE_URL}{path}"
# # # # #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# # # # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# # # # #     if not resp.ok:
# # # # #         raise RuntimeError(
# # # # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # # # #         )
# # # # #     log.debug("[Callback] %s → %d", path, resp.status_code)


# # # # # # ── Public API ───────────────────────────────────────────────────────────────

# # # # # def send_progress(
# # # # #     job_id:     str,
# # # # #     journey_id: int,
# # # # #     progress:   int,
# # # # #     message:    str,
# # # # #     status:     str = "PROCESSING",
# # # # # ) -> None:
# # # # #     """
# # # # #     POST /api/internal/analysis/progress

# # # # #     Called periodically during analysis to update the frontend progress bar
# # # # #     and the SSE stream.
# # # # #     """
# # # # #     _post(
# # # # #         "/api/internal/analysis/progress",
# # # # #         {
# # # # #             "jobId":      job_id,
# # # # #             "journeyId":  journey_id,
# # # # #             "status":     status,
# # # # #             "progress":   progress,
# # # # #             "message":    message,
# # # # #         },
# # # # #     )


# # # # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # # # #     """
# # # # #     POST /api/internal/analysis/completed

# # # # #     Called once — after ALL videos in the journey have been processed and
# # # # #     their violation frames uploaded to S3.

# # # # #     Expected shape of completion_payload
# # # # #     ─────────────────────────────────────
# # # # #     {
# # # # #         "jobId":          str,
# # # # #         "journeyId":      int,
# # # # #         "processingTime": int,          # wall-clock ms
# # # # #         "videoResults":   List[dict],   # see models.py → VideoResult
# # # # #     }
# # # # #     """
# # # # #     _post("/api/internal/analysis/completed", completion_payload)


# # # # # def send_failed(
# # # # #     job_id:        str,
# # # # #     journey_id:    int,
# # # # #     error_message: str,
# # # # # ) -> None:
# # # # #     """
# # # # #     POST /api/internal/analysis/failed

# # # # #     Called whenever an unrecoverable exception occurs during job processing.
# # # # #     Spring Boot will mark AnalysisJob and Journey as FAILED.
# # # # #     """
# # # # #     try:
# # # # #         _post(
# # # # #             "/api/internal/analysis/failed",
# # # # #             {
# # # # #                 "jobId":        job_id,
# # # # #                 "journeyId":    journey_id,
# # # # #                 "errorMessage": error_message,
# # # # #             },
# # # # #         )
# # # # #     except Exception as exc:
# # # # #         # Failure callback must never itself raise — log and swallow.
# # # # #         log.error("[Callback] send_failed itself failed: %s", exc)

# # # # """
# # # # callback_client.py
# # # # ──────────────────
# # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs
# # # # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).

# # # # Changes from previous version
# # # # ──────────────────────────────
# # # # • _BASE_URL now defaults to the live CVVRS domain (overridable via env var
# # # #   SPRING_BOOT_BASE_URL for local testing).
# # # # • send_progress()  — payload now includes the `currentVideo` field required
# # # #   by the API.
# # # # • send_completed() — accepts a fully-formed dict (built by CompletionPayload.
# # # #   to_dict()); no shape changes needed here since the caller owns the dict.
# # # # • send_failed()    — payload drops `journeyId` (not in the API spec for this
# # # #   endpoint); only jobId + errorMessage are sent.
# # # # """

# # # # from __future__ import annotations

# # # # import logging
# # # # import os
# # # # from typing import Any, Dict, Optional

# # # # import requests
# # # # from dotenv import load_dotenv

# # # # # ── Credentials / config ────────────────────────────────────────────────────
# # # # _ENV_PATH = os.path.join(
# # # #     os.path.dirname(os.path.abspath(__file__)),
# # # #     "config", "credentials.env",
# # # # )
# # # # load_dotenv(_ENV_PATH)

# # # # # Default to the live CVVRS server; override with SPRING_BOOT_BASE_URL for local dev.
# # # # _BASE_URL = os.environ.get(
# # # #     "SPRING_BOOT_BASE_URL",
# # # #     "https://cvvrsrailway-api.sconexsoft.com/cvs",
# # # # )
# # # # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # # log = logging.getLogger("callback_client")


# # # # # ── Internal helper ──────────────────────────────────────────────────────────

# # # # def _post(path: str, payload: Dict[str, Any]) -> None:
# # # #     """
# # # #     POST to a Spring Boot internal endpoint.
# # # #     No Authorization header is sent — /api/internal/* are worker-only endpoints.
# # # #     Raises RuntimeError on non-2xx so the consumer can route to failure path.
# # # #     """
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
# # # #     job_id:        str,
# # # #     journey_id:    int,
# # # #     progress:      int,
# # # #     message:       str,
# # # #     status:        str = "PROCESSING",
# # # #     current_video: Optional[int] = None,   # NEW — index of video currently being processed
# # # # ) -> None:
# # # #     """
# # # #     POST /api/internal/analysis/progress

# # # #     Called periodically during analysis to update the frontend progress bar
# # # #     and the SSE stream.

# # # #     Parameters
# # # #     ──────────
# # # #     job_id        : RabbitMQ job ID.
# # # #     journey_id    : Journey ID.
# # # #     progress      : 0–100 integer.
# # # #     message       : Human-readable status message.
# # # #     status        : "PROCESSING" | "COMPLETED" | "FAILED"
# # # #     current_video : 1-based index of the video currently being processed
# # # #                     (omitted from payload when None).
# # # #     """
# # # #     payload: Dict[str, Any] = {
# # # #         "jobId":      job_id,
# # # #         "journeyId":  journey_id,
# # # #         "status":     status,
# # # #         "progress":   progress,
# # # #         "message":    message,
# # # #     }
# # # #     if current_video is not None:
# # # #         payload["currentVideo"] = current_video

# # # #     _post("/api/internal/analysis/progress", payload)


# # # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # # #     """
# # # #     POST /api/internal/analysis/completed

# # # #     Called once after ALL videos in the journey have been processed and their
# # # #     violation frames uploaded to S3.

# # # #     Expected shape of completion_payload (built by CompletionPayload.to_dict())
# # # #     ────────────────────────────────────────────────────────────────────────────
# # # #     {
# # # #         "jobId":         str,
# # # #         "journeyId":     int,
# # # #         "batchId":       str,           # "BATCH-<jobId>" if not supplied
# # # #         "trainDetailId": int,
# # # #         "folderName":    str,
# # # #         "processingTime":int,           # wall-clock ms
# # # #         "videoResults": [
# # # #             {
# # # #                 "videoId":         str,       # STRING per API spec
# # # #                 "sequenceNo":      int,
# # # #                 "durationSeconds": float,
# # # #                 "originalS3Key":   str,
# # # #                 "violations": [
# # # #                     {
# # # #                         "violationType":          str,
# # # #                         "severity":               str,
# # # #                         "confidence":             float,
# # # #                         "riskScore":              float,
# # # #                         "timestamp":              float,   # journey-global seconds
# # # #                         "originalVideoTimestamp": float,   # local-video seconds
# # # #                         "framePaths":             [str]
# # # #                     }
# # # #                 ]
# # # #             }
# # # #         ]
# # # #     }
# # # #     """
# # # #     _post("/api/internal/analysis/completed", completion_payload)


# # # # def send_failed(
# # # #     job_id:        str,
# # # #     journey_id:    int,          # kept as parameter for caller convenience / logging
# # # #     error_message: str,
# # # # ) -> None:
# # # #     """
# # # #     POST /api/internal/analysis/failed

# # # #     Called whenever an unrecoverable exception occurs during job processing.
# # # #     Spring Boot will mark AnalysisJob and Journey as FAILED.

# # # #     Note: The API spec only requires jobId + errorMessage in the request body.
# # # #     journeyId is accepted as a parameter here for logging but is NOT included
# # # #     in the outbound payload.
# # # #     """
# # # #     try:
# # # #         _post(
# # # #             "/api/internal/analysis/failed",
# # # #             {
# # # #                 "jobId":        job_id,
# # # #                 "errorMessage": error_message,
# # # #             },
# # # #         )
# # # #     except Exception as exc:
# # # #         # Failure callback must never itself raise — log and swallow.
# # # #         log.error(
# # # #             "[Callback] send_failed itself failed (job=%s journey=%d): %s",
# # # #             job_id, journey_id, exc,
# # # #         )



# # # """

# # # callback_client.py

# # # ──────────────────

# # # Sends progress, completion, and failure callbacks to the Spring Boot backend.
 
# # # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs

# # # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).
 
# # # Fixes from previous version

# # # ─────────────────────────────

# # # • Added set_base_url() — called by consumer.py with the callbackBaseUrl from

# # #   the RabbitMQ message, so Python posts to the correct Spring Boot server

# # #   per environment without needing env var changes.

# # # • _BASE_URL still falls back to env var SPRING_BOOT_BASE_URL or the live

# # #   server when callbackBaseUrl is not provided in the message.

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
 
# # # # Default to the live CVVRS server; override with SPRING_BOOT_BASE_URL for local dev

# # # # or call set_base_url() with the callbackBaseUrl from the RabbitMQ message.

# # # _BASE_URL = os.environ.get(

# # #     "SPRING_BOOT_BASE_URL",

# # #     "https://cvvrsrailway-api.sconexsoft.com/cvs",

# # # )

# # # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))
 
# # # log = logging.getLogger("callback_client")
 
 
# # # # ── FIX: dynamic base URL setter ─────────────────────────────────────────────
 
# # # def set_base_url(url: str) -> None:

# # #     """

# # #     Override the callback base URL at runtime.
 
# # #     Called by consumer.py when the RabbitMQ message contains a callbackBaseUrl.

# # #     This allows the same Python worker to callback correctly to both local

# # #     and staging Spring Boot servers without changing env vars.
 
# # #     Example values:

# # #         "http://localhost:8093/api/internal/analysis"         (local)

# # #         "https://cvvrsrailway-api.sconexsoft.com/cvs/api/internal/analysis"  (staging)
 
# # #     Note: The URL passed here is the FULL path up to /api/internal/analysis.

# # #     The individual callbacks append /progress, /completed, /failed.

# # #     """

# # #     global _BASE_URL

# # #     # Strip the /api/internal/analysis suffix if present — we add it in _post()

# # #     base = url.rstrip("/")

# # #     if base.endswith("/api/internal/analysis"):

# # #         base = base[: -len("/api/internal/analysis")]

# # #     _BASE_URL = base

# # #     log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)
 
 
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

# # #     current_video: Optional[int] = None,

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

# # #     current_video : 1-based index of the video currently being processed.

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
 
# # #     completion_payload is built by CompletionPayload.to_dict() in models.py.

# # #     """

# # #     _post("/api/internal/analysis/completed", completion_payload)
 
 
# # # def send_failed(

# # #     job_id:        str,

# # #     journey_id:    int,          # kept for caller convenience / logging

# # #     error_message: str,

# # # ) -> None:

# # #     """

# # #     POST /api/internal/analysis/failed
 
# # #     Called whenever an unrecoverable exception occurs during job processing.

# # #     Spring Boot will mark AnalysisJob and Journey as FAILED.
 
# # #     Note: The API spec only requires jobId + errorMessage.

# # #     journeyId is accepted as a parameter for logging but is NOT included

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

# # Changes from previous version
# # ──────────────────────────────
# # • Added check_job_completed(job_id) — queries GET /api/internal/analysis/status/{jobId}
# #   and returns True when the backend reports the job is already COMPLETED.
# #   Called by consumer.py as an idempotency guard before starting any processing
# #   on a redelivered message.
# # """

# # from __future__ import annotations
# # import logging
# # import os
# # from typing import Any, Dict, Optional

# # import requests
# # from dotenv import load_dotenv

# # # ── Credentials / config ─────────────────────────────────────────────────────
# # _ENV_PATH = os.path.join(
# #     os.path.dirname(os.path.abspath(__file__)),
# #     "config", "credentials.env",
# # )
# # load_dotenv(_ENV_PATH)

# # _BASE_URL = os.environ.get(
# #     "SPRING_BOOT_BASE_URL",
# #     "https://cvvrsrailway-api.sconexsoft.com/cvs",
# # )
# # _TIMEOUT = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # log = logging.getLogger("callback_client")


# # # ── Dynamic base URL setter ───────────────────────────────────────────────────

# # def set_base_url(url: str) -> None:
# #     """
# #     Override the callback base URL at runtime.

# #     Called by consumer.py when the RabbitMQ message contains a callbackBaseUrl.
# #     Allows the same Python worker to callback correctly to both local and staging
# #     Spring Boot servers without changing env vars.

# #     The URL passed here is the FULL path up to /api/internal/analysis.
# #     Individual callbacks append /progress, /completed, /failed, /status/{id}.
# #     """
# #     global _BASE_URL
# #     base = url.rstrip("/")
# #     if base.endswith("/api/internal/analysis"):
# #         base = base[: -len("/api/internal/analysis")]
# #     _BASE_URL = base
# #     log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)


# # # ── Internal helpers ──────────────────────────────────────────────────────────

# # def _post(path: str, payload: Dict[str, Any]) -> None:
# #     """
# #     POST to a Spring Boot internal endpoint.
# #     No Authorization header — /api/internal/* are worker-only endpoints.
# #     Raises RuntimeError on non-2xx so the consumer can route to the failure path.
# #     """
# #     url = f"{_BASE_URL}{path}"
# #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# #     if not resp.ok:
# #         raise RuntimeError(
# #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# #         )
# #     log.debug("[Callback] %s → %d", path, resp.status_code)


# # def _get(path: str) -> requests.Response:
# #     """
# #     GET from a Spring Boot internal endpoint.
# #     Raises RuntimeError on non-2xx.
# #     """
# #     url = f"{_BASE_URL}{path}"
# #     log.debug("[Callback] GET %s", url)
# #     resp = requests.get(url, timeout=_TIMEOUT)
# #     if not resp.ok:
# #         raise RuntimeError(
# #             f"Callback GET {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# #         )
# #     log.debug("[Callback] %s → %d", path, resp.status_code)
# #     return resp


# # # ── Public API ────────────────────────────────────────────────────────────────

# # def check_job_completed(job_id: str) -> bool:
# #     """
# #     NEW — Idempotency check.

# #     Queries GET /api/internal/analysis/status/{jobId} and returns True when
# #     the backend reports the job status as COMPLETED.

# #     Called by consumer.py at the very start of _handle_job() so that
# #     RabbitMQ redeliveries of already-completed jobs are detected immediately
# #     and ACKed without re-running any processing.

# #     Backend contract (expected JSON shape):
# #         { "status": "COMPLETED" }   → job already done → return True
# #         { "status": "PROCESSING" }  → in-flight (rare edge-case) → return False
# #         { "status": "PENDING" }     → not yet processed → return False
# #         404 Not Found               → job unknown (treat as not completed)

# #     If the endpoint does not exist yet on your Spring Boot side, add it as:
# #         GET /api/internal/analysis/status/{jobId}
# #         → ResponseEntity<Map<String,String>>  { "status": job.getStatus().name() }

# #     Raises on network errors so the consumer can decide whether to proceed
# #     with processing or skip (consumer.py catches and proceeds on error).
# #     """
# #     try:
# #         resp = _get(f"/api/internal/analysis/status/{job_id}")
# #         data = resp.json()
# #         status = data.get("status", "").upper()
# #         is_done = status == "COMPLETED"
# #         log.info(
# #             "[Callback] check_job_completed  job=%s  status=%s  is_done=%s",
# #             job_id, status, is_done,
# #         )
# #         return is_done
# #     except RuntimeError as exc:
# #         # 404 → job not found in the backend → definitely not completed
# #         if "404" in str(exc):
# #             log.info(
# #                 "[Callback] check_job_completed  job=%s  → 404 (not found) → not completed",
# #                 job_id,
# #             )
# #             return False
# #         raise


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
# #         "jobId":     job_id,
# #         "journeyId": journey_id,
# #         "status":    status,
# #         "progress":  progress,
# #         "message":   message,
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
# #     journey_id:    int,   # kept for caller convenience / logging
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

# Changes in this version
# ────────────────────────
# • check_job_completed() now uses the EXISTING completed-callback endpoint
#   as a probe instead of a dedicated /status/{jobId} endpoint that doesn't
#   exist yet on Spring Boot.

#   Strategy: POST /api/internal/analysis/completed with only { "jobId": id,
#   "probe": true }.  Spring Boot should return 409 CONFLICT (or any non-500)
#   when the job is already done.  If it returns 500 we treat it as "unknown"
#   and fall through to processing (safe default).

#   *** TEMPORARY until the Spring Boot /status endpoint is implemented. ***
#   Once GET /api/internal/analysis/status/{jobId} is live, revert
#   check_job_completed() to use _get() as originally written.
# """

# from __future__ import annotations

# import logging
# import os
# import threading
# from typing import Any, Dict, Optional

# import requests
# from dotenv import load_dotenv

# # ── Credentials / config ─────────────────────────────────────────────────────
# _ENV_PATH = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)),
#     "config", "credentials.env",
# )
# load_dotenv(_ENV_PATH)

# _DEFAULT_BASE_URL = os.environ.get(
#     "SPRING_BOOT_BASE_URL",
#     "https://cvvrsrailway-api.sconexsoft.com/cvs",
# )
# _TIMEOUT = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# log = logging.getLogger("callback_client")

# # ── Thread-local base URL ──────────────────────────────────────────────────────
# # NOTE (worker-pool refactor): the RabbitMQ consumer can now have MULTIPLE
# # journeys in flight at once (one per GPU worker), each running _handle_job()
# # on its own thread. A single module-level _BASE_URL global would let one
# # journey's set_base_url() call silently redirect ANOTHER, concurrently
# # running journey's callbacks to the wrong host. Storing it in thread-local
# # state keeps each journey's callback_base_url isolated to the thread that's
# # actually processing it, with no change to single-journey-at-a-time behavior.
# _base_url_local = threading.local()


# def _get_base_url() -> str:
#     return getattr(_base_url_local, "value", _DEFAULT_BASE_URL)


# # ── Dynamic base URL setter ───────────────────────────────────────────────────

# def set_base_url(url: str) -> None:
#     base = url.rstrip("/")
#     if base.endswith("/api/internal/analysis"):
#         base = base[: -len("/api/internal/analysis")]
#     _base_url_local.value = base
#     log.info("[CallbackClient]  Base URL set to: %s (thread=%s)",
#              base, threading.current_thread().name)


# # ── Internal helpers ──────────────────────────────────────────────────────────

# def _post(path: str, payload: Dict[str, Any]) -> requests.Response:
#     url = f"{_get_base_url()}{path}"
#     log.debug("[Callback] POST %s  payload=%s", url, payload)
#     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
#     if not resp.ok:
#         raise RuntimeError(
#             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
#         )
#     log.debug("[Callback] %s → %d", path, resp.status_code)
#     return resp


# def _get(path: str) -> requests.Response:
#     url = f"{_get_base_url()}{path}"
#     log.debug("[Callback] GET %s", url)
#     resp = requests.get(url, timeout=_TIMEOUT)
#     if not resp.ok:
#         raise RuntimeError(
#             f"Callback GET {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
#         )
#     log.debug("[Callback] %s → %d", path, resp.status_code)
#     return resp


# # ── Public API ────────────────────────────────────────────────────────────────

# # ── Local idempotency cache ───────────────────────────────────────────────────
# #
# # When the worker finishes a job it records the job_id here so that if the
# # same message is redelivered (RabbitMQ consumer-timeout cancels the consumer
# # mid-journey, broker requeues, worker reconnects and picks it up again) the
# # idempotency check catches it locally WITHOUT relying on the backend /status
# # endpoint — which currently returns 500 for both probe strategies.
# #
# # The cache is process-lifetime only (lost on restart).  That is intentional:
# # after a genuine worker crash we WANT to reprocess any unACKed job.  The
# # repeat-job problem described in the bug report is caused by RabbitMQ's
# # consumer_timeout firing while the worker is still alive and processing,
# # so the in-memory cache is always present when the redelivery arrives.
# #
# # mark_job_completed() is called by consumer.py immediately before sending
# # the ACK, so the job is in the cache for any subsequent redelivery.
# _completed_jobs: set = set()
# _completed_jobs_lock = threading.Lock()


# def mark_job_completed(job_id: str) -> None:
#     """Record that this worker instance has successfully completed job_id."""
#     with _completed_jobs_lock:
#         _completed_jobs.add(job_id)
#     log.info("[CallbackClient]  Marked job %s as locally completed.", job_id)


# # ── In-progress job guard ─────────────────────────────────────────────────────
# #
# # _completed_jobs (above) only catches a redelivery of a job that has
# # ALREADY finished. It does NOT catch a redelivery that arrives WHILE the
# # original delivery is still being processed on another thread.
# #
# # Before the worker-pool refactor this could never happen: _on_message()
# # ran _handle_job() directly on the single pika I/O thread, so that thread
# # had to finish (or crash) before it could ever see a redelivery. Now that
# # each journey runs on its own thread (so multiple GPU workers can be busy
# # at once), a dropped/reconnected RabbitMQ connection can redeliver an
# # unacked message while the ORIGINAL job thread is still mid-journey on a
# # different GPU worker — resulting in the same journey being processed
# # twice, in parallel, by two different workers. try_start_job()/
# # finish_job() below close that gap: a job_id can only be "in progress" on
# # one thread at a time.
# _in_progress_jobs: set = set()
# _in_progress_jobs_lock = threading.Lock()


# def try_start_job(job_id: str) -> bool:
#     """
#     Attempt to claim job_id for the calling thread. Returns True if this
#     thread is now the sole owner and should proceed; returns False if
#     another thread already owns it (a redelivery arrived mid-flight) — the
#     caller should NOT reprocess the journey in that case.
#     """
#     with _in_progress_jobs_lock:
#         if job_id in _in_progress_jobs:
#             return False
#         _in_progress_jobs.add(job_id)
#         return True


# def finish_job(job_id: str) -> None:
#     """Release the claim on job_id. MUST be called from a finally block so
#     a claim is never left dangling — otherwise a genuine crash-recovery
#     redelivery (after the original truly died) would be blocked forever."""
#     with _in_progress_jobs_lock:
#         _in_progress_jobs.discard(job_id)


# def check_job_completed(job_id: str) -> bool:
#     """
#     Idempotency check — returns True if the backend already has this job as COMPLETED.

#     STAGE 0 (local cache — fastest, most reliable):
#     ─────────────────────────────────────────────────
#     If THIS worker process already completed and ACKed the job during the
#     current run, it is in _completed_jobs and we return True immediately,
#     bypassing both backend probes entirely.  This is the correct fix for the
#     RabbitMQ consumer-timeout redelivery problem: the broker requeues the
#     message and the worker's reconnect loop picks it up again, but the
#     in-memory cache catches it before any work is repeated.

#     TWO-STAGE STRATEGY (backend probes — for cross-process idempotency)
#     ──────────────────
#     Stage 1 (permanent): GET /api/internal/analysis/status/{jobId}
#         → { "status": "COMPLETED" }  → True
#         → { "status": "PENDING/PROCESSING" } → False
#         → 404 → False  (job not known yet)
#         → 500 → fall through to Stage 2

#     Stage 2 (temporary fallback until Spring Boot implements /status):
#         Uses GET /api/internal/analysis/job/{jobId} or any existing
#         read endpoint.  If that also 500s, we return False (safe default:
#         process the job rather than silently drop it).

#     WHAT TO IMPLEMENT ON SPRING BOOT (one small controller method):
#     ───────────────────────────────────────────────────────────────
#     @GetMapping("/api/internal/analysis/status/{jobId}")
#     public ResponseEntity<Map<String,String>> getJobStatus(
#             @PathVariable String jobId) {
#         return analysisJobRepository.findByJobId(jobId)
#             .map(job -> ResponseEntity.ok(Map.of("status", job.getStatus().name())))
#             .orElse(ResponseEntity.notFound().<Map<String,String>>build());
#     }

#     Once that endpoint is deployed, Stage 2 below can be deleted.
#     """

#     # ── Stage 0: local in-process cache (catches consumer-timeout redeliveries)
#     with _completed_jobs_lock:
#         already_done = job_id in _completed_jobs
#     if already_done:
#         log.warning(
#             "[Callback] idempotency check  job=%s  → found in local completed cache "
#             "— this is a redelivery of an already-ACKed job (likely caused by "
#             "RabbitMQ consumer_timeout). Skipping re-processing.", job_id,
#         )
#         return True

#     # ── Stage 1: dedicated status endpoint ───────────────────────────────────
#     try:
#         resp = _get(f"/api/internal/analysis/status/{job_id}")
#         status = resp.json().get("status", "").upper()
#         is_done = status == "COMPLETED"
#         log.info(
#             "[Callback] idempotency check (status endpoint)  job=%s  status=%s  done=%s",
#             job_id, status, is_done,
#         )
#         return is_done
#     except RuntimeError as exc:
#         if "404" in str(exc):
#             log.info(
#                 "[Callback] idempotency check  job=%s  → 404 → not completed", job_id
#             )
#             return False
#         if "500" in str(exc):
#             log.warning(
#                 "[Callback] idempotency check  job=%s  → /status endpoint returned 500 "
#                 "(not implemented yet?) — trying fallback probe.", job_id
#             )
#             # fall through to Stage 2
#         else:
#             # Network error or unexpected status — safe default: process the job
#             log.warning(
#                 "[Callback] idempotency check  job=%s  → unexpected error (%s) "
#                 "— proceeding with processing.", job_id, exc
#             )
#             return False

#     # ── Stage 2: fallback probe using progress endpoint ───────────────────────
#     # We send a lightweight progress probe at 0 % with status=CHECK.
#     # Spring Boot should:
#     #   • Return 200 if the job is still PENDING/PROCESSING (→ process it).
#     #   • Return 409 CONFLICT if the job is already COMPLETED (→ skip it).
#     #   • Return 404 if the job is unknown (→ process it).
#     # If the backend doesn't handle the CHECK status specially it will just
#     # update progress to 0 — harmless on an already-completed job.
#     #
#     # THIS IS A TEMPORARY FALLBACK.  Remove once /status endpoint is live.
#     try:
#         url = f"{_get_base_url()}/api/internal/analysis/progress"
#         resp = requests.post(
#             url,
#             json={
#                 "jobId":    job_id,
#                 "status":   "CHECK",   # sentinel value Spring Boot can detect
#                 "progress": 0,
#                 "message":  "idempotency-probe",
#             },
#             timeout=_TIMEOUT,
#         )
#         if resp.status_code == 409:
#             log.warning(
#                 "[Callback] idempotency check (probe)  job=%s  → 409 CONFLICT "
#                 "→ already COMPLETED — will skip.", job_id
#             )
#             return True
#         log.info(
#             "[Callback] idempotency check (probe)  job=%s  → HTTP %d → not completed.",
#             job_id, resp.status_code,
#         )
#         return False
#     except Exception as exc2:
#         log.warning(
#             "[Callback] idempotency check (probe) failed  job=%s: %s "
#             "— proceeding with processing.", job_id, exc2
#         )
#         return False


# def send_progress(
#     job_id:        str,
#     journey_id:    int,
#     progress:      int,
#     message:       str,
#     status:        str = "PROCESSING",
#     current_video: Optional[int] = None,
# ) -> None:
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
#     _post("/api/internal/analysis/completed", completion_payload)


# def send_failed(
#     job_id:        str,
#     journey_id:    int,
#     error_message: str,
#     video_id:      Optional[int] = None,
#     error_type:    Optional[str] = None,
#     stack_trace:   Optional[str] = None,
#     reason:        Optional[str] = None,
# ) -> None:
#     """
#     Calls POST /api/internal/analysis/failed.

#     Backward compatible: existing job-level callers (no video_id) keep
#     working exactly as before — the payload shape for that case is
#     unchanged ({"jobId", "errorMessage"}).

#     When video_id IS supplied, this becomes a PER-VIDEO failure report
#     (Phase 1 requirement: "no failed video should wait until journey
#     completion to be reported"). The payload then also carries:
#         videoId     — STRING, matches VideoResult.to_dict()'s convention
#         journeyId
#         errorType   — short classifier, e.g. "OOM", "DECODE_ERROR",
#                       "MEDIAPIPE_ERROR", "YOLO_ERROR", "TIMEOUT",
#                       "RESOURCE_EXHAUSTION", "NOT_PROCESSED"
#         stackTrace  — full traceback text, if available
#         reason      — human-readable reason string, e.g.
#                       "Not Processed - Worker Resource Exhaustion"
#                       for videos skipped after an OOM on an earlier video.
#     """
#     payload: Dict[str, Any] = {
#         "jobId":        job_id,
#         "errorMessage": error_message,
#     }
#     if video_id is not None:
#         payload["videoId"]    = str(video_id)   # STRING per API spec convention
#         payload["journeyId"]  = journey_id
#     if error_type is not None:
#         payload["errorType"] = error_type
#     if stack_trace is not None:
#         payload["stackTrace"] = stack_trace
#     if reason is not None:
#         payload["reason"] = reason

#     try:
#         _post("/api/internal/analysis/failed", payload)
#     except Exception as exc:
#         log.error(
#             "[Callback] send_failed itself failed (job=%s journey=%d video=%s): %s",
#             job_id, journey_id, video_id, exc,
#         )


# def send_video_failed(
#     job_id:        str,
#     journey_id:    int,
#     video_id:      int,
#     error_type:    str,
#     error_message: str,
#     stack_trace:   str = "",
#     reason:        Optional[str] = None,
# ) -> None:
#     """
#     Convenience wrapper for the per-video failure report required by
#     Phase 1: "Immediately call the failed endpoint for the affected video"
#     — called once per video, the moment that video's outcome is known,
#     not batched until the journey ends.

#     For resource-exhaustion cascades, pass error_type="RESOURCE_EXHAUSTION"
#     (the video that actually hit OOM) or error_type="NOT_PROCESSED" with
#     reason="Not Processed - Worker Resource Exhaustion" (videos after it
#     that were skipped as a result).
#     """
#     log.error(
#         "[Callback] video FAILED  job=%s  journey=%d  video=%d  type=%s  reason=%s: %s",
#         job_id, journey_id, video_id, error_type, reason, error_message,
#     )
#     send_failed(
#         job_id        = job_id,
#         journey_id    = journey_id,
#         error_message = error_message,
#         video_id      = video_id,
#         error_type    = error_type,
#         stack_trace   = stack_trace,
#         reason        = reason,
#     )


# # ── OOM / resource-exhaustion signature detection ────────────────────────────
# #
# # Used by analyzer.py to decide whether a per-video exception means "this
# # one video is bad" (continue with the rest) vs. "the worker itself can no
# # longer safely process more video" (stop the journey, mark every
# # remaining video FAILED with reason "Not Processed - Worker Resource
# # Exhaustion"). Matches the signatures called out explicitly in the
# # Phase 1 spec: OpenCV OutOfMemoryError, "Failed to allocate", cv2.pyd
# # memory-related failures, generic MemoryError, std::bad_alloc.
# _OOM_SIGNATURES = (
#     "outofmemoryerror",
#     "out of memory",
#     "failed to allocate",
#     "cannot allocate memory",
#     "bad_alloc",
#     "memoryerror",
#     "cv2.pyd",
#     "access violation",
#     "resource exhaust",
# )


# def is_resource_exhaustion_error(exc: BaseException) -> bool:
#     """
#     Returns True if `exc` looks like an OOM / native resource-exhaustion
#     failure rather than an ordinary recoverable per-video error.
#     """
#     text = f"{type(exc).__name__}: {exc}".lower()
#     if isinstance(exc, MemoryError):
#         return True
#     return any(sig in text for sig in _OOM_SIGNATURES)


# def classify_video_error(exc: BaseException) -> str:
#     """
#     Maps a caught per-video exception to a short errorType string for the
#     failed-endpoint payload. Best-effort classification by exception type
#     and message content — defaults to "PROCESSING_ERROR" when nothing
#     more specific matches.
#     """
#     if is_resource_exhaustion_error(exc):
#         return "RESOURCE_EXHAUSTION"
#     name = type(exc).__name__.lower()
#     text = str(exc).lower()
#     if "timeout" in name or "timeout" in text:
#         return "TIMEOUT"
#     if "mediapipe" in text:
#         return "MEDIAPIPE_ERROR"
#     if "yolo" in text or "ultralytics" in text:
#         return "YOLO_ERROR"
#     if "cv2" in text or "opencv" in text or name == "error":
#         return "OPENCV_ERROR"
#     if "cannot open" in text or "corrupt" in text or "decode" in text or "invalid" in text:
#         return "DECODE_ERROR"
#     return "PROCESSING_ERROR"


# def compute_journey_status(
#     total_videos: int,
#     succeeded_video_ids: "set[int]",
#     failed_video_ids: "set[int]",
# ) -> str:
#     """
#     Computes the journey-level terminal status per Phase 1 spec:
#         COMPLETED              — every video succeeded
#         COMPLETED_WITH_ERRORS  — at least one succeeded AND at least one failed
#         FAILED                 — every video failed (or none succeeded)

#     TIMED_OUT is NOT computed here — that's set explicitly by the watchdog
#     path in consumer.py, which knows it hit a timeout rather than inferring
#     it from video counts.
#     """
#     if total_videos == 0:
#         return "FAILED"
#     if not failed_video_ids:
#         return "COMPLETED"
#     if succeeded_video_ids:
#         return "COMPLETED_WITH_ERRORS"
#     return "FAILED"



# # # # # # """
# # # # # # callback_client.py
# # # # # # ──────────────────
# # # # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # # # # All three public helpers are fire-and-forget from the caller's perspective;
# # # # # # they raise on non-2xx so the consumer can route to the failure path.
# # # # # # """

# # # # # # from __future__ import annotations

# # # # # # import logging
# # # # # # import os
# # # # # # from typing import Any, Dict, List

# # # # # # import requests
# # # # # # from dotenv import load_dotenv

# # # # # # # ── Credentials / config ────────────────────────────────────────────────────
# # # # # # _ENV_PATH = os.path.join(
# # # # # #     os.path.dirname(os.path.abspath(__file__)),
# # # # # #     "config", "credentials.env",
# # # # # # )
# # # # # # load_dotenv(_ENV_PATH)

# # # # # # _BASE_URL = os.environ.get("SPRING_BOOT_BASE_URL", "http://localhost:8080")
# # # # # # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # # # # log = logging.getLogger("callback_client")


# # # # # # # ── Internal helper ──────────────────────────────────────────────────────────

# # # # # # def _post(path: str, payload: Dict[str, Any]) -> None:
# # # # # #     url = f"{_BASE_URL}{path}"
# # # # # #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# # # # # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# # # # # #     if not resp.ok:
# # # # # #         raise RuntimeError(
# # # # # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # # # # #         )
# # # # # #     log.debug("[Callback] %s → %d", path, resp.status_code)


# # # # # # # ── Public API ───────────────────────────────────────────────────────────────

# # # # # # def send_progress(
# # # # # #     job_id:     str,
# # # # # #     journey_id: int,
# # # # # #     progress:   int,
# # # # # #     message:    str,
# # # # # #     status:     str = "PROCESSING",
# # # # # # ) -> None:
# # # # # #     """
# # # # # #     POST /api/internal/analysis/progress

# # # # # #     Called periodically during analysis to update the frontend progress bar
# # # # # #     and the SSE stream.
# # # # # #     """
# # # # # #     _post(
# # # # # #         "/api/internal/analysis/progress",
# # # # # #         {
# # # # # #             "jobId":      job_id,
# # # # # #             "journeyId":  journey_id,
# # # # # #             "status":     status,
# # # # # #             "progress":   progress,
# # # # # #             "message":    message,
# # # # # #         },
# # # # # #     )


# # # # # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # # # # #     """
# # # # # #     POST /api/internal/analysis/completed

# # # # # #     Called once — after ALL videos in the journey have been processed and
# # # # # #     their violation frames uploaded to S3.

# # # # # #     Expected shape of completion_payload
# # # # # #     ─────────────────────────────────────
# # # # # #     {
# # # # # #         "jobId":          str,
# # # # # #         "journeyId":      int,
# # # # # #         "processingTime": int,          # wall-clock ms
# # # # # #         "videoResults":   List[dict],   # see models.py → VideoResult
# # # # # #     }
# # # # # #     """
# # # # # #     _post("/api/internal/analysis/completed", completion_payload)


# # # # # # def send_failed(
# # # # # #     job_id:        str,
# # # # # #     journey_id:    int,
# # # # # #     error_message: str,
# # # # # # ) -> None:
# # # # # #     """
# # # # # #     POST /api/internal/analysis/failed

# # # # # #     Called whenever an unrecoverable exception occurs during job processing.
# # # # # #     Spring Boot will mark AnalysisJob and Journey as FAILED.
# # # # # #     """
# # # # # #     try:
# # # # # #         _post(
# # # # # #             "/api/internal/analysis/failed",
# # # # # #             {
# # # # # #                 "jobId":        job_id,
# # # # # #                 "journeyId":    journey_id,
# # # # # #                 "errorMessage": error_message,
# # # # # #             },
# # # # # #         )
# # # # # #     except Exception as exc:
# # # # # #         # Failure callback must never itself raise — log and swallow.
# # # # # #         log.error("[Callback] send_failed itself failed: %s", exc)

# # # # # """
# # # # # callback_client.py
# # # # # ──────────────────
# # # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # # # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs
# # # # # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).

# # # # # Changes from previous version
# # # # # ──────────────────────────────
# # # # # • _BASE_URL now defaults to the live CVVRS domain (overridable via env var
# # # # #   SPRING_BOOT_BASE_URL for local testing).
# # # # # • send_progress()  — payload now includes the `currentVideo` field required
# # # # #   by the API.
# # # # # • send_completed() — accepts a fully-formed dict (built by CompletionPayload.
# # # # #   to_dict()); no shape changes needed here since the caller owns the dict.
# # # # # • send_failed()    — payload drops `journeyId` (not in the API spec for this
# # # # #   endpoint); only jobId + errorMessage are sent.
# # # # # """

# # # # # from __future__ import annotations

# # # # # import logging
# # # # # import os
# # # # # from typing import Any, Dict, Optional

# # # # # import requests
# # # # # from dotenv import load_dotenv

# # # # # # ── Credentials / config ────────────────────────────────────────────────────
# # # # # _ENV_PATH = os.path.join(
# # # # #     os.path.dirname(os.path.abspath(__file__)),
# # # # #     "config", "credentials.env",
# # # # # )
# # # # # load_dotenv(_ENV_PATH)

# # # # # # Default to the live CVVRS server; override with SPRING_BOOT_BASE_URL for local dev.
# # # # # _BASE_URL = os.environ.get(
# # # # #     "SPRING_BOOT_BASE_URL",
# # # # #     "https://cvvrsrailway-api.sconexsoft.com/cvs",
# # # # # )
# # # # # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # # # log = logging.getLogger("callback_client")


# # # # # # ── Internal helper ──────────────────────────────────────────────────────────

# # # # # def _post(path: str, payload: Dict[str, Any]) -> None:
# # # # #     """
# # # # #     POST to a Spring Boot internal endpoint.
# # # # #     No Authorization header is sent — /api/internal/* are worker-only endpoints.
# # # # #     Raises RuntimeError on non-2xx so the consumer can route to failure path.
# # # # #     """
# # # # #     url = f"{_BASE_URL}{path}"
# # # # #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# # # # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# # # # #     if not resp.ok:
# # # # #         raise RuntimeError(
# # # # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # # # #         )
# # # # #     log.debug("[Callback] %s → %d", path, resp.status_code)


# # # # # # ── Public API ───────────────────────────────────────────────────────────────

# # # # # def send_progress(
# # # # #     job_id:        str,
# # # # #     journey_id:    int,
# # # # #     progress:      int,
# # # # #     message:       str,
# # # # #     status:        str = "PROCESSING",
# # # # #     current_video: Optional[int] = None,   # NEW — index of video currently being processed
# # # # # ) -> None:
# # # # #     """
# # # # #     POST /api/internal/analysis/progress

# # # # #     Called periodically during analysis to update the frontend progress bar
# # # # #     and the SSE stream.

# # # # #     Parameters
# # # # #     ──────────
# # # # #     job_id        : RabbitMQ job ID.
# # # # #     journey_id    : Journey ID.
# # # # #     progress      : 0–100 integer.
# # # # #     message       : Human-readable status message.
# # # # #     status        : "PROCESSING" | "COMPLETED" | "FAILED"
# # # # #     current_video : 1-based index of the video currently being processed
# # # # #                     (omitted from payload when None).
# # # # #     """
# # # # #     payload: Dict[str, Any] = {
# # # # #         "jobId":      job_id,
# # # # #         "journeyId":  journey_id,
# # # # #         "status":     status,
# # # # #         "progress":   progress,
# # # # #         "message":    message,
# # # # #     }
# # # # #     if current_video is not None:
# # # # #         payload["currentVideo"] = current_video

# # # # #     _post("/api/internal/analysis/progress", payload)


# # # # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # # # #     """
# # # # #     POST /api/internal/analysis/completed

# # # # #     Called once after ALL videos in the journey have been processed and their
# # # # #     violation frames uploaded to S3.

# # # # #     Expected shape of completion_payload (built by CompletionPayload.to_dict())
# # # # #     ────────────────────────────────────────────────────────────────────────────
# # # # #     {
# # # # #         "jobId":         str,
# # # # #         "journeyId":     int,
# # # # #         "batchId":       str,           # "BATCH-<jobId>" if not supplied
# # # # #         "trainDetailId": int,
# # # # #         "folderName":    str,
# # # # #         "processingTime":int,           # wall-clock ms
# # # # #         "videoResults": [
# # # # #             {
# # # # #                 "videoId":         str,       # STRING per API spec
# # # # #                 "sequenceNo":      int,
# # # # #                 "durationSeconds": float,
# # # # #                 "originalS3Key":   str,
# # # # #                 "violations": [
# # # # #                     {
# # # # #                         "violationType":          str,
# # # # #                         "severity":               str,
# # # # #                         "confidence":             float,
# # # # #                         "riskScore":              float,
# # # # #                         "timestamp":              float,   # journey-global seconds
# # # # #                         "originalVideoTimestamp": float,   # local-video seconds
# # # # #                         "framePaths":             [str]
# # # # #                     }
# # # # #                 ]
# # # # #             }
# # # # #         ]
# # # # #     }
# # # # #     """
# # # # #     _post("/api/internal/analysis/completed", completion_payload)


# # # # # def send_failed(
# # # # #     job_id:        str,
# # # # #     journey_id:    int,          # kept as parameter for caller convenience / logging
# # # # #     error_message: str,
# # # # # ) -> None:
# # # # #     """
# # # # #     POST /api/internal/analysis/failed

# # # # #     Called whenever an unrecoverable exception occurs during job processing.
# # # # #     Spring Boot will mark AnalysisJob and Journey as FAILED.

# # # # #     Note: The API spec only requires jobId + errorMessage in the request body.
# # # # #     journeyId is accepted as a parameter here for logging but is NOT included
# # # # #     in the outbound payload.
# # # # #     """
# # # # #     try:
# # # # #         _post(
# # # # #             "/api/internal/analysis/failed",
# # # # #             {
# # # # #                 "jobId":        job_id,
# # # # #                 "errorMessage": error_message,
# # # # #             },
# # # # #         )
# # # # #     except Exception as exc:
# # # # #         # Failure callback must never itself raise — log and swallow.
# # # # #         log.error(
# # # # #             "[Callback] send_failed itself failed (job=%s journey=%d): %s",
# # # # #             job_id, journey_id, exc,
# # # # #         )



# # # # """

# # # # callback_client.py

# # # # ──────────────────

# # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.
 
# # # # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs

# # # # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).
 
# # # # Fixes from previous version

# # # # ─────────────────────────────

# # # # • Added set_base_url() — called by consumer.py with the callbackBaseUrl from

# # # #   the RabbitMQ message, so Python posts to the correct Spring Boot server

# # # #   per environment without needing env var changes.

# # # # • _BASE_URL still falls back to env var SPRING_BOOT_BASE_URL or the live

# # # #   server when callbackBaseUrl is not provided in the message.

# # # # """
 
# # # # from __future__ import annotations
 
# # # # import logging

# # # # import os

# # # # from typing import Any, Dict, Optional
 
# # # # import requests

# # # # from dotenv import load_dotenv
 
# # # # # ── Credentials / config ────────────────────────────────────────────────────

# # # # _ENV_PATH = os.path.join(

# # # #     os.path.dirname(os.path.abspath(__file__)),

# # # #     "config", "credentials.env",

# # # # )

# # # # load_dotenv(_ENV_PATH)
 
# # # # # Default to the live CVVRS server; override with SPRING_BOOT_BASE_URL for local dev

# # # # # or call set_base_url() with the callbackBaseUrl from the RabbitMQ message.

# # # # _BASE_URL = os.environ.get(

# # # #     "SPRING_BOOT_BASE_URL",

# # # #     "https://cvvrsrailway-api.sconexsoft.com/cvs",

# # # # )

# # # # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))
 
# # # # log = logging.getLogger("callback_client")
 
 
# # # # # ── FIX: dynamic base URL setter ─────────────────────────────────────────────
 
# # # # def set_base_url(url: str) -> None:

# # # #     """

# # # #     Override the callback base URL at runtime.
 
# # # #     Called by consumer.py when the RabbitMQ message contains a callbackBaseUrl.

# # # #     This allows the same Python worker to callback correctly to both local

# # # #     and staging Spring Boot servers without changing env vars.
 
# # # #     Example values:

# # # #         "http://localhost:8093/api/internal/analysis"         (local)

# # # #         "https://cvvrsrailway-api.sconexsoft.com/cvs/api/internal/analysis"  (staging)
 
# # # #     Note: The URL passed here is the FULL path up to /api/internal/analysis.

# # # #     The individual callbacks append /progress, /completed, /failed.

# # # #     """

# # # #     global _BASE_URL

# # # #     # Strip the /api/internal/analysis suffix if present — we add it in _post()

# # # #     base = url.rstrip("/")

# # # #     if base.endswith("/api/internal/analysis"):

# # # #         base = base[: -len("/api/internal/analysis")]

# # # #     _BASE_URL = base

# # # #     log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)
 
 
# # # # # ── Internal helper ──────────────────────────────────────────────────────────
 
# # # # def _post(path: str, payload: Dict[str, Any]) -> None:

# # # #     """

# # # #     POST to a Spring Boot internal endpoint.

# # # #     No Authorization header is sent — /api/internal/* are worker-only endpoints.

# # # #     Raises RuntimeError on non-2xx so the consumer can route to failure path.

# # # #     """

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

# # # #     job_id:        str,

# # # #     journey_id:    int,

# # # #     progress:      int,

# # # #     message:       str,

# # # #     status:        str = "PROCESSING",

# # # #     current_video: Optional[int] = None,

# # # # ) -> None:

# # # #     """

# # # #     POST /api/internal/analysis/progress
 
# # # #     Called periodically during analysis to update the frontend progress bar

# # # #     and the SSE stream.
 
# # # #     Parameters

# # # #     ──────────

# # # #     job_id        : RabbitMQ job ID.

# # # #     journey_id    : Journey ID.

# # # #     progress      : 0–100 integer.

# # # #     message       : Human-readable status message.

# # # #     status        : "PROCESSING" | "COMPLETED" | "FAILED"

# # # #     current_video : 1-based index of the video currently being processed.

# # # #     """

# # # #     payload: Dict[str, Any] = {

# # # #         "jobId":      job_id,

# # # #         "journeyId":  journey_id,

# # # #         "status":     status,

# # # #         "progress":   progress,

# # # #         "message":    message,

# # # #     }

# # # #     if current_video is not None:

# # # #         payload["currentVideo"] = current_video
 
# # # #     _post("/api/internal/analysis/progress", payload)
 
 
# # # # def send_completed(completion_payload: Dict[str, Any]) -> None:

# # # #     """

# # # #     POST /api/internal/analysis/completed
 
# # # #     Called once after ALL videos in the journey have been processed and their

# # # #     violation frames uploaded to S3.
 
# # # #     completion_payload is built by CompletionPayload.to_dict() in models.py.

# # # #     """

# # # #     _post("/api/internal/analysis/completed", completion_payload)
 
 
# # # # def send_failed(

# # # #     job_id:        str,

# # # #     journey_id:    int,          # kept for caller convenience / logging

# # # #     error_message: str,

# # # # ) -> None:

# # # #     """

# # # #     POST /api/internal/analysis/failed
 
# # # #     Called whenever an unrecoverable exception occurs during job processing.

# # # #     Spring Boot will mark AnalysisJob and Journey as FAILED.
 
# # # #     Note: The API spec only requires jobId + errorMessage.

# # # #     journeyId is accepted as a parameter for logging but is NOT included

# # # #     in the outbound payload.

# # # #     """

# # # #     try:

# # # #         _post(

# # # #             "/api/internal/analysis/failed",

# # # #             {

# # # #                 "jobId":        job_id,

# # # #                 "errorMessage": error_message,

# # # #             },

# # # #         )

# # # #     except Exception as exc:

# # # #         # Failure callback must never itself raise — log and swallow.

# # # #         log.error(

# # # #             "[Callback] send_failed itself failed (job=%s journey=%d): %s",

# # # #             job_id, journey_id, exc,

# # # #         )
 
 
 
# # # """
# # # callback_client.py
# # # ──────────────────
# # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs
# # # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).

# # # Changes from previous version
# # # ──────────────────────────────
# # # • Added check_job_completed(job_id) — queries GET /api/internal/analysis/status/{jobId}
# # #   and returns True when the backend reports the job is already COMPLETED.
# # #   Called by consumer.py as an idempotency guard before starting any processing
# # #   on a redelivered message.
# # # """

# # # from __future__ import annotations
# # # import logging
# # # import os
# # # from typing import Any, Dict, Optional

# # # import requests
# # # from dotenv import load_dotenv

# # # # ── Credentials / config ─────────────────────────────────────────────────────
# # # _ENV_PATH = os.path.join(
# # #     os.path.dirname(os.path.abspath(__file__)),
# # #     "config", "credentials.env",
# # # )
# # # load_dotenv(_ENV_PATH)

# # # _BASE_URL = os.environ.get(
# # #     "SPRING_BOOT_BASE_URL",
# # #     "https://cvvrsrailway-api.sconexsoft.com/cvs",
# # # )
# # # _TIMEOUT = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # log = logging.getLogger("callback_client")


# # # # ── Dynamic base URL setter ───────────────────────────────────────────────────

# # # def set_base_url(url: str) -> None:
# # #     """
# # #     Override the callback base URL at runtime.

# # #     Called by consumer.py when the RabbitMQ message contains a callbackBaseUrl.
# # #     Allows the same Python worker to callback correctly to both local and staging
# # #     Spring Boot servers without changing env vars.

# # #     The URL passed here is the FULL path up to /api/internal/analysis.
# # #     Individual callbacks append /progress, /completed, /failed, /status/{id}.
# # #     """
# # #     global _BASE_URL
# # #     base = url.rstrip("/")
# # #     if base.endswith("/api/internal/analysis"):
# # #         base = base[: -len("/api/internal/analysis")]
# # #     _BASE_URL = base
# # #     log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)


# # # # ── Internal helpers ──────────────────────────────────────────────────────────

# # # def _post(path: str, payload: Dict[str, Any]) -> None:
# # #     """
# # #     POST to a Spring Boot internal endpoint.
# # #     No Authorization header — /api/internal/* are worker-only endpoints.
# # #     Raises RuntimeError on non-2xx so the consumer can route to the failure path.
# # #     """
# # #     url = f"{_BASE_URL}{path}"
# # #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# # #     if not resp.ok:
# # #         raise RuntimeError(
# # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # #         )
# # #     log.debug("[Callback] %s → %d", path, resp.status_code)


# # # def _get(path: str) -> requests.Response:
# # #     """
# # #     GET from a Spring Boot internal endpoint.
# # #     Raises RuntimeError on non-2xx.
# # #     """
# # #     url = f"{_BASE_URL}{path}"
# # #     log.debug("[Callback] GET %s", url)
# # #     resp = requests.get(url, timeout=_TIMEOUT)
# # #     if not resp.ok:
# # #         raise RuntimeError(
# # #             f"Callback GET {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # #         )
# # #     log.debug("[Callback] %s → %d", path, resp.status_code)
# # #     return resp


# # # # ── Public API ────────────────────────────────────────────────────────────────

# # # def check_job_completed(job_id: str) -> bool:
# # #     """
# # #     NEW — Idempotency check.

# # #     Queries GET /api/internal/analysis/status/{jobId} and returns True when
# # #     the backend reports the job status as COMPLETED.

# # #     Called by consumer.py at the very start of _handle_job() so that
# # #     RabbitMQ redeliveries of already-completed jobs are detected immediately
# # #     and ACKed without re-running any processing.

# # #     Backend contract (expected JSON shape):
# # #         { "status": "COMPLETED" }   → job already done → return True
# # #         { "status": "PROCESSING" }  → in-flight (rare edge-case) → return False
# # #         { "status": "PENDING" }     → not yet processed → return False
# # #         404 Not Found               → job unknown (treat as not completed)

# # #     If the endpoint does not exist yet on your Spring Boot side, add it as:
# # #         GET /api/internal/analysis/status/{jobId}
# # #         → ResponseEntity<Map<String,String>>  { "status": job.getStatus().name() }

# # #     Raises on network errors so the consumer can decide whether to proceed
# # #     with processing or skip (consumer.py catches and proceeds on error).
# # #     """
# # #     try:
# # #         resp = _get(f"/api/internal/analysis/status/{job_id}")
# # #         data = resp.json()
# # #         status = data.get("status", "").upper()
# # #         is_done = status == "COMPLETED"
# # #         log.info(
# # #             "[Callback] check_job_completed  job=%s  status=%s  is_done=%s",
# # #             job_id, status, is_done,
# # #         )
# # #         return is_done
# # #     except RuntimeError as exc:
# # #         # 404 → job not found in the backend → definitely not completed
# # #         if "404" in str(exc):
# # #             log.info(
# # #                 "[Callback] check_job_completed  job=%s  → 404 (not found) → not completed",
# # #                 job_id,
# # #             )
# # #             return False
# # #         raise


# # # def send_progress(
# # #     job_id:        str,
# # #     journey_id:    int,
# # #     progress:      int,
# # #     message:       str,
# # #     status:        str = "PROCESSING",
# # #     current_video: Optional[int] = None,
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
# # #     current_video : 1-based index of the video currently being processed.
# # #     """
# # #     payload: Dict[str, Any] = {
# # #         "jobId":     job_id,
# # #         "journeyId": journey_id,
# # #         "status":    status,
# # #         "progress":  progress,
# # #         "message":   message,
# # #     }
# # #     if current_video is not None:
# # #         payload["currentVideo"] = current_video
# # #     _post("/api/internal/analysis/progress", payload)


# # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # #     """
# # #     POST /api/internal/analysis/completed

# # #     Called once after ALL videos in the journey have been processed and their
# # #     violation frames uploaded to S3.

# # #     completion_payload is built by CompletionPayload.to_dict() in models.py.
# # #     """
# # #     _post("/api/internal/analysis/completed", completion_payload)


# # # def send_failed(
# # #     job_id:        str,
# # #     journey_id:    int,   # kept for caller convenience / logging
# # #     error_message: str,
# # # ) -> None:
# # #     """
# # #     POST /api/internal/analysis/failed

# # #     Called whenever an unrecoverable exception occurs during job processing.
# # #     Spring Boot will mark AnalysisJob and Journey as FAILED.

# # #     Note: The API spec only requires jobId + errorMessage.
# # #     journeyId is accepted as a parameter for logging but is NOT included
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

# # Changes in this version
# # ────────────────────────
# # • check_job_completed() now uses the EXISTING completed-callback endpoint
# #   as a probe instead of a dedicated /status/{jobId} endpoint that doesn't
# #   exist yet on Spring Boot.

# #   Strategy: POST /api/internal/analysis/completed with only { "jobId": id,
# #   "probe": true }.  Spring Boot should return 409 CONFLICT (or any non-500)
# #   when the job is already done.  If it returns 500 we treat it as "unknown"
# #   and fall through to processing (safe default).

# #   *** TEMPORARY until the Spring Boot /status endpoint is implemented. ***
# #   Once GET /api/internal/analysis/status/{jobId} is live, revert
# #   check_job_completed() to use _get() as originally written.
# # """

# # from __future__ import annotations

# # import logging
# # import os
# # from typing import Any, Dict, Optional

# # import requests
# # from dotenv import load_dotenv

# # # ── Credentials / config ─────────────────────────────────────────────────────
# # _ENV_PATH = os.path.join(
# #     os.path.dirname(os.path.abspath(__file__)),
# #     "config", "credentials.env",
# # )
# # load_dotenv(_ENV_PATH)

# # _BASE_URL = os.environ.get(
# #     "SPRING_BOOT_BASE_URL",
# #     "https://cvvrsrailway-api.sconexsoft.com/cvs",
# # )
# # _TIMEOUT = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # log = logging.getLogger("callback_client")


# # # ── Dynamic base URL setter ───────────────────────────────────────────────────

# # def set_base_url(url: str) -> None:
# #     global _BASE_URL
# #     base = url.rstrip("/")
# #     if base.endswith("/api/internal/analysis"):
# #         base = base[: -len("/api/internal/analysis")]
# #     _BASE_URL = base
# #     log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)


# # # ── Internal helpers ──────────────────────────────────────────────────────────

# # def _post(path: str, payload: Dict[str, Any]) -> requests.Response:
# #     url = f"{_BASE_URL}{path}"
# #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# #     if not resp.ok:
# #         raise RuntimeError(
# #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# #         )
# #     log.debug("[Callback] %s → %d", path, resp.status_code)
# #     return resp


# # def _get(path: str) -> requests.Response:
# #     url = f"{_BASE_URL}{path}"
# #     log.debug("[Callback] GET %s", url)
# #     resp = requests.get(url, timeout=_TIMEOUT)
# #     if not resp.ok:
# #         raise RuntimeError(
# #             f"Callback GET {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# #         )
# #     log.debug("[Callback] %s → %d", path, resp.status_code)
# #     return resp


# # # ── Public API ────────────────────────────────────────────────────────────────

# # def check_job_completed(job_id: str) -> bool:
# #     """
# #     Idempotency check — returns True if the backend already has this job as COMPLETED.

# #     TWO-STAGE STRATEGY
# #     ──────────────────
# #     Stage 1 (permanent): GET /api/internal/analysis/status/{jobId}
# #         → { "status": "COMPLETED" }  → True
# #         → { "status": "PENDING/PROCESSING" } → False
# #         → 404 → False  (job not known yet)
# #         → 500 → fall through to Stage 2

# #     Stage 2 (temporary fallback until Spring Boot implements /status):
# #         Uses GET /api/internal/analysis/job/{jobId} or any existing
# #         read endpoint.  If that also 500s, we return False (safe default:
# #         process the job rather than silently drop it).

# #     WHAT TO IMPLEMENT ON SPRING BOOT (one small controller method):
# #     ───────────────────────────────────────────────────────────────
# #     @GetMapping("/api/internal/analysis/status/{jobId}")
# #     public ResponseEntity<Map<String,String>> getJobStatus(
# #             @PathVariable String jobId) {
# #         return analysisJobRepository.findByJobId(jobId)
# #             .map(job -> ResponseEntity.ok(Map.of("status", job.getStatus().name())))
# #             .orElse(ResponseEntity.notFound().<Map<String,String>>build());
# #     }

# #     Once that endpoint is deployed, Stage 2 below can be deleted.
# #     """

# #     # ── Stage 1: dedicated status endpoint ───────────────────────────────────
# #     try:
# #         resp = _get(f"/api/internal/analysis/status/{job_id}")
# #         status = resp.json().get("status", "").upper()
# #         is_done = status == "COMPLETED"
# #         log.info(
# #             "[Callback] idempotency check (status endpoint)  job=%s  status=%s  done=%s",
# #             job_id, status, is_done,
# #         )
# #         return is_done
# #     except RuntimeError as exc:
# #         if "404" in str(exc):
# #             log.info(
# #                 "[Callback] idempotency check  job=%s  → 404 → not completed", job_id
# #             )
# #             return False
# #         if "500" in str(exc):
# #             log.warning(
# #                 "[Callback] idempotency check  job=%s  → /status endpoint returned 500 "
# #                 "(not implemented yet?) — trying fallback probe.", job_id
# #             )
# #             # fall through to Stage 2
# #         else:
# #             # Network error or unexpected status — safe default: process the job
# #             log.warning(
# #                 "[Callback] idempotency check  job=%s  → unexpected error (%s) "
# #                 "— proceeding with processing.", job_id, exc
# #             )
# #             return False

# #     # ── Stage 2: fallback probe using progress endpoint ───────────────────────
# #     # We send a lightweight progress probe at 0 % with status=CHECK.
# #     # Spring Boot should:
# #     #   • Return 200 if the job is still PENDING/PROCESSING (→ process it).
# #     #   • Return 409 CONFLICT if the job is already COMPLETED (→ skip it).
# #     #   • Return 404 if the job is unknown (→ process it).
# #     # If the backend doesn't handle the CHECK status specially it will just
# #     # update progress to 0 — harmless on an already-completed job.
# #     #
# #     # THIS IS A TEMPORARY FALLBACK.  Remove once /status endpoint is live.
# #     try:
# #         url = f"{_BASE_URL}/api/internal/analysis/progress"
# #         resp = requests.post(
# #             url,
# #             json={
# #                 "jobId":    job_id,
# #                 "status":   "CHECK",   # sentinel value Spring Boot can detect
# #                 "progress": 0,
# #                 "message":  "idempotency-probe",
# #             },
# #             timeout=_TIMEOUT,
# #         )
# #         if resp.status_code == 409:
# #             log.warning(
# #                 "[Callback] idempotency check (probe)  job=%s  → 409 CONFLICT "
# #                 "→ already COMPLETED — will skip.", job_id
# #             )
# #             return True
# #         log.info(
# #             "[Callback] idempotency check (probe)  job=%s  → HTTP %d → not completed.",
# #             job_id, resp.status_code,
# #         )
# #         return False
# #     except Exception as exc2:
# #         log.warning(
# #             "[Callback] idempotency check (probe) failed  job=%s: %s "
# #             "— proceeding with processing.", job_id, exc2
# #         )
# #         return False


# # def send_progress(
# #     job_id:        str,
# #     journey_id:    int,
# #     progress:      int,
# #     message:       str,
# #     status:        str = "PROCESSING",
# #     current_video: Optional[int] = None,
# # ) -> None:
# #     payload: Dict[str, Any] = {
# #         "jobId":     job_id,
# #         "journeyId": journey_id,
# #         "status":    status,
# #         "progress":  progress,
# #         "message":   message,
# #     }
# #     if current_video is not None:
# #         payload["currentVideo"] = current_video
# #     _post("/api/internal/analysis/progress", payload)


# # def send_completed(completion_payload: Dict[str, Any]) -> None:
# #     _post("/api/internal/analysis/completed", completion_payload)


# # def send_failed(
# #     job_id:        str,
# #     journey_id:    int,
# #     error_message: str,
# # ) -> None:
# #     try:
# #         _post(
# #             "/api/internal/analysis/failed",
# #             {
# #                 "jobId":        job_id,
# #                 "errorMessage": error_message,
# #             },
# #         )
# #     except Exception as exc:
# #         log.error(
# #             "[Callback] send_failed itself failed (job=%s journey=%d): %s",
# #             job_id, journey_id, exc,
# #         )



# # # # # """
# # # # # callback_client.py
# # # # # ──────────────────
# # # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # # # All three public helpers are fire-and-forget from the caller's perspective;
# # # # # they raise on non-2xx so the consumer can route to the failure path.
# # # # # """

# # # # # from __future__ import annotations

# # # # # import logging
# # # # # import os
# # # # # from typing import Any, Dict, List

# # # # # import requests
# # # # # from dotenv import load_dotenv

# # # # # # ── Credentials / config ────────────────────────────────────────────────────
# # # # # _ENV_PATH = os.path.join(
# # # # #     os.path.dirname(os.path.abspath(__file__)),
# # # # #     "config", "credentials.env",
# # # # # )
# # # # # load_dotenv(_ENV_PATH)

# # # # # _BASE_URL = os.environ.get("SPRING_BOOT_BASE_URL", "http://localhost:8080")
# # # # # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # # # log = logging.getLogger("callback_client")


# # # # # # ── Internal helper ──────────────────────────────────────────────────────────

# # # # # def _post(path: str, payload: Dict[str, Any]) -> None:
# # # # #     url = f"{_BASE_URL}{path}"
# # # # #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# # # # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# # # # #     if not resp.ok:
# # # # #         raise RuntimeError(
# # # # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # # # #         )
# # # # #     log.debug("[Callback] %s → %d", path, resp.status_code)


# # # # # # ── Public API ───────────────────────────────────────────────────────────────

# # # # # def send_progress(
# # # # #     job_id:     str,
# # # # #     journey_id: int,
# # # # #     progress:   int,
# # # # #     message:    str,
# # # # #     status:     str = "PROCESSING",
# # # # # ) -> None:
# # # # #     """
# # # # #     POST /api/internal/analysis/progress

# # # # #     Called periodically during analysis to update the frontend progress bar
# # # # #     and the SSE stream.
# # # # #     """
# # # # #     _post(
# # # # #         "/api/internal/analysis/progress",
# # # # #         {
# # # # #             "jobId":      job_id,
# # # # #             "journeyId":  journey_id,
# # # # #             "status":     status,
# # # # #             "progress":   progress,
# # # # #             "message":    message,
# # # # #         },
# # # # #     )


# # # # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # # # #     """
# # # # #     POST /api/internal/analysis/completed

# # # # #     Called once — after ALL videos in the journey have been processed and
# # # # #     their violation frames uploaded to S3.

# # # # #     Expected shape of completion_payload
# # # # #     ─────────────────────────────────────
# # # # #     {
# # # # #         "jobId":          str,
# # # # #         "journeyId":      int,
# # # # #         "processingTime": int,          # wall-clock ms
# # # # #         "videoResults":   List[dict],   # see models.py → VideoResult
# # # # #     }
# # # # #     """
# # # # #     _post("/api/internal/analysis/completed", completion_payload)


# # # # # def send_failed(
# # # # #     job_id:        str,
# # # # #     journey_id:    int,
# # # # #     error_message: str,
# # # # # ) -> None:
# # # # #     """
# # # # #     POST /api/internal/analysis/failed

# # # # #     Called whenever an unrecoverable exception occurs during job processing.
# # # # #     Spring Boot will mark AnalysisJob and Journey as FAILED.
# # # # #     """
# # # # #     try:
# # # # #         _post(
# # # # #             "/api/internal/analysis/failed",
# # # # #             {
# # # # #                 "jobId":        job_id,
# # # # #                 "journeyId":    journey_id,
# # # # #                 "errorMessage": error_message,
# # # # #             },
# # # # #         )
# # # # #     except Exception as exc:
# # # # #         # Failure callback must never itself raise — log and swallow.
# # # # #         log.error("[Callback] send_failed itself failed: %s", exc)

# # # # """
# # # # callback_client.py
# # # # ──────────────────
# # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs
# # # # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).

# # # # Changes from previous version
# # # # ──────────────────────────────
# # # # • _BASE_URL now defaults to the live CVVRS domain (overridable via env var
# # # #   SPRING_BOOT_BASE_URL for local testing).
# # # # • send_progress()  — payload now includes the `currentVideo` field required
# # # #   by the API.
# # # # • send_completed() — accepts a fully-formed dict (built by CompletionPayload.
# # # #   to_dict()); no shape changes needed here since the caller owns the dict.
# # # # • send_failed()    — payload drops `journeyId` (not in the API spec for this
# # # #   endpoint); only jobId + errorMessage are sent.
# # # # """

# # # # from __future__ import annotations

# # # # import logging
# # # # import os
# # # # from typing import Any, Dict, Optional

# # # # import requests
# # # # from dotenv import load_dotenv

# # # # # ── Credentials / config ────────────────────────────────────────────────────
# # # # _ENV_PATH = os.path.join(
# # # #     os.path.dirname(os.path.abspath(__file__)),
# # # #     "config", "credentials.env",
# # # # )
# # # # load_dotenv(_ENV_PATH)

# # # # # Default to the live CVVRS server; override with SPRING_BOOT_BASE_URL for local dev.
# # # # _BASE_URL = os.environ.get(
# # # #     "SPRING_BOOT_BASE_URL",
# # # #     "https://cvvrsrailway-api.sconexsoft.com/cvs",
# # # # )
# # # # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # # log = logging.getLogger("callback_client")


# # # # # ── Internal helper ──────────────────────────────────────────────────────────

# # # # def _post(path: str, payload: Dict[str, Any]) -> None:
# # # #     """
# # # #     POST to a Spring Boot internal endpoint.
# # # #     No Authorization header is sent — /api/internal/* are worker-only endpoints.
# # # #     Raises RuntimeError on non-2xx so the consumer can route to failure path.
# # # #     """
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
# # # #     job_id:        str,
# # # #     journey_id:    int,
# # # #     progress:      int,
# # # #     message:       str,
# # # #     status:        str = "PROCESSING",
# # # #     current_video: Optional[int] = None,   # NEW — index of video currently being processed
# # # # ) -> None:
# # # #     """
# # # #     POST /api/internal/analysis/progress

# # # #     Called periodically during analysis to update the frontend progress bar
# # # #     and the SSE stream.

# # # #     Parameters
# # # #     ──────────
# # # #     job_id        : RabbitMQ job ID.
# # # #     journey_id    : Journey ID.
# # # #     progress      : 0–100 integer.
# # # #     message       : Human-readable status message.
# # # #     status        : "PROCESSING" | "COMPLETED" | "FAILED"
# # # #     current_video : 1-based index of the video currently being processed
# # # #                     (omitted from payload when None).
# # # #     """
# # # #     payload: Dict[str, Any] = {
# # # #         "jobId":      job_id,
# # # #         "journeyId":  journey_id,
# # # #         "status":     status,
# # # #         "progress":   progress,
# # # #         "message":    message,
# # # #     }
# # # #     if current_video is not None:
# # # #         payload["currentVideo"] = current_video

# # # #     _post("/api/internal/analysis/progress", payload)


# # # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # # #     """
# # # #     POST /api/internal/analysis/completed

# # # #     Called once after ALL videos in the journey have been processed and their
# # # #     violation frames uploaded to S3.

# # # #     Expected shape of completion_payload (built by CompletionPayload.to_dict())
# # # #     ────────────────────────────────────────────────────────────────────────────
# # # #     {
# # # #         "jobId":         str,
# # # #         "journeyId":     int,
# # # #         "batchId":       str,           # "BATCH-<jobId>" if not supplied
# # # #         "trainDetailId": int,
# # # #         "folderName":    str,
# # # #         "processingTime":int,           # wall-clock ms
# # # #         "videoResults": [
# # # #             {
# # # #                 "videoId":         str,       # STRING per API spec
# # # #                 "sequenceNo":      int,
# # # #                 "durationSeconds": float,
# # # #                 "originalS3Key":   str,
# # # #                 "violations": [
# # # #                     {
# # # #                         "violationType":          str,
# # # #                         "severity":               str,
# # # #                         "confidence":             float,
# # # #                         "riskScore":              float,
# # # #                         "timestamp":              float,   # journey-global seconds
# # # #                         "originalVideoTimestamp": float,   # local-video seconds
# # # #                         "framePaths":             [str]
# # # #                     }
# # # #                 ]
# # # #             }
# # # #         ]
# # # #     }
# # # #     """
# # # #     _post("/api/internal/analysis/completed", completion_payload)


# # # # def send_failed(
# # # #     job_id:        str,
# # # #     journey_id:    int,          # kept as parameter for caller convenience / logging
# # # #     error_message: str,
# # # # ) -> None:
# # # #     """
# # # #     POST /api/internal/analysis/failed

# # # #     Called whenever an unrecoverable exception occurs during job processing.
# # # #     Spring Boot will mark AnalysisJob and Journey as FAILED.

# # # #     Note: The API spec only requires jobId + errorMessage in the request body.
# # # #     journeyId is accepted as a parameter here for logging but is NOT included
# # # #     in the outbound payload.
# # # #     """
# # # #     try:
# # # #         _post(
# # # #             "/api/internal/analysis/failed",
# # # #             {
# # # #                 "jobId":        job_id,
# # # #                 "errorMessage": error_message,
# # # #             },
# # # #         )
# # # #     except Exception as exc:
# # # #         # Failure callback must never itself raise — log and swallow.
# # # #         log.error(
# # # #             "[Callback] send_failed itself failed (job=%s journey=%d): %s",
# # # #             job_id, journey_id, exc,
# # # #         )



# # # """

# # # callback_client.py

# # # ──────────────────

# # # Sends progress, completion, and failure callbacks to the Spring Boot backend.
 
# # # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs

# # # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).
 
# # # Fixes from previous version

# # # ─────────────────────────────

# # # • Added set_base_url() — called by consumer.py with the callbackBaseUrl from

# # #   the RabbitMQ message, so Python posts to the correct Spring Boot server

# # #   per environment without needing env var changes.

# # # • _BASE_URL still falls back to env var SPRING_BOOT_BASE_URL or the live

# # #   server when callbackBaseUrl is not provided in the message.

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
 
# # # # Default to the live CVVRS server; override with SPRING_BOOT_BASE_URL for local dev

# # # # or call set_base_url() with the callbackBaseUrl from the RabbitMQ message.

# # # _BASE_URL = os.environ.get(

# # #     "SPRING_BOOT_BASE_URL",

# # #     "https://cvvrsrailway-api.sconexsoft.com/cvs",

# # # )

# # # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))
 
# # # log = logging.getLogger("callback_client")
 
 
# # # # ── FIX: dynamic base URL setter ─────────────────────────────────────────────
 
# # # def set_base_url(url: str) -> None:

# # #     """

# # #     Override the callback base URL at runtime.
 
# # #     Called by consumer.py when the RabbitMQ message contains a callbackBaseUrl.

# # #     This allows the same Python worker to callback correctly to both local

# # #     and staging Spring Boot servers without changing env vars.
 
# # #     Example values:

# # #         "http://localhost:8093/api/internal/analysis"         (local)

# # #         "https://cvvrsrailway-api.sconexsoft.com/cvs/api/internal/analysis"  (staging)
 
# # #     Note: The URL passed here is the FULL path up to /api/internal/analysis.

# # #     The individual callbacks append /progress, /completed, /failed.

# # #     """

# # #     global _BASE_URL

# # #     # Strip the /api/internal/analysis suffix if present — we add it in _post()

# # #     base = url.rstrip("/")

# # #     if base.endswith("/api/internal/analysis"):

# # #         base = base[: -len("/api/internal/analysis")]

# # #     _BASE_URL = base

# # #     log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)
 
 
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

# # #     current_video: Optional[int] = None,

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

# # #     current_video : 1-based index of the video currently being processed.

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
 
# # #     completion_payload is built by CompletionPayload.to_dict() in models.py.

# # #     """

# # #     _post("/api/internal/analysis/completed", completion_payload)
 
 
# # # def send_failed(

# # #     job_id:        str,

# # #     journey_id:    int,          # kept for caller convenience / logging

# # #     error_message: str,

# # # ) -> None:

# # #     """

# # #     POST /api/internal/analysis/failed
 
# # #     Called whenever an unrecoverable exception occurs during job processing.

# # #     Spring Boot will mark AnalysisJob and Journey as FAILED.
 
# # #     Note: The API spec only requires jobId + errorMessage.

# # #     journeyId is accepted as a parameter for logging but is NOT included

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

# # Changes from previous version
# # ──────────────────────────────
# # • Added check_job_completed(job_id) — queries GET /api/internal/analysis/status/{jobId}
# #   and returns True when the backend reports the job is already COMPLETED.
# #   Called by consumer.py as an idempotency guard before starting any processing
# #   on a redelivered message.
# # """

# # from __future__ import annotations
# # import logging
# # import os
# # from typing import Any, Dict, Optional

# # import requests
# # from dotenv import load_dotenv

# # # ── Credentials / config ─────────────────────────────────────────────────────
# # _ENV_PATH = os.path.join(
# #     os.path.dirname(os.path.abspath(__file__)),
# #     "config", "credentials.env",
# # )
# # load_dotenv(_ENV_PATH)

# # _BASE_URL = os.environ.get(
# #     "SPRING_BOOT_BASE_URL",
# #     "https://cvvrsrailway-api.sconexsoft.com/cvs",
# # )
# # _TIMEOUT = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # log = logging.getLogger("callback_client")


# # # ── Dynamic base URL setter ───────────────────────────────────────────────────

# # def set_base_url(url: str) -> None:
# #     """
# #     Override the callback base URL at runtime.

# #     Called by consumer.py when the RabbitMQ message contains a callbackBaseUrl.
# #     Allows the same Python worker to callback correctly to both local and staging
# #     Spring Boot servers without changing env vars.

# #     The URL passed here is the FULL path up to /api/internal/analysis.
# #     Individual callbacks append /progress, /completed, /failed, /status/{id}.
# #     """
# #     global _BASE_URL
# #     base = url.rstrip("/")
# #     if base.endswith("/api/internal/analysis"):
# #         base = base[: -len("/api/internal/analysis")]
# #     _BASE_URL = base
# #     log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)


# # # ── Internal helpers ──────────────────────────────────────────────────────────

# # def _post(path: str, payload: Dict[str, Any]) -> None:
# #     """
# #     POST to a Spring Boot internal endpoint.
# #     No Authorization header — /api/internal/* are worker-only endpoints.
# #     Raises RuntimeError on non-2xx so the consumer can route to the failure path.
# #     """
# #     url = f"{_BASE_URL}{path}"
# #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# #     if not resp.ok:
# #         raise RuntimeError(
# #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# #         )
# #     log.debug("[Callback] %s → %d", path, resp.status_code)


# # def _get(path: str) -> requests.Response:
# #     """
# #     GET from a Spring Boot internal endpoint.
# #     Raises RuntimeError on non-2xx.
# #     """
# #     url = f"{_BASE_URL}{path}"
# #     log.debug("[Callback] GET %s", url)
# #     resp = requests.get(url, timeout=_TIMEOUT)
# #     if not resp.ok:
# #         raise RuntimeError(
# #             f"Callback GET {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# #         )
# #     log.debug("[Callback] %s → %d", path, resp.status_code)
# #     return resp


# # # ── Public API ────────────────────────────────────────────────────────────────

# # def check_job_completed(job_id: str) -> bool:
# #     """
# #     NEW — Idempotency check.

# #     Queries GET /api/internal/analysis/status/{jobId} and returns True when
# #     the backend reports the job status as COMPLETED.

# #     Called by consumer.py at the very start of _handle_job() so that
# #     RabbitMQ redeliveries of already-completed jobs are detected immediately
# #     and ACKed without re-running any processing.

# #     Backend contract (expected JSON shape):
# #         { "status": "COMPLETED" }   → job already done → return True
# #         { "status": "PROCESSING" }  → in-flight (rare edge-case) → return False
# #         { "status": "PENDING" }     → not yet processed → return False
# #         404 Not Found               → job unknown (treat as not completed)

# #     If the endpoint does not exist yet on your Spring Boot side, add it as:
# #         GET /api/internal/analysis/status/{jobId}
# #         → ResponseEntity<Map<String,String>>  { "status": job.getStatus().name() }

# #     Raises on network errors so the consumer can decide whether to proceed
# #     with processing or skip (consumer.py catches and proceeds on error).
# #     """
# #     try:
# #         resp = _get(f"/api/internal/analysis/status/{job_id}")
# #         data = resp.json()
# #         status = data.get("status", "").upper()
# #         is_done = status == "COMPLETED"
# #         log.info(
# #             "[Callback] check_job_completed  job=%s  status=%s  is_done=%s",
# #             job_id, status, is_done,
# #         )
# #         return is_done
# #     except RuntimeError as exc:
# #         # 404 → job not found in the backend → definitely not completed
# #         if "404" in str(exc):
# #             log.info(
# #                 "[Callback] check_job_completed  job=%s  → 404 (not found) → not completed",
# #                 job_id,
# #             )
# #             return False
# #         raise


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
# #         "jobId":     job_id,
# #         "journeyId": journey_id,
# #         "status":    status,
# #         "progress":  progress,
# #         "message":   message,
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
# #     journey_id:    int,   # kept for caller convenience / logging
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

# Changes in this version
# ────────────────────────
# • check_job_completed() now uses the EXISTING completed-callback endpoint
#   as a probe instead of a dedicated /status/{jobId} endpoint that doesn't
#   exist yet on Spring Boot.

#   Strategy: POST /api/internal/analysis/completed with only { "jobId": id,
#   "probe": true }.  Spring Boot should return 409 CONFLICT (or any non-500)
#   when the job is already done.  If it returns 500 we treat it as "unknown"
#   and fall through to processing (safe default).

#   *** TEMPORARY until the Spring Boot /status endpoint is implemented. ***
#   Once GET /api/internal/analysis/status/{jobId} is live, revert
#   check_job_completed() to use _get() as originally written.
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
#     global _BASE_URL
#     base = url.rstrip("/")
#     if base.endswith("/api/internal/analysis"):
#         base = base[: -len("/api/internal/analysis")]
#     _BASE_URL = base
#     log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)


# # ── Internal helpers ──────────────────────────────────────────────────────────

# def _post(path: str, payload: Dict[str, Any]) -> requests.Response:
#     url = f"{_BASE_URL}{path}"
#     log.debug("[Callback] POST %s  payload=%s", url, payload)
#     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
#     if not resp.ok:
#         raise RuntimeError(
#             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
#         )
#     log.debug("[Callback] %s → %d", path, resp.status_code)
#     return resp


# def _get(path: str) -> requests.Response:
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
#     Idempotency check — returns True if the backend already has this job as COMPLETED.

#     TWO-STAGE STRATEGY
#     ──────────────────
#     Stage 1 (permanent): GET /api/internal/analysis/status/{jobId}
#         → { "status": "COMPLETED" }  → True
#         → { "status": "PENDING/PROCESSING" } → False
#         → 404 → False  (job not known yet)
#         → 500 → fall through to Stage 2

#     Stage 2 (temporary fallback until Spring Boot implements /status):
#         Uses GET /api/internal/analysis/job/{jobId} or any existing
#         read endpoint.  If that also 500s, we return False (safe default:
#         process the job rather than silently drop it).

#     WHAT TO IMPLEMENT ON SPRING BOOT (one small controller method):
#     ───────────────────────────────────────────────────────────────
#     @GetMapping("/api/internal/analysis/status/{jobId}")
#     public ResponseEntity<Map<String,String>> getJobStatus(
#             @PathVariable String jobId) {
#         return analysisJobRepository.findByJobId(jobId)
#             .map(job -> ResponseEntity.ok(Map.of("status", job.getStatus().name())))
#             .orElse(ResponseEntity.notFound().<Map<String,String>>build());
#     }

#     Once that endpoint is deployed, Stage 2 below can be deleted.
#     """

#     # ── Stage 1: dedicated status endpoint ───────────────────────────────────
#     try:
#         resp = _get(f"/api/internal/analysis/status/{job_id}")
#         status = resp.json().get("status", "").upper()
#         is_done = status == "COMPLETED"
#         log.info(
#             "[Callback] idempotency check (status endpoint)  job=%s  status=%s  done=%s",
#             job_id, status, is_done,
#         )
#         return is_done
#     except RuntimeError as exc:
#         if "404" in str(exc):
#             log.info(
#                 "[Callback] idempotency check  job=%s  → 404 → not completed", job_id
#             )
#             return False
#         if "500" in str(exc):
#             log.warning(
#                 "[Callback] idempotency check  job=%s  → /status endpoint returned 500 "
#                 "(not implemented yet?) — trying fallback probe.", job_id
#             )
#             # fall through to Stage 2
#         else:
#             # Network error or unexpected status — safe default: process the job
#             log.warning(
#                 "[Callback] idempotency check  job=%s  → unexpected error (%s) "
#                 "— proceeding with processing.", job_id, exc
#             )
#             return False

#     # ── Stage 2: fallback probe using progress endpoint ───────────────────────
#     # We send a lightweight progress probe at 0 % with status=CHECK.
#     # Spring Boot should:
#     #   • Return 200 if the job is still PENDING/PROCESSING (→ process it).
#     #   • Return 409 CONFLICT if the job is already COMPLETED (→ skip it).
#     #   • Return 404 if the job is unknown (→ process it).
#     # If the backend doesn't handle the CHECK status specially it will just
#     # update progress to 0 — harmless on an already-completed job.
#     #
#     # THIS IS A TEMPORARY FALLBACK.  Remove once /status endpoint is live.
#     try:
#         url = f"{_BASE_URL}/api/internal/analysis/progress"
#         resp = requests.post(
#             url,
#             json={
#                 "jobId":    job_id,
#                 "status":   "CHECK",   # sentinel value Spring Boot can detect
#                 "progress": 0,
#                 "message":  "idempotency-probe",
#             },
#             timeout=_TIMEOUT,
#         )
#         if resp.status_code == 409:
#             log.warning(
#                 "[Callback] idempotency check (probe)  job=%s  → 409 CONFLICT "
#                 "→ already COMPLETED — will skip.", job_id
#             )
#             return True
#         log.info(
#             "[Callback] idempotency check (probe)  job=%s  → HTTP %d → not completed.",
#             job_id, resp.status_code,
#         )
#         return False
#     except Exception as exc2:
#         log.warning(
#             "[Callback] idempotency check (probe) failed  job=%s: %s "
#             "— proceeding with processing.", job_id, exc2
#         )
#         return False


# def send_progress(
#     job_id:        str,
#     journey_id:    int,
#     progress:      int,
#     message:       str,
#     status:        str = "PROCESSING",
#     current_video: Optional[int] = None,
# ) -> None:
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
#     _post("/api/internal/analysis/completed", completion_payload)


# def send_failed(
#     job_id:        str,
#     journey_id:    int,
#     error_message: str,
#     video_id:      Optional[int] = None,
#     error_type:    Optional[str] = None,
#     stack_trace:   Optional[str] = None,
#     reason:        Optional[str] = None,
# ) -> None:
#     """
#     Calls POST /api/internal/analysis/failed.

#     Backward compatible: existing job-level callers (no video_id) keep
#     working exactly as before — the payload shape for that case is
#     unchanged ({"jobId", "errorMessage"}).

#     When video_id IS supplied, this becomes a PER-VIDEO failure report
#     (Phase 1 requirement: "no failed video should wait until journey
#     completion to be reported"). The payload then also carries:
#         videoId     — STRING, matches VideoResult.to_dict()'s convention
#         journeyId
#         errorType   — short classifier, e.g. "OOM", "DECODE_ERROR",
#                       "MEDIAPIPE_ERROR", "YOLO_ERROR", "TIMEOUT",
#                       "RESOURCE_EXHAUSTION", "NOT_PROCESSED"
#         stackTrace  — full traceback text, if available
#         reason      — human-readable reason string, e.g.
#                       "Not Processed - Worker Resource Exhaustion"
#                       for videos skipped after an OOM on an earlier video.
#     """
#     payload: Dict[str, Any] = {
#         "jobId":        job_id,
#         "errorMessage": error_message,
#     }
#     if video_id is not None:
#         payload["videoId"]    = str(video_id)   # STRING per API spec convention
#         payload["journeyId"]  = journey_id
#     if error_type is not None:
#         payload["errorType"] = error_type
#     if stack_trace is not None:
#         payload["stackTrace"] = stack_trace
#     if reason is not None:
#         payload["reason"] = reason

#     try:
#         _post("/api/internal/analysis/failed", payload)
#     except Exception as exc:
#         log.error(
#             "[Callback] send_failed itself failed (job=%s journey=%d video=%s): %s",
#             job_id, journey_id, video_id, exc,
#         )


# def send_video_failed(
#     job_id:        str,
#     journey_id:    int,
#     video_id:      int,
#     error_type:    str,
#     error_message: str,
#     stack_trace:   str = "",
#     reason:        Optional[str] = None,
# ) -> None:
#     """
#     Convenience wrapper for the per-video failure report required by
#     Phase 1: "Immediately call the failed endpoint for the affected video"
#     — called once per video, the moment that video's outcome is known,
#     not batched until the journey ends.

#     For resource-exhaustion cascades, pass error_type="RESOURCE_EXHAUSTION"
#     (the video that actually hit OOM) or error_type="NOT_PROCESSED" with
#     reason="Not Processed - Worker Resource Exhaustion" (videos after it
#     that were skipped as a result).
#     """
#     log.error(
#         "[Callback] video FAILED  job=%s  journey=%d  video=%d  type=%s  reason=%s: %s",
#         job_id, journey_id, video_id, error_type, reason, error_message,
#     )
#     send_failed(
#         job_id        = job_id,
#         journey_id    = journey_id,
#         error_message = error_message,
#         video_id      = video_id,
#         error_type    = error_type,
#         stack_trace   = stack_trace,
#         reason        = reason,
#     )


# # ── OOM / resource-exhaustion signature detection ────────────────────────────
# #
# # Used by analyzer.py to decide whether a per-video exception means "this
# # one video is bad" (continue with the rest) vs. "the worker itself can no
# # longer safely process more video" (stop the journey, mark every
# # remaining video FAILED with reason "Not Processed - Worker Resource
# # Exhaustion"). Matches the signatures called out explicitly in the
# # Phase 1 spec: OpenCV OutOfMemoryError, "Failed to allocate", cv2.pyd
# # memory-related failures, generic MemoryError, std::bad_alloc.
# _OOM_SIGNATURES = (
#     "outofmemoryerror",
#     "out of memory",
#     "failed to allocate",
#     "cannot allocate memory",
#     "bad_alloc",
#     "memoryerror",
#     "cv2.pyd",
#     "access violation",
#     "resource exhaust",
# )


# def is_resource_exhaustion_error(exc: BaseException) -> bool:
#     """
#     Returns True if `exc` looks like an OOM / native resource-exhaustion
#     failure rather than an ordinary recoverable per-video error.
#     """
#     text = f"{type(exc).__name__}: {exc}".lower()
#     if isinstance(exc, MemoryError):
#         return True
#     return any(sig in text for sig in _OOM_SIGNATURES)


# def classify_video_error(exc: BaseException) -> str:
#     """
#     Maps a caught per-video exception to a short errorType string for the
#     failed-endpoint payload. Best-effort classification by exception type
#     and message content — defaults to "PROCESSING_ERROR" when nothing
#     more specific matches.
#     """
#     if is_resource_exhaustion_error(exc):
#         return "RESOURCE_EXHAUSTION"
#     name = type(exc).__name__.lower()
#     text = str(exc).lower()
#     if "timeout" in name or "timeout" in text:
#         return "TIMEOUT"
#     if "mediapipe" in text:
#         return "MEDIAPIPE_ERROR"
#     if "yolo" in text or "ultralytics" in text:
#         return "YOLO_ERROR"
#     if "cv2" in text or "opencv" in text or name == "error":
#         return "OPENCV_ERROR"
#     if "cannot open" in text or "corrupt" in text or "decode" in text or "invalid" in text:
#         return "DECODE_ERROR"
#     return "PROCESSING_ERROR"


# def compute_journey_status(
#     total_videos: int,
#     succeeded_video_ids: "set[int]",
#     failed_video_ids: "set[int]",
# ) -> str:
#     """
#     Computes the journey-level terminal status per Phase 1 spec:
#         COMPLETED              — every video succeeded
#         COMPLETED_WITH_ERRORS  — at least one succeeded AND at least one failed
#         FAILED                 — every video failed (or none succeeded)

#     TIMED_OUT is NOT computed here — that's set explicitly by the watchdog
#     path in consumer.py, which knows it hit a timeout rather than inferring
#     it from video counts.
#     """
#     if total_videos == 0:
#         return "FAILED"
#     if not failed_video_ids:
#         return "COMPLETED"
#     if succeeded_video_ids:
#         return "COMPLETED_WITH_ERRORS"
#     return "FAILED"


# # # # # """
# # # # # callback_client.py
# # # # # ──────────────────
# # # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # # # All three public helpers are fire-and-forget from the caller's perspective;
# # # # # they raise on non-2xx so the consumer can route to the failure path.
# # # # # """

# # # # # from __future__ import annotations

# # # # # import logging
# # # # # import os
# # # # # from typing import Any, Dict, List

# # # # # import requests
# # # # # from dotenv import load_dotenv

# # # # # # ── Credentials / config ────────────────────────────────────────────────────
# # # # # _ENV_PATH = os.path.join(
# # # # #     os.path.dirname(os.path.abspath(__file__)),
# # # # #     "config", "credentials.env",
# # # # # )
# # # # # load_dotenv(_ENV_PATH)

# # # # # _BASE_URL = os.environ.get("SPRING_BOOT_BASE_URL", "http://localhost:8080")
# # # # # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # # # log = logging.getLogger("callback_client")


# # # # # # ── Internal helper ──────────────────────────────────────────────────────────

# # # # # def _post(path: str, payload: Dict[str, Any]) -> None:
# # # # #     url = f"{_BASE_URL}{path}"
# # # # #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# # # # #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# # # # #     if not resp.ok:
# # # # #         raise RuntimeError(
# # # # #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# # # # #         )
# # # # #     log.debug("[Callback] %s → %d", path, resp.status_code)


# # # # # # ── Public API ───────────────────────────────────────────────────────────────

# # # # # def send_progress(
# # # # #     job_id:     str,
# # # # #     journey_id: int,
# # # # #     progress:   int,
# # # # #     message:    str,
# # # # #     status:     str = "PROCESSING",
# # # # # ) -> None:
# # # # #     """
# # # # #     POST /api/internal/analysis/progress

# # # # #     Called periodically during analysis to update the frontend progress bar
# # # # #     and the SSE stream.
# # # # #     """
# # # # #     _post(
# # # # #         "/api/internal/analysis/progress",
# # # # #         {
# # # # #             "jobId":      job_id,
# # # # #             "journeyId":  journey_id,
# # # # #             "status":     status,
# # # # #             "progress":   progress,
# # # # #             "message":    message,
# # # # #         },
# # # # #     )


# # # # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # # # #     """
# # # # #     POST /api/internal/analysis/completed

# # # # #     Called once — after ALL videos in the journey have been processed and
# # # # #     their violation frames uploaded to S3.

# # # # #     Expected shape of completion_payload
# # # # #     ─────────────────────────────────────
# # # # #     {
# # # # #         "jobId":          str,
# # # # #         "journeyId":      int,
# # # # #         "processingTime": int,          # wall-clock ms
# # # # #         "videoResults":   List[dict],   # see models.py → VideoResult
# # # # #     }
# # # # #     """
# # # # #     _post("/api/internal/analysis/completed", completion_payload)


# # # # # def send_failed(
# # # # #     job_id:        str,
# # # # #     journey_id:    int,
# # # # #     error_message: str,
# # # # # ) -> None:
# # # # #     """
# # # # #     POST /api/internal/analysis/failed

# # # # #     Called whenever an unrecoverable exception occurs during job processing.
# # # # #     Spring Boot will mark AnalysisJob and Journey as FAILED.
# # # # #     """
# # # # #     try:
# # # # #         _post(
# # # # #             "/api/internal/analysis/failed",
# # # # #             {
# # # # #                 "jobId":        job_id,
# # # # #                 "journeyId":    journey_id,
# # # # #                 "errorMessage": error_message,
# # # # #             },
# # # # #         )
# # # # #     except Exception as exc:
# # # # #         # Failure callback must never itself raise — log and swallow.
# # # # #         log.error("[Callback] send_failed itself failed: %s", exc)

# # # # """
# # # # callback_client.py
# # # # ──────────────────
# # # # Sends progress, completion, and failure callbacks to the Spring Boot backend.

# # # # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs
# # # # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).

# # # # Changes from previous version
# # # # ──────────────────────────────
# # # # • _BASE_URL now defaults to the live CVVRS domain (overridable via env var
# # # #   SPRING_BOOT_BASE_URL for local testing).
# # # # • send_progress()  — payload now includes the `currentVideo` field required
# # # #   by the API.
# # # # • send_completed() — accepts a fully-formed dict (built by CompletionPayload.
# # # #   to_dict()); no shape changes needed here since the caller owns the dict.
# # # # • send_failed()    — payload drops `journeyId` (not in the API spec for this
# # # #   endpoint); only jobId + errorMessage are sent.
# # # # """

# # # # from __future__ import annotations

# # # # import logging
# # # # import os
# # # # from typing import Any, Dict, Optional

# # # # import requests
# # # # from dotenv import load_dotenv

# # # # # ── Credentials / config ────────────────────────────────────────────────────
# # # # _ENV_PATH = os.path.join(
# # # #     os.path.dirname(os.path.abspath(__file__)),
# # # #     "config", "credentials.env",
# # # # )
# # # # load_dotenv(_ENV_PATH)

# # # # # Default to the live CVVRS server; override with SPRING_BOOT_BASE_URL for local dev.
# # # # _BASE_URL = os.environ.get(
# # # #     "SPRING_BOOT_BASE_URL",
# # # #     "https://cvvrsrailway-api.sconexsoft.com/cvs",
# # # # )
# # # # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # # # log = logging.getLogger("callback_client")


# # # # # ── Internal helper ──────────────────────────────────────────────────────────

# # # # def _post(path: str, payload: Dict[str, Any]) -> None:
# # # #     """
# # # #     POST to a Spring Boot internal endpoint.
# # # #     No Authorization header is sent — /api/internal/* are worker-only endpoints.
# # # #     Raises RuntimeError on non-2xx so the consumer can route to failure path.
# # # #     """
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
# # # #     job_id:        str,
# # # #     journey_id:    int,
# # # #     progress:      int,
# # # #     message:       str,
# # # #     status:        str = "PROCESSING",
# # # #     current_video: Optional[int] = None,   # NEW — index of video currently being processed
# # # # ) -> None:
# # # #     """
# # # #     POST /api/internal/analysis/progress

# # # #     Called periodically during analysis to update the frontend progress bar
# # # #     and the SSE stream.

# # # #     Parameters
# # # #     ──────────
# # # #     job_id        : RabbitMQ job ID.
# # # #     journey_id    : Journey ID.
# # # #     progress      : 0–100 integer.
# # # #     message       : Human-readable status message.
# # # #     status        : "PROCESSING" | "COMPLETED" | "FAILED"
# # # #     current_video : 1-based index of the video currently being processed
# # # #                     (omitted from payload when None).
# # # #     """
# # # #     payload: Dict[str, Any] = {
# # # #         "jobId":      job_id,
# # # #         "journeyId":  journey_id,
# # # #         "status":     status,
# # # #         "progress":   progress,
# # # #         "message":    message,
# # # #     }
# # # #     if current_video is not None:
# # # #         payload["currentVideo"] = current_video

# # # #     _post("/api/internal/analysis/progress", payload)


# # # # def send_completed(completion_payload: Dict[str, Any]) -> None:
# # # #     """
# # # #     POST /api/internal/analysis/completed

# # # #     Called once after ALL videos in the journey have been processed and their
# # # #     violation frames uploaded to S3.

# # # #     Expected shape of completion_payload (built by CompletionPayload.to_dict())
# # # #     ────────────────────────────────────────────────────────────────────────────
# # # #     {
# # # #         "jobId":         str,
# # # #         "journeyId":     int,
# # # #         "batchId":       str,           # "BATCH-<jobId>" if not supplied
# # # #         "trainDetailId": int,
# # # #         "folderName":    str,
# # # #         "processingTime":int,           # wall-clock ms
# # # #         "videoResults": [
# # # #             {
# # # #                 "videoId":         str,       # STRING per API spec
# # # #                 "sequenceNo":      int,
# # # #                 "durationSeconds": float,
# # # #                 "originalS3Key":   str,
# # # #                 "violations": [
# # # #                     {
# # # #                         "violationType":          str,
# # # #                         "severity":               str,
# # # #                         "confidence":             float,
# # # #                         "riskScore":              float,
# # # #                         "timestamp":              float,   # journey-global seconds
# # # #                         "originalVideoTimestamp": float,   # local-video seconds
# # # #                         "framePaths":             [str]
# # # #                     }
# # # #                 ]
# # # #             }
# # # #         ]
# # # #     }
# # # #     """
# # # #     _post("/api/internal/analysis/completed", completion_payload)


# # # # def send_failed(
# # # #     job_id:        str,
# # # #     journey_id:    int,          # kept as parameter for caller convenience / logging
# # # #     error_message: str,
# # # # ) -> None:
# # # #     """
# # # #     POST /api/internal/analysis/failed

# # # #     Called whenever an unrecoverable exception occurs during job processing.
# # # #     Spring Boot will mark AnalysisJob and Journey as FAILED.

# # # #     Note: The API spec only requires jobId + errorMessage in the request body.
# # # #     journeyId is accepted as a parameter here for logging but is NOT included
# # # #     in the outbound payload.
# # # #     """
# # # #     try:
# # # #         _post(
# # # #             "/api/internal/analysis/failed",
# # # #             {
# # # #                 "jobId":        job_id,
# # # #                 "errorMessage": error_message,
# # # #             },
# # # #         )
# # # #     except Exception as exc:
# # # #         # Failure callback must never itself raise — log and swallow.
# # # #         log.error(
# # # #             "[Callback] send_failed itself failed (job=%s journey=%d): %s",
# # # #             job_id, journey_id, exc,
# # # #         )



# # # """

# # # callback_client.py

# # # ──────────────────

# # # Sends progress, completion, and failure callbacks to the Spring Boot backend.
 
# # # Live server : https://cvvrsrailway-api.sconexsoft.com/cvs

# # # Auth        : NOT required for /api/internal/* endpoints (internal worker APIs).
 
# # # Fixes from previous version

# # # ─────────────────────────────

# # # • Added set_base_url() — called by consumer.py with the callbackBaseUrl from

# # #   the RabbitMQ message, so Python posts to the correct Spring Boot server

# # #   per environment without needing env var changes.

# # # • _BASE_URL still falls back to env var SPRING_BOOT_BASE_URL or the live

# # #   server when callbackBaseUrl is not provided in the message.

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
 
# # # # Default to the live CVVRS server; override with SPRING_BOOT_BASE_URL for local dev

# # # # or call set_base_url() with the callbackBaseUrl from the RabbitMQ message.

# # # _BASE_URL = os.environ.get(

# # #     "SPRING_BOOT_BASE_URL",

# # #     "https://cvvrsrailway-api.sconexsoft.com/cvs",

# # # )

# # # _TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))
 
# # # log = logging.getLogger("callback_client")
 
 
# # # # ── FIX: dynamic base URL setter ─────────────────────────────────────────────
 
# # # def set_base_url(url: str) -> None:

# # #     """

# # #     Override the callback base URL at runtime.
 
# # #     Called by consumer.py when the RabbitMQ message contains a callbackBaseUrl.

# # #     This allows the same Python worker to callback correctly to both local

# # #     and staging Spring Boot servers without changing env vars.
 
# # #     Example values:

# # #         "http://localhost:8093/api/internal/analysis"         (local)

# # #         "https://cvvrsrailway-api.sconexsoft.com/cvs/api/internal/analysis"  (staging)
 
# # #     Note: The URL passed here is the FULL path up to /api/internal/analysis.

# # #     The individual callbacks append /progress, /completed, /failed.

# # #     """

# # #     global _BASE_URL

# # #     # Strip the /api/internal/analysis suffix if present — we add it in _post()

# # #     base = url.rstrip("/")

# # #     if base.endswith("/api/internal/analysis"):

# # #         base = base[: -len("/api/internal/analysis")]

# # #     _BASE_URL = base

# # #     log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)
 
 
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

# # #     current_video: Optional[int] = None,

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

# # #     current_video : 1-based index of the video currently being processed.

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
 
# # #     completion_payload is built by CompletionPayload.to_dict() in models.py.

# # #     """

# # #     _post("/api/internal/analysis/completed", completion_payload)
 
 
# # # def send_failed(

# # #     job_id:        str,

# # #     journey_id:    int,          # kept for caller convenience / logging

# # #     error_message: str,

# # # ) -> None:

# # #     """

# # #     POST /api/internal/analysis/failed
 
# # #     Called whenever an unrecoverable exception occurs during job processing.

# # #     Spring Boot will mark AnalysisJob and Journey as FAILED.
 
# # #     Note: The API spec only requires jobId + errorMessage.

# # #     journeyId is accepted as a parameter for logging but is NOT included

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

# # Changes from previous version
# # ──────────────────────────────
# # • Added check_job_completed(job_id) — queries GET /api/internal/analysis/status/{jobId}
# #   and returns True when the backend reports the job is already COMPLETED.
# #   Called by consumer.py as an idempotency guard before starting any processing
# #   on a redelivered message.
# # """

# # from __future__ import annotations
# # import logging
# # import os
# # from typing import Any, Dict, Optional

# # import requests
# # from dotenv import load_dotenv

# # # ── Credentials / config ─────────────────────────────────────────────────────
# # _ENV_PATH = os.path.join(
# #     os.path.dirname(os.path.abspath(__file__)),
# #     "config", "credentials.env",
# # )
# # load_dotenv(_ENV_PATH)

# # _BASE_URL = os.environ.get(
# #     "SPRING_BOOT_BASE_URL",
# #     "https://cvvrsrailway-api.sconexsoft.com/cvs",
# # )
# # _TIMEOUT = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

# # log = logging.getLogger("callback_client")


# # # ── Dynamic base URL setter ───────────────────────────────────────────────────

# # def set_base_url(url: str) -> None:
# #     """
# #     Override the callback base URL at runtime.

# #     Called by consumer.py when the RabbitMQ message contains a callbackBaseUrl.
# #     Allows the same Python worker to callback correctly to both local and staging
# #     Spring Boot servers without changing env vars.

# #     The URL passed here is the FULL path up to /api/internal/analysis.
# #     Individual callbacks append /progress, /completed, /failed, /status/{id}.
# #     """
# #     global _BASE_URL
# #     base = url.rstrip("/")
# #     if base.endswith("/api/internal/analysis"):
# #         base = base[: -len("/api/internal/analysis")]
# #     _BASE_URL = base
# #     log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)


# # # ── Internal helpers ──────────────────────────────────────────────────────────

# # def _post(path: str, payload: Dict[str, Any]) -> None:
# #     """
# #     POST to a Spring Boot internal endpoint.
# #     No Authorization header — /api/internal/* are worker-only endpoints.
# #     Raises RuntimeError on non-2xx so the consumer can route to the failure path.
# #     """
# #     url = f"{_BASE_URL}{path}"
# #     log.debug("[Callback] POST %s  payload=%s", url, payload)
# #     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
# #     if not resp.ok:
# #         raise RuntimeError(
# #             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# #         )
# #     log.debug("[Callback] %s → %d", path, resp.status_code)


# # def _get(path: str) -> requests.Response:
# #     """
# #     GET from a Spring Boot internal endpoint.
# #     Raises RuntimeError on non-2xx.
# #     """
# #     url = f"{_BASE_URL}{path}"
# #     log.debug("[Callback] GET %s", url)
# #     resp = requests.get(url, timeout=_TIMEOUT)
# #     if not resp.ok:
# #         raise RuntimeError(
# #             f"Callback GET {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
# #         )
# #     log.debug("[Callback] %s → %d", path, resp.status_code)
# #     return resp


# # # ── Public API ────────────────────────────────────────────────────────────────

# # def check_job_completed(job_id: str) -> bool:
# #     """
# #     NEW — Idempotency check.

# #     Queries GET /api/internal/analysis/status/{jobId} and returns True when
# #     the backend reports the job status as COMPLETED.

# #     Called by consumer.py at the very start of _handle_job() so that
# #     RabbitMQ redeliveries of already-completed jobs are detected immediately
# #     and ACKed without re-running any processing.

# #     Backend contract (expected JSON shape):
# #         { "status": "COMPLETED" }   → job already done → return True
# #         { "status": "PROCESSING" }  → in-flight (rare edge-case) → return False
# #         { "status": "PENDING" }     → not yet processed → return False
# #         404 Not Found               → job unknown (treat as not completed)

# #     If the endpoint does not exist yet on your Spring Boot side, add it as:
# #         GET /api/internal/analysis/status/{jobId}
# #         → ResponseEntity<Map<String,String>>  { "status": job.getStatus().name() }

# #     Raises on network errors so the consumer can decide whether to proceed
# #     with processing or skip (consumer.py catches and proceeds on error).
# #     """
# #     try:
# #         resp = _get(f"/api/internal/analysis/status/{job_id}")
# #         data = resp.json()
# #         status = data.get("status", "").upper()
# #         is_done = status == "COMPLETED"
# #         log.info(
# #             "[Callback] check_job_completed  job=%s  status=%s  is_done=%s",
# #             job_id, status, is_done,
# #         )
# #         return is_done
# #     except RuntimeError as exc:
# #         # 404 → job not found in the backend → definitely not completed
# #         if "404" in str(exc):
# #             log.info(
# #                 "[Callback] check_job_completed  job=%s  → 404 (not found) → not completed",
# #                 job_id,
# #             )
# #             return False
# #         raise


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
# #         "jobId":     job_id,
# #         "journeyId": journey_id,
# #         "status":    status,
# #         "progress":  progress,
# #         "message":   message,
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
# #     journey_id:    int,   # kept for caller convenience / logging
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

# Changes in this version
# ────────────────────────
# • check_job_completed() now uses the EXISTING completed-callback endpoint
#   as a probe instead of a dedicated /status/{jobId} endpoint that doesn't
#   exist yet on Spring Boot.

#   Strategy: POST /api/internal/analysis/completed with only { "jobId": id,
#   "probe": true }.  Spring Boot should return 409 CONFLICT (or any non-500)
#   when the job is already done.  If it returns 500 we treat it as "unknown"
#   and fall through to processing (safe default).

#   *** TEMPORARY until the Spring Boot /status endpoint is implemented. ***
#   Once GET /api/internal/analysis/status/{jobId} is live, revert
#   check_job_completed() to use _get() as originally written.
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
#     global _BASE_URL
#     base = url.rstrip("/")
#     if base.endswith("/api/internal/analysis"):
#         base = base[: -len("/api/internal/analysis")]
#     _BASE_URL = base
#     log.info("[CallbackClient]  Base URL set to: %s", _BASE_URL)


# # ── Internal helpers ──────────────────────────────────────────────────────────

# def _post(path: str, payload: Dict[str, Any]) -> requests.Response:
#     url = f"{_BASE_URL}{path}"
#     log.debug("[Callback] POST %s  payload=%s", url, payload)
#     resp = requests.post(url, json=payload, timeout=_TIMEOUT)
#     if not resp.ok:
#         raise RuntimeError(
#             f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
#         )
#     log.debug("[Callback] %s → %d", path, resp.status_code)
#     return resp


# def _get(path: str) -> requests.Response:
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
#     Idempotency check — returns True if the backend already has this job as COMPLETED.

#     TWO-STAGE STRATEGY
#     ──────────────────
#     Stage 1 (permanent): GET /api/internal/analysis/status/{jobId}
#         → { "status": "COMPLETED" }  → True
#         → { "status": "PENDING/PROCESSING" } → False
#         → 404 → False  (job not known yet)
#         → 500 → fall through to Stage 2

#     Stage 2 (temporary fallback until Spring Boot implements /status):
#         Uses GET /api/internal/analysis/job/{jobId} or any existing
#         read endpoint.  If that also 500s, we return False (safe default:
#         process the job rather than silently drop it).

#     WHAT TO IMPLEMENT ON SPRING BOOT (one small controller method):
#     ───────────────────────────────────────────────────────────────
#     @GetMapping("/api/internal/analysis/status/{jobId}")
#     public ResponseEntity<Map<String,String>> getJobStatus(
#             @PathVariable String jobId) {
#         return analysisJobRepository.findByJobId(jobId)
#             .map(job -> ResponseEntity.ok(Map.of("status", job.getStatus().name())))
#             .orElse(ResponseEntity.notFound().<Map<String,String>>build());
#     }

#     Once that endpoint is deployed, Stage 2 below can be deleted.
#     """

#     # ── Stage 1: dedicated status endpoint ───────────────────────────────────
#     try:
#         resp = _get(f"/api/internal/analysis/status/{job_id}")
#         status = resp.json().get("status", "").upper()
#         is_done = status == "COMPLETED"
#         log.info(
#             "[Callback] idempotency check (status endpoint)  job=%s  status=%s  done=%s",
#             job_id, status, is_done,
#         )
#         return is_done
#     except RuntimeError as exc:
#         if "404" in str(exc):
#             log.info(
#                 "[Callback] idempotency check  job=%s  → 404 → not completed", job_id
#             )
#             return False
#         if "500" in str(exc):
#             log.warning(
#                 "[Callback] idempotency check  job=%s  → /status endpoint returned 500 "
#                 "(not implemented yet?) — trying fallback probe.", job_id
#             )
#             # fall through to Stage 2
#         else:
#             # Network error or unexpected status — safe default: process the job
#             log.warning(
#                 "[Callback] idempotency check  job=%s  → unexpected error (%s) "
#                 "— proceeding with processing.", job_id, exc
#             )
#             return False

#     # ── Stage 2: fallback probe using progress endpoint ───────────────────────
#     # We send a lightweight progress probe at 0 % with status=CHECK.
#     # Spring Boot should:
#     #   • Return 200 if the job is still PENDING/PROCESSING (→ process it).
#     #   • Return 409 CONFLICT if the job is already COMPLETED (→ skip it).
#     #   • Return 404 if the job is unknown (→ process it).
#     # If the backend doesn't handle the CHECK status specially it will just
#     # update progress to 0 — harmless on an already-completed job.
#     #
#     # THIS IS A TEMPORARY FALLBACK.  Remove once /status endpoint is live.
#     try:
#         url = f"{_BASE_URL}/api/internal/analysis/progress"
#         resp = requests.post(
#             url,
#             json={
#                 "jobId":    job_id,
#                 "status":   "CHECK",   # sentinel value Spring Boot can detect
#                 "progress": 0,
#                 "message":  "idempotency-probe",
#             },
#             timeout=_TIMEOUT,
#         )
#         if resp.status_code == 409:
#             log.warning(
#                 "[Callback] idempotency check (probe)  job=%s  → 409 CONFLICT "
#                 "→ already COMPLETED — will skip.", job_id
#             )
#             return True
#         log.info(
#             "[Callback] idempotency check (probe)  job=%s  → HTTP %d → not completed.",
#             job_id, resp.status_code,
#         )
#         return False
#     except Exception as exc2:
#         log.warning(
#             "[Callback] idempotency check (probe) failed  job=%s: %s "
#             "— proceeding with processing.", job_id, exc2
#         )
#         return False


# def send_progress(
#     job_id:        str,
#     journey_id:    int,
#     progress:      int,
#     message:       str,
#     status:        str = "PROCESSING",
#     current_video: Optional[int] = None,
# ) -> None:
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
#     _post("/api/internal/analysis/completed", completion_payload)


# def send_failed(
#     job_id:        str,
#     journey_id:    int,
#     error_message: str,
# ) -> None:
#     try:
#         _post(
#             "/api/internal/analysis/failed",
#             {
#                 "jobId":        job_id,
#                 "errorMessage": error_message,
#             },
#         )
#     except Exception as exc:
#         log.error(
#             "[Callback] send_failed itself failed (job=%s journey=%d): %s",
#             job_id, journey_id, exc,
#         )



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
import threading
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

# ── Credentials / config ─────────────────────────────────────────────────────
_ENV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "config", "credentials.env",
)
load_dotenv(_ENV_PATH)

_DEFAULT_BASE_URL = os.environ.get(
    "SPRING_BOOT_BASE_URL",
    "https://cvvrsrailway-api.sconexsoft.com/cvs",
)
_TIMEOUT = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

log = logging.getLogger("callback_client")

# ── Thread-local base URL ──────────────────────────────────────────────────────
# NOTE (worker-pool refactor): the RabbitMQ consumer can now have MULTIPLE
# journeys in flight at once (one per GPU worker), each running _handle_job()
# on its own thread. A single module-level _BASE_URL global would let one
# journey's set_base_url() call silently redirect ANOTHER, concurrently
# running journey's callbacks to the wrong host. Storing it in thread-local
# state keeps each journey's callback_base_url isolated to the thread that's
# actually processing it, with no change to single-journey-at-a-time behavior.
_base_url_local = threading.local()


def _get_base_url() -> str:
    return getattr(_base_url_local, "value", _DEFAULT_BASE_URL)


# ── Dynamic base URL setter ───────────────────────────────────────────────────

def set_base_url(url: str) -> None:
    base = url.rstrip("/")
    if base.endswith("/api/internal/analysis"):
        base = base[: -len("/api/internal/analysis")]
    _base_url_local.value = base
    log.info("[CallbackClient]  Base URL set to: %s (thread=%s)",
             base, threading.current_thread().name)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _post(path: str, payload: Dict[str, Any]) -> requests.Response:
    url = f"{_get_base_url()}{path}"
    log.debug("[Callback] POST %s  payload=%s", url, payload)
    resp = requests.post(url, json=payload, timeout=_TIMEOUT)
    if not resp.ok:
        raise RuntimeError(
            f"Callback POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
        )
    log.debug("[Callback] %s → %d", path, resp.status_code)
    return resp


def _get(path: str) -> requests.Response:
    url = f"{_get_base_url()}{path}"
    log.debug("[Callback] GET %s", url)
    resp = requests.get(url, timeout=_TIMEOUT)
    if not resp.ok:
        raise RuntimeError(
            f"Callback GET {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
        )
    log.debug("[Callback] %s → %d", path, resp.status_code)
    return resp


# ── Public API ────────────────────────────────────────────────────────────────

# ── Local idempotency cache ───────────────────────────────────────────────────
#
# When the worker finishes a job it records the job_id here so that if the
# same message is redelivered (RabbitMQ consumer-timeout cancels the consumer
# mid-journey, broker requeues, worker reconnects and picks it up again) the
# idempotency check catches it locally WITHOUT relying on the backend /status
# endpoint — which currently returns 500 for both probe strategies.
#
# The cache is process-lifetime only (lost on restart).  That is intentional:
# after a genuine worker crash we WANT to reprocess any unACKed job.  The
# repeat-job problem described in the bug report is caused by RabbitMQ's
# consumer_timeout firing while the worker is still alive and processing,
# so the in-memory cache is always present when the redelivery arrives.
#
# mark_job_completed() is called by consumer.py immediately before sending
# the ACK, so the job is in the cache for any subsequent redelivery.
_completed_jobs: set = set()
_completed_jobs_lock = threading.Lock()


def mark_job_completed(job_id: str) -> None:
    """Record that this worker instance has successfully completed job_id."""
    with _completed_jobs_lock:
        _completed_jobs.add(job_id)
    log.info("[CallbackClient]  Marked job %s as locally completed.", job_id)


# ── In-progress job guard ─────────────────────────────────────────────────────
#
# _completed_jobs (above) only catches a redelivery of a job that has
# ALREADY finished. It does NOT catch a redelivery that arrives WHILE the
# original delivery is still being processed on another thread.
#
# Before the worker-pool refactor this could never happen: _on_message()
# ran _handle_job() directly on the single pika I/O thread, so that thread
# had to finish (or crash) before it could ever see a redelivery. Now that
# each journey runs on its own thread (so multiple GPU workers can be busy
# at once), a dropped/reconnected RabbitMQ connection can redeliver an
# unacked message while the ORIGINAL job thread is still mid-journey on a
# different GPU worker — resulting in the same journey being processed
# twice, in parallel, by two different workers. try_start_job()/
# finish_job() below close that gap: a job_id can only be "in progress" on
# one thread at a time.
_in_progress_jobs: set = set()
_in_progress_jobs_lock = threading.Lock()


def try_start_job(job_id: str) -> bool:
    """
    Attempt to claim job_id for the calling thread. Returns True if this
    thread is now the sole owner and should proceed; returns False if
    another thread already owns it (a redelivery arrived mid-flight) — the
    caller should NOT reprocess the journey in that case.
    """
    with _in_progress_jobs_lock:
        if job_id in _in_progress_jobs:
            return False
        _in_progress_jobs.add(job_id)
        return True


def finish_job(job_id: str) -> None:
    """Release the claim on job_id. MUST be called from a finally block so
    a claim is never left dangling — otherwise a genuine crash-recovery
    redelivery (after the original truly died) would be blocked forever."""
    with _in_progress_jobs_lock:
        _in_progress_jobs.discard(job_id)


def check_job_completed(job_id: str) -> bool:
    """
    Idempotency check — returns True if the backend already has this job as COMPLETED.

    STAGE 0 (local cache — fastest, most reliable):
    ─────────────────────────────────────────────────
    If THIS worker process already completed and ACKed the job during the
    current run, it is in _completed_jobs and we return True immediately,
    bypassing both backend probes entirely.  This is the correct fix for the
    RabbitMQ consumer-timeout redelivery problem: the broker requeues the
    message and the worker's reconnect loop picks it up again, but the
    in-memory cache catches it before any work is repeated.

    TWO-STAGE STRATEGY (backend probes — for cross-process idempotency)
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

    # ── Stage 0: local in-process cache (catches consumer-timeout redeliveries)
    with _completed_jobs_lock:
        already_done = job_id in _completed_jobs
    if already_done:
        log.warning(
            "[Callback] idempotency check  job=%s  → found in local completed cache "
            "— this is a redelivery of an already-ACKed job (likely caused by "
            "RabbitMQ consumer_timeout). Skipping re-processing.", job_id,
        )
        return True

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
        url = f"{_get_base_url()}/api/internal/analysis/progress"
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
    video_id:      Optional[int] = None,
    error_type:    Optional[str] = None,
    stack_trace:   Optional[str] = None,
    reason:        Optional[str] = None,
) -> None:
    """
    Calls POST /api/internal/analysis/failed.

    Backward compatible: existing job-level callers (no video_id) keep
    working exactly as before — the payload shape for that case is
    unchanged ({"jobId", "errorMessage"}).

    When video_id IS supplied, this becomes a PER-VIDEO failure report
    (Phase 1 requirement: "no failed video should wait until journey
    completion to be reported"). The payload then also carries:
        videoId     — STRING, matches VideoResult.to_dict()'s convention
        journeyId
        errorType   — short classifier, e.g. "OOM", "DECODE_ERROR",
                      "MEDIAPIPE_ERROR", "YOLO_ERROR", "TIMEOUT",
                      "RESOURCE_EXHAUSTION", "NOT_PROCESSED"
        stackTrace  — full traceback text, if available
        reason      — human-readable reason string, e.g.
                      "Not Processed - Worker Resource Exhaustion"
                      for videos skipped after an OOM on an earlier video.
    """
    payload: Dict[str, Any] = {
        "jobId":        job_id,
        "errorMessage": error_message,
    }
    if video_id is not None:
        payload["videoId"]    = str(video_id)   # STRING per API spec convention
        payload["journeyId"]  = journey_id
    if error_type is not None:
        payload["errorType"] = error_type
    if stack_trace is not None:
        payload["stackTrace"] = stack_trace
    if reason is not None:
        payload["reason"] = reason

    try:
        _post("/api/internal/analysis/failed", payload)
    except Exception as exc:
        log.error(
            "[Callback] send_failed itself failed (job=%s journey=%d video=%s): %s",
            job_id, journey_id, video_id, exc,
        )


def send_video_failed(
    job_id:        str,
    journey_id:    int,
    video_id:      int,
    error_type:    str,
    error_message: str,
    stack_trace:   str = "",
    reason:        Optional[str] = None,
) -> None:
    """
    Convenience wrapper for the per-video failure report required by
    Phase 1: "Immediately call the failed endpoint for the affected video"
    — called once per video, the moment that video's outcome is known,
    not batched until the journey ends.

    For resource-exhaustion cascades, pass error_type="RESOURCE_EXHAUSTION"
    (the video that actually hit OOM) or error_type="NOT_PROCESSED" with
    reason="Not Processed - Worker Resource Exhaustion" (videos after it
    that were skipped as a result).
    """
    log.error(
        "[Callback] video FAILED  job=%s  journey=%d  video=%d  type=%s  reason=%s: %s",
        job_id, journey_id, video_id, error_type, reason, error_message,
    )
    send_failed(
        job_id        = job_id,
        journey_id    = journey_id,
        error_message = error_message,
        video_id      = video_id,
        error_type    = error_type,
        stack_trace   = stack_trace,
        reason        = reason,
    )


# ── OOM / resource-exhaustion signature detection ────────────────────────────
#
# Used by analyzer.py to decide whether a per-video exception means "this
# one video is bad" (continue with the rest) vs. "the worker itself can no
# longer safely process more video" (stop the journey, mark every
# remaining video FAILED with reason "Not Processed - Worker Resource
# Exhaustion"). Matches the signatures called out explicitly in the
# Phase 1 spec: OpenCV OutOfMemoryError, "Failed to allocate", cv2.pyd
# memory-related failures, generic MemoryError, std::bad_alloc.
_OOM_SIGNATURES = (
    "outofmemoryerror",
    "out of memory",
    "failed to allocate",
    "cannot allocate memory",
    "bad_alloc",
    "memoryerror",
    "cv2.pyd",
    "access violation",
    "resource exhaust",
)


def is_resource_exhaustion_error(exc: BaseException) -> bool:
    """
    Returns True if `exc` looks like an OOM / native resource-exhaustion
    failure rather than an ordinary recoverable per-video error.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    if isinstance(exc, MemoryError):
        return True
    return any(sig in text for sig in _OOM_SIGNATURES)


def classify_video_error(exc: BaseException) -> str:
    """
    Maps a caught per-video exception to a short errorType string for the
    failed-endpoint payload. Best-effort classification by exception type
    and message content — defaults to "PROCESSING_ERROR" when nothing
    more specific matches.
    """
    if is_resource_exhaustion_error(exc):
        return "RESOURCE_EXHAUSTION"
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "timeout" in name or "timeout" in text:
        return "TIMEOUT"
    if "mediapipe" in text:
        return "MEDIAPIPE_ERROR"
    if "yolo" in text or "ultralytics" in text:
        return "YOLO_ERROR"
    if "cv2" in text or "opencv" in text or name == "error":
        return "OPENCV_ERROR"
    if "cannot open" in text or "corrupt" in text or "decode" in text or "invalid" in text:
        return "DECODE_ERROR"
    return "PROCESSING_ERROR"


def compute_journey_status(
    total_videos: int,
    succeeded_video_ids: "set[int]",
    failed_video_ids: "set[int]",
) -> str:
    """
    Computes the journey-level terminal status per Phase 1 spec:
        COMPLETED              — every video succeeded
        COMPLETED_WITH_ERRORS  — at least one succeeded AND at least one failed
        FAILED                 — every video failed (or none succeeded)

    TIMED_OUT is NOT computed here — that's set explicitly by the watchdog
    path in consumer.py, which knows it hit a timeout rather than inferring
    it from video counts.
    """
    if total_videos == 0:
        return "FAILED"
    if not failed_video_ids:
        return "COMPLETED"
    if succeeded_video_ids:
        return "COMPLETED_WITH_ERRORS"
    return "FAILED"