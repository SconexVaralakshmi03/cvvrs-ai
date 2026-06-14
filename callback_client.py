"""
callback_client.py
──────────────────
Sends progress, completion, and failure callbacks to the Spring Boot backend.

All three public helpers are fire-and-forget from the caller's perspective;
they raise on non-2xx so the consumer can route to the failure path.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

# ── Credentials / config ────────────────────────────────────────────────────
_ENV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "config", "credentials.env",
)
load_dotenv(_ENV_PATH)

_BASE_URL = os.environ.get("SPRING_BOOT_BASE_URL", "http://localhost:8080")
_TIMEOUT  = int(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))

log = logging.getLogger("callback_client")


# ── Internal helper ──────────────────────────────────────────────────────────

def _post(path: str, payload: Dict[str, Any]) -> None:
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
    job_id:     str,
    journey_id: int,
    progress:   int,
    message:    str,
    status:     str = "PROCESSING",
) -> None:
    """
    POST /api/internal/analysis/progress

    Called periodically during analysis to update the frontend progress bar
    and the SSE stream.
    """
    _post(
        "/api/internal/analysis/progress",
        {
            "jobId":      job_id,
            "journeyId":  journey_id,
            "status":     status,
            "progress":   progress,
            "message":    message,
        },
    )


def send_completed(completion_payload: Dict[str, Any]) -> None:
    """
    POST /api/internal/analysis/completed

    Called once — after ALL videos in the journey have been processed and
    their violation frames uploaded to S3.

    Expected shape of completion_payload
    ─────────────────────────────────────
    {
        "jobId":          str,
        "journeyId":      int,
        "processingTime": int,          # wall-clock ms
        "videoResults":   List[dict],   # see models.py → VideoResult
    }
    """
    _post("/api/internal/analysis/completed", completion_payload)


def send_failed(
    job_id:        str,
    journey_id:    int,
    error_message: str,
) -> None:
    """
    POST /api/internal/analysis/failed

    Called whenever an unrecoverable exception occurs during job processing.
    Spring Boot will mark AnalysisJob and Journey as FAILED.
    """
    try:
        _post(
            "/api/internal/analysis/failed",
            {
                "jobId":        job_id,
                "journeyId":    journey_id,
                "errorMessage": error_message,
            },
        )
    except Exception as exc:
        # Failure callback must never itself raise — log and swallow.
        log.error("[Callback] send_failed itself failed: %s", exc)