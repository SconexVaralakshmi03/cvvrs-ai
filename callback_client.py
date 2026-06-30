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