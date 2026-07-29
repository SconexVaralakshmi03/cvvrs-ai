"""
llm_verifier.py
================

LLM-based verification gate for candidate violations, integrated into the
main journey analysis pipeline (analyzer.py).

This wraps the Qwen2.5-VL (via Ollama) verifier described in prompt.py so
that every candidate violation frame produced by the YOLO/MediaPipe
detectors (detector/*.py) is independently re-checked by a vision-language
model BEFORE:

  • its frame image is uploaded to S3, and
  • it is included in the completion payload sent to the Java/Spring Boot
    backend (i.e. before it ever reaches the DB).

If the LLM rejects a candidate, analyzer.py drops it entirely — no S3
upload, no DB row. If the LLM verifies it, the returned "role" (Loco
Pilot / Assistant Loco Pilot / Unknown) is attached to the violation and
flows through into the "role" field of the outbound ViolationResult
payload — this is the only change to the existing payload shape.

Design notes
------------
* Only violation types with defined criteria in prompt.py (Mobile Phone,
  Drowsiness, Hand Raising, Seat Absence) are sent to the LLM. Any other
  event_type is passed straight through unverified (verified=True,
  role="Unknown", skipped=True) — unchanged pipeline behaviour for those
  types, since prompt.py currently defines no verification criteria for
  them.
* If Ollama / the model is unavailable, or the LLM call errors out
  repeatedly, verify_frame() fails *open* or *closed* depending on the
  LLM_VERIFICATION_FAIL_OPEN setting (config/settings.py), so a
  temporarily-down verifier can never silently take down the whole
  pipeline in a way ops doesn't control. Default is fail CLOSED (reject)
  to match the "when in doubt, reject" philosophy of prompt.py.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
from typing import Optional

import numpy as np

from prompt import Violation, build_verification_prompt

log = logging.getLogger("llm_verifier")

# ── Config (all overridable in config/settings.py; sane defaults if absent) ──

try:
    import config.settings as _settings
except Exception:
    _settings = None


def _cfg(name: str, default):
    return getattr(_settings, name, default) if _settings is not None else default


LLM_VERIFICATION_ENABLED   = bool(_cfg("LLM_VERIFICATION_ENABLED", True))
# Fail-open = keep the candidate (verified=True) if the LLM can't be
# reached / doesn't respond after retries. Fail-closed (default) rejects
# it instead, matching the "when in doubt, reject" design of prompt.py.
LLM_VERIFICATION_FAIL_OPEN = bool(_cfg("LLM_VERIFICATION_FAIL_OPEN", False))
OLLAMA_MODEL                = str(_cfg("OLLAMA_MODEL", "qwen2.5vl:7b"))
OLLAMA_HOST                 = _cfg("OLLAMA_HOST", None)  # None = client default (localhost:11434)
OLLAMA_TIMEOUT_SECONDS       = int(_cfg("OLLAMA_TIMEOUT_SECONDS", 120))
OLLAMA_MAX_RETRIES           = int(_cfg("OLLAMA_MAX_RETRIES", 2))
OLLAMA_TEMPERATURE           = float(_cfg("OLLAMA_TEMPERATURE", 0.1))


# Internal detector event_type (as stored on _Violation.type, e.g. from
# gadget_detector.py / head_drop_detector.py / hand_raise_detector.py /
# seat_absence_detector.py) → prompt.py Violation constant. Only these are
# LLM-verifiable today, since prompt.py only defines criteria for them.
EVENT_TYPE_TO_PROMPT_VIOLATION: dict[str, str] = {
    "phone_use":       Violation.MOBILE_PHONE,
    "drowsy":          Violation.DROWSINESS,
    "sleeping":        Violation.DROWSINESS,
    "sleeping_absent": Violation.DROWSINESS,
    "hand_raise":      Violation.HAND_RAISING,
    "seat_absence":    Violation.SEAT_ABSENCE,
}

VALID_ROLES = {"Loco Pilot", "Assistant Loco Pilot", "Unknown"}


class VerificationTimeoutError(RuntimeError):
    pass


class OllamaConnectionError(RuntimeError):
    pass


# ── Ollama call ───────────────────────────────────────────────────────────

def _call_ollama(prompt_text: str, image_bgr: np.ndarray) -> str:
    """Blocking Ollama chat call with a hard timeout. Frame is passed as
    JPEG-encoded bytes so no temp file needs to be written to disk."""
    import cv2
    from ollama import Client

    ok, buf = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise ValueError("Failed to JPEG-encode frame for LLM verification")
    image_bytes = buf.tobytes()

    client = Client(host=OLLAMA_HOST) if OLLAMA_HOST else Client()

    def _do_call() -> str:
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt_text, "images": [image_bytes]}],
            options={"temperature": OLLAMA_TEMPERATURE},
        )
        try:
            return response["message"]["content"]
        except (TypeError, KeyError):
            return response.message.content  # type: ignore[union-attr]

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_do_call)
        try:
            return future.result(timeout=OLLAMA_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError as exc:
            raise VerificationTimeoutError(
                f"Model call exceeded {OLLAMA_TIMEOUT_SECONDS}s"
            ) from exc
        except Exception as exc:
            msg = str(exc).lower()
            if "connect" in msg or "connection" in msg:
                raise OllamaConnectionError(str(exc)) from exc
            raise


# ── Response parsing (mirrors the standalone verify.py safe_parse logic) ────

def _safe_parse(raw_text: str) -> Optional[dict]:
    text = (raw_text or "").strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    try:
        verified = bool(data.get("verified", False))

        role = data.get("role", "Unknown")
        if role not in VALID_ROLES:
            role = "Unknown"
        if not verified:
            role = "Unknown"

        try:
            confidence = int(data.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0
        confidence = max(0, min(100, confidence))
        if not verified:
            confidence = 0

        reason = str(data.get("reason", "")).strip() or "No reason provided."

        return {
            "verified":   verified,
            "role":       role,
            "confidence": confidence,
            "reason":     reason,
        }
    except Exception:
        return None


# ── Public entry point ───────────────────────────────────────────────────

def verify_frame(image_bgr: np.ndarray, event_type: str, log_label: str = "") -> dict:
    """
    Verify one candidate-violation frame with the LLM.

    Returns a dict:
        {
          "verified":   bool,
          "role":       "Loco Pilot" | "Assistant Loco Pilot" | "Unknown",
          "confidence": int 0-100,
          "reason":     str,
          "skipped":    bool,   # True if this event_type has no defined LLM
                                 # verification criteria, or verification is
                                 # disabled — verified is always True in that
                                 # case (unverified pass-through, unchanged
                                 # pipeline behaviour).
        }
    """
    if not LLM_VERIFICATION_ENABLED:
        return {
            "verified": True, "role": "Unknown", "confidence": 0,
            "reason": "LLM verification disabled by config.", "skipped": True,
        }

    candidate_violation = EVENT_TYPE_TO_PROMPT_VIOLATION.get((event_type or "").lower())
    if candidate_violation is None:
        return {
            "verified": True, "role": "Unknown", "confidence": 0,
            "reason": f"No LLM verification criteria defined for event_type={event_type!r}.",
            "skipped": True,
        }

    prompt_text = build_verification_prompt(candidate_violation)

    raw_text: Optional[str] = None
    last_error: Optional[Exception] = None
    for attempt in range(1, OLLAMA_MAX_RETRIES + 2):
        try:
            raw_text = _call_ollama(prompt_text, image_bgr)
            break
        except Exception as exc:
            last_error = exc
            log.warning(
                "LLM verify attempt %d/%d failed for %s (%s): %s",
                attempt, OLLAMA_MAX_RETRIES + 1, log_label, candidate_violation, exc,
            )

    if raw_text is None:
        fail_open = LLM_VERIFICATION_FAIL_OPEN
        log.error(
            "LLM verification unavailable for %s (%s) after retries: %s — %s",
            log_label, candidate_violation, last_error,
            "failing OPEN (kept, unverified)" if fail_open else "failing CLOSED (rejected)",
        )
        return {
            "verified":   fail_open,
            "role":       "Unknown",
            "confidence": 0,
            "reason":     f"LLM verification unavailable: {last_error}",
            "skipped":    False,
        }

    parsed = _safe_parse(raw_text)
    if parsed is None:
        fail_open = LLM_VERIFICATION_FAIL_OPEN
        log.warning(
            "LLM returned unparseable JSON for %s (%s) — %s",
            log_label, candidate_violation,
            "failing OPEN" if fail_open else "failing CLOSED (rejected)",
        )
        return {
            "verified":   fail_open,
            "role":       "Unknown",
            "confidence": 0,
            "reason":     "Model response could not be parsed as valid JSON.",
            "skipped":    False,
        }

    parsed["skipped"] = False
    return parsed