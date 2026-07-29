"""
llm_verifier.py
================

LLM-based verification gate for candidate violations, integrated into the
main journey analysis pipeline (analyzer.py).

This wraps the Qwen2.5-VL (via Ollama) verifier described in prompt.py so
that every candidate violation frame produced by the YOLO/MediaPipe
detectors (detector/*.py) is independently re-checked by a vision-language
model BEFORE it is included in the completion payload sent to the
Java/Spring Boot backend.

REVISION — status/role schema, no more "Unknown"
--------------------------------------------------
This version aligns with prompt.py's two-step flow and its strict output
schema: {"status": bool, "role": "LP" | "ALP" | "BOTH" | "AMBIGUOUS" | null}.

  • status = false  →  role is ALWAYS None (enforced here even if the model
                        slips and includes a role anyway).
  • status = true   →  role is ALWAYS one of "LP" / "ALP" / "BOTH" /
                        "AMBIGUOUS" — "Unknown" is never produced anywhere
                        in this module. Any invalid/missing role on a
                        verified candidate is normalized to "AMBIGUOUS"
                        (genuine, labelled uncertainty), never "Unknown".

Design notes
------------
* Only violation types with defined criteria in prompt.py (Mobile Phone,
  Drowsiness, Hand Raising, Seat Absence) are sent to the LLM. Any other
  event_type is passed straight through unverified (status=True,
  role="AMBIGUOUS", skipped=True) — unchanged pipeline behaviour for
  those types (no LLM criteria exist for them yet), just phrased with the
  new role vocabulary instead of "Unknown".
* If Ollama / the model is unavailable, or the LLM call errors out
  repeatedly, verify_frame() fails *open* or *closed* depending on the
  LLM_VERIFICATION_FAIL_OPEN setting (config/settings.py). Default is fail
  CLOSED (status=False, role=None) to match the "never default to
  status=true" / "when in doubt, reject" philosophy of prompt.py — this is
  also the fix for the DB previously showing status=true for nearly every
  violation: unresolved/failed LLM calls no longer silently pass as true.
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
# Fail-open = keep the candidate (status=True, role="AMBIGUOUS") if the LLM
# can't be reached / doesn't respond after retries. Fail-closed (default,
# recommended) rejects it instead (status=False, role=None), matching the
# "never default to status=true" design of prompt.py.
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

# Valid role values when status=true. "Unknown" is deliberately absent —
# it must never be produced by this module.
VALID_ROLES = {"LP", "ALP", "BOTH", "AMBIGUOUS"}


class VerificationTimeoutError(RuntimeError):
    pass


class OllamaConnectionError(RuntimeError):
    pass


# ── Ollama call ───────────────────────────────────────────────────────────

def _call_ollama(prompt_text: str, image_bgr: np.ndarray, log_label: str = "") -> str:
    """Blocking Ollama chat call with a hard timeout.

    Frame is written to a temp JPEG and passed as a file PATH (not raw
    bytes) — passing bytes directly was found to be handled unreliably by
    some `ollama` python client versions, degrading what the model
    actually saw. A file path matches the known-working reference
    behaviour of the standalone verify.py test tool.
    """
    import cv2
    import os
    import tempfile
    from ollama import Client

    fd, tmp_path = tempfile.mkstemp(suffix=".jpg", prefix="llmverify_")
    os.close(fd)
    try:
        ok = cv2.imwrite(tmp_path, image_bgr)
        if not ok:
            raise ValueError("Failed to JPEG-encode frame for LLM verification")

        print(f"[llm_verifier] Calling Ollama model={OLLAMA_MODEL!r} "
              f"host={OLLAMA_HOST or 'default(127.0.0.1:11434)'} "
              f"label={log_label!r} image={tmp_path} "
              f"size={os.path.getsize(tmp_path)}B")

        client = Client(host=OLLAMA_HOST) if OLLAMA_HOST else Client()

        def _do_call() -> str:
            response = client.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt_text, "images": [tmp_path]}],
                options={"temperature": OLLAMA_TEMPERATURE},
            )
            try:
                return response["message"]["content"]
            except (TypeError, KeyError):
                return response.message.content  # type: ignore[union-attr]

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_do_call)
            try:
                raw = future.result(timeout=OLLAMA_TIMEOUT_SECONDS)
                print(f"[llm_verifier] RAW response for {log_label!r}: {raw!r}")
                return raw
            except concurrent.futures.TimeoutError as exc:
                raise VerificationTimeoutError(
                    f"Model call exceeded {OLLAMA_TIMEOUT_SECONDS}s"
                ) from exc
            except Exception as exc:
                msg = str(exc).lower()
                if "connect" in msg or "connection" in msg:
                    raise OllamaConnectionError(str(exc)) from exc
                raise
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ── Response parsing ─────────────────────────────────────────────────────
#
# Expected schema from prompt.py:
#   {"status": true|false, "role": "LP"|"ALP"|"BOTH"|"AMBIGUOUS"|null}
#
# This parser is defensive against older/partial model output (e.g. a
# stray "verified" key from prompt drift, "Unknown"/legacy role strings)
# so a mid-rollout model hiccup degrades safely rather than crashing.

_LEGACY_ROLE_MAP = {
    # Back-compat normalization only — never produced going forward, but
    # if the model (or a stale cached prompt) ever emits one of these,
    # map it to the new vocabulary instead of leaking "Unknown"/old labels
    # into the payload.
    "loco pilot":           "LP",
    "assistant loco pilot": "ALP",
    "unknown":              "AMBIGUOUS",
    "ambiguous":            "AMBIGUOUS",
    "both":                 "BOTH",
    "lp":                   "LP",
    "alp":                  "ALP",
}


def _normalize_role(raw_role, status: bool) -> Optional[str]:
    """
    Enforce the hard schema rules regardless of what the model returned:
      - status is False  → role is ALWAYS None.
      - status is True   → role is ALWAYS one of VALID_ROLES; anything
        else (missing, null, "Unknown", unrecognized string) becomes
        "AMBIGUOUS" — never "Unknown".
    """
    if not status:
        return None

    if raw_role is None:
        return "AMBIGUOUS"

    role_str = str(raw_role).strip()
    if role_str.upper() in VALID_ROLES:
        return role_str.upper()

    mapped = _LEGACY_ROLE_MAP.get(role_str.strip().lower())
    if mapped:
        return mapped

    return "AMBIGUOUS"


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
        # Accept "status" (current schema) with a fallback to a stray
        # "verified" key in case of prompt/model drift during rollout.
        if "status" in data:
            status = bool(data.get("status", False))
        else:
            status = bool(data.get("verified", False))

        role = _normalize_role(data.get("role"), status)

        reason = str(data.get("reason", "")).strip()  # debug-only, not sent downstream

        return {
            "status": status,
            "role":   role,
            "reason": reason,
        }
    except Exception:
        return None


# ── Public entry point ───────────────────────────────────────────────────

def verify_frame(image_bgr: np.ndarray, event_type: str, log_label: str = "") -> dict:
    """
    Verify one candidate-violation frame with the LLM.

    Returns a dict:
        {
          "status":  bool,             # True = genuine violation, False = rejected
          "role":    str | None,       # "LP" | "ALP" | "BOTH" | "AMBIGUOUS" when
                                        # status=True; ALWAYS None when status=False.
                                        # Never "Unknown".
          "reason":  str,              # debug-only, not sent to the Java backend
          "skipped": bool,             # True if this event_type has no defined LLM
                                        # verification criteria, or verification is
                                        # disabled — unverified pass-through, unchanged
                                        # legacy pipeline behaviour for those types.
        }
    """
    if not LLM_VERIFICATION_ENABLED:
        return {
            "status": True, "role": "AMBIGUOUS",
            "reason": "LLM verification disabled by config.", "skipped": True,
        }

    candidate_violation = EVENT_TYPE_TO_PROMPT_VIOLATION.get((event_type or "").lower())
    if candidate_violation is None:
        return {
            "status": True, "role": "AMBIGUOUS",
            "reason": f"No LLM verification criteria defined for event_type={event_type!r}.",
            "skipped": True,
        }

    prompt_text = build_verification_prompt(candidate_violation)

    raw_text: Optional[str] = None
    last_error: Optional[Exception] = None
    for attempt in range(1, OLLAMA_MAX_RETRIES + 2):
        try:
            raw_text = _call_ollama(prompt_text, image_bgr, log_label=log_label)
            break
        except Exception as exc:
            last_error = exc
            log.warning(
                "LLM verify attempt %d/%d failed for %s (%s): %s",
                attempt, OLLAMA_MAX_RETRIES + 1, log_label, candidate_violation, exc,
            )
            print(f"[llm_verifier] attempt {attempt}/{OLLAMA_MAX_RETRIES + 1} "
                  f"FAILED for {log_label!r} ({candidate_violation}): {exc}")

    if raw_text is None:
        fail_open = LLM_VERIFICATION_FAIL_OPEN
        status = fail_open
        role = "AMBIGUOUS" if fail_open else None
        print(f"[llm_verifier] UNAVAILABLE for {log_label!r} ({candidate_violation}) "
              f"after retries: {last_error} — "
              f"{'failing OPEN (status=True, role=AMBIGUOUS)' if fail_open else 'failing CLOSED (status=False, role=None)'}")
        log.error(
            "LLM verification unavailable for %s (%s) after retries: %s — %s",
            log_label, candidate_violation, last_error,
            "failing OPEN" if fail_open else "failing CLOSED",
        )
        return {
            "status":  status,
            "role":    role,
            "reason":  f"LLM verification unavailable: {last_error}",
            "skipped": False,
        }

    parsed = _safe_parse(raw_text)
    if parsed is None:
        fail_open = LLM_VERIFICATION_FAIL_OPEN
        status = fail_open
        role = "AMBIGUOUS" if fail_open else None
        print(f"[llm_verifier] UNPARSEABLE JSON for {log_label!r} ({candidate_violation}) — "
              f"{'failing OPEN' if fail_open else 'failing CLOSED'}. Raw: {raw_text!r}")
        log.warning(
            "LLM returned unparseable JSON for %s (%s) — %s",
            log_label, candidate_violation,
            "failing OPEN" if fail_open else "failing CLOSED",
        )
        return {
            "status":  status,
            "role":    role,
            "reason":  "Model response could not be parsed as valid JSON.",
            "skipped": False,
        }

    parsed["skipped"] = False
    print(f"[llm_verifier] PARSED verdict for {log_label!r} "
          f"(candidate={candidate_violation}): {parsed}")
    return parsed