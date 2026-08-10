# """
# llm_verifier.py
# ================

# LLM-based verification gate for candidate violations, integrated into the
# main journey analysis pipeline (analyzer.py).

# This wraps the Qwen2.5-VL (via Ollama) verifier described in prompt.py so
# that every candidate violation frame produced by the YOLO/MediaPipe
# detectors (detector/*.py) is independently re-checked by a vision-language
# model BEFORE:

#   • its frame image is uploaded to S3, and
#   • it is included in the completion payload sent to the Java/Spring Boot
#     backend (i.e. before it ever reaches the DB).

# Every candidate is kept in the output either way and tagged with a
# "status" field:
#   • status="TRUE"  — the LLM confirmed the violation. "role" is set to
#     one of "LP" (Loco Pilot), "ALP" (Assistant Loco Pilot), "BOTH" (both
#     crew members caught committing the same violation), or "AMBIGUOUS"
#     (violation confirmed but the LLM could not confidently attribute it
#     to one specific role).
#   • status="FALSE" — the LLM rejected the candidate. "role" is always
#     None (no role is ever assigned to a non-violation).

# Design notes
# ------------
# * Only violation types with defined criteria in prompt.py (Mobile Phone,
#   Drowsiness, Hand Raising, Seat Absence) are sent to the LLM. Any other
#   event_type is passed straight through unverified (status="TRUE",
#   role="AMBIGUOUS", skipped=True, llm_invoked=False) — unchanged pipeline
#   behaviour for those types, since prompt.py currently defines no
#   verification criteria for them.
# * If Ollama / the model is unavailable, or the LLM call errors out
#   repeatedly, verify_frame() fails *open* or *closed* depending on the
#   LLM_VERIFICATION_FAIL_OPEN setting (config/settings.py), so a
#   temporarily-down verifier can never silently take down the whole
#   pipeline in a way ops doesn't control. Default is fail CLOSED (reject)
#   to match the "when in doubt, reject" philosophy of prompt.py.
# """

# from __future__ import annotations

# import concurrent.futures
# import json
# import logging
# from typing import List, Optional

# import numpy as np

# from prompt import Violation, build_verification_prompt, build_rsl_hand_brake_prompt

# log = logging.getLogger("llm_verifier")

# # ── Config (all overridable in config/settings.py; sane defaults if absent) ──

# try:
#     import config.settings as _settings
# except Exception:
#     _settings = None


# def _cfg(name: str, default):
#     return getattr(_settings, name, default) if _settings is not None else default


# LLM_VERIFICATION_ENABLED   = bool(_cfg("LLM_VERIFICATION_ENABLED", True))
# # Fail-open = keep the candidate (verified=True) if the LLM can't be
# # reached / doesn't respond after retries. Fail-closed (default) rejects
# # it instead, matching the "when in doubt, reject" design of prompt.py.
# LLM_VERIFICATION_FAIL_OPEN = bool(_cfg("LLM_VERIFICATION_FAIL_OPEN", False))
# OLLAMA_MODEL                = str(_cfg("OLLAMA_MODEL", "qwen2.5vl:7b"))
# OLLAMA_HOST                 = _cfg("OLLAMA_HOST", None)  # None = client default (localhost:11434)
# OLLAMA_TIMEOUT_SECONDS       = int(_cfg("OLLAMA_TIMEOUT_SECONDS", 120))
# OLLAMA_MAX_RETRIES           = int(_cfg("OLLAMA_MAX_RETRIES", 2))
# OLLAMA_TEMPERATURE           = float(_cfg("OLLAMA_TEMPERATURE", 0.1))


# # Internal detector event_type (as stored on _Violation.type, e.g. from
# # gadget_detector.py / head_drop_detector.py / hand_raise_detector.py /
# # seat_absence_detector.py) → prompt.py Violation constant. Only these are
# # LLM-verifiable today, since prompt.py only defines criteria for them.
# EVENT_TYPE_TO_PROMPT_VIOLATION: dict[str, str] = {
#     "phone_use":       Violation.MOBILE_PHONE,
#     "drowsy":          Violation.DROWSINESS,
#     "sleeping":        Violation.DROWSINESS,
#     "sleeping_absent": Violation.DROWSINESS,
#     "hand_raise":      Violation.HAND_RAISING,
#     "seat_absence":    Violation.SEAT_ABSENCE,
# }

# VALID_ROLES = {"Loco Pilot", "Assistant Loco Pilot", "Both", "Unknown"}

# # Canonical short role codes used on the outbound payload / DB, per task
# # spec: role is one of "LP" | "ALP" | "BOTH" | "AMBIGUOUS", or None when the
# # violation itself was rejected (status FALSE).
# ROLE_TO_CODE = {
#     "Loco Pilot":           "LP",
#     "Assistant Loco Pilot": "ALP",
#     "Both":                 "BOTH",
#     "Unknown":               "AMBIGUOUS",
# }


# def _log_verification_json(**fields) -> None:
#     """
#     Emit one structured JSON line to the terminal for EVERY candidate
#     violation that passes through verify_frame(), regardless of outcome.

#     This is the audit trail requested for task 2: from the terminal logs
#     alone it must always be possible to tell (a) whether the LLM was
#     actually invoked for a given frame ("llm_invoked"), and (b) what
#     status/role verdict was ultimately assigned to it.
#     """
#     try:
#         print("[LLM_VERIFICATION_LOG] " + json.dumps(fields, default=str))
#     except Exception:
#         log.exception("Failed to emit LLM verification JSON log")


# class VerificationTimeoutError(RuntimeError):
#     pass


# class OllamaConnectionError(RuntimeError):
#     pass


# # ── Ollama call ───────────────────────────────────────────────────────────

# def _call_ollama(prompt_text: str, image_bgr: np.ndarray, log_label: str = "") -> str:
#     """Blocking Ollama chat call with a hard timeout.

#     Frame is written to a temp JPEG and passed as a file PATH (not raw
#     bytes) — this matches the standalone verify.py tool exactly, which is
#     the known-working reference implementation. Some versions of the
#     `ollama` python client handle raw bytes for `images` less reliably
#     than a path, which can degrade what the model actually sees.
#     """
#     import cv2
#     import os
#     import tempfile
#     from ollama import Client

#     fd, tmp_path = tempfile.mkstemp(suffix=".jpg", prefix="llmverify_")
#     os.close(fd)
#     try:
#         ok = cv2.imwrite(tmp_path, image_bgr)
#         if not ok:
#             raise ValueError("Failed to JPEG-encode frame for LLM verification")

#         print(f"[llm_verifier] Calling Ollama model={OLLAMA_MODEL!r} "
#               f"host={OLLAMA_HOST or 'default(127.0.0.1:11434)'} "
#               f"label={log_label!r} image={tmp_path} "
#               f"size={os.path.getsize(tmp_path)}B")

#         client = Client(host=OLLAMA_HOST) if OLLAMA_HOST else Client()

#         def _do_call() -> str:
#             response = client.chat(
#                 model=OLLAMA_MODEL,
#                 messages=[{"role": "user", "content": prompt_text, "images": [tmp_path]}],
#                 options={"temperature": OLLAMA_TEMPERATURE},
#             )
#             try:
#                 return response["message"]["content"]
#             except (TypeError, KeyError):
#                 return response.message.content  # type: ignore[union-attr]

#         with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
#             future = executor.submit(_do_call)
#             try:
#                 raw = future.result(timeout=OLLAMA_TIMEOUT_SECONDS)
#                 print(f"[llm_verifier] RAW response for {log_label!r}: {raw!r}")
#                 return raw
#             except concurrent.futures.TimeoutError as exc:
#                 raise VerificationTimeoutError(
#                     f"Model call exceeded {OLLAMA_TIMEOUT_SECONDS}s"
#                 ) from exc
#             except Exception as exc:
#                 msg = str(exc).lower()
#                 if "connect" in msg or "connection" in msg:
#                     raise OllamaConnectionError(str(exc)) from exc
#                 raise
#     finally:
#         try:
#             os.remove(tmp_path)
#         except OSError:
#             pass


# # ── Response parsing (mirrors the standalone verify.py safe_parse logic) ────

# def _safe_parse(raw_text: str) -> Optional[dict]:
#     text = (raw_text or "").strip()

#     if text.startswith("```"):
#         lines = text.splitlines()
#         if lines and lines[0].startswith("```"):
#             lines = lines[1:]
#         if lines and lines[-1].strip().startswith("```"):
#             lines = lines[:-1]
#         text = "\n".join(lines).strip()

#     if not text.startswith("{"):
#         start, end = text.find("{"), text.rfind("}")
#         if start != -1 and end != -1 and end > start:
#             text = text[start:end + 1]

#     try:
#         data = json.loads(text)
#     except (json.JSONDecodeError, ValueError):
#         return None
#     if not isinstance(data, dict):
#         return None

#     try:
#         verified = bool(data.get("verified", False))

#         role = data.get("role", "Unknown")
#         if role not in VALID_ROLES:
#             role = "Unknown"
#         if not verified:
#             role = "Unknown"

#         try:
#             confidence = int(data.get("confidence", 0))
#         except (TypeError, ValueError):
#             confidence = 0
#         confidence = max(0, min(100, confidence))
#         if not verified:
#             confidence = 0

#         reason = str(data.get("reason", "")).strip() or "No reason provided."

#         return {
#             "verified":   verified,
#             "role":       role,
#             "confidence": confidence,
#             "reason":     reason,
#         }
#     except Exception:
#         return None


# # ── Public entry point ───────────────────────────────────────────────────

# def _status_role(verified: bool, raw_role: str) -> tuple[str, Optional[str]]:
#     """
#     Map an internal (verified: bool, raw_role: str) pair to the outbound
#     task-spec contract:
#         verified True  → status="TRUE",  role="LP"|"ALP"|"BOTH"|"AMBIGUOUS"
#         verified False → status="FALSE", role=None
#     """
#     if not verified:
#         return "FALSE", None
#     return "TRUE", ROLE_TO_CODE.get(raw_role, "AMBIGUOUS")


# def verify_frame(image_bgr: np.ndarray, event_type: str, log_label: str = "") -> dict:
#     """
#     Verify one candidate-violation frame with the LLM.

#     Returns a dict:
#         {
#           "verified":    bool,   # kept for internal/backward-compat checks
#           "status":      "TRUE" | "FALSE",
#           "role":        "LP" | "ALP" | "BOTH" | "AMBIGUOUS" | None,
#                          # None whenever status == "FALSE" — no role is
#                          # ever assigned to a rejected candidate.
#           "confidence":  int 0-100,
#           "reason":      str,
#           "skipped":     bool,  # True if this event_type has no defined LLM
#                                  # verification criteria, or verification is
#                                  # disabled — status is always "TRUE" in that
#                                  # case (unverified pass-through, unchanged
#                                  # pipeline behaviour), role "AMBIGUOUS".
#           "llm_invoked": bool,  # True iff an actual model call was made
#                                  # (i.e. this candidate was NOT skipped).
#         }

#     Every call to this function emits exactly one
#     "[LLM_VERIFICATION_LOG] {...}" JSON line to the terminal — task 2's
#     audit trail — regardless of which branch below is taken.
#     """
#     if not LLM_VERIFICATION_ENABLED:
#         status, role = "TRUE", "AMBIGUOUS"
#         _log_verification_json(
#             label=log_label, event_type=event_type, candidate_violation=None,
#             llm_invoked=False, status=status, role=role, confidence=0,
#             reason="LLM verification disabled by config.", skipped=True,
#         )
#         return {
#             "verified": True, "status": status, "role": role, "confidence": 0,
#             "reason": "LLM verification disabled by config.", "skipped": True,
#             "llm_invoked": False,
#         }

#     candidate_violation = EVENT_TYPE_TO_PROMPT_VIOLATION.get((event_type or "").lower())
#     if candidate_violation is None:
#         status, role = "TRUE", "AMBIGUOUS"
#         reason = f"No LLM verification criteria defined for event_type={event_type!r}."
#         _log_verification_json(
#             label=log_label, event_type=event_type, candidate_violation=None,
#             llm_invoked=False, status=status, role=role, confidence=0,
#             reason=reason, skipped=True,
#         )
#         return {
#             "verified": True, "status": status, "role": role, "confidence": 0,
#             "reason": reason, "skipped": True, "llm_invoked": False,
#         }

#     prompt_text = build_verification_prompt(candidate_violation)

#     raw_text: Optional[str] = None
#     last_error: Optional[Exception] = None
#     for attempt in range(1, OLLAMA_MAX_RETRIES + 2):
#         try:
#             raw_text = _call_ollama(prompt_text, image_bgr, log_label=log_label)
#             break
#         except Exception as exc:
#             last_error = exc
#             log.warning(
#                 "LLM verify attempt %d/%d failed for %s (%s): %s",
#                 attempt, OLLAMA_MAX_RETRIES + 1, log_label, candidate_violation, exc,
#             )

#     if raw_text is None:
#         fail_open = LLM_VERIFICATION_FAIL_OPEN
#         status, role = _status_role(fail_open, "Unknown")
#         reason = f"LLM verification unavailable: {last_error}"
#         log.error(
#             "LLM verification unavailable for %s (%s) after retries: %s — %s",
#             log_label, candidate_violation, last_error,
#             "failing OPEN (kept, unverified)" if fail_open else "failing CLOSED (rejected)",
#         )
#         _log_verification_json(
#             label=log_label, event_type=event_type, candidate_violation=candidate_violation,
#             llm_invoked=True, status=status, role=role, confidence=0,
#             reason=reason, skipped=False,
#         )
#         return {
#             "verified":    fail_open,
#             "status":      status,
#             "role":        role,
#             "confidence":  0,
#             "reason":      reason,
#             "skipped":     False,
#             "llm_invoked": True,
#         }

#     parsed = _safe_parse(raw_text)
#     if parsed is None:
#         fail_open = LLM_VERIFICATION_FAIL_OPEN
#         status, role = _status_role(fail_open, "Unknown")
#         reason = "Model response could not be parsed as valid JSON."
#         log.warning(
#             "LLM returned unparseable JSON for %s (%s) — %s",
#             log_label, candidate_violation,
#             "failing OPEN" if fail_open else "failing CLOSED (rejected)",
#         )
#         _log_verification_json(
#             label=log_label, event_type=event_type, candidate_violation=candidate_violation,
#             llm_invoked=True, status=status, role=role, confidence=0,
#             reason=reason, skipped=False, raw_response=raw_text,
#         )
#         return {
#             "verified":    fail_open,
#             "status":      status,
#             "role":        role,
#             "confidence":  0,
#             "reason":      reason,
#             "skipped":     False,
#             "llm_invoked": True,
#         }

#     status, role = _status_role(parsed["verified"], parsed["role"])
#     parsed["skipped"]     = False
#     parsed["llm_invoked"] = True
#     parsed["status"]      = status
#     parsed["role"]        = role
#     print(f"[llm_verifier] PARSED verdict for {log_label!r} "
#           f"(candidate={candidate_violation}): {parsed}")
#     _log_verification_json(
#         label=log_label, event_type=event_type, candidate_violation=candidate_violation,
#         llm_invoked=True, status=status, role=role, confidence=parsed["confidence"],
#         reason=parsed["reason"], skipped=False,
#     )
#     return parsed


# # ── RSL Hand Brake — single-frame entry point ───────────────────────────────
# #
# # Verifies the SAME already-confirmed hand-raise/signal frame N only — no
# # neighbouring-frame window is read or sent (the old [N-2 .. N+2] window
# # logic has been removed). One image, one Qwen-VL call, exactly like
# # verify_frame() above. If the LLM confirms the candidate, the hand-on-brake
# # crew member is by definition the Assistant Loco Pilot, so role is always
# # forced to "ALP" here rather than trusted to the model's own guess.

# def verify_rsl_hand_brake_frame(
#     image_bgr: np.ndarray,
#     log_label: str = "",
# ) -> dict:
#     """
#     Verify the RSL Hand Brake candidate from the single already-confirmed
#     hand-raise/signal frame.

#     Parameters
#     ----------
#     image_bgr:
#         The single confirmed hand-raise/signal frame (no neighbouring
#         frames are read or sent).
#     log_label:
#         Free-form label for logging/audit only.

#     Returns
#     -------
#     dict with:
#         verified, status ("TRUE"/"FALSE"), role ("ALP" iff verified,
#         else None), confidence (0-100), reason, skipped, llm_invoked.
#     """
#     if image_bgr is None:
#         return {
#             "verified": False, "status": "FALSE", "role": None, "confidence": 0,
#             "reason": "No frame available for RSL Hand Brake verification.",
#             "skipped": True, "llm_invoked": False,
#         }

#     if not LLM_VERIFICATION_ENABLED:
#         reason = "LLM verification disabled by config."
#         _log_verification_json(
#             label=log_label, event_type="rsl_hand_brake",
#             candidate_violation=Violation.RSL_HAND_BRAKE, llm_invoked=False,
#             status="TRUE", role="ALP", confidence=0, reason=reason, skipped=True,
#         )
#         return {
#             "verified": True, "status": "TRUE", "role": "ALP", "confidence": 0,
#             "reason": reason, "skipped": True, "llm_invoked": False,
#         }

#     prompt_text = build_rsl_hand_brake_prompt()

#     raw_text: Optional[str] = None
#     last_error: Optional[Exception] = None
#     for attempt in range(1, OLLAMA_MAX_RETRIES + 2):
#         try:
#             raw_text = _call_ollama(prompt_text, image_bgr, log_label=log_label)
#             break
#         except Exception as exc:
#             last_error = exc
#             log.warning(
#                 "RSL Hand Brake LLM verify attempt %d/%d failed for %s: %s",
#                 attempt, OLLAMA_MAX_RETRIES + 1, log_label, exc,
#             )

#     if raw_text is None:
#         fail_open = LLM_VERIFICATION_FAIL_OPEN
#         status, role = _status_role(fail_open, "Assistant Loco Pilot")
#         reason = f"LLM verification unavailable: {last_error}"
#         log.error(
#             "RSL Hand Brake LLM verification unavailable for %s after retries: %s — %s",
#             log_label, last_error,
#             "failing OPEN (kept, unverified)" if fail_open else "failing CLOSED (rejected)",
#         )
#         _log_verification_json(
#             label=log_label, event_type="rsl_hand_brake",
#             candidate_violation=Violation.RSL_HAND_BRAKE, llm_invoked=True,
#             status=status, role=role, confidence=0, reason=reason, skipped=False,
#         )
#         return {
#             "verified": fail_open, "status": status, "role": role, "confidence": 0,
#             "reason": reason, "skipped": False, "llm_invoked": True,
#         }

#     parsed = _safe_parse(raw_text)
#     if parsed is None:
#         fail_open = LLM_VERIFICATION_FAIL_OPEN
#         status, role = _status_role(fail_open, "Assistant Loco Pilot")
#         reason = "Model response could not be parsed as valid JSON."
#         log.warning(
#             "RSL Hand Brake LLM returned unparseable JSON for %s — %s",
#             log_label, "failing OPEN" if fail_open else "failing CLOSED (rejected)",
#         )
#         _log_verification_json(
#             label=log_label, event_type="rsl_hand_brake",
#             candidate_violation=Violation.RSL_HAND_BRAKE, llm_invoked=True,
#             status=status, role=role, confidence=0, reason=reason, skipped=False,
#             raw_response=raw_text,
#         )
#         return {
#             "verified": fail_open, "status": status, "role": role, "confidence": 0,
#             "reason": reason, "skipped": False, "llm_invoked": True,
#         }

#     # Handbrake confirmed → role is always ALP (the hand-on-brake crew
#     # member is by definition the Assistant Loco Pilot), regardless of
#     # whatever the model itself may have guessed for role.
#     status, role = _status_role(parsed["verified"], "Assistant Loco Pilot")

#     result = {
#         "verified":    parsed["verified"],
#         "status":      status,
#         "role":        role,
#         "confidence":  parsed["confidence"],
#         "reason":      parsed["reason"],
#         "skipped":     False,
#         "llm_invoked": True,
#     }
#     print(f"[llm_verifier] RSL Hand Brake PARSED verdict for {log_label!r}: {result}")
#     _log_verification_json(
#         label=log_label, event_type="rsl_hand_brake",
#         candidate_violation=Violation.RSL_HAND_BRAKE, llm_invoked=True,
#         status=status, role=role, confidence=parsed["confidence"],
#         reason=parsed["reason"], skipped=False,
#     )
#     return result


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

Every candidate is kept in the output either way and tagged with a
"status" field:
  • status="TRUE"  — the LLM confirmed the violation. "role" is set to
    one of "LP" (Loco Pilot), "ALP" (Assistant Loco Pilot), "BOTH" (both
    crew members caught committing the same violation), or "AMBIGUOUS"
    (violation confirmed but the LLM could not confidently attribute it
    to one specific role).
  • status="FALSE" — the LLM rejected the candidate. "role" is always
    None (no role is ever assigned to a non-violation).

Design notes
------------
* Only violation types with defined criteria in prompt.py (Mobile Phone,
  Drowsiness, Hand Raising, Seat Absence) are sent to the LLM. Any other
  event_type is passed straight through unverified (status="TRUE",
  role="AMBIGUOUS", skipped=True, llm_invoked=False) — unchanged pipeline
  behaviour for those types, since prompt.py currently defines no
  verification criteria for them.
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
from typing import List, Optional

import numpy as np

from prompt import Violation, build_verification_prompt, build_rsl_hand_brake_prompt

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

VALID_ROLES = {"Loco Pilot", "Assistant Loco Pilot", "Both", "Unknown"}

# Canonical short role codes used on the outbound payload / DB, per task
# spec: role is one of "LP" | "ALP" | "BOTH" | "AMBIGUOUS", or None when the
# violation itself was rejected (status FALSE).
ROLE_TO_CODE = {
    "Loco Pilot":           "LP",
    "Assistant Loco Pilot": "ALP",
    "Both":                 "BOTH",
    "Unknown":               "AMBIGUOUS",
}


def _log_verification_json(**fields) -> None:
    """
    Emit one structured JSON line to the terminal for EVERY candidate
    violation that passes through verify_frame(), regardless of outcome.

    This is the audit trail requested for task 2: from the terminal logs
    alone it must always be possible to tell (a) whether the LLM was
    actually invoked for a given frame ("llm_invoked"), and (b) what
    status/role verdict was ultimately assigned to it.
    """
    try:
        print("[LLM_VERIFICATION_LOG] " + json.dumps(fields, default=str))
    except Exception:
        log.exception("Failed to emit LLM verification JSON log")


class VerificationTimeoutError(RuntimeError):
    pass


class OllamaConnectionError(RuntimeError):
    pass


# ── Ollama call ───────────────────────────────────────────────────────────

def _call_ollama(prompt_text: str, image_bgr: np.ndarray, log_label: str = "") -> str:
    """Blocking Ollama chat call with a hard timeout.

    Frame is written to a temp JPEG and passed as a file PATH (not raw
    bytes) — this matches the standalone verify.py tool exactly, which is
    the known-working reference implementation. Some versions of the
    `ollama` python client handle raw bytes for `images` less reliably
    than a path, which can degrade what the model actually sees.
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


def _call_ollama_images(prompt_text: str, images_bgr: List[np.ndarray], log_label: str = "") -> str:
    """Blocking Ollama chat call with MULTIPLE images attached to one message.

    Used ONLY by the RSL Hand Brake verification flow (see
    detector/rsl_hand_brake_verifier.py) to send [full_frame, zoomed console
    crop] — two SPATIAL views of the SAME single already-confirmed frame,
    not different points in time. This lets the model see a small or
    partially-occluded hand near the console more clearly, without a second
    frame in time ever being read. Frames are written to temp JPEGs and
    passed as file paths (same reasoning as _call_ollama).
    """
    import cv2
    import os
    import tempfile
    from ollama import Client

    tmp_paths: List[str] = []
    try:
        for img in images_bgr:
            fd, tmp_path = tempfile.mkstemp(suffix=".jpg", prefix="llmverify_rsl_")
            os.close(fd)
            ok = cv2.imwrite(tmp_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not ok:
                raise ValueError("Failed to JPEG-encode a frame for RSL LLM verification")
            tmp_paths.append(tmp_path)

        print(f"[llm_verifier] Calling Ollama (multi-image) model={OLLAMA_MODEL!r} "
              f"host={OLLAMA_HOST or 'default(127.0.0.1:11434)'} "
              f"label={log_label!r} images={tmp_paths}")

        client = Client(host=OLLAMA_HOST) if OLLAMA_HOST else Client()

        def _do_call() -> str:
            response = client.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt_text, "images": tmp_paths}],
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
                print(f"[llm_verifier] RAW response (multi-image) for {log_label!r}: {raw!r}")
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
        for p in tmp_paths:
            try:
                os.remove(p)
            except OSError:
                pass


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

def _status_role(verified: bool, raw_role: str) -> tuple[str, Optional[str]]:
    """
    Map an internal (verified: bool, raw_role: str) pair to the outbound
    task-spec contract:
        verified True  → status="TRUE",  role="LP"|"ALP"|"BOTH"|"AMBIGUOUS"
        verified False → status="FALSE", role=None
    """
    if not verified:
        return "FALSE", None
    return "TRUE", ROLE_TO_CODE.get(raw_role, "AMBIGUOUS")


def verify_frame(image_bgr: np.ndarray, event_type: str, log_label: str = "") -> dict:
    """
    Verify one candidate-violation frame with the LLM.

    Returns a dict:
        {
          "verified":    bool,   # kept for internal/backward-compat checks
          "status":      "TRUE" | "FALSE",
          "role":        "LP" | "ALP" | "BOTH" | "AMBIGUOUS" | None,
                         # None whenever status == "FALSE" — no role is
                         # ever assigned to a rejected candidate.
          "confidence":  int 0-100,
          "reason":      str,
          "skipped":     bool,  # True if this event_type has no defined LLM
                                 # verification criteria, or verification is
                                 # disabled — status is always "TRUE" in that
                                 # case (unverified pass-through, unchanged
                                 # pipeline behaviour), role "AMBIGUOUS".
          "llm_invoked": bool,  # True iff an actual model call was made
                                 # (i.e. this candidate was NOT skipped).
        }

    Every call to this function emits exactly one
    "[LLM_VERIFICATION_LOG] {...}" JSON line to the terminal — task 2's
    audit trail — regardless of which branch below is taken.
    """
    if not LLM_VERIFICATION_ENABLED:
        status, role = "TRUE", "AMBIGUOUS"
        _log_verification_json(
            label=log_label, event_type=event_type, candidate_violation=None,
            llm_invoked=False, status=status, role=role, confidence=0,
            reason="LLM verification disabled by config.", skipped=True,
        )
        return {
            "verified": True, "status": status, "role": role, "confidence": 0,
            "reason": "LLM verification disabled by config.", "skipped": True,
            "llm_invoked": False,
        }

    candidate_violation = EVENT_TYPE_TO_PROMPT_VIOLATION.get((event_type or "").lower())
    if candidate_violation is None:
        status, role = "TRUE", "AMBIGUOUS"
        reason = f"No LLM verification criteria defined for event_type={event_type!r}."
        _log_verification_json(
            label=log_label, event_type=event_type, candidate_violation=None,
            llm_invoked=False, status=status, role=role, confidence=0,
            reason=reason, skipped=True,
        )
        return {
            "verified": True, "status": status, "role": role, "confidence": 0,
            "reason": reason, "skipped": True, "llm_invoked": False,
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

    if raw_text is None:
        fail_open = LLM_VERIFICATION_FAIL_OPEN
        status, role = _status_role(fail_open, "Unknown")
        reason = f"LLM verification unavailable: {last_error}"
        log.error(
            "LLM verification unavailable for %s (%s) after retries: %s — %s",
            log_label, candidate_violation, last_error,
            "failing OPEN (kept, unverified)" if fail_open else "failing CLOSED (rejected)",
        )
        _log_verification_json(
            label=log_label, event_type=event_type, candidate_violation=candidate_violation,
            llm_invoked=True, status=status, role=role, confidence=0,
            reason=reason, skipped=False,
        )
        return {
            "verified":    fail_open,
            "status":      status,
            "role":        role,
            "confidence":  0,
            "reason":      reason,
            "skipped":     False,
            "llm_invoked": True,
        }

    parsed = _safe_parse(raw_text)
    if parsed is None:
        fail_open = LLM_VERIFICATION_FAIL_OPEN
        status, role = _status_role(fail_open, "Unknown")
        reason = "Model response could not be parsed as valid JSON."
        log.warning(
            "LLM returned unparseable JSON for %s (%s) — %s",
            log_label, candidate_violation,
            "failing OPEN" if fail_open else "failing CLOSED (rejected)",
        )
        _log_verification_json(
            label=log_label, event_type=event_type, candidate_violation=candidate_violation,
            llm_invoked=True, status=status, role=role, confidence=0,
            reason=reason, skipped=False, raw_response=raw_text,
        )
        return {
            "verified":    fail_open,
            "status":      status,
            "role":        role,
            "confidence":  0,
            "reason":      reason,
            "skipped":     False,
            "llm_invoked": True,
        }

    status, role = _status_role(parsed["verified"], parsed["role"])
    parsed["skipped"]     = False
    parsed["llm_invoked"] = True
    parsed["status"]      = status
    parsed["role"]        = role
    print(f"[llm_verifier] PARSED verdict for {log_label!r} "
          f"(candidate={candidate_violation}): {parsed}")
    _log_verification_json(
        label=log_label, event_type=event_type, candidate_violation=candidate_violation,
        llm_invoked=True, status=status, role=role, confidence=parsed["confidence"],
        reason=parsed["reason"], skipped=False,
    )
    return parsed


# ── RSL Hand Brake — single-frame entry point (full frame + zoom crop) ──────
#
# Verifies the SAME already-confirmed hand-raise/signal frame N only — no
# neighbouring frame in TIME is ever read or sent (the old [N-2 .. N+2]
# window logic has been removed). To help the model see a small or
# partially-occluded hand near the console, TWO spatial views of that ONE
# frame are sent together in a single Qwen-VL call: the full frame, and a
# zoomed-in crop of the console/lever area cut from that identical frame
# (see detector/rsl_hand_brake_verifier.py::make_console_zoom_crop). If the
# LLM confirms the candidate, the hand-on-brake crew member is by
# definition the Assistant Loco Pilot, so role is always forced to "ALP"
# here rather than trusted to the model's own guess.

def verify_rsl_hand_brake_frame(
    image_bgr: np.ndarray,
    zoom_crop_bgr: Optional[np.ndarray] = None,
    log_label: str = "",
) -> dict:
    """
    Verify the RSL Hand Brake candidate from the single already-confirmed
    hand-raise/signal frame.

    Parameters
    ----------
    image_bgr:
        The single confirmed hand-raise/signal frame (no neighbouring
        frames in time are read or sent).
    zoom_crop_bgr:
        Optional zoomed-in crop of the console/lever area, cut from that
        SAME frame (see extract_signal_frame / make_console_zoom_crop in
        detector/rsl_hand_brake_verifier.py). When provided, it is sent
        alongside the full frame in one call so the model can see a small
        or partially-occluded hand more clearly. When None, only the full
        frame is sent (single-image call, same as before).
    log_label:
        Free-form label for logging/audit only.

    Returns
    -------
    dict with:
        verified, status ("TRUE"/"FALSE"), role ("ALP" iff verified,
        else None), confidence (0-100), reason, skipped, llm_invoked.
    """
    if image_bgr is None:
        return {
            "verified": False, "status": "FALSE", "role": None, "confidence": 0,
            "reason": "No frame available for RSL Hand Brake verification.",
            "skipped": True, "llm_invoked": False,
        }

    if not LLM_VERIFICATION_ENABLED:
        reason = "LLM verification disabled by config."
        _log_verification_json(
            label=log_label, event_type="rsl_hand_brake",
            candidate_violation=Violation.RSL_HAND_BRAKE, llm_invoked=False,
            status="TRUE", role="ALP", confidence=0, reason=reason, skipped=True,
        )
        return {
            "verified": True, "status": "TRUE", "role": "ALP", "confidence": 0,
            "reason": reason, "skipped": True, "llm_invoked": False,
        }

    prompt_text = build_rsl_hand_brake_prompt()
    images = [image_bgr] if zoom_crop_bgr is None else [image_bgr, zoom_crop_bgr]

    raw_text: Optional[str] = None
    last_error: Optional[Exception] = None
    for attempt in range(1, OLLAMA_MAX_RETRIES + 2):
        try:
            raw_text = _call_ollama_images(prompt_text, images, log_label=log_label)
            break
        except Exception as exc:
            last_error = exc
            log.warning(
                "RSL Hand Brake LLM verify attempt %d/%d failed for %s: %s",
                attempt, OLLAMA_MAX_RETRIES + 1, log_label, exc,
            )

    if raw_text is None:
        fail_open = LLM_VERIFICATION_FAIL_OPEN
        status, role = _status_role(fail_open, "Assistant Loco Pilot")
        reason = f"LLM verification unavailable: {last_error}"
        log.error(
            "RSL Hand Brake LLM verification unavailable for %s after retries: %s — %s",
            log_label, last_error,
            "failing OPEN (kept, unverified)" if fail_open else "failing CLOSED (rejected)",
        )
        _log_verification_json(
            label=log_label, event_type="rsl_hand_brake",
            candidate_violation=Violation.RSL_HAND_BRAKE, llm_invoked=True,
            status=status, role=role, confidence=0, reason=reason, skipped=False,
        )
        return {
            "verified": fail_open, "status": status, "role": role, "confidence": 0,
            "reason": reason, "skipped": False, "llm_invoked": True,
        }

    parsed = _safe_parse(raw_text)
    if parsed is None:
        fail_open = LLM_VERIFICATION_FAIL_OPEN
        status, role = _status_role(fail_open, "Assistant Loco Pilot")
        reason = "Model response could not be parsed as valid JSON."
        log.warning(
            "RSL Hand Brake LLM returned unparseable JSON for %s — %s",
            log_label, "failing OPEN" if fail_open else "failing CLOSED (rejected)",
        )
        _log_verification_json(
            label=log_label, event_type="rsl_hand_brake",
            candidate_violation=Violation.RSL_HAND_BRAKE, llm_invoked=True,
            status=status, role=role, confidence=0, reason=reason, skipped=False,
            raw_response=raw_text,
        )
        return {
            "verified": fail_open, "status": status, "role": role, "confidence": 0,
            "reason": reason, "skipped": False, "llm_invoked": True,
        }

    # Handbrake confirmed → role is always ALP (the hand-on-brake crew
    # member is by definition the Assistant Loco Pilot), regardless of
    # whatever the model itself may have guessed for role.
    status, role = _status_role(parsed["verified"], "Assistant Loco Pilot")

    result = {
        "verified":    parsed["verified"],
        "status":      status,
        "role":        role,
        "confidence":  parsed["confidence"],
        "reason":      parsed["reason"],
        "skipped":     False,
        "llm_invoked": True,
    }
    print(f"[llm_verifier] RSL Hand Brake PARSED verdict for {log_label!r}: {result}")
    _log_verification_json(
        label=log_label, event_type="rsl_hand_brake",
        candidate_violation=Violation.RSL_HAND_BRAKE, llm_invoked=True,
        status=status, role=role, confidence=parsed["confidence"],
        reason=parsed["reason"], skipped=False,
    )
    return result