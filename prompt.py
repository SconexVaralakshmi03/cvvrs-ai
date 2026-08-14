"""
prompt.py
=========

Prompt construction for the Qwen2.5-VL verification model.

Design philosophy
------------------
The model is used strictly as a *verifier*, not a detector. The
YOLO/MediaPipe pipeline (main.py / detector/*) has already produced a
*candidate* violation from a single frame. The only job of the vision
language model here is to look at that single frame and decide whether the
visual evidence genuinely supports the candidate violation, or whether it
should be rejected.

Because false positives in a railway safety system carry real operational
consequences (unwarranted disciplinary action, alarm fatigue, loss of trust
in the system), every prompt is written to bias the model heavily toward
rejection whenever there is any doubt. The prompt repeatedly and explicitly
forbids guessing, inferring context outside the single image, or inventing
violations.

Integrated into the main journey pipeline via analyzer.py + llm_verifier.py:
every candidate violation frame is run through this verification prompt
before it is uploaded to S3 or included in the completion payload sent to
the Java backend.

Prompt isolation (external prompt files)
-----------------------------------------
The natural-language prompt text and JSON response schemas are NOT
hardcoded in this file. They live under the sibling ``prompts/`` directory
(resolved relative to this file, so it works regardless of the process's
current working directory / entry point):

    prompts/
        base_rules.txt              -- shared, non-negotiable base rules
        role_rules.txt              -- shared LP/ALP role-identification rules
        seat_absence/role_rules.txt -- Seat Absence's role-rules override
        <violation_dir>/prompt.txt  -- violation-specific verification criteria
        <violation_dir>/schema.json -- violation-specific JSON response schema
                                        (holds a single "schema_text" string;
                                        the literal token "%%VIOLATION%%",
                                        where present, is substituted with
                                        the actual candidate_violation name
                                        at build time)

This keeps every violation's prompt independently editable: changing
``prompts/mobile_phone/prompt.txt`` (or its ``schema.json``) cannot affect
``hand_raising``, ``drowsiness``, ``seat_absence``, or ``rsl_hand_brake``,
since each is read from its own file and there is no shared/hardcoded copy
of any violation-specific text anywhere in the Python code. Files are read
once at import time (module load), exactly as the previous in-code string
constants were only ever built once.

Structured per-type JSON schemas (Mobile Phone / Hand Raising / Seat
Absence)
------------------------------------------------------------------------
In addition to the always-present "verified" / "role" / "confidence" /
"reason" fields, these three violation types request a handful of
EXTRA structured boolean/enum fields specific to that violation (e.g.
"device_hardware_visible" for Mobile Phone, "hand_is_elevated_in_air" for
Hand Raising, "body_part_location" for Seat Absence). The prose criteria
don't change what the model is told to look for, only what it reports back
about what it saw. They exist so llm_verifier.py's deterministic override
layer can cross-check the model's own top-level "verified" claim against
its own structured observations (e.g. reject a "verified: true" Hand
Raising candidate whose own hand_is_elevated_in_air field says false)
instead of trusting a single free-form "verified" boolean at face value.
Drowsiness and RSL Hand Brake keep their own (simpler) schemas -- no
override layer is defined for them in llm_verifier.py.
"""

from __future__ import annotations

import json
import os


class Violation:
    """String constants for the candidate violation categories."""

    MOBILE_PHONE = "Mobile Phone"
    DROWSINESS = "Drowsiness"
    HAND_RAISING = "Hand Raising"
    SEAT_ABSENCE = "Seat Absence"  # Loco cabin left unmanned -- now verified.
    RSL_HAND_BRAKE = "RSL Hand Brake Held"  # Multi-frame signal + brake check.


# --------------------------------------------------------------------------- #
# External prompt-file loading
# --------------------------------------------------------------------------- #
#
# Resolved relative to THIS file (not the process cwd), so the app can find
# the prompt files no matter where it's launched from.

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROMPTS_DIR = os.path.join(_THIS_DIR, "prompts")

# Token substituted (plain str.replace, not str.format) inside a
# violation's schema_text with the actual candidate_violation string. A
# plain token/replace is used instead of str.format() so schema files can
# contain literal "{" / "}" JSON-shaped text without needing to escape
# every brace as "{{" / "}}".
_VIOLATION_TOKEN = "%%VIOLATION%%"

# Internal violation constant -> its subdirectory name under prompts/.
_VIOLATION_DIR: dict[str, str] = {
    Violation.MOBILE_PHONE: "mobile_phone",
    Violation.DROWSINESS: "drowsiness",
    Violation.HAND_RAISING: "hand_raising",
    Violation.SEAT_ABSENCE: "seat_absence",
    Violation.RSL_HAND_BRAKE: "rsl_hand_brake",
}


def _read_text_file(*relative_parts: str) -> str:
    """Read and strip a .txt prompt file under prompts/."""
    path = os.path.join(_PROMPTS_DIR, *relative_parts)
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read().strip()


def _read_schema_text(*relative_parts: str) -> str:
    """Read a violation's schema.json and return its "schema_text" field."""
    path = os.path.join(_PROMPTS_DIR, *relative_parts)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return str(data["schema_text"]).strip()


# ── Shared rule blocks (loaded once at import time) ──────────────────────

_BASE_RULES = _read_text_file("base_rules.txt")
_ROLE_RULES = _read_text_file("role_rules.txt")
_SEAT_ABSENCE_ROLE_RULES = _read_text_file("seat_absence", "role_rules.txt")

# ── Violation-specific criteria + schema text (loaded once, one file per
#    violation type -- no violation's prompt text is shared/duplicated) ──

_CRITERIA_BY_VIOLATION: dict[str, str] = {
    violation: _read_text_file(subdir, "prompt.txt")
    for violation, subdir in _VIOLATION_DIR.items()
    if violation != Violation.RSL_HAND_BRAKE  # RSL uses its own builder below
}

_SCHEMA_TEXT_BY_VIOLATION: dict[str, str] = {
    violation: _read_schema_text(subdir, "schema.json")
    for violation, subdir in _VIOLATION_DIR.items()
    if violation != Violation.RSL_HAND_BRAKE  # RSL uses its own builder below
}

_RSL_HAND_BRAKE_CRITERIA = _read_text_file("rsl_hand_brake", "prompt.txt")
_RSL_SCHEMA_TEXT = _read_schema_text("rsl_hand_brake", "schema.json")


def build_rsl_hand_brake_prompt() -> str:
    """
    Build the single-frame (full frame + zoomed console crop of that SAME
    frame) verification prompt for the RSL Hand Brake candidate (see
    detector/rsl_hand_brake_verifier.py). Verifies only the already-
    confirmed hand-raise/signal frame itself -- no neighbouring frame in
    time is used, only two spatial views of the one frame.

    RSL Hand Brake does not use the shared role_rules.txt -- when this
    candidate is verified true, the hand-on-brake crew member is by
    definition the Assistant Loco Pilot, so role is assigned by the caller
    (llm_verifier.py), not the model.
    """
    schema = _RSL_SCHEMA_TEXT.replace(_VIOLATION_TOKEN, Violation.RSL_HAND_BRAKE)
    return "\n\n".join([_BASE_RULES, _RSL_HAND_BRAKE_CRITERIA, schema])


def build_verification_prompt(candidate_violation: str) -> str:
    """
    Build the full text prompt sent to Qwen2.5-VL for a given candidate
    violation type.

    Parameters
    ----------
    candidate_violation:
        One of the verifiable violation constants defined in
        ``Violation`` (Mobile Phone, Drowsiness, Hand Raising, Seat
        Absence).

    Returns
    -------
    str
        The complete prompt text, including base rules, violation-specific
        criteria, role rules, and the required JSON output schema.

    Raises
    ------
    KeyError
        If ``candidate_violation`` does not have defined criteria.
    """
    if candidate_violation not in _CRITERIA_BY_VIOLATION:
        raise KeyError(
            f"No verification criteria defined for violation "
            f"'{candidate_violation}'."
        )

    criteria = _CRITERIA_BY_VIOLATION[candidate_violation]
    role_rules = (
        _SEAT_ABSENCE_ROLE_RULES
        if candidate_violation == Violation.SEAT_ABSENCE
        else _ROLE_RULES
    )
    schema = _SCHEMA_TEXT_BY_VIOLATION[candidate_violation].replace(
        _VIOLATION_TOKEN, candidate_violation
    )

    return "\n\n".join([_BASE_RULES, criteria, role_rules, schema])
