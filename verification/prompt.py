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

Prompts now live in prompts/*.txt (file-based prompts)
--------------------------------------------------------------------------
Previously every prompt was assembled in this module from Python string
fragments (_BASE_RULES, _ROLE_RULES, per-violation criteria, JSON schema
templates, etc.) and returned by build_verification_prompt() /
build_rsl_hand_brake_prompt().

That is no longer the case. Each violation type now has its own standalone,
fully self-contained prompt file under prompts/ alongside this module:

    prompts/mobile_phone.txt
    prompts/drowsiness.txt
    prompts/hand_raising.txt
    prompts/seat_absence.txt
    prompts/rsl_hand_brake.txt

Each .txt file already contains the complete prompt text sent to the model
as-is (base rules + role-identification rules + violation-specific
criteria + the required JSON output schema, including the literal
candidate_violation string) -- exactly matching the standalone
llamaverify.py reference tool's load_prompt() convention, so a prompt can
be tweaked by editing the .txt file directly without touching any Python
code or redeploying application logic.

This module is now just a thin, cached loader over those files. The public
API is UNCHANGED on purpose -- build_verification_prompt(candidate_violation)
and build_rsl_hand_brake_prompt() still take the same arguments and return
the same kind of plain prompt string they always did, so llm_verifier.py
(and anything else importing from prompt.py) keeps working without any
modification.

Structured per-type JSON schemas (Mobile Phone / Hand Raising / Seat
Absence)
------------------------------------------------------------------------
In addition to the always-present "verified" / "role" / "confidence" /
"reason" fields, these three violation types' prompt files request a
handful of EXTRA structured boolean/enum fields specific to that violation
(e.g. "device_hardware_visible" for Mobile Phone, "hand_is_elevated_in_air"
for Hand Raising, "body_part_location" for Seat Absence). These extra
fields exist so llm_verifier.py's deterministic override layer can
cross-check the model's own top-level "verified" claim against its own
structured observations (e.g. reject a "verified: true" Hand Raising
candidate whose own hand_is_elevated_in_air field says false) instead of
trusting a single free-form "verified" boolean at face value. Drowsiness
and RSL Hand Brake keep their existing (simpler) schemas -- no override
layer is defined for them.
"""

from __future__ import annotations

from pathlib import Path

# Directory holding the standalone prompt .txt files, alongside this
# module -- same convention as the reference llamaverify.py CLI tool.
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class Violation:
    """String constants for the candidate violation categories."""

    MOBILE_PHONE = "Mobile Phone"
    DROWSINESS = "Drowsiness"
    HAND_RAISING = "Hand Raising"
    SEAT_ABSENCE = "Seat Absence"  # Loco cabin left unmanned -- now verified.
    RSL_HAND_BRAKE = "RSL Hand Brake Held"  # Multi-frame signal + brake check.
    # NEW — additive: curve-checking (detector/curve_checking.py). Like
    # Hand Raising, this is a positive/required-procedure event, not a
    # distraction violation -- see prompts/curve_checking.txt.
    CURVE_CHECKING = "Curve Checking"


# --------------------------------------------------------------------------- #
# Violation constant -> prompt filename under prompts/
# --------------------------------------------------------------------------- #

_PROMPT_FILES: dict[str, str] = {
    Violation.MOBILE_PHONE: "mobile_phone.txt",
    Violation.DROWSINESS: "drowsiness.txt",
    Violation.HAND_RAISING: "hand_raising.txt",
    Violation.SEAT_ABSENCE: "seat_absence.txt",
    Violation.RSL_HAND_BRAKE: "rsl_hand_brake.txt",
    Violation.CURVE_CHECKING: "curve_checking.txt",
}

# Only these four go through build_verification_prompt(); RSL Hand Brake
# has its own dedicated builder (build_rsl_hand_brake_prompt) because it
# is sent together with a second (zoomed console crop) image -- same split
# as before this change.
_VERIFIABLE_VIOLATIONS = {
    Violation.MOBILE_PHONE,
    Violation.DROWSINESS,
    Violation.HAND_RAISING,
    Violation.SEAT_ABSENCE,
    # NEW — additive: curve-checking follows the same single-frame,
    # no-extra-image verification path as the four above.
    Violation.CURVE_CHECKING,
}

# Simple in-process cache so each prompt file is only read from disk once
# per process, not on every single frame verification call.
_prompt_file_cache: dict[str, str] = {}


def _load_prompt_file(filename: str) -> str:
    """
    Read (and cache) a prompt file from PROMPTS_DIR.

    Parameters
    ----------
    filename:
        Name of the .txt file under prompts/ (e.g. "seat_absence.txt").

    Returns
    -------
    str
        The full prompt text, exactly as stored in the file (surrounding
        whitespace stripped).

    Raises
    ------
    FileNotFoundError
        If the prompt file does not exist under PROMPTS_DIR.
    """
    if filename in _prompt_file_cache:
        return _prompt_file_cache[filename]

    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}. Expected a prompts/ directory "
            f"alongside prompt.py containing one .txt file per violation "
            f"type (mobile_phone.txt, drowsiness.txt, hand_raising.txt, "
            f"seat_absence.txt, rsl_hand_brake.txt)."
        )

    text = path.read_text(encoding="utf-8").strip()
    _prompt_file_cache[filename] = text
    return text


def build_verification_prompt(candidate_violation: str) -> str:
    """
    Return the full text prompt sent to Qwen2.5-VL for a given candidate
    violation type, loaded from its standalone prompt file under prompts/.

    Parameters
    ----------
    candidate_violation:
        One of the verifiable violation constants defined in
        ``Violation`` (Mobile Phone, Drowsiness, Hand Raising, Seat
        Absence).

    Returns
    -------
    str
        The complete prompt text (base rules, violation-specific
        criteria, role rules, and the required JSON output schema) as
        stored in prompts/<violation>.txt.

    Raises
    ------
    KeyError
        If ``candidate_violation`` does not have a defined prompt file.
    """
    if candidate_violation not in _VERIFIABLE_VIOLATIONS:
        raise KeyError(
            f"No verification criteria defined for violation "
            f"'{candidate_violation}'."
        )

    filename = _PROMPT_FILES[candidate_violation]
    return _load_prompt_file(filename)


def build_rsl_hand_brake_prompt() -> str:
    """
    Return the single-frame (full frame + zoomed console crop of that SAME
    frame) verification prompt for the RSL Hand Brake candidate (see
    detector/rsl_hand_brake_verifier.py), loaded from
    prompts/rsl_hand_brake.txt. Verifies only the already-confirmed
    hand-raise/signal frame itself -- no neighbouring frame in time is
    used, only two spatial views of the one frame.
    """
    filename = _PROMPT_FILES[Violation.RSL_HAND_BRAKE]
    return _load_prompt_file(filename)