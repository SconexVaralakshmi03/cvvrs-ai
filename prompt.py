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
"""

from __future__ import annotations

class Violation:
    """String constants for the candidate violation categories."""

    MOBILE_PHONE = "Mobile Phone"
    DROWSINESS = "Drowsiness"
    HAND_RAISING = "Hand Raising"
    SEAT_ABSENCE = "Seat Absence"  # Loco cabin left unmanned -- now verified.


# --------------------------------------------------------------------------- #
# Shared, non-negotiable rules injected into every prompt
# --------------------------------------------------------------------------- #

_BASE_RULES = """
You are a strict visual VERIFICATION model for a railway locomotive cabin
safety system. You are NOT a detector. You do NOT decide what to look for.
A separate upstream system has already flagged this single image as a
CANDIDATE violation. Your only job is to verify or reject that candidate
based SOLELY on what is visibly present in this exact image.

Non-negotiable rules:
1. Analyze ONLY the current image. Do not reference, assume, or invent
   information about any previous or future frame/video context.
2. NEVER guess. If the visual evidence is ambiguous, incomplete, blurry,
   poorly lit, or open to more than one interpretation, you MUST reject
   the candidate (verified = false).
3. False positives are unacceptable. When in doubt, reject.
4. NEVER invent or hallucinate a violation, object, pose, or state that is
   not clearly and unambiguously visible in the image.
5. Do not speculate about intent, emotions, or what a person is "about to"
   do. Judge only the static visual evidence present in this frame.
6. You must always respond with a single valid JSON object and nothing
   else -- no markdown fences, no explanation outside the JSON, no
   additional commentary.
""".strip()


_ROLE_RULES = """
Role identification (only if verified = true):

The cabin normally has two crew positions: the Loco Pilot (LP), who is
seated in the primary driving seat with direct hands-on access to the
throttle/brake handle and the main control console, and the Assistant Loco
Pilot (ALP), who occupies the secondary seat/position and does not operate
the controls.

Being merely near the console, leaning over it, glancing at it, or
standing beside it is NOT enough to be classified as Loco Pilot. Only the
person who is:
  - seated in the primary driving seat, AND
  - positioned with their hand(s) on or immediately at the throttle/brake
    handle in a driving posture (not just resting nearby),
should be classified as "Loco Pilot".

Any other crew member visible in the frame -- including someone standing,
seated in the secondary seat, walking through the cabin, or interacting
with anything other than the primary control console -- must be
classified as "Assistant Loco Pilot".

Do NOT assume the person doing the flagged action (e.g. using a phone) is
automatically the Loco Pilot just because they are the most visually
prominent person in the frame, or because they are close to the panel.
Judge role independently from seat position and control-handling posture,
not from who happens to be committing the candidate violation.

If two or more people are present and you cannot clearly tell which one
occupies the primary driving seat, OR if you cannot confidently match the
flagged action to one specific person's seat position, set role to
"Unknown". Never guess the role.
""".strip()


_SEAT_ABSENCE_ROLE_RULES = """
Role identification: not applicable for this violation type. Always set
role to "Unknown" regardless of the verdict, since by definition no crew
member's role can be identified when verifying whether the cabin is
unmanned.
""".strip()


_JSON_SCHEMA = """
Respond with EXACTLY this JSON schema and nothing else:

{{
  "verified": <true or false>,
  "candidate_violation": "{violation}",
  "role": "<Loco Pilot | Assistant Loco Pilot | Unknown>",
  "confidence": <integer 0-100>,
  "reason": "<one concise sentence citing the specific visual evidence>"
}}

If verified is false, set role to "Unknown" and confidence to 0.
""".strip()


# --------------------------------------------------------------------------- #
# Per-violation verification criteria
# --------------------------------------------------------------------------- #

_MOBILE_PHONE_CRITERIA = """
Candidate violation to verify: MOBILE PHONE USAGE.

Verify (verified = true) ONLY if ALL of the following are clearly true:
- A mobile phone is clearly and unmistakably visible.
- A person is visibly holding the phone.
- The object is unambiguously a mobile phone (not a similarly shaped
  object).

Reject (verified = false) if ANY of the following apply:
- The phone (or suspected phone) is blurry or partially hidden.
- The object's identity as a phone is uncertain.
- The phone belongs to a different person than the one being evaluated.
- It is a dashboard-mounted phone / device rather than being held by a
  person.
- You are not fully certain.
""".strip()


_DROWSINESS_CRITERIA = """
Candidate violation to verify: DROWSINESS.

Verify (verified = true) ONLY if BOTH of the following are clearly true:
- The person's eyes are clearly closed.
- AND there is an obvious sleeping posture or visible head droop.

Reject (verified = false) if ANY of the following apply:
- The person appears to be blinking rather than sleeping.
- The person is simply looking down.
- Lighting/image quality is too poor to clearly judge the eyes or posture.
- The state of the eyes or posture is uncertain.
""".strip()


_HAND_RAISING_CRITERIA = """
Candidate violation to verify: HAND RAISING ON SIGNAL.

Verify (verified = true) ONLY if ALL of the following are clearly true:
- A hand is clearly raised.
- The raised hand clearly belongs to the person being evaluated.
- The gesture is clearly directed toward the front windshield (i.e. a
  signal-acknowledgement gesture), not an incidental motion.

Reject (verified = false) if the gesture instead looks like:
- Stretching.
- Scratching.
- Adjusting hair.
- Adjusting a cap.
- A resting hand.
- Any other uncertain or ambiguous pose.
""".strip()


_SEAT_ABSENCE_CRITERIA = """
Candidate violation to verify: SEAT ABSENCE (Loco Left Unmanned).

This checks whether the locomotive cabin has been left completely
unattended -- i.e. no crew member is present in the driving position at
all.

Verify (verified = true) ONLY if:
- No person is visible anywhere in the frame, OR
- The primary driving seat is clearly and unambiguously empty, with no
  part of a person (hand, arm, shoulder, head, torso) visible in or at
  the driving position.

Reject (verified = false) if ANY of the following apply:
- Any person, or any clearly identifiable body part of a person, is
  visible anywhere in the frame -- even if partially visible, at the
  edge of the frame, in the background, or momentarily leaning out of
  the seat.
- The driving seat's occupancy is unclear due to camera angle, occlusion,
  poor lighting, or motion blur.
- You are not fully certain the cabin is empty.

Because this violation carries serious safety implications, treat any
visible trace of a person -- however partial -- as sufficient to reject
the candidate.
""".strip()


_CRITERIA_BY_VIOLATION: dict[str, str] = {
    Violation.MOBILE_PHONE: _MOBILE_PHONE_CRITERIA,
    Violation.DROWSINESS: _DROWSINESS_CRITERIA,
    Violation.HAND_RAISING: _HAND_RAISING_CRITERIA,
    Violation.SEAT_ABSENCE: _SEAT_ABSENCE_CRITERIA,
}


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
    schema = _JSON_SCHEMA.format(violation=candidate_violation)

    return "\n\n".join([_BASE_RULES, criteria, role_rules, schema])
