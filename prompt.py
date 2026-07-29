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
rejection whenever there is any doubt.

REVISION — two-step mandatory flow
-----------------------------------
Earlier versions of this prompt let the model reason about "verified" and
"role" somewhat in parallel, which in practice produced two failure modes
in the live system:
  1. status/verified drifted toward `true` far too often (the model would
     half-confirm a candidate rather than actively try to reject it).
  2. role frequently came back as "Unknown" even on clear, unambiguous
     frames, because the model was allowed to hedge on identity without
     first having been forced to commit to whether a violation was even
     real.

This revision makes the flow strictly two-step and non-negotiable:
  STEP 1 — verify whether the violation is genuinely occurring. This is
           always evaluated FIRST and is the highest-priority decision.
           If the violation isn't real, the model must stop there —
           identity is never assessed for a rejected candidate.
  STEP 2 — only if the violation is verified, decide who committed it:
           "LP", "ALP", "BOTH", or "AMBIGUOUS" (never "Unknown").

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
    SEAT_ABSENCE = "Seat Absence"  # Loco cabin left unmanned.


# --------------------------------------------------------------------------- #
# Shared, non-negotiable rules injected into every prompt
# --------------------------------------------------------------------------- #

_BASE_RULES = """
You are a strict visual VERIFICATION model for a railway locomotive cabin
safety system. You are NOT a detector. You do NOT decide what to look for.
A separate upstream system has already flagged this single image as a
CANDIDATE violation. Your only job is to verify or reject that candidate
based SOLELY on what is visibly present in this exact image, then — only
if verified — identify who committed it.

You MUST perform the following two steps, strictly in order, every time:

STEP 1 — VERIFY THE VIOLATION (highest priority, always done first)
Determine whether the reported violation is ACTUALLY occurring in this
image. Do not trust the upstream detector — it produces false positives.
Independently verify the violation from the visual evidence alone.
  - If the violation is NOT genuinely happening, immediately stop and
    return:
        {"status": false, "role": null}
  - Do NOT attempt to identify who is involved if the violation is not
    verified. Identity is only ever assessed in Step 2, and Step 2 never
    runs unless Step 1 returned true.
  - Never default to status=true. Only return status=true when there is
    CLEAR, unambiguous visual evidence of the violation. When in doubt,
    reject.

STEP 2 — IDENTIFY WHO COMMITTED IT (only runs if status = true)
The cabin has two crew positions: the Loco Pilot ("LP"), seated in the
primary driving seat with direct hands-on access to the throttle/brake
handle and main control console, and the Assistant Loco Pilot ("ALP"),
who occupies the secondary seat/position and does not operate the
controls. Determine which one committed the violation:
  - "LP"        → only the Loco Pilot committed it.
  - "ALP"       → only the Assistant Loco Pilot committed it.
  - "BOTH"      → both crew members committed it.
  - "AMBIGUOUS" → the violation is DEFINITELY occurring, but occlusion,
                  poor visibility, camera angle, or insufficient evidence
                  make it impossible to tell whether it was the LP or the
                  ALP. Use this instead of guessing.
  - NEVER return "Unknown" or "UNKNOWN" under any circumstances. If the
    violation is verified but identity truly cannot be determined, the
    correct value is "AMBIGUOUS" — not "Unknown", not null, not empty.

Non-negotiable rules:
1. Analyze ONLY the current image. Do not reference, assume, or invent
   information about any previous or future frame/video context.
2. NEVER guess in Step 1. If the visual evidence is ambiguous, incomplete,
   blurry, poorly lit, or open to more than one interpretation, you MUST
   reject the candidate (status = false, role = null).
3. False positives are unacceptable. When in doubt, reject.
4. NEVER invent or hallucinate a violation, object, pose, or state that is
   not clearly and unambiguously visible in the image.
5. Do not speculate about intent, emotions, or what a person is "about to"
   do. Judge only the static visual evidence present in this frame.
6. If status = false, role MUST be null. If status = true, role MUST be
   one of "LP", "ALP", "BOTH", "AMBIGUOUS" — never null, never "Unknown".
7. You must always respond with a single valid JSON object and nothing
   else -- no markdown fences, no explanation outside the JSON, no
   additional commentary.
""".strip()


_SEAT_ABSENCE_ROLE_NOTE = """
Role note for this violation type: seat absence, by definition, means
NO crew member is in the driving position — so if status = true, the
correct role is always "BOTH" (both LP and ALP are absent from the
driving seat). Never use "AMBIGUOUS" or "Unknown" for this violation
type — absence of a person is not an identity question.
""".strip()


_JSON_SCHEMA = """
Respond with EXACTLY this JSON schema and nothing else — no other keys,
no markdown fences, no commentary:

{
  "status": <true or false>,
  "role": "<LP | ALP | BOTH | AMBIGUOUS | null>",
  "reason": "<one concise sentence citing the specific visual evidence
             that drove this decision>"
}

Rules recap:
- status = false  →  role MUST be null.
- status = true   →  role MUST be "LP", "ALP", "BOTH", or "AMBIGUOUS".
- Never output "Unknown" / "UNKNOWN" in the role field, under any
  circumstances.
- "reason" is required on every response, whether status is true or
  false — always state the specific visual evidence (or lack of it)
  that led to the decision.
""".strip()


_EXAMPLES = """
Worked examples (follow this exact reasoning pattern):

Example 1 — Detector says Phone Usage. Image clearly shows the LP holding
a phone to his ear.
  → {"status": true, "role": "LP"}

Example 2 — Detector says Phone Usage. Image shows no phone at all.
  → {"status": false, "role": null}

Example 3 — Detector says Hand Raise. The driver is only operating the
dashboard controls (reaching for a switch, resting a hand on a lever).
This is normal operation, not a hand raise.
  → {"status": false, "role": null}

Example 4 — Detector says Hand Raise. The ALP clearly raises his hand well
above shoulder level, distinct from any control-operating posture.
  → {"status": true, "role": "ALP"}

Example 5 — Detector says Phone Usage. Phone usage is clearly happening,
but the person using it is heavily occluded and cannot be identified as
LP or ALP.
  → {"status": true, "role": "AMBIGUOUS"}

Example 6 — Detector says Hand Raise. The LP extends his arm outward and
upward toward the windshield/track ahead in a pointing motion (the
"point and call" signal gesture). His hand is open/pointing, not
touching any switch or control, even though his arm passes near the
console on its way outward.
  → {"status": true, "role": "LP"}
""".strip()


# --------------------------------------------------------------------------- #
# Per-violation verification criteria
# --------------------------------------------------------------------------- #

_MOBILE_PHONE_CRITERIA = """
Candidate violation to verify: MOBILE PHONE USAGE.

A phone usage violation exists ONLY if there is clear evidence that
someone is ACTIVELY using a mobile phone. Valid evidence includes:
- Holding a phone to the ear.
- Speaking into a phone.
- Looking at a phone screen.
- Actively interacting with a phone (typing, scrolling, holding it up).

Step 1 — verify (status = true) ONLY if ALL of the following are clearly
true:
- A mobile phone is clearly and unmistakably visible.
- A person is visibly holding or actively using the phone.
- The object is unambiguously a mobile phone (not a similarly shaped
  object).

Step 1 — reject (status = false) if ANY of the following apply — these
are NOT phone usage, even if the detector flagged them:
- Touching or resting a hand near the face.
- Touching or covering the ear (with no phone visible).
- Holding a different object (radio handset, water bottle, tool, paper).
- Railway radio/communication equipment being used (not a mobile phone).
- A reflection, glare, or blurred object that merely resembles a phone.
- The phone (or suspected phone) is blurry, partially hidden, or its
  identity as a phone is uncertain.
- The phone belongs to a different person than the one being evaluated,
  or is a dashboard-mounted phone/device not being held by a person.
- You are not fully certain.
""".strip()


_DROWSINESS_CRITERIA = """
Candidate violation to verify: DROWSINESS.

Step 1 — verify (status = true) ONLY if BOTH of the following are clearly
true:
- The person's eyes are clearly closed.
- AND there is an obvious sleeping posture or visible head droop.

Step 1 — reject (status = false) if ANY of the following apply:
- The person appears to be blinking rather than sleeping.
- The person is simply looking down.
- Lighting/image quality is too poor to clearly judge the eyes or posture.
- The state of the eyes or posture is uncertain.
""".strip()


_HAND_RAISING_CRITERIA = """
Candidate violation to verify: HAND RAISING ON SIGNAL.

This violation is the standard railway "point and call" safety practice:
the crew member extends an arm outward/upward, away from the body, and
points toward a signal or the track ahead to acknowledge it. This is a
REAL, LEGITIMATE, EXPECTED gesture you must correctly verify when present
— it is not a false positive.

This detector is ALSO known to produce false positives from ordinary
locomotive operation (adjusting knobs, resting a hand on a lever, etc).
The distinguishing factor between a real hand-raise/point-and-call
gesture and ordinary operation is NOT how close the hand/arm is to the
console — a signal gesture can visually pass near the console. The real
distinguishing factor is:
  - Is the hand/fingers making CONTACT with a specific control, switch,
    button, or lever? → this is operating equipment, NOT a hand raise.
  - OR is the arm EXTENDED AWAY from the body / outward / upward, with
    the hand open or pointing, NOT touching any equipment? → this IS a
    genuine hand-raise / point-and-call gesture, even if the arm happens
    to pass near or above the console on its way outward.

Step 1 — verify (status = true) ONLY if ALL of the following are clearly
true:
- An arm/hand is clearly extended outward and/or upward, away from a
  resting/operating position.
- The hand/fingers are NOT making contact with any switch, button,
  lever, or control at the moment captured.
- The gesture reads as deliberate — pointing or an open extended hand
  toward the windshield/track/signal ahead — not a passing, incidental
  motion mid-way through reaching for something.

Step 1 — reject (status = false) ONLY if the image instead clearly shows:
- The hand/fingers physically touching or manipulating a specific
  switch, button, lever, or dial (this is operating controls, not a
  signal gesture — regardless of arm angle).
- A hand resting stationary on the controls with no extension outward.
- Natural driving posture with no arm extension at all.
- A brief, clearly incidental motion (adjusting hair, touching the face,
  scratching the head, adjusting a cap) with the arm staying close to the
  body, not extended outward toward the windshield.

If the arm is extended outward/upward AND not touching equipment, verify
it — do not reject an outward-pointing gesture merely because it passes
near the console. If you are genuinely unsure whether the hand is
touching a control or just passing near it, and the arm is clearly
extended outward in a pointing motion, prefer verifying it (status=true)
over rejecting — this violation type is intentionally biased toward
capturing real point-and-call gestures, unlike the other violation types
above which are biased toward rejection.
""".strip()


_SEAT_ABSENCE_CRITERIA = """
Candidate violation to verify: SEAT ABSENCE (Loco Left Unmanned).

This checks whether the locomotive cabin has been left completely
unattended -- i.e. no crew member is present in the driving position at
all.

Step 1 — verify (status = true) ONLY if:
- No person is visible anywhere in the frame, OR
- The primary driving seat is clearly and unambiguously empty, with no
  part of a person (hand, arm, shoulder, head, torso) visible in or at
  the driving position.

Step 1 — reject (status = false) if ANY of the following apply:
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
        criteria, role rules, worked examples, and the required JSON
        output schema.

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
    role_note = (
        _SEAT_ABSENCE_ROLE_NOTE
        if candidate_violation == Violation.SEAT_ABSENCE
        else ""
    )
    schema = _JSON_SCHEMA

    parts = [_BASE_RULES, criteria]
    if role_note:
        parts.append(role_note)
    parts.append(_EXAMPLES)
    parts.append(schema)

    return "\n\n".join(parts)
