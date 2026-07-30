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

Before assigning a role, first determine who is operating the locomotive.
The role must be identified independently from the violation itself.

STEP 1 – Identify the Loco Pilot (LP)

The Loco Pilot is the crew member who is clearly and visibly:

- Seated in the primary driving seat directly in front of the main driving console.
- Facing the front windshield while operating the locomotive.
- Actively controlling or immediately positioned to control the locomotive.
- Having one or both hands on, or immediately adjacent to, the throttle, brake handle, or main driving controls.
- Occupying the driver's position throughout the visible scene.

Only this person should be classified as "Loco Pilot".

STEP 2 – Identify the Assistant Loco Pilot (ALP)

Any other crew member visible in the cabin who is NOT occupying the primary
driving position must be classified as "Assistant Loco Pilot".

This includes a crew member who is:

- Sitting in the secondary seat.
- Sitting beside or behind the driver.
- Standing anywhere inside the cabin.
- Walking inside the cabin.
- Looking outside.
- Holding a phone.
- Sleeping.
- Performing any other activity away from the primary driving controls.

The activity being performed DOES NOT determine the role.

STEP 3 – Match the violation

After identifying the LP and ALP, determine which person is committing
the verified violation.

If the violating person is the identified driver,
role = "Loco Pilot".

If the violating person is any other crew member,
role = "Assistant Loco Pilot".

If BOTH the Loco Pilot and Assistant Loco Pilot are clearly and independently
committing the SAME verified violation simultaneously, set role = "Both".

IMPORTANT RULES

- Never assume the person closest to the camera is the Loco Pilot.
- Never assume the largest or most prominent person in the image is the Loco Pilot.
- Never assign the Loco Pilot role solely because a person is holding a phone,
  appears drowsy, or is raising a hand.
- Never infer the role based on the candidate violation.
- Determine the driver's position first, then identify who committed the violation.
- If the driver's position cannot be identified with high confidence, or if the
  violating person cannot be confidently matched to either crew member,
  set role = "Unknown".
- False role assignments are unacceptable. When in doubt, return "Unknown"
  instead of guessing.
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
  "role": "<Loco Pilot | Assistant Loco Pilot | Both | Unknown>",
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

This violation represents the standard railway signal acknowledgement
gesture performed by a crew member while driving.

Verify (verified = true) ONLY if ALL of the following conditions are clearly
visible in this single image:

- At least ONE hand is clearly raised or intentionally extended toward the
  front windshield / railway signal.
- The raised hand clearly belongs to the person being evaluated.
- The gesture is deliberate and appears to acknowledge an external railway
  signal rather than interact with the locomotive controls.
- The raised hand is clearly visible and distinguishable from normal driving
  movements.

A raised hand does NOT need to be above the person's head. A forward-raised
or forward-extended arm directed toward the front windshield is sufficient
provided it is clearly a signal acknowledgement gesture.

Reject (verified = false) if ANY of the following apply:

- The hand is resting on the control console.
- The hand is holding or operating a lever, switch, throttle, brake, or
  any locomotive control.
- The arm movement appears to be part of normal driving operations.
- The person is reaching for an object inside the cabin.
- The gesture appears to be stretching, scratching, adjusting clothing,
  adjusting hair, adjusting a cap, or any other non-signal action.
- The raised hand is partially hidden, blurred, or not clearly visible.
- The direction or purpose of the gesture cannot be confidently determined.
- There is any reasonable ambiguity about whether the gesture is a railway
  signal acknowledgement.

False positives are unacceptable. If the gesture is not clearly identifiable
as a railway signal acknowledgement from this single frame, reject the
candidate (verified = false).
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