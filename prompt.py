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
    RSL_HAND_BRAKE = "RSL Hand Brake"  # NEW -- opposite-hand brake grip during a hand-raise.


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


_HAND_RAISING_CRITERIA = _HAND_RAISING_CRITERIA = """
Candidate violation to verify: HAND RAISING ON SIGNAL.

This violation represents the standard railway signal acknowledgement
performed by railway crew members.

Verify (verified = true) if ALL of the following are clearly visible in this
single image:

- At least one arm is raised or intentionally extended toward the front
  windshield.
- The raised or extended arm clearly belongs to the evaluated person.
- The arm direction is consistent with acknowledging an external railway
  signal.
- The gesture is visually distinguishable from normal resting posture.

A valid signal acknowledgement DOES NOT require:

- the hand to be above the head,
- the elbow to be fully extended,
- the entire hand or fingertips to be visible,
- both hands to be raised.

It is acceptable if:

- only one hand is raised,
- one hand remains on the driving controls,
- part of the raised hand extends outside the camera frame,
- the fingertips are cropped but the arm direction is clearly visible.

Reject (verified = false) only if:

- there is no raised or forward-extended arm,
- the arm is clearly resting,
- the person is reaching for an object inside the cabin,
- the movement is clearly unrelated to signal acknowledgement
  (scratching, adjusting clothing, adjusting hair, stretching, etc.),
- the arm direction cannot be determined because of severe blur or
  occlusion,
- there is insufficient visual evidence to distinguish the gesture.

If the raised arm is clearly directed toward the front windshield,
prefer verification rather than rejection.

False positives should still be avoided, but do not reject solely because
part of the hand or forearm lies outside the image boundary.
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


_RSL_HAND_BRAKE_CRITERIA = """
Candidate violation to verify: RSL HAND BRAKE (Deadman's / Vigilance Brake)
HELD DURING SIGNAL ACKNOWLEDGEMENT.

An upstream pose-geometry check has already flagged this frame as: one
hand/arm raised or extended for signal acknowledgement, AND the person's
OTHER (opposite) hand positioned near the driving console in a location
consistent with the fixed RSL Hand Brake lever. That geometry check is
approximate -- it cannot see the actual physical lever, only where the
hand sits relative to the body. Your job is to look at the real console
in this image and decide whether the opposite hand is genuinely resting
on / gripping a fixed lever or handle there.

Verify (verified = true) ONLY if ALL of the following are clearly true:
- One hand/arm is visibly raised or extended for signaling.
- The OPPOSITE hand is clearly touching a lever, handle, or knob-like
  fixed control mounted on the console directly in front of the person
  -- not merely hovering near the console's general area.
- The hand's grip/contact looks like genuine, settled holding (fingers
  wrapped around or firmly resting on the control), not a hand caught
  mid-motion, reaching, pointing, or passing through that space toward
  something else.
- The lever/control being touched is plausibly the brake/vigilance
  control given its position on the console (not clearly some other
  switch, gauge, or unrelated object).

Reject (verified = false) if ANY of the following apply:
- The opposite hand is not visible, is out of frame, or is occluded.
- The opposite hand is resting on the person's own body, clothing, lap,
  the seat, or any surface that is not a fixed console control.
- The hand appears to be moving, reaching, adjusting something, or
  merely passing through the console area rather than settled in a
  holding grip.
- The identity of the object under/near the hand is uncertain -- you
  cannot confidently tell it is a lever/brake handle at all.
- The raised/signaling arm itself is not clearly visible or convincing
  in this same image.
- Image quality, blur, glare, or occlusion make the grip impossible to
  confirm with confidence.
- You are not fully certain.

This is a compliance/procedure confirmation, not a hazard alert, but the
same "when in doubt, reject" rule still applies -- do not verify a brake
grip unless it is clearly and unambiguously visible in this image.
""".strip()


_CRITERIA_BY_VIOLATION: dict[str, str] = {
    Violation.MOBILE_PHONE: _MOBILE_PHONE_CRITERIA,
    Violation.DROWSINESS: _DROWSINESS_CRITERIA,
    Violation.HAND_RAISING: _HAND_RAISING_CRITERIA,
    Violation.SEAT_ABSENCE: _SEAT_ABSENCE_CRITERIA,
    Violation.RSL_HAND_BRAKE: _RSL_HAND_BRAKE_CRITERIA,
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
