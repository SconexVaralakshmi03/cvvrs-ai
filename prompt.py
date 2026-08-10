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
    RSL_HAND_BRAKE = "RSL Hand Brake Held"  # Multi-frame signal + brake check.


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
performed by railway crew members: a deliberate arm-raise/point gesture
directed OUT toward the front windshield / an external signal, NOT any
hand movement toward the driving console itself.

Verify (verified = true) ONLY if ALL of the following are clearly true:

- At least one arm is raised or intentionally extended toward the front
  windshield / outside the cabin, clearly aimed away from the console.
- The hand is open or pointing (e.g. an extended index finger, an open
  palm) -- NOT gripping, curled around, or in contact with any lever,
  joystick, throttle handle, switch, knob, or other console control.
- The raised or extended arm clearly belongs to the evaluated person.
- The gesture is visually distinguishable from normal resting or
  operating posture.

A valid signal acknowledgement DOES NOT require:

- the hand to be above the head,
- the elbow to be fully extended,
- the entire hand or fingertips to be visible,
- both hands to be raised.

It is acceptable if:

- only one hand is raised,
- the OTHER hand (not the one being evaluated) remains on the driving
  controls,
- part of the raised hand extends outside the camera frame,
- the fingertips are cropped but the arm direction is clearly visible.

Reject (verified = false) if ANY of the following apply:

- There is no raised or forward-extended arm.
- The arm is resting, or is reaching toward / gripping / resting on any
  part of the driving console -- including the throttle handle, brake
  lever, joystick, switches, or gauges -- even if the arm looks
  moderately extended. Operating or holding a control is NOT a signal
  acknowledgement, regardless of arm angle or extension.
- The person is leaning forward to operate equipment rather than
  gesturing outward/upward away from the console.
- The movement is clearly unrelated to signal acknowledgement
  (scratching, adjusting clothing, adjusting hair, stretching, reaching
  for an object inside the cabin, etc.).
- The arm direction cannot be determined because of severe blur or
  occlusion.
- There is insufficient visual evidence to distinguish the gesture from
  routine console operation.
- You are not fully certain this is a genuine signal-acknowledgement
  gesture rather than routine cabin activity.

When in doubt, reject -- false positives here (routine console/throttle
operation misread as a signal gesture) are as unacceptable as missed
genuine gestures. Do not reject solely because part of the hand or
forearm lies outside the image boundary, but never let arm extension
alone substitute for a hand that is visibly open/pointing and clearly
NOT in contact with a control.
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


# --------------------------------------------------------------------------- #
# RSL Hand Brake -- single-frame verification (full frame + zoomed crop)
# --------------------------------------------------------------------------- #
#
# Verified from the SAME single already-confirmed hand-raise/signal frame N
# only -- no neighbouring frame in TIME is read or sent. To help the model
# see a small/partially-occluded hand near the console, TWO images of that
# SAME frame are sent together: the full frame, and a zoomed-in crop of the
# console/lever area cut from that identical frame (see
# detector/rsl_hand_brake_verifier.py::extract_signal_frame /
# make_console_zoom_crop). This is spatial cropping of one frame, not a
# temporal window.

_RSL_HAND_BRAKE_CRITERIA = """
Candidate violation to verify: RSL HAND BRAKE HELD (Deadman's / Vigilance
Brake), together with the SIGNAL ACKNOWLEDGEMENT that triggered it.

An upstream system has already detected a raised/extended-arm
signal-acknowledgement gesture in THIS image. You are shown TWO pictures
of the exact SAME moment/frame -- they are not different points in time:

  Image 1: the full cabin frame.
  Image 2: a zoomed-in crop of the driving console / lever area cut from
           that identical frame, provided ONLY to help you see a small or
           partially-occluded hand near the console more clearly. Use it
           together with Image 1, not as separate evidence of a different
           moment.

Verify (verified = true) ONLY if BOTH of the following are clearly true:

1. SIGNAL ACKNOWLEDGEMENT -- at least one crew member's arm is raised or
   intentionally extended toward the front windshield, clearly
   distinguishable from a normal resting posture.
2. HAND ON THE RSL HAND BRAKE -- the OTHER crew member -- the one who is
   NOT performing the signal acknowledgement, the Assistant Loco Pilot --
   has one hand, typically the right hand, resting on or gripping the RSL
   Hand Brake / Vigilance Brake lever on the driving console, at console
   height, in front of their body.

A valid signal acknowledgement DOES NOT require the hand to be above the
head, the elbow fully extended, or the fingertips visible; part of the
raised arm may extend outside the camera frame.

For the hand-on-brake check, a hand does NOT need to be fully unobstructed
to count -- if the wrist/forearm position and angle are clearly consistent
with reaching to and resting on the lever, partial occlusion by a cloth,
cable, or another object on the console is acceptable evidence, especially
when Image 2 makes the wrist/forearm position clearer than Image 1 alone.

Reject (verified = false) if ANY of the following apply:
- There is no raised/extended arm performing a signal acknowledgement in
  this image.
- No hand, wrist, or forearm is visible on or near the RSL Hand Brake
  console lever in EITHER image.
- The image(s) are too blurry, dark, or occluded to judge either the
  signal or the brake-holding hand's position with reasonable confidence.
- You are not fully certain.

When in doubt, reject.
""".strip()


_RSL_JSON_SCHEMA = """
Respond with EXACTLY this JSON schema and nothing else -- no markdown
fences, no explanation outside the JSON:

{{
  "verified": <true or false>,
  "candidate_violation": "{violation}",
  "confidence": <integer 0-100>,
  "reason": "<one concise sentence citing the specific visual evidence,
             noting if Image 2 (the zoomed crop) was what made the
             hand/lever visible>"
}}

If verified is false, set confidence to 0. Do not include a "role" field
-- when this candidate is verified true, the hand-on-brake crew member is
by definition the Assistant Loco Pilot (ALP), so role is assigned by the
caller, not the model.
""".strip()


def build_rsl_hand_brake_prompt() -> str:
    """
    Build the single-frame (full frame + zoomed console crop of that SAME
    frame) verification prompt for the RSL Hand Brake candidate (see
    detector/rsl_hand_brake_verifier.py). Verifies only the already-
    confirmed hand-raise/signal frame itself -- no neighbouring frame in
    time is used, only two spatial views of the one frame.
    """
    schema = _RSL_JSON_SCHEMA.format(violation=Violation.RSL_HAND_BRAKE)
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
    schema = _JSON_SCHEMA.format(violation=candidate_violation)

    return "\n\n".join([_BASE_RULES, criteria, role_rules, schema])