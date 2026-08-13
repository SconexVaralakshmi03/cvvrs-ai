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

Structured per-type JSON schemas (Mobile Phone / Hand Raising / Seat
Absence)
------------------------------------------------------------------------
In addition to the always-present "verified" / "role" / "confidence" /
"reason" fields, these three violation types now request a handful of
EXTRA structured boolean/enum fields specific to that violation (e.g.
"device_hardware_visible" for Mobile Phone, "hand_is_elevated_in_air" for
Hand Raising, "body_part_location" for Seat Absence). The prose criteria
above are UNCHANGED — these extra fields don't change what the model is
told to look for, only what it reports back about what it saw. They exist
so llm_verifier.py's deterministic override layer can cross-check the
model's own top-level "verified" claim against its own structured
observations (e.g. reject a "verified: true" Hand Raising candidate whose
own hand_is_elevated_in_air field says false) instead of trusting a
single free-form "verified" boolean at face value. Drowsiness and RSL
Hand Brake keep their existing schemas (no override layer is defined for
them).
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
# Structured per-type JSON schemas — Mobile Phone / Hand Raising / Seat
# Absence request a few EXTRA structured fields (on top of the standard
# verified/role/confidence/reason) so llm_verifier.py's deterministic
# override layer can cross-check the model's own "verified" claim against
# its own reported observations. The prose criteria above (unchanged) still
# govern what the model is told to look for — these schemas only change
# what it reports back.
# --------------------------------------------------------------------------- #

_MOBILE_PHONE_JSON_SCHEMA = """
Respond with EXACTLY this JSON schema and nothing else:

{{
  "device_hardware_visible": <true or false -- is physical phone body/screen clearly visible?>,
  "object_visually_separate_from_skin": <true or false>,
  "hand_gripping_object": <true or false>,
  "hand_to_head_pose": <true or false>,
  "verified": <true or false>,
  "candidate_violation": "{violation}",
  "role": "<Loco Pilot | Assistant Loco Pilot | Both | Unknown>",
  "confidence": <integer 0-100>,
  "reason": "<one concise sentence citing visual evidence>"
}}

If verified is false, set role to "Unknown" and confidence to 0.
""".strip()


_HAND_RAISING_JSON_SCHEMA = """
CRITICAL VISUAL GROUNDING RULES FOR HAND RAISING:
1. Examine the height of the hands relative to the DASHBOARD/CONSOLE LEDGE.
2. If hands are resting on switches, levers, throttle, desk, paper logbook, or window sill, they are AT DESK LEVEL. You MUST set hand_is_elevated_in_air to false and hand_height to "at_desk_level".
3. A hand is ONLY "elevated in air" if there is clear daylight/space between the hand/arm and any console/desk surface, reaching UP towards the windshield or ceiling.
4. Do NOT hallucinate an elevated arm if both crew members have their hands on or near the control desk.
5. ONLY classify as STRETCHING/YAWNING if BOTH arms are raised overhead, hands are clasped, or the person is leaning/tilted back in a relaxation posture.

Respond with EXACTLY this JSON schema and nothing else:

{{
  "persons_observed": <integer count of visible crew members>,
  "person_descriptions": [
    "<for EACH visible person, write a fresh 8-15 word description covering: (a) position, (b) exact hand location relative to console desk, (c) whether hand is in air or on desk.>"
  ],
  "signaling_person": "<loco_pilot | assistant_loco_pilot | none>",
  "signaling_person_description": "<exact string from person_descriptions for the signaling person, or '' if none>",
  "hand_is_elevated_in_air": <true or false -- false if hand is on/near desk or controls>,
  "arms_raised_count": <0, 1, or 2>,
  "hand_height": "<at_desk_level | shoulder_height | above_head | unclear>",
  "torso_leaning_back": <true or false>,
  "head_tilted_back": <true or false>,
  "hands_clasped_overhead": <true or false>,
  "arm_direction": "<toward_windshield | toward_window_or_wall | straight_up_ceiling | low_at_console | unclear>",
  "verified": <true or false>,
  "candidate_violation": "Hand Raising",
  "role": "<Loco Pilot | Assistant Loco Pilot | Both | Unknown>",
  "confidence": <integer 0-100>,
  "reason": "<one concise sentence>"
}}
""".strip()


_SEAT_ABSENCE_JSON_SCHEMA = """
Respond with EXACTLY this JSON schema and nothing else:

{{
  "human_body_part_visible": <true or false>,
  "body_part_location": "<none | driving_seat | secondary_seat | edge_of_frame | background | floor>",
  "body_part_type": "<none | head_hair | arm_hand | leg_foot | torso | partial>",
  "seat_and_cabin_empty": <true or false>,
  "verified": <true or false>,
  "candidate_violation": "Seat Absence",
  "role": "Unknown",
  "confidence": <integer 0-100>,
  "reason": "<one concise sentence>"
}}
""".strip()

# Dispatch table used by build_verification_prompt() below — only the
# three types above have an extra structured schema; every other
# verifiable type keeps using the generic _JSON_SCHEMA.
_STRUCTURED_JSON_SCHEMA_BY_VIOLATION: dict[str, str] = {
    Violation.MOBILE_PHONE: _MOBILE_PHONE_JSON_SCHEMA,
    Violation.HAND_RAISING: _HAND_RAISING_JSON_SCHEMA,
    Violation.SEAT_ABSENCE: _SEAT_ABSENCE_JSON_SCHEMA,
}


# --------------------------------------------------------------------------- #
# Per-violation verification criteria
# --------------------------------------------------------------------------- #

_MOBILE_PHONE_CRITERIA = """
Candidate violation to verify: MOBILE PHONE USAGE.

Verify (verified = true) ONLY if ALL of the following criteria are unambiguously met:

1. HARDWARE IDENTIFICATION: An actual physical mobile phone device is directly visible.
   You must clearly identify physical electronic device attributes, such as a flat
   rectangular face, glass screen reflection, camera bump, straight device edges, or
   a distinct phone case/bezel.
2. PHYSICAL GRASP: Human fingers or a hand are visibly gripping or holding that
   distinct physical electronic object.
3. VISUAL SEPARATION: The phone object must be visually distinct from human skin,
   ears, hair, clothing, background shadows, or cabin hardware.

Reject (verified = false) if ANY of the following apply:

- HAND-TO-HEAD POSES (FALSE POSITIVE HAZARD): The person's hand, fist, thumb, or
  fingers are resting on or positioned near the ear, cheek, temple, chin, or mouth,
  BUT no physical rectangular electronic device is visible between/under the fingers.
- NATURAL CABIN GESTURES: The person is rubbing their face, resting their head on
  their hand, scratching their head/ear, resting their chin, or adjusting eyeglasses/headsets.
- AMBIGUOUS OCCLUSION: The space between the hand and face is shadowed, dark, or
  partially hidden such that you must GUESS or INFER that a phone is present.
- OTHER CABIN OBJECTS: The object held is a walkie-talkie, microphone, logbook, pen,
  cup, paper, or cabin lever/switch.
- DASHBOARD DEVICES: The screen or device is fixed to the cabin console or mounted
  rather than actively held in a crew member's hand.

CRITICAL DISAMBIGUATION RULE:
A hand held near the face or ear is NOT evidence of a mobile phone. You are strictly
forbidden from assuming a phone is present behind or inside a closed hand. If the
physical hardware frame/screen of the phone itself cannot be clearly seen separate
from skin tones, set verified = false.
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
IMPORTANT:
This is a STRICT VISUAL VERIFICATION task. Judge only what is visibly
present in this single image. Do NOT infer the person's intention or
assume that an extended arm is a signal.
A valid hand-raising signal requires CLEAR visual evidence that at least
one crew member is performing a distinct raised-arm gesture away from the
normal working/resting position.
Verify (verified = true) ONLY if ALL of the following are clearly visible:

1. At least one arm is clearly RAISED or distinctly EXTENDED away from the
   person's normal resting/working position.
2. The raised hand and forearm are clearly AWAY FROM and UNSUPPORTED BY the
   table, console, dashboard, or other working surface.
3. The raised hand is NOT touching, holding, gripping, resting on, or
   operating a control, lever, switch, throttle, brake handle, joystick,
   knob, gauge, or other console equipment.
4. The arm posture is clearly distinguishable from a normal relaxed,
   resting, reaching, or equipment-operating posture.
5. The raised/extended arm clearly belongs to the crew member being
   evaluated.
   HAND SHAPE:
   The exact finger configuration is NOT important.
   Accept a hand that is:

- fully open,
- partially open,
- pointing with one or more fingers,
- in a salute-like configuration,
- partially closed,
- or otherwise visibly raised.
  Do NOT require all fingers to be open and do NOT require a specific
  finger position.
  ARM ANGLE:
  There is NO fixed required arm angle.
  The arm does NOT need to form exactly 90 degrees with the shoulder,
  upper arm, or forearm.
  Accept reasonable variation in:
- shoulder position,
- elbow bend,
- forearm direction,
- wrist angle,
- hand height,
- and overall arm orientation.
  The hand does NOT need to be above the head.
  The elbow does NOT need to be fully straight.
  Only one hand needs to be raised.
  CRITICAL TABLE / CONSOLE RULE:
  A hand or forearm resting on, supported by, or extended across the
  table/console/work surface is NOT a hand-raising signal.
  Do NOT classify an arm as raised merely because it is extended forward
  over the console or table.
  Reject (verified = false) if ANY of the following apply:
- The hand or forearm is resting on the table, console, dashboard, or
  other working surface.
- The arm is extended across or toward the console/table as part of
  normal cabin activity.
- The person is reaching for an object inside the cabin.
- The hand is touching, holding, gripping, or operating any control or
  equipment.
- The arm is in a normal relaxed or working position.
- The posture is consistent with stretching, reaching, resting, adjusting
  clothing, adjusting hair, or another routine cabin activity.
- The image does not clearly show that the arm is actually raised away
  from the working/resting surface.
- The raised arm cannot be confidently distinguished from ordinary cabin
  activity.
- The arm direction or hand position cannot be determined because of
  severe blur, occlusion, cropping, or poor image quality.
- You are not fully certain that the visible posture satisfies the
  criteria above.
  DO NOT infer intent:
  Do not decide that a person is signalling simply because the arm is
  extended, pointing, elevated, or directed toward the front of the cabin.
  The visual posture itself must provide sufficient evidence of a distinct
  raised-arm gesture.
  When in doubt, reject the candidate.
  False positives caused by interpreting stretching, reaching, resting,
  or console/table activity as a hand-raising signal are unacceptable.
  """.strip()


_SEAT_ABSENCE_CRITERIA = """
Candidate violation to verify: SEAT ABSENCE / LOCO LEFT UNMANNED.

This candidate checks whether the cabin is completely deserted and unmanned.

MANDATORY EDGE-TO-EDGE SCAN (Check BEFORE verdict):
You MUST scan the entire frame, paying particular attention to the BOTTOM, LEFT, RIGHT,
and TOP EDGES, as well as background areas. Look for ANY visible part of a human body
(head, hair, forehead, glasses, shoulder, arm, hand, leg, shirt collar, or silhouette).

Verify (verified = true) ONLY if ALL of the following conditions are met:

1. ABSOLUTE ZERO HUMAN PRESENCE: There is NO human person visible anywhere in the image.
2. EMPTY DRIVING SEAT & CABIN: The primary driving position, secondary seat, and cabin
   floor/perimeter are entirely devoid of human presence.

Reject (verified = false) IMMEDIATELY if ANY of the following apply:

- CROPPED / PARTIAL PERSON AT EDGE: Any portion of a person's head, forehead, hair,
  glasses, ear, shoulder, arm, or hand is visible at the edge or bottom of the frame
  (even if they are leaning down, bent over, or partially cut off by the camera boundary).
- SEATED OR STANDING CREW: A crew member is sitting, standing, walking, bending down,
  or moving anywhere inside the cabin.
- POOR LIGHTING / MOTION BLUR: You cannot determine with 100% certainty that the cabin
  is empty due to darkness, shadow, or blur.

STRICT RULE: "Unmanned" means ZERO traces of a human body anywhere in the frame. If even
a tiny fraction of a head, hair, or forehead is visible at the bottom or sides of the
image, verified MUST be false.
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
    # Fix (integration): Mobile Phone / Hand Raising / Seat Absence get the
    # richer structured schema (extra fields for llm_verifier.py's
    # deterministic override layer to cross-check); every other verifiable
    # type keeps the original generic schema — unchanged behaviour for
    # Drowsiness.
    if candidate_violation in _STRUCTURED_JSON_SCHEMA_BY_VIOLATION:
        schema = _STRUCTURED_JSON_SCHEMA_BY_VIOLATION[candidate_violation].format(
            violation=candidate_violation
        )
    else:
        schema = _JSON_SCHEMA.format(violation=candidate_violation)

    return "\n\n".join([_BASE_RULES, criteria, role_rules, schema])