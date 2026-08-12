# # # # YOLO_MODEL = "yolov8m.pt" 
# # # # GADGET_CLASSES = ["cell phone"] 
# # # # GADGET_CONFIDENCE_THRESHOLD = 0.50 
# # # # PILOT_CONFIDENCE_THRESHOLD = 0.40 
# # # # MAX_PILOTS = 2 
# # # # GADGET_ALLOWED_DURATION = 0.20 
# # # # OUTPUT_PATH = "outputs/gadget_detection_output.mp4" 
# # # # LOG_PATH = "logs/distraction_log.txt" 
# # # # WINDOW_NAME = "Loco Pilot Monitoring" 
# # # # DISPLAY_SCALE = 1.0 
# # # # GADGET_MIN_AREA = 700
# # # # GADGET_MIN_ASPECT = 0.4
# # # # GADGET_MAX_ASPECT = 3.0
# # # # GADGET_MAX_WIDTH_FRACTION = 0.30
# # # # GADGET_MIN_WIDTH_PX = 22
# # # # GADGET_MIN_HEIGHT_PX = 30
# # # # GADGET_MIN_EDGE_VARIANCE = 130.0
# # # # GADGET_EAR_PROXIMITY_MARGIN = 0.20
# # # # ABSENCE_ALLOWED_DURATION = 4.0 
# # # # ABSENCE_CALIBRATION_FRAMES = 60 
# # # # ABSENCE_OVERLAP_THRESH = 0.10 
# # # # HEAD_PITCH_DOWN = 10.0 
# # # # HEAD_PITCH_BACK = 15.0 
# # # # EAR_THRESHOLD = 0.20 
# # # # HEAD_DROP_DURATION = 1.5 
# # # # STANDING_RATIO_THRESHOLD = 1.3 
# # # # BODY_MOTION_THRESHOLD = 15 
# # # # RELOG_INTERVAL = 30.0 
# # # # STILL_MOTION_MAX = 6.0 
# # # # STILL_FRAMES_THRESHOLD = 5 
# # # # NOSE_EAR_DROP_FRACTION = 0.08 
# # # # HEAD_DROOP_SCORE_THRESHOLD = 0.60 
# # # # NOSE_EAR_RISE_FRACTION = 0.10 
# # # # HEAD_BACK_SCORE_THRESHOLD = 0.50 
# # # # TORSO_RECLINE_MIN = -10.0 
# # # # TORSO_HUNCH_MAX = 10.0 
# # # # DROWSY_SCORE_THRESHOLD = 0.5



# # # YOLO_MODEL = "yolov8m.pt"
# # # GADGET_CLASSES = ["cell phone"]
# # # GADGET_CONFIDENCE_THRESHOLD = 0.50
# # # PILOT_CONFIDENCE_THRESHOLD = 0.40
# # # MAX_PILOTS = 2

# # # # FIX (Bug #6): was 0.20s — so short that a single YOLO frame triggered a
# # # # violation. 2.0s requires the phone to be continuously detected for two full
# # # # seconds of video time before an alert fires, eliminating single-frame spurious
# # # # hits from dashboard reflections or brief instrument contacts.
# # # GADGET_ALLOWED_DURATION = 2.0

# # # OUTPUT_PATH = "outputs/gadget_detection_output.mp4"
# # # LOG_PATH = "logs/distraction_log.txt"
# # # WINDOW_NAME = "Loco Pilot Monitoring"
# # # DISPLAY_SCALE = 1.0
# # # GADGET_MIN_AREA = 700
# # # GADGET_MIN_ASPECT = 0.4
# # # GADGET_MAX_ASPECT = 3.0
# # # GADGET_MAX_WIDTH_FRACTION = 0.30
# # # GADGET_MIN_WIDTH_PX = 22
# # # GADGET_MIN_HEIGHT_PX = 30
# # # GADGET_MIN_EDGE_VARIANCE = 130.0

# # # # FIX (Bug #3): was 0.20 — added 20% slack on all sides of the pilot bbox,
# # # # which allowed phone hits on the instrument panel directly in front of the
# # # # pilot to pass Filter C. Tightened to 0.05 (5%) so only detections very
# # # # close to the pilot's body boundary pass.
# # # GADGET_EAR_PROXIMITY_MARGIN = 0.05

# # # ABSENCE_ALLOWED_DURATION = 4.0
# # # ABSENCE_CALIBRATION_FRAMES = 60
# # # ABSENCE_OVERLAP_THRESH = 0.10
# # # HEAD_PITCH_DOWN = 10.0
# # # HEAD_PITCH_BACK = 15.0
# # # EAR_THRESHOLD = 0.20
# # # HEAD_DROP_DURATION = 1.5
# # # STANDING_RATIO_THRESHOLD = 1.3
# # # BODY_MOTION_THRESHOLD = 15
# # # RELOG_INTERVAL = 30.0
# # # STILL_MOTION_MAX = 6.0
# # # STILL_FRAMES_THRESHOLD = 5
# # # NOSE_EAR_DROP_FRACTION = 0.08
# # # HEAD_DROOP_SCORE_THRESHOLD = 0.60
# # # NOSE_EAR_RISE_FRACTION = 0.10
# # # HEAD_BACK_SCORE_THRESHOLD = 0.50
# # # TORSO_RECLINE_MIN = -10.0
# # # TORSO_HUNCH_MAX = 10.0
# # # DROWSY_SCORE_THRESHOLD = 0.5



# # YOLO_MODEL = "yolov8m.pt"
# # GADGET_CLASSES = ["cell phone"]
# # GADGET_CONFIDENCE_THRESHOLD = 0.50
# # PILOT_CONFIDENCE_THRESHOLD = 0.40
# # MAX_PILOTS = 2

# # # FIX (Bug #6): was 0.20s — so short that a single YOLO frame triggered a
# # # violation. 2.0s requires the phone to be continuously detected for two full
# # # seconds of video time before an alert fires.
# # GADGET_ALLOWED_DURATION = 2.0

# # OUTPUT_PATH = "outputs/gadget_detection_output.mp4"
# # LOG_PATH = "logs/distraction_log.txt"
# # WINDOW_NAME = "Loco Pilot Monitoring"
# # DISPLAY_SCALE = 1.0
# # GADGET_MIN_AREA = 700

# # # FIX (Radio handset): Tightened aspect ratio to portrait-only.
# # # A real phone held to the ear is portrait: width < height → aspect < 1.0.
# # # The walkie-talkie/radio handset in this cab is detected with aspect ~1.1–1.35
# # # (nearly square or slightly landscape). Tightening GADGET_MAX_ASPECT to 0.85
# # # rejects near-square and landscape detections, eliminating the radio handset
# # # false positives while still catching phones held in portrait orientation.
# # GADGET_MIN_ASPECT = 0.30
# # GADGET_MAX_ASPECT = 0.85

# # GADGET_MAX_WIDTH_FRACTION = 0.30
# # GADGET_MIN_WIDTH_PX = 22
# # GADGET_MIN_HEIGHT_PX = 30
# # GADGET_MIN_EDGE_VARIANCE = 90.0

# # # FIX (Bug #3): was 0.20 — added 20% slack allowing dashboard hits to pass
# # # Filter C. Tightened to 0.05 (5%).
# # GADGET_EAR_PROXIMITY_MARGIN = 0.05

# # ABSENCE_ALLOWED_DURATION = 4.0
# # ABSENCE_CALIBRATION_FRAMES = 60
# # ABSENCE_OVERLAP_THRESH = 0.10
# # HEAD_PITCH_DOWN = 10.0
# # HEAD_PITCH_BACK = 15.0
# # EAR_THRESHOLD = 0.20
# # HEAD_DROP_DURATION = 1.5
# # STANDING_RATIO_THRESHOLD = 1.3
# # BODY_MOTION_THRESHOLD = 15
# # RELOG_INTERVAL = 30.0
# # STILL_MOTION_MAX = 6.0
# # STILL_FRAMES_THRESHOLD = 5
# # NOSE_EAR_DROP_FRACTION = 0.08
# # HEAD_DROOP_SCORE_THRESHOLD = 0.60
# # NOSE_EAR_RISE_FRACTION = 0.10
# # HEAD_BACK_SCORE_THRESHOLD = 0.50
# # TORSO_RECLINE_MIN = -10.0
# # TORSO_HUNCH_MAX = 10.0
# # DROWSY_SCORE_THRESHOLD = 0.5



# YOLO_MODEL = "yolov8m.pt" 
# GADGET_CLASSES = ["cell phone"] 
# GADGET_CONFIDENCE_THRESHOLD = 0.50 
# PILOT_CONFIDENCE_THRESHOLD = 0.40 
# MAX_PILOTS = 2 
# GADGET_ALLOWED_DURATION = 0.20 
# OUTPUT_PATH = "outputs/gadget_detection_output.mp4" 
# LOG_PATH = "logs/distraction_log.txt" 
# WINDOW_NAME = "Loco Pilot Monitoring" 
# DISPLAY_SCALE = 1.0 
# GADGET_MIN_AREA = 700
# GADGET_MIN_ASPECT = 0.4
# GADGET_MAX_ASPECT = 3.0
# GADGET_MAX_WIDTH_FRACTION = 0.30
# GADGET_MIN_WIDTH_PX = 22
# GADGET_MIN_HEIGHT_PX = 30
# GADGET_MIN_EDGE_VARIANCE = 130.0
# GADGET_EAR_PROXIMITY_MARGIN = 0.20
# ABSENCE_ALLOWED_DURATION = 4.0 
# ABSENCE_CALIBRATION_FRAMES = 60 
# ABSENCE_OVERLAP_THRESH = 0.10 
# HEAD_PITCH_DOWN = 10.0 
# HEAD_PITCH_BACK = 15.0 
# EAR_THRESHOLD = 0.20 
# HEAD_DROP_DURATION = 1.5 
# STANDING_RATIO_THRESHOLD = 1.3 
# BODY_MOTION_THRESHOLD = 15 
# RELOG_INTERVAL = 30.0 
# STILL_MOTION_MAX = 6.0 
# STILL_FRAMES_THRESHOLD = 5 
# NOSE_EAR_DROP_FRACTION = 0.08 
# HEAD_DROOP_SCORE_THRESHOLD = 0.60 
# NOSE_EAR_RISE_FRACTION = 0.10 
# HEAD_BACK_SCORE_THRESHOLD = 0.50 
# TORSO_RECLINE_MIN = -10.0 
# TORSO_HUNCH_MAX = 10.0 
# DROWSY_SCORE_THRESHOLD = 0.5

# config/settings.py
# ── Active configuration ──────────────────────────────────────────────────────

YOLO_MODEL                 = "yolov8m.pt"
GADGET_CLASSES             = ["cell phone"]
GADGET_CONFIDENCE_THRESHOLD = 0.55
PILOT_CONFIDENCE_THRESHOLD  = 0.40
MAX_PILOTS                 = 2

# How long (video seconds) a phone must be continuously detected before an
# alert fires.  2.0 s eliminates single-frame spurious hits from reflections
# or brief instrument contacts.
GADGET_ALLOWED_DURATION    = 2.0

OUTPUT_PATH                = "outputs/gadget_detection_output.mp4"
LOG_PATH                   = "logs/distraction_log.txt"
WINDOW_NAME                = "Loco Pilot Monitoring"
DISPLAY_SCALE              = 1.0

# ── Shape filter (Filter A) ───────────────────────────────────────────────────
GADGET_MIN_AREA            = 700
GADGET_MIN_ASPECT          = 0.30   # portrait phones: width < height
GADGET_MAX_ASPECT          = 0.85   # rejects walkie-talkie (aspect ~1.1–1.35)
GADGET_MAX_WIDTH_FRACTION  = 0.30
GADGET_MIN_WIDTH_PX        = 22
GADGET_MIN_HEIGHT_PX       = 30

# ── Edge variance filter (Filter D) ──────────────────────────────────────────
GADGET_MIN_EDGE_VARIANCE   = 90.0   # floored at 35.0 in code for IR footage

# ── Legacy proximity margin (kept for imports; not used in new logic) ─────────
# The new detector uses landmark-based ear proximity (_EAR_RADIUS_PX = 80 px)
# and the fallback bbox-zone margin is hardcoded to 0.05 in gadget_detector.py.
GADGET_EAR_PROXIMITY_MARGIN = 0.05

# ── Seat absence ──────────────────────────────────────────────────────────────
ABSENCE_ALLOWED_DURATION   = 2.0
ABSENCE_CALIBRATION_FRAMES = 60
ABSENCE_OVERLAP_THRESH     = 0.10

# ── Head-drop / drowsiness ────────────────────────────────────────────────────
HEAD_PITCH_DOWN            = 10.0
HEAD_PITCH_BACK            = 15.0
EAR_THRESHOLD              = 0.20
HEAD_DROP_DURATION         = 1.5
STANDING_RATIO_THRESHOLD   = 1.3
BODY_MOTION_THRESHOLD      = 15
RELOG_INTERVAL             = 30.0
STILL_MOTION_MAX           = 6.0
STILL_FRAMES_THRESHOLD     = 5
NOSE_EAR_DROP_FRACTION     = 0.08
HEAD_DROOP_SCORE_THRESHOLD = 0.40
NOSE_EAR_RISE_FRACTION     = 0.10
HEAD_BACK_SCORE_THRESHOLD  = 0.50
TORSO_RECLINE_MIN          = -10.0
TORSO_HUNCH_MAX            = 10.0
DROWSY_SCORE_THRESHOLD     = 0.5

# ── Hand-raise / signaling gesture (NEW — additive, does not touch any ────────
#    existing detector's constants above) ──────────────────────────────────
# Detects the railway "call out and point / hand-raise acknowledgement"
# gesture — either a straight OVERHEAD salute or an extended forward/
# outward POINT towards the window/signal/gauge. Runs off the SAME
# per-pilot MediaPipe landmarks the gadget detector already computes each
# cycle (main.py's _get_pose_landmarks) — no extra pose inference.

# Minimum MediaPipe landmark visibility to trust a shoulder/elbow/wrist
# point at all.
HAND_RAISE_VIS_THRESHOLD            = 0.5

# Wrist must be at least this many pixels ABOVE the shoulder (smaller y)
# to count as "raised" rather than a small hand fidget near the panel.
# FIX (real-footage false positives on 001_handgestures.mp4): 20px is a
# flat pixel count with no relation to frame resolution or how far the
# pilot is from the camera. On this footage (1904x1004) it's a trivial
# ~1% of frame height — almost ANY upward hand motion clears it, so this
# gate was doing effectively nothing and all the real filtering rested on
# the ratio/angle branches below, which turned out not to be tight enough
# either (6 of 6 "hand raise" hits on a 4-min routine-panel-operation
# clip, only 1 genuine). Now combined with HAND_RAISE_MARGIN_FRACTION so
# the gate scales with the person's actual size in frame.
HAND_RAISE_MARGIN_PX                = 20

# NEW — companion proportional floor for the rule above: wrist must ALSO
# clear this fraction of the person's own torso height above the
# shoulder, not just a flat pixel count. The effective margin used is
# max(HAND_RAISE_MARGIN_PX, HAND_RAISE_MARGIN_FRACTION * torso_h) — see
# _classify_side(). 5% of torso height is small enough to still pass a
# genuine raise/point gesture comfortably, but on a high-resolution/
# close-up frame it's meaningfully larger than 20px, closing the gap a
# routine reach toward the panel was sneaking through.
HAND_RAISE_MARGIN_FRACTION          = 0.05

# Elbow angle (shoulder-elbow-wrist), degrees. A locked-out overhead
# salute sits ~150-180°; resting/bent arms sit ~105-135°.
HAND_RAISE_ELBOW_STRAIGHT_MIN_DEG   = 150.0

# "Point towards the window/signal" gesture support: elbow noticeably
# bent (~120-130°) but the wrist still reaches far from the shoulder
# relative to the person's own torso height. dist(shoulder, wrist) /
# torso_height must reach this ratio.
#
# FIX (real-footage false positives): 0.55 was measured against only the
# two reference frames and turned out to also be satisfied by ordinary
# reaches to operate this cab's panel/gauges — they sit at a similar
# shoulder-to-head reach height, so a routine switch press was scoring
# as high as ~0.55-0.65. Raised to 0.72: the two validated true-positive
# reference frames measured 0.74-0.88, comfortably clear of this floor,
# while most panel-operation reaches (shorter, closer to the body) fall
# below it.
#
# FIX #2 (001_handgestures.mp4 — 6 confirmed hits in a 4-min clip of
# routine panel operation, only 1 genuine): 0.72 left only a 0.02 margin
# below the documented true-positive floor (0.74), so reaches that
# nearly-but-not-quite matched a real gesture still got through. Raised
# to match the true-positive floor exactly — still passes both reference
# frames (0.74, 0.88) at the boundary and above, while closing that gap.
HAND_RAISE_EXTENSION_RATIO_MIN      = 0.74

# NEW — companion floor for the extension-ratio rule above: the elbow
# must ALSO be at least moderately opened out (not a tight, tucked bend)
# for the "POINT" branch to fire. Reference frames measured 122-129°,
# well clear of this floor; this mainly guards against a landmark
# artifact inflating the ratio on a genuinely tucked-in arm.
#
# FIX #2 (001_handgestures.mp4): 100° was loose enough that a pilot
# leaning IN toward the console with a moderately tucked elbow (typical
# of pressing a sequence of panel switches) could still pass. Raised to
# 115° — still well clear of the 122-129° reference range for a genuine
# point/signal gesture, but above the tucked-elbow posture of routine
# panel operation.
HAND_RAISE_EXTENDED_MIN_ANGLE_DEG   = 115.0

# Fallback: wrist at/above head (nose) height by this fraction of torso
# height, even when the elbow bend is in between the two rules above.
HAND_RAISE_WRIST_ABOVE_NOSE_FRACTION = 0.02

# FIX (real-footage false positives, confirmed on TWO independent videos):
# this branch (cond_overhead in detector/hand_raise_detector.py::
# _classify_side) previously had NO elbow-angle requirement at all — a
# wrist within ~2% of torso height of nose level was enough to mark the
# arm "raised" regardless of elbow bend. That posture also covers holding
# a phone to the ear, and — as seen on 001_handgestures.mp4 — leaning
# forward with a sharply tucked elbow to operate upper-console switches,
# where the wrist crosses above nose height purely from the reach/lean,
# not a gesture. Tightening HAND_RAISE_EXTENSION_RATIO_MIN /
# HAND_RAISE_EXTENDED_MIN_ANGLE_DEG alone did not fix this because those
# only gate the POINT branch — this OVERHEAD fallback branch was
# untouched and kept firing independently. Requiring the elbow to also be
# at least moderately open here closes that gap while a genuine overhead
# raise (elbow well past 150deg, or comfortably open even mid-raise)
# clears this floor easily.
HAND_RAISE_OVERHEAD_MIN_ANGLE_DEG   = 120.0

# How many consecutive detector CYCLES (not raw frames — this detector
# runs on the same cadence as the gadget detector, GADGET_EVERY) a side
# must be "raised" before it's confirmed. Kills single-cycle jitter.
# Raised from 2 -> 3: an extra cycle of persistence further separates a
# quick, incidental panel touch from a held, deliberate signaling gesture.
HAND_RAISE_CONFIRM_FRAMES           = 3

# Minimum true (onset-to-now) duration, in video seconds, before a
# gesture is logged — now that the onset-tracking bug is fixed (see
# detector/hand_raise_detector.py, raw_start), this is a genuine
# elapsed-time gate rather than firing instantly at 0.0. 0.5s filters out
# single-frame flicker while still catching brief-but-real gestures.
HAND_RAISE_ALLOWED_DURATION         = 0.5

# How many consecutive missed detector cycles (no pose landmarks AT ALL
# for this pilot) before an in-progress gesture episode is considered
# ended outright.
HAND_RAISE_MISS_TOLERANCE           = 1

# NEW — hysteresis for the RAW (pre-confirmation) geometric raise signal.
# Allows this many consecutive cycles where NEITHER side geometrically
# qualifies before the episode is considered truly over. Without this, a
# single dropped cycle mid-gesture (pose jitter, or the hand briefly
# dipping between two motions of one continuous reach) split what was
# really one gesture into two separate logged episodes, each with a
# misleadingly tiny reported duration.
HAND_RAISE_RAW_MISS_TOLERANCE       = 1

# NEW — zone split ratio for HandRaisePoseEngine (detector/hand_raise_detector.py).
# Matches the 0.57 top/bottom split main.py's _get_pose_landmarks() already
# uses for the gadget detector, so hand-raise pilot ids (2=top, 1=bottom)
# line up with the rest of the pipeline's pilot numbering. Kept as its own
# constant (rather than importing main.py's literal) so this detector stays
# fully self-contained.
HAND_RAISE_ZONE_SPLIT_RATIO         = 0.57

# ── RSL Hand Brake post-verification (NEW — additive) ─────────────────────────
# See detector/rsl_hand_brake_verifier.py. Runs ONLY after hand_raise has
# already confirmed a violation on the signal frame and the pipeline has
# selected its final representative frame — never on every frame, never a
# standalone detector.
#
# ONLY that single already-selected signal frame N is sent DIRECTLY to
# Qwen-VL (see llm_verifier.verify_rsl_hand_brake_frame /
# prompt.build_rsl_hand_brake_prompt) for signal-acknowledgement +
# brake-hand checking, in place of a second MediaPipe pose pass. No
# neighbouring-frame-in-TIME window is read or sent. If confirmed, role is
# always "ALP".
#
# To help the model see a small/partially-occluded hand near the console
# (root cause of missed RSL confirmations on otherwise-correct frames),
# TWO images of that SAME frame are sent together: the full frame, and a
# zoomed-in crop of the console/lever area cut from that identical frame.
# This is spatial cropping of one frame, NOT a second frame in time.
# Fractions are relative to full frame width/height (0.0-1.0), tuned to
# this fleet's fixed in-cab camera framing (console lower-left, ALP
# seat/console-edge center-right); adjust if a different camera mount is
# introduced.
RSL_BRAKE_ZOOM_X_START_FRACTION        = 0.05
RSL_BRAKE_ZOOM_X_END_FRACTION          = 0.95
RSL_BRAKE_ZOOM_Y_START_FRACTION        = 0.15
RSL_BRAKE_ZOOM_Y_END_FRACTION          = 0.95

# How much to upscale the cropped console region before sending it to
# Qwen-VL, so a small hand/lever region isn't just a handful of pixels
# once cropped out of a full cabin-wide frame.
RSL_BRAKE_ZOOM_SCALE_FACTOR            = 2.0

# LEGACY — the ±N frame verification window around frame N has been
# removed; RSL Hand Brake verification now reads/sends ONLY the single
# already-selected signal frame. This constant is no longer read anywhere
# and is kept only so old configs don't fail to import.
RSL_BRAKE_WINDOW_RADIUS                = 2

# LEGACY — used only by the old geometry-based (MediaPipe landmark) window
# check that verify_rsl_hand_brake() used before it was switched to send
# frames directly to Qwen-VL. No longer read by detector/rsl_hand_brake_verifier.py;
# kept here in case that geometry-only path is ever reinstated.
RSL_BRAKE_VIS_THRESHOLD                = 0.5
RSL_BRAKE_REGION_Y_MARGIN_FRACTION     = 0.20
RSL_BRAKE_REGION_X_FRACTION            = 0.55
RSL_BRAKE_MIN_VALID_FRAMES             = 3
RSL_BRAKE_MIN_PASS_FRAMES              = 3
RSL_BRAKE_MIN_PASS_RATIO               = 0.8
RSL_BRAKE_MAX_POSITION_SPREAD_FRACTION = 0.35

# ── LLM verification gate (NEW — additive) ────────────────────────────────────
# Every candidate violation (Mobile Phone, Drowsiness, Hand Raising,
# Seat Absence) produced by the detectors above is re-checked by the
# Qwen2.5-VL verification model (prompt.py / llm_verifier.py) inside
# analyzer.py, BEFORE its frame is uploaded to S3 or the violation is
# included in the completion payload sent to the Java backend. See
# llm_verifier.py for the full flow.

# Master switch. False = skip the LLM entirely and keep the old
# detector-only behaviour (every candidate is treated as verified).
LLM_VERIFICATION_ENABLED    = True

# If the LLM can't be reached / times out / returns unparseable JSON after
# retries: False (default) = fail CLOSED — reject the candidate, matching
# the "when in doubt, reject" philosophy of prompt.py. True = fail OPEN —
# keep the candidate unverified rather than silently drop real violations
# during an LLM outage.
LLM_VERIFICATION_FAIL_OPEN  = False

OLLAMA_MODEL                 = "qwen2.5vl:7b"
OLLAMA_HOST                  = None   # None = ollama client default (http://localhost:11434)
# Worst-case time a single stuck/slow violation frame can block the
# journey: OLLAMA_TIMEOUT_SECONDS x (OLLAMA_MAX_RETRIES + 1) attempts.
# Was 120s x 3 = 360s (6 min) PER FRAME, run serially for every candidate
# violation in the journey — on a journey with many violations this can
# silently consume most of the journey's timeout budget. Lowered to
# 45s x 2 = 90s per frame; tune these based on your Ollama server's
# actual observed p99 response time, not a guess.
OLLAMA_TIMEOUT_SECONDS        = 45
OLLAMA_MAX_RETRIES            = 1
OLLAMA_TEMPERATURE            = 0.1