# config/settings.py
# ── Active configuration ──────────────────────────────────────────────────────

# ── Calibration & Auto-Detection ──────────────────────────────────────────────
CALIBRATION_FRAMES = 60
CALIBRATION_FACE_FRONT_MIN = 12  # >= 12/60 frames with faces -> FRONT
CALIBRATION_FACE_BACK_MAX = 3    # <= 3/60 frames with faces -> BACK
CALIBRATION_TIMEOUT_FRAMES = 300 

# ── Gadget Specifics ─────────────────────────────────────────────────────────
FRONT_VIEW_TORSO_ZONE_FRACTION = 0.40  # Phone must be in lower 60%
GADGET_CONFIDENCE_BACK_VIEW = 0.70     # Higher strictness for YOLO-only mode

YOLO_MODEL                 = "yolov8m.pt"
GADGET_CLASSES             = ["cell phone"]
GADGET_CONFIDENCE_THRESHOLD = 0.60
PILOT_CONFIDENCE_THRESHOLD  = 0.40
MAX_PILOTS                 = 2

# How long (video seconds) a phone must be continuously detected before an
# alert fires.  2.0 s eliminates single-frame spurious hits from reflections
# or brief instrument contacts.
GADGET_ALLOWED_DURATION    = 3.0

OUTPUT_PATH                = "outputs/gadget_detection_output.mp4"
LOG_PATH                   = "logs/distraction_log.txt"
WINDOW_NAME                = "Loco Pilot Monitoring"
DISPLAY_SCALE              = 1.0

# ── Shape filter (Filter A) ───────────────────────────────────────────────────
GADGET_MIN_AREA            = 700
GADGET_MIN_ASPECT          = 0.30   # portrait phones: width < height
GADGET_MAX_ASPECT          = 0.85   
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
ABSENCE_ALLOWED_DURATION   = 15.0
ABSENCE_CALIBRATION_FRAMES = 60
ABSENCE_OVERLAP_THRESH     = 0.20

# ── Head-drop / drowsiness ────────────────────────────────────────────────────
HEAD_PITCH_DOWN            = 15.0
HEAD_PITCH_BACK            = 15.0
EAR_THRESHOLD              = 0.25
HEAD_DROP_DURATION         = 5.0
STANDING_RATIO_THRESHOLD   = 1.3
BODY_MOTION_THRESHOLD      = 15
RELOG_INTERVAL             = 60.0
STILL_MOTION_MAX           = 12.0
STILL_FRAMES_THRESHOLD     = 5
NOSE_EAR_DROP_FRACTION     = 0.08
HEAD_DROOP_SCORE_THRESHOLD = 0.50
NOSE_EAR_RISE_FRACTION     = 0.10
HEAD_BACK_SCORE_THRESHOLD  = 0.40
TORSO_RECLINE_MIN          = -10.0
TORSO_HUNCH_MAX            = 10.0
DROWSY_SCORE_THRESHOLD     = 0.5