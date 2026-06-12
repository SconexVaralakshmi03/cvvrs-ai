# config/settings.py
# ── Active configuration ──────────────────────────────────────────────────────

YOLO_MODEL                 = "yolov8m.pt"
GADGET_CLASSES             = ["cell phone"]
GADGET_CONFIDENCE_THRESHOLD = 0.25  # Decreased from 0.35 to detect more phones
PILOT_CONFIDENCE_THRESHOLD  = 0.55
MAX_PILOTS                 = 2

# How long (video seconds) a phone must be continuously detected before an
# alert fires.  1.0 s allows phones that may briefly be occluded.
GADGET_ALLOWED_DURATION    = 1.0

LOG_PATH                   = "logs/distraction_log.txt"
WINDOW_NAME                = "Loco Pilot Monitoring"
DISPLAY_SCALE              = 1.0

# ── Shape filter (Filter A) ───────────────────────────────────────────────────
GADGET_MIN_AREA            = 500
GADGET_MIN_ASPECT          = 0.30   # portrait phones: width < height
GADGET_MAX_ASPECT          = 0.98   # rejects walkie-talkie but allows slanted phones
GADGET_MAX_WIDTH_FRACTION  = 0.30
GADGET_MIN_WIDTH_PX        = 22
GADGET_MIN_HEIGHT_PX       = 30

# ── Edge variance filter (Filter D) ──────────────────────────────────────────
GADGET_MIN_EDGE_VARIANCE   = 25.0   # further relaxed for real-world motion/blur

# ── Ear proximity margin (fallback when no landmarks) ────────────────────────
GADGET_EAR_PROXIMITY_MARGIN = 0.05

# ── Seat absence ──────────────────────────────────────────────────────────────
ABSENCE_ALLOWED_DURATION   = 4.0
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