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
HEAD_DROOP_SCORE_THRESHOLD = 0.60
NOSE_EAR_RISE_FRACTION     = 0.10
HEAD_BACK_SCORE_THRESHOLD  = 0.50
TORSO_RECLINE_MIN          = -10.0
TORSO_HUNCH_MAX            = 10.0
DROWSY_SCORE_THRESHOLD     = 0.5