from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# DEBUG LOGGING — phone validation path tracing
# ─────────────────────────────────────────────────────────────────────────────
# Set this logger's level to DEBUG (e.g. logging.getLogger("gadget_detector")
# .setLevel(logging.DEBUG)) to see, for every YOLO "cell phone" detection,
# which validation path it took (landmark vs fallback) and exactly which
# check accepted or rejected it. This is the quickest way to find out
# whether false positives are coming from the loose bbox-fallback path
# or from the landmark-based ear/wrist checks.
logger = logging.getLogger("gadget_detector")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.WARNING)  # default: quiet. Flip to DEBUG to trace.
from config.settings import (
    YOLO_MODEL,
    GADGET_CLASSES,
    GADGET_CONFIDENCE_THRESHOLD,
    PILOT_CONFIDENCE_THRESHOLD,
    MAX_PILOTS,
    GADGET_ALLOWED_DURATION,
    RELOG_INTERVAL,
    GADGET_MIN_AREA,
    GADGET_MIN_ASPECT,
    GADGET_MAX_ASPECT,
    GADGET_MAX_WIDTH_FRACTION,
    GADGET_MIN_WIDTH_PX,
    GADGET_MIN_HEIGHT_PX,
    GADGET_MIN_EDGE_VARIANCE,
    FRONT_VIEW_TORSO_ZONE_FRACTION,
    GADGET_CONFIDENCE_BACK_VIEW,
)
from detector.enums import CameraView

# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

GREEN_LINE_RATIO = 0.57

# How many consecutive detector calls YOLO can miss the phone before the
# distraction timer resets.
GADGET_MISS_TOLERANCE = 3

# ── MediaPipe landmark indices (COCO-17 / BlazePose-33) ──────────────────────
_LM_NOSE        = 0
_LM_LEFT_EAR    = 7
_LM_RIGHT_EAR   = 8
_LM_LEFT_WRIST  = 15
_LM_RIGHT_WRIST = 16

# Minimum landmark visibility score to be trusted.
# MediaPipe assigns ~0.0–0.3 to occluded/estimated landmarks and >0.5 to
# landmarks it actually sees in the image.
_VIS_THRESH = 0.5

# ── Ear-proximity thresholds ─────────────────────────────────────────────────
# Phone centre must be within this many pixels of a visible ear landmark.
# At 848×480 a phone held to the ear sits ~40–70 px from the ear landmark.
# 80 px gives safe headroom without reaching the mouth region (~120 px below).
_EAR_RADIUS_PX = 80

# Phone centre Y must be ABOVE (ear_y + this offset) to be considered near the
# ear rather than near the mouth. The mouth is roughly 80–100 px below the ear
# on a seated pilot in this camera geometry. 65 px is the safe cut-off.
_MOUTH_EXCLUSION_OFFSET_PX = 65

# ── Wrist-to-object confirmation threshold ────────────────────────────────────
# At least one visible wrist must be within this many pixels of the phone
# centre to confirm the pilot is physically holding something.
# Wrist landmark → phone centre when gripping ≈ 60–100 px; 120 px is safe.
_WRIST_TO_OBJ_PX = 120

# ── Fallback (no landmarks available) ────────────────────────────────────────
# When MediaPipe landmarks are absent for a frame we fall back to a simple
# bbox-zone check: phone centre must be inside the top HEAD_ZONE_FRACTION of
# the pilot's body bbox with a small lateral margin.
_HEAD_ZONE_FRACTION = 0.25
_FALLBACK_MARGIN    = 0.02  # 2 % of bbox width/height — tight, avoids mouth zone


# ─────────────────────────────────────────────────────────────────────────────
# YOLO MODEL (lazy singleton)
# ─────────────────────────────────────────────────────────────────────────────

_model = None


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        import torch
        try:
            from ultralytics.nn.tasks import DetectionModel
            torch.serialization.add_safe_globals([DetectionModel])
        except Exception:
            pass
        _model = YOLO(YOLO_MODEL)
    return _model


def get_shared_yolo_model():
    return _get_model()


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GadgetHit:
    class_name: str
    confidence: float
    bbox:       Tuple[int, int, int, int]
    near_ear:   bool = False
    from_pose:  bool = False
    validation_path: str = ""   # "landmark" or "fallback" — set when accepted


@dataclass
class PilotResult:
    pilot_id:    int
    bbox:        Tuple[int, int, int, int]
    gadgets:     List[GadgetHit] = field(default_factory=list)
    distracted:  bool  = False
    timer_value: float = 0.0


@dataclass
class FrameDetections:
    person_boxes: List[Tuple[Tuple[int, int, int, int], float]]
    gadgets:      List[GadgetHit]
    pilot_crops:  Dict[int, Tuple[np.ndarray, int, int, int, int]]
    layout_type:  str  # 'SIDE_BY_SIDE' or 'TOP_BOTTOM'
    split_val:    float
    frame_shape:  Tuple[int, int, int]


# ─────────────────────────────────────────────────────────────────────────────
# PER-PILOT TIMER  (video_time-based, not wall-clock)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _PilotTimer:
    pilot_id:               int
    start_vtime:            Optional[float] = None
    last_logged:            Optional[float] = None
    miss_frames:            int             = 0
    last_person_seen_vtime: Optional[float] = None

    def activate(self, video_time: float) -> None:
        self.miss_frames = 0
        if self.start_vtime is None:
            self.start_vtime = video_time

    def miss(self) -> bool:
        """Increment miss counter. Returns True when tolerance is exceeded."""
        self.miss_frames += 1
        return self.miss_frames > GADGET_MISS_TOLERANCE

    def reset(self) -> None:
        self.start_vtime  = None
        self.last_logged  = None
        self.miss_frames  = 0

    def elapsed(self, video_time: float) -> float:
        if self.start_vtime is None:
            return 0.0
        return max(0.0, video_time - self.start_vtime)

    def should_log(self, video_time: float) -> bool:
        if self.elapsed(video_time) < GADGET_ALLOWED_DURATION:
            return False
        if self.last_logged is None:
            return True
        return (video_time - self.last_logged) >= RELOG_INTERVAL

    def mark_logged(self, video_time: float) -> None:
        self.last_logged = video_time


# ─────────────────────────────────────────────────────────────────────────────
# LANDMARK HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _lm_pixel(lm, frame_w: int, frame_h: int) -> Tuple[float, float]:
    """Convert a MediaPipe landmark to pixel coordinates.

    Handles both normalised objects (lm.x, lm.y) and raw pixel tuples/lists.
    """
    if hasattr(lm, "x"):
        return lm.x * frame_w, lm.y * frame_h
    return float(lm[0]), float(lm[1])


def _lm_visibility(lm) -> float:
    """Return the visibility score of a landmark, or 1.0 if not available."""
    if hasattr(lm, "visibility"):
        return float(lm.visibility)
    # Plain tuple/list landmarks have no visibility — assume visible
    return 1.0


def _visible_landmarks(
    landmarks: list,
    indices:   List[int],
    frame_w:   int,
    frame_h:   int,
) -> List[Tuple[float, float]]:
    """
    Return pixel (x, y) for each landmark index that:
      - exists in the list
      - has visibility >= _VIS_THRESH
      - has pixel coordinates within the frame bounds (basic sanity check)
    """
    results = []
    for idx in indices:
        if idx >= len(landmarks):
            continue
        lm  = landmarks[idx]
        vis = _lm_visibility(lm)
        if vis < _VIS_THRESH:
            continue
        x, y = _lm_pixel(lm, frame_w, frame_h)
        if 0 <= x <= frame_w and 0 <= y <= frame_h:
            results.append((x, y))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SHAPE FILTER  (Filter A)
# ─────────────────────────────────────────────────────────────────────────────

def _is_valid_gadget_shape(bbox: Tuple[int, int, int, int], frame_w: int) -> bool:
    """
    Rejects detections that are too small, too large, wrong aspect ratio,
    or implausibly wide (dashboard panel, instrument cluster etc.).
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return False
    if (w * h) < GADGET_MIN_AREA:
        return False
    if w < GADGET_MIN_WIDTH_PX or h < GADGET_MIN_HEIGHT_PX:
        return False
    aspect = w / h
    if aspect < GADGET_MIN_ASPECT or aspect > GADGET_MAX_ASPECT:
        return False
    if w > frame_w * GADGET_MAX_WIDTH_FRACTION:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# EDGE VARIANCE FILTER  (Filter D)
# ─────────────────────────────────────────────────────────────────────────────

def _has_phone_like_edges(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
    """
    A real phone has sharp rectangular edges → high Laplacian variance.
    Soft objects (hands, clothing, dashboard plastic) have lower variance.
    Threshold is floored at 35.0 for low-light / IR footage.
    """
    x1, y1, x2, y2 = bbox
    fh, fw = frame.shape[:2]

    x1c = max(0, x1);  y1c = max(0, y1)
    x2c = min(fw, x2); y2c = min(fh, y2)
    if x2c <= x1c or y2c <= y1c:
        return False

    crop = frame[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return False

    gray      = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    lap       = cv2.Laplacian(gray, cv2.CV_64F)
    variance  = float(lap.var())
    threshold = max(GADGET_MIN_EDGE_VARIANCE, 35.0)   # floor for IR footage
    return variance >= threshold


# ─────────────────────────────────────────────────────────────────────────────
# PHONE-NEAR-EAR CHECK  (landmark-based, per pilot)
# ─────────────────────────────────────────────────────────────────────────────

def _phone_passes_ear_check(
    phone_cx:   float,
    phone_cy:   float,
    landmarks:  list,
    frame_w:    int,
    frame_h:    int,
) -> bool:
    """
    Returns True ONLY when ALL of the following hold:

    1. At least one ear landmark (left 7 / right 8) is visible (vis >= 0.5).
       — Handles the one-ear-visible situation: only the camera-facing ear
         is used; the occluded ear's estimated landmark is skipped.

    2. The phone centre is within _EAR_RADIUS_PX of that visible ear.

    3. Mouth exclusion: the phone centre Y must be ABOVE
       (ear_y + _MOUTH_EXCLUSION_OFFSET_PX).
       — Walkie-talkies held at the mouth are ~80–100 px below the ear
         in this camera geometry. 65 px is the safe rejection threshold.
       — This check uses Y of the CLOSEST visible ear so it adapts
         automatically regardless of which side is visible.

    If no ear landmark is visible the function returns False
    (caller falls back to bbox-zone check).
    """
    visible_ears = _visible_landmarks(
        landmarks, [_LM_LEFT_EAR, _LM_RIGHT_EAR], frame_w, frame_h
    )
    if not visible_ears:
        logger.debug(
            "EAR_CHECK reject: no visible ear landmark (vis>=%.1f)", _VIS_THRESH
        )
        return False   # no reliable ear → caller must use fallback

    # Distance from phone centre to each visible ear
    dists = [np.hypot(phone_cx - ex, phone_cy - ey) for ex, ey in visible_ears]
    min_dist = min(dists)
    closest_ear_y = visible_ears[int(np.argmin(dists))][1]

    # Rule 1: phone must be close to a visible ear
    if min_dist > _EAR_RADIUS_PX:
        logger.debug(
            "EAR_CHECK reject: phone %.0fpx from nearest ear (limit %dpx)",
            min_dist, _EAR_RADIUS_PX,
        )
        return False

    # Rule 2: phone must NOT be below the mouth-exclusion line
    #   (walkie-talkie held at mouth → phone_cy > closest_ear_y + offset)
    if phone_cy > closest_ear_y + _MOUTH_EXCLUSION_OFFSET_PX:
        logger.debug(
            "EAR_CHECK reject (MOUTH GATE): phone_cy=%.0f > ear_y(%.0f)+%dpx "
            "→ looks like mouth-level object (walkie-talkie)",
            phone_cy, closest_ear_y, _MOUTH_EXCLUSION_OFFSET_PX,
        )
        return False

    logger.debug(
        "EAR_CHECK pass: dist=%.0fpx (limit %d), phone_cy=%.0f vs ear_y+offset=%.0f",
        min_dist, _EAR_RADIUS_PX, phone_cy, closest_ear_y + _MOUTH_EXCLUSION_OFFSET_PX,
    )
    return True


# ─────────────────────────────────────────────────────────────────────────────
# WRIST-CONFIRMATION CHECK
# ─────────────────────────────────────────────────────────────────────────────

def _wrist_confirms_object(
    phone_cx:  float,
    phone_cy:  float,
    landmarks: list,
    frame_w:   int,
    frame_h:   int,
) -> bool:
    """
    Returns True if at least one VISIBLE wrist landmark is within
    _WRIST_TO_OBJ_PX of the phone centre.

    Purpose: eliminates false positives where YOLO detects an object
    (dashboard instrument, reflection) with no hand nearby.

    If no wrist landmark is visible (both hands occluded / landmarks absent)
    we return True conservatively — we don't want to block a real detection
    just because the wrist is hidden. The ear check is the primary gate;
    this is the secondary confirmation.
    """
    visible_wrists = _visible_landmarks(
        landmarks, [_LM_LEFT_WRIST, _LM_RIGHT_WRIST], frame_w, frame_h
    )
    if not visible_wrists:
        # No visible wrist landmarks — cannot confirm but also cannot deny.
        # Return True so we don't silently block real detections.
        logger.debug("WRIST_CHECK pass (no visible wrist landmarks — conservative pass)")
        return True

    for wx, wy in visible_wrists:
        dist = np.hypot(phone_cx - wx, phone_cy - wy)
        if dist < _WRIST_TO_OBJ_PX:
            logger.debug("WRIST_CHECK pass: wrist %.0fpx from phone (limit %d)", dist, _WRIST_TO_OBJ_PX)
            return True

    logger.debug(
        "WRIST_CHECK reject: nearest wrist %.0fpx from phone (limit %d) — no hand near object",
        min(np.hypot(phone_cx - wx, phone_cy - wy) for wx, wy in visible_wrists),
        _WRIST_TO_OBJ_PX,
    )
    return False


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK: bbox-zone check  (used when landmarks are absent)
# ─────────────────────────────────────────────────────────────────────────────

def _phone_passes_bbox_fallback(
    phone_cx: float,
    phone_cy: float,
    pbox:     Tuple[int, int, int, int],
) -> bool:
    """
    When MediaPipe landmarks are unavailable, check that the phone centre
    falls inside the top HEAD_ZONE_FRACTION of the pilot's body bbox with
    a tight 5 % margin.

    This is intentionally stricter than the old 45 %+20 % zone to reduce
    walkie-talkie false positives even in degraded mode.
    """
    px1, py1, px2, py2 = pbox
    p_h = py2 - py1
    p_w = px2 - px1
    if p_h <= 0:
        logger.debug("FALLBACK_CHECK reject: invalid pilot bbox height")
        return False

    head_bottom = py1 + _HEAD_ZONE_FRACTION * p_h
    my = _FALLBACK_MARGIN * p_h
    mx = _FALLBACK_MARGIN * p_w

    passed = (
        px1 - mx <= phone_cx <= px2 + mx
        and py1 - my <= phone_cy <= head_bottom + my
    )
    if passed:
        logger.debug(
            "FALLBACK_CHECK pass: phone centre (%.0f,%.0f) inside top %.0f%% "
            "of pilot bbox (head_bottom=%.0f, x-range=[%.0f,%.0f])",
            phone_cx, phone_cy, _HEAD_ZONE_FRACTION * 100, head_bottom, px1 - mx, px2 + mx,
        )
    else:
        logger.debug(
            "FALLBACK_CHECK reject: phone centre (%.0f,%.0f) outside top %.0f%% "
            "zone (head_bottom=%.0f, x-range=[%.0f,%.0f]) — no ear/mouth/wrist "
            "checks ran for this hit",
            phone_cx, phone_cy, _HEAD_ZONE_FRACTION * 100, head_bottom, px1 - mx, px2 + mx,
        )
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# PILOT-ZONE ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def _assign_pilots_by_zone(
    boxes:   List[Tuple[int, int, int, int]],
    split_y: int,
) -> List[Tuple[int, Tuple[int, int, int, int]]]:
    """
    Split detected persons into Pilot 2 (upper zone, pid=2) and
    Pilot 1 (lower zone, pid=1) using the horizontal split line at
    57 % of frame height.  When multiple persons fall in the same zone
    (e.g. a bystander) the largest bbox wins.
    """
    if not boxes:
        return []
    upper, lower = [], []
    for box in boxes:
        cy = (box[1] + box[3]) / 2.0
        (upper if cy < split_y else lower).append(box)

    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
    result = []
    if upper:
        result.append((2, max(upper, key=area)))
    if lower:
        result.append((1, max(lower, key=area)))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# IR ENHANCEMENT
# ─────────────────────────────────────────────────────────────────────────────

def _smart_enhance(frame: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE + unsharp-mask when the frame is dark (mean brightness < 160).
    IR / night-vision footage benefits significantly; daytime footage is left
    unchanged to avoid over-sharpening.
    """
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))

    if mean_val < 160:
        clahe     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced  = clahe.apply(gray)
        blurred   = cv2.GaussianBlur(enhanced, (0, 0), 3)
        sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DETECTOR CLASS
# ─────────────────────────────────────────────────────────────────────────────

class GadgetDetector:
    """
    Detects mobile phone usage by loco pilots using a multi-stage pipeline.

    DETECTION LOGIC (all conditions must hold to activate the timer):
    ──────────────────────────────────────────────────────────────────

    Stage 1 — YOLO object detection
        YOLO must detect a "cell phone" class object.
        Confidence must exceed GADGET_CONFIDENCE_THRESHOLD (0.50).
        No YOLO detection → no alert. The old wrist-alone heuristic
        (which fired on empty hands) has been completely removed.

    Stage 2 — Shape filter (Filter A)
        The detected bbox must have realistic phone dimensions:
        area, width, height, aspect ratio, and max width fraction.

    Stage 3 — Edge variance filter (Filter D)
        The image patch under the bbox must have sharp rectangular
        edges (high Laplacian variance). Soft objects like hands,
        clothing, or reflections are rejected here.

    Stage 4 — Person bbox required (Filter B)
        A person must be detected in the same pilot zone, or a
        cached bbox from the last 3.0 s of video time is used.
        No person → phone hit is discarded.

    Stage 5 — Ear-proximity + mouth-exclusion check (Filter C, NEW)
        When MediaPipe pose landmarks are available:

          a) Find VISIBLE ear landmarks (visibility >= 0.5).
             Only the camera-facing ear is reliable; the occluded ear
             has visibility < 0.5 and is automatically skipped.
             Handles the one-ear-visible geometry of this cab camera.

          b) Phone centre must be within _EAR_RADIUS_PX (80 px) of
             a visible ear.

          c) Mouth exclusion: phone centre Y must be ABOVE
             (ear_y + _MOUTH_EXCLUSION_OFFSET_PX = 65 px).
             Walkie-talkies held at the mouth sit ~80–100 px below
             the ear in this camera geometry and are rejected here.

        When landmarks are unavailable, fall back to: phone centre
        inside top 45 % of pilot body bbox with 5 % margin (tight).

    Stage 6 — Wrist confirmation (NEW)
        At least one visible wrist must be within _WRIST_TO_OBJ_PX
        (120 px) of the phone centre. This confirms the pilot is
        physically holding something (not a dashboard reflection).
        If no wrist landmark is visible, this check passes conservatively
        so real detections are not silently blocked.

    Stage 7 — Per-pilot distraction timer
        Phone must be continuously detected for GADGET_ALLOWED_DURATION
        seconds of video time. YOLO can miss for GADGET_MISS_TOLERANCE
        consecutive detector calls before the timer resets.

    PER-PILOT ISOLATION:
    ────────────────────
    All state (timers, bbox cache, gadget cache) is maintained separately
    for Pilot 1 and Pilot 2. A violation by one pilot never affects the
    other's timer or detection state.
    """

    # Person-miss tolerance: how long (video seconds) a pilot can be absent
    # from YOLO detections before the cached bbox is no longer trusted.
    # 18 raw frames @ 25 fps ≈ 0.72 s between YOLO calls; 3.0 s is safe.
    _PERSON_MISS_TOLERANCE_S = 3.0

    def __init__(self) -> None:
        self.timers: Dict[int, _PilotTimer] = {
            1: _PilotTimer(1),
            2: _PilotTimer(2),
        }
        self.last_gadget_hits:       List[GadgetHit] = []
        self._last_gadgets_by_pilot: Dict[int, List[GadgetHit]] = {1: [], 2: []}
        self.last_frame_detections:  Optional[FrameDetections]  = None

        # phone_frame_counter: threshold = 1 (YOLO runs every 18 raw frames,
        # so counter can never reach > 1 before next gadget call resets it).
        self._phone_frame_counter: Dict[int, int] = {1: 0, 2: 0}

        # Last known person bbox per pilot (survives YOLO skip frames).
        self._last_known_bbox: Dict[int, Optional[Tuple[int, int, int, int]]] = {
            1: None,
            2: None,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # RESOURCE CLEANUP
    # ─────────────────────────────────────────────────────────────────────────

    def close(self) -> None:
        """
        Release any resources owned by this GadgetDetector instance.

        GadgetDetector holds only pure-Python state (timer dicts, gadget
        lists). The YOLO model is a module-level lazy singleton shared
        across all instances and intentionally kept alive for the life of
        the process, so it is NOT released here.

        This method exists so main.py's finally block can call
        self.detector.close() symmetrically alongside
        self.droop_detector.close() and self.absence_detector.close()
        without raising AttributeError. Safe to call multiple times.
        """
        # Clear instance-level caches to release numpy array references
        # (pilot crop arrays stored inside last_frame_detections).
        self.last_gadget_hits      = []
        self._last_gadgets_by_pilot = {1: [], 2: []}
        self.last_frame_detections  = None
        self._last_known_bbox       = {1: None, 2: None}

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC ENTRY POINT
    # ─────────────────────────────────────────────────────────────────────────

    def process(
        self,
        frame:          np.ndarray,
        video_time:     float,
        zone_manager:   'DynamicZoneManager',
        current_view:   CameraView,
        pose_landmarks: Optional[Dict[int, list]] = None,
    ) -> Tuple[List[PilotResult], List[Tuple[int, str]]]:
        """
        Process one video frame.

        Parameters
        ----------
        frame          : BGR image (numpy array).
        video_time     : Video-timeline seconds for this frame (not wall clock).
        pose_landmarks : Optional dict mapping pilot_id → list of MediaPipe
                         landmarks for that pilot.  When provided, the
                         ear-proximity and wrist-confirmation checks use
                         real landmark coordinates; otherwise the fallback
                         bbox-zone check is used.

        Returns
        -------
        results    : List[PilotResult] — one entry per detected pilot.
        log_events : List[(pilot_id, event_str)] — violations ready to log.
        """
        enhanced = _smart_enhance(frame)
        raw_boxes, raw_gadgets = self._run_yolo(enhanced, frame, current_view)

        frame_h, frame_w = frame.shape[:2]

        # ── Assign persons to pilot zones ─────────────────────────────────────
        if zone_manager.is_calibrated and current_view != CameraView.UNKNOWN:
            bbox_by_pid_raw = zone_manager.assign_pilots(raw_boxes)
            pilot_boxes = [(pid, box) for pid, box in bbox_by_pid_raw.items()]
            layout_type = zone_manager.layout_type
            split_val = zone_manager.split_val
        else:
            split_y = int(frame_h * 0.57) # Fallback GREEN_LINE_RATIO
            pilot_boxes = _assign_pilots_by_zone(raw_boxes, split_y)
            layout_type = 'TOP_BOTTOM'
            split_val = float(split_y)

        bbox_by_pid: Dict[int, Optional[Tuple[int, int, int, int]]] = {1: None, 2: None}
        for pid, pbox in pilot_boxes:
            bbox_by_pid[pid] = pbox

        # ── Update bbox cache (survives YOLO skip frames) ─────────────────────
        for pid in [1, 2]:
            if bbox_by_pid[pid] is not None:
                self._last_known_bbox[pid] = bbox_by_pid[pid]
            elif self._last_known_bbox[pid] is not None:
                bbox_by_pid[pid] = self._last_known_bbox[pid]

        # ── Assign validated gadget hits to pilots ────────────────────────────
        gadgets_by_pilot = self._validate_and_assign_gadgets(
            gadgets        = raw_gadgets,
            bbox_by_pid    = bbox_by_pid,
            pose_landmarks = pose_landmarks,
            video_time     = video_time,
            frame          = frame,
            frame_w        = frame_w,
            frame_h        = frame_h,
            current_view   = current_view,
        )

        # ── Per-pilot timer logic ─────────────────────────────────────────────
        results:    List[PilotResult]     = []
        log_events: List[Tuple[int, str]] = []

        for pid in [1, 2]:
            pbox    = bbox_by_pid[pid]
            matched = gadgets_by_pilot.get(pid, [])
            timer   = self.timers[pid]

            if pbox is not None:
                timer.last_person_seen_vtime = video_time

            if matched:
                self._phone_frame_counter[pid] += 1
                # Threshold = 1: activate timer on first confirmed detection.
                if self._phone_frame_counter[pid] >= 1:
                    timer.activate(video_time)
                self._last_gadgets_by_pilot[pid] = matched
            else:
                self._phone_frame_counter[pid] = 0
                if timer.miss():
                    timer.reset()
                    self._last_gadgets_by_pilot[pid] = []

            distracted = timer.elapsed(video_time) >= GADGET_ALLOWED_DURATION

            if distracted and timer.should_log(video_time):
                last_known = self._last_gadgets_by_pilot[pid]
                best = (
                    max(matched, key=lambda g: g.confidence)
                    if matched
                    else (last_known[0] if last_known else None)
                )
                name = best.class_name if best else "cell phone"
                log_events.append((pid, name))
                timer.mark_logged(video_time)

            if pbox is not None or matched:
                if pbox is None:
                    # Placeholder bbox when person is temporarily missed
                    if layout_type == 'SIDE_BY_SIDE':
                        sv = int(split_val)
                        pbox = (0, frame_h//4, sv, 3*frame_h//4) if pid == 1 else (sv, frame_h//4, frame_w, 3*frame_h//4)
                    else:
                        sv = int(split_val)
                        pbox = (frame_w//4, 0, 3*frame_w//4, sv) if pid == 2 else (frame_w//4, sv, 3*frame_w//4, frame_h)
                results.append(PilotResult(
                    pilot_id    = pid,
                    bbox        = pbox,
                    gadgets     = matched,
                    distracted  = distracted,
                    timer_value = timer.elapsed(video_time),
                ))

        self.last_gadget_hits = raw_gadgets

        # ── Build FrameDetections for downstream consumers (head-droop etc.) ──
        pilot_crops: Dict[int, Tuple[np.ndarray, int, int, int, int]] = {}
        for pid, pbox in pilot_boxes:
            x1, y1, x2, y2 = pbox
            x1c = max(0, x1);  y1c = max(0, y1)
            x2c = min(frame_w, x2); y2c = min(frame_h, y2)
            if x2c > x1c and y2c > y1c:
                pilot_crops[pid] = (frame[y1c:y2c, x1c:x2c], x1c, y1c, x2c, y2c)

        self.last_frame_detections = FrameDetections(
            person_boxes = [(b, 1.0) for b in raw_boxes],
            gadgets      = raw_gadgets,
            pilot_crops  = pilot_crops,
            layout_type  = layout_type,
            split_val    = split_val,
            frame_shape  = frame.shape,
        )

        return results, log_events

    # ─────────────────────────────────────────────────────────────────────────
    # GADGET VALIDATION + PILOT ASSIGNMENT  (core new logic)
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_and_assign_gadgets(
        self,
        gadgets:        List[GadgetHit],
        bbox_by_pid:    Dict[int, Optional[Tuple[int, int, int, int]]],
        pose_landmarks: Optional[Dict[int, list]],
        video_time:     float,
        frame:          np.ndarray,
        frame_w:        int,
        frame_h:        int,
        current_view:   CameraView,
    ) -> Dict[int, List[GadgetHit]]:
        """
        For each YOLO gadget hit, determine which pilot (if any) it belongs to
        and whether it passes all validation filters.

        Runs independently for EACH pilot — a phone near Pilot 1's ear never
        affects Pilot 2's result.

        Filter pipeline per (gadget, pilot) pair:
          B  — Person bbox exists (live or cached within tolerance)
          C1 — Ear-proximity check (landmark-based when available)
          C2 — Mouth exclusion (vertical Y guard)
          W  — Wrist confirmation (object must be near a visible wrist)
          FB — Fallback bbox-zone check (when no landmarks available)
        """
        by_pilot: Dict[int, List[GadgetHit]] = {1: [], 2: []}

        for g in gadgets:
            gx1, gy1, gx2, gy2 = g.bbox
            gcx = (gx1 + gx2) / 2.0
            gcy = (gy1 + gy2) / 2.0

            for pid in [1, 2]:
                # ── Filter B: person bbox required ────────────────────────────
                pbox = bbox_by_pid.get(pid)

                if pbox is None:
                    # No current detection — check cached bbox tolerance
                    timer = self.timers.get(pid)
                    if (
                        timer is None
                        or timer.last_person_seen_vtime is None
                        or (video_time - timer.last_person_seen_vtime)
                           > self._PERSON_MISS_TOLERANCE_S
                    ):
                        logger.debug(
                            "pid=%d gadget=%s REJECTED: no person bbox (live or cached)",
                            pid, g.class_name,
                        )
                        continue   # no person, no cached bbox in time → skip
                    pbox = self._last_known_bbox.get(pid)
                    if pbox is None:
                        logger.debug(
                            "pid=%d gadget=%s REJECTED: no cached bbox available",
                            pid, g.class_name,
                        )
                        continue

                # ── Spatial pre-filter: phone must be in the pilot's half ──────
                # Quick rejection using the pilot bbox before expensive checks.
                px1, py1, px2, py2 = pbox
                p_h = py2 - py1
                p_w = px2 - px1
                if p_h <= 0 or p_w <= 0:
                    continue
                # Phone centre must be horizontally within pilot bbox ± 30 %
                h_margin = 0.30 * p_w
                if not (px1 - h_margin <= gcx <= px2 + h_margin):
                    logger.debug(
                        "pid=%d gadget=%s REJECTED: outside horizontal zone "
                        "(gcx=%.0f, allowed=[%.0f,%.0f])",
                        pid, g.class_name, gcx, px1 - h_margin, px2 + h_margin,
                    )
                    continue

                if current_view == CameraView.BACK:
                    # BACK view: YOLO confidence (already checked) is sufficient. Bypass C and W.
                    logger.debug(
                        "pid=%d gadget=%s conf=%.2f -> BACK_VIEW path (bypass C/W)",
                        pid, g.class_name, g.confidence
                    )
                    g.validation_path = "back_view_bypass"
                    g.near_ear = True
                    by_pilot[pid].append(g)
                    continue

                if current_view == CameraView.FRONT:
                    # FRONT view: Phone must be in the lower portion of the torso.
                    torso_y_threshold = py1 + p_h * FRONT_VIEW_TORSO_ZONE_FRACTION
                    if gcy < torso_y_threshold:
                        logger.debug(
                            "pid=%d gadget=%s REJECTED: above FRONT view torso threshold",
                            pid, g.class_name
                        )
                        continue

                # ── Filter C + W: landmark-based checks ───────────────────────
                lms = (pose_landmarks or {}).get(pid)

                if lms is not None and len(lms) >= 17:
                    # Primary path: use MediaPipe landmarks
                    logger.debug(
                        "pid=%d gadget=%s conf=%.2f bbox=%s -> LANDMARK path "
                        "(%d landmarks available)",
                        pid, g.class_name, g.confidence, g.bbox, len(lms),
                    )

                    # C1 + C2: ear proximity and mouth exclusion
                    if not _phone_passes_ear_check(gcx, gcy, lms, frame_w, frame_h):
                        continue

                    # W: wrist confirmation
                    if not _wrist_confirms_object(gcx, gcy, lms, frame_w, frame_h):
                        continue

                    g.validation_path = "landmark"

                else:
                    # Fallback path: no landmarks — use tight bbox zone
                    logger.debug(
                        "pid=%d gadget=%s conf=%.2f bbox=%s -> FALLBACK path "
                        "(landmarks=%s)",
                        pid, g.class_name, g.confidence, g.bbox,
                        "none" if lms is None else f"only {len(lms)}",
                    )
                    if not _phone_passes_bbox_fallback(gcx, gcy, pbox):
                        continue

                    g.validation_path = "fallback"

                # ── All filters passed ────────────────────────────────────────
                logger.debug(
                    "pid=%d gadget=%s ACCEPTED via %s path",
                    pid, g.class_name, g.validation_path,
                )
                g.near_ear = True
                by_pilot[pid].append(g)

        return by_pilot

    # ─────────────────────────────────────────────────────────────────────────
    # YOLO INFERENCE
    # ─────────────────────────────────────────────────────────────────────────

    def _run_yolo(
        self,
        enhanced_frame: np.ndarray,
        original_frame: np.ndarray,
        current_view:   CameraView,
    ) -> Tuple[List[Tuple[int, int, int, int]], List[GadgetHit]]:
        """
        Run YOLOv8 on the (possibly enhanced) frame.

        Returns
        -------
        person_boxes : List of (x1,y1,x2,y2) for detected persons,
                       sorted largest-first, capped at MAX_PILOTS.
        gadget_hits  : List of GadgetHit for objects passing shape +
                       edge-variance filters (Filters A and D).
        """
        model = _get_model()
        res   = model(enhanced_frame, verbose=False)[0]
        _, frame_w = enhanced_frame.shape[:2]

        persons: List[Tuple[Tuple[int, int, int, int], float]] = []
        gadgets: List[GadgetHit] = []

        for box in res.boxes:
            cls_id  = int(box.cls[0])
            conf    = float(box.conf[0])
            name    = model.names[cls_id].lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox    = (x1, y1, x2, y2)

            if name == "person" and conf > PILOT_CONFIDENCE_THRESHOLD:
                persons.append((bbox, conf))

            elif name in GADGET_CLASSES:
                req_conf = GADGET_CONFIDENCE_BACK_VIEW if current_view == CameraView.BACK else GADGET_CONFIDENCE_THRESHOLD
                if conf <= req_conf:
                    continue
                
                logger.debug(
                    "YOLO raw candidate: class=%s conf=%.2f bbox=%s",
                    name, conf, bbox,
                )
                # Filter A — shape
                if not _is_valid_gadget_shape(bbox, frame_w):
                    logger.debug("  -> REJECTED by shape filter (Filter A)")
                    continue
                # Filter D — edge variance (run on original, not enhanced,
                # to avoid artefact-induced false sharpness)
                if not _has_phone_like_edges(original_frame, bbox):
                    logger.debug("  -> REJECTED by edge-variance filter (Filter D)")
                    continue
                gadgets.append(GadgetHit(name, conf, bbox))

        # Sort persons by area descending; keep at most MAX_PILOTS
        persons.sort(key=lambda p: (p[0][2] - p[0][0]) * (p[0][3] - p[0][1]), reverse=True)
        return [p[0] for p in persons[:MAX_PILOTS]], gadgets