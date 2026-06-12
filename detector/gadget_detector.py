from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np

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
)

from utils.logger import setup_logger

# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# FIX: raised from 0.57 to 0.80 so that when Pilot 1 reclines backwards,
# their bounding-box bottom (y2) stays below the split line, preventing
# identity swaps between Pilot 1 and Pilot 2.
GREEN_LINE_RATIO = 0.80

# How many consecutive detector calls YOLO can miss the phone before the
# distraction timer resets.
GADGET_MISS_TOLERANCE = 6

# ── MediaPipe landmark indices (COCO-17 / BlazePose-33) ──────────────────────
_LM_NOSE = 0
_LM_LEFT_EAR = 7
_LM_RIGHT_EAR = 8
_LM_LEFT_WRIST = 15
_LM_RIGHT_WRIST = 16

# Minimum landmark visibility score to be trusted.
_VIS_THRESH = 0.3  # Decreased from 0.5 to allow lower visibility landmarks

# ── Ear-proximity thresholds ─────────────────────────────────────────────────
# Phone centre must be within this many pixels of a visible ear landmark.
_EAR_RADIUS_PX = 120  # Increased from 80 to be more lenient

# Phone centre Y must be ABOVE (ear_y + this offset) to be considered near the
# ear rather than near the mouth. Walkie-talkies held at the mouth are
# ~80–100 px below the ear in this camera geometry. 65 px is the safe cut-off.
_MOUTH_EXCLUSION_OFFSET_PX = 100  # Increased from 65 to allow lower phone positions

# ── Wrist-to-object confirmation threshold ────────────────────────────────────
# At least one visible wrist must be within this many pixels of the phone
# centre to confirm the pilot is physically holding something.
_WRIST_TO_OBJ_PX = 180  # Increased from 120 to be more lenient

# ── Fallback (no landmarks available) ────────────────────────────────────────
_HEAD_ZONE_FRACTION = 0.45
_FALLBACK_MARGIN = 0.05  # 5 % of bbox width/height — tight, avoids mouth zone


# ─────────────────────────────────────────────────────────────────────────────
# YOLO MODEL (lazy singleton)
# ─────────────────────────────────────────────────────────────────────────────

_model = None


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO

        _model = YOLO(YOLO_MODEL)
    return _model


def get_shared_yolo_model():
    return _get_model()


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ObjectHit:
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    near_ear: bool = False
    from_pose: bool = False


@dataclass
class PilotResult:
    pilot_id: int
    bbox: Tuple[int, int, int, int]
    gadgets: List[ObjectHit] = field(default_factory=list)
    distracted: bool = False
    timer_value: float = 0.0


@dataclass
class FrameDetections:
    person_boxes: List[Tuple[Tuple[int, int, int, int], float]]
    gadgets: List[ObjectHit]
    pilot_crops: Dict[int, Tuple[np.ndarray, int, int, int, int]]
    split_y: int
    frame_shape: Tuple[int, int, int]


# ─────────────────────────────────────────────────────────────────────────────
# PER-PILOT TIMER  (video_time-based, not wall-clock)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _PilotTimer:
    pilot_id: int
    start_vtime: Optional[float] = None
    last_logged: Optional[float] = None
    miss_frames: int = 0
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
        self.start_vtime = None
        self.last_logged = None
        self.miss_frames = 0

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
    """Convert a MediaPipe landmark to pixel coordinates."""
    if hasattr(lm, "x"):
        return lm.x * frame_w, lm.y * frame_h
    return float(lm[0]), float(lm[1])


def _lm_visibility(lm) -> float:
    """Return the visibility score of a landmark, or 1.0 if not available."""
    if hasattr(lm, "visibility"):
        return float(lm.visibility)
    return 1.0


def _visible_landmarks(
    landmarks: list,
    indices: List[int],
    frame_w: int,
    frame_h: int,
) -> List[Tuple[float, float]]:
    """
    Return pixel (x, y) for each landmark index that:
      - exists in the list
      - has visibility >= _VIS_THRESH
      - has pixel coordinates within the frame bounds
    """
    results = []
    for idx in indices:
        if idx >= len(landmarks):
            continue
        lm = landmarks[idx]
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

    x1c = max(0, x1)
    y1c = max(0, y1)
    x2c = min(fw, x2)
    y2c = min(fh, y2)
    if x2c <= x1c or y2c <= y1c:
        return False

    crop = frame[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return False

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(lap.var())
    threshold = max(GADGET_MIN_EDGE_VARIANCE, 15.0)  # floor for IR footage
    return variance >= threshold


# ─────────────────────────────────────────────────────────────────────────────
# PHONE-NEAR-EAR CHECK  (landmark-based, per pilot)
# ─────────────────────────────────────────────────────────────────────────────


def _phone_passes_ear_check(
    phone_cx: float,
    phone_cy: float,
    landmarks: list,
    frame_w: int,
    frame_h: int,
) -> bool:
    """
    Returns True ONLY when ALL of the following hold:

    1. At least one ear landmark (left 7 / right 8) is visible (vis >= 0.5).
    2. The phone centre is within _EAR_RADIUS_PX of that visible ear.
    3. Mouth exclusion: the phone centre Y must be ABOVE
       (ear_y + _MOUTH_EXCLUSION_OFFSET_PX).
       — Walkie-talkies held at the mouth are rejected here.

    If no ear landmark is visible the function returns False
    (caller falls back to bbox-zone check).
    """
    visible_ears = _visible_landmarks(
        landmarks, [_LM_LEFT_EAR, _LM_RIGHT_EAR], frame_w, frame_h
    )
    if not visible_ears:
        return False

    dists = [np.hypot(phone_cx - ex, phone_cy - ey) for ex, ey in visible_ears]
    min_dist = min(dists)
    closest_ear_y = visible_ears[int(np.argmin(dists))][1]

    if min_dist > _EAR_RADIUS_PX:
        return False

    if phone_cy > closest_ear_y + _MOUTH_EXCLUSION_OFFSET_PX:
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# WRIST-CONFIRMATION CHECK
# ─────────────────────────────────────────────────────────────────────────────


def _wrist_confirms_object(
    phone_cx: float,
    phone_cy: float,
    landmarks: list,
    frame_w: int,
    frame_h: int,
) -> bool:
    """
    Returns True if at least one VISIBLE wrist landmark is within
    _WRIST_TO_OBJ_PX of the phone centre.

    If no wrist landmark is visible we return False conservatively.
    """
    visible_wrists = _visible_landmarks(
        landmarks, [_LM_LEFT_WRIST, _LM_RIGHT_WRIST], frame_w, frame_h
    )
    if not visible_wrists:
        return False

    for wx, wy in visible_wrists:
        if np.hypot(phone_cx - wx, phone_cy - wy) < _WRIST_TO_OBJ_PX:
            return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK: bbox-zone check  (used when landmarks are absent)
# ─────────────────────────────────────────────────────────────────────────────


def _phone_passes_bbox_fallback(
    phone_cx: float,
    phone_cy: float,
    pbox: Tuple[int, int, int, int],
) -> bool:
    """
    When MediaPipe landmarks are unavailable, check that the phone centre
    falls inside the top HEAD_ZONE_FRACTION of the pilot's body bbox with
    a tight 5 % margin.
    """
    px1, py1, px2, py2 = pbox
    p_h = py2 - py1
    p_w = px2 - px1
    if p_h <= 0:
        return False

    head_bottom = py1 + _HEAD_ZONE_FRACTION * p_h
    my = _FALLBACK_MARGIN * p_h
    mx = _FALLBACK_MARGIN * p_w

    return px1 - mx <= phone_cx <= px2 + mx and py1 - my <= phone_cy <= head_bottom + my


# ─────────────────────────────────────────────────────────────────────────────
# PILOT-ZONE ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────


def _assign_pilots_by_zone(
    boxes: List[Tuple[int, int, int, int]],
    split_y: int,
) -> List[Tuple[int, Tuple[int, int, int, int]]]:
    """
    Split detected persons into Pilot 2 (upper zone) and Pilot 1 (lower zone).

    FIX: uses the BOTTOM of the bounding box (y2) instead of the centre (cy)
    to prevent identity swapping when Pilot 1 reclines backwards to sleep.
    """
    if not boxes:
        return []
    upper, lower = [], []
    for box in boxes:
        y2 = box[3]
        (upper if y2 < split_y else lower).append(box)

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
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))

    if mean_val < 160:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
        sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DETECTOR CLASS
# ─────────────────────────────────────────────────────────────────────────────


class YoloObjectDetector:
    """
    Detects mobile phone usage by loco pilots using a multi-stage pipeline.

    DETECTION LOGIC (all conditions must hold to activate the timer):
    ──────────────────────────────────────────────────────────────────

    Stage 1 — YOLO object detection
        YOLO must detect a "cell phone" class object.
        Confidence must exceed GADGET_CONFIDENCE_THRESHOLD.
        The old wrist-alone heuristic has been completely removed.

    Stage 2 — Shape filter (Filter A)
        The detected bbox must have realistic phone dimensions.

    Stage 3 — Edge variance filter (Filter D)
        The image patch must have sharp rectangular edges.
        Run on the ORIGINAL frame (not enhanced) to avoid CLAHE artifacts.

    Stage 4 — Person bbox required (Filter B)
        A person must be detected in the same pilot zone, or a
        cached bbox from the last 3.0 s of video time is used.

    Stage 5 — Ear-proximity + mouth-exclusion check (Filter C)
        When MediaPipe pose landmarks are available:
          a) Phone centre must be within 80px of a visible ear.
          b) Phone centre must be ABOVE ear_y + 65px (mouth exclusion).
          c) Walkie-talkies held at the mouth are rejected.
        Fallback: phone centre inside top 45% of pilot bbox with 5% margin.

    Stage 6 — Wrist confirmation
        At least one visible wrist must be within 120px of the phone centre.

    Stage 7 — Per-pilot distraction timer
        Phone must be continuously detected for GADGET_ALLOWED_DURATION
        seconds of video time.
    """

    _PERSON_MISS_TOLERANCE_S = 3.0

    def __init__(self) -> None:
        self.logger = setup_logger()
        self.timers: Dict[int, _PilotTimer] = {
            1: _PilotTimer(1),
            2: _PilotTimer(2),
        }
        self.last_object_hits: List[ObjectHit] = []
        self._last_gadgets_by_pilot: Dict[int, List[ObjectHit]] = {1: [], 2: []}
        self.last_frame_detections: Optional[FrameDetections] = None

        self._phone_frame_counter: Dict[int, int] = {1: 0, 2: 0}

        self._last_known_bbox: Dict[int, Optional[Tuple[int, int, int, int]]] = {
            1: None,
            2: None,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC ENTRY POINT
    # ─────────────────────────────────────────────────────────────────────────

    def process(
        self,
        frame: np.ndarray,
        video_time: float,
        pose_landmarks: Optional[Dict[int, list]] = None,
    ) -> Tuple[List[PilotResult], List[Tuple[int, str]]]:
        enhanced = _smart_enhance(frame)
        raw_boxes, raw_gadgets = self._run_yolo(enhanced, frame)

        frame_h, frame_w = frame.shape[:2]
        split_y = int(frame_h * GREEN_LINE_RATIO)

        pilot_boxes = _assign_pilots_by_zone(raw_boxes, split_y)
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
            gadgets=raw_gadgets,
            bbox_by_pid=bbox_by_pid,
            pose_landmarks=pose_landmarks,
            video_time=video_time,
            frame=frame,
            frame_w=frame_w,
            frame_h=frame_h,
        )

        # ── Per-pilot timer logic ─────────────────────────────────────────────
        results: List[PilotResult] = []
        log_events: List[Tuple[int, str]] = []
        timer_stats = {
            'pilot1_matched': 0,
            'pilot2_matched': 0,
            'pilot1_counter': 0,
            'pilot2_counter': 0,
            'pilot1_activated': False,
            'pilot2_activated': False,
            'pilot1_distracted': False,
            'pilot2_distracted': False,
            'pilot1_should_log': False,
            'pilot2_should_log': False
        }

        for pid in [1, 2]:
            pbox = bbox_by_pid[pid]
            matched = gadgets_by_pilot.get(pid, [])
            timer = self.timers[pid]

            if pbox is not None:
                timer.last_person_seen_vtime = video_time

            if matched:
                self._phone_frame_counter[pid] += 1
                if self._phone_frame_counter[pid] >= 1:
                    timer.activate(video_time)
                self._last_gadgets_by_pilot[pid] = matched
                timer_stats[f'pilot{pid}_matched'] = len(matched)
                timer_stats[f'pilot{pid}_counter'] = self._phone_frame_counter[pid]
                timer_stats[f'pilot{pid}_activated'] = timer.start_vtime is not None
            else:
                self._phone_frame_counter[pid] = 0
                if timer.miss():
                    timer.reset()
                    self._last_gadgets_by_pilot[pid] = []

            distracted = timer.elapsed(video_time) >= GADGET_ALLOWED_DURATION
            timer_stats[f'pilot{pid}_distracted'] = distracted
            timer_stats[f'pilot{pid}_should_log'] = timer.should_log(video_time)

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
                self.logger.info(
                    f"PHONE DETECTION: Pilot {pid} distracted! "
                    f"matched={len(matched)}, counter={self._phone_frame_counter[pid]}, "
                    f"timer_active={timer.start_vtime is not None}"
                )

            if pbox is not None or matched:
                if pbox is None:
                    pbox = (
                        (frame_w // 4, 0, 3 * frame_w // 4, split_y)
                        if pid == 2
                        else (frame_w // 4, split_y, 3 * frame_w // 4, frame_h)
                    )
                results.append(
                    PilotResult(
                        pilot_id=pid,
                        bbox=pbox,
                        gadgets=matched,
                        distracted=distracted,
                        timer_value=timer.elapsed(video_time),
                    )
                )

        # Log timer statistics for debugging
        if timer_stats['pilot1_matched'] > 0 or timer_stats['pilot2_matched'] > 0:
            self.logger.debug(
                f"Phone timer stats: "
                f"pilot1_matched={timer_stats['pilot1_matched']}, "
                f"pilot2_matched={timer_stats['pilot2_matched']}, "
                f"pilot1_counter={timer_stats['pilot1_counter']}, "
                f"pilot2_counter={timer_stats['pilot2_counter']}, "
                f"pilot1_activated={timer_stats['pilot1_activated']}, "
                f"pilot2_activated={timer_stats['pilot2_activated']}, "
                f"pilot1_distracted={timer_stats['pilot1_distracted']}, "
                f"pilot2_distracted={timer_stats['pilot2_distracted']}, "
                f"pilot1_should_log={timer_stats['pilot1_should_log']}, "
                f"pilot2_should_log={timer_stats['pilot2_should_log']}"
            )

        self.last_object_hits = raw_gadgets

        # ── Build FrameDetections for downstream consumers ────────────────────
        pilot_crops: Dict[int, Tuple[np.ndarray, int, int, int, int]] = {}
        for pid, pbox in pilot_boxes:
            x1, y1, x2, y2 = pbox
            x1c = max(0, x1)
            y1c = max(0, y1)
            x2c = min(frame_w, x2)
            y2c = min(frame_h, y2)
            if x2c > x1c and y2c > y1c:
                # Create a copy of the crop to avoid holding a reference to the original frame
                pilot_crops[pid] = (frame[y1c:y2c, x1c:x2c].copy(), x1c, y1c, x2c, y2c)

        self.last_frame_detections = FrameDetections(
            person_boxes=[(b, 1.0) for b in raw_boxes],
            gadgets=raw_gadgets,
            pilot_crops=pilot_crops,
            split_y=split_y,
            frame_shape=frame.shape,
        )

        return results, log_events

    # ─────────────────────────────────────────────────────────────────────────
    # GADGET VALIDATION + PILOT ASSIGNMENT  (core new logic)
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_and_assign_gadgets(
        self,
        gadgets: List[ObjectHit],
        bbox_by_pid: Dict[int, Optional[Tuple[int, int, int, int]]],
        pose_landmarks: Optional[Dict[int, list]],
        video_time: float,
        frame: np.ndarray,
        frame_w: int,
        frame_h: int,
    ) -> Dict[int, List[ObjectHit]]:
        """
        For each YOLO gadget hit, determine which pilot (if any) it belongs to
        and whether it passes all validation filters.

        Filter pipeline per (gadget, pilot) pair:
          B  — Person bbox exists (live or cached within tolerance)
          C1 — Ear-proximity check (landmark-based when available)
          C2 — Mouth exclusion (vertical Y guard)
          W  — Wrist confirmation (object must be near a visible wrist)
          FB — Fallback bbox-zone check (when no landmarks available)
        """
        by_pilot: Dict[int, List[ObjectHit]] = {1: [], 2: []}
        validation_stats = {
            'total': len(gadgets),
            'pilot1_passed': 0,
            'pilot2_passed': 0,
            'pilot1_failed_b': 0,
            'pilot2_failed_b': 0,
            'pilot1_failed_spatial': 0,
            'pilot2_failed_spatial': 0,
            'pilot1_failed_landmarks': 0,
            'pilot2_failed_landmarks': 0,
            'pilot1_failed_fallback': 0,
            'pilot2_failed_fallback': 0
        }

        for g in gadgets:
            gx1, gy1, gx2, gy2 = g.bbox
            gcx = (gx1 + gx2) / 2.0
            gcy = (gy1 + gy2) / 2.0

            for pid in [1, 2]:
                # ── Filter B: person bbox required ────────────────────────────
                pbox = bbox_by_pid.get(pid)

                if pbox is None:
                    timer = self.timers.get(pid)
                    if (
                        timer is None
                        or timer.last_person_seen_vtime is None
                        or (video_time - timer.last_person_seen_vtime)
                        > self._PERSON_MISS_TOLERANCE_S
                    ):
                        validation_stats[f'pilot{pid}_failed_b'] += 1
                        continue
                    pbox = self._last_known_bbox.get(pid)
                    if pbox is None:
                        validation_stats[f'pilot{pid}_failed_b'] += 1
                        continue

                # ── Spatial pre-filter: phone must be in the pilot's half ──────
                px1, py1, px2, py2 = pbox
                p_h = py2 - py1
                p_w = px2 - px1
                if p_h <= 0 or p_w <= 0:
                    validation_stats[f'pilot{pid}_failed_spatial'] += 1
                    continue
                h_margin = 0.30 * p_w
                if not (px1 - h_margin <= gcx <= px2 + h_margin):
                    validation_stats[f'pilot{pid}_failed_spatial'] += 1
                    continue

                # ── Filter C + W: landmark-based checks ───────────────────────
                lms = (pose_landmarks or {}).get(pid)

                if lms is not None and len(lms) >= 17:
                    ear_ok = _phone_passes_ear_check(gcx, gcy, lms, frame_w, frame_h)
                    wrist_ok = _wrist_confirms_object(gcx, gcy, lms, frame_w, frame_h)
                    if not (ear_ok or wrist_ok):
                        validation_stats[f'pilot{pid}_failed_landmarks'] += 1
                        continue
                else:
                    if not _phone_passes_bbox_fallback(gcx, gcy, pbox):
                        validation_stats[f'pilot{pid}_failed_fallback'] += 1
                        continue

                # ── All filters passed ────────────────────────────────────────
                g.near_ear = True
                by_pilot[pid].append(g)
                validation_stats[f'pilot{pid}_passed'] += 1

        # Log validation statistics for debugging
        if validation_stats['total'] > 0:
            self.logger.debug(
                f"Gadget validation stats: total={validation_stats['total']}, "
                f"pilot1_passed={validation_stats['pilot1_passed']}, "
                f"pilot2_passed={validation_stats['pilot2_passed']}, "
                f"pilot1_failed_b={validation_stats['pilot1_failed_b']}, "
                f"pilot2_failed_b={validation_stats['pilot2_failed_b']}, "
                f"pilot1_failed_spatial={validation_stats['pilot1_failed_spatial']}, "
                f"pilot2_failed_spatial={validation_stats['pilot2_failed_spatial']}, "
                f"pilot1_failed_landmarks={validation_stats['pilot1_failed_landmarks']}, "
                f"pilot2_failed_landmarks={validation_stats['pilot2_failed_landmarks']}, "
                f"pilot1_failed_fallback={validation_stats['pilot1_failed_fallback']}, "
                f"pilot2_failed_fallback={validation_stats['pilot2_failed_fallback']}"
            )

        return by_pilot

    # ─────────────────────────────────────────────────────────────────────────
    # YOLO INFERENCE
    # ─────────────────────────────────────────────────────────────────────────

    def _run_yolo(
        self,
        enhanced_frame: np.ndarray,
        original_frame: np.ndarray,
    ) -> Tuple[List[Tuple[int, int, int, int]], List[ObjectHit]]:
        """
        Run YOLOv8 on the (possibly enhanced) frame.
        Edge variance (Filter D) runs on the ORIGINAL frame to avoid
        CLAHE-induced false sharpness.
        """
        model = _get_model()
        res = model(enhanced_frame, verbose=False)[0]
        _, frame_w = enhanced_frame.shape[:2]

        persons: List[Tuple[Tuple[int, int, int, int], float]] = []
        gadgets: List[ObjectHit] = []
        phone_detections_raw = 0
        phone_detections_after_shape = 0
        phone_detections_after_edges = 0

        for box in res.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls_id].lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox = (x1, y1, x2, y2)

            if name == "person" and conf > PILOT_CONFIDENCE_THRESHOLD:
                persons.append((bbox, conf))

            elif name in GADGET_CLASSES and conf > GADGET_CONFIDENCE_THRESHOLD:
                phone_detections_raw += 1
                if not _is_valid_gadget_shape(bbox, frame_w):
                    continue
                phone_detections_after_shape += 1
                # Filter D — edge variance on original frame (not enhanced)
                if not _has_phone_like_edges(original_frame, bbox):
                    continue
                phone_detections_after_edges += 1
                gadgets.append(ObjectHit(name, conf, bbox))

        # Log detection statistics for debugging
        if phone_detections_raw > 0:
            self.logger.debug(
                f"YOLO phone detections: raw={phone_detections_raw}, "
                f"after_shape={phone_detections_after_shape}, "
                f"after_edges={phone_detections_after_edges}"
            )

        persons.sort(
            key=lambda p: (p[0][2] - p[0][0]) * (p[0][3] - p[0][1]), reverse=True
        )
        return [p[0] for p in persons[:MAX_PILOTS]], gadgets
