from __future__ import annotations

import time
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
    GADGET_EAR_PROXIMITY_MARGIN,
)

_model = None

def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(YOLO_MODEL)
    return _model

def get_shared_yolo_model():
    return _get_model()

GREEN_LINE_RATIO = 0.57

# ── Miss tolerance ────────────────────────────────────────────────
# How many consecutive frames YOLO can miss the phone before the
# distraction timer resets.
# Set to 3 (was 8) — shorter window prevents brief false hits from
# lingering on screen and in logs after the phone disappears.
GADGET_MISS_TOLERANCE = 3


# ─────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────

@dataclass
class GadgetHit:
    class_name: str
    confidence: float
    bbox:       Tuple[int, int, int, int]
    near_ear:   bool = False

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
    split_y:      int
    frame_shape:  Tuple[int, int, int]


# ─────────────────────────────────────────
# PILOT TIMER  (wall clock only)
# ─────────────────────────────────────────

@dataclass
class _PilotTimer:
    pilot_id:    int
    start_time:  Optional[float] = None
    last_logged: Optional[float] = None
    miss_frames: int = 0
    last_person_seen: Optional[float] = None
    def activate(self):
        self.miss_frames = 0
        if self.start_time is None:
            self.start_time = time.monotonic()

    def miss(self) -> bool:
        self.miss_frames += 1
        return self.miss_frames > GADGET_MISS_TOLERANCE

    def reset(self):
        self.start_time  = None
        self.last_logged = None
        self.miss_frames = 0

    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.monotonic() - self.start_time

    def should_log(self) -> bool:
        if self.elapsed() < GADGET_ALLOWED_DURATION:
            return False
        if self.last_logged is None:
            return True
        return (time.monotonic() - self.last_logged) >= RELOG_INTERVAL

    def mark_logged(self):
        self.last_logged = time.monotonic()


# ─────────────────────────────────────────
# MAIN DETECTOR
# ─────────────────────────────────────────

class GadgetDetector:
    """
    HOW GADGET DETECTION WORKS — plain English
    ───────────────────────────────────────────

    Step 1 — YOLO scans the frame for persons and "cell phone" objects.

    Step 2 — Each detected person is assigned to Pilot 1 or Pilot 2
             based on whether their centre is above or below the
             horizontal split line (57% of frame height).

    Step 3 — PHONE VALIDATION (3 filters must ALL pass):

        Filter A — Shape check
            Phone bbox must have realistic size and aspect ratio.
            Rejects tiny blobs, thin arms, wide panels/papers.

        Filter B — Person bbox required
            If YOLO did not detect a person in that pilot zone,
            the phone is IGNORED completely.
            → No person detected = no phone alert (prevents false
              alerts from instrument panels, papers, empty zones).

        Filter C — Ear/head proximity check
            Phone centre must be inside the TOP 40% of the pilot's
            person bbox (= head/shoulder region), expanded by 15%
            margin for YOLO jitter tolerance.
            → Phone on console/lap = ignored (centre below head zone)
            → Phone held to ear   = valid hit

    Step 4 — Timer:
        Phone must be continuously valid for GADGET_ALLOWED_DURATION
        seconds. YOLO can miss for up to GADGET_MISS_TOLERANCE frames
        without resetting the timer (handles brief detection gaps).
        After GADGET_ALLOWED_DURATION → distracted=True → red border
        + banner appear + log entry written.

    WHY HAND NEAR EAR (NO PHONE) CAN STILL FALSE-TRIGGER:
        YOLO sometimes misclassifies a dark hand or fist held near
        the face as "cell phone". This is a YOLO model limitation.
        The confidence threshold (0.65) and shape filters reduce this
        but cannot eliminate it entirely with a generic YOLO model.
        A custom-trained model on locomotive cabin footage would
        remove this class of false positive permanently.
    """

    HEAD_ZONE_FRACTION = 0.45   # top 40% of pilot body = head region

    def __init__(self) -> None:
        self.timers: Dict[int, _PilotTimer] = {
            1: _PilotTimer(1),
            2: _PilotTimer(2),
        }
        self.last_gadget_hits:       List[GadgetHit] = []
        self._last_gadgets_by_pilot: Dict[int, List[GadgetHit]] = {1: [], 2: []}
        self.last_frame_detections:  Optional[FrameDetections] = None
        self.phone_frame_counter = {1: 0, 2: 0}
    def process(
        self,
        frame:      np.ndarray,
        video_time: float,
    ) -> Tuple[List[PilotResult], List[Tuple[int, str]]]:

        enhanced = self._smart_enhance(frame)
        raw_boxes, raw_gadgets = self._run_yolo(enhanced)

        frame_h, frame_w = frame.shape[:2]
        split_y = int(frame_h * GREEN_LINE_RATIO)

        pilot_boxes = self._assign_pilots_by_zone(raw_boxes, split_y)
        bbox_by_pid: Dict[int, Optional[Tuple[int,int,int,int]]] = {1: None, 2: None}
        for pid, pbox in pilot_boxes:
            bbox_by_pid[pid] = pbox

        # ── GADGET ASSIGNMENT ─────────────────────────────────────
        # Only phones that pass ALL 3 filters count.
        gadgets_by_pilot = self._assign_gadgets_near_ear(
            raw_gadgets, bbox_by_pid
        )

        results:    List[PilotResult]     = []
        log_events: List[Tuple[int, str]] = []

        for pid in [1, 2]:
            pbox    = bbox_by_pid[pid]
            matched = gadgets_by_pilot.get(pid, [])
            timer   = self.timers[pid]
            if pbox is not None:
                timer.last_person_seen = time.monotonic()
            if matched:
                self.phone_frame_counter[pid] += 1
                if self.phone_frame_counter[pid] >= 3:
                    timer.activate()
                self._last_gadgets_by_pilot[pid] = matched
            else:
                if timer.miss():
                    timer.reset()
                    self._last_gadgets_by_pilot[pid] = []

            distracted = timer.elapsed() >= GADGET_ALLOWED_DURATION

            display_gadgets = matched 

            if distracted and timer.should_log():
                last_known = self._last_gadgets_by_pilot[pid]
                best = (max(matched, key=lambda g: g.confidence)
                        if matched else
                        (last_known[0] if last_known else None))
                name = best.class_name if best else "cell phone"
                log_events.append((pid, name))
                timer.mark_logged()

            if pbox is not None or matched:
                if pbox is None:
                    if pid == 2:
                        pbox = (frame_w//4, 0, 3*frame_w//4, split_y)
                    else:
                        pbox = (frame_w//4, split_y, 3*frame_w//4, frame_h)

                results.append(PilotResult(
                    pilot_id    = pid,
                    bbox        = pbox,
                    gadgets     = display_gadgets,
                    distracted  = distracted,
                    timer_value = timer.elapsed(),
                ))

        self.last_gadget_hits = raw_gadgets

        pilot_crops: Dict[int, Tuple[np.ndarray, int, int, int, int]] = {}
        for pid, pbox in pilot_boxes:
            x1, y1, x2, y2 = pbox
            x1c = max(0, x1); y1c = max(0, y1)
            x2c = min(frame_w, x2); y2c = min(frame_h, y2)
            if x2c > x1c and y2c > y1c:
                pilot_crops[pid] = (frame[y1c:y2c, x1c:x2c], x1c, y1c, x2c, y2c)

        self.last_frame_detections = FrameDetections(
            person_boxes = [(b, 1.0) for b in raw_boxes],
            gadgets      = raw_gadgets,
            pilot_crops  = pilot_crops,
            split_y      = split_y,
            frame_shape  = frame.shape,
        )

        return results, log_events

    # ─────────────────────────────────────────────────────────────
    # FILTER: ear proximity  (Filter B + C combined)
    # ─────────────────────────────────────────────────────────────

    def _assign_gadgets_near_ear(
        self,
        gadgets:     List[GadgetHit],
        bbox_by_pid: Dict[int, Optional[Tuple[int,int,int,int]]],
    ) -> Dict[int, List[GadgetHit]]:
        """
        Filter B — person bbox MUST exist for this pilot.
                   No person detected → phone completely ignored.
                   (Removes false alerts from empty zones, panels,
                    papers, and instrument screens.)

        Filter C — phone centre must be in head zone (top 40% of
                   person bbox + 15% margin).
                   Phone on desk/lap → ignored.
                   Phone at ear      → valid.
        """
        by_pilot: Dict[int, List[GadgetHit]] = {1: [], 2: []}

        for g in gadgets:
            gx1, gy1, gx2, gy2 = g.bbox
            gcx = (gx1 + gx2) / 2
            gcy = (gy1 + gy2) / 2

            for pid in [1, 2]:
                pbox = bbox_by_pid.get(pid)

                # ── Filter B: person must be detected ────────────
                PERSON_MISS_TOLERANCE_TIME = 1.0
                if pbox is None:
                    timer = self.timers.get(pid)
                    if timer is None or timer.last_person_seen is None:
                        continue
                    if (time.monotonic() - timer.last_person_seen) > PERSON_MISS_TOLERANCE_TIME:
                        continue
                    # bbox missing this frame but seen recently — skip to avoid unpacking None
                    continue
                px1, py1, px2, py2 = pbox
                p_h = py2 - py1
                p_w = px2 - px1
                if p_h <= 0:
                    continue

                # ── Filter C: phone must be in head zone ─────────
                head_bottom = py1 + self.HEAD_ZONE_FRACTION * p_h
                my = GADGET_EAR_PROXIMITY_MARGIN * p_h
                mx = GADGET_EAR_PROXIMITY_MARGIN * p_w

                if (px1 - mx <= gcx <= px2 + mx and
                        py1 - my <= gcy <= head_bottom + my):
                    g.near_ear = True
                    by_pilot[pid].append(g)

        return by_pilot

    @staticmethod
    def _assign_pilots_by_zone(
        boxes:   List[Tuple[int,int,int,int]],
        split_y: int,
    ) -> List[Tuple[int, Tuple[int,int,int,int]]]:
        if not boxes:
            return []
        upper, lower = [], []
        for box in boxes:
            cy = (box[1] + box[3]) / 2
            (upper if cy < split_y else lower).append(box)
        area = lambda b: (b[2]-b[0]) * (b[3]-b[1])
        result = []
        if upper: result.append((2, max(upper, key=area)))
        if lower: result.append((1, max(lower, key=area)))
        return result

    def _smart_enhance(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if np.mean(gray) < 100:
            clahe    = cv2.createCLAHE(2.5, (8, 8))
            enhanced = clahe.apply(gray)
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        return frame

    def _run_yolo(
        self,
        frame: np.ndarray,
    ) -> Tuple[List[Tuple[int,int,int,int]], List[GadgetHit]]:
        model = _get_model()
        res   = model(frame, verbose=False)[0]
        _, frame_w = frame.shape[:2]

        persons: List[Tuple[Tuple[int,int,int,int], float]] = []
        gadgets: List[GadgetHit] = []

        for box in res.boxes:
            cls_id       = int(box.cls[0])
            conf         = float(box.conf[0])
            name         = model.names[cls_id].lower()
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            bbox         = (x1, y1, x2, y2)

            if name == "person" and conf > PILOT_CONFIDENCE_THRESHOLD:
                persons.append((bbox, conf))
            elif name in GADGET_CLASSES and conf > GADGET_CONFIDENCE_THRESHOLD:
                # Filter A: shape check (geometry)
                # Filter D: pixel content check (real phone has edges)
                if (_is_valid_gadget_shape(bbox, frame_w) and
                        _has_phone_like_edges(frame, bbox)):
                    gadgets.append(GadgetHit(name, conf, bbox))

        persons.sort(
            key=lambda p: (p[0][2]-p[0][0]) * (p[0][3]-p[0][1]),
            reverse=True,
        )
        return [p[0] for p in persons[:MAX_PILOTS]], gadgets


# ─────────────────────────────────────────
# SHAPE FILTER  (Filter A)
# ─────────────────────────────────────────

def _is_valid_gadget_shape(bbox: Tuple[int,int,int,int], frame_w: int) -> bool:
    """
    Filter A — geometric shape sanity check.

    Rejects detections impossible for a real phone:
      • Too small area
      • Width or height below minimum pixel size
      • Wrong aspect ratio (thin arms, wide panels)
      • Too wide relative to frame
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return False
    if (w * h) < GADGET_MIN_AREA:
        return False
    # Both dimensions must meet minimums independently.
    # A shadow blob may pass area check but fail width or height.
    if w < GADGET_MIN_WIDTH_PX or h < GADGET_MIN_HEIGHT_PX:
        return False
    aspect = w / h
    if aspect < GADGET_MIN_ASPECT or aspect > GADGET_MAX_ASPECT:
        return False
    if w > frame_w * GADGET_MAX_WIDTH_FRACTION:
        return False
    return True


def _has_phone_like_edges(
    frame: np.ndarray,
    bbox:  Tuple[int,int,int,int],
) -> bool:
    """
    Pixel content check — separates real phone from shadow/hand.

    HOW IT WORKS:
      A real phone is a manufactured rectangular object with sharp,
      high-contrast edges (screen border, body outline).
      The Laplacian operator highlights edges. The variance of the
      Laplacian inside the detection bbox tells us how many sharp
      edges are present.

      Real phone   → HIGH Laplacian variance (lots of clear edges)
      Dark shadow  → LOW  Laplacian variance (no edges, just dark blur)
      Hand/skin    → MEDIUM variance but usually fails shape filter first

    Threshold: GADGET_MIN_EDGE_VARIANCE (default 80.0)
    Tune down if real phones are missed. Tune up if shadows still pass.
    """
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]

    # Clamp to frame bounds
    x1c = max(0, x1); y1c = max(0, y1)
    x2c = min(w, x2); y2c = min(h, y2)

    if x2c <= x1c or y2c <= y1c:
        return False

    crop = frame[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return False

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    lap  = cv2.Laplacian(gray, cv2.CV_64F)
    variance = lap.var()

    return variance >= GADGET_MIN_EDGE_VARIANCE


def _intersection_area(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    return max(0, ix2-ix1) * max(0, iy2-iy1)

def _iou(a, b):
    inter = _intersection_area(a, b)
    if inter == 0: return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)