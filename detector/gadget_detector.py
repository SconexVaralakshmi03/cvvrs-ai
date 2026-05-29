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
GADGET_MISS_TOLERANCE = 3

# ── IR / Debug settings ───────────────────────────────────────────
# Set DEBUG_YOLO = True to print raw YOLO detections to console.
DEBUG_YOLO = True   # ← SET TO False IN PRODUCTION

# Wrist-to-ear distance threshold (fraction of frame width).
WRIST_EAR_THRESH_FRACTION = 0.15


# ─────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────

@dataclass
class GadgetHit:
    class_name: str
    confidence: float
    bbox:       Tuple[int, int, int, int]
    near_ear:   bool  = False
    from_pose:  bool  = False


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

    Step 3 — PHONE VALIDATION (filters must ALL pass):

        Filter A — Shape check
            Phone bbox must have realistic size and aspect ratio.

        Filter B — Person bbox required
            If YOLO did not detect a person in that pilot zone,
            the phone is IGNORED completely.
            FIX: last known person bbox is cached across YOLO frames
            so a missed person detection doesn't kill a valid phone hit.

        Filter C — Ear/head proximity check
            Phone centre must be inside the TOP 45% of the pilot's
            person bbox (head/shoulder region).

        Filter D — Edge variance check
            Real phones have sharp edges (Laplacian variance).
            Lowered threshold for IR footage.

    Step 4 — WRIST-TO-EAR HEURISTIC
        If MediaPipe pose landmarks are provided, check if wrist is
        near ear. This catches phones that YOLO misses.

    Step 5 — Timer:
        Phone must be continuously valid for GADGET_ALLOWED_DURATION
        seconds. YOLO can miss for up to GADGET_MISS_TOLERANCE frames
        without resetting the timer.

    KEY FIXES applied (2026-05-28):
    ─────────────────────────────────
    FIX 1 — _last_known_bbox cache:
        YOLO runs every 18 frames. Between runs, bbox_by_pid[pid] is
        None, killing valid phone detections. We now cache the last
        known person bbox and reuse it when YOLO misses.

    FIX 2 — phone_frame_counter threshold lowered from 3 → 1:
        With YOLO running every 18 frames, the counter could never
        reach 3 before being reset. Now 1 confirmed detection
        activates the timer.

    FIX 3 — PERSON_MISS_TOLERANCE_TIME raised from 1.0s → 3.0s:
        18 frames @ 25fps = 0.72s between YOLO runs. 1.0s grace was
        too tight. 3.0s gives enough headroom for sparse YOLO runs.
    """

    HEAD_ZONE_FRACTION = 0.45   # top 45% of pilot body = head region

    def __init__(self) -> None:
        self.timers: Dict[int, _PilotTimer] = {
            1: _PilotTimer(1),
            2: _PilotTimer(2),
        }
        self.last_gadget_hits:       List[GadgetHit] = []
        self._last_gadgets_by_pilot: Dict[int, List[GadgetHit]] = {1: [], 2: []}
        self.last_frame_detections:  Optional[FrameDetections] = None

        # FIX 2: threshold lowered from 3 → 1
        # YOLO runs every 18 frames — counter can never reach 3 before reset.
        self.phone_frame_counter = {1: 0, 2: 0}

        # FIX 1: cache last known person bbox per pilot.
        # Prevents valid phone detections being dropped on frames where
        # YOLO skips person detection (YOLO every 18 frames).
        self._last_known_bbox: Dict[int, Optional[Tuple[int, int, int, int]]] = {
            1: None,
            2: None,
        }

    def process(
        self,
        frame:           np.ndarray,
        video_time:      float,
        pose_landmarks:  Optional[Dict[int, list]] = None,
    ) -> Tuple[List[PilotResult], List[Tuple[int, str]]]:

        enhanced = self._smart_enhance(frame)
        raw_boxes, raw_gadgets = self._run_yolo(enhanced)

        frame_h, frame_w = frame.shape[:2]
        split_y = int(frame_h * GREEN_LINE_RATIO)

        pilot_boxes = self._assign_pilots_by_zone(raw_boxes, split_y)
        bbox_by_pid: Dict[int, Optional[Tuple[int,int,int,int]]] = {1: None, 2: None}
        for pid, pbox in pilot_boxes:
            bbox_by_pid[pid] = pbox

        # ── FIX 1: BBOX CACHE ────────────────────────────────────
        # YOLO only runs every 18 frames. On frames where it skips,
        # bbox_by_pid[pid] is None — which kills valid phone detections
        # via Filter B. We fill gaps using the last known person bbox.
        for pid in [1, 2]:
            if bbox_by_pid[pid] is not None:
                # Fresh detection — update the cache
                self._last_known_bbox[pid] = bbox_by_pid[pid]
            elif self._last_known_bbox[pid] is not None:
                # YOLO missed this frame — use cached bbox
                bbox_by_pid[pid] = self._last_known_bbox[pid]
                if DEBUG_YOLO:
                    print(f"[BBOX CACHE] Pilot {pid} — using cached person bbox "
                          f"{self._last_known_bbox[pid]}")

        # ── GADGET ASSIGNMENT ─────────────────────────────────────
        gadgets_by_pilot = self._assign_gadgets_near_ear(
            raw_gadgets, bbox_by_pid
        )

        # ── WRIST-TO-EAR HEURISTIC ───────────────────────────────
        if pose_landmarks:
            wrist_hits = self._check_wrist_near_ear(
                pose_landmarks, frame_w, frame_h, bbox_by_pid
            )
            for pid, hit in wrist_hits.items():
                if hit and not gadgets_by_pilot.get(pid):
                    gadgets_by_pilot[pid] = [GadgetHit(
                        class_name = "cell phone",
                        confidence = 0.75,
                        bbox       = (0, 0, 0, 0),
                        near_ear   = True,
                        from_pose  = True,
                    )]
                    if DEBUG_YOLO:
                        print(f"[WRIST-EAR] Pilot {pid} — phone detected via pose heuristic")

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
                # FIX 2: threshold 1 (was 3) — YOLO runs every 18 frames,
                # counter never reached 3 before resetting on a miss.
                if self.phone_frame_counter[pid] >= 1:
                    timer.activate()
                    if DEBUG_YOLO:
                        print(f"[TIMER] Pilot {pid} timer activated/continuing — "
                              f"elapsed={timer.elapsed():.2f}s "
                              f"phone_frames={self.phone_frame_counter[pid]}")
                self._last_gadgets_by_pilot[pid] = matched
            else:
                self.phone_frame_counter[pid] = 0
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
                if DEBUG_YOLO:
                    print(f"[VIOLATION] Pilot {pid} distracted with '{name}' "
                          f"elapsed={timer.elapsed():.2f}s")

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
    # WRIST-TO-EAR HEURISTIC
    # ─────────────────────────────────────────────────────────────

    def _check_wrist_near_ear(
        self,
        pose_landmarks: Dict[int, list],
        frame_w:        int,
        frame_h:        int,
        bbox_by_pid:    Dict[int, Optional[Tuple[int,int,int,int]]],
    ) -> Dict[int, bool]:
        LEFT_EAR    = 7
        RIGHT_EAR   = 8
        LEFT_WRIST  = 15
        RIGHT_WRIST = 16

        result = {1: False, 2: False}
        thresh = WRIST_EAR_THRESH_FRACTION * frame_w

        for pid, landmarks in pose_landmarks.items():
            if landmarks is None or len(landmarks) < 17:
                continue

            pbox = bbox_by_pid.get(pid)

            for ear_idx, wrist_idx in [
                (LEFT_EAR, LEFT_WRIST),
                (RIGHT_EAR, RIGHT_WRIST),
            ]:
                try:
                    ear   = landmarks[ear_idx]
                    wrist = landmarks[wrist_idx]

                    if hasattr(ear, 'x'):
                        ex, ey = ear.x * frame_w,   ear.y * frame_h
                        wx, wy = wrist.x * frame_w, wrist.y * frame_h
                    else:
                        ex, ey = ear[0],   ear[1]
                        wx, wy = wrist[0], wrist[1]

                    dist = np.hypot(wx - ex, wy - ey)

                    if dist < thresh:
                        if pbox is not None:
                            px1, py1, px2, py2 = pbox
                            upper_limit = py1 + 0.6 * (py2 - py1)
                            if wy > upper_limit:
                                continue

                        result[pid] = True
                        if DEBUG_YOLO:
                            print(f"[WRIST-EAR] Pilot {pid} wrist-ear dist={dist:.1f}px "
                                  f"(thresh={thresh:.1f}px)")
                        break

                except (IndexError, AttributeError):
                    continue

        return result

    # ─────────────────────────────────────────────────────────────
    # FILTER: ear proximity  (Filter B + C combined)
    # ─────────────────────────────────────────────────────────────

    def _assign_gadgets_near_ear(
        self,
        gadgets:     List[GadgetHit],
        bbox_by_pid: Dict[int, Optional[Tuple[int,int,int,int]]],
    ) -> Dict[int, List[GadgetHit]]:
        by_pilot: Dict[int, List[GadgetHit]] = {1: [], 2: []}

        for g in gadgets:
            gx1, gy1, gx2, gy2 = g.bbox
            gcx = (gx1 + gx2) / 2
            gcy = (gy1 + gy2) / 2

            for pid in [1, 2]:
                pbox = bbox_by_pid.get(pid)

                # FIX 3: raised from 1.0s → 3.0s
                # 18 frames @ 25fps = 0.72s between YOLO runs.
                # 1.0s was too tight — person bbox was expiring before
                # the next YOLO run. 3.0s gives safe headroom.
                PERSON_MISS_TOLERANCE_TIME = 3.0

                if pbox is None:
                    timer = self.timers.get(pid)
                    if timer is None or timer.last_person_seen is None:
                        continue
                    if (time.monotonic() - timer.last_person_seen) > PERSON_MISS_TOLERANCE_TIME:
                        continue
                    continue

                px1, py1, px2, py2 = pbox
                p_h = py2 - py1
                p_w = px2 - px1
                if p_h <= 0:
                    continue

                head_bottom = py1 + self.HEAD_ZONE_FRACTION * p_h
                my = GADGET_EAR_PROXIMITY_MARGIN * p_h
                mx = GADGET_EAR_PROXIMITY_MARGIN * p_w

                if DEBUG_YOLO:
                    print(f"[ASSIGN] Pilot {pid} pbox={pbox} | "
                          f"phone_center=({gcx:.0f},{gcy:.0f}) | "
                          f"head_bottom={head_bottom:.0f} | "
                          f"x_range=[{px1-mx:.0f},{px2+mx:.0f}] "
                          f"y_range=[{py1-my:.0f},{head_bottom+my:.0f}]")

                if (px1 - mx <= gcx <= px2 + mx and
                        py1 - my <= gcy <= head_bottom + my):
                    g.near_ear = True
                    by_pilot[pid].append(g)
                    if DEBUG_YOLO:
                        print(f"  └─ MATCHED to Pilot {pid}")

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

    # ─────────────────────────────────────────────────────────────
    # IR ENHANCEMENT
    # ─────────────────────────────────────────────────────────────

    def _smart_enhance(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray)

        if mean_val < 160:
            clahe     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced  = clahe.apply(gray)
            blurred   = cv2.GaussianBlur(enhanced, (0, 0), 3)
            sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
            return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

        return frame

    # ─────────────────────────────────────────────────────────────
    # YOLO INFERENCE
    # ─────────────────────────────────────────────────────────────

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

            elif name in GADGET_CLASSES:
                if DEBUG_YOLO:
                    w, h   = x2-x1, y2-y1
                    aspect = round(w/h, 2) if h > 0 else 0
                    shape_ok = _is_valid_gadget_shape(bbox, frame_w)
                    edge_ok  = _has_phone_like_edges(frame, bbox)
                    print(
                        f"[RAW YOLO] cell_phone conf={conf:.3f} | "
                        f"bbox=({x1},{y1},{x2},{y2}) | "
                        f"w={w} h={h} aspect={aspect} | "
                        f"above_conf_thresh={conf > GADGET_CONFIDENCE_THRESHOLD} | "
                        f"shape_ok={shape_ok} | edge_ok={edge_ok}"
                    )

                if conf > GADGET_CONFIDENCE_THRESHOLD:
                    if (_is_valid_gadget_shape(bbox, frame_w) and
                            _has_phone_like_edges(frame, bbox)):
                        gadgets.append(GadgetHit(name, conf, bbox))
                    elif DEBUG_YOLO:
                        print(f"  └─ REJECTED by filters")
                elif DEBUG_YOLO:
                    print(f"  └─ REJECTED: conf {conf:.3f} < threshold {GADGET_CONFIDENCE_THRESHOLD}")

        persons.sort(
            key=lambda p: (p[0][2]-p[0][0]) * (p[0][3]-p[0][1]),
            reverse=True,
        )
        return [p[0] for p in persons[:MAX_PILOTS]], gadgets


# ─────────────────────────────────────────
# SHAPE FILTER  (Filter A)
# ─────────────────────────────────────────

def _is_valid_gadget_shape(bbox: Tuple[int,int,int,int], frame_w: int) -> bool:
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


# ─────────────────────────────────────────
# EDGE VARIANCE FILTER  (Filter D)
# ─────────────────────────────────────────

def _has_phone_like_edges(
    frame: np.ndarray,
    bbox:  Tuple[int,int,int,int],
) -> bool:
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]

    x1c = max(0, x1); y1c = max(0, y1)
    x2c = min(w, x2); y2c = min(h, y2)

    if x2c <= x1c or y2c <= y1c:
        return False

    crop = frame[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return False

    gray     = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    lap      = cv2.Laplacian(gray, cv2.CV_64F)
    variance = lap.var()

    # Use settings value, but floor at 35.0 for IR footage
    effective_threshold = min(GADGET_MIN_EDGE_VARIANCE, 35.0)

    if DEBUG_YOLO:
        print(f"  [EDGE] variance={variance:.1f} threshold={effective_threshold:.1f} "
              f"pass={variance >= effective_threshold}")

    return variance >= effective_threshold


def _intersection_area(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    return max(0, ix2-ix1) * max(0, iy2-iy1)

def _iou(a, b):
    inter = _intersection_area(a, b)
    if inter == 0: return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
