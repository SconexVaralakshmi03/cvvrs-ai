from __future__ import annotations

import os as _os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import (
    CURVE_CHECK_POSE_MODEL,
    CURVE_CHECK_PERSON_CONFIDENCE,
    CURVE_CHECK_KEYPOINT_CONFIDENCE,
    CURVE_CHECK_POSE_IMGSZ,
    CURVE_CHECK_SCORE_THRESHOLD,
    CURVE_CHECK_DOOR_ROI,
    CURVE_CHECK_CONFIRM_FRAMES,
    CURVE_CHECK_MISS_TOLERANCE,
    CURVE_CHECK_ALLOWED_DURATION,
    CURVE_CHECK_PILOT_ID,
    RELOG_INTERVAL,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

PERSON_CLASS_ID = 0

# COCO-17 keypoint indices (ultralytics pose output layout) — same
# convention the standalone prototype used.
_KP_NOSE          = 0
_KP_LEFT_EYE      = 1
_KP_RIGHT_EYE     = 2
_KP_LEFT_EAR      = 3
_KP_RIGHT_EAR     = 4
_KP_LEFT_SHOULDER = 5
_KP_RIGHT_SHOULDER = 6
_KP_LEFT_ANKLE    = 15
_KP_RIGHT_ANKLE   = 16


# ─────────────────────────────────────────────────────────────────────────────
# YOLO POSE MODEL (lazy, per-process singleton)
# ─────────────────────────────────────────────────────────────────────────────
#
# Mirrors detector/gadget_detector.py's _get_model()/release_model() pattern:
# loaded once per worker process and reused across every video/journey. This
# detector needs FULL-FRAME person+pose detection (the ALP leaning out the
# door leaves the seat-zone crops the rest of the pipeline works with), which
# is why — like HandRaisePoseEngine in detector/hand_raise_detector.py — it
# owns a dedicated model instead of reusing main.py's MediaPipe self._pose or
# GadgetDetector's YOLO instance.

_model = None
_MODEL_DEVICE = _os.environ.get("CURVE_CHECK_YOLO_DEVICE", "cuda:0")


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        try:
            import torch
            _has_cuda = torch.cuda.is_available()
        except Exception:
            torch = None
            _has_cuda = False

        if _MODEL_DEVICE.startswith("cuda") and _has_cuda:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        _model = YOLO(CURVE_CHECK_POSE_MODEL)

        if _MODEL_DEVICE.startswith("cuda") and _has_cuda:
            try:
                _model.to(_MODEL_DEVICE)
            except Exception:
                pass
        # No CUDA → leave on default (CPU) device.

    return _model


def release_model() -> None:
    """
    Explicitly drop the pose-model singleton and free its CUDA memory.
    Not currently wired into resource_manager.py's per-journey cleanup (see
    the note in that file about gadget_detector.release_model() never
    actually being called — per-video/journey model teardown there happens
    via gc.collect() + the CUDA-cache flush instead). Kept here so it's
    available the same way GadgetDetector's equivalent is, if that wiring
    is ever added.
    """
    global _model
    if _model is not None:
        try:
            del _model
        finally:
            _model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def _point_inside_polygon(x: float, y: float, polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0


def _get_point(
    kpts: np.ndarray,
    index: int,
    conf_thresh: float = CURVE_CHECK_KEYPOINT_CONFIDENCE,
) -> Optional[Tuple[float, float, float]]:
    if kpts is None or index >= len(kpts):
        return None
    kp = kpts[index]
    x, y = float(kp[0]), float(kp[1])
    conf = float(kp[2]) if len(kp) >= 3 else 1.0
    if conf < conf_thresh:
        return None
    return (x, y, conf)


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CurveCheckCandidate:
    """One detected, non-pilot person evaluated against the door zone this frame."""
    bbox:            Tuple[int, int, int, int]
    score:           float
    door_zone:       bool
    outside_looking: bool
    reason:          str


@dataclass
class CurveCheckResult:
    """Per-frame state of the (single) curve-checking episode, for drawing/logging."""
    pilot_id:        int
    outside_looking: bool = False
    timer_value:     float = 0.0
    bbox:            Optional[Tuple[int, int, int, int]] = None
    score:           float = 0.0
    reason:          str = ""


# ─────────────────────────────────────────────────────────────────────────────
# TIMER  (video_time-based duration, detector-cycle-based debounce/miss —
# same split used by _PilotTimer in gadget_detector.py and hand_raise_detector.py)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _CurveCheckTimer:
    pilot_id:     int
    start_vtime:  Optional[float] = None
    last_logged:  Optional[float] = None
    miss_cycles:  int             = 0
    confirmed:    bool            = False

    def activate(self, video_time: float) -> None:
        self.miss_cycles = 0
        if self.start_vtime is None:
            self.start_vtime = video_time

    def miss(self) -> bool:
        """Call once per detector cycle with no qualifying candidate.
        Returns True once CURVE_CHECK_MISS_TOLERANCE is exceeded."""
        self.miss_cycles += 1
        return self.miss_cycles > CURVE_CHECK_MISS_TOLERANCE

    def reset(self) -> None:
        self.start_vtime = None
        self.last_logged = None
        self.miss_cycles = 0
        self.confirmed   = False

    def elapsed(self, video_time: float) -> float:
        if self.start_vtime is None:
            return 0.0
        return max(0.0, video_time - self.start_vtime)

    def should_log(self, video_time: float) -> bool:
        if self.elapsed(video_time) < CURVE_CHECK_ALLOWED_DURATION:
            return False
        if self.last_logged is None:
            return True
        return (video_time - self.last_logged) >= RELOG_INTERVAL

    def mark_logged(self, video_time: float) -> None:
        self.last_logged = video_time
        self.confirmed   = True

    def close_if_confirmed(self, end_video_time: float) -> Optional[Tuple[float, float, float]]:
        result = None
        if self.confirmed and self.start_vtime is not None:
            result = (self.start_vtime, end_video_time, max(0.0, end_video_time - self.start_vtime))
            print(f"[TIMER-CLOSE][curve_checking] pilot={self.pilot_id} "
                  f"start={self.start_vtime:.2f} end={end_video_time:.2f} "
                  f"true_dur={result[2]:.2f}")
        self.reset()
        return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class CurveCheckingDetector:
    """
    Detects the assistant loco pilot (ALP) leaning out of the cabin door to
    visually check the curve ahead — a REQUIRED positive procedure, logged
    the same way HandRaiseDetector logs signaling (LOW severity, not a
    distraction), not a violation.

    Ported from a standalone YOLO26-pose prototype (DOOR_ROI + per-person
    head-orientation/doorway-posture scoring). Restructured here so it
    plugs into the same call/return contract as every other detector in
    detector/*.py:

        results, log_events, completed_events = detector.process(...)

      results          : List[CurveCheckResult]   (len 1 — single tracked slot)
      log_events        : List[(pilot_id, event_str)]
      completed_events  : List[(pilot_id, start_vtime, end_vtime, true_duration)]

    Frame-skip note: the 8-consecutive-frame / 60-frame-cooldown thresholds
    in the original prototype only meant "~0.3s / ~2.4s" because that script
    ran on every raw frame. This pipeline sub-samples (RAW_FRAME_SKIP) and
    additionally calls each detector on its own cadence (e.g. every
    CURVE_CHECK_EVERY processed frames), so a fixed frame count would mean a
    different amount of real time depending on those settings. This
    detector fixes that: CURVE_CHECK_ALLOWED_DURATION (the "is this real"
    gate) is in VIDEO SECONDS via video_time, so the detector behaves
    identically regardless of frame-skip/cadence tuning. Only the
    single-cycle jitter debounce (CURVE_CHECK_CONFIRM_FRAMES /
    CURVE_CHECK_MISS_TOLERANCE) is counted in detector-cycle units — that's
    intentional and matches HAND_RAISE_CONFIRM_FRAMES / GADGET_MISS_TOLERANCE
    elsewhere in this codebase, since a "cycle" is a fixed unit of this
    detector's own invocation cadence, not of raw video frames.
    """

    def __init__(
        self,
        door_roi_norm: Optional[Tuple[Tuple[float, float], ...]] = None,
        pilot_id: int = CURVE_CHECK_PILOT_ID,
    ) -> None:
        self._door_roi_norm = door_roi_norm or CURVE_CHECK_DOOR_ROI
        self._timer          = _CurveCheckTimer(pilot_id)
        self._confirm_cycles = 0
        self.last_candidates: List[CurveCheckCandidate] = []

    # ──────────────────────────────────────────────────────────────
    # RESOURCE CLEANUP
    # ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        """
        Release this instance's state. The pose model itself is a
        per-process singleton (see _get_model() above) and is intentionally
        NOT torn down here — same lifecycle contract as GadgetDetector's
        YOLO model (see resource_manager.py's note on model-handle
        ownership). Safe to call multiple times.
        """
        self._timer.reset()
        self._confirm_cycles = 0
        self.last_candidates = []

    # ──────────────────────────────────────────────────────────────
    # INTERNAL — door ROI
    # ──────────────────────────────────────────────────────────────

    def _door_roi_px(self, frame_width: int, frame_height: int) -> np.ndarray:
        pts = [
            (int(x * frame_width), int(y * frame_height))
            for x, y in self._door_roi_norm
        ]
        return np.array(pts, dtype=np.int32)

    # ──────────────────────────────────────────────────────────────
    # INTERNAL — per-person scoring (ported from the standalone prototype)
    # ──────────────────────────────────────────────────────────────

    def _analyze_person(
        self,
        kpts: np.ndarray,
        bbox: Tuple[float, float, float, float],
        door_roi: np.ndarray,
    ) -> Tuple[bool, float, bool, str]:
        x1, y1, x2, y2 = bbox
        person_width  = x2 - x1
        person_height = y2 - y1
        if person_width <= 0 or person_height <= 0:
            return False, 0.0, False, ""

        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

        left_ankle  = _get_point(kpts, _KP_LEFT_ANKLE)
        right_ankle = _get_point(kpts, _KP_RIGHT_ANKLE)
        if left_ankle and right_ankle:
            foot_x = (left_ankle[0] + right_ankle[0]) / 2
            foot_y = (left_ankle[1] + right_ankle[1]) / 2
        else:
            foot_x, foot_y = center_x, y2

        left_shoulder  = _get_point(kpts, _KP_LEFT_SHOULDER)
        right_shoulder = _get_point(kpts, _KP_RIGHT_SHOULDER)

        center_in_door   = _point_inside_polygon(center_x, center_y, door_roi)
        feet_in_door     = _point_inside_polygon(foot_x, foot_y, door_roi)
        shoulder_in_door = False
        if left_shoulder:
            shoulder_in_door |= _point_inside_polygon(left_shoulder[0], left_shoulder[1], door_roi)
        if right_shoulder:
            shoulder_in_door |= _point_inside_polygon(right_shoulder[0], right_shoulder[1], door_roi)

        door_person = center_in_door or feet_in_door or shoulder_in_door
        if not door_person:
            return False, 0.0, False, ""

        nose      = _get_point(kpts, _KP_NOSE)
        left_eye  = _get_point(kpts, _KP_LEFT_EYE)
        right_eye = _get_point(kpts, _KP_RIGHT_EYE)
        left_ear  = _get_point(kpts, _KP_LEFT_EAR)
        right_ear = _get_point(kpts, _KP_RIGHT_EAR)

        head_score = 0.0
        reasons: List[str] = []

        face_points = sum(p is not None for p in (nose, left_eye, right_eye))
        ears        = sum(p is not None for p in (left_ear, right_ear))

        # Back-facing (looking away from the cabin, out the door)
        if face_points == 0 and ears >= 1:
            head_score += 0.55
            reasons.append("back-facing head")
        elif face_points == 0 and ears == 0:
            head_score += 0.25
            reasons.append("face not visible")

        # Side-facing (leaning, head turned toward the door)
        if nose and left_ear and right_ear:
            ear_mid_x  = (left_ear[0] + right_ear[0]) / 2
            head_width = abs(left_ear[0] - right_ear[0])
            if head_width > 5:
                nose_offset = (nose[0] - ear_mid_x) / head_width
                if abs(nose_offset) > 0.45:
                    head_score += 0.45
                    reasons.append("side-facing head")

        body_score = 0.0
        if left_shoulder and right_shoulder:
            shoulder_width = _distance(
                (left_shoulder[0], left_shoulder[1]),
                (right_shoulder[0], right_shoulder[1]),
            )
            aspect = person_height / max(shoulder_width, 1)
            if aspect > 1.4:
                body_score += 0.20
                reasons.append("standing at doorway")

        head_near_door = False
        for candidate_pt in (nose, left_ear, right_ear):
            if candidate_pt:
                head_near_door = _point_inside_polygon(candidate_pt[0], candidate_pt[1], door_roi)
                if head_near_door:
                    break
        if head_near_door:
            head_score += 0.20
            reasons.append("head near doorway")

        score = min(head_score + body_score, 1.0)
        outside_looking = door_person and score >= CURVE_CHECK_SCORE_THRESHOLD

        return door_person, score, outside_looking, ", ".join(reasons)

    # ──────────────────────────────────────────────────────────────
    # PUBLIC — call once per detector cycle
    # ──────────────────────────────────────────────────────────────

    def process(
        self,
        frame: np.ndarray,
        video_time: float,
        known_pilot_boxes: Optional[List[Tuple[int, Tuple[int, int, int, int]]]] = None,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        exclude_iou: float = 0.30,
    ) -> Tuple[List[CurveCheckResult], List[Tuple[int, str]], List[Tuple[int, float, float, float]]]:
        """
        Parameters
        ----------
        frame              : BGR image for THIS detector cycle (already
                              subject to whatever RAW_FRAME_SKIP / cadence
                              main.py is calling this at).
        video_time          : Video-timeline seconds for this frame.
        known_pilot_boxes   : Optional [(pilot_id, bbox), ...] from
                              GadgetDetector's own results this cycle. Any
                              detected person overlapping one of these
                              (IoU >= exclude_iou) is excluded from
                              candidates — the seated LP/ALP are never
                              themselves flagged as "leaning out the door".
                              If omitted, every detected person is a
                              candidate (only the door-zone gate applies).
        frame_width/height  : Defaults to frame.shape if not given.

        Returns
        -------
        results          : List[CurveCheckResult] — always length 1 (the
                            single tracked door-zone slot).
        log_events       : List[(pilot_id, event_str)]
        completed_events : List[(pilot_id, start_vtime, end_vtime, true_duration)]
        """
        h, w = frame.shape[:2]
        frame_width  = frame_width or w
        frame_height = frame_height or h

        door_roi = self._door_roi_px(frame_width, frame_height)

        model = _get_model()
        yolo_results = model.predict(
            frame,
            conf=CURVE_CHECK_PERSON_CONFIDENCE,
            imgsz=CURVE_CHECK_POSE_IMGSZ,
            device=_MODEL_DEVICE,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )

        candidates: List[CurveCheckCandidate] = []

        if yolo_results:
            result = yolo_results[0]
            if result.boxes is not None and result.keypoints is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                keypoints = result.keypoints.data.cpu().numpy()

                for i in range(len(boxes)):
                    bbox = tuple(map(int, boxes[i]))

                    if known_pilot_boxes:
                        overlaps_pilot = any(
                            _bbox_iou(bbox, pbox) >= exclude_iou
                            for _pid, pbox in known_pilot_boxes
                            if pbox is not None
                        )
                        if overlaps_pilot:
                            continue

                    door_zone, score, outside_looking, reason = self._analyze_person(
                        keypoints[i], boxes[i], door_roi
                    )
                    if not door_zone:
                        continue

                    candidates.append(CurveCheckCandidate(
                        bbox=bbox, score=score, door_zone=door_zone,
                        outside_looking=outside_looking, reason=reason,
                    ))

        self.last_candidates = candidates

        best = max(
            (c for c in candidates if c.outside_looking),
            key=lambda c: c.score,
            default=None,
        )

        timer = self._timer
        log_events: List[Tuple[int, str]] = []
        completed_events: List[Tuple[int, float, float, float]] = []

        if best is not None:
            self._confirm_cycles += 1
            if self._confirm_cycles >= CURVE_CHECK_CONFIRM_FRAMES:
                timer.activate(video_time)
        else:
            self._confirm_cycles = 0
            if timer.miss():
                closed = timer.close_if_confirmed(video_time)
                if closed is not None:
                    start_v, end_v, true_dur = closed
                    completed_events.append((timer.pilot_id, start_v, end_v, true_dur))

        elapsed  = timer.elapsed(video_time)
        confirmed_now = elapsed >= CURVE_CHECK_ALLOWED_DURATION

        if confirmed_now and timer.should_log(video_time):
            log_events.append((timer.pilot_id, "Curve checking — ALP looking outside door"))
            timer.mark_logged(video_time)

        results = [CurveCheckResult(
            pilot_id        = timer.pilot_id,
            outside_looking = confirmed_now,
            timer_value     = elapsed,
            bbox            = best.bbox if best else None,
            score           = best.score if best else 0.0,
            reason          = best.reason if best else "",
        )]

        return results, log_events, completed_events


# ─────────────────────────────────────────────────────────────────────────────
# DRAW HELPERS (optional — mirrors utils/draw.py's style; only called when
# main.py's DRAW flag is True, same as every other detector's overlay)
# ─────────────────────────────────────────────────────────────────────────────

def draw_door_roi(frame: np.ndarray, door_roi_px: np.ndarray) -> np.ndarray:
    overlay = frame.copy()
    cv2.fillPoly(overlay, [door_roi_px], (255, 150, 0))
    frame = cv2.addWeighted(overlay, 0.12, frame, 0.88, 0)
    cv2.polylines(frame, [door_roi_px], True, (255, 150, 0), 3)
    cv2.putText(frame, "DOOR / CURVE-CHECK ZONE", tuple(door_roi_px[0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 0), 2)
    return frame


def draw_curve_check_overlay(frame: np.ndarray, result: CurveCheckResult) -> np.ndarray:
    if result.bbox is None:
        return frame
    x1, y1, x2, y2 = result.bbox
    color = (0, 165, 255) if result.outside_looking else (255, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"CURVE CHECK {result.score:.2f}" if result.outside_looking else f"door zone {result.score:.2f}"
    cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame