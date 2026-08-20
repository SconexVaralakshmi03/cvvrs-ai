from __future__ import annotations

import os as _os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import (
    CURVE_CHECK_POSE_MODEL,
    CURVE_CHECK_PERSON_CONFIDENCE,
    CURVE_CHECK_KEYPOINT_CONFIDENCE,
    CURVE_CHECK_POSE_IMGSZ,
    CURVE_CHECK_SCORE_THRESHOLD,
    CURVE_CHECK_DRIVER_ROI,
    CURVE_CHECK_DOOR_ROI,
    CURVE_CHECK_MIN_CONSECUTIVE_FRAMES,
    CURVE_CHECK_EVENT_COOLDOWN_FRAMES,
    CURVE_CHECK_PILOT_ID,
)

# ─────────────────────────────────────────────────────────────────────────────
# SOURCE OF TRUTH
# ─────────────────────────────────────────────────────────────────────────────
#
# This file is a line-for-line port of the standalone YOLO26-pose
# curve-checking prototype's detection logic (DRIVER_ROI/select_driver,
# DOOR_ROI/analyze_person, and the outside_counter/MIN_CONSECUTIVE_FRAMES/
# EVENT_COOLDOWN_FRAMES temporal filter). Every function below that mirrors
# a standalone function keeps the standalone name (in _private form) and
# body unchanged — thresholds, ROI values, and control flow are not
# modified or "improved" here. The only things this file adds beyond the
# standalone script are the class wrapper (so per-video state persists
# across process() calls the way the standalone script's loop-local
# variables persisted across frames) and the (results, log_events,
# completed_events) return contract main.py's detector-dispatch pattern
# expects. completed_events is always [] — the standalone script has no
# start/end episode concept, only a cooldown-gated snapshot event, so
# there is nothing to "complete".
#
# Two standalone globals became per-instance config instead of module
# globals (CURVE_CHECK_DRIVER_ROI, CURVE_CHECK_DOOR_ROI in
# config/settings.py) purely so multiple detector instances don't share
# mutable state — their VALUES are unchanged from the standalone script.

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

PERSON_CLASS_ID = 0

# COCO-17 keypoint indices (ultralytics pose output layout) — same
# convention the standalone prototype used.
_KP_NOSE           = 0
_KP_LEFT_EYE       = 1
_KP_RIGHT_EYE      = 2
_KP_LEFT_EAR       = 3
_KP_RIGHT_EAR      = 4
_KP_LEFT_SHOULDER  = 5
_KP_RIGHT_SHOULDER = 6
_KP_LEFT_ANKLE     = 15
_KP_RIGHT_ANKLE    = 16


# ─────────────────────────────────────────────────────────────────────────────
# YOLO POSE MODEL (lazy, per-process singleton)
# ─────────────────────────────────────────────────────────────────────────────
#
# Mirrors detector/gadget_detector.py's _get_model()/release_model() pattern:
# loaded once per worker process and reused across every video/journey. This
# detector needs FULL-FRAME person+pose detection — exactly what the
# standalone script's pose_model.predict(frame, ...) call does (no ROI
# cropping beforehand) — which is why, like HandRaisePoseEngine in
# detector/hand_raise_detector.py, it owns a dedicated model instead of
# reusing main.py's MediaPipe self._pose or GadgetDetector's YOLO instance.

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
# UTILITY — ported verbatim from the standalone script
# ─────────────────────────────────────────────────────────────────────────────

def _distance(p1, p2) -> float:
    """Exact port of the standalone script's distance()."""
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def _point_inside_polygon(x: float, y: float, polygon: np.ndarray) -> bool:
    """Exact port of the standalone script's point_inside_polygon()."""
    return cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0


def _get_point(
    kpts: np.ndarray,
    index: int,
    conf_thresh: float = CURVE_CHECK_KEYPOINT_CONFIDENCE,
) -> Optional[Tuple[float, float, float]]:
    """Exact port of the standalone script's get_point()."""
    if kpts is None or index >= len(kpts):
        return None
    kp = kpts[index]
    x, y = float(kp[0]), float(kp[1])
    conf = float(kp[2]) if len(kp) >= 3 else 1.0
    if conf < conf_thresh:
        return None
    return (x, y, conf)


def _inside_driver_roi(
    bbox,
    frame_width: int,
    frame_height: int,
    driver_roi: Tuple[float, float, float, float],
) -> bool:
    """Exact port of the standalone script's inside_driver_roi()."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    rx1 = driver_roi[0] * frame_width
    ry1 = driver_roi[1] * frame_height
    rx2 = driver_roi[2] * frame_width
    ry2 = driver_roi[3] * frame_height

    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def _select_driver(
    persons: List[dict],
    frame_width: int,
    frame_height: int,
    driver_roi: Tuple[float, float, float, float],
) -> Optional[dict]:
    """Exact port of the standalone script's select_driver(): largest-area
    person whose bbox center falls inside driver_roi, or None."""
    candidates = []
    for person in persons:
        bbox = person["bbox"]
        if not _inside_driver_roi(bbox, frame_width, frame_height, driver_roi):
            continue
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        candidates.append((area, person))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _analyze_person(kpts, bbox, door_roi: np.ndarray) -> Optional[dict]:
    """
    Exact port of the standalone script's analyze_person(). Returns None
    only for a degenerate (non-positive width/height) bbox — otherwise
    always returns a dict with door_person/outside_looking/score/reason,
    exactly as the standalone script did (including its "hard gate" early
    return for persons who never reach DOOR_ROI, and its exact scoring
    weights: 0.55 back-facing / 0.25 face-not-visible / 0.45 side-facing /
    0.20 standing-at-doorway / 0.20 head-near-doorway).
    """
    x1, y1, x2, y2 = bbox
    person_width = x2 - x1
    person_height = y2 - y1

    if person_width <= 0 or person_height <= 0:
        return None

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # -- FEET --------------------------------------------------------------
    left_ankle = _get_point(kpts, _KP_LEFT_ANKLE)
    right_ankle = _get_point(kpts, _KP_RIGHT_ANKLE)
    if left_ankle and right_ankle:
        foot_x = (left_ankle[0] + right_ankle[0]) / 2
        foot_y = (left_ankle[1] + right_ankle[1]) / 2
    else:
        foot_x = center_x
        foot_y = y2

    # -- DOOR / OUTSIDE RANGE GATE ------------------------------------------
    center_in_door = _point_inside_polygon(center_x, center_y, door_roi)
    feet_in_door = _point_inside_polygon(foot_x, foot_y, door_roi)

    left_shoulder = _get_point(kpts, _KP_LEFT_SHOULDER)
    right_shoulder = _get_point(kpts, _KP_RIGHT_SHOULDER)

    shoulder_in_door = False
    if left_shoulder:
        shoulder_in_door |= _point_inside_polygon(left_shoulder[0], left_shoulder[1], door_roi)
    if right_shoulder:
        shoulder_in_door |= _point_inside_polygon(right_shoulder[0], right_shoulder[1], door_roi)

    # HARD GATE: no door-zone presence = never POSSIBLE OUTSIDE.
    door_person = center_in_door or feet_in_door or shoulder_in_door

    if not door_person:
        return {
            "door_person": False,
            "outside_looking": False,
            "score": 0.0,
            "reason": "Outside range not reached",
        }

    # -- HEAD ----------------------------------------------------------------
    nose = _get_point(kpts, _KP_NOSE)
    left_eye = _get_point(kpts, _KP_LEFT_EYE)
    right_eye = _get_point(kpts, _KP_RIGHT_EYE)
    left_ear = _get_point(kpts, _KP_LEFT_EAR)
    right_ear = _get_point(kpts, _KP_RIGHT_EAR)

    head_score = 0.0
    reasons: List[str] = []

    face_points = sum(p is not None for p in (nose, left_eye, right_eye))
    ears = sum(p is not None for p in (left_ear, right_ear))

    # BACK-FACING
    if face_points == 0 and ears >= 1:
        head_score += 0.55
        reasons.append("back-facing head")
    elif face_points == 0 and ears == 0:
        head_score += 0.25
        reasons.append("face not visible")

    # SIDE-FACING
    if nose and left_ear and right_ear:
        ear_mid_x = (left_ear[0] + right_ear[0]) / 2
        head_width = abs(left_ear[0] - right_ear[0])
        if head_width > 5:
            nose_offset = (nose[0] - ear_mid_x) / head_width
            if abs(nose_offset) > 0.45:
                head_score += 0.45
                reasons.append("side-facing head")

    # BODY
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

    # HEAD NEAR DOOR
    head_near_door = False
    if nose:
        head_near_door = _point_inside_polygon(nose[0], nose[1], door_roi)
    elif left_ear:
        head_near_door = _point_inside_polygon(left_ear[0], left_ear[1], door_roi)
    elif right_ear:
        head_near_door = _point_inside_polygon(right_ear[0], right_ear[1], door_roi)

    if head_near_door:
        head_score += 0.20
        reasons.append("head near doorway")

    # -- FINAL -----------------------------------------------------------
    score = min(head_score + body_score, 1.0)
    outside_looking = door_person and score >= CURVE_CHECK_SCORE_THRESHOLD

    return {
        "door_person": door_person,
        "outside_looking": outside_looking,
        "score": score,
        "reason": ", ".join(reasons),
    }


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CurveCheckCandidate:
    """One detected, non-driver person evaluated against the door zone this
    cycle — kept for drawing/diagnostics, mirrors the fields analyze_person()
    computes per person in the standalone script's outside-analysis loop."""
    bbox:            Tuple[int, int, int, int]
    score:           float
    door_person:     bool
    outside_looking: bool
    reason:          str


@dataclass
class CurveCheckResult:
    """Per-cycle state, for drawing/logging. outside_counter mirrors the
    standalone script's outside_counter local variable exactly."""
    pilot_id:        int
    outside_looking: bool = False   # == standalone's looking_outside_confirmed
    outside_counter: int = 0
    bbox:            Optional[Tuple[int, int, int, int]] = None
    score:           float = 0.0
    reason:          str = ""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class CurveCheckingDetector:
    """
    Detects the assistant loco pilot (ALP) leaning out of the cabin door to
    visually check the curve ahead — a REQUIRED positive procedure, logged
    the same way HandRaiseDetector logs signaling (LOW severity, not a
    distraction), not a violation.

    Detection logic is a faithful port of the standalone YOLO26-pose
    prototype: this detector runs its own full-frame YOLO pose pass each
    cycle, picks "the driver" (LP) via DRIVER_ROI/select_driver exactly as
    the standalone script did, and scores every OTHER detected person
    against the literal-pixel DOOR_ROI via analyze_person(). It does NOT
    take pilot boxes from GadgetDetector and does NOT do any IoU-based
    pilot exclusion — that was integrated-pipeline-only logic layered on
    top of the standalone script in an earlier version of this file, and
    has been removed so behavior matches the standalone script exactly.

    Temporal filter is the standalone script's exact outside_counter /
    MIN_CONSECUTIVE_FRAMES / EVENT_COOLDOWN_FRAMES logic — a
    increment-on-hit / decay-by-1-on-miss counter gating a confirmed flag,
    then a cooldown between logged events, counted in units of "calls to
    process()" (this detector's own cycles) the same way the standalone
    script counted raw frames. See CURVE_CHECK_EVERY in config/settings.py
    for what one "cycle" corresponds to in main.py's frame cadence.

    Call/return contract matches every other detector.py in this pipeline:

        results, log_events, completed_events = detector.process(...)

      results          : List[CurveCheckResult]   (len 1 — single tracked slot)
      log_events        : List[(pilot_id, event_str)]
      completed_events  : List[(pilot_id, start_vtime, end_vtime, true_duration)]
                          — always [] here; the standalone script has no
                          start/end episode concept, only a cooldown-gated
                          snapshot event (see save_event() in the
                          standalone script), so there is nothing to
                          "complete". Kept in the return tuple purely for
                          call-site compatibility with main.py's
                          detector-dispatch pattern.
    """

    def __init__(
        self,
        driver_roi: Optional[Tuple[float, float, float, float]] = None,
        door_roi: Optional[Tuple[Tuple[int, int], ...]] = None,
        pilot_id: int = CURVE_CHECK_PILOT_ID,
    ) -> None:
        self._driver_roi = driver_roi or CURVE_CHECK_DRIVER_ROI
        # Literal pixel polygon — NOT normalized, NOT rescaled per frame,
        # exactly like the standalone script's module-level DOOR_ROI.
        self._door_roi = np.array(door_roi or CURVE_CHECK_DOOR_ROI, dtype=np.int32)
        self._pilot_id = pilot_id

        # Exact standalone temporal-filter state (outside_counter,
        # last_outside_event_frame), now per-instance instead of
        # per-process-loop-local so it persists correctly across process()
        # calls for this video without leaking into a different video's
        # detector instance.
        self.outside_counter = 0
        self.last_outside_event_frame = -999999
        self._call_number = 0

        self.last_candidates: List[CurveCheckCandidate] = []

    # ──────────────────────────────────────────────────────────────
    # RESOURCE CLEANUP
    # ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        """
        Reset this instance's per-video state. The pose model itself is a
        per-process singleton (see _get_model() above) and is intentionally
        NOT torn down here — same lifecycle contract as GadgetDetector's
        YOLO model. Safe to call multiple times.
        """
        self.outside_counter = 0
        self.last_outside_event_frame = -999999
        self._call_number = 0
        self.last_candidates = []

    # ──────────────────────────────────────────────────────────────
    # PUBLIC — call once per detector cycle
    # ──────────────────────────────────────────────────────────────

    def process(
        self,
        frame: np.ndarray,
        video_time: float,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ) -> Tuple[List[CurveCheckResult], List[Tuple[int, str]], List[Tuple[int, float, float, float]]]:
        """
        Parameters
        ----------
        frame              : BGR image for THIS detector cycle.
        video_time          : Video-timeline seconds for this frame (not
                              used in the detection logic itself — kept
                              only so record_violation() call sites in
                              main.py have a timestamp; the standalone
                              script's own event timestamp came from
                              frame_number / fps the same way).
        frame_width/height  : Defaults to frame.shape if not given.

        Returns
        -------
        results          : List[CurveCheckResult] — always length 1.
        log_events       : List[(pilot_id, event_str)]
        completed_events : List[(pilot_id, start_vtime, end_vtime, true_duration)]
                            — always [].
        """
        h, w = frame.shape[:2]
        frame_width = frame_width or w
        frame_height = frame_height or h

        self._call_number += 1
        frame_number = self._call_number  # standalone's frame_number, in this detector's own cycle units

        # ====================================================
        # YOLO POSE  (exact standalone predict() call/shape)
        # ====================================================
        model = _get_model()
        pose_results = model.predict(
            frame,
            conf=CURVE_CHECK_PERSON_CONFIDENCE,
            imgsz=CURVE_CHECK_POSE_IMGSZ,
            device=_MODEL_DEVICE,
            verbose=False,
        )

        detected_persons: List[dict] = []

        if pose_results:
            pose_result = pose_results[0]
            if pose_result.boxes is not None and pose_result.keypoints is not None:
                boxes = pose_result.boxes.xyxy.cpu().numpy()
                classes = pose_result.boxes.cls.cpu().numpy()
                confidences = pose_result.boxes.conf.cpu().numpy()
                keypoints = pose_result.keypoints.data.cpu().numpy()

                for i in range(len(boxes)):
                    if int(classes[i]) != PERSON_CLASS_ID:
                        continue
                    if confidences[i] < CURVE_CHECK_PERSON_CONFIDENCE:
                        continue
                    detected_persons.append({
                        "index": i,
                        "bbox": boxes[i],
                        "kpts": keypoints[i],
                        "confidence": float(confidences[i]),
                    })

        # ====================================================
        # DRIVER SELECTION  (exact standalone select_driver())
        # ====================================================
        driver = _select_driver(detected_persons, frame_width, frame_height, self._driver_roi)

        # ====================================================
        # OUTSIDE DOOR ANALYSIS  (exact standalone loop, including
        # its "last is_outside person wins the reason string" quirk —
        # outside_score uses max(), outside_reason does not)
        # ====================================================
        outside_found = False
        outside_score = 0.0
        outside_reason = ""
        best_bbox: Optional[Tuple[int, int, int, int]] = None

        candidates: List[CurveCheckCandidate] = []

        for person in detected_persons:
            # The selected driver is never classified as POSSIBLE OUTSIDE.
            if driver is not None and np.array_equal(person["bbox"], driver["bbox"]):
                continue

            analysis = _analyze_person(person["kpts"], person["bbox"], self._door_roi)
            if analysis is None:
                continue

            bbox = tuple(map(int, person["bbox"]))
            score = analysis["score"]
            is_outside = analysis["outside_looking"]

            candidates.append(CurveCheckCandidate(
                bbox=bbox, score=score, door_person=analysis["door_person"],
                outside_looking=is_outside, reason=analysis["reason"],
            ))

            if is_outside:
                outside_found = True
                outside_score = max(outside_score, score)
                outside_reason = analysis["reason"]
                best_bbox = bbox

        self.last_candidates = candidates

        # ====================================================
        # OUTSIDE TEMPORAL FILTER  (exact standalone counter logic)
        # ====================================================
        if outside_found:
            self.outside_counter += 1
        else:
            self.outside_counter = max(0, self.outside_counter - 1)

        looking_outside_confirmed = self.outside_counter >= CURVE_CHECK_MIN_CONSECUTIVE_FRAMES

        # ====================================================
        # SAVE LOOKING OUTSIDE  (exact standalone cooldown gate)
        # ====================================================
        log_events: List[Tuple[int, str]] = []
        completed_events: List[Tuple[int, float, float, float]] = []  # standalone has no episode/duration concept

        if looking_outside_confirmed:
            if (frame_number - self.last_outside_event_frame) >= CURVE_CHECK_EVENT_COOLDOWN_FRAMES:
                log_events.append((self._pilot_id, "Curve checking — ALP looking outside door"))
                self.last_outside_event_frame = frame_number

        results = [CurveCheckResult(
            pilot_id        = self._pilot_id,
            outside_looking = looking_outside_confirmed,
            outside_counter = self.outside_counter,
            bbox            = best_bbox,
            score           = outside_score,
            reason          = outside_reason,
        )]

        return results, log_events, completed_events


# ─────────────────────────────────────────────────────────────────────────────
# DRAW HELPERS — exact standalone draw_door_roi()/draw_driver_roi() coloring
# and text, plus per-candidate coloring matching the standalone script's
# outside-analysis drawing loop. Only called when main.py's DRAW flag is
# True, same as every other detector's overlay.
# ─────────────────────────────────────────────────────────────────────────────

def draw_door_roi(frame: np.ndarray, door_roi_px: np.ndarray) -> np.ndarray:
    """Exact port of the standalone script's draw_door_roi()."""
    overlay = frame.copy()
    cv2.fillPoly(overlay, [door_roi_px], (255, 150, 0))
    frame = cv2.addWeighted(overlay, 0.12, frame, 0.88, 0)
    cv2.polylines(frame, [door_roi_px], True, (255, 150, 0), 3)
    cv2.putText(frame, "DOOR / OUTSIDE RANGE", tuple(door_roi_px[0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 150, 0), 2)
    return frame


def draw_driver_roi(frame: np.ndarray, driver_roi_norm: Tuple[float, float, float, float]) -> np.ndarray:
    """Exact port of the standalone script's draw_driver_roi()."""
    h, w = frame.shape[:2]
    x1 = int(driver_roi_norm[0] * w)
    y1 = int(driver_roi_norm[1] * h)
    x2 = int(driver_roi_norm[2] * w)
    y2 = int(driver_roi_norm[3] * h)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
    cv2.putText(frame, "DRIVER SEARCH ROI", (x1 + 5, max(25, y1 + 25)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    return frame


def draw_curve_check_overlay(frame: np.ndarray, result: CurveCheckResult) -> np.ndarray:
    """Draws every evaluated non-driver candidate this cycle, colored the
    same way the standalone script's outside-analysis loop did: orange for
    outside_looking, yellow for door_person-but-not-outside, green for
    everyone else. result.bbox/score label the strongest outside_looking
    hit this cycle (== standalone's outside_score / last-wins reason)."""
    if result.bbox is not None:
        color = (0, 165, 255) if result.outside_looking else (255, 255, 0)
        x1, y1, x2, y2 = result.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if result.outside_looking:
            label = f"POSSIBLE OUTSIDE {result.score:.2f}"
            cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame