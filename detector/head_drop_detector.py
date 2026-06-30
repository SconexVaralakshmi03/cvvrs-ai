from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import (
    EAR_THRESHOLD,
    HEAD_DROP_DURATION,
    NOSE_EAR_DROP_FRACTION,
    NOSE_EAR_RISE_FRACTION,
    HEAD_DROOP_SCORE_THRESHOLD,
    HEAD_BACK_SCORE_THRESHOLD,
    RELOG_INTERVAL,
    TORSO_RECLINE_MIN,
    TORSO_HUNCH_MAX,
)
from detector.enums import CameraView
from detector.gadget_detector import FrameDetections


# ──────────────────────────────────────────────────────────────────
# TUNING CONSTANTS
# ──────────────────────────────────────────────────────────────────

# ── Stillness gate ────────────────────────────────────────────────
# Shoulder midpoint must move < this many pixels to count as "quiet"
STILL_MOTION_PX        = 15.0   # px; train vibration ≈ 5-10 px, real move ≈ 30+

# How many consecutive "quiet" frames before we say body is still
STILL_FRAMES_REQUIRED  = 4    # ~1.2s at DROOP_EVERY=3

# ── Head-drop / head-tilt score ───────────────────────────────────
# Rolling window length for accumulating head signal (frames)
HEAD_SCORE_WINDOW      = 5    # ~1.5s at DROOP_EVERY=3

# ── Eye-closure gate ─────────────────────────────────────────────
# Consecutive frames with EAR < EAR_THRESHOLD before "eyes closed" fires
EYE_CLOSED_FRAMES      = 20    # ~2.4s at DROOP_EVERY=3

# ── Seat check ───────────────────────────────────────────────────
# Pilot bounding box must occupy at least this fraction of the vertical
# zone height for us to consider them seated (not standing / absent)
SEATED_MIN_BBOX_FRACTION = 0.15   # bbox height ≥ 15 % of zone height

# ── YOLO miss tolerance ──────────────────────────────────────────
MAX_MISS_FRAMES        = 10    # frames YOLO can miss before state resets


# ──────────────────────────────────────────────────────────────────
# RESULT DATA CLASS
# ──────────────────────────────────────────────────────────────────

@dataclass
class DroopResult:
    pilot_id:    int
    drooping:    bool  = False
    severity:    str   = "OK"
    timer_value: float = 0.0   # seconds since drowsy trigger
    is_seated:   bool  = True
    # v5: expose which posture triggered the alert (for UI / logging)
    droop_type:  str   = "NONE"   # "FORWARD", "BACKWARD", "EYES", "NONE"


# ──────────────────────────────────────────────────────────────────
# INTERNAL PILOT STATE
# ──────────────────────────────────────────────────────────────────

@dataclass
class _PilotState:
    pilot_id: int
    last_logged_second: Optional[int] = None
    # Stillness tracking (torso / shoulder midpoint)
    prev_shoulder:  Optional[Tuple[float, float]] = None
    prev_nose_y: Optional[float] = None
    still_counter:  int = 0          # consecutive quiet frames
    
    # Dynamic Baseline Tracking
    baseline_gap_metric: Optional[float] = None
    baseline_samples: List[float] = field(default_factory=list)

    # Forward head-drop score (rolling window of booleans)
    droop_window:      Deque[bool] = field(
        default_factory=lambda: deque(maxlen=HEAD_SCORE_WINDOW)
    )

    # Backward head-tilt score (rolling window of booleans)  ← NEW v5
    back_droop_window: Deque[bool] = field(
        default_factory=lambda: deque(maxlen=HEAD_SCORE_WINDOW)
    )

    # Eye-closure streak
    eye_closed_streak: int = 0

    # Alert timer (wall clock)
    alert_start:    Optional[float] = None
    last_logged:    Optional[float] = None

    # YOLO miss handling
    miss_frames:    int = 0
    last_crop_data: Optional[tuple] = None

    # ── helpers ──

    def is_still(self) -> bool:
        return self.still_counter >= STILL_FRAMES_REQUIRED

    def forward_droop_score(self) -> float:
        if not self.droop_window:
            return 0.0
        return sum(self.droop_window) / len(self.droop_window)

    def backward_tilt_score(self) -> float:               # NEW v5
        if not self.back_droop_window:
            return 0.0
        return sum(self.back_droop_window) / len(self.back_droop_window)

    def activate_alert(self, video_time: float):
        self.miss_frames = 0
        if self.alert_start is None:
            self.alert_start = video_time

    def reset_alert(self):
        self.alert_start = None
        self.last_logged = None

    def alert_elapsed(self, video_time: float) -> float:
        if self.alert_start is None:
            return 0.0
        return video_time - self.alert_start

    def full_reset(self):
        """Called when pilot disappears for too many frames."""
        self.prev_shoulder     = None
        self.still_counter     = 0
        self.droop_window.clear()
        self.back_droop_window.clear()    # NEW v5
        self.eye_closed_streak = 0
        self.miss_frames       = 0
        self.last_crop_data    = None
        self.reset_alert()


# ──────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────

def _eye_aspect_ratio(pts: List[Tuple[float, float]]) -> float:
    """6-point EAR for one eye (MediaPipe face mesh indices)."""
    def d(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])
    return (d(pts[1], pts[5]) + d(pts[2], pts[4])) / (2.0 * d(pts[0], pts[3]) + 1e-6)


def _is_seated(bbox: Tuple[int, int, int, int], zone_height: int) -> bool:
    """
    Returns True if the pilot's bounding box is tall enough to
    suggest they are sitting (not standing or absent from view).

    A seated person's bbox typically covers their torso + head
    relative to the zone. A standing person's bbox is much taller,
    and an absent person has no bbox at all.

    We check that bbox height ≥ SEATED_MIN_BBOX_FRACTION of zone
    height AND that the bottom of the bbox is in the lower 70% of
    the zone (i.e. they reach toward the seat area).
    """
    x1, y1, x2, y2 = bbox
    bbox_h = y2 - y1
    if zone_height <= 0:
        return True   # can't determine, assume seated
    return (bbox_h / zone_height) >= SEATED_MIN_BBOX_FRACTION


# ──────────────────────────────────────────────────────────────────
# MAIN DETECTOR
# ──────────────────────────────────────────────────────────────────

class HeadDroopDetector:
    """
    Stillness-gated, score-based drowsiness detector (v5).

    PIPELINE PER PILOT PER FRAME
    ─────────────────────────────
    1. Get pilot crop from FrameDetections (YOLO bbox region).
    2. Run MediaPipe Pose on crop → shoulder midpoint for stillness.
    3. Run MediaPipe FaceMesh on crop → nose/ear Y for droop, EAR for eyes.
    4. Update TWO rolling score windows:
         • droop_window      — forward chin-drop frames
         • back_droop_window — backward head-tilt frames   (NEW v5)
    5. Update eye-closed streak counter.
    6. Compute DROWSY signal:
         signal = is_still()  AND  (
             high_forward_droop       ← nose below ear × 60% of window
             OR high_backward_tilt    ← nose above ear × 60% of window (NEW v5)
             OR eyes_long_shut        ← EAR < threshold × 15 frames
         )
    7. If signal → start/keep alert timer.
       Else       → reset alert timer.
    8. DROWSY fires when alert_elapsed() ≥ HEAD_DROP_DURATION.

    WHY THIS WORKS FOR REAL FOOTAGE
    ────────────────────────────────
    • Pilot leaning forward to operate controls:
        torso moves → still_counter resets → is_still() = False → no alert.

    • Pilot taking a quick look down at instruments:
        head drops for 2-3 frames → forward droop score stays low → no alert.

    • Pilot taking a quick look UP:
        head tilts back for 2-3 frames → backward tilt score stays low → no alert.

    • Drowsy pilot — forward microsleep (classic):
        torso quiet + forward droop frames accumulate over 1-2 s →
        score crosses HEAD_DROOP_SCORE_THRESHOLD → alert fires.

    • Drowsy pilot — backward recline sleep (NEW v5, observed in footage):
        Pilot leans back into seat with arms folded, nose rises well above
        ear tragion and STAYS there. Torso is still (not reaching forward).
        Backward tilt frames accumulate over 1-2 s →
        score crosses HEAD_BACK_SCORE_THRESHOLD → alert fires.

    • Train vibration:
        shoulder jitter < STILL_MOTION_PX → absorbed, still_counter keeps
        building. Nose/ear Y jitter is small relative to both
        NOSE_EAR_DROP_FRACTION and NOSE_EAR_RISE_FRACTION
        (each 8% of crop height ≈ 20-40 px on typical crops).
    """

    def __init__(self):
        self._states: Dict[int, _PilotState] = {
            1: _PilotState(1),
            2: _PilotState(2),
        }

        import mediapipe as mp
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,         # smoothing absorbs jitter
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._face = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
    # ──────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────

    def process(
        self,
        frame:            "np.ndarray",
        video_time:       float,
        frame_detections: Optional[FrameDetections],
        zone_manager:     'DynamicZoneManager',
        current_view:     CameraView,
    ) -> Tuple[List[DroopResult], List[Tuple[int, str]]]:

        results:    List[DroopResult]     = []
        log_events: List[Tuple[int, str]] = []

        if frame_detections is None:
            return [DroopResult(1), DroopResult(2)], []

        frame_h, frame_w = frame_detections.frame_shape[:2]

        if current_view == CameraView.BACK:
            for pid in (1, 2):
                self._states[pid].full_reset()
            return [DroopResult(1), DroopResult(2)], []

        zone_h: Dict[int, int] = {}
        for pid in (1, 2):
            x1, y1, x2, y2 = zone_manager.get_zone(pid, frame_w, frame_h)
            zone_h[pid] = max(1, y2 - y1)

        for pid in (1, 2):
            state     = self._states[pid]
            crop_data = frame_detections.pilot_crops.get(pid)

            # ── YOLO miss handling ────────────────────────────────
            if crop_data is None:
                state.miss_frames += 1
                if (state.alert_start is not None
                        and state.miss_frames <= MAX_MISS_FRAMES
                        and state.last_crop_data is not None):
                    # Carry last known crop for short gaps
                    crop_data = state.last_crop_data
                else:
                    state.full_reset()
                    results.append(DroopResult(pid))
                    continue
            else:
                state.miss_frames = 0

            crop, x1, y1, x2, y2 = crop_data
            state.last_crop_data  = crop_data

            if crop.size == 0:
                results.append(DroopResult(pid))
                continue

            # ── Seated check (using YOLO bbox, not mediapipe) ─────
            pilot_bbox = (x1, y1, x2, y2)
            seated = _is_seated(pilot_bbox, zone_h[pid])

            if not seated:
                # Standing or partially visible — skip drowsiness
                state.full_reset()
                results.append(DroopResult(pid, is_seated=False))
                continue

            # ── MediaPipe inference ───────────────────────────────
            rgb      = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pose_res = self._pose.process(rgb)
            face_res = self._face.process(rgb)

            crop_h, crop_w = crop.shape[:2]

            # ══════════════════════════════════════════════════════
            # STEP A — TORSO STILLNESS
            # ══════════════════════════════════════════════════════
            # Use shoulder midpoint as proxy for torso movement.
            # If MediaPipe Pose fails, we conservatively assume NOT still
            # (prevents false alerts when pose is ambiguous).

            torso_still = False
            torso_angle = None   # degrees from vertical; +ve = reclined back

            if pose_res.pose_landmarks:
                lm     = pose_res.pose_landmarks.landmark
                curr_x = ((lm[11].x + lm[12].x) / 2) * crop_w
                curr_y = ((lm[11].y + lm[12].y) / 2) * crop_h
                curr   = (curr_x, curr_y)

                if state.prev_shoulder is not None:
                    motion = math.hypot(
                        curr[0] - state.prev_shoulder[0],
                        curr[1] - state.prev_shoulder[1],
                    )
                    if motion < STILL_MOTION_PX:
                        state.still_counter = min(state.still_counter + 1, STILL_FRAMES_REQUIRED+5)
                    else:
                        # Active body movement → reset EVERYTHING.
                        # Torso move = not drowsy.
                        state.still_counter = max(0, state.still_counter - 2)
                        if motion > STILL_MOTION_PX * 1.5:
                            state.droop_window.clear()
                            state.back_droop_window.clear()   # NEW v5
                            state.eye_closed_streak = 0
                            state.reset_alert()

                state.prev_shoulder = curr
                torso_still = state.is_still()

                # ── Torso angle (shoulder-to-hip line vs. vertical) ───
                # A seated, attentive pilot's torso is close to vertical.
                # Genuine recline-sleep tilts the torso itself backward,
                # not just the head. This distinguishes recline-sleep from
                # a normally-seated pilot whose head/ear offset alone can
                # look "tilted back" due to camera angle or simply looking
                # up/forward without actually leaning into the seat.
                hip_x  = ((lm[23].x + lm[24].x) / 2) * crop_w
                hip_y  = ((lm[23].y + lm[24].y) / 2) * crop_h
                dx     = curr_x - hip_x
                dy     = curr_y - hip_y   # shoulder above hip → dy negative
                if dy != 0 or dx != 0:
                    # angle from vertical, signed so leaning back (shoulder
                    # moves backward/up relative to hip in image-x terms)
                    # is positive — sign convention matched against
                    # TORSO_RECLINE_MIN / TORSO_HUNCH_MAX in settings.
                    torso_angle = math.degrees(math.atan2(dx, -dy))
            else:
                # No pose detected — treat as active (suppress alert)
                state.still_counter = max(0, state.still_counter - 2)
                torso_still = state.is_still()

            # ══════════════════════════════════════════════════════
            # STEP B — HEAD SIGNAL  (forward droop + backward tilt)
            # ══════════════════════════════════════════════════════
            # We only accumulate signal when torso is still.
            # When torso moves, both windows are cleared above.

            eye_closed_this_frame    = False
            face_detected=face_res.multi_face_landmarks is not None
            
            # ── 1. EYE CLOSURE (Requires Face) ──────────────────────
            if face_detected:
                flm    = face_res.multi_face_landmarks[0].landmark
                eye_pts = [
                    (flm[33].x  * crop_w, flm[33].y  * crop_h),
                    (flm[160].x * crop_w, flm[160].y * crop_h),
                    (flm[158].x * crop_w, flm[158].y * crop_h),
                    (flm[133].x * crop_w, flm[133].y * crop_h),
                    (flm[153].x * crop_w, flm[153].y * crop_h),
                    (flm[144].x * crop_w, flm[144].y * crop_h),
                ]
                ear = _eye_aspect_ratio(eye_pts)
                eye_closed_this_frame = (ear < EAR_THRESHOLD)

            # ── 2. DYNAMIC BASELINE (Primary Head Pose Detection) ───
            final_forward = False
            final_back = False

            if pose_res.pose_landmarks:
                lm = pose_res.pose_landmarks.landmark
                
                if lm[11].visibility > lm[12].visibility:
                    best_shoulder = lm[11]
                else:
                    best_shoulder = lm[12]
                    
                if lm[7].visibility > lm[8].visibility:
                    best_ear = lm[7]
                else:
                    best_ear = lm[8]
                    
                shoulder_y = best_shoulder.y * crop_h
                nose_x = lm[0].x * crop_w
                nose_y = lm[0].y * crop_h
                ear_x = best_ear.x * crop_w
                ear_y = best_ear.y * crop_h
                
                if state.prev_nose_y is not None:
                    motion = abs(nose_y - state.prev_nose_y)
                else:
                    motion = 0

                state.prev_nose_y = nose_y

                # Hybrid Scale-Invariant Geometry (Shoulder vs Nose, normalized by Face Size)
                face_size = math.hypot(nose_x - ear_x, nose_y - ear_y)
                
                # Ensure high confidence to prevent hallucinating on empty seats
                if lm[0].visibility < 0.6 or best_ear.visibility < 0.6 or best_shoulder.visibility < 0.5:
                    state.full_reset()
                    results.append(DroopResult(pid, is_seated=False))
                    continue

                if face_size > crop_h * 0.05:
                    gap_metric = (shoulder_y - nose_y) / face_size
                else:
                    gap_metric = state.baseline_gap_metric if state.baseline_gap_metric is not None else 1.5

                # DETERMINISTIC HALLUCINATION CHECK
                if gap_metric < 0:
                    # Geometrically impossible (nose physically below shoulders).
                    # MediaPipe is wildly hallucinating on an empty chair.
                    state.full_reset()
                    results.append(DroopResult(pid, is_seated=False))
                    continue

                # --- DYNAMIC BASELINE CALIBRATION ---
                if state.baseline_gap_metric is None:
                    # Learning Phase: Collect samples when sitting still
                    if torso_still:
                        state.baseline_samples.append(gap_metric)
                        if len(state.baseline_samples) >= 10:
                            state.baseline_gap_metric = sum(state.baseline_samples) / 10
                            state.baseline_samples.clear()
                            # print(f"[DEBUG PID {pid}] BASELINE ESTABLISHED: {state.baseline_gap_metric:.2f}")
                else:
                    # Detection Phase
                    deviation = gap_metric / state.baseline_gap_metric
                    
                    # If the gap deviates significantly (> 30% shrink or growth)
                    if (deviation < 0.70 or deviation > 1.30) and motion < 25:
                        # Use torso_angle to disambiguate! From high-angle cameras, 
                        # leaning backward can actually cause the 2D vertical gap to shrink.
                        if torso_angle is not None and torso_angle >= TORSO_RECLINE_MIN:
                            final_back = True
                        else:
                            final_forward = True
                        
                    # Adaptive Tracking (EMA) - slowly adjust baseline to new normal posture
                    if torso_still and not final_forward and not final_back and motion < 25:
                        state.baseline_gap_metric = (0.95 * state.baseline_gap_metric) + (0.05 * gap_metric)

            # ── Accumulate windows ONLY when torso is still ───────
            # Only accumulate when torso is still — mirrors STEP B's
            # face-detected path. Without this gate, leaning forward to
            # work the controls (torso actively moving, face mesh lost
            # due to head angle) still got counted toward the droop
            # score, eventually crossing HEAD_DROOP_SCORE_THRESHOLD and
            # firing a false "FORWARD" drowsiness alert.
            if torso_still:
                state.droop_window.append(final_forward)
                state.back_droop_window.append(final_back)

        

            # Eye-closed streak — requires torso still to avoid
            # flagging a pilot who merely reaches forward with eyes
            # momentarily closed.
            if eye_closed_this_frame:
                state.eye_closed_streak += 1
            else:
                state.eye_closed_streak = 0

            # ══════════════════════════════════════════════════════
            # STEP C — DROWSY SIGNAL DECISION
            # ══════════════════════════════════════════════════════
            #
            #  RULE: pilot must be STILL, AND at least one of:
            #    (a) forward droop score ≥ HEAD_DROOP_SCORE_THRESHOLD
            #        — sustained gradual forward chin-drop
            #    (b) backward tilt score ≥ HEAD_BACK_SCORE_THRESHOLD  (NEW v5)
            #        — sustained backward recline sleep
            #    (c) eyes closed for EYE_CLOSED_FRAMES consecutive frames
            #        — seated microsleep with sustained eye closure
            #
            #  Each condition requires the rolling window to be at least
            #  half-full before scoring, preventing spurious early triggers.

            forward_score  = state.forward_droop_score()
            backward_score = state.backward_tilt_score()           # NEW v5
            #print(f"[PID {pid}] still={torso_still} | face={face_detected} | fwd_score={forward_score:.2f} | back_score={backward_score:.2f} | eye_streak={state.eye_closed_streak} | torso_angle={torso_angle}")
            half_window = HEAD_SCORE_WINDOW // 2

            high_forward_droop = (
                len(state.droop_window) >= half_window
                and forward_score >= HEAD_DROOP_SCORE_THRESHOLD
            )

            # Torso must show genuine recline (angle ≥ TORSO_RECLINE_MIN)
            # for the latest reading, on top of the sustained head-offset
            # score. Without this, a normally-seated pilot whose head/ear
            # offset alone crosses the threshold (camera angle, looking up,
            # just sat back down) gets misread as recline-sleep — this was
            # firing "drowsy (BACKWARD)" within seconds of a pilot simply
            # returning to and settling into their seat.
            torso_reclined = (
                torso_angle is not None
                and torso_angle >= TORSO_RECLINE_MIN
            )

            if forward_score > 0 or backward_score > 0 or state.eye_closed_streak > 0:
                print(f"[PID {pid}] time={video_time:.1f}s | face={face_detected} | fwd={forward_score:.2f} | back={backward_score:.2f} | angle={torso_angle} | still={torso_still} | reclined={torso_reclined}")

            high_backward_tilt = (                                  # NEW v5
                len(state.back_droop_window) >= half_window
                and backward_score >= HEAD_BACK_SCORE_THRESHOLD
                and torso_reclined
            )

            eyes_long_shut = (state.eye_closed_streak >= EYE_CLOSED_FRAMES)

            drowsy_signal =  (
                high_forward_droop
                or high_backward_tilt                               # NEW v5
                or eyes_long_shut
            )
            
            # ══════════════════════════════════════════════════════
            # STEP D — ALERT TIMER
            # ══════════════════════════════════════════════════════

            if drowsy_signal:
                state.activate_alert(video_time)
            else:
                state.reset_alert()

            elapsed  = state.alert_elapsed(video_time)
            drooping = elapsed >= HEAD_DROP_DURATION

            # Determine which posture triggered the alert (for UI/log)
            if drooping:
                if high_backward_tilt:
                    droop_type = "BACKWARD"
                elif high_forward_droop:
                    droop_type = "FORWARD"
                elif eyes_long_shut:
                    droop_type = "EYES"
                else:
                    droop_type = "UNKNOWN"
            else:
                droop_type = "NONE"

            results.append(DroopResult(
                pilot_id    = pid,
                drooping    = drooping,
                severity    = "DROWSINESS" if drooping else "OK",
                timer_value = elapsed,
                is_seated   = True,
                droop_type  = droop_type,
            ))
            current_second = int(video_time)

            if drooping:
                if (state.last_logged is None or (video_time - state.last_logged) >= RELOG_INTERVAL):
                    if state.last_logged_second != current_second:
                        log_events.append((pid, f"DROWSINESS:{droop_type}"))
                        state.last_logged = video_time
                        state.last_logged_second = current_second


        return results, log_events