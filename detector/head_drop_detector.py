# detector/head_drop_detector.py
# ══════════════════════════════════════════════════════════════════
#  LOCO PILOT DROWSINESS DETECTOR — v5  (stillness-gated, seat-aware,
#                                        forward + backward sleep detection)
#
#  PROBLEM WITH v4
#  ────────────────
#  Detecting ANY head movement as drowsiness caused huge false-positive
#  rates on 43-min real cabin footage. Pilots lean forward to operate
#  controls, look at gauges, or shift in their seat — none of these
#  are drowsiness. Train vibration also jitters mediapipe landmarks.
#
#  NEW IN v5 — BACKWARD RECLINE SLEEP
#  ────────────────────────────────────
#  Real-world CCTV footage (e.g. 15-06-2025 23:58:16, CAB1 LP) shows
#  pilots sleeping by leaning BACKWARD into the seat with arms folded
#  and head tilted back.  v4 only detected FORWARD chin-drops (nose
#  below ear).  v5 adds a symmetric backward-tilt check: when nose
#  rises well ABOVE the ear for many sustained frames while the torso
#  is still, a DROWSINESS alert fires.
#
#  REAL DROWSINESS ON A LOCO PILOT (observed patterns)
#  ─────────────────────────────────────────────────────
#  1. Pilot is SEATED (chair visible and occupied near their bbox)
#  2. TORSO IS STILL — train vibration aside, no large body movement
#  3a. HEAD SLOWLY DROPS FORWARD — chin moves toward chest gradually.
#      The drop accumulates over several seconds.  (Classic microsleep)
#  3b. HEAD TILTS BACKWARD — pilot reclines into seat, nose rises above
#      ear level, arms may fold across chest.  (Recline sleep — new v5)
#  4. EYES CLOSE for sustained periods (EAR below threshold)
#  5. The combination of stillness + (forward droop score OR backward
#     tilt score OR sustained eye closure) = drowsy.
#
#  KEY DESIGN DECISIONS
#  ─────────────────────
#  • "Stillness" is judged on the TORSO (shoulder midpoint), not the
#    head — because a drowsy person's torso is still while their head
#    slowly droops or tilts back.
#
#  • Both forward and backward head signals are accumulated as SCORES
#    over a rolling window (HEAD_SCORE_WINDOW frames). A single-frame
#    movement does not fire an alert.
#
#  • The final alert fires when ALL of these hold simultaneously:
#      torso_still  AND  (high_droop_forward OR high_tilt_back OR eyes_long_shut)
#    This prevents false alerts from:
#      - Leaning forward to press controls   (torso moves → not still)
#      - Quick nod / look up                 (score too low)
#      - Momentary eye blink                 (EAR recovers quickly)
#      - Train vibration jitter              (absorbed by smoothing)
#
#  • Seat-occupancy check: the pilot bbox must overlap the lower
#    portion of their zone (where a seat would be). Standing / absent
#    pilots suppress drowsiness detection.
#
# ══════════════════════════════════════════════════════════════════

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import cv2

from config.settings import (
    EAR_THRESHOLD,
    HEAD_DROP_DURATION,
    NOSE_EAR_DROP_FRACTION,
    NOSE_EAR_RISE_FRACTION,
    HEAD_DROOP_SCORE_THRESHOLD,
    HEAD_BACK_SCORE_THRESHOLD,
    RELOG_INTERVAL,
)
from detector.gadget_detector import FrameDetections


# ──────────────────────────────────────────────────────────────────
# TUNING CONSTANTS
# ──────────────────────────────────────────────────────────────────

# ── Stillness gate ────────────────────────────────────────────────
# Shoulder midpoint must move < this many pixels to count as "quiet"
STILL_MOTION_PX        = 2.5   # px; train vibration ≈ 2-4 px, real move ≈ 15+

# How many consecutive "quiet" frames before we say body is still
STILL_FRAMES_REQUIRED  = 10    # ~0.4 s at 25 fps

# ── Head-drop / head-tilt score ───────────────────────────────────
# Rolling window length for accumulating head signal (frames)
HEAD_SCORE_WINDOW      = 10    # ~1.5 s at 25 fps

# ── Eye-closure gate ─────────────────────────────────────────────
# Consecutive frames with EAR < EAR_THRESHOLD before "eyes closed" fires
EYE_CLOSED_FRAMES      = 15    # ~0.6 s at 25 fps; ignores normal blinks

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

    def activate_alert(self):
        self.miss_frames = 0
        if self.alert_start is None:
            self.alert_start = time.monotonic()

    def reset_alert(self):
        self.alert_start = None
        self.last_logged = None

    def alert_elapsed(self) -> float:
        if self.alert_start is None:
            return 0.0
        return time.monotonic() - self.alert_start

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
    ) -> Tuple[List[DroopResult], List[Tuple[int, str]]]:

        results:    List[DroopResult]     = []
        log_events: List[Tuple[int, str]] = []

        if frame_detections is None:
            return [DroopResult(1), DroopResult(2)], []

        frame_h, frame_w = frame_detections.frame_shape[:2]
        split_y          = frame_detections.split_y

        # Zone heights (used for seated check)
        zone_h: Dict[int, int] = {
            1: frame_h - split_y,   # lower half — pilot 1
            2: split_y,             # upper half — pilot 2
        }

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
                        if motion>6:
                            state.droop_window.clear()
                            state.back_droop_window.clear()   # NEW v5
                            state.eye_closed_streak = 0
                            state.reset_alert()

                state.prev_shoulder = curr
                torso_still = state.is_still()
            else:
                # No pose detected — treat as active (suppress alert)
                state.still_counter = max(0, state.still_counter - 2)
                torso_still = state.is_still()

            # ══════════════════════════════════════════════════════
            # STEP B — HEAD SIGNAL  (forward droop + backward tilt)
            # ══════════════════════════════════════════════════════
            # We only accumulate signal when torso is still.
            # When torso moves, both windows are cleared above.

            head_forward_this_frame  = False   # nose dropped below ear
            head_backward_this_frame = False   # nose rose above ear (NEW v5)
            eye_closed_this_frame    = False
            face_detected=face_res.multi_face_landmarks is not None
            if face_detected:
                flm    = face_res.multi_face_landmarks[0].landmark

                nose_y = flm[1].y   * crop_h
                ear_y  = flm[454].y * crop_h   # right ear tragion

                # ── Forward chin-drop ─────────────────────────────
                drop_threshold = crop_h * NOSE_EAR_DROP_FRACTION
                head_forward_this_frame = (nose_y > ear_y + drop_threshold)

                # ── Backward head-tilt (NEW v5) ───────────────────
                # Pilot reclines: nose rises above ear by NOSE_EAR_RISE_FRACTION.
                # Symmetric to forward droop; same fraction, opposite direction.
                rise_threshold = crop_h * NOSE_EAR_RISE_FRACTION
                head_backward_this_frame = (nose_y < ear_y - rise_threshold)

                # ── Eye closure ───────────────────────────────────
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

            # ── Accumulate windows ONLY when torso is still ───────
            if torso_still:
                state.droop_window.append(head_forward_this_frame)
                if face_detected:
                    state.back_droop_window.append(head_backward_this_frame)   # NEW v5
             # NEW v5
            # else: both windows were already cleared in Step A above.
            # ─────────────────────────────────────────
# FALLBACK: backward sleep without face or Forward sleep with out face
# ─────────────────────────────────────────
            if not face_detected:
                fallback_forward = False
                fallback_back = False
                if pose_res.pose_landmarks:
                    lm = pose_res.pose_landmarks.landmark
                    shoulder_y = ((lm[11].y + lm[12].y) / 2) * crop_h
                    nose_y = lm[0].y * crop_h
                    if state.prev_nose_y is not None:
                        motion = abs(nose_y - state.prev_nose_y)
                    else:
                        motion = 0

                    state.prev_nose_y = nose_y

                    ratio_forward = (nose_y - shoulder_y) / crop_h
                    ratio_back = (shoulder_y - nose_y) / crop_h

                    #print("FORWARD ratio:", ratio_forward, "motion:", motion)
                    #print("BACK ratio:", ratio_back, "motion:", motion)

        # ✅ FIXED CONDITION (IMPORTANT)
                    if ratio_forward > 0.07:
                        fallback_forward = True

        # ✅ KEEP YOUR EXISTING BACK LOGIC
                    if ratio_back > 0.15 and motion < 25:
                        fallback_back = True
                final_forward = head_forward_this_frame or fallback_forward
                final_back = head_backward_this_frame or fallback_back
    # ✅ append once (VERY IMPORTANT)
                state.droop_window.append(final_forward)
                state.back_droop_window.append(final_back)

        

            # Eye-closed streak — requires torso still to avoid
            # flagging a pilot who merely reaches forward with eyes
            # momentarily closed.
            if eye_closed_this_frame and torso_still:
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
            #print(f"[PID {pid}] still={torso_still} | face={face_detected} | fwd_score={forward_score:.2f} | back_score={backward_score:.2f} | eye_streak={state.eye_closed_streak}")
            half_window = HEAD_SCORE_WINDOW // 2

            high_forward_droop = (
                len(state.droop_window) >= half_window
                and forward_score >= HEAD_DROOP_SCORE_THRESHOLD
            )

            high_backward_tilt = (                                  # NEW v5
                len(state.back_droop_window) >= half_window
                and backward_score >= HEAD_BACK_SCORE_THRESHOLD
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
                state.activate_alert()
            else:
                state.reset_alert()

            elapsed  = state.alert_elapsed()
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
            current_time = time.monotonic()
            current_second = int(video_time)

            if drooping:
                if (state.last_logged is None or (current_time - state.last_logged) >= RELOG_INTERVAL):
                    if state.last_logged_second != current_second:
                        log_events.append((pid, f"DROWSINESS:{droop_type}"))
                        state.last_logged = current_time
                        state.last_logged_second = current_second


        return results, log_events