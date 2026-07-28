# # detector/hand_raise_detector.py
# # ══════════════════════════════════════════════════════════════════════════
# #  LOCO PILOT / ASSISTANT LOCO PILOT — HAND-RAISE / SIGNALING DETECTOR
# #  ──────────────────────────────────────────────────────────────────────
# #  NEW DETECTOR — added alongside the existing three (gadget / seat-absence
# #  / head-drop). None of those three detectors or their logic are touched
# #  by this file; it is purely additive and wired into main.py the same way
# #  SeatAbsenceDetector is (a lightweight, pure-Python state machine with a
# #  .process(...) / .close() interface, called once per detector cycle).
# #
# #  WHAT IT DETECTS
# #  ──────────────────────────────────────────────────────────────────────
# #  The railway "call out and point / hand-raise acknowledgement" signaling
# #  gesture, in either of its two real-world variants:
# #    (a) OVERHEAD — arm extended fairly straight, wrist well above
# #        shoulder/head level (classic overhead salute).
# #    (b) POINT    — arm extended OUTWARD/FORWARD towards the window,
# #        signal, or gauge; elbow visibly bent (~120-130°) rather than
# #        locked straight, but the wrist still reaches well away from the
# #        shoulder relative to the person's own torso size.
# #  The other arm normally stays at rest; either crew member may perform
# #  either variant, and both may gesture at the same time (tracked
# #  independently per pilot_id, left/right side each tracked separately).
# #
# #  NO EXTRA POSE INFERENCE
# #  ──────────────────────────────────────────────────────────────────────
# #  This detector does NOT run MediaPipe itself. It consumes the SAME
# #  per-pilot MediaPipe Pose landmarks main.py already builds once per
# #  gadget-detector cycle via GadgetDetectionPipeline._get_pose_landmarks()
# #  — a dict {pilot_id: [landmark_0 .. landmark_32]} in full-frame-relative
# #  coordinates (BlazePose-33, same indices used by gadget_detector.py).
# #  Reusing that means zero additional model calls / zero additional GPU or
# #  CPU cost beyond what the pipeline already pays for the gadget detector.
# # ══════════════════════════════════════════════════════════════════════════

# from __future__ import annotations

# import math
# from dataclasses import dataclass
# from typing import Dict, List, Optional, Tuple

# import cv2

# from config.settings import (
#     HAND_RAISE_VIS_THRESHOLD,
#     HAND_RAISE_MARGIN_PX,
#     HAND_RAISE_ELBOW_STRAIGHT_MIN_DEG,
#     HAND_RAISE_EXTENSION_RATIO_MIN,
#     HAND_RAISE_EXTENDED_MIN_ANGLE_DEG,
#     HAND_RAISE_WRIST_ABOVE_NOSE_FRACTION,
#     HAND_RAISE_CONFIRM_FRAMES,
#     HAND_RAISE_ALLOWED_DURATION,
#     HAND_RAISE_MISS_TOLERANCE,
#     HAND_RAISE_RAW_MISS_TOLERANCE,
#     HAND_RAISE_ZONE_SPLIT_RATIO,
#     RELOG_INTERVAL,
# )

# # ── Dedicated MediaPipe import for the hand-raise pose engine ──────────────
# # Kept local to this file (mirrors main.py's own _MP_AVAILABLE guard) so
# # this module stays a self-contained, additive drop-in and never has to
# # import anything from main.py.
# try:
#     import mediapipe as _mp
#     _MP_AVAILABLE = True
# except Exception:
#     _mp = None
#     _MP_AVAILABLE = False

# # ── MediaPipe landmark indices (BlazePose-33) ──────────────────────────────
# # Identical convention/indices to the ones already used in gadget_detector.py.
# _LM_NOSE                       = 0
# _LM_L_SHOULDER, _LM_R_SHOULDER = 11, 12
# _LM_L_ELBOW,    _LM_R_ELBOW    = 13, 14
# _LM_L_WRIST,    _LM_R_WRIST    = 15, 16
# _LM_L_HIP,      _LM_R_HIP      = 23, 24


# # ──────────────────────────────────────────────────────────────────────────
# # DEDICATED POSE ENGINE — hand-raise ONLY (bug fix, fully additive)
# # ──────────────────────────────────────────────────────────────────────────
# # main.py's _get_pose_landmarks() (used by the gadget detector) shares ONE
# # mediapipe.solutions.pose.Pose(static_image_mode=False, ...) instance
# # across TWO different crops (top zone / bottom zone) every cycle.
# # static_image_mode=False turns on MediaPipe's internal temporal tracker,
# # which assumes consecutive .process() calls belong to the SAME subject.
# # Alternating two unrelated crops through that one tracking-mode instance
# # corrupts its internal ROI state on every call, which is the primary
# # source of the noisy/wrong landmarks causing hand-raise false positives
# # in the pipeline (the standalone script never does this — it uses two
# # independent, static_image_mode=True Pose objects, or the stateless
# # Tasks API).
# #
# # This engine fixes that WITHOUT touching main.py's self._pose or
# # _get_pose_landmarks() at all, so GadgetDetector / SeatAbsenceDetector /
# # HeadDroopDetector keep consuming exactly what they always have, byte for
# # byte unchanged. It keeps the SAME zone split ratio and pilot-id
# # convention (pilot 2 = top zone, pilot 1 = bottom zone) as
# # _get_pose_landmarks() so hand-raise results stay aligned with the rest
# # of the pipeline's pilot numbering.

# class _HandRaiseLandmark:
#     """Minimal landmark stand-in — same .x / .y / .visibility surface the
#     rest of this file (and _classify_side) already expects, whether the
#     real landmarks come from main.py's patched objects or here."""
#     __slots__ = ("x", "y", "visibility")

#     def __init__(self, x: float, y: float, visibility: float) -> None:
#         self.x = x
#         self.y = y
#         self.visibility = visibility


# class HandRaisePoseEngine:
#     """Self-contained MediaPipe Pose source used ONLY by HandRaiseDetector.

#     Public surface:
#         engine = HandRaisePoseEngine()
#         landmarks_by_pilot = engine.get_landmarks(frame, frame_w, frame_h)
#         engine.close()

#     Returns the same shape as main.py's _get_pose_landmarks():
#         { pilot_id: [landmark_0 .. landmark_32] }   (or None)
#     """

#     def __init__(self, zone_split_ratio: float = HAND_RAISE_ZONE_SPLIT_RATIO) -> None:
#         self._split_ratio = zone_split_ratio
#         self._pose_top = None
#         self._pose_bottom = None
#         self._available = _MP_AVAILABLE

#         if self._available:
#             try:
#                 mp_pose = _mp.solutions.pose
#                 # static_image_mode=True => every .process() call is an
#                 # independent, from-scratch detection with no cross-call
#                 # tracking state. Alternating between the two zones each
#                 # cycle can no longer corrupt a shared tracker this way.
#                 self._pose_top = mp_pose.Pose(
#                     static_image_mode=True,
#                     model_complexity=1,
#                     enable_segmentation=False,
#                     min_detection_confidence=0.5,
#                 )
#                 self._pose_bottom = mp_pose.Pose(
#                     static_image_mode=True,
#                     model_complexity=1,
#                     enable_segmentation=False,
#                     min_detection_confidence=0.5,
#                 )
#             except Exception:
#                 self._available = False
#                 self._pose_top = None
#                 self._pose_bottom = None

#     def get_landmarks(
#         self, frame, frame_w: int, frame_h: int
#     ) -> Optional[Dict[int, list]]:
#         if not self._available or self._pose_top is None or self._pose_bottom is None:
#             return None
#         try:
#             h, w = frame.shape[:2]
#             split_y = int(h * self._split_ratio)
#             zones = {
#                 2: (self._pose_top,    frame[0:split_y, 0:w], 0),
#                 1: (self._pose_bottom, frame[split_y:h, 0:w], split_y),
#             }

#             result: Dict[int, list] = {}
#             for pid, (pose_obj, crop, y_offset) in zones.items():
#                 if crop.size == 0:
#                     continue
#                 rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
#                 rgb.flags.writeable = False
#                 mp_result = pose_obj.process(rgb)
#                 if mp_result.pose_landmarks is None:
#                     continue

#                 crop_h = crop.shape[0]
#                 patched = []
#                 for lm in mp_result.pose_landmarks.landmark:
#                     full_y = (lm.y * crop_h + y_offset) / h
#                     patched.append(_HandRaiseLandmark(
#                         x=lm.x,
#                         y=full_y,
#                         visibility=getattr(lm, "visibility", 1.0),
#                     ))
#                 result[pid] = patched

#             return result if result else None
#         except Exception:
#             return None

#     def close(self) -> None:
#         try:
#             if self._pose_top is not None:
#                 self._pose_top.close()
#         except Exception:
#             pass
#         try:
#             if self._pose_bottom is not None:
#                 self._pose_bottom.close()
#         except Exception:
#             pass
#         self._pose_top = None
#         self._pose_bottom = None


# # ──────────────────────────────────────────────────────────────────────────
# # RESULT DATA CLASS  (mirrors DroopResult / PilotResult naming conventions
# # used by the other detectors in this project)
# # ──────────────────────────────────────────────────────────────────────────

# @dataclass
# class HandRaiseResult:
#     pilot_id:     int
#     hand_raised:  bool  = False
#     gesture_type: str   = "NONE"   # "OVERHEAD" | "POINT" | "NONE"
#     side:         str   = "NONE"   # "LEFT" | "RIGHT" | "BOTH" | "NONE"
#     confidence:   float = 0.0      # 0-1, landmark-visibility based
#     timer_value:  float = 0.0      # seconds since gesture confirmed (video time)
#     severity:     str   = "OK"     # "SIGNALING" once confirmed_duration is met


# # ──────────────────────────────────────────────────────────────────────────
# # INTERNAL PER-PILOT STATE
# # ──────────────────────────────────────────────────────────────────────────

# @dataclass
# class _PilotState:
#     pilot_id:      int
#     left_streak:   int             = 0
#     right_streak:  int             = 0
#     # BUGFIX — see process(): this must be set on the RAW first-detected
#     # cycle, not on the (later) confirmed cycle, or every reported
#     # duration undercounts by the confirmation delay.
#     raw_start:     Optional[float] = None   # video_time the RAW geometric raise began
#     raw_miss:      int             = 0      # consecutive cycles since raw_start with NO raise on either side (hysteresis)
#     alert_start:   Optional[float] = None   # video_time the CONFIRMED gesture started being logged
#     last_logged:   Optional[float] = None   # video_time it was last logged
#     miss_frames:   int             = 0      # consecutive cycles with no pose landmarks at all
#     confirmed:     bool            = False

#     def elapsed(self, video_time: float) -> float:
#         if self.raw_start is None:
#             return 0.0
#         return max(0.0, video_time - self.raw_start)


# # ──────────────────────────────────────────────────────────────────────────
# # GEOMETRY HELPERS
# # ──────────────────────────────────────────────────────────────────────────

# def _elbow_angle(shoulder, elbow, wrist) -> float:
#     """Angle at the elbow (shoulder-elbow-wrist) in degrees. 180 = dead straight."""
#     v1 = (shoulder[0] - elbow[0], shoulder[1] - elbow[1])
#     v2 = (wrist[0] - elbow[0], wrist[1] - elbow[1])
#     n1, n2 = math.hypot(*v1), math.hypot(*v2)
#     if n1 * n2 == 0:
#         return 0.0
#     cosv = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
#     return math.degrees(math.acos(cosv))


# def _lm_xy(lm, frame_w: int, frame_h: int) -> Tuple[float, float]:
#     return (lm.x * frame_w, lm.y * frame_h)


# def _lm_vis(lm) -> float:
#     return getattr(lm, "visibility", 1.0)


# def _classify_side(landmarks, sh_idx: int, el_idx: int, wr_idx: int,
#                     nose_xy: Tuple[float, float], torso_h: float,
#                     frame_w: int, frame_h: int) -> Tuple[bool, str, float]:
#     """Returns (raised, gesture_type, confidence) for one arm side."""
#     if len(landmarks) <= max(sh_idx, el_idx, wr_idx):
#         return False, "NONE", 0.0

#     vis_sh, vis_el, vis_wr = _lm_vis(landmarks[sh_idx]), _lm_vis(landmarks[el_idx]), _lm_vis(landmarks[wr_idx])
#     conf = min(vis_sh, vis_el, vis_wr)
#     if conf < HAND_RAISE_VIS_THRESHOLD:
#         return False, "NONE", conf

#     sh = _lm_xy(landmarks[sh_idx], frame_w, frame_h)
#     el = _lm_xy(landmarks[el_idx], frame_w, frame_h)
#     wr = _lm_xy(landmarks[wr_idx], frame_w, frame_h)

#     angle = _elbow_angle(sh, el, wr)
#     dist_shoulder_wrist = math.hypot(sh[0] - wr[0], sh[1] - wr[1])
#     extension_ratio = dist_shoulder_wrist / max(1.0, torso_h)

#     wrist_above_shoulder = (sh[1] - wr[1]) >= HAND_RAISE_MARGIN_PX
#     wrist_above_nose = (nose_xy[1] - wr[1]) >= (HAND_RAISE_WRIST_ABOVE_NOSE_FRACTION * torso_h)

#     # PRIMARY SAFETY GATE — wrist must be at least somewhat above shoulder
#     # level. Keeps a resting/bent arm on the panel, a hand propped under
#     # the chin, etc. from ever being flagged, however far it reaches out.
#     cond_straight = angle >= HAND_RAISE_ELBOW_STRAIGHT_MIN_DEG
#     # BUGFIX — the "POINT" branch used extension_ratio ALONE, which is
#     # also satisfied by an ordinary reach to operate a panel switch/gauge
#     # at shoulder-to-head height (this cab's panel sits exactly in that
#     # reach envelope). Requiring the elbow to ALSO be at least
#     # moderately extended (not a tight, tucked-in bend) removes most of
#     # that overlap while still comfortably passing the genuine pointing
#     # gesture measured on reference frames (elbow ~122-129°, ratio
#     # ~0.74-0.88 — both well clear of the raised floors below).
#     cond_extended = (extension_ratio >= HAND_RAISE_EXTENSION_RATIO_MIN
#                       and angle >= HAND_RAISE_EXTENDED_MIN_ANGLE_DEG)
#     cond_overhead = wrist_above_nose

#     raised = wrist_above_shoulder and (cond_straight or cond_extended or cond_overhead)
#     gesture_type = "NONE"
#     if raised:
#         gesture_type = "OVERHEAD" if (cond_straight or cond_overhead) else "POINT"

#     return raised, gesture_type, conf


# # ──────────────────────────────────────────────────────────────────────────
# # DETECTOR
# # ──────────────────────────────────────────────────────────────────────────

# class HandRaiseDetector:
#     """
#     Same call shape as the other detectors in this project:

#         results, log_events, completed_events = hand_raise_detector.process(
#             pose_landmarks_by_pilot, video_time, frame_w, frame_h
#         )

#     - results:          List[HandRaiseResult], one per pilot seen this cycle.
#     - log_events:        List[(pilot_id, "HAND_RAISE:<gesture_type>")] — only
#                           populated the cycle a gesture is FIRST confirmed
#                           (or re-confirmed past RELOG_INTERVAL), mirroring
#                           gadget/absence/droop's log_events convention.
#     - completed_events:  List[(pilot_id, start_video_time, end_video_time,
#                           true_duration, gesture_type)] — mirrors droop's
#                           completed_events shape (pid, start, end, dur, type).
#     """

#     def __init__(self) -> None:
#         self._states: Dict[int, _PilotState] = {}

#     def _state(self, pilot_id: int) -> _PilotState:
#         if pilot_id not in self._states:
#             self._states[pilot_id] = _PilotState(pilot_id=pilot_id)
#         return self._states[pilot_id]

#     def process(
#         self,
#         pose_landmarks_by_pilot: Optional[Dict[int, list]],
#         video_time: float,
#         frame_w: int,
#         frame_h: int,
#     ) -> Tuple[List[HandRaiseResult], List[Tuple[int, str]], List[Tuple[int, float, float, float, str]]]:
#         results: List[HandRaiseResult] = []
#         log_events: List[Tuple[int, str]] = []
#         completed_events: List[Tuple[int, float, float, float, str]] = []

#         if not pose_landmarks_by_pilot:
#             # No landmarks this cycle (MediaPipe unavailable this run, or
#             # no person detected). Age out every tracked pilot's miss
#             # counter and close any in-progress episode past tolerance —
#             # same YOLO/pose-miss handling pattern the other detectors use.
#             for pid, state in list(self._states.items()):
#                 state.miss_frames += 1
#                 if state.miss_frames > HAND_RAISE_MISS_TOLERANCE and state.raw_start is not None:
#                     start_v  = state.raw_start
#                     true_dur = max(0.0, video_time - start_v)
#                     completed_events.append((pid, start_v, video_time, true_dur, "SIGNALING"))
#                     state.raw_start    = None
#                     state.raw_miss     = 0
#                     state.alert_start  = None
#                     state.confirmed    = False
#                     state.left_streak  = 0
#                     state.right_streak = 0
#             return results, log_events, completed_events

#         for pid, landmarks in pose_landmarks_by_pilot.items():
#             state = self._state(pid)
#             state.miss_frames = 0

#             if len(landmarks) <= max(_LM_L_HIP, _LM_R_HIP, _LM_NOSE):
#                 continue

#             l_hip, r_hip = landmarks[_LM_L_HIP], landmarks[_LM_R_HIP]
#             hip_y   = (l_hip.y + r_hip.y) / 2.0 * frame_h
#             nose_xy = _lm_xy(landmarks[_LM_NOSE], frame_w, frame_h)
#             torso_h = max(1.0, abs(hip_y - nose_xy[1]))

#             # Quality gate (added) — torso landmarks (both shoulders + both
#             # hips) must be reasonably confident, or this is more likely a
#             # spurious pose fit on cab clutter than an actual human body.
#             # Mirrors the same gate already present in the standalone
#             # script (hand_raise_salute_detector.py, analyze_frame). A
#             # failing gate is treated as "not raised" for this cycle only
#             # — it flows through the existing streak/miss/episode logic
#             # below exactly like a normal negative reading would, so no
#             # other behavior in this file changes.
#             _torso_idx = (_LM_L_SHOULDER, _LM_R_SHOULDER, _LM_L_HIP, _LM_R_HIP)
#             if len(landmarks) > max(_torso_idx):
#                 _avg_torso_vis = sum(_lm_vis(landmarks[i]) for i in _torso_idx) / len(_torso_idx)
#             else:
#                 _avg_torso_vis = 0.0

#             if _avg_torso_vis < HAND_RAISE_VIS_THRESHOLD:
#                 left_raised,  left_type,  left_conf  = False, "NONE", _avg_torso_vis
#                 right_raised, right_type, right_conf = False, "NONE", _avg_torso_vis
#             else:
#                 left_raised, left_type, left_conf = _classify_side(
#                     landmarks, _LM_L_SHOULDER, _LM_L_ELBOW, _LM_L_WRIST, nose_xy, torso_h, frame_w, frame_h)
#                 right_raised, right_type, right_conf = _classify_side(
#                     landmarks, _LM_R_SHOULDER, _LM_R_ELBOW, _LM_R_WRIST, nose_xy, torso_h, frame_w, frame_h)

#             # Temporal confirmation — kills single-cycle jitter. Used only
#             # to decide WHETHER to ever log the gesture at all.
#             state.left_streak  = state.left_streak  + 1 if left_raised  else 0
#             state.right_streak = state.right_streak + 1 if right_raised else 0
#             left_confirmed  = state.left_streak  >= HAND_RAISE_CONFIRM_FRAMES
#             right_confirmed = state.right_streak >= HAND_RAISE_CONFIRM_FRAMES

#             hand_raised = left_confirmed or right_confirmed
#             both        = left_confirmed and right_confirmed

#             if both:
#                 side, gesture_type = "BOTH", (right_type if right_type != "NONE" else left_type)
#                 confidence = (left_conf + right_conf) / 2.0
#             elif right_confirmed:
#                 side, gesture_type, confidence = "RIGHT", right_type, right_conf
#             elif left_confirmed:
#                 side, gesture_type, confidence = "LEFT", left_type, left_conf
#             else:
#                 side, gesture_type, confidence = "NONE", "NONE", max(left_conf, right_conf)

#             # ── Onset timer (BUGFIX) ────────────────────────────────────────
#             # raw_start marks the FIRST cycle either side geometrically
#             # qualified as raised — BEFORE confirmation — so elapsed/
#             # reported duration reflects the gesture's true onset instead
#             # of the (later) confirmation moment. This is what
#             # close_violation_episode() uses to fill in the PDF/JSON
#             # "Duration" field, so getting this right matters for report
#             # accuracy, not just internal state.
#             #
#             # HAND_RAISE_RAW_MISS_TOLERANCE gives a little hysteresis: one
#             # single cycle where the geometric check drops out mid-gesture
#             # (pose jitter, a brief hand dip between two panel presses,
#             # etc.) does NOT immediately end the episode and force a
#             # separate re-confirmation — it only does after that many
#             # consecutive raw-negative cycles. Without this, one
#             # continuous multi-second reach could get reported as two (or
#             # more) separate near-zero-duration "signaling" events, which
#             # is exactly the pattern seen in earlier runs (e.g. two
#             # events only 13s apart, both ~0-1s "duration").
#             raw_positive = left_raised or right_raised
#             if raw_positive:
#                 state.raw_miss = 0
#                 if state.raw_start is None:
#                     state.raw_start = video_time
#             else:
#                 state.raw_miss += 1
#                 if state.raw_miss > HAND_RAISE_RAW_MISS_TOLERANCE and state.raw_start is not None:
#                     # Genuinely ended — close it out below.
#                     pass

#             episode_active = state.raw_start is not None and state.raw_miss <= HAND_RAISE_RAW_MISS_TOLERANCE

#             if not episode_active and state.raw_start is not None:
#                 start_v  = state.raw_start
#                 true_dur = max(0.0, video_time - start_v)
#                 completed_events.append((pid, start_v, video_time, true_dur, gesture_type))
#                 state.raw_start    = None
#                 state.raw_miss     = 0
#                 state.alert_start  = None
#                 state.confirmed    = False
#                 state.left_streak  = 0
#                 state.right_streak = 0

#             elapsed = state.elapsed(video_time)
#             confirmed_duration = hand_raised and elapsed >= HAND_RAISE_ALLOWED_DURATION

#             if confirmed_duration and state.alert_start is None:
#                 state.alert_start = state.raw_start   # kept for parity/telemetry only

#             results.append(HandRaiseResult(
#                 pilot_id=pid, hand_raised=confirmed_duration, gesture_type=gesture_type,
#                 side=side, confidence=confidence, timer_value=elapsed,
#                 severity="SIGNALING" if confirmed_duration else "OK",
#             ))

#             if confirmed_duration:
#                 if state.last_logged is None or (video_time - state.last_logged) >= RELOG_INTERVAL:
#                     log_events.append((pid, f"HAND_RAISE:{gesture_type}"))
#                     state.last_logged = video_time
#                     state.confirmed   = True

#         return results, log_events, completed_events

#     def close(self) -> None:
#         """No native resources owned directly — this detector reuses the
#         pipeline's shared MediaPipe Pose instance (via the landmarks dict
#         passed into process()), it doesn't create its own. Present only
#         for interface parity with GadgetDetector.close(),
#         HeadDroopDetector.close(), SeatAbsenceDetector.close() so main.py's
#         _release_pipeline_resources() can call it unconditionally."""
#         pass


# detector/hand_raise_detector.py
# ══════════════════════════════════════════════════════════════════════════
#  LOCO PILOT / ASSISTANT LOCO PILOT — HAND-RAISE / SIGNALING DETECTOR
#  ──────────────────────────────────────────────────────────────────────
#  NEW DETECTOR — added alongside the existing three (gadget / seat-absence
#  / head-drop). None of those three detectors or their logic are touched
#  by this file; it is purely additive and wired into main.py the same way
#  SeatAbsenceDetector is (a lightweight, pure-Python state machine with a
#  .process(...) / .close() interface, called once per detector cycle).
#
#  WHAT IT DETECTS
#  ──────────────────────────────────────────────────────────────────────
#  The railway "call out and point / hand-raise acknowledgement" signaling
#  gesture, in either of its two real-world variants:
#    (a) OVERHEAD — arm extended fairly straight, wrist well above
#        shoulder/head level (classic overhead salute).
#    (b) POINT    — arm extended OUTWARD/FORWARD towards the window,
#        signal, or gauge; elbow visibly bent (~120-130°) rather than
#        locked straight, but the wrist still reaches well away from the
#        shoulder relative to the person's own torso size.
#  The other arm normally stays at rest; either crew member may perform
#  either variant, and both may gesture at the same time (tracked
#  independently per pilot_id, left/right side each tracked separately).
#
#  NO EXTRA POSE INFERENCE
#  ──────────────────────────────────────────────────────────────────────
#  This detector does NOT run MediaPipe itself. It consumes the SAME
#  per-pilot MediaPipe Pose landmarks main.py already builds once per
#  gadget-detector cycle via GadgetDetectionPipeline._get_pose_landmarks()
#  — a dict {pilot_id: [landmark_0 .. landmark_32]} in full-frame-relative
#  coordinates (BlazePose-33, same indices used by gadget_detector.py).
#  Reusing that means zero additional model calls / zero additional GPU or
#  CPU cost beyond what the pipeline already pays for the gadget detector.
# ══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2

from config.settings import (
    HAND_RAISE_VIS_THRESHOLD,
    HAND_RAISE_MARGIN_PX,
    HAND_RAISE_MARGIN_FRACTION,
    HAND_RAISE_ELBOW_STRAIGHT_MIN_DEG,
    HAND_RAISE_EXTENSION_RATIO_MIN,
    HAND_RAISE_EXTENDED_MIN_ANGLE_DEG,
    HAND_RAISE_WRIST_ABOVE_NOSE_FRACTION,
    HAND_RAISE_OVERHEAD_MIN_ANGLE_DEG,
    HAND_RAISE_CONFIRM_FRAMES,
    HAND_RAISE_ALLOWED_DURATION,
    HAND_RAISE_MISS_TOLERANCE,
    HAND_RAISE_RAW_MISS_TOLERANCE,
    HAND_RAISE_ZONE_SPLIT_RATIO,
    RELOG_INTERVAL,
)

# ── Dedicated MediaPipe import for the hand-raise pose engine ──────────────
# Kept local to this file (mirrors main.py's own _MP_AVAILABLE guard) so
# this module stays a self-contained, additive drop-in and never has to
# import anything from main.py.
try:
    import mediapipe as _mp
    _MP_AVAILABLE = True
except Exception:
    _mp = None
    _MP_AVAILABLE = False

# ── MediaPipe landmark indices (BlazePose-33) ──────────────────────────────
# Identical convention/indices to the ones already used in gadget_detector.py.
_LM_NOSE                       = 0
_LM_L_SHOULDER, _LM_R_SHOULDER = 11, 12
_LM_L_ELBOW,    _LM_R_ELBOW    = 13, 14
_LM_L_WRIST,    _LM_R_WRIST    = 15, 16
_LM_L_HIP,      _LM_R_HIP      = 23, 24


# ──────────────────────────────────────────────────────────────────────────
# DEDICATED POSE ENGINE — hand-raise ONLY (bug fix, fully additive)
# ──────────────────────────────────────────────────────────────────────────
# main.py's _get_pose_landmarks() (used by the gadget detector) shares ONE
# mediapipe.solutions.pose.Pose(static_image_mode=False, ...) instance
# across TWO different crops (top zone / bottom zone) every cycle.
# static_image_mode=False turns on MediaPipe's internal temporal tracker,
# which assumes consecutive .process() calls belong to the SAME subject.
# Alternating two unrelated crops through that one tracking-mode instance
# corrupts its internal ROI state on every call, which is the primary
# source of the noisy/wrong landmarks causing hand-raise false positives
# in the pipeline (the standalone script never does this — it uses two
# independent, static_image_mode=True Pose objects, or the stateless
# Tasks API).
#
# This engine fixes that WITHOUT touching main.py's self._pose or
# _get_pose_landmarks() at all, so GadgetDetector / SeatAbsenceDetector /
# HeadDroopDetector keep consuming exactly what they always have, byte for
# byte unchanged. It keeps the SAME zone split ratio and pilot-id
# convention (pilot 2 = top zone, pilot 1 = bottom zone) as
# _get_pose_landmarks() so hand-raise results stay aligned with the rest
# of the pipeline's pilot numbering.

class _HandRaiseLandmark:
    """Minimal landmark stand-in — same .x / .y / .visibility surface the
    rest of this file (and _classify_side) already expects, whether the
    real landmarks come from main.py's patched objects or here."""
    __slots__ = ("x", "y", "visibility")

    def __init__(self, x: float, y: float, visibility: float) -> None:
        self.x = x
        self.y = y
        self.visibility = visibility


class HandRaisePoseEngine:
    """Self-contained MediaPipe Pose source used ONLY by HandRaiseDetector.

    Public surface:
        engine = HandRaisePoseEngine()
        landmarks_by_pilot = engine.get_landmarks(frame, frame_w, frame_h)
        engine.close()

    Returns the same shape as main.py's _get_pose_landmarks():
        { pilot_id: [landmark_0 .. landmark_32] }   (or None)
    """

    def __init__(self, zone_split_ratio: float = HAND_RAISE_ZONE_SPLIT_RATIO) -> None:
        self._split_ratio = zone_split_ratio
        self._pose_top = None
        self._pose_bottom = None
        self._available = _MP_AVAILABLE

        if self._available:
            try:
                mp_pose = _mp.solutions.pose
                # static_image_mode=True => every .process() call is an
                # independent, from-scratch detection with no cross-call
                # tracking state. Alternating between the two zones each
                # cycle can no longer corrupt a shared tracker this way.
                self._pose_top = mp_pose.Pose(
                    static_image_mode=True,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                )
                self._pose_bottom = mp_pose.Pose(
                    static_image_mode=True,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                )
            except Exception:
                self._available = False
                self._pose_top = None
                self._pose_bottom = None

    def get_landmarks(
        self, frame, frame_w: int, frame_h: int
    ) -> Optional[Dict[int, list]]:
        if not self._available or self._pose_top is None or self._pose_bottom is None:
            return None
        try:
            h, w = frame.shape[:2]
            split_y = int(h * self._split_ratio)
            zones = {
                2: (self._pose_top,    frame[0:split_y, 0:w], 0),
                1: (self._pose_bottom, frame[split_y:h, 0:w], split_y),
            }

            result: Dict[int, list] = {}
            for pid, (pose_obj, crop, y_offset) in zones.items():
                if crop.size == 0:
                    continue
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                mp_result = pose_obj.process(rgb)
                if mp_result.pose_landmarks is None:
                    continue

                crop_h = crop.shape[0]
                patched = []
                for lm in mp_result.pose_landmarks.landmark:
                    full_y = (lm.y * crop_h + y_offset) / h
                    patched.append(_HandRaiseLandmark(
                        x=lm.x,
                        y=full_y,
                        visibility=getattr(lm, "visibility", 1.0),
                    ))
                result[pid] = patched

            return result if result else None
        except Exception:
            return None

    def close(self) -> None:
        try:
            if self._pose_top is not None:
                self._pose_top.close()
        except Exception:
            pass
        try:
            if self._pose_bottom is not None:
                self._pose_bottom.close()
        except Exception:
            pass
        self._pose_top = None
        self._pose_bottom = None


# ──────────────────────────────────────────────────────────────────────────
# RESULT DATA CLASS  (mirrors DroopResult / PilotResult naming conventions
# used by the other detectors in this project)
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class HandRaiseResult:
    pilot_id:     int
    hand_raised:  bool  = False
    gesture_type: str   = "NONE"   # "OVERHEAD" | "POINT" | "NONE"
    side:         str   = "NONE"   # "LEFT" | "RIGHT" | "BOTH" | "NONE"
    confidence:   float = 0.0      # 0-1, landmark-visibility based
    timer_value:  float = 0.0      # seconds since gesture confirmed (video time)
    severity:     str   = "OK"     # "SIGNALING" once confirmed_duration is met


# ──────────────────────────────────────────────────────────────────────────
# INTERNAL PER-PILOT STATE
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class _PilotState:
    pilot_id:      int
    left_streak:   int             = 0
    right_streak:  int             = 0
    # BUGFIX — see process(): this must be set on the RAW first-detected
    # cycle, not on the (later) confirmed cycle, or every reported
    # duration undercounts by the confirmation delay.
    raw_start:     Optional[float] = None   # video_time the RAW geometric raise began
    raw_miss:      int             = 0      # consecutive cycles since raw_start with NO raise on either side (hysteresis)
    alert_start:   Optional[float] = None   # video_time the CONFIRMED gesture started being logged
    last_logged:   Optional[float] = None   # video_time it was last logged
    miss_frames:   int             = 0      # consecutive cycles with no pose landmarks at all
    confirmed:     bool            = False

    def elapsed(self, video_time: float) -> float:
        if self.raw_start is None:
            return 0.0
        return max(0.0, video_time - self.raw_start)


# ──────────────────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ──────────────────────────────────────────────────────────────────────────

def _elbow_angle(shoulder, elbow, wrist) -> float:
    """Angle at the elbow (shoulder-elbow-wrist) in degrees. 180 = dead straight."""
    v1 = (shoulder[0] - elbow[0], shoulder[1] - elbow[1])
    v2 = (wrist[0] - elbow[0], wrist[1] - elbow[1])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 * n2 == 0:
        return 0.0
    cosv = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cosv))


def _lm_xy(lm, frame_w: int, frame_h: int) -> Tuple[float, float]:
    return (lm.x * frame_w, lm.y * frame_h)


def _lm_vis(lm) -> float:
    return getattr(lm, "visibility", 1.0)


def _classify_side(landmarks, sh_idx: int, el_idx: int, wr_idx: int,
                    nose_xy: Tuple[float, float], torso_h: float,
                    frame_w: int, frame_h: int) -> Tuple[bool, str, float]:
    """Returns (raised, gesture_type, confidence) for one arm side."""
    if len(landmarks) <= max(sh_idx, el_idx, wr_idx):
        return False, "NONE", 0.0

    vis_sh, vis_el, vis_wr = _lm_vis(landmarks[sh_idx]), _lm_vis(landmarks[el_idx]), _lm_vis(landmarks[wr_idx])
    conf = min(vis_sh, vis_el, vis_wr)
    if conf < HAND_RAISE_VIS_THRESHOLD:
        return False, "NONE", conf

    sh = _lm_xy(landmarks[sh_idx], frame_w, frame_h)
    el = _lm_xy(landmarks[el_idx], frame_w, frame_h)
    wr = _lm_xy(landmarks[wr_idx], frame_w, frame_h)

    angle = _elbow_angle(sh, el, wr)
    dist_shoulder_wrist = math.hypot(sh[0] - wr[0], sh[1] - wr[1])
    extension_ratio = dist_shoulder_wrist / max(1.0, torso_h)

    # FIX (real-footage false positives): a flat pixel margin is
    # meaningless without knowing the frame resolution / how close the
    # person is to the camera — on a ~1900px-wide frame, 20px is
    # negligible and this gate was passing almost any upward hand
    # motion. Now requires the larger of the flat pixel floor and a
    # fraction of the person's own torso height, so it scales with the
    # footage instead of silently doing nothing on high-res video.
    required_margin = max(HAND_RAISE_MARGIN_PX, HAND_RAISE_MARGIN_FRACTION * torso_h)
    wrist_above_shoulder = (sh[1] - wr[1]) >= required_margin
    wrist_above_nose = (nose_xy[1] - wr[1]) >= (HAND_RAISE_WRIST_ABOVE_NOSE_FRACTION * torso_h)

    # PRIMARY SAFETY GATE — wrist must be at least somewhat above shoulder
    # level. Keeps a resting/bent arm on the panel, a hand propped under
    # the chin, etc. from ever being flagged, however far it reaches out.
    cond_straight = angle >= HAND_RAISE_ELBOW_STRAIGHT_MIN_DEG
    # BUGFIX — the "POINT" branch used extension_ratio ALONE, which is
    # also satisfied by an ordinary reach to operate a panel switch/gauge
    # at shoulder-to-head height (this cab's panel sits exactly in that
    # reach envelope). Requiring the elbow to ALSO be at least
    # moderately extended (not a tight, tucked-in bend) removes most of
    # that overlap while still comfortably passing the genuine pointing
    # gesture measured on reference frames (elbow ~122-129°, ratio
    # ~0.74-0.88 — both well clear of the raised floors below).
    cond_extended = (extension_ratio >= HAND_RAISE_EXTENSION_RATIO_MIN
                      and angle >= HAND_RAISE_EXTENDED_MIN_ANGLE_DEG)
    # FIX — this branch previously was just `wrist_above_nose`, with NO
    # elbow-angle requirement. Confirmed on two independent real videos
    # (a phone-to-ear frame, and repeated tucked-elbow reaches to upper-
    # console switches) that this let ordinary hand-to-head-height
    # activity pass regardless of elbow bend. Requiring the elbow to also
    # be at least moderately open rejects those tucked-elbow postures
    # while a genuine overhead raise clears this floor easily.
    cond_overhead = wrist_above_nose and (angle >= HAND_RAISE_OVERHEAD_MIN_ANGLE_DEG)

    raised = wrist_above_shoulder and (cond_straight or cond_extended or cond_overhead)
    gesture_type = "NONE"
    if raised:
        gesture_type = "OVERHEAD" if (cond_straight or cond_overhead) else "POINT"

    return raised, gesture_type, conf


# ──────────────────────────────────────────────────────────────────────────
# DETECTOR
# ──────────────────────────────────────────────────────────────────────────

class HandRaiseDetector:
    """
    Same call shape as the other detectors in this project:

        results, log_events, completed_events = hand_raise_detector.process(
            pose_landmarks_by_pilot, video_time, frame_w, frame_h
        )

    - results:          List[HandRaiseResult], one per pilot seen this cycle.
    - log_events:        List[(pilot_id, "HAND_RAISE:<gesture_type>")] — only
                          populated the cycle a gesture is FIRST confirmed
                          (or re-confirmed past RELOG_INTERVAL), mirroring
                          gadget/absence/droop's log_events convention.
    - completed_events:  List[(pilot_id, start_video_time, end_video_time,
                          true_duration, gesture_type)] — mirrors droop's
                          completed_events shape (pid, start, end, dur, type).
    """

    def __init__(self) -> None:
        self._states: Dict[int, _PilotState] = {}

    def _state(self, pilot_id: int) -> _PilotState:
        if pilot_id not in self._states:
            self._states[pilot_id] = _PilotState(pilot_id=pilot_id)
        return self._states[pilot_id]

    def process(
        self,
        pose_landmarks_by_pilot: Optional[Dict[int, list]],
        video_time: float,
        frame_w: int,
        frame_h: int,
    ) -> Tuple[List[HandRaiseResult], List[Tuple[int, str]], List[Tuple[int, float, float, float, str]]]:
        results: List[HandRaiseResult] = []
        log_events: List[Tuple[int, str]] = []
        completed_events: List[Tuple[int, float, float, float, str]] = []

        if not pose_landmarks_by_pilot:
            # No landmarks this cycle (MediaPipe unavailable this run, or
            # no person detected). Age out every tracked pilot's miss
            # counter and close any in-progress episode past tolerance —
            # same YOLO/pose-miss handling pattern the other detectors use.
            for pid, state in list(self._states.items()):
                state.miss_frames += 1
                if state.miss_frames > HAND_RAISE_MISS_TOLERANCE and state.raw_start is not None:
                    start_v  = state.raw_start
                    true_dur = max(0.0, video_time - start_v)
                    completed_events.append((pid, start_v, video_time, true_dur, "SIGNALING"))
                    state.raw_start    = None
                    state.raw_miss     = 0
                    state.alert_start  = None
                    state.confirmed    = False
                    state.left_streak  = 0
                    state.right_streak = 0
            return results, log_events, completed_events

        for pid, landmarks in pose_landmarks_by_pilot.items():
            state = self._state(pid)
            state.miss_frames = 0

            if len(landmarks) <= max(_LM_L_HIP, _LM_R_HIP, _LM_NOSE):
                continue

            l_hip, r_hip = landmarks[_LM_L_HIP], landmarks[_LM_R_HIP]
            hip_y   = (l_hip.y + r_hip.y) / 2.0 * frame_h
            nose_xy = _lm_xy(landmarks[_LM_NOSE], frame_w, frame_h)
            torso_h = max(1.0, abs(hip_y - nose_xy[1]))

            # Quality gate (added) — torso landmarks (both shoulders + both
            # hips) must be reasonably confident, or this is more likely a
            # spurious pose fit on cab clutter than an actual human body.
            # Mirrors the same gate already present in the standalone
            # script (hand_raise_salute_detector.py, analyze_frame). A
            # failing gate is treated as "not raised" for this cycle only
            # — it flows through the existing streak/miss/episode logic
            # below exactly like a normal negative reading would, so no
            # other behavior in this file changes.
            _torso_idx = (_LM_L_SHOULDER, _LM_R_SHOULDER, _LM_L_HIP, _LM_R_HIP)
            if len(landmarks) > max(_torso_idx):
                _avg_torso_vis = sum(_lm_vis(landmarks[i]) for i in _torso_idx) / len(_torso_idx)
            else:
                _avg_torso_vis = 0.0

            if _avg_torso_vis < HAND_RAISE_VIS_THRESHOLD:
                left_raised,  left_type,  left_conf  = False, "NONE", _avg_torso_vis
                right_raised, right_type, right_conf = False, "NONE", _avg_torso_vis
            else:
                left_raised, left_type, left_conf = _classify_side(
                    landmarks, _LM_L_SHOULDER, _LM_L_ELBOW, _LM_L_WRIST, nose_xy, torso_h, frame_w, frame_h)
                right_raised, right_type, right_conf = _classify_side(
                    landmarks, _LM_R_SHOULDER, _LM_R_ELBOW, _LM_R_WRIST, nose_xy, torso_h, frame_w, frame_h)

            # Temporal confirmation — kills single-cycle jitter. Used only
            # to decide WHETHER to ever log the gesture at all.
            state.left_streak  = state.left_streak  + 1 if left_raised  else 0
            state.right_streak = state.right_streak + 1 if right_raised else 0
            left_confirmed  = state.left_streak  >= HAND_RAISE_CONFIRM_FRAMES
            right_confirmed = state.right_streak >= HAND_RAISE_CONFIRM_FRAMES

            hand_raised = left_confirmed or right_confirmed
            both        = left_confirmed and right_confirmed

            if both:
                side, gesture_type = "BOTH", (right_type if right_type != "NONE" else left_type)
                confidence = (left_conf + right_conf) / 2.0
            elif right_confirmed:
                side, gesture_type, confidence = "RIGHT", right_type, right_conf
            elif left_confirmed:
                side, gesture_type, confidence = "LEFT", left_type, left_conf
            else:
                side, gesture_type, confidence = "NONE", "NONE", max(left_conf, right_conf)

            # ── Onset timer (BUGFIX) ────────────────────────────────────────
            # raw_start marks the FIRST cycle either side geometrically
            # qualified as raised — BEFORE confirmation — so elapsed/
            # reported duration reflects the gesture's true onset instead
            # of the (later) confirmation moment. This is what
            # close_violation_episode() uses to fill in the PDF/JSON
            # "Duration" field, so getting this right matters for report
            # accuracy, not just internal state.
            #
            # HAND_RAISE_RAW_MISS_TOLERANCE gives a little hysteresis: one
            # single cycle where the geometric check drops out mid-gesture
            # (pose jitter, a brief hand dip between two panel presses,
            # etc.) does NOT immediately end the episode and force a
            # separate re-confirmation — it only does after that many
            # consecutive raw-negative cycles. Without this, one
            # continuous multi-second reach could get reported as two (or
            # more) separate near-zero-duration "signaling" events, which
            # is exactly the pattern seen in earlier runs (e.g. two
            # events only 13s apart, both ~0-1s "duration").
            raw_positive = left_raised or right_raised
            if raw_positive:
                state.raw_miss = 0
                if state.raw_start is None:
                    state.raw_start = video_time
            else:
                state.raw_miss += 1
                if state.raw_miss > HAND_RAISE_RAW_MISS_TOLERANCE and state.raw_start is not None:
                    # Genuinely ended — close it out below.
                    pass

            episode_active = state.raw_start is not None and state.raw_miss <= HAND_RAISE_RAW_MISS_TOLERANCE

            if not episode_active and state.raw_start is not None:
                start_v  = state.raw_start
                true_dur = max(0.0, video_time - start_v)
                completed_events.append((pid, start_v, video_time, true_dur, gesture_type))
                state.raw_start    = None
                state.raw_miss     = 0
                state.alert_start  = None
                state.confirmed    = False
                state.left_streak  = 0
                state.right_streak = 0

            elapsed = state.elapsed(video_time)
            confirmed_duration = hand_raised and elapsed >= HAND_RAISE_ALLOWED_DURATION

            if confirmed_duration and state.alert_start is None:
                state.alert_start = state.raw_start   # kept for parity/telemetry only

            results.append(HandRaiseResult(
                pilot_id=pid, hand_raised=confirmed_duration, gesture_type=gesture_type,
                side=side, confidence=confidence, timer_value=elapsed,
                severity="SIGNALING" if confirmed_duration else "OK",
            ))

            if confirmed_duration:
                if state.last_logged is None or (video_time - state.last_logged) >= RELOG_INTERVAL:
                    log_events.append((pid, f"HAND_RAISE:{gesture_type}"))
                    state.last_logged = video_time
                    state.confirmed   = True

        return results, log_events, completed_events

    def close(self) -> None:
        """No native resources owned directly — this detector reuses the
        pipeline's shared MediaPipe Pose instance (via the landmarks dict
        passed into process()), it doesn't create its own. Present only
        for interface parity with GadgetDetector.close(),
        HeadDroopDetector.close(), SeatAbsenceDetector.close() so main.py's
        _release_pipeline_resources() can call it unconditionally."""
        pass