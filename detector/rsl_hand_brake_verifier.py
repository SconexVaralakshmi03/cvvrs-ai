# detector/rsl_hand_brake_verifier.py
# ══════════════════════════════════════════════════════════════════════════
#  RSL HAND BRAKE POST-VERIFICATION (Deadman's Brake / Vigilance Brake)
#  ──────────────────────────────────────────────────────────────────────
#  NOT a standalone detector — this module is never run on every frame and
#  owns no per-frame pipeline hook of its own. It is only ever invoked
#  AFTER detector/hand_raise_detector.py has already:
#    1. processed the complete video,
#    2. confirmed a genuine ALP/LP hand-raise episode, and
#    3. the pipeline has selected/finalized the single representative
#       violation frame for that episode (post dedup + time-window merge
#       in analyzer.py / utils/violation_store.py).
#
#  Given that already-selected frame (frame N) and the pilot/side that
#  triggered it, this module re-derives pose landmarks for ONLY the small
#  ±2 frame verification window [N-2, N-1, N, N+1, N+2] (never the whole
#  video) and decides whether the OPPOSITE hand is consistently resting
#  on / gripping the fixed RSL Hand Brake (Deadman's / Vigilance Brake)
#  console lever throughout that window.
#
#  WHY NOT BODY GEOMETRY ALONE
#  ──────────────────────────────────────────────────────────────────────
#  The RSL Hand Brake is a FIXED control mounted on the console — it is
#  not part of the human body, so it cannot be found the way the raised
#  hand itself is (elbow angle / wrist-above-shoulder). Instead this
#  module checks the opposite wrist's position relative to the pilot's
#  own torso (a proxy for "in front of the console, at console height")
#  and — critically — its TEMPORAL STABILITY across the 5-frame window:
#  a hand that is genuinely gripping a fixed lever barely moves frame to
#  frame, while a hand merely passing through that area on its way
#  elsewhere does not stay put.
#
#  REUSE, NOT MODIFICATION
#  ──────────────────────────────────────────────────────────────────────
#  This module imports (never edits) detector/hand_raise_detector.py's
#  existing pose engine (HandRaisePoseEngine) and per-side classification
#  helper (_classify_side) so:
#    • zero new MediaPipe graphs/config are invented,
#    • the "which side is raised" question is answered with the EXACT
#      same geometry the live hand-raise detector already trusts, and
#    • hand_raise_detector.py itself is never touched.
# ══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from detector.hand_raise_detector import (
    HandRaisePoseEngine,
    _classify_side,
    _lm_xy,
    _lm_vis,
    _LM_NOSE,
    _LM_L_SHOULDER, _LM_R_SHOULDER,
    _LM_L_ELBOW,    _LM_R_ELBOW,
    _LM_L_WRIST,    _LM_R_WRIST,
    _LM_L_HIP,      _LM_R_HIP,
)

try:
    from config.settings import (
        RSL_BRAKE_VIS_THRESHOLD,
        RSL_BRAKE_REGION_Y_MARGIN_FRACTION,
        RSL_BRAKE_REGION_X_FRACTION,
        RSL_BRAKE_MIN_VALID_FRAMES,
        RSL_BRAKE_MIN_PASS_FRAMES,
        RSL_BRAKE_MIN_PASS_RATIO,
        RSL_BRAKE_MAX_POSITION_SPREAD_FRACTION,
        RSL_BRAKE_WINDOW_RADIUS,
    )
except Exception:
    # Sane fallbacks so this module never hard-fails if settings.py hasn't
    # been updated yet — mirrors the defensive `_cfg()` pattern already
    # used by llm_verifier.py for the same reason.
    RSL_BRAKE_VIS_THRESHOLD                = 0.5
    RSL_BRAKE_REGION_Y_MARGIN_FRACTION     = 0.20
    RSL_BRAKE_REGION_X_FRACTION            = 0.55
    RSL_BRAKE_MIN_VALID_FRAMES             = 3
    RSL_BRAKE_MIN_PASS_FRAMES              = 3
    RSL_BRAKE_MIN_PASS_RATIO               = 0.8
    RSL_BRAKE_MAX_POSITION_SPREAD_FRACTION = 0.35
    RSL_BRAKE_WINDOW_RADIUS                = 2


# ──────────────────────────────────────────────────────────────────────────
# Frame window extraction — reads ONLY the small verification window, never
# a full video scan.
# ──────────────────────────────────────────────────────────────────────────

def extract_frame_window(
    video_path: str,
    local_secs: float,
    fps: float,
    window: int = RSL_BRAKE_WINDOW_RADIUS,
) -> List[Optional[np.ndarray]]:
    """
    Read frames [N-window .. N+window] (2*window+1 frames total) around the
    already-selected hand-raise frame, using the SAME local-time seek
    strategy already used elsewhere in this project (CAP_PROP_POS_MSEC with
    a CAP_PROP_POS_FRAMES fallback) — see ViolationStore.extract_violation_frames
    / analyzer.py Case 3.

    Returns a list of length (2*window + 1); an entry is None if that
    particular frame could not be read (e.g. window runs off the start/end
    of the video). The center entry (index == window) is the ORIGINAL
    selected representative frame itself.
    """
    n = 2 * window + 1
    frames: List[Optional[np.ndarray]] = [None] * n

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return frames

    try:
        frame_dur = (1.0 / fps) if fps and fps > 0 else (1.0 / 25.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for i, offset in enumerate(range(-window, window + 1)):
            t = max(0.0, local_secs + offset * frame_dur)
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ret, frame = cap.read()
            if not ret:
                local_frame = int(t * (fps or 25.0))
                local_frame = min(max(0, local_frame), max(0, total - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, local_frame)
                ret, frame = cap.read()
            frames[i] = frame if ret else None
    finally:
        cap.release()

    return frames


# ──────────────────────────────────────────────────────────────────────────
# Re-identify which pilot/side was raised in a given frame's landmarks
# (reuses hand_raise_detector's own _classify_side geometry unchanged).
# ──────────────────────────────────────────────────────────────────────────

def _find_raised_pilot_and_side(
    landmarks_by_pilot: Optional[Dict[int, list]],
    frame_w: int,
    frame_h: int,
) -> Optional[Tuple[int, str, float]]:
    if not landmarks_by_pilot:
        return None

    best: Optional[Tuple[int, str, float]] = None
    for pid, landmarks in landmarks_by_pilot.items():
        if len(landmarks) <= max(_LM_L_HIP, _LM_R_HIP, _LM_NOSE):
            continue

        l_hip, r_hip = landmarks[_LM_L_HIP], landmarks[_LM_R_HIP]
        hip_y   = (l_hip.y + r_hip.y) / 2.0 * frame_h
        nose_xy = _lm_xy(landmarks[_LM_NOSE], frame_w, frame_h)
        torso_h = max(1.0, abs(hip_y - nose_xy[1]))

        left_raised, _, left_conf = _classify_side(
            landmarks, _LM_L_SHOULDER, _LM_L_ELBOW, _LM_L_WRIST, nose_xy, torso_h, frame_w, frame_h)
        right_raised, _, right_conf = _classify_side(
            landmarks, _LM_R_SHOULDER, _LM_R_ELBOW, _LM_R_WRIST, nose_xy, torso_h, frame_w, frame_h)

        for raised, side, conf in ((left_raised, "LEFT", left_conf), (right_raised, "RIGHT", right_conf)):
            if raised and (best is None or conf > best[2]):
                best = (pid, side, conf)

    return best


def _opposite_side(side: str) -> str:
    return "RIGHT" if side == "LEFT" else "LEFT"


def _side_indices(side: str) -> Tuple[int, int, int, int]:
    """Returns (shoulder_idx, elbow_idx, wrist_idx, hip_idx) for one side."""
    if side == "LEFT":
        return _LM_L_SHOULDER, _LM_L_ELBOW, _LM_L_WRIST, _LM_L_HIP
    return _LM_R_SHOULDER, _LM_R_ELBOW, _LM_R_WRIST, _LM_R_HIP


# ──────────────────────────────────────────────────────────────────────────
# Per-frame "is the opposite hand in the fixed brake-lever region" check.
# ──────────────────────────────────────────────────────────────────────────

def _brake_region_check(landmarks: list, side: str, frame_w: int, frame_h: int) -> Optional[dict]:
    sh_idx, el_idx, wr_idx, hip_idx = _side_indices(side)
    needed = (sh_idx, el_idx, wr_idx, _LM_L_HIP, _LM_R_HIP)
    if len(landmarks) <= max(needed):
        return None

    vis = min(_lm_vis(landmarks[sh_idx]), _lm_vis(landmarks[wr_idx]))
    if vis < RSL_BRAKE_VIS_THRESHOLD:
        return None

    sh = _lm_xy(landmarks[sh_idx], frame_w, frame_h)
    wr = _lm_xy(landmarks[wr_idx], frame_w, frame_h)

    l_hip, r_hip = landmarks[_LM_L_HIP], landmarks[_LM_R_HIP]
    hip_y = (l_hip.y + r_hip.y) / 2.0 * frame_h
    hip_x = (l_hip.x + r_hip.x) / 2.0 * frame_w
    torso_h = max(1.0, abs(hip_y - sh[1]))

    # The console/brake lever sits roughly between shoulder height and a
    # little below hip height, in front of the pilot — NOT above the
    # shoulder (that would be the raised hand's own territory) and not
    # far out to the side of the body.
    y_min = sh[1] - RSL_BRAKE_REGION_Y_MARGIN_FRACTION * torso_h
    y_max = hip_y + RSL_BRAKE_REGION_Y_MARGIN_FRACTION * torso_h
    in_y  = y_min <= wr[1] <= y_max

    x_span = RSL_BRAKE_REGION_X_FRACTION * torso_h
    in_x   = abs(wr[0] - hip_x) <= x_span

    return {
        "pass":     bool(in_y and in_x),
        "wrist_xy": wr,
        "vis":      vis,
        "torso_h":  torso_h,
    }


def _verify_window(
    landmarks_seq: List[Optional[Dict[int, list]]],
    pilot_id: int,
    raised_side: str,
    frame_w: int,
    frame_h: int,
) -> Tuple[bool, float]:
    opposite = _opposite_side(raised_side)

    checks = []
    for landmarks_by_pilot in landmarks_seq:
        if not landmarks_by_pilot or pilot_id not in landmarks_by_pilot:
            checks.append(None)
            continue
        checks.append(_brake_region_check(landmarks_by_pilot[pilot_id], opposite, frame_w, frame_h))

    valid = [c for c in checks if c is not None]
    if len(valid) < RSL_BRAKE_MIN_VALID_FRAMES:
        return False, 0.0

    passed = [c for c in valid if c["pass"]]
    pass_ratio = len(passed) / len(valid)
    if len(passed) < RSL_BRAKE_MIN_PASS_FRAMES or pass_ratio < RSL_BRAKE_MIN_PASS_RATIO:
        return False, round(0.5 * pass_ratio, 3)

    # Temporal consistency — a hand truly holding a fixed lever barely
    # moves across the window; a hand merely passing through does.
    xs = [c["wrist_xy"][0] for c in passed]
    ys = [c["wrist_xy"][1] for c in passed]
    avg_torso_h = sum(c["torso_h"] for c in passed) / len(passed)
    spread = ((max(xs) - min(xs)) + (max(ys) - min(ys))) / max(1.0, avg_torso_h)
    consistent = spread <= RSL_BRAKE_MAX_POSITION_SPREAD_FRACTION

    avg_vis = sum(c["vis"] for c in passed) / len(passed)
    confidence = round(min(1.0, 0.5 * pass_ratio + 0.3 * (1.0 if consistent else 0.0) + 0.2 * avg_vis), 3)

    return bool(consistent), confidence


# ──────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────

def verify_rsl_hand_brake(
    frames: List[Optional[np.ndarray]],
    engine: HandRaisePoseEngine,
    frame_w: int,
    frame_h: int,
) -> dict:
    """
    frames: the [N-2, N-1, N, N+1, N+2] window from extract_frame_window()
            (or an equivalent 5-frame window with the selected frame in the
            middle). Entries may be None where a read failed.
    engine: a HandRaisePoseEngine instance — reused across calls by the
            caller so no extra MediaPipe graphs get created per violation.

    Returns:
        {
          "confirmed":  bool,   # Hand Raise == TRUE (by construction) AND
                                 # opposite hand consistently on the brake
          "confidence": float,  # 0.0-1.0 dedicated RSL confidence
          "pilot_id":   int | None,
          "side":       "LEFT" | "RIGHT" | None,   # the RAISED side
          "reason":     str,
        }
    """
    landmarks_seq: List[Optional[Dict[int, list]]] = []
    for f in frames:
        landmarks_seq.append(engine.get_landmarks(f, frame_w, frame_h) if f is not None else None)

    center_idx = len(landmarks_seq) // 2
    raised = _find_raised_pilot_and_side(landmarks_seq[center_idx], frame_w, frame_h)

    # Fall back to scanning the rest of the window if the exact center
    # frame's pose fit was momentarily noisy (mirrors the small hysteresis
    # tolerance already used elsewhere in hand_raise_detector.py).
    if raised is None:
        for lm in landmarks_seq:
            raised = _find_raised_pilot_and_side(lm, frame_w, frame_h)
            if raised:
                break

    if raised is None:
        return {
            "confirmed": False, "confidence": 0.0, "pilot_id": None, "side": None,
            "reason": "Could not re-identify the raised hand/pilot within the verification window.",
        }

    pilot_id, side, _raise_conf = raised
    confirmed, confidence = _verify_window(landmarks_seq, pilot_id, side, frame_w, frame_h)

    return {
        "confirmed":  confirmed,
        "confidence": confidence,
        "pilot_id":   pilot_id,
        "side":       side,
        "reason": (
            "Opposite hand consistently held in the RSL Hand Brake region "
            "across the verification window." if confirmed else
            "Opposite hand was not consistently confirmed on the RSL Hand Brake."
        ),
    }
