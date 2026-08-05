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
#  FLOW (unchanged)
#  ──────────────────────────────────────────────────────────────────────
#      YOLO → Hand Raise Candidate → Representative Frame N
#           → extract [N-2, N-1, N, N+1, N+2]
#           → send ALL frames to Qwen-VL
#           → Qwen checks: signal acknowledgement, hand on brake,
#             consistency across the window, LP/ALP
#           → Verified
#
#  Given that already-selected frame (frame N) and the pilot/side that
#  triggered it, this module re-reads the small ±2 frame verification
#  window [N-2, N-1, N, N+1, N+2] (never the whole video) and hands the
#  frames straight to Qwen-VL (llm_verifier.verify_rsl_hand_brake_frames /
#  prompt.build_rsl_hand_brake_prompt), which judges signal acknowledgement,
#  opposite/ALP-hand-on-brake, and temporal consistency directly from the
#  images in one call — NOT via a second MediaPipe pass over the window.
#  Qwen also picks the single best/most-confident frame out of the window
#  to use as the evidence frame for the resulting violation record.
#
#  WHY NOT A SECOND MEDIAPIPE PASS
#  ──────────────────────────────────────────────────────────────────────
#  The RSL Hand Brake is a FIXED control mounted on the console — it is
#  not part of the human body, so pose landmarks alone can't tell "hand
#  resting near torso" apart from "hand actually gripping the lever", and
#  a second geometry pass over the window added complexity without
#  actually looking at the console. Qwen-VL, given the raw frames, can
#  directly see the lever and judge grip + consistency + role in one shot.
# ══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from llm_verifier import verify_rsl_hand_brake_frames

try:
    from config.settings import RSL_BRAKE_WINDOW_RADIUS
except Exception:
    # Sane fallback so this module never hard-fails if settings.py hasn't
    # been updated yet — mirrors the defensive `_cfg()` pattern already
    # used by llm_verifier.py for the same reason.
    RSL_BRAKE_WINDOW_RADIUS = 2


# ──────────────────────────────────────────────────────────────────────────
# Frame window extraction — reads ONLY the small verification window, never
# a full video scan. Unchanged from before: this is still the only place
# that touches the video file for RSL verification.
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
# Public entry point
# ──────────────────────────────────────────────────────────────────────────

def verify_rsl_hand_brake(
    frames: List[Optional[np.ndarray]],
    log_label: str = "",
) -> dict:
    """
    frames: the [N-2, N-1, N, N+1, N+2] window from extract_frame_window()
            (or an equivalent window with the selected frame N in the
            middle). Entries may be None where a read failed.
    log_label: free-form label forwarded to the LLM call for logging/audit
            only (e.g. f"{src_file}@{local_secs:.2f}s").

    All non-None frames in the window are sent to Qwen-VL TOGETHER in a
    single call — no per-window MediaPipe re-pass. The model itself
    verifies signal acknowledgement + hand-on-brake + temporal consistency
    + LP/ALP role, and picks the single best evidence frame out of the
    ones it was shown.

    Returns:
        {
          "confirmed":  bool,    # Qwen verified the full RSL Hand Brake
                                  # candidate (signal + brake hand +
                                  # consistency) across the window
          "confidence": float,   # 0.0-1.0
          "role":       "LP" | "ALP" | "BOTH" | "AMBIGUOUS" | None,
          "reason":     str,
          "best_frame": np.ndarray | None,  # the single frame (out of the
                                  # 5) Qwen judged most clearly shows the
                                  # evidence — this is what should be kept
                                  # as the violation's representative frame
        }
    """
    original_center = len(frames) // 2

    valid_indices = [i for i, f in enumerate(frames) if f is not None]
    valid_frames = [frames[i] for i in valid_indices]

    if not valid_frames:
        return {
            "confirmed":  False,
            "confidence": 0.0,
            "role":       None,
            "reason":     "No readable frames in the verification window.",
            "best_frame": None,
        }

    # Map the original center position (index == window radius) into the
    # filtered "valid frames only" list — falls back to the nearest valid
    # frame if the exact center frame couldn't be read from disk.
    if original_center in valid_indices:
        center_in_valid = valid_indices.index(original_center)
    else:
        center_in_valid = min(
            range(len(valid_indices)),
            key=lambda i: abs(valid_indices[i] - original_center),
        )

    verdict = verify_rsl_hand_brake_frames(
        valid_frames, center_in_valid, log_label=log_label,
    )

    best_idx = verdict.get("best_frame_index", center_in_valid)
    if not (0 <= best_idx < len(valid_frames)):
        best_idx = center_in_valid
    best_frame = valid_frames[best_idx]

    return {
        "confirmed":  bool(verdict["verified"]),
        "confidence": round(float(verdict.get("confidence", 0)) / 100.0, 3),
        "role":       verdict.get("role"),
        "reason":     verdict.get("reason", ""),
        "best_frame": best_frame,
    }
