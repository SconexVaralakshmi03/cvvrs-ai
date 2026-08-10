# detector/rsl_hand_brake_verifier.py
# ══════════════════════════════════════════════════════════════════════════
#  RSL HAND BRAKE POST-VERIFICATION (Deadman's Brake / Vigilance Brake)
#  ──────────────────────────────────────────────────────────────────────
#  NOT a standalone detector — this module is never run on every frame and
#  owns no per-frame pipeline hook of its own. It is only ever invoked
#  AFTER detector/hand_raise_detector.py has already:
#    1. processed the complete video,
#    2. confirmed a genuine ALP/LP hand-raise episode on the SIGNAL frame, and
#    3. the pipeline has selected/finalized the single representative
#       violation frame for that episode (post dedup + time-window merge
#       in analyzer.py / utils/violation_store.py).
#
#  FLOW (unchanged hand-raise logic; RSL verification is single-frame)
#  ──────────────────────────────────────────────────────────────────────
#      YOLO → Hand Raise Candidate confirmed on Signal Frame N
#           → take ONLY frame N (no neighbouring frames)
#           → send frame N to Qwen-VL
#           → Qwen checks: signal acknowledgement + hand on brake, in
#             that SAME frame only
#           → if confirmed, role is always ALP
#
#  Given that already-selected frame (frame N) and the pilot/side that
#  triggered it, this module re-reads ONLY that single frame (never the
#  whole video, and no ±N neighbour window) and hands it straight to
#  Qwen-VL (llm_verifier.verify_rsl_hand_brake_frame /
#  prompt.build_rsl_hand_brake_prompt), which judges signal acknowledgement
#  and opposite/ALP-hand-on-brake directly from that one image — NOT via a
#  second MediaPipe pass, and NOT across a multi-frame window.
#
#  WHY NOT A SECOND MEDIAPIPE PASS
#  ──────────────────────────────────────────────────────────────────────
#  The RSL Hand Brake is a FIXED control mounted on the console — it is
#  not part of the human body, so pose landmarks alone can't tell "hand
#  resting near torso" apart from "hand actually gripping the lever".
#  Qwen-VL, given the raw frame, can directly see the lever and judge grip
#  + role in one shot.
# ══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from llm_verifier import verify_rsl_hand_brake_frame


# ──────────────────────────────────────────────────────────────────────────
# Frame extraction — reads ONLY the single already-selected signal frame N,
# never a full video scan and never a ±N neighbour window.
# ──────────────────────────────────────────────────────────────────────────

def extract_signal_frame(
    video_path: str,
    local_secs: float,
    fps: float,
) -> Optional[np.ndarray]:
    """
    Read exactly the already-selected hand-raise/signal frame N, using the
    SAME local-time seek strategy already used elsewhere in this project
    (CAP_PROP_POS_MSEC with a CAP_PROP_POS_FRAMES fallback) — see
    ViolationStore.extract_violation_frames / analyzer.py Case 3.

    Returns the frame (np.ndarray), or None if it could not be read.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        t = max(0.0, local_secs)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ret, frame = cap.read()
        if not ret:
            local_frame = int(t * (fps or 25.0))
            local_frame = min(max(0, local_frame), max(0, total - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, local_frame)
            ret, frame = cap.read()
        return frame if ret else None
    finally:
        cap.release()


# ──────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────

def verify_rsl_hand_brake(
    frame: Optional[np.ndarray],
    log_label: str = "",
) -> dict:
    """
    frame: the single already-confirmed hand-raise/signal frame N from
            extract_signal_frame() — no neighbouring frames are involved.
    log_label: free-form label forwarded to the LLM call for logging/audit
            only (e.g. f"{src_file}@{local_secs:.2f}s").

    The frame is sent to Qwen-VL on its own — no window, no per-frame
    MediaPipe re-pass. The model verifies signal acknowledgement + hand-on-
    brake in that single image. If confirmed, role is always "ALP".

    Returns:
        {
          "confirmed":  bool,    # Qwen verified the RSL Hand Brake
                                  # candidate (signal + brake hand) in
                                  # this single frame
          "confidence": float,   # 0.0-1.0
          "role":       "ALP" | None,   # always "ALP" when confirmed,
                                         # else None
          "reason":     str,
          "best_frame": np.ndarray | None,  # the SAME frame passed in,
                                  # returned for compatibility with
                                  # callers that use it as the violation's
                                  # representative frame
        }
    """
    if frame is None:
        return {
            "confirmed":  False,
            "confidence": 0.0,
            "role":       None,
            "reason":     "No readable frame for RSL Hand Brake verification.",
            "best_frame": None,
        }

    verdict = verify_rsl_hand_brake_frame(frame, log_label=log_label)

    return {
        "confirmed":  bool(verdict["verified"]),
        "confidence": round(float(verdict.get("confidence", 0)) / 100.0, 3),
        "role":       verdict.get("role"),
        "reason":     verdict.get("reason", ""),
        "best_frame": frame if verdict["verified"] else None,
    }