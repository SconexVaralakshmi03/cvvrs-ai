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
#           → take ONLY frame N (no neighbouring frame in TIME)
#           → also cut a zoomed-in crop of the console/lever area from
#             that SAME frame N (spatial crop, not a different moment)
#           → send BOTH images (full frame + zoom crop) to Qwen-VL in
#             ONE call
#           → Qwen checks: signal acknowledgement + hand on brake, using
#             the full frame for context and the zoom crop for a closer
#             look at a small/partially-occluded hand
#           → if confirmed, role is always ALP
#
#  Given that already-selected frame (frame N) and the pilot/side that
#  triggered it, this module re-reads ONLY that single frame (never the
#  whole video, and no ±N neighbour window) and hands it — together with a
#  zoomed spatial crop of the same frame — straight to Qwen-VL
#  (llm_verifier.verify_rsl_hand_brake_frame /
#  prompt.build_rsl_hand_brake_prompt), which judges signal acknowledgement
#  and opposite/ALP-hand-on-brake directly from that one moment — NOT via a
#  second MediaPipe pass, and NOT across a temporal multi-frame window.
#
#  WHY A ZOOMED CROP OF THE SAME FRAME (root-cause fix)
#  ──────────────────────────────────────────────────────────────────────
#  Observed in production logs/reports: a frame that visually DOES show a
#  hand near the RSL lever was still rejected by Qwen ("no hand on or near
#  the RSL Hand Brake console lever") because that hand occupies a small,
#  partially-occluded region of a cabin-wide frame — easy to miss at full-
#  frame resolution, especially after JPEG compression. Cropping the
#  console/lever region out of that SAME frame and upscaling it (config:
#  RSL_BRAKE_ZOOM_* in config/settings.py) gives the model a second, much
#  larger look at exactly that region without reading any additional frame
#  from the video — still strictly single-frame verification.
#
#  WHY NOT A SECOND MEDIAPIPE PASS
#  ──────────────────────────────────────────────────────────────────────
#  The RSL Hand Brake is a FIXED control mounted on the console — it is
#  not part of the human body, so pose landmarks alone can't tell "hand
#  resting near torso" apart from "hand actually gripping the lever".
#  Qwen-VL, given the raw frame (now with an extra zoomed-in look), can
#  directly see the lever and judge grip + role in one shot.
# ══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from llm_verifier import verify_rsl_hand_brake_frame

try:
    from config.settings import (
        RSL_BRAKE_ZOOM_X_START_FRACTION,
        RSL_BRAKE_ZOOM_X_END_FRACTION,
        RSL_BRAKE_ZOOM_Y_START_FRACTION,
        RSL_BRAKE_ZOOM_Y_END_FRACTION,
        RSL_BRAKE_ZOOM_SCALE_FACTOR,
    )
except Exception:
    # Sane fallback so this module never hard-fails if settings.py hasn't
    # been updated yet — mirrors the defensive `_cfg()` pattern already
    # used by llm_verifier.py for the same reason.
    RSL_BRAKE_ZOOM_X_START_FRACTION = 0.10
    RSL_BRAKE_ZOOM_X_END_FRACTION   = 0.85
    RSL_BRAKE_ZOOM_Y_START_FRACTION = 0.35
    RSL_BRAKE_ZOOM_Y_END_FRACTION   = 0.95
    RSL_BRAKE_ZOOM_SCALE_FACTOR     = 2.0


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
# Zoomed console/lever crop — SPATIAL crop of the SAME frame, not a second
# frame in time. Purely a visibility aid for the LLM call below.
# ──────────────────────────────────────────────────────────────────────────

def make_console_zoom_crop(frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Cut a zoomed-in crop of the driving console / RSL lever region out of
    the SAME frame passed in (no other frame is read), and upscale it so a
    small/partially-occluded hand is easier for the model to see.

    Returns None if `frame` is None or the crop region is degenerate.
    """
    if frame is None:
        return None

    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return None

    x0 = max(0, min(w - 1, int(w * RSL_BRAKE_ZOOM_X_START_FRACTION)))
    x1 = max(x0 + 1, min(w, int(w * RSL_BRAKE_ZOOM_X_END_FRACTION)))
    y0 = max(0, min(h - 1, int(h * RSL_BRAKE_ZOOM_Y_START_FRACTION)))
    y1 = max(y0 + 1, min(h, int(h * RSL_BRAKE_ZOOM_Y_END_FRACTION)))

    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    scale = max(1.0, float(RSL_BRAKE_ZOOM_SCALE_FACTOR))
    if scale > 1.0:
        crop = cv2.resize(
            crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)),
            interpolation=cv2.INTER_CUBIC,
        )
    return crop


# ──────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────

def verify_rsl_hand_brake(
    frame: Optional[np.ndarray],
    log_label: str = "",
) -> dict:
    """
    frame: the single already-confirmed hand-raise/signal frame N from
            extract_signal_frame() — no neighbouring frame in time is
            involved.
    log_label: free-form label forwarded to the LLM call for logging/audit
            only (e.g. f"{src_file}@{local_secs:.2f}s").

    The frame is sent to Qwen-VL together with a zoomed-in crop of the
    console/lever region cut from that SAME frame (see
    make_console_zoom_crop) — two spatial views of one moment, no window,
    no per-frame MediaPipe re-pass. The model verifies signal
    acknowledgement + hand-on-brake using both images together. If
    confirmed, role is always "ALP".

    Returns:
        {
          "confirmed":  bool,    # Qwen verified the RSL Hand Brake
                                  # candidate (signal + brake hand) for
                                  # this single frame
          "confidence": float,   # 0.0-1.0
          "role":       "ALP" | None,   # always "ALP" when confirmed,
                                         # else None
          "reason":     str,
          "best_frame": np.ndarray | None,  # the SAME full frame passed
                                  # in, returned for compatibility with
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

    zoom_crop = make_console_zoom_crop(frame)
    verdict = verify_rsl_hand_brake_frame(frame, zoom_crop, log_label=log_label)

    return {
        "confirmed":  bool(verdict["verified"]),
        "confidence": round(float(verdict.get("confidence", 0)) / 100.0, 3),
        "role":       verdict.get("role"),
        "reason":     verdict.get("reason", ""),
        "best_frame": frame if verdict["verified"] else None,
    }