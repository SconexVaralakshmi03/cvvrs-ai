# utils/draw.py


from __future__ import annotations
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ── Colour palette (BGR) ───────────────────────────────────────────
CLR_PILOT_1   = (0,   200, 255)   # cyan   — Pilot 1 box, NEVER changes
CLR_PILOT_2   = (255, 165,   0)   # amber  — Pilot 2 box, NEVER changes
CLR_ALERT     = (0,     0, 255)   # red    — distraction OVERLAY only
CLR_SAFE      = (0,   220,   0)   # green
CLR_GADGET    = (0,    50, 255)   # bright red for gadget box
CLR_TEXT_BG   = (20,   20,  20)   # near-black background for text

CLR_ABSENCE   = (200,   0, 200)   # magenta – pilot away from seat
CLR_CALIB     = (180, 180,   0)   # yellow  – calibration in progress
CLR_SEAT_ZONE = (0,  215, 255)    # yellow  — fixed seat zone border
CLR_TRACKING  = (255, 120,   0)   # blue    — person tracking box

# Drowsiness — single colour (SLEEPING removed)
CLR_DROWSY    = (0,  120, 255)    # orange-blue for drowsiness overlay
CLR_DROOP     = CLR_DROWSY
CLR_STANDING  = (150, 150, 150)   # grey — standing-person label


def put_text(
    frame:  np.ndarray,
    text:   str,
    pos:    Tuple[int, int],
    colour: Tuple[int, int, int] = (220, 220, 220),
    scale:  float = 0.52,
    thick:  int   = 1,
) -> None:
    """Draw text with a dark background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    x, y = pos
    cv2.rectangle(frame,
                  (x - 2,      y - th - 3),
                  (x + tw + 2, y + bl + 1),
                  CLR_TEXT_BG, cv2.FILLED)
    cv2.putText(frame, text, (x, y), font, scale, colour, thick, cv2.LINE_AA)


def draw_pilot_box(
    frame:      np.ndarray,
    bbox:       Tuple[int, int, int, int],
    pilot_id:   int,
    distracted: bool,
    gadgets:    List[str],
) -> None:
    
    x1, y1, x2, y2 = bbox
    base_colour = CLR_PILOT_1 if pilot_id == 1 else CLR_PILOT_2
    cv2.rectangle(frame, (x1, y1), (x2, y2), base_colour, 2)

    if distracted:
        cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2),
                      CLR_ALERT, 3)
        put_text(frame, "!! PHONE DETECTED", (x1, y1 - 8),
                 CLR_ALERT, scale=0.55)


def draw_gadget_box(
    frame: np.ndarray,
    bbox:  Tuple[int, int, int, int],
    label: str,
    conf:  float,
) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), CLR_GADGET, 2)
    put_text(frame, f"{label} {conf:.0%}", (x1, y1 - 6), CLR_GADGET, scale=0.45)


def draw_hud(
    frame:       np.ndarray,
    video_time:  float,
    frame_no:    int,
    pilot_count: int,
) -> None:
    hh = int(video_time) // 3600
    mm = (int(video_time) % 3600) // 60
    ss = int(video_time) % 60
    put_text(
        frame,
        f"Time {hh:02d}:{mm:02d}:{ss:02d}  |  Frame {frame_no}  |  Pilots detected: {pilot_count}",
        (10, 22), (200, 200, 200), scale=0.50,
    )


def draw_alert_banner(frame: np.ndarray, pilot_id: int, gadget: str) -> None:
    h, w = frame.shape[:2]
    banner_h = 36
    overlay  = frame.copy()
    cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0, 0, 180), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    msg = "  !! CRITICAL - One of the pilots is using a mobile phone !!"
    put_text(frame, msg, (8, h - 10), (255, 255, 255), scale=0.58, thick=1)



# SEAT ABSENCE DRAWING

def draw_seat_zone(
    frame:     np.ndarray,
    seat_zone: Tuple[int, int, int, int],
    pilot_id:  int,
) -> None:
    x1, y1, x2, y2 = seat_zone
    cv2.rectangle(frame, (x1, y1), (x2, y2), CLR_SEAT_ZONE, 2)


def draw_tracking_box(
    frame:         np.ndarray,
    tracking_bbox: Optional[Tuple[int, int, int, int]],
) -> None:
    if tracking_bbox is None:
        return
    x1, y1, x2, y2 = tracking_bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), CLR_TRACKING, 2)


def draw_absence_overlay(
    frame:      np.ndarray,
    bbox:       Optional[Tuple[int, int, int, int]],
    pilot_id:   int,
    absent:     bool,
    timer_val:  float,
    calibrated: bool,
) -> None:
    if not calibrated:
        put_text(frame, "Calibrating seat zone...", (10, 45),
                 CLR_CALIB, scale=0.48)
        return

    if absent:
        if bbox is not None:
            x1, y1 = bbox[0], bbox[1]
            put_text(frame, f"Away From Seat  {timer_val:.1f}s",
                     (x1, y1 - 8), CLR_ABSENCE, scale=0.52)
        else:
            put_text(frame, f"!! PILOT NOT IN FRAME  [{timer_val:.1f}s] !!",
                     (10, 45), CLR_ABSENCE, scale=0.52)


def draw_absence_banner(
    frame:    np.ndarray,
    pilot_id: int,
    duration: float,
) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 72), (w, h - 36), (150, 0, 150), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    msg = f"  !! CRITICAL - One of the pilots is away from the seat  ({duration:.1f}s) !!"
    put_text(frame, msg, (8, h - 46), (255, 255, 255), scale=0.58, thick=1)



# HEAD DROOP / DROWSINESS DRAWING


def _severity_colour(severity: str) -> tuple:
    """
    Only two valid severities now: "DROWSINESS" or "OK".
    SLEEPING has been removed from the detector output.
    """
    if severity == "DROWSINESS":
        return CLR_DROWSY
    return (0, 210, 0)   # no droop


def draw_droop_keypoints(
    frame:     np.ndarray,
    keypoints: dict,
    pilot_id:  int,
    drooping:  bool,
    angle:     float,
) -> None:
    colour = CLR_DROWSY if drooping else (0, 210, 0)

    if "hip_mid" in keypoints and "shoulder_mid" in keypoints:
        cv2.line(frame, keypoints["hip_mid"], keypoints["shoulder_mid"],
                 colour, 2, cv2.LINE_AA)
        cv2.circle(frame, keypoints["hip_mid"],      4, colour, -1)
        cv2.circle(frame, keypoints["shoulder_mid"], 4, colour, -1)

    if "nose" in keypoints and "ear" in keypoints:
        cv2.line(frame, keypoints["ear"], keypoints["nose"], colour, 2, cv2.LINE_AA)
        cv2.circle(frame, keypoints["nose"], 5, colour, -1)
        cv2.circle(frame, keypoints["ear"],  4, colour, -1)

    if "face_ear" in keypoints and "face_nose" in keypoints:
        cv2.line(frame, keypoints["face_ear"], keypoints["face_nose"],
                 colour, 2, cv2.LINE_AA)
        cv2.circle(frame, keypoints["face_nose"], 5, colour, -1)
        cv2.circle(frame, keypoints["face_ear"],  4, colour, -1)
    if "face_chin" in keypoints and "face_nose" in keypoints:
        cv2.line(frame, keypoints["face_chin"], keypoints["face_nose"],
                 colour, 1, cv2.LINE_AA)

    if "shoulder_mid" in keypoints:
        put_text(frame, f"T:{angle:.0f}deg",
                 (keypoints["shoulder_mid"][0]+6, keypoints["shoulder_mid"][1]),
                 colour, scale=0.38)


def draw_droop_overlay(
    frame:     np.ndarray,
    pilot_id:  int,
    drooping:  bool,
    timer_val: float,
    bbox:      Optional[Tuple[int, int, int, int]],
    severity:  str = "DROWSINESS",
) -> None:
    """Draw the drowsiness label and border on the pilot bbox."""
    if not drooping:
        return

    colour = _severity_colour(severity)
    label  = f"DROWSINESS  {timer_val:.1f}s"   # always "DROWSINESS", no ???

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
        put_text(frame, label, (x1, y1 - 8), colour, scale=0.56)
    else:
        put_text(frame, f"!! {label} !!", (10, 68), colour, scale=0.55)


def draw_standing_label(
    frame:    np.ndarray,
    pilot_id: int,
    bbox:     Optional[Tuple[int, int, int, int]],
) -> None:
    if bbox is None:
        return
    x1, y1, x2, y2 = bbox
    put_text(frame, "[STANDING — droop N/A]",
             (x1, y2 + 18), CLR_STANDING, scale=0.42)


def draw_droop_banner(
    frame:    np.ndarray,
    pilot_id: int,
    duration: float,
    severity: str = "DROWSINESS",
) -> None:
   
    h, w  = frame.shape[:2]
    bgr   = (0, 80, 200)          # uniform dark-blue for DROWSINESS
    ov    = frame.copy()
    cv2.rectangle(ov, (0, h-108), (w, h-72), bgr, cv2.FILLED)
    cv2.addWeighted(ov, 0.70, frame, 0.30, 0, frame)

    # Clean message — no "???" possible
    msg = f"  !! CRITICAL — One of the pilots is drowsy  ({duration:.1f}s) !!"
    put_text(frame, msg, (8, h-82), (255, 255, 255), scale=0.58, thick=1)
