# detector/seat_absence_detector.py
# ──────────────────────────────────────────────────────────────────
# Seat Absence Detection
#
# VISUAL LAYOUT
# ──────────────
#   ┌──────────────────────────────────────┐  y = 0
#   │         PILOT 2 ZONE  (upper)        │   ← blue box must stay here
#   ├──────────────────────────── split_y ─┤  y ≈ 57% H   (yellow dashed line)
#   │    LOCO PILOT ZONE / PILOT 1 (lower) │   ← crossing into here = DISTRACTION
#   └──────────────────────────────────────┘  y = H
#
# YELLOW BOX  = fixed seat zone, locked once at startup, never moves
# BLUE BOX    = live person tracking box each frame
#
# RULE
# ─────
# For EACH detected person (blue box):
#   • Find which pilot zone their centre belongs to (above/below split_y)
#   • Their yellow seat zone is the zone they started in
#   • If their blue box centre crosses INTO the loco pilot zone (below split_y)
#     → that pilot is away from their seat → start timer → alert
#
# Box colours NEVER change — yellow stays yellow, blue stays blue.
# Alert = banner + log only.
# ──────────────────────────────────────────────────────────────────

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from config.settings import (
    ABSENCE_ALLOWED_DURATION,
    RELOG_INTERVAL,
)

GREEN_LINE_RATIO = 0.57   # must match gadget_detector.py


# ──────────────────────────────────────────────────────────────────
# RESULT
# ──────────────────────────────────────────────────────────────────

@dataclass
class AbsenceResult:
    pilot_id:      int
    absent:        bool  = False
    timer_value:   float = 0.0
    calibrated:    bool  = True
    seat_zone:     Optional[Tuple[int, int, int, int]] = None  # yellow box
    tracking_bbox: Optional[Tuple[int, int, int, int]] = None  # blue box


# ──────────────────────────────────────────────────────────────────
# ABSENCE TIMER  — wall clock only, no video_time mixing
# ──────────────────────────────────────────────────────────────────

# Minimum consecutive frames pilot must be absent before timer starts.
# Prevents single-frame YOLO miss from triggering absence alert.
ABSENCE_MISS_FRAMES_BEFORE_START = 15   # ~0.5s at 25fps

@dataclass
class _AbsenceTimer:
    pilot_id:    int
    start_time:  Optional[float] = None
    last_logged: Optional[float] = None
    miss_count:  int = 0   # consecutive frames with no detection

    def activate(self):
        """Call every frame the pilot is NOT in their seat zone."""
        self.miss_count += 1
        # Only start the wall-clock timer after enough consecutive
        # missed frames — prevents YOLO jitter from firing alerts.
        if self.miss_count >= ABSENCE_MISS_FRAMES_BEFORE_START:
            if self.start_time is None:
                self.start_time = time.monotonic()

    def reset(self):
        self.start_time  = None
        self.last_logged = None
        self.miss_count  = 0

    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.monotonic() - self.start_time

    def should_log(self) -> bool:
        if self.elapsed() < ABSENCE_ALLOWED_DURATION:
            return False
        if self.last_logged is None:
            return True
        return (time.monotonic() - self.last_logged) >= RELOG_INTERVAL

    def mark_logged(self):
        self.last_logged = time.monotonic()


# ──────────────────────────────────────────────────────────────────
# MAIN DETECTOR
# ──────────────────────────────────────────────────────────────────

class SeatAbsenceDetector:
    """
    Tracks each pilot using their blue bounding box.
    Each pilot has a fixed yellow seat zone (their natural half of frame).

    DISTRACTION RULE:
    ─────────────────
    Pilot 2 (upper zone person) → alert if their blue box centre
    drops BELOW split_y into the loco pilot zone.

    Pilot 1 (lower zone person) → alert if their blue box is
    completely absent from the lower zone (they left the frame).

    This matches exactly what is visible in the camera:
    the horizontal dashed line is the boundary — any person
    crossing below it into the driving zone triggers the alert.
    """

    def __init__(self) -> None:
        self._timers: Dict[int, _AbsenceTimer] = {
            1: _AbsenceTimer(1),
            2: _AbsenceTimer(2),
        }

    # ──────────────────────────────────────────────────────────────
    # PUBLIC — call once per frame
    # ──────────────────────────────────────────────────────────────

    def process(
        self,
        pilot_boxes:  List[Tuple[int, Tuple[int, int, int, int]]],
        video_time:   float,
        frame_width:  int = 848,
        frame_height: int = 480,
    ) -> Tuple[List[AbsenceResult], List[Tuple[int, str]]]:

        split_y = int(frame_height * GREEN_LINE_RATIO)

        # Build bbox lookup by pilot_id
        bbox_by_pid: Dict[int, Optional[Tuple[int,int,int,int]]] = {
            1: None, 2: None
        }
        for pid, bbox in pilot_boxes:
            bbox_by_pid[pid] = bbox

        # Fixed yellow seat zones — upper half for P2, lower half for P1
        seat_zones: Dict[int, Tuple[int,int,int,int]] = {
            2: (0, 0,        frame_width, split_y),       # upper zone
            1: (0, split_y,  frame_width, frame_height),  # lower zone
        }

        results:    List[AbsenceResult]   = []
        log_events: List[Tuple[int, str]] = []

        for pid in [1, 2]:
            timer     = self._timers[pid]
            seat_zone = seat_zones[pid]
            bbox      = bbox_by_pid.get(pid)

            # ── DISTRACTION CHECK ─────────────────────────────────
            # For Pilot 2: distracted when their blue box centre
            #   crosses BELOW split_y (into the loco pilot zone)
            # For Pilot 1: distracted when not detected in lower zone
            in_seat = self._pilot_in_seat(bbox, pid, split_y)

            if in_seat:
                timer.reset()
                results.append(AbsenceResult(
                    pilot_id      = pid,
                    absent        = False,
                    timer_value   = 0.0,
                    calibrated    = True,
                    seat_zone     = seat_zone,
                    tracking_bbox = bbox,
                ))
            else:
                timer.activate()
                elapsed = timer.elapsed()
                absent  = elapsed >= ABSENCE_ALLOWED_DURATION

                if absent and timer.should_log():
                    log_events.append((pid, "Pilot Away From Seat"))
                    timer.mark_logged()

                results.append(AbsenceResult(
                    pilot_id      = pid,
                    absent        = absent,
                    timer_value   = elapsed,
                    calibrated    = True,
                    seat_zone     = seat_zone,
                    tracking_bbox = bbox,
                ))

        return results, log_events

    # ──────────────────────────────────────────────────────────────
    # SEAT CHECK
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _pilot_in_seat(
        bbox:    Optional[Tuple[int,int,int,int]],
        pid:     int,
        split_y: int,
    ) -> bool:
        """
        Pilot 2 (upper zone):
            IN SEAT  → blue box centre is ABOVE split_y
            ABSENT   → blue box centre is BELOW split_y (crossed into
                       loco pilot zone) OR no detection at all

        Pilot 1 (lower zone):
            IN SEAT  → blue box detected anywhere in lower zone
            ABSENT   → no blue box detected in lower zone
        """
        if bbox is None:
            return False

        x1, y1, x2, y2 = bbox

        if pid == 2:
            # Pilot 2 must stay ABOVE the horizontal line
            return y1 < split_y
        else:
            # Pilot 1 must be in the lower zone
            return y2 >= split_y
