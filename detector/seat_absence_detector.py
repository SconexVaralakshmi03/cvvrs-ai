from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from config.settings import (
    ABSENCE_ALLOWED_DURATION,
    RELOG_INTERVAL,
)

GREEN_LINE_RATIO = 0.57


# ──────────────────────────────────────────────────────────────────
# RESULT
# ──────────────────────────────────────────────────────────────────

@dataclass
class AbsenceResult:
    pilot_id:      int
    absent:        bool  = False
    timer_value:   float = 0.0
    calibrated:    bool  = True
    seat_zone:     Optional[Tuple[int, int, int, int]] = None
    tracking_bbox: Optional[Tuple[int, int, int, int]] = None


# ──────────────────────────────────────────────────────────────────
# ABSENCE TIMER  — uses video_time, not wall clock
# ──────────────────────────────────────────────────────────────────

# FIX (Bug #4): The old value of 15 frames was counted in detector-call frames,
# not raw frames. With RAW_FRAME_SKIP=3 and ABSENCE_EVERY=4, the detector is
# called once every 12 raw frames (~0.48s at 25fps). So 15 detector-frames was
# actually ~7.2s of real video delay. Replaced with a time-based threshold of
# 0.5s using video_time so it behaves as originally intended regardless of
# frame-skip settings.
ABSENCE_GRACE_SECONDS = 0.5   # pilot must be absent this long before timer starts

@dataclass
class _AbsenceTimer:
    pilot_id:     int
    start_vtime:  Optional[float] = None   # video_time when absence started
    last_logged:  Optional[float] = None   # video_time of last log
    # FIX (Bug #4): replaced frame counter with video-time-based grace period.
    grace_start:  Optional[float] = None   # video_time when first missed

    def activate(self, video_time: float):
        """Call every detector frame the pilot is NOT in their seat zone."""
        # Start grace period on first miss
        if self.grace_start is None:
            self.grace_start = video_time
        # Only start the absence timer after the grace period expires
        grace_elapsed = video_time - self.grace_start
        if grace_elapsed >= ABSENCE_GRACE_SECONDS:
            if self.start_vtime is None:
                self.start_vtime = video_time

    def reset(self):
        self.start_vtime = None
        self.last_logged = None
        self.grace_start = None

    def elapsed(self, video_time: float) -> float:
        if self.start_vtime is None:
            return 0.0
        return max(0.0, video_time - self.start_vtime)

    def should_log(self, video_time: float) -> bool:
        if self.elapsed(video_time) < ABSENCE_ALLOWED_DURATION:
            return False
        if self.last_logged is None:
            return True
        return (video_time - self.last_logged) >= RELOG_INTERVAL

    def mark_logged(self, video_time: float):
        self.last_logged = video_time


# ──────────────────────────────────────────────────────────────────
# MAIN DETECTOR
# ──────────────────────────────────────────────────────────────────

class SeatAbsenceDetector:
    """
    Tracks each pilot using their bounding box.
    Each pilot has a fixed yellow seat zone (their natural half of frame).

    DISTRACTION RULE:
    ─────────────────
    Pilot 2 (upper zone person) → alert if their box centre
    drops BELOW split_y into the loco pilot zone.

    Pilot 1 (lower zone person) → alert if their box centre
    is no longer in the lower zone (consistent with Pilot 2 logic).

    FIXES APPLIED:
    ──────────────
    FIX (Bug #3 — seat check): Pilot 1 check now uses bbox CENTRE y
    instead of y2. The old y2 check meant a pilot who stood up but
    whose feet/legs were still partially in frame (bbox bottom still
    touching the lower zone) was incorrectly marked "in seat".

    FIX (Bug #4 — miss threshold): Replaced frame-count-based grace
    period (15 frames, effectively ~7.2s) with a video-time-based
    grace period (ABSENCE_GRACE_SECONDS = 0.5s) so behaviour is
    correct regardless of frame-skip settings.

    FIX (Video time): All timers use video_time instead of
    time.monotonic() so a 43-min video processed faster than real
    time does not cause timers to fire incorrectly.
    """

    def __init__(self) -> None:
        self._timers: Dict[int, _AbsenceTimer] = {
            1: _AbsenceTimer(1),
            2: _AbsenceTimer(2),
        }

    # ──────────────────────────────────────────────────────────────
    # PUBLIC — call once per detector frame
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
        bbox_by_pid: Dict[int, Optional[Tuple[int, int, int, int]]] = {
            1: None, 2: None
        }
        for pid, bbox in pilot_boxes:
            bbox_by_pid[pid] = bbox

        # Fixed yellow seat zones — upper half for P2, lower half for P1
        seat_zones: Dict[int, Tuple[int, int, int, int]] = {
            2: (0, 0,        frame_width, split_y),
            1: (0, split_y,  frame_width, frame_height),
        }

        results:    List[AbsenceResult]   = []
        log_events: List[Tuple[int, str]] = []

        for pid in [1, 2]:
            timer     = self._timers[pid]
            seat_zone = seat_zones[pid]
            bbox      = bbox_by_pid.get(pid)

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
                timer.activate(video_time)
                elapsed = timer.elapsed(video_time)
                absent  = elapsed >= ABSENCE_ALLOWED_DURATION

                if absent and timer.should_log(video_time):
                    log_events.append((pid, "Pilot Away From Seat"))
                    timer.mark_logged(video_time)

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
        bbox:    Optional[Tuple[int, int, int, int]],
        pid:     int,
        split_y: int,
    ) -> bool:
        """
        Pilot 2 (upper zone):
            IN SEAT  → box centre is ABOVE split_y
            ABSENT   → box centre is BELOW split_y OR no detection

        Pilot 1 (lower zone):
            IN SEAT  → box CENTRE is in the lower zone (>= split_y)
            ABSENT   → box centre is above split_y OR no detection

        FIX (Bug #3): Previously Pilot 1 used `y2 >= split_y` which
        allowed a standing pilot whose feet were still visible to be
        counted as seated. Now uses the bbox centre, consistent with
        the Pilot 2 logic.
        """
        if bbox is None:
            return False

        x1, y1, x2, y2 = bbox
        cy = (y1 + y2) / 2  # use centre for both pilots — consistent & robust

        if pid == 2:
            return cy < split_y
        else:
            # FIX: was `y2 >= split_y`; now uses centre
            return cy >= split_y