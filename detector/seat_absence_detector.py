# # from __future__ import annotations

# # import time
# # from dataclasses import dataclass
# # from typing import Dict, List, Optional, Tuple

# # from config.settings import (
# #     ABSENCE_ALLOWED_DURATION,
# #     RELOG_INTERVAL,
# # )

# # GREEN_LINE_RATIO = 0.57


# # # ──────────────────────────────────────────────────────────────────
# # # RESULT
# # # ──────────────────────────────────────────────────────────────────

# # @dataclass
# # class AbsenceResult:
# #     pilot_id:      int
# #     absent:        bool  = False
# #     timer_value:   float = 0.0
# #     calibrated:    bool  = True
# #     seat_zone:     Optional[Tuple[int, int, int, int]] = None
# #     tracking_bbox: Optional[Tuple[int, int, int, int]] = None


# # # ──────────────────────────────────────────────────────────────────
# # # ABSENCE TIMER  — uses video_time, not wall clock
# # # ──────────────────────────────────────────────────────────────────

# # # FIX (Bug #4): The old value of 15 frames was counted in detector-call frames,
# # # not raw frames. With RAW_FRAME_SKIP=3 and ABSENCE_EVERY=4, the detector is
# # # called once every 12 raw frames (~0.48s at 25fps). So 15 detector-frames was
# # # actually ~7.2s of real video delay. Replaced with a time-based threshold of
# # # 0.5s using video_time so it behaves as originally intended regardless of
# # # frame-skip settings.
# # ABSENCE_GRACE_SECONDS = 0.5   # pilot must be absent this long before timer starts

# # @dataclass
# # class _AbsenceTimer:
# #     pilot_id:     int
# #     start_vtime:  Optional[float] = None   # video_time when absence started
# #     last_logged:  Optional[float] = None   # video_time of last log
# #     # FIX (Bug #4): replaced frame counter with video-time-based grace period.
# #     grace_start:  Optional[float] = None   # video_time when first missed

# #     def activate(self, video_time: float):
# #         """Call every detector frame the pilot is NOT in their seat zone."""
# #         # Start grace period on first miss
# #         if self.grace_start is None:
# #             self.grace_start = video_time
# #         # Only start the absence timer after the grace period expires
# #         grace_elapsed = video_time - self.grace_start
# #         if grace_elapsed >= ABSENCE_GRACE_SECONDS:
# #             if self.start_vtime is None:
# #                 self.start_vtime = video_time

# #     def reset(self):
# #         self.start_vtime = None
# #         self.last_logged = None
# #         self.grace_start = None

# #     def elapsed(self, video_time: float) -> float:
# #         if self.start_vtime is None:
# #             return 0.0
# #         return max(0.0, video_time - self.start_vtime)

# #     def should_log(self, video_time: float) -> bool:
# #         if self.elapsed(video_time) < ABSENCE_ALLOWED_DURATION:
# #             return False
# #         if self.last_logged is None:
# #             return True
# #         return (video_time - self.last_logged) >= RELOG_INTERVAL

# #     def mark_logged(self, video_time: float):
# #         self.last_logged = video_time


# # # ──────────────────────────────────────────────────────────────────
# # # MAIN DETECTOR
# # # ──────────────────────────────────────────────────────────────────

# # class SeatAbsenceDetector:
# #     """
# #     Tracks each pilot using their bounding box.
# #     Each pilot has a fixed yellow seat zone (their natural half of frame).

# #     DISTRACTION RULE:
# #     ─────────────────
# #     Pilot 2 (upper zone person) → alert if their box centre
# #     drops BELOW split_y into the loco pilot zone.

# #     Pilot 1 (lower zone person) → alert if their box centre
# #     is no longer in the lower zone (consistent with Pilot 2 logic).

# #     FIXES APPLIED:
# #     ──────────────
# #     FIX (Bug #3 — seat check): Pilot 1 check now uses bbox CENTRE y
# #     instead of y2. The old y2 check meant a pilot who stood up but
# #     whose feet/legs were still partially in frame (bbox bottom still
# #     touching the lower zone) was incorrectly marked "in seat".

# #     FIX (Bug #4 — miss threshold): Replaced frame-count-based grace
# #     period (15 frames, effectively ~7.2s) with a video-time-based
# #     grace period (ABSENCE_GRACE_SECONDS = 0.5s) so behaviour is
# #     correct regardless of frame-skip settings.

# #     FIX (Video time): All timers use video_time instead of
# #     time.monotonic() so a 43-min video processed faster than real
# #     time does not cause timers to fire incorrectly.
# #     """

# #     def __init__(self) -> None:
# #         self._timers: Dict[int, _AbsenceTimer] = {
# #             1: _AbsenceTimer(1),
# #             2: _AbsenceTimer(2),
# #         }

# #     # ──────────────────────────────────────────────────────────────
# #     # PUBLIC — call once per detector frame
# #     # ──────────────────────────────────────────────────────────────

# #     def process(
# #         self,
# #         pilot_boxes:  List[Tuple[int, Tuple[int, int, int, int]]],
# #         video_time:   float,
# #         frame_width:  int = 848,
# #         frame_height: int = 480,
# #     ) -> Tuple[List[AbsenceResult], List[Tuple[int, str]]]:

# #         split_y = int(frame_height * GREEN_LINE_RATIO)

# #         # Build bbox lookup by pilot_id
# #         bbox_by_pid: Dict[int, Optional[Tuple[int, int, int, int]]] = {
# #             1: None, 2: None
# #         }
# #         for pid, bbox in pilot_boxes:
# #             bbox_by_pid[pid] = bbox

# #         # Fixed yellow seat zones — upper half for P2, lower half for P1
# #         seat_zones: Dict[int, Tuple[int, int, int, int]] = {
# #             2: (0, 0,        frame_width, split_y),
# #             1: (0, split_y,  frame_width, frame_height),
# #         }

# #         results:    List[AbsenceResult]   = []
# #         log_events: List[Tuple[int, str]] = []

# #         for pid in [1, 2]:
# #             timer     = self._timers[pid]
# #             seat_zone = seat_zones[pid]
# #             bbox      = bbox_by_pid.get(pid)

# #             in_seat = self._pilot_in_seat(bbox, pid, split_y)

# #             if in_seat:
# #                 timer.reset()
# #                 results.append(AbsenceResult(
# #                     pilot_id      = pid,
# #                     absent        = False,
# #                     timer_value   = 0.0,
# #                     calibrated    = True,
# #                     seat_zone     = seat_zone,
# #                     tracking_bbox = bbox,
# #                 ))
# #             else:
# #                 timer.activate(video_time)
# #                 elapsed = timer.elapsed(video_time)
# #                 absent  = elapsed >= ABSENCE_ALLOWED_DURATION

# #                 if absent and timer.should_log(video_time):
# #                     log_events.append((pid, "Pilot Away From Seat"))
# #                     timer.mark_logged(video_time)

# #                 results.append(AbsenceResult(
# #                     pilot_id      = pid,
# #                     absent        = absent,
# #                     timer_value   = elapsed,
# #                     calibrated    = True,
# #                     seat_zone     = seat_zone,
# #                     tracking_bbox = bbox,
# #                 ))

# #         return results, log_events

# #     # ──────────────────────────────────────────────────────────────
# #     # SEAT CHECK
# #     # ──────────────────────────────────────────────────────────────

# #     @staticmethod
# #     def _pilot_in_seat(
# #         bbox:    Optional[Tuple[int, int, int, int]],
# #         pid:     int,
# #         split_y: int,
# #     ) -> bool:
# #         """
# #         Pilot 2 (upper zone):
# #             IN SEAT  → box centre is ABOVE split_y
# #             ABSENT   → box centre is BELOW split_y OR no detection

# #         Pilot 1 (lower zone):
# #             IN SEAT  → box CENTRE is in the lower zone (>= split_y)
# #             ABSENT   → box centre is above split_y OR no detection

# #         FIX (Bug #3): Previously Pilot 1 used `y2 >= split_y` which
# #         allowed a standing pilot whose feet were still visible to be
# #         counted as seated. Now uses the bbox centre, consistent with
# #         the Pilot 2 logic.
# #         """
# #         if bbox is None:
# #             return False

# #         x1, y1, x2, y2 = bbox
# #         cy = (y1 + y2) / 2  # use centre for both pilots — consistent & robust

# #         if pid == 2:
# #             return cy < split_y
# #         else:
# #             # FIX: was `y2 >= split_y`; now uses centre
# #             return cy >= split_y



# from __future__ import annotations

# import time
# from dataclasses import dataclass
# from typing import Dict, List, Optional, Tuple

# from config.settings import (
#     ABSENCE_ALLOWED_DURATION,
#     RELOG_INTERVAL,
# )

# GREEN_LINE_RATIO = 0.57


# # ──────────────────────────────────────────────────────────────────
# # RESULT
# # ──────────────────────────────────────────────────────────────────

# @dataclass
# class AbsenceResult:
#     pilot_id:      int
#     absent:        bool  = False
#     timer_value:   float = 0.0
#     calibrated:    bool  = True
#     seat_zone:     Optional[Tuple[int, int, int, int]] = None
#     tracking_bbox: Optional[Tuple[int, int, int, int]] = None


# # ──────────────────────────────────────────────────────────────────
# # ABSENCE TIMER  — uses video_time, not wall clock
# # ──────────────────────────────────────────────────────────────────

# # FIX (Bug #4): The old value of 15 frames was counted in detector-call frames,
# # not raw frames. With RAW_FRAME_SKIP=3 and ABSENCE_EVERY=4, the detector is
# # called once every 12 raw frames (~0.48s at 25fps). So 15 detector-frames was
# # actually ~7.2s of real video delay. Replaced with a time-based threshold of
# # 0.5s using video_time so it behaves as originally intended regardless of
# # frame-skip settings.
# ABSENCE_GRACE_SECONDS = 0.5   # pilot must be absent this long before timer starts

# @dataclass
# class _AbsenceTimer:
#     pilot_id:     int
#     start_vtime:  Optional[float] = None   # video_time when absence started
#     last_logged:  Optional[float] = None   # video_time of last log
#     # FIX (Bug #4): replaced frame counter with video-time-based grace period.
#     grace_start:  Optional[float] = None   # video_time when first missed

#     def activate(self, video_time: float):
#         """Call every detector frame the pilot is NOT in their seat zone."""
#         # Start grace period on first miss
#         if self.grace_start is None:
#             self.grace_start = video_time
#         # Only start the absence timer after the grace period expires
#         grace_elapsed = video_time - self.grace_start
#         if grace_elapsed >= ABSENCE_GRACE_SECONDS:
#             if self.start_vtime is None:
#                 self.start_vtime = video_time

#     def reset(self):
#         self.start_vtime = None
#         self.last_logged = None
#         self.grace_start = None

#     def elapsed(self, video_time: float) -> float:
#         if self.start_vtime is None:
#             return 0.0
#         return max(0.0, video_time - self.start_vtime)

#     def should_log(self, video_time: float) -> bool:
#         if self.elapsed(video_time) < ABSENCE_ALLOWED_DURATION:
#             return False
#         if self.last_logged is None:
#             return True
#         return (video_time - self.last_logged) >= RELOG_INTERVAL

#     def mark_logged(self, video_time: float):
#         self.last_logged = video_time


# # ──────────────────────────────────────────────────────────────────
# # MAIN DETECTOR
# # ──────────────────────────────────────────────────────────────────

# class SeatAbsenceDetector:
#     """
#     Tracks each pilot using their bounding box.
#     Each pilot has a fixed yellow seat zone (their natural half of frame).

#     DISTRACTION RULE:
#     ─────────────────
#     Pilot 2 (upper zone person) → alert if their box centre
#     drops BELOW split_y into the loco pilot zone.

#     Pilot 1 (lower zone person) → alert if their box centre
#     is no longer in the lower zone (consistent with Pilot 2 logic).

#     FIXES APPLIED:
#     ──────────────
#     FIX (Bug #3 — seat check): Pilot 1 check now uses bbox CENTRE y
#     instead of y2. The old y2 check meant a pilot who stood up but
#     whose feet/legs were still partially in frame (bbox bottom still
#     touching the lower zone) was incorrectly marked "in seat".

#     FIX (Bug #4 — miss threshold): Replaced frame-count-based grace
#     period (15 frames, effectively ~7.2s) with a video-time-based
#     grace period (ABSENCE_GRACE_SECONDS = 0.5s) so behaviour is
#     correct regardless of frame-skip settings.

#     FIX (Video time): All timers use video_time instead of
#     time.monotonic() so a 43-min video processed faster than real
#     time does not cause timers to fire incorrectly.
#     """

#     def __init__(self) -> None:
#         self._timers: Dict[int, _AbsenceTimer] = {
#             1: _AbsenceTimer(1),
#             2: _AbsenceTimer(2),
#         }

#     # ──────────────────────────────────────────────────────────────
#     # RESOURCE CLEANUP
#     # ──────────────────────────────────────────────────────────────

#     def close(self) -> None:
#         """
#         Release any resources owned by this SeatAbsenceDetector instance.

#         SeatAbsenceDetector is a pure-Python state machine with no native
#         MediaPipe graphs or file handles — all state is Python dicts and
#         dataclasses. This method exists so main.py's finally block can call
#         self.absence_detector.close() symmetrically alongside the other
#         detector cleanup calls without raising AttributeError. Safe to call
#         multiple times.
#         """
#         # Reset timer state so no references to old video-time values linger.
#         for timer in self._timers.values():
#             timer.reset()

#     # ──────────────────────────────────────────────────────────────
#     # PUBLIC — call once per detector frame
#     # ──────────────────────────────────────────────────────────────

#     def process(
#         self,
#         pilot_boxes:  List[Tuple[int, Tuple[int, int, int, int]]],
#         video_time:   float,
#         frame_width:  int = 848,
#         frame_height: int = 480,
#     ) -> Tuple[List[AbsenceResult], List[Tuple[int, str]]]:

#         split_y = int(frame_height * GREEN_LINE_RATIO)

#         # Build bbox lookup by pilot_id
#         bbox_by_pid: Dict[int, Optional[Tuple[int, int, int, int]]] = {
#             1: None, 2: None
#         }
#         for pid, bbox in pilot_boxes:
#             bbox_by_pid[pid] = bbox

#         # Fixed yellow seat zones — upper half for P2, lower half for P1
#         seat_zones: Dict[int, Tuple[int, int, int, int]] = {
#             2: (0, 0,        frame_width, split_y),
#             1: (0, split_y,  frame_width, frame_height),
#         }

#         results:    List[AbsenceResult]   = []
#         log_events: List[Tuple[int, str]] = []

#         for pid in [1, 2]:
#             timer     = self._timers[pid]
#             seat_zone = seat_zones[pid]
#             bbox      = bbox_by_pid.get(pid)

#             in_seat = self._pilot_in_seat(bbox, pid, split_y)

#             if in_seat:
#                 timer.reset()
#                 results.append(AbsenceResult(
#                     pilot_id      = pid,
#                     absent        = False,
#                     timer_value   = 0.0,
#                     calibrated    = True,
#                     seat_zone     = seat_zone,
#                     tracking_bbox = bbox,
#                 ))
#             else:
#                 timer.activate(video_time)
#                 elapsed = timer.elapsed(video_time)
#                 absent  = elapsed >= ABSENCE_ALLOWED_DURATION

#                 if absent and timer.should_log(video_time):
#                     log_events.append((pid, "Pilot Away From Seat"))
#                     timer.mark_logged(video_time)

#                 results.append(AbsenceResult(
#                     pilot_id      = pid,
#                     absent        = absent,
#                     timer_value   = elapsed,
#                     calibrated    = True,
#                     seat_zone     = seat_zone,
#                     tracking_bbox = bbox,
#                 ))

#         return results, log_events

#     # ──────────────────────────────────────────────────────────────
#     # SEAT CHECK
#     # ──────────────────────────────────────────────────────────────

#     @staticmethod
#     def _pilot_in_seat(
#         bbox:    Optional[Tuple[int, int, int, int]],
#         pid:     int,
#         split_y: int,
#     ) -> bool:
#         """
#         Pilot 2 (upper zone):
#             IN SEAT  → box centre is ABOVE split_y
#             ABSENT   → box centre is BELOW split_y OR no detection

#         Pilot 1 (lower zone):
#             IN SEAT  → box CENTRE is in the lower zone (>= split_y)
#             ABSENT   → box centre is above split_y OR no detection

#         FIX (Bug #3): Previously Pilot 1 used `y2 >= split_y` which
#         allowed a standing pilot whose feet were still visible to be
#         counted as seated. Now uses the bbox centre, consistent with
#         the Pilot 2 logic.
#         """
#         if bbox is None:
#             return False

#         x1, y1, x2, y2 = bbox
#         cy = (y1 + y2) / 2  # use centre for both pilots — consistent & robust

#         if pid == 2:
#             return cy < split_y
#         else:
#             # FIX: was `y2 >= split_y`; now uses centre
#             return cy >= split_y


# from __future__ import annotations

# import time
# from dataclasses import dataclass
# from typing import Dict, List, Optional, Tuple

# from config.settings import (
#     ABSENCE_ALLOWED_DURATION,
#     RELOG_INTERVAL,
# )

# GREEN_LINE_RATIO = 0.57


# # ──────────────────────────────────────────────────────────────────
# # RESULT
# # ──────────────────────────────────────────────────────────────────

# @dataclass
# class AbsenceResult:
#     pilot_id:      int
#     absent:        bool  = False
#     timer_value:   float = 0.0
#     calibrated:    bool  = True
#     seat_zone:     Optional[Tuple[int, int, int, int]] = None
#     tracking_bbox: Optional[Tuple[int, int, int, int]] = None


# # ──────────────────────────────────────────────────────────────────
# # ABSENCE TIMER  — uses video_time, not wall clock
# # ──────────────────────────────────────────────────────────────────

# # FIX (Bug #4): The old value of 15 frames was counted in detector-call frames,
# # not raw frames. With RAW_FRAME_SKIP=3 and ABSENCE_EVERY=4, the detector is
# # called once every 12 raw frames (~0.48s at 25fps). So 15 detector-frames was
# # actually ~7.2s of real video delay. Replaced with a time-based threshold of
# # 0.5s using video_time so it behaves as originally intended regardless of
# # frame-skip settings.
# ABSENCE_GRACE_SECONDS = 0.5   # pilot must be absent this long before timer starts

# @dataclass
# class _AbsenceTimer:
#     pilot_id:     int
#     start_vtime:  Optional[float] = None   # video_time when absence started
#     last_logged:  Optional[float] = None   # video_time of last log
#     # FIX (Bug #4): replaced frame counter with video-time-based grace period.
#     grace_start:  Optional[float] = None   # video_time when first missed

#     def activate(self, video_time: float):
#         """Call every detector frame the pilot is NOT in their seat zone."""
#         # Start grace period on first miss
#         if self.grace_start is None:
#             self.grace_start = video_time
#         # Only start the absence timer after the grace period expires
#         grace_elapsed = video_time - self.grace_start
#         if grace_elapsed >= ABSENCE_GRACE_SECONDS:
#             if self.start_vtime is None:
#                 self.start_vtime = video_time

#     def reset(self):
#         self.start_vtime = None
#         self.last_logged = None
#         self.grace_start = None

#     def elapsed(self, video_time: float) -> float:
#         if self.start_vtime is None:
#             return 0.0
#         return max(0.0, video_time - self.start_vtime)

#     def should_log(self, video_time: float) -> bool:
#         if self.elapsed(video_time) < ABSENCE_ALLOWED_DURATION:
#             return False
#         if self.last_logged is None:
#             return True
#         return (video_time - self.last_logged) >= RELOG_INTERVAL

#     def mark_logged(self, video_time: float):
#         self.last_logged = video_time


# # ──────────────────────────────────────────────────────────────────
# # MAIN DETECTOR
# # ──────────────────────────────────────────────────────────────────

# class SeatAbsenceDetector:
#     """
#     Tracks each pilot using their bounding box.
#     Each pilot has a fixed yellow seat zone (their natural half of frame).

#     DISTRACTION RULE:
#     ─────────────────
#     Pilot 2 (upper zone person) → alert if their box centre
#     drops BELOW split_y into the loco pilot zone.

#     Pilot 1 (lower zone person) → alert if their box centre
#     is no longer in the lower zone (consistent with Pilot 2 logic).

#     FIXES APPLIED:
#     ──────────────
#     FIX (Bug #3 — seat check): Pilot 1 check now uses bbox CENTRE y
#     instead of y2. The old y2 check meant a pilot who stood up but
#     whose feet/legs were still partially in frame (bbox bottom still
#     touching the lower zone) was incorrectly marked "in seat".

#     FIX (Bug #4 — miss threshold): Replaced frame-count-based grace
#     period (15 frames, effectively ~7.2s) with a video-time-based
#     grace period (ABSENCE_GRACE_SECONDS = 0.5s) so behaviour is
#     correct regardless of frame-skip settings.

#     FIX (Video time): All timers use video_time instead of
#     time.monotonic() so a 43-min video processed faster than real
#     time does not cause timers to fire incorrectly.
#     """

#     def __init__(self) -> None:
#         self._timers: Dict[int, _AbsenceTimer] = {
#             1: _AbsenceTimer(1),
#             2: _AbsenceTimer(2),
#         }

#     # ──────────────────────────────────────────────────────────────
#     # PUBLIC — call once per detector frame
#     # ──────────────────────────────────────────────────────────────

#     def process(
#         self,
#         pilot_boxes:  List[Tuple[int, Tuple[int, int, int, int]]],
#         video_time:   float,
#         frame_width:  int = 848,
#         frame_height: int = 480,
#     ) -> Tuple[List[AbsenceResult], List[Tuple[int, str]]]:

#         split_y = int(frame_height * GREEN_LINE_RATIO)

#         # Build bbox lookup by pilot_id
#         bbox_by_pid: Dict[int, Optional[Tuple[int, int, int, int]]] = {
#             1: None, 2: None
#         }
#         for pid, bbox in pilot_boxes:
#             bbox_by_pid[pid] = bbox

#         # Fixed yellow seat zones — upper half for P2, lower half for P1
#         seat_zones: Dict[int, Tuple[int, int, int, int]] = {
#             2: (0, 0,        frame_width, split_y),
#             1: (0, split_y,  frame_width, frame_height),
#         }

#         results:    List[AbsenceResult]   = []
#         log_events: List[Tuple[int, str]] = []

#         for pid in [1, 2]:
#             timer     = self._timers[pid]
#             seat_zone = seat_zones[pid]
#             bbox      = bbox_by_pid.get(pid)

#             in_seat = self._pilot_in_seat(bbox, pid, split_y)

#             if in_seat:
#                 timer.reset()
#                 results.append(AbsenceResult(
#                     pilot_id      = pid,
#                     absent        = False,
#                     timer_value   = 0.0,
#                     calibrated    = True,
#                     seat_zone     = seat_zone,
#                     tracking_bbox = bbox,
#                 ))
#             else:
#                 timer.activate(video_time)
#                 elapsed = timer.elapsed(video_time)
#                 absent  = elapsed >= ABSENCE_ALLOWED_DURATION

#                 if absent and timer.should_log(video_time):
#                     log_events.append((pid, "Pilot Away From Seat"))
#                     timer.mark_logged(video_time)

#                 results.append(AbsenceResult(
#                     pilot_id      = pid,
#                     absent        = absent,
#                     timer_value   = elapsed,
#                     calibrated    = True,
#                     seat_zone     = seat_zone,
#                     tracking_bbox = bbox,
#                 ))

#         return results, log_events

#     # ──────────────────────────────────────────────────────────────
#     # SEAT CHECK
#     # ──────────────────────────────────────────────────────────────

#     @staticmethod
#     def _pilot_in_seat(
#         bbox:    Optional[Tuple[int, int, int, int]],
#         pid:     int,
#         split_y: int,
#     ) -> bool:
#         """
#         Pilot 2 (upper zone):
#             IN SEAT  → box centre is ABOVE split_y
#             ABSENT   → box centre is BELOW split_y OR no detection

#         Pilot 1 (lower zone):
#             IN SEAT  → box CENTRE is in the lower zone (>= split_y)
#             ABSENT   → box centre is above split_y OR no detection

#         FIX (Bug #3): Previously Pilot 1 used `y2 >= split_y` which
#         allowed a standing pilot whose feet were still visible to be
#         counted as seated. Now uses the bbox centre, consistent with
#         the Pilot 2 logic.
#         """
#         if bbox is None:
#             return False

#         x1, y1, x2, y2 = bbox
#         cy = (y1 + y2) / 2  # use centre for both pilots — consistent & robust

#         if pid == 2:
#             return cy < split_y
#         else:
#             # FIX: was `y2 >= split_y`; now uses centre
#             return cy >= split_y



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
# NEW — symmetric debounce for the *return* trigger. Without this, a single
# detector-call frame where in_seat flips True (YOLO noise, brief occlusion,
# reflection, motion blur) instantly closed out the whole episode via
# close_if_confirmed(), truncating true_duration to whatever elapsed by that
# one flickery frame instead of the pilot's real return time. Mirrors the
# GADGET_MISS_TOLERANCE debounce gadget_detector.py already has for its own
# "end" condition (phone no longer seen) — this detector was missing the
# equivalent for "pilot back in seat".
RETURN_GRACE_SECONDS = 0.5   # pilot must be continuously back in seat this long before we call it a real return

@dataclass
class _AbsenceTimer:
    pilot_id:     int
    start_vtime:  Optional[float] = None   # video_time when absence started
    last_logged:  Optional[float] = None   # video_time of last log
    # FIX (Bug #4): replaced frame counter with video-time-based grace period.
    grace_start:  Optional[float] = None   # video_time when first missed
    # NEW — return-debounce tracking (see RETURN_GRACE_SECONDS above).
    return_grace_start: Optional[float] = None   # video_time when in_seat first seen
    # NEW — true trigger→end duration tracking (additive, does not affect
    # any of the existing threshold/relog behaviour above). Set True the
    # moment this episode is first confirmed/logged (see should_log() call
    # site in process()), so reset() knows whether this was a real,
    # already-reported violation (and therefore worth closing out with a
    # true end time) versus a sub-threshold blip that never got logged.
    confirmed:    bool = False

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
        self.start_vtime        = None
        self.last_logged        = None
        self.grace_start        = None
        self.return_grace_start = None
        self.confirmed          = False

    # NEW — call every detector frame the pilot IS in their seat zone.
    # Returns True only once the pilot has been continuously in-seat for
    # RETURN_GRACE_SECONDS — a genuine return. Returns False while still
    # inside the debounce window, so the caller should NOT close the
    # episode yet (treat it as a possible single-frame flicker and keep
    # the absence state exactly as it was).
    def note_in_seat(self, video_time: float) -> bool:
        if self.return_grace_start is None:
            self.return_grace_start = video_time
        return (video_time - self.return_grace_start) >= RETURN_GRACE_SECONDS

    # NEW — call every detector frame the pilot is NOT in their seat zone,
    # to cancel any pending return-debounce window (a real absence frame
    # means the earlier in-seat frame(s) really were just a flicker).
    def note_absent(self) -> None:
        self.return_grace_start = None

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
        self.confirmed   = True

    # NEW — call this instead of reset() at the point where the pilot
    # returns to their seat. If this episode was ever actually confirmed
    # as a violation (i.e. mark_logged() fired at least once), returns
    # (start_vtime, end_vtime, true_duration) for the caller to report as
    # the real trigger→end span. Otherwise returns None (nothing to
    # report — it never crossed the threshold). Always resets afterward,
    # so behaviour at the call site is identical to the old bare
    # `timer.reset()` except for this extra return value.
    def close_if_confirmed(self, end_video_time: float) -> "Optional[Tuple[float, float, float]]":
        result = None
        if self.confirmed and self.start_vtime is not None:
            result = (self.start_vtime, end_video_time, max(0.0, end_video_time - self.start_vtime))
            print(f"[TIMER-CLOSE][absence] pilot={self.pilot_id} start={self.start_vtime:.2f} "
                  f"end={end_video_time:.2f} true_dur={result[2]:.2f}")
        else:
            print(f"[TIMER-CLOSE][absence] pilot={self.pilot_id} skipped — "
                  f"confirmed={self.confirmed} start_vtime={self.start_vtime}")
        self.reset()
        return result


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
    # RESOURCE CLEANUP
    # ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        """
        Release any resources owned by this SeatAbsenceDetector instance.

        SeatAbsenceDetector is a pure-Python state machine with no native
        MediaPipe graphs or file handles — all state is Python dicts and
        dataclasses. This method exists so main.py's finally block can call
        self.absence_detector.close() symmetrically alongside the other
        detector cleanup calls without raising AttributeError. Safe to call
        multiple times.
        """
        # Reset timer state so no references to old video-time values linger.
        for timer in self._timers.values():
            timer.reset()

    # ──────────────────────────────────────────────────────────────
    # PUBLIC — call once per detector frame
    # ──────────────────────────────────────────────────────────────

    def process(
        self,
        pilot_boxes:  List[Tuple[int, Tuple[int, int, int, int]]],
        video_time:   float,
        frame_width:  int = 848,
        frame_height: int = 480,
    ) -> Tuple[List[AbsenceResult], List[Tuple[int, str]], List[Tuple[int, float, float, float]]]:

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

        results:         List[AbsenceResult]              = []
        log_events:      List[Tuple[int, str]]             = []
        # NEW — additive: (pilot_id, start_vtime, end_vtime, true_duration)
        # for every absence episode that gets confirmed (logged) and then
        # actually ends (pilot returns to seat) within this video. Does
        # not change any existing return value or behaviour above.
        completed_events: List[Tuple[int, float, float, float]] = []

        for pid in [1, 2]:
            timer     = self._timers[pid]
            seat_zone = seat_zones[pid]
            bbox      = bbox_by_pid.get(pid)

            in_seat = self._pilot_in_seat(bbox, pid, split_y)

            if in_seat:
                # NEW — only treat this as a genuine return once the pilot
                # has been continuously in-seat for RETURN_GRACE_SECONDS.
                # A single flickery in_seat frame no longer instantly
                # truncates true_duration — it just doesn't reset anything
                # until confirmed by note_in_seat().
                returned = timer.note_in_seat(video_time)
                if returned:
                    closed = timer.close_if_confirmed(video_time)
                    if closed is not None:
                        start_v, end_v, true_dur = closed
                        completed_events.append((pid, start_v, end_v, true_dur))
                    results.append(AbsenceResult(
                        pilot_id      = pid,
                        absent        = False,
                        timer_value   = 0.0,
                        calibrated    = True,
                        seat_zone     = seat_zone,
                        tracking_bbox = bbox,
                    ))
                else:
                    # Still inside the return-debounce window: keep
                    # reporting the ongoing absence unchanged rather than
                    # prematurely clearing it.
                    elapsed = timer.elapsed(video_time)
                    absent  = timer.start_vtime is not None and elapsed >= ABSENCE_ALLOWED_DURATION
                    results.append(AbsenceResult(
                        pilot_id      = pid,
                        absent        = absent,
                        timer_value   = elapsed,
                        calibrated    = True,
                        seat_zone     = seat_zone,
                        tracking_bbox = bbox,
                    ))
            else:
                timer.note_absent()
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

        return results, log_events, completed_events

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