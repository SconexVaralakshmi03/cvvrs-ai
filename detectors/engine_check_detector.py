from __future__ import annotations

import os as _os
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.settings import (
    ENGINE_CHECK_MODEL,
    ENGINE_CHECK_PERSON_CLASS_ID,
    ENGINE_CHECK_PERSON_CONFIDENCE,
    ENGINE_CHECK_TRACKER,
    ENGINE_CHECK_DOOR_X,
    ENGINE_CHECK_DOOR_TOP_Y,
    ENGINE_CHECK_DOOR_BOTTOM_Y,
    ENGINE_CHECK_DOOR_DISTANCE_THRESHOLD,
    ENGINE_CHECK_MIN_MOVEMENT_PIXELS,
    ENGINE_CHECK_MOVING_TOWARD_THRESHOLD,
    ENGINE_CHECK_MIN_CLOSER_FRAMES,
    ENGINE_CHECK_MISSING_FRAMES_REQUIRED,
    ENGINE_CHECK_HISTORY_SIZE,
    ENGINE_CHECK_MAX_TRACK_AGE,
)

# ─────────────────────────────────────────────────────────────────────────────
# SOURCE OF TRUTH
# ─────────────────────────────────────────────────────────────────────────────
#
# This file is a faithful, frame-by-frame port of a standalone YOLOv8m +
# ByteTrack prototype (person approaches an engine-room door, then
# disappears from view => "engine check"). The centroid/distance/movement/
# missing-frame arming logic below is unchanged from that script — see
# config/settings.py's ENGINE_CHECK_* block for the exact constants that
# were ported and the one caveat (literal pixel door coordinates) that was
# deliberately left as-is rather than "fixed" like curve_checking's
# DOOR_ROI was.
#
# TWO things ARE different from the standalone script, both required
# purely because this runs inside a long-lived multi-video pipeline
# instead of a single `python script.py <one video>` invocation, and
# NEITHER changes the frame-by-frame detection outcome for any given
# video:
#
#   1. _reset_tracker() — the standalone script only ever ran once per
#      process, so ByteTrack's persist=True internal state never needed
#      resetting. This pipeline's YOLO model is a per-process singleton
#      (see _get_model(), mirroring gadget_detector.py / curve_checking.py)
#      reused across every video a worker processes, so without an
#      explicit reset at the start of each video, track IDs (and
#      therefore candidate/history state) would bleed across unrelated
#      videos.
#
#   2. ENGINE_CHECK_MAX_TRACK_AGE dead-track cleanup — purely a memory
#      hygiene addition for long-running processes. It only ever deletes
#      state for a track_id that has already been absent for many frames
#      and is therefore no longer reachable by any of the arming/missing
#      logic below — it cannot change what fires.
#
# Everything else — the exact three arming conditions (inside the
# approach radius, moving toward the door, distance trending down), the
# exact missing-frame confirmation count, and calling this detector on
# EVERY raw frame with no skipping (main.py calls this before its
# RAW_FRAME_SKIP gate, unlike every other detector in this pipeline) — is
# unchanged.

PERSON_CLASS_ID = ENGINE_CHECK_PERSON_CLASS_ID


# ─────────────────────────────────────────────────────────────────────────────
# YOLO + BYTETRACK MODEL (lazy, per-process singleton)
# ─────────────────────────────────────────────────────────────────────────────

_model = None
_MODEL_DEVICE = _os.environ.get("ENGINE_CHECK_YOLO_DEVICE", "cuda:0")


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        try:
            import torch
            _has_cuda = torch.cuda.is_available()
        except Exception:
            torch = None
            _has_cuda = False

        if _MODEL_DEVICE.startswith("cuda") and _has_cuda:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        _model = YOLO(ENGINE_CHECK_MODEL)

        if _MODEL_DEVICE.startswith("cuda") and _has_cuda:
            try:
                _model.to(_MODEL_DEVICE)
            except Exception:
                pass
        # No CUDA → leave on default (CPU) device.

    return _model


def _reset_tracker(model) -> None:
    """
    Force ByteTrack to reinitialize at the start of a new video.

    model.track(persist=True) keeps its tracker state living on the
    shared model.predictor object across every call — including calls
    from a PREVIOUS video, since _get_model() above is a per-process
    singleton reused across every video a worker processes (same pattern
    as gadget_detector.py / curve_checking.py). Without this reset, a
    track_id from the tail of one video could continue straight into the
    next video's frame 1, corrupting that track's history/candidate
    state. The standalone script never hit this because it only ever
    processed one video per process run. Version-robust: just drops the
    predictor so ultralytics rebuilds a fresh tracker on the next
    .track() call, rather than poking at internal tracker fields that
    vary across ultralytics versions.
    """
    try:
        model.predictor = None
    except Exception:
        pass


def release_model() -> None:
    """Explicitly drop the model singleton and free its CUDA memory."""
    global _model
    if _model is not None:
        try:
            del _model
        finally:
            _model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# MATH HELPERS — exact port of the standalone script's helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _centroid(x1: float, y1: float, x2: float, y2: float) -> Tuple[int, int]:
    """Exact port of the standalone script's calculate_centroid()."""
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Exact port of the standalone script's calculate_distance()."""
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def _vector(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
    """Exact port of the standalone script's calculate_vector()."""
    return (p2[0] - p1[0], p2[1] - p1[1])


def _magnitude(v: Tuple[float, float]) -> float:
    """Exact port of the standalone script's vector_magnitude()."""
    return float(np.hypot(v[0], v[1]))


def _cosine_similarity(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """Exact port of the standalone script's cosine_similarity()."""
    m1, m2 = _magnitude(v1), _magnitude(v2)
    if m1 == 0 or m2 == 0:
        return 0.0
    return ((v1[0] * v2[0]) + (v1[1] * v2[1])) / (m1 * m2)


def _is_distance_decreasing(distance_history: deque) -> bool:
    """Exact port of the standalone script's is_distance_decreasing()."""
    required = ENGINE_CHECK_MIN_CLOSER_FRAMES + 1
    if len(distance_history) < required:
        return False
    recent = list(distance_history)[-required:]
    decreasing_count = sum(
        1 for i in range(1, len(recent)) if recent[i] < recent[i - 1]
    )
    return decreasing_count >= ENGINE_CHECK_MIN_CLOSER_FRAMES


# ─────────────────────────────────────────────────────────────────────────────
# TRACK STATE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Track:
    history:           deque = field(default_factory=lambda: deque(maxlen=ENGINE_CHECK_HISTORY_SIZE))
    distance_history:  deque = field(default_factory=lambda: deque(maxlen=10))
    candidate:         bool  = False
    candidate_frame:   Optional[int] = None
    missing_frames:    int   = 0
    last_seen:         int   = 0
    last_similarity:   float = 0.0


@dataclass
class EngineCheckResult:
    """One tracked person, this frame — used for the debug overlay."""
    track_id:            int
    bbox:                 Tuple[int, int, int, int]
    center:               Tuple[int, int]
    distance_to_door:     float
    inside_door_zone:     bool
    moving_toward_door:   bool
    similarity:           float
    distance_decreasing:  bool
    candidate:            bool
    history:              List[Tuple[int, int]] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class EngineCheckDetector:
    """
    Faithful port of the standalone engine-check prototype.

    Call process() on EVERY raw frame (see module docstring) — main.py
    calls this before its RAW_FRAME_SKIP gate specifically so this
    detector never skips a frame, unlike every other detector in this
    pipeline.

    A track is armed as a "candidate" the first frame all three hold:
      1. within ENGINE_CHECK_DOOR_DISTANCE_THRESHOLD px of the door
      2. moving toward the door (cosine similarity check)
      3. distance-to-door has been decreasing for
         ENGINE_CHECK_MIN_CLOSER_FRAMES consecutive samples

    Once armed, if that track then goes missing (not detected in the
    current frame) for ENGINE_CHECK_MISSING_FRAMES_REQUIRED consecutive
    frames, "engine check" fires exactly once for that track_id.
    """

    def __init__(self) -> None:
        self._door_center: Tuple[int, int] = (
            ENGINE_CHECK_DOOR_X,
            int((ENGINE_CHECK_DOOR_TOP_Y + ENGINE_CHECK_DOOR_BOTTOM_Y) / 2),
        )
        self._tracks: Dict[int, _Track] = {}
        # Mirrors the standalone script's `detected_engine_checks` +
        # `already_detected = any(event["track_id"] == track_id ...)`
        # dedup check — one event per track_id, ever, for this video.
        self._logged_track_ids: set = set()

        _reset_tracker(_get_model())

    @property
    def door_center(self) -> Tuple[int, int]:
        return self._door_center

    # ──────────────────────────────────────────────────────────────
    # RESOURCE CLEANUP
    # ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        """
        Release this instance's per-video track state. The underlying
        YOLO model is a per-process singleton (see _get_model()) shared
        with the next video's EngineCheckDetector instance, and is reset
        via _reset_tracker() in __init__ rather than here.
        """
        self._tracks.clear()
        self._logged_track_ids.clear()

    # ──────────────────────────────────────────────────────────────
    # PUBLIC — call once per RAW frame (no skipping)
    # ──────────────────────────────────────────────────────────────

    def process(
        self,
        frame: np.ndarray,
        frame_number: int,
        video_time: float,
    ) -> Tuple[List[EngineCheckResult], List[Tuple[int, str]], List]:
        """
        Parameters
        ----------
        frame        : full raw BGR frame (no cropping — exact standalone
                       model.track(frame, ...) call).
        frame_number : raw frame index (used for candidate_frame /
                       last_seen bookkeeping and missing-frame counting —
                       exact standalone frame_number semantics).
        video_time   : this frame's timestamp, for the caller's
                       violation-store bookkeeping only; not used in any
                       arming/missing decision below (the standalone
                       script had no such concept either).

        Returns
        -------
        results           : List[EngineCheckResult] — every person tracked
                             this frame, for the optional debug overlay.
        log_events        : List[(track_id, "Engine check detected")] —
                             fires exactly once per track_id, the frame its
                             missing-frame count first reaches
                             ENGINE_CHECK_MISSING_FRAMES_REQUIRED.
        completed_events  : always [] — the standalone script has no
                             start/end episode/duration concept, only a
                             one-shot detection event (same shape as
                             detectors/curve_checking.py's contract).
        """
        model = _get_model()

        yolo_results = model.track(
            frame,
            persist=True,
            tracker=ENGINE_CHECK_TRACKER,
            classes=[PERSON_CLASS_ID],
            conf=ENGINE_CHECK_PERSON_CONFIDENCE,
            verbose=False,
        )
        result = yolo_results[0]

        current_track_ids: set = set()
        results: List[EngineCheckResult] = []
        log_events: List[Tuple[int, str]] = []
        completed_events: List = []

        if result.boxes is not None and result.boxes.id is not None:
            track_ids   = result.boxes.id.int().cpu().tolist()
            coordinates = result.boxes.xyxy.cpu().tolist()

            for track_id, box in zip(track_ids, coordinates):
                track_id = int(track_id)
                current_track_ids.add(track_id)

                x1, y1, x2, y2 = map(int, box)
                center = _centroid(x1, y1, x2, y2)

                track = self._tracks.setdefault(track_id, _Track())
                track.history.append(center)
                track.missing_frames = 0
                track.last_seen = frame_number

                door_distance = _distance(center, self._door_center)
                track.distance_history.append(door_distance)

                moving_toward = False
                similarity    = 0.0
                if len(track.history) >= 2:
                    previous_center = track.history[-2]
                    movement = _vector(previous_center, center)
                    if _magnitude(movement) >= ENGINE_CHECK_MIN_MOVEMENT_PIXELS:
                        to_door    = _vector(center, self._door_center)
                        similarity = _cosine_similarity(movement, to_door)
                        moving_toward = similarity >= ENGINE_CHECK_MOVING_TOWARD_THRESHOLD
                track.last_similarity = similarity

                distance_decreasing = _is_distance_decreasing(track.distance_history)
                inside_door_zone    = door_distance <= ENGINE_CHECK_DOOR_DISTANCE_THRESHOLD

                # ── ARM CANDIDATE — exact standalone condition (no region
                # check — the final standalone version doesn't have one) ──
                if inside_door_zone and moving_toward and distance_decreasing:
                    if not track.candidate:
                        track.candidate       = True
                        track.candidate_frame = frame_number
                        print(
                            f"[EngineCheck] CANDIDATE track={track_id} "
                            f"frame={frame_number} dist={door_distance:.1f}px "
                            f"sim={similarity:.2f}"
                        )

                results.append(EngineCheckResult(
                    track_id             = track_id,
                    bbox                 = (x1, y1, x2, y2),
                    center               = center,
                    distance_to_door     = door_distance,
                    inside_door_zone     = inside_door_zone,
                    moving_toward_door   = moving_toward,
                    similarity           = similarity,
                    distance_decreasing  = distance_decreasing,
                    candidate            = track.candidate,
                    history              = list(track.history),
                ))

        # ── MISSING-TRACK BOOKKEEPING — exact port ──────────────────────
        for track_id, track in self._tracks.items():
            if track_id in current_track_ids:
                continue

            if not track.candidate:
                track.missing_frames = 0
                continue

            track.missing_frames += 1
            print(
                f"[EngineCheck] frame={frame_number} track={track_id} "
                f"missing {track.missing_frames}/{ENGINE_CHECK_MISSING_FRAMES_REQUIRED}"
            )

            if track.missing_frames >= ENGINE_CHECK_MISSING_FRAMES_REQUIRED:
                if track_id not in self._logged_track_ids:
                    self._logged_track_ids.add(track_id)
                    log_events.append((track_id, "Engine check detected"))
                    print(
                        f"[EngineCheck] DETECTED track={track_id} "
                        f"candidate_frame={track.candidate_frame} "
                        f"confirm_frame={frame_number} "
                        f"missing_frames={track.missing_frames}"
                    )

        # ── dead-track cleanup — NEW, memory hygiene only (see module
        # docstring point 2) ─────────────────────────────────────────────
        stale = [
            tid for tid, t in self._tracks.items()
            if frame_number - t.last_seen > ENGINE_CHECK_MAX_TRACK_AGE
        ]
        for tid in stale:
            del self._tracks[tid]

        return results, log_events, completed_events


# ─────────────────────────────────────────────────────────────────────────────
# DRAW HELPERS — only called when main.py's DRAW flag is True, same as
# every other detector's overlay (see detectors/curve_checking.py).
# ─────────────────────────────────────────────────────────────────────────────

def draw_engine_door_zone(frame: np.ndarray, door_center: Tuple[int, int]) -> np.ndarray:
    """Exact port of the standalone script's door-center + approach-circle
    drawing (door line itself is resolution/ROI-specific and left to the
    caller if desired — the centroid + radius are the load-bearing part)."""
    import cv2
    cv2.circle(frame, door_center, 10, (0, 0, 255), -1)
    cv2.circle(frame, door_center, ENGINE_CHECK_DOOR_DISTANCE_THRESHOLD, (0, 0, 255), 2)
    cv2.putText(frame, "ENGINE DOOR", (door_center[0] + 15, door_center[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame


def draw_engine_check_overlay(frame: np.ndarray, result: EngineCheckResult) -> np.ndarray:
    """Exact port of the standalone script's per-person box/trajectory/
    label drawing and color scheme (yellow=candidate, orange=in zone,
    green=neither)."""
    import cv2

    if result.candidate:
        color = (0, 255, 255)
    elif result.inside_door_zone:
        color = (0, 165, 255)
    else:
        color = (0, 255, 0)

    x1, y1, x2, y2 = result.bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    cv2.circle(frame, result.center, 7, color, -1)

    history = result.history
    for i in range(1, len(history)):
        cv2.line(frame, history[i - 1], history[i], (255, 255, 0), 2)

    cv2.putText(frame, f"ID:{result.track_id} dist:{result.distance_to_door:.0f}px",
                (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    if result.moving_toward_door:
        cv2.putText(frame, f"TOWARD DOOR {result.similarity:.2f}",
                    (x1, min(frame.shape[0] - 30, y2 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    if result.candidate:
        cv2.putText(frame, "ENGINE CANDIDATE",
                    (x1, min(frame.shape[0] - 5, y2 + 42)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    return frame
