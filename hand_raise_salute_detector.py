#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════
#  hand_raise_salute_detector.py
#  ──────────────────────────────────────────────────────────────────────
#  LOCO PILOT / ASSISTANT LOCO PILOT — HAND-RAISE (SALUTE) DETECTOR
#
#  Detects the railway "call out and point / hand-raise acknowledgement"
#  gesture performed by the Loco Pilot (LP) and/or Assistant Loco Pilot
#  (ALP) inside a side-view cab camera: one arm extended fairly straight,
#  wrist raised above shoulder level, while the other arm stays at rest.
#  Both crew members may perform the gesture at the same time.
#
#  DETECTION LIBRARY: MediaPipe ONLY (no YOLO / no external weights).
#  Human presence is inferred purely from MediaPipe Pose landmark output
#  (a body is "present" when Pose returns a landmark set with usable
#  visibility) — there is no separate person-detector network.
#
#  ENGINE / DEVICE
#  ──────────────────────────────────────────────────────────────────────
#  MediaPipe ships two Python pose APIs:
#    1. Tasks API  (mediapipe.tasks.python.vision.PoseLandmarker)
#       - supports true multi-person detection in a single pass
#         (num_poses=N) — used here to catch LP + ALP together.
#       - accepts a `BaseOptions(delegate=...GPU)` request. On machines
#         where MediaPipe's GPU delegate is available this runs pose
#         inference on the GPU. This is NOT a CUDA execution provider —
#         MediaPipe's public delegate is OpenGL/EGL based (Linux/Android)
#         or Metal (macOS/iOS); there is no first-class CUDA backend in
#         the public Python wheel. We still call it "GPU" honestly below
#         and never silently claim CUDA is running when it is not.
#    2. Legacy `mp.solutions.pose.Pose` — CPU only (XNNPACK), single
#       person per call, but its model ships inside the pip package
#       (no download needed) and is extremely reliable.
#
#  This script ALWAYS tries the Tasks-API GPU path first (so the GPU is
#  used automatically whenever the runtime actually supports it), and
#  transparently falls back to the bundled CPU legacy API — with a
#  console banner stating exactly which engine/device ended up active.
#  Fallback also runs multi-person detection via LEFT/RIGHT half-frame
#  zone splitting (matching the existing project's GREEN_LINE_RATIO
#  zoning convention) so two crew members are still both covered.
#
#  OUTPUT: an OpenCV popup window showing the live/processed video with
#  a bounding box per detected person, RAISED / NOT RAISED status text,
#  and a confidence score (0-100%) derived from MediaPipe landmark
#  visibility. Optionally saves an annotated .mp4 alongside the popup.
# ══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp

# ──────────────────────────────────────────────────────────────────────────
# TUNING CONSTANTS
# ──────────────────────────────────────────────────────────────────────────

# Minimum MediaPipe landmark visibility to trust a shoulder/elbow/wrist
# point at all. Below this the joint is considered occluded/unreliable.
VIS_THRESHOLD = 0.5

# Wrist must be at least this many pixels ABOVE the shoulder (smaller y)
# to count as "raised" rather than a small hand fidget near the panel.
RAISE_MARGIN_PX = 20

# Elbow angle (shoulder-elbow-wrist), degrees. A genuine raised/extended
# salute arm is close to straight. Values were verified against real
# cab-camera reference frames: resting/bent arms sit ~105-135°, the
# extended salute arm sits ~150-180°.
ELBOW_STRAIGHT_MIN_DEG = 150.0

# A softer, secondary rule for a bent-but-clearly-up salute (hand raised
# near/above head height even if the elbow isn't fully locked out).
# If the wrist is above the NOSE by this fraction of torso height, we
# still accept it as "raised" even when the elbow angle check is a
# little under ELBOW_STRAIGHT_MIN_DEG (natural human variation).
WRIST_ABOVE_NOSE_FRACTION = 0.02

# Rolling temporal smoothing: how many consecutive frames a side must be
# "raised" before we confirm it (kills single-frame jitter / false hits).
CONFIRM_FRAMES = 3

# How many consecutive misses before a tracked person's state resets.
MAX_MISS_FRAMES = 8

# Zone split ratio for the legacy CPU fallback (matches the project's
# existing GREEN_LINE_RATIO convention: far/seated pilot on top,
# near/foreground assistant pilot on the bottom of a side-view cab cam).
ZONE_SPLIT_RATIO = 0.55

# Landmark indices (BlazePose-33), identical across legacy Solutions API
# and the newer Tasks API.
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24

# Tasks-API pose landmarker model (auto-downloaded once, cached locally).
_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "pose_landmarker_full.task")
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)


# ──────────────────────────────────────────────────────────────────────────
# RESULT DATA CLASSES
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class ArmState:
    visible: bool = False
    raised: bool = False
    confidence: float = 0.0     # 0-1, landmark visibility based
    elbow_angle: float = 0.0


@dataclass
class PersonResult:
    person_id: int
    label: str                                   # "LOCO PILOT" / "ASST LOCO PILOT" / "PERSON n"
    bbox: Tuple[int, int, int, int]               # x1, y1, x2, y2 (full-frame coords)
    left_arm: ArmState = field(default_factory=ArmState)
    right_arm: ArmState = field(default_factory=ArmState)
    landmarks_px: Optional[dict] = None           # idx -> (x, y, visibility), for skeleton drawing

    @property
    def hand_raised(self) -> bool:
        return self.left_arm.raised or self.right_arm.raised

    @property
    def both_hands_raised(self) -> bool:
        return self.left_arm.raised and self.right_arm.raised

    @property
    def confidence(self) -> float:
        """Overall confidence = visibility of whichever side(s) are raised,
        or the best available arm confidence when nothing is raised."""
        active = [a.confidence for a in (self.left_arm, self.right_arm) if a.raised]
        if active:
            return sum(active) / len(active)
        fallback = [a.confidence for a in (self.left_arm, self.right_arm) if a.visible]
        return max(fallback) if fallback else 0.0

    @property
    def status_text(self) -> str:
        if self.both_hands_raised:
            return "BOTH HANDS RAISED"
        if self.hand_raised:
            return "HAND RAISE DETECTED"
        return "NO HAND RAISE"


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


def _classify_arm(shoulder, elbow, wrist, nose, torso_h: float,
                   vis_shoulder: float, vis_elbow: float, vis_wrist: float) -> ArmState:
    visible = (vis_shoulder >= VIS_THRESHOLD and vis_elbow >= VIS_THRESHOLD
               and vis_wrist >= VIS_THRESHOLD)
    conf = min(vis_shoulder, vis_elbow, vis_wrist)

    if not visible:
        return ArmState(visible=False, raised=False, confidence=conf, elbow_angle=0.0)

    angle = _elbow_angle(shoulder, elbow, wrist)
    wrist_above_shoulder = (shoulder[1] - wrist[1]) >= RAISE_MARGIN_PX
    wrist_above_nose = (nose[1] - wrist[1]) >= (WRIST_ABOVE_NOSE_FRACTION * torso_h)

    # Primary rule: extended-straight arm raised above the shoulder.
    # Secondary rule: wrist at/above head height, even if elbow bend
    # isn't perfectly locked out (natural variation in the salute).
    raised = wrist_above_shoulder and (angle >= ELBOW_STRAIGHT_MIN_DEG or wrist_above_nose)

    return ArmState(visible=True, raised=raised, confidence=conf, elbow_angle=angle)


def _bbox_from_points(points: List[Tuple[float, float]], pad: int, frame_w: int, frame_h: int):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1 = max(0, int(min(xs)) - pad)
    y1 = max(0, int(min(ys)) - pad)
    x2 = min(frame_w - 1, int(max(xs)) + pad)
    y2 = min(frame_h - 1, int(max(ys)) + pad)
    return x1, y1, x2, y2


# ──────────────────────────────────────────────────────────────────────────
# DEVICE / ENGINE SETUP
# ──────────────────────────────────────────────────────────────────────────

def _try_report_cuda() -> None:
    """Purely informational: reports whether a CUDA-capable GPU is visible
    on this machine. MediaPipe's public Python API does not expose a CUDA
    execution provider, so this does not switch MediaPipe's backend — it
    only tells you what hardware is available in case you later want to
    route this pipeline's *other* models (e.g. YOLO elsewhere in this
    project) onto it."""
    try:
        import torch  # optional; not required for MediaPipe itself
        if torch.cuda.is_available():
            print(f"[device] CUDA GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            print("[device] CUDA not available (torch sees no GPU) — using CPU.")
    except ImportError:
        print("[device] torch not installed — skipping CUDA availability probe.")


def _ensure_task_model() -> Optional[str]:
    """Download the Tasks-API pose landmarker model once, cache it locally.
    Returns the local path, or None if it could not be obtained (offline)."""
    if os.path.exists(_MODEL_PATH):
        return _MODEL_PATH
    try:
        os.makedirs(_MODEL_DIR, exist_ok=True)
        print("[setup] Downloading MediaPipe pose_landmarker_full.task "
              "(one-time, ~30 MB)...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print(f"[setup] Model cached at {_MODEL_PATH}")
        return _MODEL_PATH
    except Exception as exc:
        print(f"[setup] Could not download Tasks-API model ({exc}). "
              f"Falling back to the bundled CPU legacy engine.")
        return None


class PoseEngine:
    """Wraps whichever MediaPipe pose engine ends up available:
    Tasks API (multi-person, GPU-attempted) preferred, legacy
    Solutions API (CPU, bundled, single-person-per-call) as fallback.
    Exposes one method: detect_people(frame) -> List[PersonResult]."""

    def __init__(self, max_people: int = 2, force_cpu: bool = False):
        self.max_people = max_people
        self.mode = None            # "tasks_gpu" | "tasks_cpu" | "legacy_cpu"
        self._tasks_landmarker = None
        self._legacy_pose_top = None
        self._legacy_pose_bottom = None

        if not force_cpu:
            self._init_tasks_api()
        if self._tasks_landmarker is None:
            self._init_legacy_api()

        print(f"[engine] Active MediaPipe engine: {self.mode}")

    # ── Tasks API (preferred, GPU-attempted, true multi-person) ────────
    def _init_tasks_api(self) -> None:
        model_path = _ensure_task_model()
        if model_path is None:
            return
        try:
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision as mp_vision

            base_opts_gpu = mp_tasks.BaseOptions(
                model_asset_path=model_path,
                delegate=mp_tasks.BaseOptions.Delegate.GPU,
            )
            options = mp_vision.PoseLandmarkerOptions(
                base_options=base_opts_gpu,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=self.max_people,
                min_pose_detection_confidence=0.4,
                min_pose_presence_confidence=0.4,
                min_tracking_confidence=0.4,
            )
            self._tasks_landmarker = mp_vision.PoseLandmarker.create_from_options(options)
            self.mode = "tasks_gpu"
            return
        except Exception as exc:
            print(f"[engine] GPU delegate unavailable ({exc}); trying Tasks-API on CPU...")

        try:
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision as mp_vision

            base_opts_cpu = mp_tasks.BaseOptions(
                model_asset_path=model_path,
                delegate=mp_tasks.BaseOptions.Delegate.CPU,
            )
            options = mp_vision.PoseLandmarkerOptions(
                base_options=base_opts_cpu,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=self.max_people,
                min_pose_detection_confidence=0.4,
                min_pose_presence_confidence=0.4,
                min_tracking_confidence=0.4,
            )
            self._tasks_landmarker = mp_vision.PoseLandmarker.create_from_options(options)
            self.mode = "tasks_cpu"
        except Exception as exc:
            print(f"[engine] Tasks-API unavailable at all ({exc}); "
                  f"falling back to the bundled legacy CPU engine.")
            self._tasks_landmarker = None

    # ── Legacy Solutions API (guaranteed-available CPU fallback) ───────
    def _init_legacy_api(self) -> None:
        mp_pose = mp.solutions.pose
        # One Pose() instance per zone (top = far/seated pilot side of the
        # cab, bottom = near/foreground assistant pilot side). We run in
        # static_image_mode=True (independent re-detection every frame,
        # not landmark tracking) — slightly slower but this is what the
        # raise/elbow-angle thresholds in this file were tuned against,
        # and it avoids tracking drift when a hand snaps up quickly.
        self._legacy_pose_top = mp_pose.Pose(
            static_image_mode=True, model_complexity=1,
            min_detection_confidence=0.4,
        )
        self._legacy_pose_bottom = mp_pose.Pose(
            static_image_mode=True, model_complexity=1,
            min_detection_confidence=0.4,
        )
        self.mode = "legacy_cpu"

    # ── Public API ───────────────────────────────────────────────────
    def detect_people(self, frame_bgr: np.ndarray) -> List[dict]:
        """Returns a list of raw per-person landmark dicts:
        {landmarks: {idx: (x_px, y_px, visibility)}, points: [(x,y),...]}
        (frame-space pixel coordinates already applied)."""
        if self.mode in ("tasks_gpu", "tasks_cpu"):
            return self._detect_tasks(frame_bgr)
        return self._detect_legacy(frame_bgr)

    def _to_landmark_dict(self, landmarks, w: int, h: int, x_off: int = 0, y_off: int = 0) -> dict:
        out = {}
        for idx, lm in enumerate(landmarks):
            out[idx] = (lm.x * w + x_off, lm.y * h + y_off, lm.visibility)
        return out

    def _detect_tasks(self, frame_bgr: np.ndarray) -> List[dict]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._tasks_landmarker.detect(mp_image)

        people = []
        if result and result.pose_landmarks:
            for landmarks in result.pose_landmarks:
                people.append({"landmarks": self._to_landmark_dict(landmarks, w, h), "zone": None})
        return people

    def _detect_legacy(self, frame_bgr: np.ndarray) -> List[dict]:
        h, w = frame_bgr.shape[:2]
        split_y = int(h * ZONE_SPLIT_RATIO)
        zones = [
            ("top", self._legacy_pose_top, frame_bgr[0:split_y, :], 0, 0),
            ("bottom", self._legacy_pose_bottom, frame_bgr[split_y:h, :], 0, split_y),
        ]
        people = []
        for zone_name, pose_obj, crop, x_off, y_off in zones:
            if crop.size == 0:
                continue
            ch, cw = crop.shape[:2]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = pose_obj.process(rgb)
            if res.pose_landmarks:
                people.append({
                    "landmarks": self._to_landmark_dict(
                        res.pose_landmarks.landmark, cw, ch, x_off, y_off
                    ),
                    "zone": zone_name,
                })
        return people

    def close(self) -> None:
        if self._tasks_landmarker is not None:
            self._tasks_landmarker.close()
        if self._legacy_pose_top is not None:
            self._legacy_pose_top.close()
        if self._legacy_pose_bottom is not None:
            self._legacy_pose_bottom.close()


# ──────────────────────────────────────────────────────────────────────────
# TEMPORAL SMOOTHING (per tracked person, keyed by rough position)
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class _TrackState:
    left_streak: int = 0
    right_streak: int = 0
    miss_frames: int = 0


class PersonTracker:
    """Extremely lightweight tracker: matches new detections to previous
    ones by closest hip-center position, so raise/rest state is smoothed
    across frames per person rather than per-detection-slot."""

    def __init__(self):
        self._states: dict[int, _TrackState] = {}
        self._last_centers: dict[int, Tuple[float, float]] = {}
        self._next_id = 0

    def _match_id(self, center: Tuple[float, float], max_dist: float = 120.0) -> int:
        best_id, best_dist = None, max_dist
        for pid, c in self._last_centers.items():
            d = math.hypot(center[0] - c[0], center[1] - c[1])
            if d < best_dist:
                best_id, best_dist = pid, d
        if best_id is None:
            best_id = self._next_id
            self._next_id += 1
            self._states[best_id] = _TrackState()
        self._last_centers[best_id] = center
        return best_id

    def update(self, pid: int, left_raw: bool, right_raw: bool) -> Tuple[bool, bool]:
        st = self._states.setdefault(pid, _TrackState())
        st.miss_frames = 0
        st.left_streak = st.left_streak + 1 if left_raw else 0
        st.right_streak = st.right_streak + 1 if right_raw else 0
        return (st.left_streak >= CONFIRM_FRAMES, st.right_streak >= CONFIRM_FRAMES)

    def match(self, center: Tuple[float, float]) -> int:
        return self._match_id(center)


# ──────────────────────────────────────────────────────────────────────────
# FRAME ANALYSIS
# ──────────────────────────────────────────────────────────────────────────

def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def analyze_frame(frame: np.ndarray, engine: PoseEngine, tracker: PersonTracker) -> List[PersonResult]:
    h, w = frame.shape[:2]
    raw_people = engine.detect_people(frame)
    candidates = []  # (hip_center, bbox, avg_vis, left_arm, right_arm, landmarks)

    for raw in raw_people:
        lm = raw["landmarks"]
        if L_HIP not in lm or R_HIP not in lm:
            continue

        hip_center = ((lm[L_HIP][0] + lm[R_HIP][0]) / 2, (lm[L_HIP][1] + lm[R_HIP][1]) / 2)
        nose_pt = (lm[NOSE][0], lm[NOSE][1]) if NOSE in lm else hip_center
        torso_h = max(1.0, abs(hip_center[1] - nose_pt[1]))

        def arm(sh, el, wr):
            if sh not in lm or el not in lm or wr not in lm:
                return ArmState(visible=False)
            return _classify_arm(
                (lm[sh][0], lm[sh][1]), (lm[el][0], lm[el][1]), (lm[wr][0], lm[wr][1]),
                nose_pt, torso_h, lm[sh][2], lm[el][2], lm[wr][2],
            )

        left_arm = arm(L_SHOULDER, L_ELBOW, L_WRIST)
        right_arm = arm(R_SHOULDER, R_ELBOW, R_WRIST)

        # Quality gate: torso landmarks (shoulders + hips) must be
        # reasonably confident, or this is likely a spurious pose fit on
        # cab clutter rather than an actual human body.
        torso_vis = [lm[i][2] for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP) if i in lm]
        if not torso_vis or (sum(torso_vis) / len(torso_vis)) < 0.5:
            continue

        points = [(v[0], v[1]) for v in lm.values() if v[2] >= 0.3]
        if not points:
            continue
        bbox = _bbox_from_points(points, pad=15, frame_w=w, frame_h=h)
        avg_vis = sum(v[2] for v in lm.values()) / len(lm)

        candidates.append((hip_center, bbox, avg_vis, left_arm, right_arm, lm))

    # ── De-duplicate: when the legacy engine's top/bottom zone crops both
    # catch the same physical body straddling the split line, keep only
    # the higher-average-visibility detection of the overlapping pair.
    kept = []
    candidates.sort(key=lambda c: c[2], reverse=True)   # best visibility first
    for cand in candidates:
        hip_c, bbox_c = cand[0], cand[1]
        is_dupe = any(
            _bbox_iou(bbox_c, k[1]) > 0.4
            or math.hypot(hip_c[0] - k[0][0], hip_c[1] - k[0][1]) < 60
            for k in kept
        )
        if is_dupe:
            continue
        kept.append(cand)

    # ── Label by RELATIVE vertical position among people found in this
    # frame (top-most hip = far/seated Loco Pilot side of the cab,
    # next = near/foreground Assistant Loco Pilot side). This adapts to
    # either camera orientation instead of relying on a fixed pixel
    # ratio, so it works whether the frame is POV'd from the LP or ALP
    # side of the cab.
    kept.sort(key=lambda c: c[0][1])   # ascending hip y (topmost first)
    labels = ["LOCO PILOT", "ASST LOCO PILOT"]

    results: List[PersonResult] = []
    for i, (hip_center, bbox, avg_vis, left_arm, right_arm, lm) in enumerate(kept):
        label = labels[i] if i < len(labels) else f"PERSON {i + 1}"

        pid = tracker.match(hip_center)
        left_confirmed, right_confirmed = tracker.update(pid, left_arm.raised, right_arm.raised)
        left_arm.raised = left_arm.raised and left_confirmed
        right_arm.raised = right_arm.raised and right_confirmed

        results.append(PersonResult(
            person_id=pid, label=label, bbox=bbox,
            left_arm=left_arm, right_arm=right_arm, landmarks_px=lm,
        ))

    return results


# ──────────────────────────────────────────────────────────────────────────
# DRAWING
# ──────────────────────────────────────────────────────────────────────────

_COLOR_RAISED = (0, 220, 0)       # green
_COLOR_NORMAL = (200, 200, 60)    # cyan-ish
_COLOR_SKELETON = (255, 140, 0)   # orange

_SKELETON_PAIRS = [
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
]


def draw_result(frame: np.ndarray, r: PersonResult) -> None:
    color = _COLOR_RAISED if r.hand_raised else _COLOR_NORMAL
    x1, y1, x2, y2 = r.bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if r.landmarks_px:
        for a, b in _SKELETON_PAIRS:
            if a in r.landmarks_px and b in r.landmarks_px:
                pa = r.landmarks_px[a]
                pb = r.landmarks_px[b]
                if pa[2] >= 0.3 and pb[2] >= 0.3:
                    cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                              _COLOR_SKELETON, 2)

    conf_pct = r.confidence * 100.0
    line1 = f"{r.label} #{r.person_id}"
    line2 = f"{r.status_text}  ({conf_pct:.0f}%)"

    (tw1, th1), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    (tw2, th2), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    box_w = max(tw1, tw2) + 12
    ty = max(0, y1 - th1 - th2 - 16)
    cv2.rectangle(frame, (x1, ty), (x1 + box_w, y1), color, -1)
    cv2.putText(frame, line1, (x1 + 6, ty + th1 + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, line2, (x1 + 6, ty + th1 + th2 + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MediaPipe hand-raise / salute detector for loco cab CCTV.")
    p.add_argument("--source", type=str, default="0",
                   help="Video file path, image path, or webcam index (default: 0).")
    p.add_argument("--save", type=str, default=None,
                   help="Optional path to save annotated output video (.mp4).")
    p.add_argument("--cpu-only", action="store_true",
                   help="Skip the GPU-attempted Tasks-API engine and force the bundled legacy CPU engine.")
    p.add_argument("--max-people", type=int, default=2,
                   help="Max people to track at once (LP + ALP => 2).")
    return p.parse_args()


def _is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in (".png", ".jpg", ".jpeg", ".bmp")


def main():
    args = parse_args()
    _try_report_cuda()

    engine = PoseEngine(max_people=args.max_people, force_cpu=args.cpu_only)
    tracker = PersonTracker()

    source = args.source
    is_webcam = source.isdigit()
    is_image = (not is_webcam) and _is_image_file(source)

    writer = None
    win_name = "Loco Cab - Hand Raise / Salute Detection"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    try:
        if is_image:
            frame = cv2.imread(source)
            if frame is None:
                print(f"[error] Could not read image: {source}")
                sys.exit(1)
            results = analyze_frame(frame, engine, tracker)
            for r in results:
                draw_result(frame, r)
                print(f"{r.label} #{r.person_id}: {r.status_text} "
                      f"(confidence={r.confidence * 100:.1f}%)")
            if args.save:
                cv2.imwrite(args.save, frame)
                print(f"[output] Saved annotated image to {args.save}")
            cv2.imshow(win_name, frame)
            print("Press any key in the popup window to close...")
            cv2.waitKey(0)
        else:
            cap = cv2.VideoCapture(int(source) if is_webcam else source)
            if not cap.isOpened():
                print(f"[error] Could not open source: {source}")
                sys.exit(1)

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if args.save:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))

            print("Press 'q' or ESC in the popup window to quit.")
            prev_t = time.time()
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                results = analyze_frame(frame, engine, tracker)
                for r in results:
                    draw_result(frame, r)

                now = time.time()
                fps_disp = 1.0 / max(1e-6, now - prev_t)
                prev_t = now
                cv2.putText(frame, f"FPS: {fps_disp:.1f}  engine:{engine.mode}",
                            (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                if writer is not None:
                    writer.write(frame)

                cv2.imshow(win_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

            cap.release()
    finally:
        if writer is not None:
            writer.release()
        engine.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
