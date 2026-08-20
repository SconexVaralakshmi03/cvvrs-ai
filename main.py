"""
main.py
────────
GadgetDetectionPipeline — the core per-video AI processing pipeline
(YOLO gadget/phone detection, seat-absence detection, head-droop/
drowsiness detection, hand-raise detection). Detection logic, violation
criteria, confidence thresholds and frame-sampling are UNCHANGED.

CVVRS reliability fixes applied in this file (see project fix list):
  • Fix 6  — per-video progress reporting: an optional `progress_cb`
             callback is invoked periodically (frame-count and
             wall-clock throttled) with (current_frame, processed_frames,
             total_frames, last_progress_time) so the watchdog in
             consumer.py can tell a genuinely slow-but-healthy video
             apart from a stalled one, without guessing from wall-clock
             time alone. The callback fires ONLY when a frame has
             actually been processed — never on a timer by itself — per
             the "do not update the heartbeat merely because the
             watchdog itself is alive" requirement.
  • Fix 8/9/10 — resource lifecycle: `_release_pipeline_resources()`
             (pre-existing in this codebase) already explicitly shuts
             down the ThreadPoolExecutor and closes every MediaPipe/
             native detector resource this pipeline instance owns, on
             every exit path (including the "video failed to open"
             early-exit). Left unchanged here — verified correct.
"""

from __future__ import annotations

import argparse
import os
import queue
import re
import sys
import threading
import traceback
import uuid
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
import warnings
import cv2
import numpy as np

print("[main] ✅ NEW main.py loaded — v5 (merged: mediapipe ear-check, mouth-exclusion, wrist-confirm, fixed drowsiness)")

DRAW           = False
RAW_FRAME_SKIP = 3
GADGET_EVERY   = 6
ABSENCE_EVERY  = 4
DROOP_EVERY    = 15
# NEW — additive: curve-checking (detector/curve_checking.py) cadence.
# Runs its own full-frame YOLO pose pass, so it gets its own cadence
# knob rather than piggy-backing on GADGET_EVERY — tune independently
# if this proves too slow/fast relative to the gadget detector.
CURVE_EVERY    = 6

# ── Fix 6: per-video progress reporting cadence ────────────────────────────
# Purely a reporting cadence — does NOT affect frame sampling, detection
# logic, or which frames are analyzed. Overridable via env for tuning.
PROGRESS_HEARTBEAT_FRAMES  = int(os.environ.get("PROGRESS_HEARTBEAT_FRAMES", "50"))
PROGRESS_HEARTBEAT_SECONDS = float(os.environ.get("PROGRESS_HEARTBEAT_SECONDS", "5"))

# ── MediaPipe pose (optional — graceful degradation if not installed) ─────────
try:
    import mediapipe as mp
    _mp_pose    = mp.solutions.pose
    _MP_AVAILABLE = True
except ImportError:
    _mp_pose      = None
    _MP_AVAILABLE = False
    print("[main] WARNING: mediapipe not installed — gadget detector will use bbox fallback")

from config.settings import OUTPUT_PATH, WINDOW_NAME, DISPLAY_SCALE, GADGET_ALLOWED_DURATION, ABSENCE_ALLOWED_DURATION, HEAD_DROP_DURATION
from utils.logger import setup_logger, log_distraction, finalize_report
from utils.violation_store import ViolationStore
from utils.draw import (
    draw_pilot_box, draw_gadget_box, draw_hud, draw_alert_banner,
    draw_seat_zone, draw_absence_overlay, draw_absence_banner,
    draw_droop_keypoints, draw_droop_overlay, draw_droop_banner,
    draw_standing_label,
)
from detector.gadget_detector import GadgetDetector
from detector.seat_absence_detector import SeatAbsenceDetector
from detector.head_drop_detector import HeadDroopDetector
from detector.hand_raise_detector import HandRaiseDetector, HandRaisePoseEngine
# NEW — additive: curve-checking (ALP looking outside the door on a curve).
from detector.curve_checking import CurveCheckingDetector, draw_door_roi, draw_curve_check_overlay

_STOP = object()
READ_QUEUE_MAXSIZE  = 8
WRITE_QUEUE_MAXSIZE = 8
warnings.filterwarnings("ignore", category=UserWarning)


# Simple container so MediaPipe landmark coordinates can be patched to
# full-frame space before being passed to the gadget detector.
class _PatchedLandmark:
    __slots__ = ("x", "y", "visibility")
    def __init__(self, x: float, y: float, visibility: float):
        self.x          = x
        self.y          = y
        self.visibility = visibility


def _draw_distraction_label(frame, bbox, distraction_type, timer_val, color=(0,0,255)):
    if bbox is None:
        return
    x1, y1, x2, y2 = bbox
    label = f"{distraction_type}  {timer_val:.1f}s"
    font, font_scale, thickness = cv2.FONT_HERSHEY_DUPLEX, 0.52, 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    pad    = 4
    tag_y2 = max(y1, th + pad * 2)
    tag_y1 = tag_y2 - th - pad * 2
    tag_x2 = x1 + tw + pad * 2
    cv2.rectangle(frame, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
    cv2.putText(frame, label, (x1+pad, tag_y2-pad-baseline//2),
                font, font_scale, (255,255,255), thickness, cv2.LINE_AA)


class GadgetDetectionPipeline:

    def __init__(
        self,
        source:           str | int,
        analysis_id:      Optional[str]            = None,
        train_detail_id:  int                      = 0,
        save:             bool                     = False,
        display:          bool                     = False,
        # ── batch-mode parameters ──────────────────────────────────────────
        shared_vstore:    Optional[ViolationStore]  = None,  # shared store for whole folder
        time_offset:      float                    = 0.0,   # cumulative seconds before this video
        frame_offset:     int                      = 0,     # cumulative frames before this video
        source_filename:  str                      = "",    # real DB filename e.g. "mobile.mp4"
        # ── Fix 6: per-video progress reporting ──────────────────────────────
        # Optional callable(current_frame: int, processed_frames: int,
        # total_frames: int, last_progress_time: float) invoked periodically
        # while frames are actually being processed. Never invoked on a bare
        # timer — only alongside real processing progress. video_id is bound
        # by the caller (analyzer.py) via functools.partial / a closure, not
        # passed here, since this pipeline only ever handles one video.
        progress_cb                               = None,
    ) -> None:
        self.source          = source
        self.train_detail_id = train_detail_id
        self.save            = save
        self.display         = display
        self.shared_vstore   = shared_vstore
        self.time_offset     = time_offset
        self.frame_offset    = frame_offset
        self.source_filename = source_filename
        self._progress_cb    = progress_cb

        if analysis_id:
            self.analysis_id = analysis_id
        elif isinstance(source, str) and source not in ("0",) and os.path.isfile(source):
            stem             = os.path.splitext(os.path.basename(source))[0]
            self.analysis_id = re.sub(r"[^A-Za-z0-9_-]", "_", stem)
        else:
            self.analysis_id = uuid.uuid4().hex[:8]

        self.logger           = setup_logger()
        self.detector         = GadgetDetector()
        self.absence_detector = SeatAbsenceDetector()
        self.droop_detector   = HeadDroopDetector()
        # NEW — additive: hand-raise detector + its OWN dedicated pose
        # engine (HandRaisePoseEngine). This does NOT reuse self._pose /
        # _get_pose_landmarks() below — those stay exactly as they were
        # and keep feeding GadgetDetector unchanged. See
        # detector/hand_raise_detector.py for why hand-raise needed its
        # own pose source (shared-tracker-state bug fix).
        self.hand_raise_detector = HandRaiseDetector()
        self._hand_raise_pose    = HandRaisePoseEngine()
        # NEW — additive: curve-checking. Owns its own full-frame YOLO-pose
        # model (see detector/curve_checking.py) — doesn't reuse self._pose
        # or GadgetDetector's model, same reasoning as HandRaisePoseEngine.
        self.curve_check_detector = CurveCheckingDetector()

        # MediaPipe pose — reused across frames, graceful if not installed
        if _MP_AVAILABLE:
            self._pose = _mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            self._pose = None
        self._frame_height: int = 480
        self._frame_width:  int = 848
        self._writer:  Optional[cv2.VideoWriter] = None
        self.vstore:   Optional[ViolationStore]  = None
        self.executor  = ThreadPoolExecutor(max_workers=3)

        self._prev_pilot_boxes      = []
        self._prev_frame_detections = None
        self._processed_frame_no    = 0
        self._absence_pilot_boxes: list = []
        self._yolo_empty_run_count: int = 0

        self._read_queue:  queue.Queue = queue.Queue(maxsize=READ_QUEUE_MAXSIZE)
        self._write_queue: queue.Queue = queue.Queue(maxsize=WRITE_QUEUE_MAXSIZE)

    # ──────────────────────────────────────────────────────────────────────────
    # run()
    # ──────────────────────────────────────────────────────────────────────────

    def _release_pipeline_resources(self) -> None:
        """
        Release the ThreadPoolExecutor and all detector-owned native
        resources (MediaPipe graphs, etc.) constructed in __init__.

        Extracted as its own method so it can be called from BOTH:
          1. The normal run() finally block (cap opened successfully,
             frames were processed).
          2. The early-exit path in run() when cv2.VideoCapture fails to
             open the source — __init__ already constructed self.executor,
             self.detector, self.absence_detector, self.droop_detector,
             and self._pose before that failure is even detected, so
             without this call those resources would leak every time a
             video file can't be opened.

        Idempotent / safe to call more than once — every step is wrapped
        and a second call simply finds nothing left to close.
        """
        try:
            self.executor.shutdown(wait=True)
        except Exception:
            self.logger.error("executor.shutdown failed:\n" + traceback.format_exc())

        try:
            if self._pose is not None:
                self._pose.close()
                self._pose = None
        except Exception:
            self.logger.error("MediaPipe pose.close() failed:\n" + traceback.format_exc())

        try:
            self.droop_detector.close()
        except Exception:
            self.logger.error("HeadDroopDetector.close() failed:\n" + traceback.format_exc())

        try:
            self.hand_raise_detector.close()
        except Exception:
            self.logger.error("HandRaiseDetector.close() failed:\n" + traceback.format_exc())

        try:
            if getattr(self, "_hand_raise_pose", None) is not None:
                self._hand_raise_pose.close()
                self._hand_raise_pose = None
        except Exception:
            self.logger.error("HandRaisePoseEngine.close() failed:\n" + traceback.format_exc())

        try:
            self.absence_detector.close()
        except Exception:
            self.logger.error("SeatAbsenceDetector.close() failed:\n" + traceback.format_exc())

        try:
            self.curve_check_detector.close()
        except Exception:
            self.logger.error("CurveCheckingDetector.close() failed:\n" + traceback.format_exc())

        try:
            self.detector.close()
        except Exception:
            self.logger.error("GadgetDetector.close() failed:\n" + traceback.format_exc())

    def run(self) -> str:
        import time
        start_time = time.time()

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.logger.error(f"Cannot open source: {self.source!r}")
            # FIX (resource leak): __init__ already constructed the
            # executor and all three detectors (with their native
            # MediaPipe/YOLO handles) before we ever got here. Without
            # this call, every "video won't open" case leaked a
            # ThreadPoolExecutor and several MediaPipe graphs — and this
            # path is reached once per bad video file in a long-running
            # worker, so the leak compounds over time.
            self._release_pipeline_resources()
            try:
                cap.release()
            except Exception:
                pass
            sys.exit(1)

        _raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps    = _raw_fps or 25.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_width  = width
        self._frame_height = height
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video      : {width}x{height} @ {fps:.1f}fps  {total/fps:.1f}s  {total} frames")
        print(f"Analysis ID: {self.analysis_id}")
        print(f"time_offset={self.time_offset:.2f}s  frame_offset={self.frame_offset}  source_filename={self.source_filename!r}")

        source_str = str(self.source)
        # Always use the DB filename for display; fall back only for standalone CLI runs
        display_filename = (
            self.source_filename
            if self.source_filename
            else (os.path.basename(source_str) if isinstance(self.source, str) else "webcam")
        )

        duration_s = round(total / fps, 3) if total > 0 and fps > 0 else 0.0
        h, m, s    = int(duration_s)//3600, (int(duration_s)%3600)//60, int(duration_s)%60
        size_mb    = (
            round(os.path.getsize(source_str)/1_000_000, 2)
            if isinstance(self.source, str) and os.path.isfile(source_str) else 0
        )

        video_info = {
            "filename":          display_filename,
            "videoPath":         source_str,
            "durationSeconds":   duration_s,
            "durationFormatted": f"{h}:{m:02d}:{s:02d}",
            "resolution":        f"{width}x{height}",
            "fps":               round(fps, 3),
            "totalFrames":       total,
            "sizeMb":            size_mb,
        }

        if self.shared_vstore is not None:
            # ── BATCH MODE ─────────────────────────────────────────────────
            # Use the shared store; register this video's info; do NOT finalize here.
            self.vstore = self.shared_vstore
            self.vstore.add_video_info(video_info)
            print(f"[Pipeline] BATCH MODE — using shared ViolationStore, will NOT finalize here")
        else:
            # ── STANDALONE MODE ────────────────────────────────────────────
            self.vstore = ViolationStore(
                analysis_id     = self.analysis_id,
                train_detail_id = self.train_detail_id,
                video_info      = video_info,
            )
            print(f"[Pipeline] STANDALONE MODE — fresh ViolationStore, will finalize at end")

        self._print_banner(fps, width, height, total)

        if self.save:
            os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
            self._writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        raw_frame_no = 0
        report_path  = ""

        # ── Fix 6: per-video progress reporting ──────────────────────────────
        # Throttled two ways: at most once every PROGRESS_HEARTBEAT_FRAMES
        # processed frames AND at most once every PROGRESS_HEARTBEAT_SECONDS
        # wall-clock seconds (whichever comes first), so a fast video doesn't
        # flood events_q and a slow video still reports promptly. Only ever
        # called from inside the loop that just processed a real frame —
        # never from an idle timer — so "last_progress_time" genuinely means
        # "processing was still moving forward" to the watchdog reading it.
        _progress_last_emit_frame = 0
        _progress_last_emit_time  = time.time()

        def _maybe_report_progress(force: bool = False) -> None:
            nonlocal _progress_last_emit_frame, _progress_last_emit_time
            if self._progress_cb is None:
                return
            now = time.time()
            due_by_frames = (self._processed_frame_no - _progress_last_emit_frame) >= PROGRESS_HEARTBEAT_FRAMES
            due_by_time   = (now - _progress_last_emit_time) >= PROGRESS_HEARTBEAT_SECONDS
            if not (force or due_by_frames or due_by_time):
                return
            _progress_last_emit_frame = self._processed_frame_no
            _progress_last_emit_time  = now
            try:
                self._progress_cb(raw_frame_no, self._processed_frame_no, total, now)
            except Exception:
                self.logger.error("progress_cb failed (non-fatal):\n" + traceback.format_exc())

        reader_thread = threading.Thread(target=self._reader_loop, args=(cap,), daemon=True, name="FrameReader")
        writer_thread = threading.Thread(target=self._writer_loop, daemon=True, name="FrameWriter")
        reader_thread.start()
        writer_thread.start()

        try:
            while True:
                item = self._read_queue.get()
                if item is _STOP:
                    break
                raw_frame, raw_frame_no, video_time = item
                if raw_frame_no % RAW_FRAME_SKIP != 0:
                    self._write_queue.put(raw_frame)
                    continue
                self._processed_frame_no += 1
                annotated = self._process_frame(raw_frame, video_time, raw_frame_no, self._processed_frame_no)
                self._write_queue.put(annotated)
                _maybe_report_progress()
                if self.display:
                    show = annotated
                    if DISPLAY_SCALE != 1.0:
                        show = cv2.resize(annotated, (int(width*DISPLAY_SCALE), int(height*DISPLAY_SCALE)))
                    cv2.imshow(WINDOW_NAME, show)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break

        except KeyboardInterrupt:
            self.logger.info("\nInterrupted by user.")
        except Exception:
            self.logger.error("Unexpected error:\n" + traceback.format_exc())
        finally:
            # Force one last progress report at whatever frame count we
            # actually reached — covers both "finished normally" (report
            # shows 100%) and "broke out via exception" (report shows the
            # true last-known-good frame, useful for diagnosing where a
            # video actually stopped making progress).
            _maybe_report_progress(force=True)

            self._write_queue.put(_STOP)
            writer_thread.join(timeout=30)

            # FIX (orphan reader thread): if the main loop broke out early
            # due to an exception, the reader thread may still be blocked
            # on self._read_queue.put(...) because nothing is draining the
            # queue anymore. Drain it briefly so the reader can unblock and
            # exit on its own, then join with a bounded timeout so we never
            # hang the worker process.
            if reader_thread.is_alive():
                drain_deadline = time.time() + 5.0
                while reader_thread.is_alive() and time.time() < drain_deadline:
                    try:
                        while True:
                            self._read_queue.get_nowait()
                    except queue.Empty:
                        pass
                    reader_thread.join(timeout=0.25)
            reader_thread.join(timeout=5)
            if reader_thread.is_alive():
                self.logger.warning("FrameReader thread did not terminate cleanly.")

            cap.release()
            if self._writer:
                self._writer.release()
            if self.display:
                cv2.destroyAllWindows()

            # FIX (resource leak): explicitly release the ThreadPoolExecutor
            # and all MediaPipe native graphs owned by this pipeline instance.
            # A fresh GadgetDetectionPipeline (and therefore a fresh executor
            # + fresh MediaPipe Pose/FaceMesh graphs) is created per video in
            # batch/journey mode (see analyzer.py), so without explicit
            # cleanup here these accumulate across every video and every
            # journey processed by a long-running worker.
            #
            # Fix 1: also closes MediaPipe resources owned by absence_detector
            # and gadget detector — both expose close() and are released here
            # via the same helper used by the early cap.isOpened()-failure
            # exit path above, so there is exactly one place this logic lives.
            self._release_pipeline_resources()

            processing_time = round(time.time() - start_time, 3)
            self._print_summary(raw_frame_no, processing_time)
            finalize_report()

            if self.shared_vstore is None:
                # STANDALONE — finalize now
                report_path = self.vstore.finalize(processing_time=processing_time)
            # BATCH — api.py calls finalize() once after all videos; return empty string
            else:
                report_path = ""

        actual_fps = raw_frame_no / processing_time if processing_time > 0 else 0
        print(f"\nTotal Time : {processing_time:.2f}s   FPS : {actual_fps:.2f}")
        return report_path

    # ──────────────────────────────────────────────────────────────────────────
    # reader / writer threads
    # ──────────────────────────────────────────────────────────────────────────

    def _reader_loop(self, cap: cv2.VideoCapture) -> None:
        frame_no = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_no  += 1
                video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                self._read_queue.put((frame, frame_no, video_time))
        except Exception:
            self.logger.error("Reader error:\n" + traceback.format_exc())
        finally:
            self._read_queue.put(_STOP)

    def _writer_loop(self) -> None:
        try:
            while True:
                item = self._write_queue.get()
                if item is _STOP:
                    break
                if self._writer:
                    self._writer.write(item)
        except Exception:
            self.logger.error("Writer error:\n" + traceback.format_exc())

    # ──────────────────────────────────────────────────────────────────────────
    # _process_frame  — all timestamps are offset-adjusted here
    # ──────────────────────────────────────────────────────────────────────────

    def _process_frame(self, frame, video_time, raw_frame_no, processed_frame_no):
        annotated = frame

        # ── global values for this frame ──────────────────────────────────────
        global_time  = video_time  + self.time_offset   # HH:MM:SS into the full recording
        global_frame = raw_frame_no + self.frame_offset  # unique frame index across all videos

        run_gadget  = (processed_frame_no % GADGET_EVERY  == 0)
        run_absence = run_gadget
        run_droop   = (processed_frame_no % DROOP_EVERY   == 0)
        run_curve   = (processed_frame_no % CURVE_EVERY   == 0)

        prev_frame_detection = self._prev_frame_detections

        # ── Build per-pilot MediaPipe landmarks for gadget detector ───────────
        pose_landmarks_by_pilot = self._get_pose_landmarks(frame) if run_gadget else None

        future_gadget  = self.executor.submit(self.detector.process, frame, round(global_time, 3), pose_landmarks_by_pilot) if run_gadget  else None
        future_droop   = self.executor.submit(self.droop_detector.process, frame, global_time, prev_frame_detection) if run_droop else None
        # NEW — additive: curve-checking uses self._prev_pilot_boxes (the
        # LP/ALP boxes from the most recent gadget-detector cycle) purely to
        # EXCLUDE the seated pilots from candidates — same "most recent
        # known" pattern SeatAbsenceDetector uses via self._absence_pilot_boxes.
        future_curve   = self.executor.submit(
            self.curve_check_detector.process, frame, global_time,
            self._prev_pilot_boxes, frame.shape[1], frame.shape[0],
        ) if run_curve else None

        results,         log_events,         gadget_completed  = [], [], []
        absence_results, absence_log_events, absence_completed = [], [], []
        droop_results,   droop_log_events,   droop_completed   = [], [], []
        curve_results,   curve_log_events,   curve_completed   = [], [], []

        try:
            if future_gadget  is not None: results,         log_events,         gadget_completed  = future_gadget.result()
        except Exception as exc:
            self.logger.error(f"Gadget error frame {global_frame}: {exc}", exc_info=True)

        if run_gadget:
            new_boxes = [(r.pilot_id, r.bbox) for r in results]
            self._prev_pilot_boxes      = new_boxes
            self._prev_frame_detections = self.detector.last_frame_detections
            if new_boxes:
                self._absence_pilot_boxes  = new_boxes
                self._yolo_empty_run_count = 0
            else:
                self._yolo_empty_run_count += 1
                if self._yolo_empty_run_count >= 3:
                    self._absence_pilot_boxes = []

        # FIX (false seat-absence at video start): absence used to be
        # submitted to the executor ALONGSIDE future_gadget, using
        # self._absence_pilot_boxes from BEFORE this frame's gadget result
        # was folded back in — i.e. always one detector-cycle stale. That's
        # normally a harmless ~1.2s lag, but self._absence_pilot_boxes
        # starts as [] in __init__ and a fresh pipeline/detector set is
        # created per video, so on the first tick (or two) of EVERY video
        # the absence detector saw an "empty cabin" even when both pilots
        # were sitting right there on THAT SAME frame — just because the
        # real detection hadn't been written back yet. With
        # ABSENCE_GRACE_SECONDS + ABSENCE_ALLOWED_DURATION that startup lag
        # was enough to build a false multi-second "absence" before real
        # detections caught up and cleared it.
        #
        # Fix: run absence AFTER self._absence_pilot_boxes has been updated
        # from THIS frame's gadget result, so it always sees current data.
        # SeatAbsenceDetector.process() is a lightweight pure-Python state
        # machine (no model inference), so calling it synchronously here
        # instead of via the executor costs negligible time.
        if run_absence:
            try:
                absence_results, absence_log_events, absence_completed = self.absence_detector.process(
                    self._absence_pilot_boxes, global_time, frame.shape[1], frame.shape[0]
                )
            except Exception as exc:
                self.logger.error(f"Absence error frame {global_frame}: {exc}", exc_info=True)

        try:
            if future_droop   is not None: droop_results,   droop_log_events,   droop_completed   = future_droop.result()
        except Exception as exc:
            self.logger.error(f"Droop error frame {global_frame}: {exc}", exc_info=True)

        try:
            if future_curve   is not None: curve_results,   curve_log_events,   curve_completed   = future_curve.result()
        except Exception as exc:
            self.logger.error(f"CurveCheck error frame {global_frame}: {exc}", exc_info=True)

        # ── NEW — Hand-raise / signaling detector (additive) ───────────────────
        # Uses its OWN dedicated pose landmarks (self._hand_raise_pose),
        # NOT pose_landmarks_by_pilot above. pose_landmarks_by_pilot keeps
        # going to the gadget detector exactly as before, untouched.
        # HandRaisePoseEngine runs on the same cadence (run_gadget) so
        # timing/cost stays identical to before this change. Pure Python
        # state machine downstream, same as the absence detector, so it's
        # called synchronously rather than via the executor.
        hand_raise_results, hand_raise_log_events, hand_raise_completed = [], [], []
        if run_gadget:
            try:
                hand_raise_pose_landmarks = self._hand_raise_pose.get_landmarks(
                    frame, frame.shape[1], frame.shape[0]
                )
                hand_raise_results, hand_raise_log_events, hand_raise_completed = self.hand_raise_detector.process(
                    hand_raise_pose_landmarks, global_time, frame.shape[1], frame.shape[0]
                )
            except Exception as exc:
                self.logger.error(f"HandRaise error frame {global_frame}: {exc}", exc_info=True)


        # ── Draw (only when DRAW=True) ────────────────────────────────────────
        if DRAW:
            for g in self.detector.last_gadget_hits:
                draw_gadget_box(annotated, g.bbox, g.class_name, g.confidence)
            for ar in absence_results:
                if ar.calibrated and ar.seat_zone is not None:
                    draw_seat_zone(annotated, ar.seat_zone, ar.pilot_id)
            # NEW — additive: curve-checking door-zone overlay.
            door_roi_px = self.curve_check_detector._door_roi_px(frame.shape[1], frame.shape[0])
            annotated = draw_door_roi(annotated, door_roi_px)
            for cr in curve_results:
                annotated = draw_curve_check_overlay(annotated, cr)

        any_gadget_distracted = False
        last_gadget_pilot, last_gadget_name = None, ""

        for r in results:
            gadget_names = [g.class_name for g in r.gadgets]
            if DRAW:
                draw_pilot_box(annotated, r.bbox, r.pilot_id, r.distracted, [])
            if r.distracted:
                any_gadget_distracted = True
                last_gadget_pilot     = r.pilot_id
                last_gadget_name      = gadget_names[0] if gadget_names else "gadget"
                if DRAW:
                    _draw_distraction_label(annotated, r.bbox, "Phone Usage", r.timer_value, color=(0,0,220))

        any_absence_distracted = False
        last_absent_pilot, last_absent_duration = None, 0.0

        for ar in absence_results:
            current_bbox = next((r.bbox for r in results if r.pilot_id == ar.pilot_id), None)
            if DRAW:
                draw_absence_overlay(frame=annotated, bbox=current_bbox, pilot_id=ar.pilot_id,
                                     absent=ar.absent, timer_val=ar.timer_value, calibrated=ar.calibrated)
            if ar.absent:
                any_absence_distracted = True
                last_absent_pilot      = ar.pilot_id
                last_absent_duration   = ar.timer_value
                if DRAW:
                    _draw_distraction_label(annotated, current_bbox, "Away From Seat", ar.timer_value, color=(0,140,255))

        any_droop_distracted = False
        last_droop_pilot, last_droop_duration, last_droop_severity = None, 0.0, "DROWSINESS"
        bbox_by_pid = {r.pilot_id: r.bbox for r in results} if droop_results else {}

        for dr in droop_results:
            current_bbox = bbox_by_pid.get(dr.pilot_id)
            if not dr.is_seated:
                if DRAW: draw_standing_label(annotated, dr.pilot_id, current_bbox)
                continue
            if DRAW:
                if hasattr(dr, "keypoints") and dr.keypoints:
                    draw_droop_keypoints(frame=annotated, keypoints=dr.keypoints, pilot_id=dr.pilot_id,
                                         drooping=dr.drooping, angle=getattr(dr, "angle", 0.0))
                draw_droop_overlay(frame=annotated, pilot_id=dr.pilot_id, drooping=dr.drooping,
                                   timer_val=dr.timer_value, bbox=current_bbox,
                                   severity=getattr(dr, "severity", "DROWSINESS"))
            if dr.drooping:
                any_droop_distracted = True
                last_droop_pilot     = dr.pilot_id
                last_droop_duration  = dr.timer_value
                last_droop_severity  = getattr(dr, "severity", "DROWSINESS")
                if DRAW:
                    _draw_distraction_label(annotated, current_bbox, last_droop_severity,
                                            dr.timer_value*(38/25.0),
                                            color=(0,200,255) if last_droop_severity=="DROWSINESS" else (0,80,200))

        # ── NEW — Hand-raise / signaling draw (additive, DRAW-gated like the
        # other three) — reuses the existing _draw_distraction_label helper,
        # no new drawing utility needed. Green so it reads as a POSITIVE
        # compliance event rather than a violation, unlike the red/orange
        # distraction labels above.
        any_hand_raise = False
        last_hand_raise_pilot, last_hand_raise_label = None, ""
        for hr in hand_raise_results:
            current_bbox = bbox_by_pid.get(hr.pilot_id) if bbox_by_pid else next(
                (r.bbox for r in results if r.pilot_id == hr.pilot_id), None)
            if hr.hand_raised:
                any_hand_raise = True
                last_hand_raise_pilot = hr.pilot_id
                last_hand_raise_label = f"Hand Raise Signaling ({hr.side})"
                if DRAW:
                    _draw_distraction_label(annotated, current_bbox, last_hand_raise_label,
                                            hr.timer_value, color=(0, 200, 0))

        if DRAW:
            if any_gadget_distracted  and last_gadget_pilot  is not None: draw_alert_banner(annotated, last_gadget_pilot,  last_gadget_name)
            if any_absence_distracted and last_absent_pilot  is not None: draw_absence_banner(annotated, last_absent_pilot, last_absent_duration)
            if any_droop_distracted   and last_droop_pilot   is not None: draw_droop_banner(annotated,  last_droop_pilot,  last_droop_duration, severity=last_droop_severity)
            for dr in droop_results:
                if not dr.drooping: continue
                if any(ar.absent and ar.pilot_id == dr.pilot_id for ar in absence_results):
                    _draw_distraction_label(annotated, bbox_by_pid.get(dr.pilot_id), "SLEEPING / ABSENT", dr.timer_value, color=(0,50,200))
            draw_hud(annotated, global_time, global_frame, len(results))

        # ── Record violations ─────────────────────────────────────────────────
        # video_time      = local time within THIS file  → local_video_time param
        # global_time     = video_time + time_offset     → video_time param (→ "timestamp")
        # global_frame    = raw_frame_no + frame_offset  → frame_index param
        # source_filename = DB filename                  → shown in original_video_timestamp

        if log_events:
            r_ref        = next((r for r in results if r.distracted), None)
            conf         = r_ref.gadgets[0].confidence if (r_ref and r_ref.gadgets) else 0.9
            dur          = r_ref.timer_value if r_ref else 0.0
            event_global = max(0.0, global_time - GADGET_ALLOWED_DURATION)
            event_local  = max(0.0, video_time  - GADGET_ALLOWED_DURATION)
            # FIX — Evidence frame didn't match reported timestamp:
            # video_time/local_video_time below are backdated by
            # GADGET_ALLOWED_DURATION (reporting when the violation
            # actually STARTED), but annotated/global_frame are the LIVE
            # current frame at CONFIRMATION time — several seconds later.
            # Passing that live frame as annotated_frame short-circuits
            # the store's frame extraction (it just uploads whatever's in
            # memory), so the saved evidence image shows whatever was
            # happening at confirmation time, not at the reported onset
            # — e.g. reported "0:00:21" but the frame's on-screen clock
            # reads "0:00:28", by which point the phone may already be
            # down. Passing annotated_frame=None lets the downstream
            # extraction re-seek the source video using local_time_str
            # (the backdated onset time), so the saved frame always
            # matches the reported timestamp.
            self.vstore.record_violation(
                annotated_frame=None, original_frame=frame,
                video_time=event_global, frame_index=global_frame,
                event_type="phone_use", severity="CRITICAL",
                confidence=conf, risk_score=80, risk_level="CRITICAL",
                factors=["phone_use", "distraction"], duration=dur,
                source_filename=self.source_filename, local_video_time=event_local,
            )
            log_distraction(self.logger, event_global, event="One of the pilots is using a mobile phone", severity="CRITICAL", frame=annotated)

        if absence_log_events:
            ar_ref       = next((ar for ar in absence_results if ar.absent), None)
            dur_abs      = ar_ref.timer_value if ar_ref else 0.0
            event_global = max(0.0, global_time - ABSENCE_ALLOWED_DURATION)
            event_local  = max(0.0, video_time  - ABSENCE_ALLOWED_DURATION)
            # FIX — see phone_use call above: timestamp is backdated by
            # ABSENCE_ALLOWED_DURATION but the live frame is not; pass
            # annotated_frame=None so the evidence frame is re-seeked at
            # the correct (backdated) local_time_str instead of showing
            # whatever the current confirmation-time frame happened to be.
            self.vstore.record_violation(
                annotated_frame=None, original_frame=frame,
                video_time=event_global, frame_index=global_frame,
                event_type="seat_absence", severity="CRITICAL",
                confidence=1.0, risk_score=70, risk_level="CRITICAL",
                factors=["seat_absence"], duration=dur_abs,
                source_filename=self.source_filename, local_video_time=event_local,
            )
            log_distraction(self.logger, event_global, event="One of the pilots is away from the seat", severity="CRITICAL", frame=annotated)

        if droop_log_events:
            severities  = [e[1] for e in droop_log_events]
            is_sleeping = any("SLEEPING" in s for s in severities)
            droop_pids  = {e[0] for e in droop_log_events}
            absent_pids = {ar.pilot_id for ar in absence_results if ar.absent}
            also_absent = bool(droop_pids & absent_pids)

            if also_absent:  event_msg, etype = "One of the pilots is sleeping / slumped in seat", "sleeping_absent"
            elif is_sleeping: event_msg, etype = "One of the pilots is sleeping", "sleeping"
            else:             event_msg, etype = "One of the pilots is drowsy",   "drowsy"

            dr_ref       = next((dr for dr in droop_results if dr.drooping), None)
            dur_drp      = dr_ref.timer_value if dr_ref else 0.0
            event_global = max(0.0, global_time - HEAD_DROP_DURATION)
            event_local  = max(0.0, video_time  - HEAD_DROP_DURATION)
            # FIX — see phone_use call above: timestamp is backdated by
            # HEAD_DROP_DURATION but the live frame is not; pass
            # annotated_frame=None so the evidence frame is re-seeked at
            # the correct (backdated) local_time_str instead of showing
            # whatever the current confirmation-time frame happened to be.
            self.vstore.record_violation(
                annotated_frame=None, original_frame=frame,
                video_time=event_global, frame_index=global_frame,
                event_type=etype, severity="CRITICAL",
                confidence=0.9, risk_score=75, risk_level="HIGH",
                factors=["drowsy", "head_droop"], duration=dur_drp,
                source_filename=self.source_filename, local_video_time=event_local,
            )
            log_distraction(self.logger, event_global, event=event_msg, severity="CRITICAL", frame=annotated)

        # ── NEW — Hand-raise / signaling record (additive) ─────────────────────
        # Unlike the three blocks above (all sustained ABNORMAL/violation
        # states), this is a brief, deliberate, CORRECT-procedure gesture —
        # recorded as a LOW-severity/LOW-risk event rather than a violation,
        # but still goes through the exact same vstore.record_violation() /
        # log_distraction() calls so it shows up in the same JSON report and
        # the same plain-text distraction log as the other three.
        # event_type="hand_raise" maps to the violationType string
        # "Hand Raise on signaling" — see the mapping added to
        # analyzer.py::_event_type_to_violation_type().
        #
        # BUGFIX (both LP and ALP signaling together only ever showed ONE of
        # them) — hand_raise_log_events can contain MORE THAN ONE entry in
        # the same cycle (one per pilot_id) when both crew members raise a
        # hand at once, e.g. both acknowledging the same external signal.
        # The old code took `next((hr for hr in hand_raise_results if
        # hr.hand_raised), None)` — a SINGLE result — and called
        # record_violation() ONCE regardless of how many pilots were in
        # hand_raise_log_events that cycle, so the second pilot's gesture
        # was silently dropped before it ever reached dedup/merge/LLM
        # verification. Now loops over hand_raise_log_events and records
        # one violation per pilot that actually signaled this cycle, each
        # tagged with its own pilot_id (record_violation()'s dedup key now
        # includes pilot_id — see utils/violation_store.py — so two
        # simultaneous different-pilot events on the same global frame no
        # longer collide into one).
        #
        # Also passes the LIVE in-memory `frame` as annotated_frame instead
        # of None: unlike the phone_use/seat_absence/droop blocks above,
        # hand_raise's event_local/event_global are NOT backdated
        # (event_local = video_time, taken directly, no ALLOWED_DURATION
        # subtraction), so the live frame at confirmation time IS already
        # the correct evidence frame for the reported timestamp — no
        # re-seek needed. This also means the RSL Hand Brake post-
        # verification step (detector/rsl_hand_brake_verifier.py) can reuse
        # this EXACT frame instead of re-seeking the video by timestamp a
        # second time, removing any chance of the two checks ending up
        # looking at two slightly different frames.
        if hand_raise_log_events:
            event_global = global_time
            event_local  = video_time
            hr_by_pilot  = {hr.pilot_id: hr for hr in hand_raise_results}
            for _pid, _gesture_label in hand_raise_log_events:
                hr_ref  = hr_by_pilot.get(_pid)
                conf_hr = hr_ref.confidence  if hr_ref else 0.9
                dur_hr  = hr_ref.timer_value if hr_ref else 0.0
                self.vstore.record_violation(
                    annotated_frame=frame.copy(), original_frame=frame,
                    video_time=event_global, frame_index=global_frame,
                    event_type="hand_raise", severity="LOW",
                    confidence=conf_hr, risk_score=0, risk_level="LOW",
                    factors=["hand_raise", "signaling"], duration=dur_hr,
                    source_filename=self.source_filename, local_video_time=event_local,
                    pilot_id=_pid,
                )
            log_distraction(self.logger, event_global, event="Hand Raise on signaling", severity="LOW", frame=annotated)

        # ── NEW — curve-checking record (additive) ──────────────────────────────
        # Same shape as the hand-raise block above: a brief, deliberate,
        # CORRECT-procedure event (ALP looked outside at the door during a
        # curve) rather than a sustained ABNORMAL state — LOW severity/risk,
        # not backdated (event_local/event_global = the live confirmation-
        # time values, matching hand_raise's reasoning: no re-seek needed).
        if curve_log_events:
            event_global = global_time
            event_local  = video_time
            cr_by_pilot  = {cr.pilot_id: cr for cr in curve_results}
            for _pid, _event_label in curve_log_events:
                cr_ref  = cr_by_pilot.get(_pid)
                conf_cc = cr_ref.score       if cr_ref else 0.9
                dur_cc  = cr_ref.timer_value if cr_ref else 0.0
                self.vstore.record_violation(
                    annotated_frame=frame.copy(), original_frame=frame,
                    video_time=event_global, frame_index=global_frame,
                    event_type="curve_checking", severity="LOW",
                    confidence=conf_cc, risk_score=0, risk_level="LOW",
                    factors=["curve_checking", "outside_view"], duration=dur_cc,
                    source_filename=self.source_filename, local_video_time=event_local,
                    pilot_id=_pid,
                )
            log_distraction(self.logger, event_global, event="Curve Checking on outside view", severity="LOW", frame=annotated)

        # ── NEW — close out true trigger→end durations (additive) ──────────────
        # These fire independently of the log_events blocks above: a
        # completed_events entry means the episode that was already logged
        # earlier has now genuinely ENDED (phone put down, pilot back in
        # seat, no longer drowsy). We look up that already-recorded
        # violation and fill in how long it truly lasted, trigger to end.
        # This never creates a new violation record and never changes any
        # of the fields set above.
        if gadget_completed:
            print(f"[MAIN] gadget_completed frame={global_frame} src={self.source_filename!r}: {gadget_completed}")
        for _pid, _start_v, _end_v, _true_dur in gadget_completed:
            self.vstore.close_violation_episode(
                source_filename=self.source_filename, event_type="phone_use",
                start_video_time=_start_v, end_video_time=_end_v, duration=_true_dur,
            )

        if absence_completed:
            print(f"[MAIN] absence_completed frame={global_frame} src={self.source_filename!r}: {absence_completed}")
        for _pid, _start_v, _end_v, _true_dur in absence_completed:
            self.vstore.close_violation_episode(
                source_filename=self.source_filename, event_type="seat_absence",
                start_video_time=_start_v, end_video_time=_end_v, duration=_true_dur,
            )

        # Droop's event_type is chosen dynamically at log time (drowsy /
        # sleeping / sleeping_absent, depending on whether the pilot was
        # also absent) and isn't knowable from the detector alone, so try
        # each candidate — close_violation_episode() is a no-op if a given
        # type doesn't match, and it always matches at most one violation.
        if droop_completed:
            print(f"[MAIN] droop_completed frame={global_frame} src={self.source_filename!r}: {droop_completed}")
        for _pid, _start_v, _end_v, _true_dur, _dtype in droop_completed:
            for _etype in ("drowsy", "sleeping", "sleeping_absent"):
                self.vstore.close_violation_episode(
                    source_filename=self.source_filename, event_type=_etype,
                    start_video_time=_start_v, end_video_time=_end_v, duration=_true_dur,
                )

        # NEW — additive: closes out the hand-raise episode's true
        # trigger→end duration the same way the three blocks above do.
        # No-op if the episode was too short to have ever been logged in
        # the first place (close_violation_episode() only matches an
        # already-recorded violation).
        if hand_raise_completed:
            print(f"[MAIN] hand_raise_completed frame={global_frame} src={self.source_filename!r}: {hand_raise_completed}")
        for _pid, _start_v, _end_v, _true_dur, _gtype in hand_raise_completed:
            self.vstore.close_violation_episode(
                source_filename=self.source_filename, event_type="hand_raise",
                start_video_time=_start_v, end_video_time=_end_v, duration=_true_dur,
                pilot_id=_pid,
            )

        # NEW — additive: closes out the curve-checking episode's true
        # trigger→end duration, same pattern as the blocks above.
        if curve_completed:
            print(f"[MAIN] curve_completed frame={global_frame} src={self.source_filename!r}: {curve_completed}")
        for _pid, _start_v, _end_v, _true_dur in curve_completed:
            self.vstore.close_violation_episode(
                source_filename=self.source_filename, event_type="curve_checking",
                start_video_time=_start_v, end_video_time=_end_v, duration=_true_dur,
                pilot_id=_pid,
            )

        return annotated

    def _get_pose_landmarks(self, frame: np.ndarray) -> Optional[dict]:
        """
        Run MediaPipe Pose on the full frame and return a dict:
            { pilot_id: [landmark_0, ..., landmark_32] }

        The frame is split at split_y (57 % of height) to assign each
        detected person's landmarks to Pilot 1 or Pilot 2 independently,
        using the same zone logic as the YOLO pilot assignment.

        MediaPipe processes the full frame in one call (it detects up to 1
        person by default). For a two-person frame we crop each pilot's half
        and run MediaPipe on each crop separately.

        Returns None if MediaPipe is not installed or fails.
        """
        if self._pose is None:
            return None

        try:
            h = self._frame_height
            w = self._frame_width
            split_y = int(h * 0.57)

            # Crop each pilot zone and run MediaPipe independently.
            # Pilot 2 = top half (y: 0 → split_y)
            # Pilot 1 = bottom half (y: split_y → h)
            zones = {
                2: frame[0:split_y, 0:w],
                1: frame[split_y:h,  0:w],
            }

            result: dict = {}
            for pid, crop in zones.items():
                if crop.size == 0:
                    continue
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                mp_result = self._pose.process(rgb)
                if mp_result.pose_landmarks is None:
                    continue

                lms = mp_result.pose_landmarks.landmark

                # Adjust Y coordinates back to full-frame pixel space.
                # Pilot 1 crop starts at split_y in the full frame.
                y_offset = split_y if pid == 1 else 0

                # We store the raw landmark objects but patch their .y so
                # that downstream code using lm.y * frame_h gives the
                # correct FULL-FRAME pixel coordinate.
                patched = []
                for lm in lms:
                    crop_h = crop.shape[0]
                    # lm.y is normalised to the crop height; convert to
                    # normalised full-frame Y.
                    full_y = (lm.y * crop_h + y_offset) / h
                    # Build a simple namespace so downstream can use lm.x / lm.y
                    patched.append(_PatchedLandmark(
                        x          = lm.x,          # X is same (full width used)
                        y          = full_y,
                        visibility = getattr(lm, "visibility", 1.0),
                    ))
                result[pid] = patched

            return result if result else None

        except Exception:
            return None

    def _print_banner(self, fps, w, h, total):
        self.logger.info(f"\n{'='*60}\n  LOCO PILOT DISTRACTION DETECTION\n  Analysis ID : {self.analysis_id}\n  Source      : {self.source}\n  Video       : {w}x{h} @ {fps:.1f} fps ({total} frames)\n  Output      : {OUTPUT_PATH if self.save else 'disabled'}\n{'='*60}\n")

    def _print_summary(self, frame_no, processing_time):
        self.logger.info(f"\n{'='*60}\n  Processing complete\n  Raw frames  : {frame_no}\n  Processed   : {self._processed_frame_no} (1 in every {RAW_FRAME_SKIP})\n  Time        : {processing_time:.2f}s\n  Report : outputs/{self.analysis_id}/analysis_report.json\n  Frames : outputs/{self.analysis_id}/frames/\n{'='*60}\n")


def parse_args():
    p = argparse.ArgumentParser(description="Loco Pilot Distraction Detection")
    p.add_argument("--source",          default=0)
    p.add_argument("--analysis-id",     default=None)
    p.add_argument("--train-detail-id", default=0, type=int)
    p.add_argument("--no-display",      action="store_true")
    p.add_argument("--no-save",         action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    GadgetDetectionPipeline(
        source=source, analysis_id=args.analysis_id,
        train_detail_id=args.train_detail_id,
        save=not args.no_save, display=False,
    ).run()