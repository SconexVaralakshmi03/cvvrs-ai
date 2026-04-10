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


DRAW           = False  # set True only for visual debug
RAW_FRAME_SKIP = 3      # process 1 in every N raw frames
GADGET_EVERY   = 6      # YOLO  every Nth processed frame
ABSENCE_EVERY  = 4      # absence every Nth processed frame
DROOP_EVERY    = 15     # droop every Nth processed frame
 # allowed duration in seconds before logging violation


from config.settings import OUTPUT_PATH, WINDOW_NAME, DISPLAY_SCALE,GADGET_ALLOWED_DURATION,ABSENCE_ALLOWED_DURATION,HEAD_DROP_DURATION
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

_STOP = object()

READ_QUEUE_MAXSIZE  = 8
WRITE_QUEUE_MAXSIZE = 8

warnings.filterwarnings("ignore", category=UserWarning)


def _draw_distraction_label(
    frame: np.ndarray,
    bbox: tuple,
    distraction_type: str,
    timer_val: float,
    color: tuple = (0, 0, 255),
) -> None:
    if bbox is None:
        return
    x1, y1, x2, y2 = bbox
    label = f"{distraction_type}  {timer_val:.1f}s"
    font       = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.52
    thickness  = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    pad    = 4
    tag_y2 = max(y1, th + pad * 2)
    tag_y1 = tag_y2 - th - pad * 2
    tag_x2 = x1 + tw + pad * 2
    cv2.rectangle(frame, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
    cv2.putText(
        frame, label,
        (x1 + pad, tag_y2 - pad - baseline // 2),
        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
    )


class GadgetDetectionPipeline:

    def __init__(
        self,
        source:          str | int,
        analysis_id:     Optional[str] = None,
        train_detail_id: int           = 0,
        save:            bool          = False,
        display:         bool          = False,
    ) -> None:
        self.source          = source
        self.train_detail_id = train_detail_id
        self.save            = save
        self.display         = display

        if analysis_id:
            self.analysis_id = analysis_id
        elif (
            isinstance(source, str)
            and source not in ("0",)
            and os.path.isfile(source)
        ):
            stem             = os.path.splitext(os.path.basename(source))[0]
            self.analysis_id = re.sub(r"[^A-Za-z0-9_-]", "_", stem)
        else:
            self.analysis_id = uuid.uuid4().hex[:8]

        self.logger           = setup_logger()
        self.detector         = GadgetDetector()
        self.absence_detector = SeatAbsenceDetector()
        self.droop_detector   = HeadDroopDetector()
        self._writer:  Optional[cv2.VideoWriter] = None
        self.vstore:   Optional[ViolationStore]  = None

        # 3 workers: one per detector, no excess overhead
        self.executor = ThreadPoolExecutor(max_workers=3)

        self._prev_pilot_boxes      = []
        self._prev_frame_detections = None
        self._processed_frame_no    = 0   

        self._read_queue:  queue.Queue = queue.Queue(maxsize=READ_QUEUE_MAXSIZE)
        self._write_queue: queue.Queue = queue.Queue(maxsize=WRITE_QUEUE_MAXSIZE)

    
    # ENTRY POINT
    

    def run(self) -> str:
        import time
        start_time = time.time()

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.logger.error(f"Cannot open source: {self.source!r}")
            sys.exit(1)

        _raw_fps = cap.get(cv2.CAP_PROP_FPS)
        if not _raw_fps:
            print("[WARNING] FPS not detected — defaulting to 25.0")
        fps    = _raw_fps or 25.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video      : {width}x{height} @ {fps:.1f}fps  "
              f"{total/fps:.1f}s  {total} frames")
        print(f"Analysis ID: {self.analysis_id}")
        print(f"Processing : every {RAW_FRAME_SKIP}rd raw frame  |  "
              f"YOLO every {RAW_FRAME_SKIP * GADGET_EVERY} raw frames  |  "
              f"Droop every {RAW_FRAME_SKIP * DROOP_EVERY} raw frames")

        source_str  = str(self.source)
        source_name = (
            os.path.basename(source_str)
            if isinstance(self.source, str) else "webcam"
        )
        duration_s = round(total / fps, 3) if total > 0 and fps > 0 else 0.0
        h, m, s    = (
            int(duration_s) // 3600,
            (int(duration_s) % 3600) // 60,
            int(duration_s) % 60,
        )
        size_mb = (
            round(os.path.getsize(source_str) / 1_000_000, 2)
            if isinstance(self.source, str) and os.path.isfile(source_str) else 0
        )

        video_info = {
            "filename":          source_name,
            "videoPath":         source_str,
            "durationSeconds":   duration_s,
            "durationFormatted": f"{h}:{m:02d}:{s:02d}",
            "resolution":        f"{width}x{height}",
            "fps":               round(fps, 3),
            "totalFrames":       total,
            "sizeMb":            size_mb,
        }

        self.vstore = ViolationStore(
            analysis_id     = self.analysis_id,
            train_detail_id = self.train_detail_id,
            video_info      = video_info,
        )
        self._print_banner(fps, width, height, total)

        if self.save:
            os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
            self._writer = cv2.VideoWriter(
                OUTPUT_PATH,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

        raw_frame_no = 0
        report_path  = ""

        reader_thread = threading.Thread(
            target=self._reader_loop, args=(cap,),
            daemon=True, name="FrameReader",
        )
        writer_thread = threading.Thread(
            target=self._writer_loop,
            daemon=True, name="FrameWriter",
        )
        reader_thread.start()
        writer_thread.start()

        try:
            while True:
                item = self._read_queue.get()
                if item is _STOP:
                    break

                raw_frame, raw_frame_no, video_time = item

                # ── Skip most raw frames — pass through as-is ─────
                if raw_frame_no % RAW_FRAME_SKIP != 0:
                    self._write_queue.put(raw_frame)
                    continue

                # ── Process this frame ────────────────────────────
                self._processed_frame_no += 1
                annotated = self._process_frame(
                    raw_frame, video_time, raw_frame_no, self._processed_frame_no
                )
                self._write_queue.put(annotated)

                if self.display:
                    show = annotated
                    if DISPLAY_SCALE != 1.0:
                        show = cv2.resize(
                            annotated,
                            (int(width * DISPLAY_SCALE), int(height * DISPLAY_SCALE)),
                        )
                    cv2.imshow(WINDOW_NAME, show)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        self.logger.info("Quit by user.")
                        break

        except KeyboardInterrupt:
            self.logger.info("\nInterrupted by user.")
        except Exception:
            self.logger.error("Unexpected error:\n" + traceback.format_exc())
        finally:
            self._write_queue.put(_STOP)
            writer_thread.join(timeout=30)
            cap.release()
            if self._writer:
                self._writer.release()
            if self.display:
                cv2.destroyAllWindows()

            processing_time = round(time.time() - start_time, 3)
            self._print_summary(raw_frame_no, processing_time)
            finalize_report()
            report_path = self.vstore.finalize(processing_time=processing_time)

        actual_fps = raw_frame_no / processing_time if processing_time > 0 else 0
        print(f"\nTotal Time : {processing_time:.2f}s   FPS : {actual_fps:.2f}")
        return report_path

 
    # READER THREAD
   

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

    
    # WRITER THREAD
    

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

    
    # PER-FRAME PROCESSING
    

    def _process_frame(
        self,
        frame:              np.ndarray,
        video_time:         float,
        raw_frame_no:       int,
        processed_frame_no: int,
    ) -> np.ndarray:
        annotated = frame

        # Cadences are relative to processed_frame_no so that the effective ML frequency is consistent regardless of RAW_FRAME_SKIP.
        run_gadget  = (processed_frame_no % GADGET_EVERY  == 0)
        run_absence = (processed_frame_no % ABSENCE_EVERY == 0)
        run_droop   = (processed_frame_no % DROOP_EVERY   == 0)

        prev_pilot_boxes     = self._prev_pilot_boxes
        prev_frame_detection = self._prev_frame_detections

        future_gadget = (
            self.executor.submit(self.detector.process, frame, round(video_time, 0))
            if run_gadget else None
        )
        future_absence = (
            self.executor.submit(
                self.absence_detector.process,
                prev_pilot_boxes, video_time, frame.shape[1], frame.shape[0],
            )
            if run_absence else None
        )
        future_droop = (
            self.executor.submit(
                self.droop_detector.process,
                frame, video_time, prev_frame_detection,
            )
            if run_droop else None
        )

        results,         log_events         = [], []
        absence_results, absence_log_events = [], []
        droop_results,   droop_log_events   = [], []

        try:
            if future_gadget is not None:
                results, log_events = future_gadget.result()
        except Exception as exc:
            self.logger.error(f"Gadget error frame {raw_frame_no}: {exc}", exc_info=True)

        try:
            if future_absence is not None:
                absence_results, absence_log_events = future_absence.result()
        except Exception as exc:
            self.logger.error(f"Absence error frame {raw_frame_no}: {exc}", exc_info=True)

        try:
            if future_droop is not None:
                droop_results, droop_log_events = future_droop.result()
        except Exception as exc:
            self.logger.error(f"Droop error frame {raw_frame_no}: {exc}", exc_info=True)

        if run_gadget:
            self._prev_pilot_boxes      = [(r.pilot_id, r.bbox) for r in results]
            self._prev_frame_detections = self.detector.last_frame_detections

        #  Draw (skipped entirely when DRAW=False
        if DRAW:
            for g in self.detector.last_gadget_hits:
                draw_gadget_box(annotated, g.bbox, g.class_name, g.confidence)
            for ar in absence_results:
                if ar.calibrated and ar.seat_zone is not None:
                    draw_seat_zone(annotated, ar.seat_zone, ar.pilot_id)

        any_gadget_distracted = False
        last_gadget_pilot     = None
        last_gadget_name      = ""

        for r in results:
            gadget_names = [g.class_name for g in r.gadgets]
            if DRAW:
                draw_pilot_box(annotated, r.bbox, r.pilot_id, r.distracted, [])
            if r.distracted:
                any_gadget_distracted = True
                last_gadget_pilot     = r.pilot_id
                last_gadget_name      = gadget_names[0] if gadget_names else "gadget"
                if DRAW:
                    _draw_distraction_label(annotated, r.bbox, "Phone Usage",
                                            r.timer_value, color=(0, 0, 220))

        any_absence_distracted = False
        last_absent_pilot      = None
        last_absent_duration   = 0.0

        for ar in absence_results:
            current_bbox = next(
                (r.bbox for r in results if r.pilot_id == ar.pilot_id), None
            )
            if DRAW:
                draw_absence_overlay(
                    frame=annotated, bbox=current_bbox, pilot_id=ar.pilot_id,
                    absent=ar.absent, timer_val=ar.timer_value,
                    calibrated=ar.calibrated,
                )
            if ar.absent:
                any_absence_distracted = True
                last_absent_pilot      = ar.pilot_id
                last_absent_duration   = ar.timer_value
                if DRAW:
                    _draw_distraction_label(annotated, current_bbox, "Away From Seat",
                                            ar.timer_value, color=(0, 140, 255))

        any_droop_distracted = False
        last_droop_pilot     = None
        last_droop_duration  = 0.0
        last_droop_severity  = "DROWSINESS"
        bbox_by_pid          = {}

        if droop_results:
            bbox_by_pid = {r.pilot_id: r.bbox for r in results}

        for dr in droop_results:
            current_bbox = bbox_by_pid.get(dr.pilot_id)
            if not dr.is_seated:
                if DRAW:
                    draw_standing_label(annotated, dr.pilot_id, current_bbox)
                continue
            if hasattr(dr, "keypoints") and dr.keypoints:
                if DRAW:
                    draw_droop_keypoints(
                        frame=annotated, keypoints=dr.keypoints,
                        pilot_id=dr.pilot_id, drooping=dr.drooping,
                        angle=getattr(dr, "angle", 0.0),
                    )
            if DRAW:
                draw_droop_overlay(
                    frame=annotated, pilot_id=dr.pilot_id,
                    drooping=dr.drooping, timer_val=dr.timer_value,
                    bbox=current_bbox,
                    severity=getattr(dr, "severity", "DROWSINESS"),
                )
            if dr.drooping:
                any_droop_distracted = True
                last_droop_pilot     = dr.pilot_id
                last_droop_duration  = dr.timer_value
                last_droop_severity  = getattr(dr, "severity", "DROWSINESS")
                display_secs         = dr.timer_value * (38 / 25.0)
                if DRAW:
                    _draw_distraction_label(
                        annotated, current_bbox, last_droop_severity, display_secs,
                        color=(0, 200, 255)
                        if last_droop_severity == "DROWSINESS" else (0, 80, 200),
                    )

        if DRAW:
            if any_gadget_distracted and last_gadget_pilot is not None:
                draw_alert_banner(annotated, last_gadget_pilot, last_gadget_name)
            if any_absence_distracted and last_absent_pilot is not None:
                draw_absence_banner(annotated, last_absent_pilot, last_absent_duration)
            if any_droop_distracted and last_droop_pilot is not None:
                draw_droop_banner(annotated, last_droop_pilot, last_droop_duration,
                                  severity=last_droop_severity)
            for dr in droop_results:
                if not dr.drooping:
                    continue
                if any(ar.absent and ar.pilot_id == dr.pilot_id
                       for ar in absence_results):
                    cb = bbox_by_pid.get(dr.pilot_id)
                    _draw_distraction_label(annotated, cb, "SLEEPING / ABSENT",
                                            dr.timer_value, color=(0, 50, 200))
            draw_hud(annotated, video_time, raw_frame_no, len(results))

        #  Log + store violations
        if log_events:
            r_ref = next((r for r in results if r.distracted), None)
            conf  = r_ref.gadgets[0].confidence if (r_ref and r_ref.gadgets) else 0.9
            dur   = r_ref.timer_value if r_ref else 0.0
            event_time = max(0, video_time - GADGET_ALLOWED_DURATION)
            self.vstore.record_violation(
                annotated_frame=annotated, original_frame=frame,
                video_time=event_time, frame_index=raw_frame_no,
                event_type="phone_use", severity="CRITICAL",
                confidence=conf, risk_score=80, risk_level="CRITICAL",
                factors=["phone_use", "distraction"], duration=dur,
            )
            log_distraction(self.logger, event_time,
                            event="One of the pilots is using a mobile phone",
                            severity="CRITICAL", frame=annotated)

        if absence_log_events:
            ar_ref  = next((ar for ar in absence_results if ar.absent), None)
            dur_abs = ar_ref.timer_value if ar_ref else 0.0
            event_time = max(0, video_time - ABSENCE_ALLOWED_DURATION)
            self.vstore.record_violation(
                annotated_frame=annotated, original_frame=frame,
                video_time=event_time, frame_index=raw_frame_no,
                event_type="seat_absence", severity="CRITICAL",
                confidence=1.0, risk_score=70, risk_level="CRITICAL",
                factors=["seat_absence"], duration=dur_abs,
            )
            log_distraction(self.logger, event_time,
                            event="One of the pilots is away from the seat",
                            severity="CRITICAL", frame=annotated)

        if droop_log_events:
            severities  = [e[1] for e in droop_log_events]
            is_sleeping = any("SLEEPING" in s for s in severities)
            droop_pids  = {e[0] for e in droop_log_events}
            absent_pids = {ar.pilot_id for ar in absence_results if ar.absent}
            also_absent = bool(droop_pids & absent_pids)

            if also_absent:
                event_msg = "One of the pilots is sleeping / slumped in seat"
                etype     = "sleeping_absent"
            elif is_sleeping:
                event_msg = "One of the pilots is sleeping"
                etype     = "sleeping"
            else:
                event_msg = "One of the pilots is drowsy"
                etype     = "drowsy"

            dr_ref  = next((dr for dr in droop_results if dr.drooping), None)
            dur_drp = dr_ref.timer_value if dr_ref else 0.0
            event_time = max(0, video_time - HEAD_DROP_DURATION)
            self.vstore.record_violation(
                annotated_frame=annotated, original_frame=frame,
                video_time=event_time, frame_index=raw_frame_no,
                event_type=etype, severity="CRITICAL",
                confidence=0.9, risk_score=75, risk_level="HIGH",
                factors=["drowsy", "head_droop"], duration=dur_drp,
            )
            log_distraction(self.logger, event_time, event=event_msg,
                            severity="CRITICAL", frame=annotated)

        return annotated

    
    # HELPERS
    

    def _print_banner(self, fps: float, w: int, h: int, total: int) -> None:
        self.logger.info(
            f"\n{'='*60}\n"
            f"  LOCO PILOT DISTRACTION DETECTION\n"
            f"  Analysis ID : {self.analysis_id}\n"
            f"  Source      : {self.source}\n"
            f"  Video       : {w}x{h} @ {fps:.1f} fps ({total} frames)\n"
            f"  Output      : {OUTPUT_PATH if self.save else 'disabled'}\n"
            f"{'='*60}\n"
        )

    def _print_summary(self, frame_no: int, processing_time: float) -> None:
        self.logger.info(
            f"\n{'='*60}\n"
            f"  Processing complete\n"
            f"  Raw frames  : {frame_no}\n"
            f"  Processed   : {self._processed_frame_no} "
            f"(1 in every {RAW_FRAME_SKIP})\n"
            f"  Time        : {processing_time:.2f}s\n"
            f"  Report : outputs/{self.analysis_id}/analysis_report.json\n"
            f"  Frames : outputs/{self.analysis_id}/frames/\n"
            f"{'='*60}\n"
        )



# CLI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Loco Pilot Distraction Detection")
    p.add_argument("--source",          default=0,
                   help="Video file path or camera index (default: 0 = webcam)")
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
        source          = source,
        analysis_id     = args.analysis_id,
        train_detail_id = args.train_detail_id,
        save            = not args.no_save,
        display         = False,
    ).run()