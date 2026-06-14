# # # from __future__ import annotations

# # # import argparse
# # # import os
# # # import queue
# # # import re
# # # import sys
# # # import threading
# # # import traceback
# # # import uuid
# # # from typing import Optional
# # # from concurrent.futures import ThreadPoolExecutor
# # # import warnings
# # # import cv2
# # # import numpy as np


# # # DRAW           = False  # set True only for visual debug
# # # RAW_FRAME_SKIP = 3      # process 1 in every N raw frames
# # # GADGET_EVERY   = 6      # YOLO  every Nth processed frame
# # # ABSENCE_EVERY  = 4      # absence every Nth processed frame
# # # DROOP_EVERY    = 15     # droop every Nth processed frame
# # #  # allowed duration in seconds before logging violation


# # # from config.settings import OUTPUT_PATH, WINDOW_NAME, DISPLAY_SCALE,GADGET_ALLOWED_DURATION,ABSENCE_ALLOWED_DURATION,HEAD_DROP_DURATION
# # # from utils.logger import setup_logger, log_distraction, finalize_report
# # # from utils.violation_store import ViolationStore
# # # from utils.draw import (
# # #     draw_pilot_box, draw_gadget_box, draw_hud, draw_alert_banner,
# # #     draw_seat_zone, draw_absence_overlay, draw_absence_banner,
# # #     draw_droop_keypoints, draw_droop_overlay, draw_droop_banner,
# # #     draw_standing_label,
# # # )
# # # from detector.gadget_detector import GadgetDetector
# # # from detector.seat_absence_detector import SeatAbsenceDetector
# # # from detector.head_drop_detector import HeadDroopDetector

# # # _STOP = object()

# # # READ_QUEUE_MAXSIZE  = 8
# # # WRITE_QUEUE_MAXSIZE = 8

# # # warnings.filterwarnings("ignore", category=UserWarning)


# # # def _draw_distraction_label(
# # #     frame: np.ndarray,
# # #     bbox: tuple,
# # #     distraction_type: str,
# # #     timer_val: float,
# # #     color: tuple = (0, 0, 255),
# # # ) -> None:
# # #     if bbox is None:
# # #         return
# # #     x1, y1, x2, y2 = bbox
# # #     label = f"{distraction_type}  {timer_val:.1f}s"
# # #     font       = cv2.FONT_HERSHEY_DUPLEX
# # #     font_scale = 0.52
# # #     thickness  = 1
# # #     (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
# # #     pad    = 4
# # #     tag_y2 = max(y1, th + pad * 2)
# # #     tag_y1 = tag_y2 - th - pad * 2
# # #     tag_x2 = x1 + tw + pad * 2
# # #     cv2.rectangle(frame, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
# # #     cv2.putText(
# # #         frame, label,
# # #         (x1 + pad, tag_y2 - pad - baseline // 2),
# # #         font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
# # #     )


# # # class GadgetDetectionPipeline:

# # #     def __init__(
# # #         self,
# # #         source:          str | int,
# # #         analysis_id:     Optional[str] = None,
# # #         train_detail_id: int           = 0,
# # #         save:            bool          = False,
# # #         display:         bool          = False,
# # #         time_offset:     float         = 0.0,
# # #         frame_offset:    int           = 0,       # ← NEW: cumulative frame count before this video
# # #         shared_vstore=None,
# # #         original_filename: Optional[str] = None,
# # #     ) -> None:
# # #         self.source            = source
# # #         self.train_detail_id   = train_detail_id
# # #         self.save              = save
# # #         self.display           = display
# # #         self.time_offset       = time_offset      # cumulative seconds before this video
# # #         self.frame_offset      = frame_offset     # cumulative frames before this video
# # #         self.shared_vstore     = shared_vstore    # if set, use this instead of creating new one
# # #         self.original_filename = original_filename  # real upload name, overrides tmp path basename

# # #         if analysis_id:
# # #             self.analysis_id = analysis_id
# # #         elif (
# # #             isinstance(source, str)
# # #             and source not in ("0",)
# # #             and os.path.isfile(source)
# # #         ):
# # #             stem             = os.path.splitext(os.path.basename(source))[0]
# # #             self.analysis_id = re.sub(r"[^A-Za-z0-9_-]", "_", stem)
# # #         else:
# # #             self.analysis_id = uuid.uuid4().hex[:8]

# # #         self.logger           = setup_logger()
# # #         self.detector         = GadgetDetector()
# # #         self.absence_detector = SeatAbsenceDetector()
# # #         self.droop_detector   = HeadDroopDetector()
# # #         self._writer:  Optional[cv2.VideoWriter] = None
# # #         self.vstore:   Optional[ViolationStore]  = None

# # #         # 3 workers: one per detector, no excess overhead
# # #         self.executor = ThreadPoolExecutor(max_workers=3)

# # #         self._prev_pilot_boxes      = []
# # #         self._prev_frame_detections = None
# # #         self._processed_frame_no    = 0   

# # #         self._read_queue:  queue.Queue = queue.Queue(maxsize=READ_QUEUE_MAXSIZE)
# # #         self._write_queue: queue.Queue = queue.Queue(maxsize=WRITE_QUEUE_MAXSIZE)

    
# # #     # ENTRY POINT
    

# # #     def run(self) -> tuple:
# # #         """Returns (report_path, duration_seconds, total_frame_count)."""
# # #         import time
# # #         start_time = time.time()

# # #         cap = cv2.VideoCapture(self.source)
# # #         if not cap.isOpened():
# # #             self.logger.error(f"Cannot open source: {self.source!r}")
# # #             sys.exit(1)

# # #         _raw_fps = cap.get(cv2.CAP_PROP_FPS)
# # #         if not _raw_fps:
# # #             print("[WARNING] FPS not detected — defaulting to 25.0")
# # #         fps    = _raw_fps or 25.0
# # #         width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# # #         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# # #         total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# # #         print(f"Video      : {width}x{height} @ {fps:.1f}fps  "
# # #               f"{total/fps:.1f}s  {total} frames")
# # #         print(f"Analysis ID: {self.analysis_id}")
# # #         print(f"Processing : every {RAW_FRAME_SKIP}rd raw frame  |  "
# # #               f"YOLO every {RAW_FRAME_SKIP * GADGET_EVERY} raw frames  |  "
# # #               f"Droop every {RAW_FRAME_SKIP * DROOP_EVERY} raw frames")

# # #         source_str  = str(self.source)
# # #         # Prefer the real uploaded filename passed in from api.py;
# # #         # fall back to basename of the (temp) path for direct/CLI use.
# # #         source_name = (
# # #             self.original_filename
# # #             if self.original_filename
# # #             else (os.path.basename(source_str) if isinstance(self.source, str) else "webcam")
# # #         )
# # #         # Seek to end to get true duration — handles VFR and mismatched fps tags
# # #         # (e.g. cabin_video.mp4 has container fps=30 but actual fps=6)
# # #         cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
# # #         duration_s = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 3)
# # #         cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # rewind for processing
# # #         if duration_s <= 0:
# # #             # Fallback for sources that don't support seek-to-end
# # #             duration_s = round(total / fps, 3) if total > 0 and fps > 0 else 0.0
# # #         h, m, s    = (
# # #             int(duration_s) // 3600,
# # #             (int(duration_s) % 3600) // 60,
# # #             int(duration_s) % 60,
# # #         )
# # #         size_mb = (
# # #             round(os.path.getsize(source_str) / 1_000_000, 2)
# # #             if isinstance(self.source, str) and os.path.isfile(source_str) else 0
# # #         )

# # #         video_info = {
# # #             "filename":          source_name,
# # #             "videoPath":         source_str,
# # #             "durationSeconds":   duration_s,
# # #             "durationFormatted": f"{h}:{m:02d}:{s:02d}",
# # #             "resolution":        f"{width}x{height}",
# # #             "fps":               round(fps, 3),
# # #             "totalFrames":       total,
# # #             "sizeMb":            size_mb,
# # #         }

# # #         if self.shared_vstore is not None:
# # #             # Batch mode: attach this video's metadata to the shared store
# # #             self.vstore = self.shared_vstore
# # #             self.vstore.add_video_info(video_info)
# # #         else:
# # #             self.vstore = ViolationStore(
# # #                 analysis_id     = self.analysis_id,
# # #                 train_detail_id = self.train_detail_id,
# # #                 video_info      = video_info,
# # #             )
# # #         self._print_banner(fps, width, height, total)

# # #         if self.save:
# # #             os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
# # #             self._writer = cv2.VideoWriter(
# # #                 OUTPUT_PATH,
# # #                 cv2.VideoWriter_fourcc(*"mp4v"),
# # #                 fps,
# # #                 (width, height),
# # #             )

# # #         raw_frame_no = 0
# # #         report_path  = ""

# # #         reader_thread = threading.Thread(
# # #             target=self._reader_loop, args=(cap,),
# # #             daemon=True, name="FrameReader",
# # #         )
# # #         writer_thread = threading.Thread(
# # #             target=self._writer_loop,
# # #             daemon=True, name="FrameWriter",
# # #         )
# # #         reader_thread.start()
# # #         writer_thread.start()

# # #         try:
# # #             while True:
# # #                 item = self._read_queue.get()
# # #                 if item is _STOP:
# # #                     break

# # #                 raw_frame, raw_frame_no, video_time = item

# # #                 # ── Skip most raw frames — pass through as-is ─────
# # #                 # raw_frame_no here is already globally offset so the
# # #                 # modulo cadence is kept consistent across videos.
# # #                 if raw_frame_no % RAW_FRAME_SKIP != 0:
# # #                     self._write_queue.put(raw_frame)
# # #                     continue

# # #                 # ── Process this frame ────────────────────────────
# # #                 self._processed_frame_no += 1
# # #                 annotated = self._process_frame(
# # #                     raw_frame, video_time, raw_frame_no, self._processed_frame_no
# # #                 )
# # #                 self._write_queue.put(annotated)

# # #                 if self.display:
# # #                     show = annotated
# # #                     if DISPLAY_SCALE != 1.0:
# # #                         show = cv2.resize(
# # #                             annotated,
# # #                             (int(width * DISPLAY_SCALE), int(height * DISPLAY_SCALE)),
# # #                         )
# # #                     cv2.imshow(WINDOW_NAME, show)
# # #                     key = cv2.waitKey(1) & 0xFF
# # #                     if key in (ord("q"), 27):
# # #                         self.logger.info("Quit by user.")
# # #                         break

# # #         except KeyboardInterrupt:
# # #             self.logger.info("\nInterrupted by user.")
# # #         except Exception:
# # #             self.logger.error("Unexpected error:\n" + traceback.format_exc())
# # #         finally:
# # #             self._write_queue.put(_STOP)
# # #             writer_thread.join(timeout=30)
# # #             cap.release()
# # #             if self._writer:
# # #                 self._writer.release()
# # #             if self.display:
# # #                 cv2.destroyAllWindows()

# # #             processing_time = round(time.time() - start_time, 3)
# # #             self._print_summary(raw_frame_no, processing_time)
# # #             finalize_report()
# # #             # In batch mode (shared_vstore) we do NOT finalize here —
# # #             # api.py finalizes the shared store once after ALL videos are done.
# # #             if self.shared_vstore is None:
# # #                 report_path = self.vstore.finalize(processing_time=processing_time)
# # #             else:
# # #                 report_path = ""   # will be set by api.py after last video

# # #         actual_fps = raw_frame_no / processing_time if processing_time > 0 else 0
# # #         print(f"\nTotal Time : {processing_time:.2f}s   FPS : {actual_fps:.2f}")
# # #         # Return duration_s and total so api.py can accumulate offsets
# # #         # without re-opening the (already-deleted) temp file.
# # #         return report_path, duration_s, total

 
# # #     # READER THREAD
   

# # #     def _reader_loop(self, cap: cv2.VideoCapture) -> None:
# # #         # frame_no counts frames within THIS video (1-based).
# # #         # We add self.frame_offset so every frame has a globally unique
# # #         # index across the entire batch — prevents deduplication collisions
# # #         # in ViolationStore._seen_frames when two videos share the same
# # #         # local frame numbers.
# # #         frame_no = 0
# # #         try:
# # #             while True:
# # #                 ret, frame = cap.read()
# # #                 if not ret:
# # #                     break
# # #                 frame_no  += 1
# # #                 global_frame_no = frame_no + self.frame_offset          # ← CHANGED
# # #                 video_time      = (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0) + self.time_offset
# # #                 self._read_queue.put((frame, global_frame_no, video_time))
# # #         except Exception:
# # #             self.logger.error("Reader error:\n" + traceback.format_exc())
# # #         finally:
# # #             self._read_queue.put(_STOP)

    
# # #     # WRITER THREAD
    

# # #     def _writer_loop(self) -> None:
# # #         try:
# # #             while True:
# # #                 item = self._write_queue.get()
# # #                 if item is _STOP:
# # #                     break
# # #                 if self._writer:
# # #                     self._writer.write(item)
# # #         except Exception:
# # #             self.logger.error("Writer error:\n" + traceback.format_exc())

    
# # #     # PER-FRAME PROCESSING
    

# # #     def _process_frame(
# # #         self,
# # #         frame:              np.ndarray,
# # #         video_time:         float,
# # #         raw_frame_no:       int,
# # #         processed_frame_no: int,
# # #     ) -> np.ndarray:
# # #         annotated = frame

# # #         # Cadences are relative to processed_frame_no so that the effective ML frequency is consistent regardless of RAW_FRAME_SKIP.
# # #         run_gadget  = (processed_frame_no % GADGET_EVERY  == 0)
# # #         run_absence = (processed_frame_no % ABSENCE_EVERY == 0)
# # #         run_droop   = (processed_frame_no % DROOP_EVERY   == 0)

# # #         prev_pilot_boxes     = self._prev_pilot_boxes
# # #         prev_frame_detection = self._prev_frame_detections

# # #         future_gadget = (
# # #             self.executor.submit(self.detector.process, frame, round(video_time, 0))
# # #             if run_gadget else None
# # #         )
# # #         future_absence = (
# # #             self.executor.submit(
# # #                 self.absence_detector.process,
# # #                 prev_pilot_boxes, video_time, frame.shape[1], frame.shape[0],
# # #             )
# # #             if run_absence else None
# # #         )
# # #         future_droop = (
# # #             self.executor.submit(
# # #                 self.droop_detector.process,
# # #                 frame, video_time, prev_frame_detection,
# # #             )
# # #             if run_droop else None
# # #         )

# # #         results,         log_events         = [], []
# # #         absence_results, absence_log_events = [], []
# # #         droop_results,   droop_log_events   = [], []

# # #         try:
# # #             if future_gadget is not None:
# # #                 results, log_events = future_gadget.result()
# # #         except Exception as exc:
# # #             self.logger.error(f"Gadget error frame {raw_frame_no}: {exc}", exc_info=True)

# # #         try:
# # #             if future_absence is not None:
# # #                 absence_results, absence_log_events = future_absence.result()
# # #         except Exception as exc:
# # #             self.logger.error(f"Absence error frame {raw_frame_no}: {exc}", exc_info=True)

# # #         try:
# # #             if future_droop is not None:
# # #                 droop_results, droop_log_events = future_droop.result()
# # #         except Exception as exc:
# # #             self.logger.error(f"Droop error frame {raw_frame_no}: {exc}", exc_info=True)

# # #         if run_gadget:
# # #             self._prev_pilot_boxes      = [(r.pilot_id, r.bbox) for r in results]
# # #             self._prev_frame_detections = self.detector.last_frame_detections

# # #         #  Draw (skipped entirely when DRAW=False
# # #         if DRAW:
# # #             for g in self.detector.last_gadget_hits:
# # #                 draw_gadget_box(annotated, g.bbox, g.class_name, g.confidence)
# # #             for ar in absence_results:
# # #                 if ar.calibrated and ar.seat_zone is not None:
# # #                     draw_seat_zone(annotated, ar.seat_zone, ar.pilot_id)

# # #         any_gadget_distracted = False
# # #         last_gadget_pilot     = None
# # #         last_gadget_name      = ""

# # #         for r in results:
# # #             gadget_names = [g.class_name for g in r.gadgets]
# # #             if DRAW:
# # #                 draw_pilot_box(annotated, r.bbox, r.pilot_id, r.distracted, [])
# # #             if r.distracted:
# # #                 any_gadget_distracted = True
# # #                 last_gadget_pilot     = r.pilot_id
# # #                 last_gadget_name      = gadget_names[0] if gadget_names else "gadget"
# # #                 if DRAW:
# # #                     _draw_distraction_label(annotated, r.bbox, "Phone Usage",
# # #                                             r.timer_value, color=(0, 0, 220))

# # #         any_absence_distracted = False
# # #         last_absent_pilot      = None
# # #         last_absent_duration   = 0.0

# # #         for ar in absence_results:
# # #             current_bbox = next(
# # #                 (r.bbox for r in results if r.pilot_id == ar.pilot_id), None
# # #             )
# # #             if DRAW:
# # #                 draw_absence_overlay(
# # #                     frame=annotated, bbox=current_bbox, pilot_id=ar.pilot_id,
# # #                     absent=ar.absent, timer_val=ar.timer_value,
# # #                     calibrated=ar.calibrated,
# # #                 )
# # #             if ar.absent:
# # #                 any_absence_distracted = True
# # #                 last_absent_pilot      = ar.pilot_id
# # #                 last_absent_duration   = ar.timer_value
# # #                 if DRAW:
# # #                     _draw_distraction_label(annotated, current_bbox, "Away From Seat",
# # #                                             ar.timer_value, color=(0, 140, 255))

# # #         any_droop_distracted = False
# # #         last_droop_pilot     = None
# # #         last_droop_duration  = 0.0
# # #         last_droop_severity  = "DROWSINESS"
# # #         bbox_by_pid          = {}

# # #         if droop_results:
# # #             bbox_by_pid = {r.pilot_id: r.bbox for r in results}

# # #         for dr in droop_results:
# # #             current_bbox = bbox_by_pid.get(dr.pilot_id)
# # #             if not dr.is_seated:
# # #                 if DRAW:
# # #                     draw_standing_label(annotated, dr.pilot_id, current_bbox)
# # #                 continue
# # #             if hasattr(dr, "keypoints") and dr.keypoints:
# # #                 if DRAW:
# # #                     draw_droop_keypoints(
# # #                         frame=annotated, keypoints=dr.keypoints,
# # #                         pilot_id=dr.pilot_id, drooping=dr.drooping,
# # #                         angle=getattr(dr, "angle", 0.0),
# # #                     )
# # #             if DRAW:
# # #                 draw_droop_overlay(
# # #                     frame=annotated, pilot_id=dr.pilot_id,
# # #                     drooping=dr.drooping, timer_val=dr.timer_value,
# # #                     bbox=current_bbox,
# # #                     severity=getattr(dr, "severity", "DROWSINESS"),
# # #                 )
# # #             if dr.drooping:
# # #                 any_droop_distracted = True
# # #                 last_droop_pilot     = dr.pilot_id
# # #                 last_droop_duration  = dr.timer_value
# # #                 last_droop_severity  = getattr(dr, "severity", "DROWSINESS")
# # #                 display_secs         = dr.timer_value * (38 / 25.0)
# # #                 if DRAW:
# # #                     _draw_distraction_label(
# # #                         annotated, current_bbox, last_droop_severity, display_secs,
# # #                         color=(0, 200, 255)
# # #                         if last_droop_severity == "DROWSINESS" else (0, 80, 200),
# # #                     )

# # #         if DRAW:
# # #             if any_gadget_distracted and last_gadget_pilot is not None:
# # #                 draw_alert_banner(annotated, last_gadget_pilot, last_gadget_name)
# # #             if any_absence_distracted and last_absent_pilot is not None:
# # #                 draw_absence_banner(annotated, last_absent_pilot, last_absent_duration)
# # #             if any_droop_distracted and last_droop_pilot is not None:
# # #                 draw_droop_banner(annotated, last_droop_pilot, last_droop_duration,
# # #                                   severity=last_droop_severity)
# # #             for dr in droop_results:
# # #                 if not dr.drooping:
# # #                     continue
# # #                 if any(ar.absent and ar.pilot_id == dr.pilot_id
# # #                        for ar in absence_results):
# # #                     cb = bbox_by_pid.get(dr.pilot_id)
# # #                     _draw_distraction_label(annotated, cb, "SLEEPING / ABSENT",
# # #                                             dr.timer_value, color=(0, 50, 200))
# # #             draw_hud(annotated, video_time, raw_frame_no, len(results))

# # #         #  Log + store violations
# # #         if log_events:
# # #             r_ref = next((r for r in results if r.distracted), None)
# # #             conf  = r_ref.gadgets[0].confidence if (r_ref and r_ref.gadgets) else 0.9
# # #             dur   = r_ref.timer_value if r_ref else 0.0
# # #             # Clamp to time_offset floor so we never produce a timestamp
# # #             # earlier than the start of this video in the combined timeline.
# # #             event_time = max(self.time_offset, video_time - GADGET_ALLOWED_DURATION)  # ← CHANGED
# # #             self.vstore.record_violation(
# # #                 annotated_frame=annotated, original_frame=frame,
# # #                 video_time=event_time, frame_index=raw_frame_no,
# # #                 event_type="phone_use", severity="CRITICAL",
# # #                 confidence=conf, risk_score=80, risk_level="CRITICAL",
# # #                 factors=["phone_use"], duration=dur,
# # #             )
# # #             log_distraction(self.logger, event_time,
# # #                             event="One of the pilots is using a mobile phone",
# # #                             severity="CRITICAL", frame=annotated)

# # #         if absence_log_events:
# # #             ar_ref  = next((ar for ar in absence_results if ar.absent), None)
# # #             dur_abs = ar_ref.timer_value if ar_ref else 0.0
# # #             event_time = max(self.time_offset, video_time - ABSENCE_ALLOWED_DURATION)  # ← CHANGED
# # #             self.vstore.record_violation(
# # #                 annotated_frame=annotated, original_frame=frame,
# # #                 video_time=event_time, frame_index=raw_frame_no,
# # #                 event_type="seat_absence", severity="CRITICAL",
# # #                 confidence=1.0, risk_score=70, risk_level="CRITICAL",
# # #                 factors=["seat_absence"], duration=dur_abs,
# # #             )
# # #             log_distraction(self.logger, event_time,
# # #                             event="One of the pilots is away from the seat",
# # #                             severity="CRITICAL", frame=annotated)

# # #         if droop_log_events:
# # #             severities  = [e[1] for e in droop_log_events]
# # #             is_sleeping = any("SLEEPING" in s for s in severities)
# # #             droop_pids  = {e[0] for e in droop_log_events}
# # #             absent_pids = {ar.pilot_id for ar in absence_results if ar.absent}
# # #             also_absent = bool(droop_pids & absent_pids)

# # #             if also_absent:
# # #                 event_msg = "One of the pilots is sleeping / slumped in seat"
# # #                 etype     = "sleeping_absent"
# # #             elif is_sleeping:
# # #                 event_msg = "One of the pilots is sleeping"
# # #                 etype     = "sleeping"
# # #             else:
# # #                 event_msg = "One of the pilots is drowsy"
# # #                 etype     = "drowsy"

# # #             dr_ref  = next((dr for dr in droop_results if dr.drooping), None)
# # #             dur_drp = dr_ref.timer_value if dr_ref else 0.0
# # #             event_time = max(self.time_offset, video_time - HEAD_DROP_DURATION)  # ← CHANGED
# # #             self.vstore.record_violation(
# # #                 annotated_frame=annotated, original_frame=frame,
# # #                 video_time=event_time, frame_index=raw_frame_no,
# # #                 event_type=etype, severity="CRITICAL",
# # #                 confidence=0.9, risk_score=75, risk_level="HIGH",
# # #                 factors=["drowsy", "head_droop"], duration=dur_drp,
# # #             )
# # #             log_distraction(self.logger, event_time, event=event_msg,
# # #                             severity="CRITICAL", frame=annotated)

# # #         return annotated

    
# # #     # HELPERS
    

# # #     def _print_banner(self, fps: float, w: int, h: int, total: int) -> None:
# # #         self.logger.info(
# # #             f"\n{'='*60}\n"
# # #             f"  LOCO PILOT DISTRACTION DETECTION\n"
# # #             f"  Analysis ID : {self.analysis_id}\n"
# # #             f"  Source      : {self.source}\n"
# # #             f"  Video       : {w}x{h} @ {fps:.1f} fps ({total} frames)\n"
# # #             f"  Output      : {OUTPUT_PATH if self.save else 'disabled'}\n"
# # #             f"{'='*60}\n"
# # #         )

# # #     def _print_summary(self, frame_no: int, processing_time: float) -> None:
# # #         self.logger.info(
# # #             f"\n{'='*60}\n"
# # #             f"  Processing complete\n"
# # #             f"  Raw frames  : {frame_no}\n"
# # #             f"  Processed   : {self._processed_frame_no} "
# # #             f"(1 in every {RAW_FRAME_SKIP})\n"
# # #             f"  Time        : {processing_time:.2f}s\n"
# # #             f"  Report : outputs/{self.analysis_id}/analysis_report.json\n"
# # #             f"  Frames : outputs/{self.analysis_id}/frames/\n"
# # #             f"{'='*60}\n"
# # #         )



# # # # CLI


# # # def parse_args() -> argparse.Namespace:
# # #     p = argparse.ArgumentParser(description="Loco Pilot Distraction Detection")
# # #     p.add_argument("--source",          default=0,
# # #                    help="Video file path or camera index (default: 0 = webcam)")
# # #     p.add_argument("--analysis-id",     default=None)
# # #     p.add_argument("--train-detail-id", default=0, type=int)
# # #     p.add_argument("--no-display",      action="store_true")
# # #     p.add_argument("--no-save",         action="store_true")
# # #     return p.parse_args()


# # # if __name__ == "__main__":
# # #     args   = parse_args()
# # #     source = args.source
# # #     if isinstance(source, str) and source.isdigit():
# # #         source = int(source)

# # #     GadgetDetectionPipeline(
# # #         source          = source,
# # #         analysis_id     = args.analysis_id,
# # #         train_detail_id = args.train_detail_id,
# # #         save            = not args.no_save,
# # #         display         = False,
# # #     ).run()



# # from __future__ import annotations

# # import argparse
# # import os
# # import queue
# # import re
# # import sys
# # import threading
# # import traceback
# # import uuid
# # from typing import Optional
# # from concurrent.futures import ThreadPoolExecutor
# # import warnings
# # import cv2
# # import numpy as np


# # DRAW           = False  # set True only for visual debug
# # RAW_FRAME_SKIP = 3      # process 1 in every N raw frames
# # GADGET_EVERY   = 6      # YOLO  every Nth processed frame
# # ABSENCE_EVERY  = 4      # absence every Nth processed frame
# # DROOP_EVERY    = 15     # droop every Nth processed frame


# # from config.settings import (
# #     OUTPUT_PATH, WINDOW_NAME, DISPLAY_SCALE,
# #     GADGET_ALLOWED_DURATION, ABSENCE_ALLOWED_DURATION, HEAD_DROP_DURATION,
# # )
# # from utils.logger import setup_logger, log_distraction, finalize_report
# # from utils.violation_store import ViolationStore
# # from utils.draw import (
# #     draw_pilot_box, draw_gadget_box, draw_hud, draw_alert_banner,
# #     draw_seat_zone, draw_absence_overlay, draw_absence_banner,
# #     draw_droop_keypoints, draw_droop_overlay, draw_droop_banner,
# #     draw_standing_label,
# # )
# # from detector.gadget_detector import GadgetDetector
# # from detector.seat_absence_detector import SeatAbsenceDetector
# # from detector.head_drop_detector import HeadDroopDetector

# # _STOP = object()

# # READ_QUEUE_MAXSIZE  = 8
# # WRITE_QUEUE_MAXSIZE = 8

# # warnings.filterwarnings("ignore", category=UserWarning)


# # def _draw_distraction_label(
# #     frame: np.ndarray,
# #     bbox: tuple,
# #     distraction_type: str,
# #     timer_val: float,
# #     color: tuple = (0, 0, 255),
# # ) -> None:
# #     if bbox is None:
# #         return
# #     x1, y1, x2, y2 = bbox
# #     label = f"{distraction_type}  {timer_val:.1f}s"
# #     font       = cv2.FONT_HERSHEY_DUPLEX
# #     font_scale = 0.52
# #     thickness  = 1
# #     (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
# #     pad    = 4
# #     tag_y2 = max(y1, th + pad * 2)
# #     tag_y1 = tag_y2 - th - pad * 2
# #     tag_x2 = x1 + tw + pad * 2
# #     cv2.rectangle(frame, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
# #     cv2.putText(
# #         frame, label,
# #         (x1 + pad, tag_y2 - pad - baseline // 2),
# #         font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
# #     )


# # class GadgetDetectionPipeline:

# #     def __init__(
# #         self,
# #         source:            str | int,
# #         analysis_id:       Optional[str] = None,
# #         train_detail_id:   int           = 0,
# #         save:              bool          = False,
# #         display:           bool          = False,
# #         time_offset:       float         = 0.0,
# #         frame_offset:      int           = 0,
# #         shared_vstore                    = None,
# #         original_filename: Optional[str] = None,
# #     ) -> None:
# #         self.source            = source
# #         self.train_detail_id   = train_detail_id
# #         self.save              = save
# #         self.display           = display
# #         self.time_offset       = time_offset       # cumulative seconds before this video
# #         self.frame_offset      = frame_offset      # cumulative frames before this video
# #         self.shared_vstore     = shared_vstore
# #         self.original_filename = original_filename # real upload filename, not the tmp path

# #         if analysis_id:
# #             self.analysis_id = analysis_id
# #         elif (
# #             isinstance(source, str)
# #             and source not in ("0",)
# #             and os.path.isfile(source)
# #         ):
# #             stem             = os.path.splitext(os.path.basename(source))[0]
# #             self.analysis_id = re.sub(r"[^A-Za-z0-9_-]", "_", stem)
# #         else:
# #             self.analysis_id = uuid.uuid4().hex[:8]

# #         self.logger           = setup_logger()
# #         self.detector         = GadgetDetector()
# #         self.absence_detector = SeatAbsenceDetector()
# #         self.droop_detector   = HeadDroopDetector()
# #         self._writer: Optional[cv2.VideoWriter] = None
# #         self.vstore: Optional[ViolationStore]   = None

# #         self.executor = ThreadPoolExecutor(max_workers=3)

# #         self._prev_pilot_boxes      = []
# #         self._prev_frame_detections = None
# #         self._processed_frame_no    = 0

# #         self._read_queue:  queue.Queue = queue.Queue(maxsize=READ_QUEUE_MAXSIZE)
# #         self._write_queue: queue.Queue = queue.Queue(maxsize=WRITE_QUEUE_MAXSIZE)


# #     # ── ENTRY POINT ───────────────────────────────────────────────

# #     def run(self) -> tuple:
# #         """Returns (report_path, duration_seconds, total_frame_count)."""
# #         import time
# #         start_time = time.time()

# #         cap = cv2.VideoCapture(self.source)
# #         if not cap.isOpened():
# #             self.logger.error(f"Cannot open source: {self.source!r}")
# #             sys.exit(1)

# #         _raw_fps = cap.get(cv2.CAP_PROP_FPS)
# #         if not _raw_fps:
# #             print("[WARNING] FPS not detected — defaulting to 25.0")
# #         fps    = _raw_fps or 25.0
# #         width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# #         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# #         total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# #         print(f"Video      : {width}x{height} @ {fps:.1f}fps  "
# #               f"{total/fps:.1f}s  {total} frames")
# #         print(f"Analysis ID: {self.analysis_id}")
# #         print(f"Processing : every {RAW_FRAME_SKIP}rd raw frame  |  "
# #               f"YOLO every {RAW_FRAME_SKIP * GADGET_EVERY} raw frames  |  "
# #               f"Droop every {RAW_FRAME_SKIP * DROOP_EVERY} raw frames")

# #         source_str  = str(self.source)
# #         # Use real uploaded filename if provided; fall back to temp file basename
# #         source_name = (
# #             self.original_filename
# #             if self.original_filename
# #             else (os.path.basename(source_str) if isinstance(self.source, str) else "webcam")
# #         )

# #         # Seek to end for accurate duration (handles VFR / mismatched fps tags)
# #         cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
# #         duration_s = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 3)
# #         cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # rewind
# #         if duration_s <= 0:
# #             duration_s = round(total / fps, 3) if total > 0 and fps > 0 else 0.0

# #         h = int(duration_s) // 3600
# #         m = (int(duration_s) % 3600) // 60
# #         s = int(duration_s) % 60
# #         size_mb = (
# #             round(os.path.getsize(source_str) / 1_000_000, 2)
# #             if isinstance(self.source, str) and os.path.isfile(source_str) else 0
# #         )

# #         video_info = {
# #             "filename":          source_name,
# #             "videoPath":         source_str,
# #             "durationSeconds":   duration_s,
# #             "durationFormatted": f"{h}:{m:02d}:{s:02d}",
# #             "resolution":        f"{width}x{height}",
# #             "fps":               round(fps, 3),
# #             "totalFrames":       total,
# #             "sizeMb":            size_mb,
# #         }

# #         if self.shared_vstore is not None:
# #             # Batch mode: attach this video's metadata to the shared store
# #             self.vstore = self.shared_vstore
# #             self.vstore.add_video_info(video_info)
# #         else:
# #             self.vstore = ViolationStore(
# #                 analysis_id     = self.analysis_id,
# #                 train_detail_id = self.train_detail_id,
# #                 video_info      = video_info,
# #             )

# #         self._print_banner(fps, width, height, total)

# #         if self.save:
# #             os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
# #             self._writer = cv2.VideoWriter(
# #                 OUTPUT_PATH,
# #                 cv2.VideoWriter_fourcc(*"mp4v"),
# #                 fps,
# #                 (width, height),
# #             )

# #         raw_frame_no = 0
# #         report_path  = ""

# #         reader_thread = threading.Thread(
# #             target=self._reader_loop, args=(cap,),
# #             daemon=True, name="FrameReader",
# #         )
# #         writer_thread = threading.Thread(
# #             target=self._writer_loop,
# #             daemon=True, name="FrameWriter",
# #         )
# #         reader_thread.start()
# #         writer_thread.start()

# #         try:
# #             while True:
# #                 item = self._read_queue.get()
# #                 if item is _STOP:
# #                     break

# #                 raw_frame, raw_frame_no, video_time = item

# #                 if raw_frame_no % RAW_FRAME_SKIP != 0:
# #                     self._write_queue.put(raw_frame)
# #                     continue

# #                 self._processed_frame_no += 1
# #                 annotated = self._process_frame(
# #                     raw_frame, video_time, raw_frame_no, self._processed_frame_no
# #                 )
# #                 self._write_queue.put(annotated)

# #                 if self.display:
# #                     show = annotated
# #                     if DISPLAY_SCALE != 1.0:
# #                         show = cv2.resize(
# #                             annotated,
# #                             (int(width * DISPLAY_SCALE), int(height * DISPLAY_SCALE)),
# #                         )
# #                     cv2.imshow(WINDOW_NAME, show)
# #                     key = cv2.waitKey(1) & 0xFF
# #                     if key in (ord("q"), 27):
# #                         self.logger.info("Quit by user.")
# #                         break

# #         except KeyboardInterrupt:
# #             self.logger.info("\nInterrupted by user.")
# #         except Exception:
# #             self.logger.error("Unexpected error:\n" + traceback.format_exc())
# #         finally:
# #             self._write_queue.put(_STOP)
# #             writer_thread.join(timeout=30)
# #             cap.release()
# #             if self._writer:
# #                 self._writer.release()
# #             if self.display:
# #                 cv2.destroyAllWindows()

# #             processing_time = round(time.time() - start_time, 3)
# #             self._print_summary(raw_frame_no, processing_time)
# #             finalize_report()

# #             # In batch mode the caller (api.py) finalizes after ALL videos are done
# #             if self.shared_vstore is None:
# #                 report_path = self.vstore.finalize(processing_time=processing_time)
# #             else:
# #                 report_path = ""

# #         actual_fps = raw_frame_no / processing_time if processing_time > 0 else 0
# #         print(f"\nTotal Time : {processing_time:.2f}s   FPS : {actual_fps:.2f}")
# #         return report_path, duration_s, total


# #     # ── READER THREAD ─────────────────────────────────────────────

# #     def _reader_loop(self, cap: cv2.VideoCapture) -> None:
# #         # frame_no counts frames within THIS video (1-based).
# #         # Adding self.frame_offset gives every frame a globally unique
# #         # index across the entire batch — prevents dedup collisions in
# #         # ViolationStore._seen_frames when two videos share local frame numbers.
# #         frame_no = 0
# #         try:
# #             while True:
# #                 ret, frame = cap.read()
# #                 if not ret:
# #                     break
# #                 frame_no        += 1
# #                 global_frame_no  = frame_no + self.frame_offset
# #                 video_time       = (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0) + self.time_offset
# #                 self._read_queue.put((frame, global_frame_no, video_time))
# #         except Exception:
# #             self.logger.error("Reader error:\n" + traceback.format_exc())
# #         finally:
# #             self._read_queue.put(_STOP)


# #     # ── WRITER THREAD ─────────────────────────────────────────────

# #     def _writer_loop(self) -> None:
# #         try:
# #             while True:
# #                 item = self._write_queue.get()
# #                 if item is _STOP:
# #                     break
# #                 if self._writer:
# #                     self._writer.write(item)
# #         except Exception:
# #             self.logger.error("Writer error:\n" + traceback.format_exc())


# #     # ── PER-FRAME PROCESSING ──────────────────────────────────────

# #     def _process_frame(
# #         self,
# #         frame:              np.ndarray,
# #         video_time:         float,
# #         raw_frame_no:       int,
# #         processed_frame_no: int,
# #     ) -> np.ndarray:
# #         annotated = frame

# #         run_gadget  = (processed_frame_no % GADGET_EVERY  == 0)
# #         run_absence = (processed_frame_no % ABSENCE_EVERY == 0)
# #         run_droop   = (processed_frame_no % DROOP_EVERY   == 0)

# #         prev_pilot_boxes     = self._prev_pilot_boxes
# #         prev_frame_detection = self._prev_frame_detections

# #         future_gadget = (
# #             self.executor.submit(self.detector.process, frame, round(video_time, 0))
# #             if run_gadget else None
# #         )
# #         future_absence = (
# #             self.executor.submit(
# #                 self.absence_detector.process,
# #                 prev_pilot_boxes, video_time, frame.shape[1], frame.shape[0],
# #             )
# #             if run_absence else None
# #         )
# #         future_droop = (
# #             self.executor.submit(
# #                 self.droop_detector.process,
# #                 frame, video_time, prev_frame_detection,
# #             )
# #             if run_droop else None
# #         )

# #         results,         log_events         = [], []
# #         absence_results, absence_log_events = [], []
# #         droop_results,   droop_log_events   = [], []

# #         try:
# #             if future_gadget is not None:
# #                 results, log_events = future_gadget.result()
# #         except Exception as exc:
# #             self.logger.error(f"Gadget error frame {raw_frame_no}: {exc}", exc_info=True)

# #         try:
# #             if future_absence is not None:
# #                 absence_results, absence_log_events = future_absence.result()
# #         except Exception as exc:
# #             self.logger.error(f"Absence error frame {raw_frame_no}: {exc}", exc_info=True)

# #         try:
# #             if future_droop is not None:
# #                 droop_results, droop_log_events = future_droop.result()
# #         except Exception as exc:
# #             self.logger.error(f"Droop error frame {raw_frame_no}: {exc}", exc_info=True)

# #         if run_gadget:
# #             self._prev_pilot_boxes      = [(r.pilot_id, r.bbox) for r in results]
# #             self._prev_frame_detections = self.detector.last_frame_detections

# #         # ── Draw (skipped entirely when DRAW=False) ───────────────
# #         if DRAW:
# #             for g in self.detector.last_gadget_hits:
# #                 draw_gadget_box(annotated, g.bbox, g.class_name, g.confidence)
# #             for ar in absence_results:
# #                 if ar.calibrated and ar.seat_zone is not None:
# #                     draw_seat_zone(annotated, ar.seat_zone, ar.pilot_id)

# #         any_gadget_distracted = False
# #         last_gadget_pilot     = None
# #         last_gadget_name      = ""

# #         for r in results:
# #             gadget_names = [g.class_name for g in r.gadgets]
# #             if DRAW:
# #                 draw_pilot_box(annotated, r.bbox, r.pilot_id, r.distracted, [])
# #             if r.distracted:
# #                 any_gadget_distracted = True
# #                 last_gadget_pilot     = r.pilot_id
# #                 last_gadget_name      = gadget_names[0] if gadget_names else "gadget"
# #                 if DRAW:
# #                     _draw_distraction_label(annotated, r.bbox, "Phone Usage",
# #                                             r.timer_value, color=(0, 0, 220))

# #         any_absence_distracted = False
# #         last_absent_pilot      = None
# #         last_absent_duration   = 0.0

# #         for ar in absence_results:
# #             current_bbox = next(
# #                 (r.bbox for r in results if r.pilot_id == ar.pilot_id), None
# #             )
# #             if DRAW:
# #                 draw_absence_overlay(
# #                     frame=annotated, bbox=current_bbox, pilot_id=ar.pilot_id,
# #                     absent=ar.absent, timer_val=ar.timer_value,
# #                     calibrated=ar.calibrated,
# #                 )
# #             if ar.absent:
# #                 any_absence_distracted = True
# #                 last_absent_pilot      = ar.pilot_id
# #                 last_absent_duration   = ar.timer_value
# #                 if DRAW:
# #                     _draw_distraction_label(annotated, current_bbox, "Away From Seat",
# #                                             ar.timer_value, color=(0, 140, 255))

# #         any_droop_distracted = False
# #         last_droop_pilot     = None
# #         last_droop_duration  = 0.0
# #         last_droop_severity  = "DROWSINESS"
# #         bbox_by_pid          = {}

# #         if droop_results:
# #             bbox_by_pid = {r.pilot_id: r.bbox for r in results}

# #         for dr in droop_results:
# #             current_bbox = bbox_by_pid.get(dr.pilot_id)
# #             if not dr.is_seated:
# #                 if DRAW:
# #                     draw_standing_label(annotated, dr.pilot_id, current_bbox)
# #                 continue
# #             if hasattr(dr, "keypoints") and dr.keypoints:
# #                 if DRAW:
# #                     draw_droop_keypoints(
# #                         frame=annotated, keypoints=dr.keypoints,
# #                         pilot_id=dr.pilot_id, drooping=dr.drooping,
# #                         angle=getattr(dr, "angle", 0.0),
# #                     )
# #             if DRAW:
# #                 draw_droop_overlay(
# #                     frame=annotated, pilot_id=dr.pilot_id,
# #                     drooping=dr.drooping, timer_val=dr.timer_value,
# #                     bbox=current_bbox,
# #                     severity=getattr(dr, "severity", "DROWSINESS"),
# #                 )
# #             if dr.drooping:
# #                 any_droop_distracted = True
# #                 last_droop_pilot     = dr.pilot_id
# #                 last_droop_duration  = dr.timer_value
# #                 last_droop_severity  = getattr(dr, "severity", "DROWSINESS")
# #                 display_secs         = dr.timer_value * (38 / 25.0)
# #                 if DRAW:
# #                     _draw_distraction_label(
# #                         annotated, current_bbox, last_droop_severity, display_secs,
# #                         color=(0, 200, 255) if last_droop_severity == "DROWSINESS"
# #                         else (0, 80, 200),
# #                     )

# #         if DRAW:
# #             if any_gadget_distracted and last_gadget_pilot is not None:
# #                 draw_alert_banner(annotated, last_gadget_pilot, last_gadget_name)
# #             if any_absence_distracted and last_absent_pilot is not None:
# #                 draw_absence_banner(annotated, last_absent_pilot, last_absent_duration)
# #             if any_droop_distracted and last_droop_pilot is not None:
# #                 draw_droop_banner(annotated, last_droop_pilot, last_droop_duration,
# #                                   severity=last_droop_severity)
# #             for dr in droop_results:
# #                 if not dr.drooping:
# #                     continue
# #                 if any(ar.absent and ar.pilot_id == dr.pilot_id
# #                        for ar in absence_results):
# #                     cb = bbox_by_pid.get(dr.pilot_id)
# #                     _draw_distraction_label(annotated, cb, "SLEEPING / ABSENT",
# #                                             dr.timer_value, color=(0, 50, 200))
# #             draw_hud(annotated, video_time, raw_frame_no, len(results))

# #         # ── CHANGED: compute local_video_time so original_video_timestamp ──
# #         # in the JSON report shows the correct time WITHIN the source file,
# #         # not the global cumulative timeline position.
# #         # source_filename is passed so the field reads e.g.:
# #         #   "ch06_...mp4 00:00:17"  instead of  " 00:00:00"
# #         local_video_time = video_time - self.time_offset   # time within THIS video
# #         src_filename     = self.original_filename or ""    # real filename, not tmp path

# #         # ── Log + store violations ────────────────────────────────

# #         if log_events:
# #             r_ref = next((r for r in results if r.distracted), None)
# #             conf  = r_ref.gadgets[0].confidence if (r_ref and r_ref.gadgets) else 0.9
# #             dur   = r_ref.timer_value if r_ref else 0.0
# #             event_time = max(self.time_offset, video_time - GADGET_ALLOWED_DURATION)
# #             self.vstore.record_violation(
# #                 annotated_frame  = annotated,
# #                 original_frame   = frame,
# #                 video_time       = event_time,
# #                 frame_index      = raw_frame_no,
# #                 event_type       = "phone_use",
# #                 severity         = "CRITICAL",
# #                 confidence       = conf,
# #                 risk_score       = 80,
# #                 risk_level       = "CRITICAL",
# #                 factors          = ["phone_use"],
# #                 duration         = dur,
# #                 source_filename  = src_filename,       # ← CHANGED
# #                 local_video_time = local_video_time,   # ← CHANGED
# #             )
# #             log_distraction(self.logger, event_time,
# #                             event="One of the pilots is using a mobile phone",
# #                             severity="CRITICAL", frame=annotated)

# #         if absence_log_events:
# #             ar_ref  = next((ar for ar in absence_results if ar.absent), None)
# #             dur_abs = ar_ref.timer_value if ar_ref else 0.0
# #             event_time = max(self.time_offset, video_time - ABSENCE_ALLOWED_DURATION)
# #             self.vstore.record_violation(
# #                 annotated_frame  = annotated,
# #                 original_frame   = frame,
# #                 video_time       = event_time,
# #                 frame_index      = raw_frame_no,
# #                 event_type       = "seat_absence",
# #                 severity         = "CRITICAL",
# #                 confidence       = 1.0,
# #                 risk_score       = 70,
# #                 risk_level       = "CRITICAL",
# #                 factors          = ["seat_absence"],
# #                 duration         = dur_abs,
# #                 source_filename  = src_filename,       # ← CHANGED
# #                 local_video_time = local_video_time,   # ← CHANGED
# #             )
# #             log_distraction(self.logger, event_time,
# #                             event="One of the pilots is away from the seat",
# #                             severity="CRITICAL", frame=annotated)

# #         if droop_log_events:
# #             severities  = [e[1] for e in droop_log_events]
# #             is_sleeping = any("SLEEPING" in s for s in severities)
# #             droop_pids  = {e[0] for e in droop_log_events}
# #             absent_pids = {ar.pilot_id for ar in absence_results if ar.absent}
# #             also_absent = bool(droop_pids & absent_pids)

# #             if also_absent:
# #                 event_msg = "One of the pilots is sleeping / slumped in seat"
# #                 etype     = "sleeping_absent"
# #             elif is_sleeping:
# #                 event_msg = "One of the pilots is sleeping"
# #                 etype     = "sleeping"
# #             else:
# #                 event_msg = "One of the pilots is drowsy"
# #                 etype     = "drowsy"

# #             dr_ref  = next((dr for dr in droop_results if dr.drooping), None)
# #             dur_drp = dr_ref.timer_value if dr_ref else 0.0
# #             event_time = max(self.time_offset, video_time - HEAD_DROP_DURATION)
# #             self.vstore.record_violation(
# #                 annotated_frame  = annotated,
# #                 original_frame   = frame,
# #                 video_time       = event_time,
# #                 frame_index      = raw_frame_no,
# #                 event_type       = etype,
# #                 severity         = "CRITICAL",
# #                 confidence       = 0.9,
# #                 risk_score       = 75,
# #                 risk_level       = "HIGH",
# #                 factors          = ["drowsy", "head_droop"],
# #                 duration         = dur_drp,
# #                 source_filename  = src_filename,       # ← CHANGED
# #                 local_video_time = local_video_time,   # ← CHANGED
# #             )
# #             log_distraction(self.logger, event_time, event=event_msg,
# #                             severity="CRITICAL", frame=annotated)

# #         return annotated


# #     # ── HELPERS ───────────────────────────────────────────────────

# #     def _print_banner(self, fps: float, w: int, h: int, total: int) -> None:
# #         self.logger.info(
# #             f"\n{'='*60}\n"
# #             f"  LOCO PILOT DISTRACTION DETECTION\n"
# #             f"  Analysis ID : {self.analysis_id}\n"
# #             f"  Source      : {self.source}\n"
# #             f"  Video       : {w}x{h} @ {fps:.1f} fps ({total} frames)\n"
# #             f"  Output      : {OUTPUT_PATH if self.save else 'disabled'}\n"
# #             f"{'='*60}\n"
# #         )

# #     def _print_summary(self, frame_no: int, processing_time: float) -> None:
# #         self.logger.info(
# #             f"\n{'='*60}\n"
# #             f"  Processing complete\n"
# #             f"  Raw frames  : {frame_no}\n"
# #             f"  Processed   : {self._processed_frame_no} "
# #             f"(1 in every {RAW_FRAME_SKIP})\n"
# #             f"  Time        : {processing_time:.2f}s\n"
# #             f"  Report : outputs/{self.analysis_id}/analysis_report.json\n"
# #             f"  Frames : outputs/{self.analysis_id}/frames/\n"
# #             f"{'='*60}\n"
# #         )


# # # ── CLI ───────────────────────────────────────────────────────────

# # def parse_args() -> argparse.Namespace:
# #     p = argparse.ArgumentParser(description="Loco Pilot Distraction Detection")
# #     p.add_argument("--source",          default=0,
# #                    help="Video file path or camera index (default: 0 = webcam)")
# #     p.add_argument("--analysis-id",     default=None)
# #     p.add_argument("--train-detail-id", default=0, type=int)
# #     p.add_argument("--no-display",      action="store_true")
# #     p.add_argument("--no-save",         action="store_true")
# #     return p.parse_args()


# # if __name__ == "__main__":
# #     args   = parse_args()
# #     source = args.source
# #     if isinstance(source, str) and source.isdigit():
# #         source = int(source)

# #     GadgetDetectionPipeline(
# #         source          = source,
# #         analysis_id     = args.analysis_id,
# #         train_detail_id = args.train_detail_id,
# #         save            = not args.no_save,
# #         display         = False,
# #     ).run()

# # # from __future__ import annotations

# # # import argparse
# # # import os
# # # import queue
# # # import re
# # # import sys
# # # import threading
# # # import traceback
# # # import uuid
# # # from typing import Optional
# # # from concurrent.futures import ThreadPoolExecutor
# # # import warnings
# # # import cv2
# # # import numpy as np


# # # DRAW           = False  # set True only for visual debug
# # # RAW_FRAME_SKIP = 3      # process 1 in every N raw frames
# # # GADGET_EVERY   = 6      # YOLO  every Nth processed frame
# # # ABSENCE_EVERY  = 4      # absence every Nth processed frame
# # # DROOP_EVERY    = 15     # droop every Nth processed frame
# # #  # allowed duration in seconds before logging violation


# # # from config.settings import OUTPUT_PATH, WINDOW_NAME, DISPLAY_SCALE,GADGET_ALLOWED_DURATION,ABSENCE_ALLOWED_DURATION,HEAD_DROP_DURATION
# # # from utils.logger import setup_logger, log_distraction, finalize_report
# # # from utils.violation_store import ViolationStore
# # # from utils.draw import (
# # #     draw_pilot_box, draw_gadget_box, draw_hud, draw_alert_banner,
# # #     draw_seat_zone, draw_absence_overlay, draw_absence_banner,
# # #     draw_droop_keypoints, draw_droop_overlay, draw_droop_banner,
# # #     draw_standing_label,
# # # )
# # # from detector.gadget_detector import GadgetDetector
# # # from detector.seat_absence_detector import SeatAbsenceDetector
# # # from detector.head_drop_detector import HeadDroopDetector

# # # _STOP = object()

# # # READ_QUEUE_MAXSIZE  = 8
# # # WRITE_QUEUE_MAXSIZE = 8

# # # warnings.filterwarnings("ignore", category=UserWarning)


# # # def _draw_distraction_label(
# # #     frame: np.ndarray,
# # #     bbox: tuple,
# # #     distraction_type: str,
# # #     timer_val: float,
# # #     color: tuple = (0, 0, 255),
# # # ) -> None:
# # #     if bbox is None:
# # #         return
# # #     x1, y1, x2, y2 = bbox
# # #     label = f"{distraction_type}  {timer_val:.1f}s"
# # #     font       = cv2.FONT_HERSHEY_DUPLEX
# # #     font_scale = 0.52
# # #     thickness  = 1
# # #     (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
# # #     pad    = 4
# # #     tag_y2 = max(y1, th + pad * 2)
# # #     tag_y1 = tag_y2 - th - pad * 2
# # #     tag_x2 = x1 + tw + pad * 2
# # #     cv2.rectangle(frame, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
# # #     cv2.putText(
# # #         frame, label,
# # #         (x1 + pad, tag_y2 - pad - baseline // 2),
# # #         font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
# # #     )


# # # class GadgetDetectionPipeline:

# # #     def __init__(
# # #         self,
# # #         source:          str | int,
# # #         analysis_id:     Optional[str] = None,
# # #         train_detail_id: int           = 0,
# # #         save:            bool          = False,
# # #         display:         bool          = False,
# # #         time_offset:     float         = 0.0,
# # #         frame_offset:    int           = 0,       # ← NEW: cumulative frame count before this video
# # #         shared_vstore=None,
# # #         original_filename: Optional[str] = None,
# # #     ) -> None:
# # #         self.source            = source
# # #         self.train_detail_id   = train_detail_id
# # #         self.save              = save
# # #         self.display           = display
# # #         self.time_offset       = time_offset      # cumulative seconds before this video
# # #         self.frame_offset      = frame_offset     # cumulative frames before this video
# # #         self.shared_vstore     = shared_vstore    # if set, use this instead of creating new one
# # #         self.original_filename = original_filename  # real upload name, overrides tmp path basename

# # #         if analysis_id:
# # #             self.analysis_id = analysis_id
# # #         elif (
# # #             isinstance(source, str)
# # #             and source not in ("0",)
# # #             and os.path.isfile(source)
# # #         ):
# # #             stem             = os.path.splitext(os.path.basename(source))[0]
# # #             self.analysis_id = re.sub(r"[^A-Za-z0-9_-]", "_", stem)
# # #         else:
# # #             self.analysis_id = uuid.uuid4().hex[:8]

# # #         self.logger           = setup_logger()
# # #         self.detector         = GadgetDetector()
# # #         self.absence_detector = SeatAbsenceDetector()
# # #         self.droop_detector   = HeadDroopDetector()
# # #         self._writer:  Optional[cv2.VideoWriter] = None
# # #         self.vstore:   Optional[ViolationStore]  = None

# # #         # 3 workers: one per detector, no excess overhead
# # #         self.executor = ThreadPoolExecutor(max_workers=3)

# # #         self._prev_pilot_boxes      = []
# # #         self._prev_frame_detections = None
# # #         self._processed_frame_no    = 0   

# # #         self._read_queue:  queue.Queue = queue.Queue(maxsize=READ_QUEUE_MAXSIZE)
# # #         self._write_queue: queue.Queue = queue.Queue(maxsize=WRITE_QUEUE_MAXSIZE)

    
# # #     # ENTRY POINT
    

# # #     def run(self) -> tuple:
# # #         """Returns (report_path, duration_seconds, total_frame_count)."""
# # #         import time
# # #         start_time = time.time()

# # #         cap = cv2.VideoCapture(self.source)
# # #         if not cap.isOpened():
# # #             self.logger.error(f"Cannot open source: {self.source!r}")
# # #             sys.exit(1)

# # #         _raw_fps = cap.get(cv2.CAP_PROP_FPS)
# # #         if not _raw_fps:
# # #             print("[WARNING] FPS not detected — defaulting to 25.0")
# # #         fps    = _raw_fps or 25.0
# # #         width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# # #         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# # #         total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# # #         print(f"Video      : {width}x{height} @ {fps:.1f}fps  "
# # #               f"{total/fps:.1f}s  {total} frames")
# # #         print(f"Analysis ID: {self.analysis_id}")
# # #         print(f"Processing : every {RAW_FRAME_SKIP}rd raw frame  |  "
# # #               f"YOLO every {RAW_FRAME_SKIP * GADGET_EVERY} raw frames  |  "
# # #               f"Droop every {RAW_FRAME_SKIP * DROOP_EVERY} raw frames")

# # #         source_str  = str(self.source)
# # #         # Prefer the real uploaded filename passed in from api.py;
# # #         # fall back to basename of the (temp) path for direct/CLI use.
# # #         source_name = (
# # #             self.original_filename
# # #             if self.original_filename
# # #             else (os.path.basename(source_str) if isinstance(self.source, str) else "webcam")
# # #         )
# # #         # Seek to end to get true duration — handles VFR and mismatched fps tags
# # #         # (e.g. cabin_video.mp4 has container fps=30 but actual fps=6)
# # #         cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
# # #         duration_s = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 3)
# # #         cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # rewind for processing
# # #         if duration_s <= 0:
# # #             # Fallback for sources that don't support seek-to-end
# # #             duration_s = round(total / fps, 3) if total > 0 and fps > 0 else 0.0
# # #         h, m, s    = (
# # #             int(duration_s) // 3600,
# # #             (int(duration_s) % 3600) // 60,
# # #             int(duration_s) % 60,
# # #         )
# # #         size_mb = (
# # #             round(os.path.getsize(source_str) / 1_000_000, 2)
# # #             if isinstance(self.source, str) and os.path.isfile(source_str) else 0
# # #         )

# # #         video_info = {
# # #             "filename":          source_name,
# # #             "videoPath":         source_str,
# # #             "durationSeconds":   duration_s,
# # #             "durationFormatted": f"{h}:{m:02d}:{s:02d}",
# # #             "resolution":        f"{width}x{height}",
# # #             "fps":               round(fps, 3),
# # #             "totalFrames":       total,
# # #             "sizeMb":            size_mb,
# # #         }

# # #         if self.shared_vstore is not None:
# # #             # Batch mode: attach this video's metadata to the shared store
# # #             self.vstore = self.shared_vstore
# # #             self.vstore.add_video_info(video_info)
# # #         else:
# # #             self.vstore = ViolationStore(
# # #                 analysis_id     = self.analysis_id,
# # #                 train_detail_id = self.train_detail_id,
# # #                 video_info      = video_info,
# # #             )
# # #         self._print_banner(fps, width, height, total)

# # #         if self.save:
# # #             os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
# # #             self._writer = cv2.VideoWriter(
# # #                 OUTPUT_PATH,
# # #                 cv2.VideoWriter_fourcc(*"mp4v"),
# # #                 fps,
# # #                 (width, height),
# # #             )

# # #         raw_frame_no = 0
# # #         report_path  = ""

# # #         reader_thread = threading.Thread(
# # #             target=self._reader_loop, args=(cap,),
# # #             daemon=True, name="FrameReader",
# # #         )
# # #         writer_thread = threading.Thread(
# # #             target=self._writer_loop,
# # #             daemon=True, name="FrameWriter",
# # #         )
# # #         reader_thread.start()
# # #         writer_thread.start()

# # #         try:
# # #             while True:
# # #                 item = self._read_queue.get()
# # #                 if item is _STOP:
# # #                     break

# # #                 raw_frame, raw_frame_no, video_time = item

# # #                 # ── Skip most raw frames — pass through as-is ─────
# # #                 # raw_frame_no here is already globally offset so the
# # #                 # modulo cadence is kept consistent across videos.
# # #                 if raw_frame_no % RAW_FRAME_SKIP != 0:
# # #                     self._write_queue.put(raw_frame)
# # #                     continue

# # #                 # ── Process this frame ────────────────────────────
# # #                 self._processed_frame_no += 1
# # #                 annotated = self._process_frame(
# # #                     raw_frame, video_time, raw_frame_no, self._processed_frame_no
# # #                 )
# # #                 self._write_queue.put(annotated)

# # #                 if self.display:
# # #                     show = annotated
# # #                     if DISPLAY_SCALE != 1.0:
# # #                         show = cv2.resize(
# # #                             annotated,
# # #                             (int(width * DISPLAY_SCALE), int(height * DISPLAY_SCALE)),
# # #                         )
# # #                     cv2.imshow(WINDOW_NAME, show)
# # #                     key = cv2.waitKey(1) & 0xFF
# # #                     if key in (ord("q"), 27):
# # #                         self.logger.info("Quit by user.")
# # #                         break

# # #         except KeyboardInterrupt:
# # #             self.logger.info("\nInterrupted by user.")
# # #         except Exception:
# # #             self.logger.error("Unexpected error:\n" + traceback.format_exc())
# # #         finally:
# # #             self._write_queue.put(_STOP)
# # #             writer_thread.join(timeout=30)
# # #             cap.release()
# # #             if self._writer:
# # #                 self._writer.release()
# # #             if self.display:
# # #                 cv2.destroyAllWindows()

# # #             processing_time = round(time.time() - start_time, 3)
# # #             self._print_summary(raw_frame_no, processing_time)
# # #             finalize_report()
# # #             # In batch mode (shared_vstore) we do NOT finalize here —
# # #             # api.py finalizes the shared store once after ALL videos are done.
# # #             if self.shared_vstore is None:
# # #                 report_path = self.vstore.finalize(processing_time=processing_time)
# # #             else:
# # #                 report_path = ""   # will be set by api.py after last video

# # #         actual_fps = raw_frame_no / processing_time if processing_time > 0 else 0
# # #         print(f"\nTotal Time : {processing_time:.2f}s   FPS : {actual_fps:.2f}")
# # #         # Return duration_s and total so api.py can accumulate offsets
# # #         # without re-opening the (already-deleted) temp file.
# # #         return report_path, duration_s, total

 
# # #     # READER THREAD
   

# # #     def _reader_loop(self, cap: cv2.VideoCapture) -> None:
# # #         # frame_no counts frames within THIS video (1-based).
# # #         # We add self.frame_offset so every frame has a globally unique
# # #         # index across the entire batch — prevents deduplication collisions
# # #         # in ViolationStore._seen_frames when two videos share the same
# # #         # local frame numbers.
# # #         frame_no = 0
# # #         try:
# # #             while True:
# # #                 ret, frame = cap.read()
# # #                 if not ret:
# # #                     break
# # #                 frame_no  += 1
# # #                 global_frame_no = frame_no + self.frame_offset          # ← CHANGED
# # #                 video_time      = (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0) + self.time_offset
# # #                 self._read_queue.put((frame, global_frame_no, video_time))
# # #         except Exception:
# # #             self.logger.error("Reader error:\n" + traceback.format_exc())
# # #         finally:
# # #             self._read_queue.put(_STOP)

    
# # #     # WRITER THREAD
    

# # #     def _writer_loop(self) -> None:
# # #         try:
# # #             while True:
# # #                 item = self._write_queue.get()
# # #                 if item is _STOP:
# # #                     break
# # #                 if self._writer:
# # #                     self._writer.write(item)
# # #         except Exception:
# # #             self.logger.error("Writer error:\n" + traceback.format_exc())

    
# # #     # PER-FRAME PROCESSING
    

# # #     def _process_frame(
# # #         self,
# # #         frame:              np.ndarray,
# # #         video_time:         float,
# # #         raw_frame_no:       int,
# # #         processed_frame_no: int,
# # #     ) -> np.ndarray:
# # #         annotated = frame

# # #         # Cadences are relative to processed_frame_no so that the effective ML frequency is consistent regardless of RAW_FRAME_SKIP.
# # #         run_gadget  = (processed_frame_no % GADGET_EVERY  == 0)
# # #         run_absence = (processed_frame_no % ABSENCE_EVERY == 0)
# # #         run_droop   = (processed_frame_no % DROOP_EVERY   == 0)

# # #         prev_pilot_boxes     = self._prev_pilot_boxes
# # #         prev_frame_detection = self._prev_frame_detections

# # #         future_gadget = (
# # #             self.executor.submit(self.detector.process, frame, round(video_time, 0))
# # #             if run_gadget else None
# # #         )
# # #         future_absence = (
# # #             self.executor.submit(
# # #                 self.absence_detector.process,
# # #                 prev_pilot_boxes, video_time, frame.shape[1], frame.shape[0],
# # #             )
# # #             if run_absence else None
# # #         )
# # #         future_droop = (
# # #             self.executor.submit(
# # #                 self.droop_detector.process,
# # #                 frame, video_time, prev_frame_detection,
# # #             )
# # #             if run_droop else None
# # #         )

# # #         results,         log_events         = [], []
# # #         absence_results, absence_log_events = [], []
# # #         droop_results,   droop_log_events   = [], []

# # #         try:
# # #             if future_gadget is not None:
# # #                 results, log_events = future_gadget.result()
# # #         except Exception as exc:
# # #             self.logger.error(f"Gadget error frame {raw_frame_no}: {exc}", exc_info=True)

# # #         try:
# # #             if future_absence is not None:
# # #                 absence_results, absence_log_events = future_absence.result()
# # #         except Exception as exc:
# # #             self.logger.error(f"Absence error frame {raw_frame_no}: {exc}", exc_info=True)

# # #         try:
# # #             if future_droop is not None:
# # #                 droop_results, droop_log_events = future_droop.result()
# # #         except Exception as exc:
# # #             self.logger.error(f"Droop error frame {raw_frame_no}: {exc}", exc_info=True)

# # #         if run_gadget:
# # #             self._prev_pilot_boxes      = [(r.pilot_id, r.bbox) for r in results]
# # #             self._prev_frame_detections = self.detector.last_frame_detections

# # #         #  Draw (skipped entirely when DRAW=False
# # #         if DRAW:
# # #             for g in self.detector.last_gadget_hits:
# # #                 draw_gadget_box(annotated, g.bbox, g.class_name, g.confidence)
# # #             for ar in absence_results:
# # #                 if ar.calibrated and ar.seat_zone is not None:
# # #                     draw_seat_zone(annotated, ar.seat_zone, ar.pilot_id)

# # #         any_gadget_distracted = False
# # #         last_gadget_pilot     = None
# # #         last_gadget_name      = ""

# # #         for r in results:
# # #             gadget_names = [g.class_name for g in r.gadgets]
# # #             if DRAW:
# # #                 draw_pilot_box(annotated, r.bbox, r.pilot_id, r.distracted, [])
# # #             if r.distracted:
# # #                 any_gadget_distracted = True
# # #                 last_gadget_pilot     = r.pilot_id
# # #                 last_gadget_name      = gadget_names[0] if gadget_names else "gadget"
# # #                 if DRAW:
# # #                     _draw_distraction_label(annotated, r.bbox, "Phone Usage",
# # #                                             r.timer_value, color=(0, 0, 220))

# # #         any_absence_distracted = False
# # #         last_absent_pilot      = None
# # #         last_absent_duration   = 0.0

# # #         for ar in absence_results:
# # #             current_bbox = next(
# # #                 (r.bbox for r in results if r.pilot_id == ar.pilot_id), None
# # #             )
# # #             if DRAW:
# # #                 draw_absence_overlay(
# # #                     frame=annotated, bbox=current_bbox, pilot_id=ar.pilot_id,
# # #                     absent=ar.absent, timer_val=ar.timer_value,
# # #                     calibrated=ar.calibrated,
# # #                 )
# # #             if ar.absent:
# # #                 any_absence_distracted = True
# # #                 last_absent_pilot      = ar.pilot_id
# # #                 last_absent_duration   = ar.timer_value
# # #                 if DRAW:
# # #                     _draw_distraction_label(annotated, current_bbox, "Away From Seat",
# # #                                             ar.timer_value, color=(0, 140, 255))

# # #         any_droop_distracted = False
# # #         last_droop_pilot     = None
# # #         last_droop_duration  = 0.0
# # #         last_droop_severity  = "DROWSINESS"
# # #         bbox_by_pid          = {}

# # #         if droop_results:
# # #             bbox_by_pid = {r.pilot_id: r.bbox for r in results}

# # #         for dr in droop_results:
# # #             current_bbox = bbox_by_pid.get(dr.pilot_id)
# # #             if not dr.is_seated:
# # #                 if DRAW:
# # #                     draw_standing_label(annotated, dr.pilot_id, current_bbox)
# # #                 continue
# # #             if hasattr(dr, "keypoints") and dr.keypoints:
# # #                 if DRAW:
# # #                     draw_droop_keypoints(
# # #                         frame=annotated, keypoints=dr.keypoints,
# # #                         pilot_id=dr.pilot_id, drooping=dr.drooping,
# # #                         angle=getattr(dr, "angle", 0.0),
# # #                     )
# # #             if DRAW:
# # #                 draw_droop_overlay(
# # #                     frame=annotated, pilot_id=dr.pilot_id,
# # #                     drooping=dr.drooping, timer_val=dr.timer_value,
# # #                     bbox=current_bbox,
# # #                     severity=getattr(dr, "severity", "DROWSINESS"),
# # #                 )
# # #             if dr.drooping:
# # #                 any_droop_distracted = True
# # #                 last_droop_pilot     = dr.pilot_id
# # #                 last_droop_duration  = dr.timer_value
# # #                 last_droop_severity  = getattr(dr, "severity", "DROWSINESS")
# # #                 display_secs         = dr.timer_value * (38 / 25.0)
# # #                 if DRAW:
# # #                     _draw_distraction_label(
# # #                         annotated, current_bbox, last_droop_severity, display_secs,
# # #                         color=(0, 200, 255)
# # #                         if last_droop_severity == "DROWSINESS" else (0, 80, 200),
# # #                     )

# # #         if DRAW:
# # #             if any_gadget_distracted and last_gadget_pilot is not None:
# # #                 draw_alert_banner(annotated, last_gadget_pilot, last_gadget_name)
# # #             if any_absence_distracted and last_absent_pilot is not None:
# # #                 draw_absence_banner(annotated, last_absent_pilot, last_absent_duration)
# # #             if any_droop_distracted and last_droop_pilot is not None:
# # #                 draw_droop_banner(annotated, last_droop_pilot, last_droop_duration,
# # #                                   severity=last_droop_severity)
# # #             for dr in droop_results:
# # #                 if not dr.drooping:
# # #                     continue
# # #                 if any(ar.absent and ar.pilot_id == dr.pilot_id
# # #                        for ar in absence_results):
# # #                     cb = bbox_by_pid.get(dr.pilot_id)
# # #                     _draw_distraction_label(annotated, cb, "SLEEPING / ABSENT",
# # #                                             dr.timer_value, color=(0, 50, 200))
# # #             draw_hud(annotated, video_time, raw_frame_no, len(results))

# # #         #  Log + store violations
# # #         if log_events:
# # #             r_ref = next((r for r in results if r.distracted), None)
# # #             conf  = r_ref.gadgets[0].confidence if (r_ref and r_ref.gadgets) else 0.9
# # #             dur   = r_ref.timer_value if r_ref else 0.0
# # #             # Clamp to time_offset floor so we never produce a timestamp
# # #             # earlier than the start of this video in the combined timeline.
# # #             event_time = max(self.time_offset, video_time - GADGET_ALLOWED_DURATION)  # ← CHANGED
# # #             self.vstore.record_violation(
# # #                 annotated_frame=annotated, original_frame=frame,
# # #                 video_time=event_time, frame_index=raw_frame_no,
# # #                 event_type="phone_use", severity="CRITICAL",
# # #                 confidence=conf, risk_score=80, risk_level="CRITICAL",
# # #                 factors=["phone_use"], duration=dur,
# # #             )
# # #             log_distraction(self.logger, event_time,
# # #                             event="One of the pilots is using a mobile phone",
# # #                             severity="CRITICAL", frame=annotated)

# # #         if absence_log_events:
# # #             ar_ref  = next((ar for ar in absence_results if ar.absent), None)
# # #             dur_abs = ar_ref.timer_value if ar_ref else 0.0
# # #             event_time = max(self.time_offset, video_time - ABSENCE_ALLOWED_DURATION)  # ← CHANGED
# # #             self.vstore.record_violation(
# # #                 annotated_frame=annotated, original_frame=frame,
# # #                 video_time=event_time, frame_index=raw_frame_no,
# # #                 event_type="seat_absence", severity="CRITICAL",
# # #                 confidence=1.0, risk_score=70, risk_level="CRITICAL",
# # #                 factors=["seat_absence"], duration=dur_abs,
# # #             )
# # #             log_distraction(self.logger, event_time,
# # #                             event="One of the pilots is away from the seat",
# # #                             severity="CRITICAL", frame=annotated)

# # #         if droop_log_events:
# # #             severities  = [e[1] for e in droop_log_events]
# # #             is_sleeping = any("SLEEPING" in s for s in severities)
# # #             droop_pids  = {e[0] for e in droop_log_events}
# # #             absent_pids = {ar.pilot_id for ar in absence_results if ar.absent}
# # #             also_absent = bool(droop_pids & absent_pids)

# # #             if also_absent:
# # #                 event_msg = "One of the pilots is sleeping / slumped in seat"
# # #                 etype     = "sleeping_absent"
# # #             elif is_sleeping:
# # #                 event_msg = "One of the pilots is sleeping"
# # #                 etype     = "sleeping"
# # #             else:
# # #                 event_msg = "One of the pilots is drowsy"
# # #                 etype     = "drowsy"

# # #             dr_ref  = next((dr for dr in droop_results if dr.drooping), None)
# # #             dur_drp = dr_ref.timer_value if dr_ref else 0.0
# # #             event_time = max(self.time_offset, video_time - HEAD_DROP_DURATION)  # ← CHANGED
# # #             self.vstore.record_violation(
# # #                 annotated_frame=annotated, original_frame=frame,
# # #                 video_time=event_time, frame_index=raw_frame_no,
# # #                 event_type=etype, severity="CRITICAL",
# # #                 confidence=0.9, risk_score=75, risk_level="HIGH",
# # #                 factors=["drowsy", "head_droop"], duration=dur_drp,
# # #             )
# # #             log_distraction(self.logger, event_time, event=event_msg,
# # #                             severity="CRITICAL", frame=annotated)

# # #         return annotated

    
# # #     # HELPERS
    

# # #     def _print_banner(self, fps: float, w: int, h: int, total: int) -> None:
# # #         self.logger.info(
# # #             f"\n{'='*60}\n"
# # #             f"  LOCO PILOT DISTRACTION DETECTION\n"
# # #             f"  Analysis ID : {self.analysis_id}\n"
# # #             f"  Source      : {self.source}\n"
# # #             f"  Video       : {w}x{h} @ {fps:.1f} fps ({total} frames)\n"
# # #             f"  Output      : {OUTPUT_PATH if self.save else 'disabled'}\n"
# # #             f"{'='*60}\n"
# # #         )

# # #     def _print_summary(self, frame_no: int, processing_time: float) -> None:
# # #         self.logger.info(
# # #             f"\n{'='*60}\n"
# # #             f"  Processing complete\n"
# # #             f"  Raw frames  : {frame_no}\n"
# # #             f"  Processed   : {self._processed_frame_no} "
# # #             f"(1 in every {RAW_FRAME_SKIP})\n"
# # #             f"  Time        : {processing_time:.2f}s\n"
# # #             f"  Report : outputs/{self.analysis_id}/analysis_report.json\n"
# # #             f"  Frames : outputs/{self.analysis_id}/frames/\n"
# # #             f"{'='*60}\n"
# # #         )



# # # # CLI


# # # def parse_args() -> argparse.Namespace:
# # #     p = argparse.ArgumentParser(description="Loco Pilot Distraction Detection")
# # #     p.add_argument("--source",          default=0,
# # #                    help="Video file path or camera index (default: 0 = webcam)")
# # #     p.add_argument("--analysis-id",     default=None)
# # #     p.add_argument("--train-detail-id", default=0, type=int)
# # #     p.add_argument("--no-display",      action="store_true")
# # #     p.add_argument("--no-save",         action="store_true")
# # #     return p.parse_args()


# # # if __name__ == "__main__":
# # #     args   = parse_args()
# # #     source = args.source
# # #     if isinstance(source, str) and source.isdigit():
# # #         source = int(source)

# # #     GadgetDetectionPipeline(
# # #         source          = source,
# # #         analysis_id     = args.analysis_id,
# # #         train_detail_id = args.train_detail_id,
# # #         save            = not args.no_save,
# # #         display         = False,
# # #     ).run()



# # from __future__ import annotations

# # import argparse
# # import os
# # import queue
# # import re
# # import sys
# # import threading
# # import traceback
# # import uuid
# # from typing import Optional
# # from concurrent.futures import ThreadPoolExecutor
# # import warnings
# # import cv2
# # import numpy as np


# # DRAW           = False  # set True only for visual debug
# # RAW_FRAME_SKIP = 3      # process 1 in every N raw frames
# # GADGET_EVERY   = 6      # YOLO  every Nth processed frame
# # ABSENCE_EVERY  = 4      # absence every Nth processed frame
# # DROOP_EVERY    = 15     # droop every Nth processed frame


# # from config.settings import (
# #     OUTPUT_PATH, WINDOW_NAME, DISPLAY_SCALE,
# #     GADGET_ALLOWED_DURATION, ABSENCE_ALLOWED_DURATION, HEAD_DROP_DURATION,
# # )
# # from utils.logger import setup_logger, log_distraction, finalize_report
# # from utils.violation_store import ViolationStore
# # from utils.draw import (
# #     draw_pilot_box, draw_gadget_box, draw_hud, draw_alert_banner,
# #     draw_seat_zone, draw_absence_overlay, draw_absence_banner,
# #     draw_droop_keypoints, draw_droop_overlay, draw_droop_banner,
# #     draw_standing_label,
# # )
# # from detector.gadget_detector import GadgetDetector
# # from detector.seat_absence_detector import SeatAbsenceDetector
# # from detector.head_drop_detector import HeadDroopDetector

# # _STOP = object()

# # READ_QUEUE_MAXSIZE  = 8
# # WRITE_QUEUE_MAXSIZE = 8

# # warnings.filterwarnings("ignore", category=UserWarning)


# # def _draw_distraction_label(
# #     frame: np.ndarray,
# #     bbox: tuple,
# #     distraction_type: str,
# #     timer_val: float,
# #     color: tuple = (0, 0, 255),
# # ) -> None:
# #     if bbox is None:
# #         return
# #     x1, y1, x2, y2 = bbox
# #     label = f"{distraction_type}  {timer_val:.1f}s"
# #     font       = cv2.FONT_HERSHEY_DUPLEX
# #     font_scale = 0.52
# #     thickness  = 1
# #     (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
# #     pad    = 4
# #     tag_y2 = max(y1, th + pad * 2)
# #     tag_y1 = tag_y2 - th - pad * 2
# #     tag_x2 = x1 + tw + pad * 2
# #     cv2.rectangle(frame, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
# #     cv2.putText(
# #         frame, label,
# #         (x1 + pad, tag_y2 - pad - baseline // 2),
# #         font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
# #     )


# # class GadgetDetectionPipeline:

# #     def __init__(
# #         self,
# #         source:            str | int,
# #         analysis_id:       Optional[str] = None,
# #         train_detail_id:   int           = 0,
# #         save:              bool          = False,
# #         display:           bool          = False,
# #         time_offset:       float         = 0.0,
# #         frame_offset:      int           = 0,
# #         shared_vstore                    = None,
# #         original_filename: Optional[str] = None,
# #     ) -> None:
# #         self.source            = source
# #         self.train_detail_id   = train_detail_id
# #         self.save              = save
# #         self.display           = display
# #         self.time_offset       = time_offset       # cumulative seconds before this video
# #         self.frame_offset      = frame_offset      # cumulative frames before this video
# #         self.shared_vstore     = shared_vstore
# #         self.original_filename = original_filename # real upload filename, not the tmp path

# #         if analysis_id:
# #             self.analysis_id = analysis_id
# #         elif (
# #             isinstance(source, str)
# #             and source not in ("0",)
# #             and os.path.isfile(source)
# #         ):
# #             stem             = os.path.splitext(os.path.basename(source))[0]
# #             self.analysis_id = re.sub(r"[^A-Za-z0-9_-]", "_", stem)
# #         else:
# #             self.analysis_id = uuid.uuid4().hex[:8]

# #         self.logger           = setup_logger()
# #         self.detector         = GadgetDetector()
# #         self.absence_detector = SeatAbsenceDetector()
# #         self.droop_detector   = HeadDroopDetector()
# #         self._writer: Optional[cv2.VideoWriter] = None
# #         self.vstore: Optional[ViolationStore]   = None

# #         self.executor = ThreadPoolExecutor(max_workers=3)

# #         self._prev_pilot_boxes      = []
# #         self._prev_frame_detections = None
# #         self._processed_frame_no    = 0

# #         self._read_queue:  queue.Queue = queue.Queue(maxsize=READ_QUEUE_MAXSIZE)
# #         self._write_queue: queue.Queue = queue.Queue(maxsize=WRITE_QUEUE_MAXSIZE)


# #     # ── ENTRY POINT ───────────────────────────────────────────────

# #     def run(self) -> tuple:
# #         """Returns (report_path, duration_seconds, total_frame_count)."""
# #         import time
# #         start_time = time.time()

# #         cap = cv2.VideoCapture(self.source)
# #         if not cap.isOpened():
# #             self.logger.error(f"Cannot open source: {self.source!r}")
# #             sys.exit(1)

# #         _raw_fps = cap.get(cv2.CAP_PROP_FPS)
# #         if not _raw_fps:
# #             print("[WARNING] FPS not detected — defaulting to 25.0")
# #         fps    = _raw_fps or 25.0
# #         width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# #         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# #         total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# #         print(f"Video      : {width}x{height} @ {fps:.1f}fps  "
# #               f"{total/fps:.1f}s  {total} frames")
# #         print(f"Analysis ID: {self.analysis_id}")
# #         print(f"Processing : every {RAW_FRAME_SKIP}rd raw frame  |  "
# #               f"YOLO every {RAW_FRAME_SKIP * GADGET_EVERY} raw frames  |  "
# #               f"Droop every {RAW_FRAME_SKIP * DROOP_EVERY} raw frames")

# #         source_str  = str(self.source)
# #         # Use real uploaded filename if provided; fall back to temp file basename
# #         source_name = (
# #             self.original_filename
# #             if self.original_filename
# #             else (os.path.basename(source_str) if isinstance(self.source, str) else "webcam")
# #         )

# #         # Seek to end for accurate duration (handles VFR / mismatched fps tags)
# #         cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
# #         duration_s = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 3)
# #         cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # rewind
# #         if duration_s <= 0:
# #             duration_s = round(total / fps, 3) if total > 0 and fps > 0 else 0.0

# #         h = int(duration_s) // 3600
# #         m = (int(duration_s) % 3600) // 60
# #         s = int(duration_s) % 60
# #         size_mb = (
# #             round(os.path.getsize(source_str) / 1_000_000, 2)
# #             if isinstance(self.source, str) and os.path.isfile(source_str) else 0
# #         )

# #         video_info = {
# #             "filename":          source_name,
# #             "videoPath":         source_str,
# #             "durationSeconds":   duration_s,
# #             "durationFormatted": f"{h}:{m:02d}:{s:02d}",
# #             "resolution":        f"{width}x{height}",
# #             "fps":               round(fps, 3),
# #             "totalFrames":       total,
# #             "sizeMb":            size_mb,
# #         }

# #         if self.shared_vstore is not None:
# #             # Batch mode: attach this video's metadata to the shared store
# #             self.vstore = self.shared_vstore
# #             self.vstore.add_video_info(video_info)
# #         else:
# #             self.vstore = ViolationStore(
# #                 analysis_id     = self.analysis_id,
# #                 train_detail_id = self.train_detail_id,
# #                 video_info      = video_info,
# #             )

# #         self._print_banner(fps, width, height, total)

# #         if self.save:
# #             os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
# #             self._writer = cv2.VideoWriter(
# #                 OUTPUT_PATH,
# #                 cv2.VideoWriter_fourcc(*"mp4v"),
# #                 fps,
# #                 (width, height),
# #             )

# #         raw_frame_no = 0
# #         report_path  = ""

# #         reader_thread = threading.Thread(
# #             target=self._reader_loop, args=(cap,),
# #             daemon=True, name="FrameReader",
# #         )
# #         writer_thread = threading.Thread(
# #             target=self._writer_loop,
# #             daemon=True, name="FrameWriter",
# #         )
# #         reader_thread.start()
# #         writer_thread.start()

# #         try:
# #             while True:
# #                 item = self._read_queue.get()
# #                 if item is _STOP:
# #                     break

# #                 raw_frame, raw_frame_no, video_time = item

# #                 if raw_frame_no % RAW_FRAME_SKIP != 0:
# #                     self._write_queue.put(raw_frame)
# #                     continue

# #                 self._processed_frame_no += 1
# #                 annotated = self._process_frame(
# #                     raw_frame, video_time, raw_frame_no, self._processed_frame_no
# #                 )
# #                 self._write_queue.put(annotated)

# #                 if self.display:
# #                     show = annotated
# #                     if DISPLAY_SCALE != 1.0:
# #                         show = cv2.resize(
# #                             annotated,
# #                             (int(width * DISPLAY_SCALE), int(height * DISPLAY_SCALE)),
# #                         )
# #                     cv2.imshow(WINDOW_NAME, show)
# #                     key = cv2.waitKey(1) & 0xFF
# #                     if key in (ord("q"), 27):
# #                         self.logger.info("Quit by user.")
# #                         break

# #         except KeyboardInterrupt:
# #             self.logger.info("\nInterrupted by user.")
# #         except Exception:
# #             self.logger.error("Unexpected error:\n" + traceback.format_exc())
# #         finally:
# #             self._write_queue.put(_STOP)
# #             writer_thread.join(timeout=30)
# #             cap.release()
# #             if self._writer:
# #                 self._writer.release()
# #             if self.display:
# #                 cv2.destroyAllWindows()

# #             processing_time = round(time.time() - start_time, 3)
# #             self._print_summary(raw_frame_no, processing_time)
# #             finalize_report()

# #             # In batch mode the caller (api.py) finalizes after ALL videos are done
# #             if self.shared_vstore is None:
# #                 report_path = self.vstore.finalize(processing_time=processing_time)
# #             else:
# #                 report_path = ""

# #         actual_fps = raw_frame_no / processing_time if processing_time > 0 else 0
# #         print(f"\nTotal Time : {processing_time:.2f}s   FPS : {actual_fps:.2f}")
# #         return report_path, duration_s, total


# #     # ── READER THREAD ─────────────────────────────────────────────

# #     def _reader_loop(self, cap: cv2.VideoCapture) -> None:
# #         # frame_no counts frames within THIS video (1-based).
# #         # Adding self.frame_offset gives every frame a globally unique
# #         # index across the entire batch — prevents dedup collisions in
# #         # ViolationStore._seen_frames when two videos share local frame numbers.
# #         frame_no = 0
# #         try:
# #             while True:
# #                 ret, frame = cap.read()
# #                 if not ret:
# #                     break
# #                 frame_no        += 1
# #                 global_frame_no  = frame_no + self.frame_offset
# #                 video_time       = (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0) + self.time_offset
# #                 self._read_queue.put((frame, global_frame_no, video_time))
# #         except Exception:
# #             self.logger.error("Reader error:\n" + traceback.format_exc())
# #         finally:
# #             self._read_queue.put(_STOP)


# #     # ── WRITER THREAD ─────────────────────────────────────────────

# #     def _writer_loop(self) -> None:
# #         try:
# #             while True:
# #                 item = self._write_queue.get()
# #                 if item is _STOP:
# #                     break
# #                 if self._writer:
# #                     self._writer.write(item)
# #         except Exception:
# #             self.logger.error("Writer error:\n" + traceback.format_exc())


# #     # ── PER-FRAME PROCESSING ──────────────────────────────────────

# #     def _process_frame(
# #         self,
# #         frame:              np.ndarray,
# #         video_time:         float,
# #         raw_frame_no:       int,
# #         processed_frame_no: int,
# #     ) -> np.ndarray:
# #         annotated = frame

# #         run_gadget  = (processed_frame_no % GADGET_EVERY  == 0)
# #         run_absence = (processed_frame_no % ABSENCE_EVERY == 0)
# #         run_droop   = (processed_frame_no % DROOP_EVERY   == 0)

# #         prev_pilot_boxes     = self._prev_pilot_boxes
# #         prev_frame_detection = self._prev_frame_detections

# #         future_gadget = (
# #             self.executor.submit(self.detector.process, frame, round(video_time, 0))
# #             if run_gadget else None
# #         )
# #         future_absence = (
# #             self.executor.submit(
# #                 self.absence_detector.process,
# #                 prev_pilot_boxes, video_time, frame.shape[1], frame.shape[0],
# #             )
# #             if run_absence else None
# #         )
# #         future_droop = (
# #             self.executor.submit(
# #                 self.droop_detector.process,
# #                 frame, video_time, prev_frame_detection,
# #             )
# #             if run_droop else None
# #         )

# #         results,         log_events         = [], []
# #         absence_results, absence_log_events = [], []
# #         droop_results,   droop_log_events   = [], []

# #         try:
# #             if future_gadget is not None:
# #                 results, log_events = future_gadget.result()
# #         except Exception as exc:
# #             self.logger.error(f"Gadget error frame {raw_frame_no}: {exc}", exc_info=True)

# #         try:
# #             if future_absence is not None:
# #                 absence_results, absence_log_events = future_absence.result()
# #         except Exception as exc:
# #             self.logger.error(f"Absence error frame {raw_frame_no}: {exc}", exc_info=True)

# #         try:
# #             if future_droop is not None:
# #                 droop_results, droop_log_events = future_droop.result()
# #         except Exception as exc:
# #             self.logger.error(f"Droop error frame {raw_frame_no}: {exc}", exc_info=True)

# #         if run_gadget:
# #             self._prev_pilot_boxes      = [(r.pilot_id, r.bbox) for r in results]
# #             self._prev_frame_detections = self.detector.last_frame_detections

# #         # ── Draw (skipped entirely when DRAW=False) ───────────────
# #         if DRAW:
# #             for g in self.detector.last_gadget_hits:
# #                 draw_gadget_box(annotated, g.bbox, g.class_name, g.confidence)
# #             for ar in absence_results:
# #                 if ar.calibrated and ar.seat_zone is not None:
# #                     draw_seat_zone(annotated, ar.seat_zone, ar.pilot_id)

# #         any_gadget_distracted = False
# #         last_gadget_pilot     = None
# #         last_gadget_name      = ""

# #         for r in results:
# #             gadget_names = [g.class_name for g in r.gadgets]
# #             if DRAW:
# #                 draw_pilot_box(annotated, r.bbox, r.pilot_id, r.distracted, [])
# #             if r.distracted:
# #                 any_gadget_distracted = True
# #                 last_gadget_pilot     = r.pilot_id
# #                 last_gadget_name      = gadget_names[0] if gadget_names else "gadget"
# #                 if DRAW:
# #                     _draw_distraction_label(annotated, r.bbox, "Phone Usage",
# #                                             r.timer_value, color=(0, 0, 220))

# #         any_absence_distracted = False
# #         last_absent_pilot      = None
# #         last_absent_duration   = 0.0

# #         for ar in absence_results:
# #             current_bbox = next(
# #                 (r.bbox for r in results if r.pilot_id == ar.pilot_id), None
# #             )
# #             if DRAW:
# #                 draw_absence_overlay(
# #                     frame=annotated, bbox=current_bbox, pilot_id=ar.pilot_id,
# #                     absent=ar.absent, timer_val=ar.timer_value,
# #                     calibrated=ar.calibrated,
# #                 )
# #             if ar.absent:
# #                 any_absence_distracted = True
# #                 last_absent_pilot      = ar.pilot_id
# #                 last_absent_duration   = ar.timer_value
# #                 if DRAW:
# #                     _draw_distraction_label(annotated, current_bbox, "Away From Seat",
# #                                             ar.timer_value, color=(0, 140, 255))

# #         any_droop_distracted = False
# #         last_droop_pilot     = None
# #         last_droop_duration  = 0.0
# #         last_droop_severity  = "DROWSINESS"
# #         bbox_by_pid          = {}

# #         if droop_results:
# #             bbox_by_pid = {r.pilot_id: r.bbox for r in results}

# #         for dr in droop_results:
# #             current_bbox = bbox_by_pid.get(dr.pilot_id)
# #             if not dr.is_seated:
# #                 if DRAW:
# #                     draw_standing_label(annotated, dr.pilot_id, current_bbox)
# #                 continue
# #             if hasattr(dr, "keypoints") and dr.keypoints:
# #                 if DRAW:
# #                     draw_droop_keypoints(
# #                         frame=annotated, keypoints=dr.keypoints,
# #                         pilot_id=dr.pilot_id, drooping=dr.drooping,
# #                         angle=getattr(dr, "angle", 0.0),
# #                     )
# #             if DRAW:
# #                 draw_droop_overlay(
# #                     frame=annotated, pilot_id=dr.pilot_id,
# #                     drooping=dr.drooping, timer_val=dr.timer_value,
# #                     bbox=current_bbox,
# #                     severity=getattr(dr, "severity", "DROWSINESS"),
# #                 )
# #             if dr.drooping:
# #                 any_droop_distracted = True
# #                 last_droop_pilot     = dr.pilot_id
# #                 last_droop_duration  = dr.timer_value
# #                 last_droop_severity  = getattr(dr, "severity", "DROWSINESS")
# #                 display_secs         = dr.timer_value * (38 / 25.0)
# #                 if DRAW:
# #                     _draw_distraction_label(
# #                         annotated, current_bbox, last_droop_severity, display_secs,
# #                         color=(0, 200, 255) if last_droop_severity == "DROWSINESS"
# #                         else (0, 80, 200),
# #                     )

# #         if DRAW:
# #             if any_gadget_distracted and last_gadget_pilot is not None:
# #                 draw_alert_banner(annotated, last_gadget_pilot, last_gadget_name)
# #             if any_absence_distracted and last_absent_pilot is not None:
# #                 draw_absence_banner(annotated, last_absent_pilot, last_absent_duration)
# #             if any_droop_distracted and last_droop_pilot is not None:
# #                 draw_droop_banner(annotated, last_droop_pilot, last_droop_duration,
# #                                   severity=last_droop_severity)
# #             for dr in droop_results:
# #                 if not dr.drooping:
# #                     continue
# #                 if any(ar.absent and ar.pilot_id == dr.pilot_id
# #                        for ar in absence_results):
# #                     cb = bbox_by_pid.get(dr.pilot_id)
# #                     _draw_distraction_label(annotated, cb, "SLEEPING / ABSENT",
# #                                             dr.timer_value, color=(0, 50, 200))
# #             draw_hud(annotated, video_time, raw_frame_no, len(results))

# #         # ── CHANGED: compute local_video_time so original_video_timestamp ──
# #         # in the JSON report shows the correct time WITHIN the source file,
# #         # not the global cumulative timeline position.
# #         # source_filename is passed so the field reads e.g.:
# #         #   "ch06_...mp4 00:00:17"  instead of  " 00:00:00"
# #         local_video_time = video_time - self.time_offset   # time within THIS video
# #         src_filename     = self.original_filename or ""    # real filename, not tmp path

# #         # ── Log + store violations ────────────────────────────────

# #         if log_events:
# #             r_ref = next((r for r in results if r.distracted), None)
# #             conf  = r_ref.gadgets[0].confidence if (r_ref and r_ref.gadgets) else 0.9
# #             dur   = r_ref.timer_value if r_ref else 0.0
# #             event_time = max(self.time_offset, video_time - GADGET_ALLOWED_DURATION)
# #             self.vstore.record_violation(
# #                 annotated_frame  = annotated,
# #                 original_frame   = frame,
# #                 video_time       = event_time,
# #                 frame_index      = raw_frame_no,
# #                 event_type       = "phone_use",
# #                 severity         = "CRITICAL",
# #                 confidence       = conf,
# #                 risk_score       = 80,
# #                 risk_level       = "CRITICAL",
# #                 factors          = ["phone_use"],
# #                 duration         = dur,
# #                 source_filename  = src_filename,       # ← CHANGED
# #                 local_video_time = local_video_time,   # ← CHANGED
# #             )
# #             log_distraction(self.logger, event_time,
# #                             event="One of the pilots is using a mobile phone",
# #                             severity="CRITICAL", frame=annotated)

# #         if absence_log_events:
# #             ar_ref  = next((ar for ar in absence_results if ar.absent), None)
# #             dur_abs = ar_ref.timer_value if ar_ref else 0.0
# #             event_time = max(self.time_offset, video_time - ABSENCE_ALLOWED_DURATION)
# #             self.vstore.record_violation(
# #                 annotated_frame  = annotated,
# #                 original_frame   = frame,
# #                 video_time       = event_time,
# #                 frame_index      = raw_frame_no,
# #                 event_type       = "seat_absence",
# #                 severity         = "CRITICAL",
# #                 confidence       = 1.0,
# #                 risk_score       = 70,
# #                 risk_level       = "CRITICAL",
# #                 factors          = ["seat_absence"],
# #                 duration         = dur_abs,
# #                 source_filename  = src_filename,       # ← CHANGED
# #                 local_video_time = local_video_time,   # ← CHANGED
# #             )
# #             log_distraction(self.logger, event_time,
# #                             event="One of the pilots is away from the seat",
# #                             severity="CRITICAL", frame=annotated)

# #         if droop_log_events:
# #             severities  = [e[1] for e in droop_log_events]
# #             is_sleeping = any("SLEEPING" in s for s in severities)
# #             droop_pids  = {e[0] for e in droop_log_events}
# #             absent_pids = {ar.pilot_id for ar in absence_results if ar.absent}
# #             also_absent = bool(droop_pids & absent_pids)

# #             if also_absent:
# #                 event_msg = "One of the pilots is sleeping / slumped in seat"
# #                 etype     = "sleeping_absent"
# #             elif is_sleeping:
# #                 event_msg = "One of the pilots is sleeping"
# #                 etype     = "sleeping"
# #             else:
# #                 event_msg = "One of the pilots is drowsy"
# #                 etype     = "drowsy"

# #             dr_ref  = next((dr for dr in droop_results if dr.drooping), None)
# #             dur_drp = dr_ref.timer_value if dr_ref else 0.0
# #             event_time = max(self.time_offset, video_time - HEAD_DROP_DURATION)
# #             self.vstore.record_violation(
# #                 annotated_frame  = annotated,
# #                 original_frame   = frame,
# #                 video_time       = event_time,
# #                 frame_index      = raw_frame_no,
# #                 event_type       = etype,
# #                 severity         = "CRITICAL",
# #                 confidence       = 0.9,
# #                 risk_score       = 75,
# #                 risk_level       = "HIGH",
# #                 factors          = ["drowsy", "head_droop"],
# #                 duration         = dur_drp,
# #                 source_filename  = src_filename,       # ← CHANGED
# #                 local_video_time = local_video_time,   # ← CHANGED
# #             )
# #             log_distraction(self.logger, event_time, event=event_msg,
# #                             severity="CRITICAL", frame=annotated)

# #         return annotated


# #     # ── HELPERS ───────────────────────────────────────────────────

# #     def _print_banner(self, fps: float, w: int, h: int, total: int) -> None:
# #         self.logger.info(
# #             f"\n{'='*60}\n"
# #             f"  LOCO PILOT DISTRACTION DETECTION\n"
# #             f"  Analysis ID : {self.analysis_id}\n"
# #             f"  Source      : {self.source}\n"
# #             f"  Video       : {w}x{h} @ {fps:.1f} fps ({total} frames)\n"
# #             f"  Output      : {OUTPUT_PATH if self.save else 'disabled'}\n"
# #             f"{'='*60}\n"
# #         )

# #     def _print_summary(self, frame_no: int, processing_time: float) -> None:
# #         self.logger.info(
# #             f"\n{'='*60}\n"
# #             f"  Processing complete\n"
# #             f"  Raw frames  : {frame_no}\n"
# #             f"  Processed   : {self._processed_frame_no} "
# #             f"(1 in every {RAW_FRAME_SKIP})\n"
# #             f"  Time        : {processing_time:.2f}s\n"
# #             f"  Report : outputs/{self.analysis_id}/analysis_report.json\n"
# #             f"  Frames : outputs/{self.analysis_id}/frames/\n"
# #             f"{'='*60}\n"
# #         )


# # # ── CLI ───────────────────────────────────────────────────────────

# # def parse_args() -> argparse.Namespace:
# #     p = argparse.ArgumentParser(description="Loco Pilot Distraction Detection")
# #     p.add_argument("--source",          default=0,
# #                    help="Video file path or camera index (default: 0 = webcam)")
# #     p.add_argument("--analysis-id",     default=None)
# #     p.add_argument("--train-detail-id", default=0, type=int)
# #     p.add_argument("--no-display",      action="store_true")
# #     p.add_argument("--no-save",         action="store_true")
# #     return p.parse_args()


# # if __name__ == "__main__":
# #     args   = parse_args()
# #     source = args.source
# #     if isinstance(source, str) and source.isdigit():
# #         source = int(source)

# #     GadgetDetectionPipeline(
# #         source          = source,
# #         analysis_id     = args.analysis_id,
# #         train_detail_id = args.train_detail_id,
# #         save            = not args.no_save,
# #         display         = False,
# #     ).run()

# from __future__ import annotations

# import argparse
# import os
# import queue
# import re
# import sys
# import threading
# import traceback
# import uuid
# from typing import Optional
# from concurrent.futures import ThreadPoolExecutor
# import warnings
# import cv2
# import numpy as np

# print("[main] ✅ NEW main.py loaded — v6 (GADGET_EVERY=2, GADGET_MISS_TOLERANCE=5)")

# DRAW           = False
# RAW_FRAME_SKIP = 3
# # FIX: was 6 → YOLO ran every 18 raw frames = 0.72s apart.
# # With GADGET_ALLOWED_DURATION=2.0s that needed 3 perfect consecutive hits.
# # A single missed YOLO run (lighting change, pose shift) would reset progress.
# # At GADGET_EVERY=2 → YOLO runs every 6 raw frames = 0.24s apart.
# # Now only 9 hits needed for 2.0s, and a miss costs 0.24s instead of 0.72s.
# GADGET_EVERY   = 2
# ABSENCE_EVERY  = 4
# DROOP_EVERY    = 15

# # ── MediaPipe pose (optional — graceful degradation if not installed) ─────────
# try:
#     import mediapipe as mp
#     _mp_pose    = mp.solutions.pose
#     _MP_AVAILABLE = True
# except ImportError:
#     _mp_pose      = None
#     _MP_AVAILABLE = False
#     print("[main] WARNING: mediapipe not installed — gadget detector will use bbox fallback")

# from config.settings import OUTPUT_PATH, WINDOW_NAME, DISPLAY_SCALE, GADGET_ALLOWED_DURATION, ABSENCE_ALLOWED_DURATION, HEAD_DROP_DURATION
# from utils.logger import setup_logger, log_distraction, finalize_report
# from utils.violation_store import ViolationStore
# from utils.draw import (
#     draw_pilot_box, draw_gadget_box, draw_hud, draw_alert_banner,
#     draw_seat_zone, draw_absence_overlay, draw_absence_banner,
#     draw_droop_keypoints, draw_droop_overlay, draw_droop_banner,
#     draw_standing_label,
# )
# from detector.gadget_detector import GadgetDetector
# from detector.seat_absence_detector import SeatAbsenceDetector
# from detector.head_drop_detector import HeadDroopDetector

# _STOP = object()
# READ_QUEUE_MAXSIZE  = 8
# WRITE_QUEUE_MAXSIZE = 8
# warnings.filterwarnings("ignore", category=UserWarning)


# # Simple container so MediaPipe landmark coordinates can be patched to
# # full-frame space before being passed to the gadget detector.
# class _PatchedLandmark:
#     __slots__ = ("x", "y", "visibility")
#     def __init__(self, x: float, y: float, visibility: float):
#         self.x          = x
#         self.y          = y
#         self.visibility = visibility


# def _draw_distraction_label(frame, bbox, distraction_type, timer_val, color=(0,0,255)):
#     if bbox is None:
#         return
#     x1, y1, x2, y2 = bbox
#     label = f"{distraction_type}  {timer_val:.1f}s"
#     font, font_scale, thickness = cv2.FONT_HERSHEY_DUPLEX, 0.52, 1
#     (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
#     pad    = 4
#     tag_y2 = max(y1, th + pad * 2)
#     tag_y1 = tag_y2 - th - pad * 2
#     tag_x2 = x1 + tw + pad * 2
#     cv2.rectangle(frame, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
#     cv2.putText(frame, label, (x1+pad, tag_y2-pad-baseline//2),
#                 font, font_scale, (255,255,255), thickness, cv2.LINE_AA)


# class GadgetDetectionPipeline:

#     def __init__(
#         self,
#         source:           str | int,
#         analysis_id:      Optional[str]            = None,
#         train_detail_id:  int                      = 0,
#         save:             bool                     = False,
#         display:          bool                     = False,
#         # ── batch-mode parameters ──────────────────────────────────────────
#         shared_vstore:    Optional[ViolationStore]  = None,  # shared store for whole folder
#         time_offset:      float                    = 0.0,   # cumulative seconds before this video
#         frame_offset:     int                      = 0,     # cumulative frames before this video
#         source_filename:  str                      = "",    # real DB filename e.g. "mobile.mp4"
#     ) -> None:
#         self.source          = source
#         self.train_detail_id = train_detail_id
#         self.save            = save
#         self.display         = display
#         self.shared_vstore   = shared_vstore
#         self.time_offset     = time_offset
#         self.frame_offset    = frame_offset
#         self.source_filename = source_filename

#         if analysis_id:
#             self.analysis_id = analysis_id
#         elif isinstance(source, str) and source not in ("0",) and os.path.isfile(source):
#             stem             = os.path.splitext(os.path.basename(source))[0]
#             self.analysis_id = re.sub(r"[^A-Za-z0-9_-]", "_", stem)
#         else:
#             self.analysis_id = uuid.uuid4().hex[:8]

#         self.logger           = setup_logger()
#         self.detector         = GadgetDetector()
#         self.absence_detector = SeatAbsenceDetector()
#         self.droop_detector   = HeadDroopDetector()

#         # MediaPipe pose — reused across frames, graceful if not installed
#         if _MP_AVAILABLE:
#             self._pose = _mp_pose.Pose(
#                 static_image_mode=False,
#                 model_complexity=1,
#                 enable_segmentation=False,
#                 min_detection_confidence=0.5,
#                 min_tracking_confidence=0.5,
#             )
#         else:
#             self._pose = None
#         self._frame_height: int = 480
#         self._frame_width:  int = 848
#         self._writer:  Optional[cv2.VideoWriter] = None
#         self.vstore:   Optional[ViolationStore]  = None
#         self.executor  = ThreadPoolExecutor(max_workers=3)

#         self._prev_pilot_boxes      = []
#         self._prev_frame_detections = None
#         self._processed_frame_no    = 0
#         self._absence_pilot_boxes: list = []
#         self._yolo_empty_run_count: int = 0

#         self._read_queue:  queue.Queue = queue.Queue(maxsize=READ_QUEUE_MAXSIZE)
#         self._write_queue: queue.Queue = queue.Queue(maxsize=WRITE_QUEUE_MAXSIZE)

#     # ──────────────────────────────────────────────────────────────────────────
#     # run()
#     # ──────────────────────────────────────────────────────────────────────────

#     def run(self) -> str:
#         import time
#         start_time = time.time()

#         cap = cv2.VideoCapture(self.source)
#         if not cap.isOpened():
#             self.logger.error(f"Cannot open source: {self.source!r}")
#             sys.exit(1)

#         _raw_fps = cap.get(cv2.CAP_PROP_FPS)
#         fps    = _raw_fps or 25.0
#         width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         self._frame_width  = width
#         self._frame_height = height
#         total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#         print(f"Video      : {width}x{height} @ {fps:.1f}fps  {total/fps:.1f}s  {total} frames")
#         print(f"Analysis ID: {self.analysis_id}")
#         print(f"time_offset={self.time_offset:.2f}s  frame_offset={self.frame_offset}  source_filename={self.source_filename!r}")

#         source_str = str(self.source)
#         # Always use the DB filename for display; fall back only for standalone CLI runs
#         display_filename = (
#             self.source_filename
#             if self.source_filename
#             else (os.path.basename(source_str) if isinstance(self.source, str) else "webcam")
#         )

#         duration_s = round(total / fps, 3) if total > 0 and fps > 0 else 0.0
#         h, m, s    = int(duration_s)//3600, (int(duration_s)%3600)//60, int(duration_s)%60
#         size_mb    = (
#             round(os.path.getsize(source_str)/1_000_000, 2)
#             if isinstance(self.source, str) and os.path.isfile(source_str) else 0
#         )

#         video_info = {
#             "filename":          display_filename,
#             "videoPath":         source_str,
#             "durationSeconds":   duration_s,
#             "durationFormatted": f"{h}:{m:02d}:{s:02d}",
#             "resolution":        f"{width}x{height}",
#             "fps":               round(fps, 3),
#             "totalFrames":       total,
#             "sizeMb":            size_mb,
#         }

#         if self.shared_vstore is not None:
#             # ── BATCH MODE ─────────────────────────────────────────────────
#             # Use the shared store; register this video's info; do NOT finalize here.
#             self.vstore = self.shared_vstore
#             self.vstore.add_video_info(video_info)
#             print(f"[Pipeline] BATCH MODE — using shared ViolationStore, will NOT finalize here")
#         else:
#             # ── STANDALONE MODE ────────────────────────────────────────────
#             self.vstore = ViolationStore(
#                 analysis_id     = self.analysis_id,
#                 train_detail_id = self.train_detail_id,
#                 video_info      = video_info,
#             )
#             print(f"[Pipeline] STANDALONE MODE — fresh ViolationStore, will finalize at end")

#         self._print_banner(fps, width, height, total)

#         if self.save:
#             os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
#             self._writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

#         raw_frame_no = 0
#         report_path  = ""

#         reader_thread = threading.Thread(target=self._reader_loop, args=(cap,), daemon=True, name="FrameReader")
#         writer_thread = threading.Thread(target=self._writer_loop, daemon=True, name="FrameWriter")
#         reader_thread.start()
#         writer_thread.start()

#         try:
#             while True:
#                 item = self._read_queue.get()
#                 if item is _STOP:
#                     break
#                 raw_frame, raw_frame_no, video_time = item
#                 if raw_frame_no % RAW_FRAME_SKIP != 0:
#                     self._write_queue.put(raw_frame)
#                     continue
#                 self._processed_frame_no += 1
#                 annotated = self._process_frame(raw_frame, video_time, raw_frame_no, self._processed_frame_no)
#                 self._write_queue.put(annotated)
#                 if self.display:
#                     show = annotated
#                     if DISPLAY_SCALE != 1.0:
#                         show = cv2.resize(annotated, (int(width*DISPLAY_SCALE), int(height*DISPLAY_SCALE)))
#                     cv2.imshow(WINDOW_NAME, show)
#                     if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
#                         break

#         except KeyboardInterrupt:
#             self.logger.info("\nInterrupted by user.")
#         except Exception:
#             self.logger.error("Unexpected error:\n" + traceback.format_exc())
#         finally:
#             self._write_queue.put(_STOP)
#             writer_thread.join(timeout=30)
#             cap.release()
#             if self._writer:
#                 self._writer.release()
#             if self.display:
#                 cv2.destroyAllWindows()

#             processing_time = round(time.time() - start_time, 3)
#             self._print_summary(raw_frame_no, processing_time)
#             finalize_report()

#             if self.shared_vstore is None:
#                 # STANDALONE — finalize now
#                 report_path = self.vstore.finalize(processing_time=processing_time)
#             # BATCH — api.py calls finalize() once after all videos; return empty string
#             else:
#                 report_path = ""

#         actual_fps = raw_frame_no / processing_time if processing_time > 0 else 0
#         print(f"\nTotal Time : {processing_time:.2f}s   FPS : {actual_fps:.2f}")
#         return report_path

#     # ──────────────────────────────────────────────────────────────────────────
#     # reader / writer threads
#     # ──────────────────────────────────────────────────────────────────────────

#     def _reader_loop(self, cap: cv2.VideoCapture) -> None:
#         frame_no = 0
#         try:
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
#                 frame_no  += 1
#                 video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
#                 self._read_queue.put((frame, frame_no, video_time))
#         except Exception:
#             self.logger.error("Reader error:\n" + traceback.format_exc())
#         finally:
#             self._read_queue.put(_STOP)

#     def _writer_loop(self) -> None:
#         try:
#             while True:
#                 item = self._write_queue.get()
#                 if item is _STOP:
#                     break
#                 if self._writer:
#                     self._writer.write(item)
#         except Exception:
#             self.logger.error("Writer error:\n" + traceback.format_exc())

#     # ──────────────────────────────────────────────────────────────────────────
#     # _process_frame  — all timestamps are offset-adjusted here
#     # ──────────────────────────────────────────────────────────────────────────

#     def _process_frame(self, frame, video_time, raw_frame_no, processed_frame_no):
#         annotated = frame

#         # ── global values for this frame ──────────────────────────────────────
#         global_time  = video_time  + self.time_offset   # HH:MM:SS into the full recording
#         global_frame = raw_frame_no + self.frame_offset  # unique frame index across all videos

#         run_gadget  = (processed_frame_no % GADGET_EVERY  == 0)
#         run_absence = run_gadget
#         run_droop   = (processed_frame_no % DROOP_EVERY   == 0)

#         prev_frame_detection = self._prev_frame_detections

#         # ── Build per-pilot MediaPipe landmarks for gadget detector ───────────
#         pose_landmarks_by_pilot = self._get_pose_landmarks(frame) if run_gadget else None

#         future_gadget  = self.executor.submit(self.detector.process, frame, round(global_time, 3), pose_landmarks_by_pilot) if run_gadget  else None
#         future_absence = self.executor.submit(self.absence_detector.process, self._absence_pilot_boxes, global_time, frame.shape[1], frame.shape[0]) if run_absence else None
#         future_droop   = self.executor.submit(self.droop_detector.process, frame, global_time, prev_frame_detection) if run_droop else None

#         results,         log_events         = [], []
#         absence_results, absence_log_events = [], []
#         droop_results,   droop_log_events   = [], []

#         try:
#             if future_gadget  is not None: results,         log_events         = future_gadget.result()
#         except Exception as exc:
#             self.logger.error(f"Gadget error frame {global_frame}: {exc}", exc_info=True)
#         try:
#             if future_absence is not None: absence_results, absence_log_events = future_absence.result()
#         except Exception as exc:
#             self.logger.error(f"Absence error frame {global_frame}: {exc}", exc_info=True)
#         try:
#             if future_droop   is not None: droop_results,   droop_log_events   = future_droop.result()
#         except Exception as exc:
#             self.logger.error(f"Droop error frame {global_frame}: {exc}", exc_info=True)

#         if run_gadget:
#             new_boxes = [(r.pilot_id, r.bbox) for r in results]
#             self._prev_pilot_boxes      = new_boxes
#             self._prev_frame_detections = self.detector.last_frame_detections
#             if new_boxes:
#                 self._absence_pilot_boxes  = new_boxes
#                 self._yolo_empty_run_count = 0
#             else:
#                 self._yolo_empty_run_count += 1
#                 if self._yolo_empty_run_count >= 3:
#                     self._absence_pilot_boxes = []

#         # ── Draw (only when DRAW=True) ────────────────────────────────────────
#         if DRAW:
#             for g in self.detector.last_gadget_hits:
#                 draw_gadget_box(annotated, g.bbox, g.class_name, g.confidence)
#             for ar in absence_results:
#                 if ar.calibrated and ar.seat_zone is not None:
#                     draw_seat_zone(annotated, ar.seat_zone, ar.pilot_id)

#         any_gadget_distracted = False
#         last_gadget_pilot, last_gadget_name = None, ""

#         for r in results:
#             gadget_names = [g.class_name for g in r.gadgets]
#             if DRAW:
#                 draw_pilot_box(annotated, r.bbox, r.pilot_id, r.distracted, [])
#             if r.distracted:
#                 any_gadget_distracted = True
#                 last_gadget_pilot     = r.pilot_id
#                 last_gadget_name      = gadget_names[0] if gadget_names else "gadget"
#                 if DRAW:
#                     _draw_distraction_label(annotated, r.bbox, "Phone Usage", r.timer_value, color=(0,0,220))

#         any_absence_distracted = False
#         last_absent_pilot, last_absent_duration = None, 0.0

#         for ar in absence_results:
#             current_bbox = next((r.bbox for r in results if r.pilot_id == ar.pilot_id), None)
#             if DRAW:
#                 draw_absence_overlay(frame=annotated, bbox=current_bbox, pilot_id=ar.pilot_id,
#                                      absent=ar.absent, timer_val=ar.timer_value, calibrated=ar.calibrated)
#             if ar.absent:
#                 any_absence_distracted = True
#                 last_absent_pilot      = ar.pilot_id
#                 last_absent_duration   = ar.timer_value
#                 if DRAW:
#                     _draw_distraction_label(annotated, current_bbox, "Away From Seat", ar.timer_value, color=(0,140,255))

#         any_droop_distracted = False
#         last_droop_pilot, last_droop_duration, last_droop_severity = None, 0.0, "DROWSINESS"
#         bbox_by_pid = {r.pilot_id: r.bbox for r in results} if droop_results else {}

#         for dr in droop_results:
#             current_bbox = bbox_by_pid.get(dr.pilot_id)
#             if not dr.is_seated:
#                 if DRAW: draw_standing_label(annotated, dr.pilot_id, current_bbox)
#                 continue
#             if DRAW:
#                 if hasattr(dr, "keypoints") and dr.keypoints:
#                     draw_droop_keypoints(frame=annotated, keypoints=dr.keypoints, pilot_id=dr.pilot_id,
#                                          drooping=dr.drooping, angle=getattr(dr, "angle", 0.0))
#                 draw_droop_overlay(frame=annotated, pilot_id=dr.pilot_id, drooping=dr.drooping,
#                                    timer_val=dr.timer_value, bbox=current_bbox,
#                                    severity=getattr(dr, "severity", "DROWSINESS"))
#             if dr.drooping:
#                 any_droop_distracted = True
#                 last_droop_pilot     = dr.pilot_id
#                 last_droop_duration  = dr.timer_value
#                 last_droop_severity  = getattr(dr, "severity", "DROWSINESS")
#                 if DRAW:
#                     _draw_distraction_label(annotated, current_bbox, last_droop_severity,
#                                             dr.timer_value*(38/25.0),
#                                             color=(0,200,255) if last_droop_severity=="DROWSINESS" else (0,80,200))

#         if DRAW:
#             if any_gadget_distracted  and last_gadget_pilot  is not None: draw_alert_banner(annotated, last_gadget_pilot,  last_gadget_name)
#             if any_absence_distracted and last_absent_pilot  is not None: draw_absence_banner(annotated, last_absent_pilot, last_absent_duration)
#             if any_droop_distracted   and last_droop_pilot   is not None: draw_droop_banner(annotated,  last_droop_pilot,  last_droop_duration, severity=last_droop_severity)
#             for dr in droop_results:
#                 if not dr.drooping: continue
#                 if any(ar.absent and ar.pilot_id == dr.pilot_id for ar in absence_results):
#                     _draw_distraction_label(annotated, bbox_by_pid.get(dr.pilot_id), "SLEEPING / ABSENT", dr.timer_value, color=(0,50,200))
#             draw_hud(annotated, global_time, global_frame, len(results))

#         # ── Record violations ─────────────────────────────────────────────────
#         # video_time      = local time within THIS file  → local_video_time param
#         # global_time     = video_time + time_offset     → video_time param (→ "timestamp")
#         # global_frame    = raw_frame_no + frame_offset  → frame_index param
#         # source_filename = DB filename                  → shown in original_video_timestamp

#         if log_events:
#             r_ref        = next((r for r in results if r.distracted), None)
#             conf         = r_ref.gadgets[0].confidence if (r_ref and r_ref.gadgets) else 0.9
#             dur          = r_ref.timer_value if r_ref else 0.0
#             event_global = max(0.0, global_time - GADGET_ALLOWED_DURATION)
#             event_local  = max(0.0, video_time  - GADGET_ALLOWED_DURATION)
#             self.vstore.record_violation(
#                 annotated_frame=annotated, original_frame=frame,
#                 video_time=event_global, frame_index=global_frame,
#                 event_type="phone_use", severity="CRITICAL",
#                 confidence=conf, risk_score=80, risk_level="CRITICAL",
#                 factors=["phone_use", "distraction"], duration=dur,
#                 source_filename=self.source_filename, local_video_time=event_local,
#             )
#             log_distraction(self.logger, event_global, event="One of the pilots is using a mobile phone", severity="CRITICAL", frame=annotated)

#         if absence_log_events:
#             ar_ref       = next((ar for ar in absence_results if ar.absent), None)
#             dur_abs      = ar_ref.timer_value if ar_ref else 0.0
#             event_global = max(0.0, global_time - ABSENCE_ALLOWED_DURATION)
#             event_local  = max(0.0, video_time  - ABSENCE_ALLOWED_DURATION)
#             self.vstore.record_violation(
#                 annotated_frame=annotated, original_frame=frame,
#                 video_time=event_global, frame_index=global_frame,
#                 event_type="seat_absence", severity="CRITICAL",
#                 confidence=1.0, risk_score=70, risk_level="CRITICAL",
#                 factors=["seat_absence"], duration=dur_abs,
#                 source_filename=self.source_filename, local_video_time=event_local,
#             )
#             log_distraction(self.logger, event_global, event="One of the pilots is away from the seat", severity="CRITICAL", frame=annotated)

#         if droop_log_events:
#             severities  = [e[1] for e in droop_log_events]
#             is_sleeping = any("SLEEPING" in s for s in severities)
#             droop_pids  = {e[0] for e in droop_log_events}
#             absent_pids = {ar.pilot_id for ar in absence_results if ar.absent}
#             also_absent = bool(droop_pids & absent_pids)

#             if also_absent:  event_msg, etype = "One of the pilots is sleeping / slumped in seat", "sleeping_absent"
#             elif is_sleeping: event_msg, etype = "One of the pilots is sleeping", "sleeping"
#             else:             event_msg, etype = "One of the pilots is drowsy",   "drowsy"

#             dr_ref       = next((dr for dr in droop_results if dr.drooping), None)
#             dur_drp      = dr_ref.timer_value if dr_ref else 0.0
#             event_global = max(0.0, global_time - HEAD_DROP_DURATION)
#             event_local  = max(0.0, video_time  - HEAD_DROP_DURATION)
#             self.vstore.record_violation(
#                 annotated_frame=annotated, original_frame=frame,
#                 video_time=event_global, frame_index=global_frame,
#                 event_type=etype, severity="CRITICAL",
#                 confidence=0.9, risk_score=75, risk_level="HIGH",
#                 factors=["drowsy", "head_droop"], duration=dur_drp,
#                 source_filename=self.source_filename, local_video_time=event_local,
#             )
#             log_distraction(self.logger, event_global, event=event_msg, severity="CRITICAL", frame=annotated)

#         return annotated

#     def _get_pose_landmarks(self, frame: np.ndarray) -> Optional[dict]:
#         """
#         Run MediaPipe Pose on the full frame and return a dict:
#             { pilot_id: [landmark_0, ..., landmark_32] }

#         The frame is split at split_y (57 % of height) to assign each
#         detected person's landmarks to Pilot 1 or Pilot 2 independently,
#         using the same zone logic as the YOLO pilot assignment.

#         MediaPipe processes the full frame in one call (it detects up to 1
#         person by default). For a two-person frame we crop each pilot's half
#         and run MediaPipe on each crop separately.

#         Returns None if MediaPipe is not installed or fails.
#         """
#         if self._pose is None:
#             return None

#         try:
#             h = self._frame_height
#             w = self._frame_width
#             split_y = int(h * 0.57)

#             # Crop each pilot zone and run MediaPipe independently.
#             # Pilot 2 = top half (y: 0 → split_y)
#             # Pilot 1 = bottom half (y: split_y → h)
#             zones = {
#                 2: frame[0:split_y, 0:w],
#                 1: frame[split_y:h,  0:w],
#             }

#             result: dict = {}
#             for pid, crop in zones.items():
#                 if crop.size == 0:
#                     continue
#                 rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
#                 mp_result = self._pose.process(rgb)
#                 if mp_result.pose_landmarks is None:
#                     continue

#                 lms = mp_result.pose_landmarks.landmark

#                 # Adjust Y coordinates back to full-frame pixel space.
#                 # Pilot 1 crop starts at split_y in the full frame.
#                 y_offset = split_y if pid == 1 else 0

#                 # We store the raw landmark objects but patch their .y so
#                 # that downstream code using lm.y * frame_h gives the
#                 # correct FULL-FRAME pixel coordinate.
#                 patched = []
#                 for lm in lms:
#                     crop_h = crop.shape[0]
#                     # lm.y is normalised to the crop height; convert to
#                     # normalised full-frame Y.
#                     full_y = (lm.y * crop_h + y_offset) / h
#                     # Build a simple namespace so downstream can use lm.x / lm.y
#                     patched.append(_PatchedLandmark(
#                         x          = lm.x,          # X is same (full width used)
#                         y          = full_y,
#                         visibility = getattr(lm, "visibility", 1.0),
#                     ))
#                 result[pid] = patched

#             return result if result else None

#         except Exception:
#             return None

#     def _print_banner(self, fps, w, h, total):
#         self.logger.info(f"\n{'='*60}\n  LOCO PILOT DISTRACTION DETECTION\n  Analysis ID : {self.analysis_id}\n  Source      : {self.source}\n  Video       : {w}x{h} @ {fps:.1f} fps ({total} frames)\n  Output      : {OUTPUT_PATH if self.save else 'disabled'}\n{'='*60}\n")

#     def _print_summary(self, frame_no, processing_time):
#         self.logger.info(f"\n{'='*60}\n  Processing complete\n  Raw frames  : {frame_no}\n  Processed   : {self._processed_frame_no} (1 in every {RAW_FRAME_SKIP})\n  Time        : {processing_time:.2f}s\n  Report : outputs/{self.analysis_id}/analysis_report.json\n  Frames : outputs/{self.analysis_id}/frames/\n{'='*60}\n")


# def parse_args():
#     p = argparse.ArgumentParser(description="Loco Pilot Distraction Detection")
#     p.add_argument("--source",          default=0)
#     p.add_argument("--analysis-id",     default=None)
#     p.add_argument("--train-detail-id", default=0, type=int)
#     p.add_argument("--no-display",      action="store_true")
#     p.add_argument("--no-save",         action="store_true")
#     return p.parse_args()


# if __name__ == "__main__":
#     args   = parse_args()
#     source = args.source
#     if isinstance(source, str) and source.isdigit():
#         source = int(source)
#     GadgetDetectionPipeline(
#         source=source, analysis_id=args.analysis_id,
#         # train_detail_id=args.train_detail_id,
#         save=not args.no_save, display=False,
#     ).run() 


# # from __future__ import annotations

# # import argparse
# # import os
# # import queue
# # import re
# # import sys
# # import threading
# # import traceback
# # import uuid
# # from typing import Optional
# # from concurrent.futures import ThreadPoolExecutor
# # import warnings
# # import cv2
# # import numpy as np


# # DRAW           = False  # set True only for visual debug
# # RAW_FRAME_SKIP = 3      # process 1 in every N raw frames
# # GADGET_EVERY   = 6      # YOLO  every Nth processed frame
# # ABSENCE_EVERY  = 4      # absence every Nth processed frame
# # DROOP_EVERY    = 15     # droop every Nth processed frame
# #  # allowed duration in seconds before logging violation


# # from config.settings import OUTPUT_PATH, WINDOW_NAME, DISPLAY_SCALE,GADGET_ALLOWED_DURATION,ABSENCE_ALLOWED_DURATION,HEAD_DROP_DURATION
# # from utils.logger import setup_logger, log_distraction, finalize_report
# # from utils.violation_store import ViolationStore
# # from utils.draw import (
# #     draw_pilot_box, draw_gadget_box, draw_hud, draw_alert_banner,
# #     draw_seat_zone, draw_absence_overlay, draw_absence_banner,
# #     draw_droop_keypoints, draw_droop_overlay, draw_droop_banner,
# #     draw_standing_label,
# # )
# # from detector.gadget_detector import GadgetDetector
# # from detector.seat_absence_detector import SeatAbsenceDetector
# # from detector.head_drop_detector import HeadDroopDetector

# # _STOP = object()

# # READ_QUEUE_MAXSIZE  = 8
# # WRITE_QUEUE_MAXSIZE = 8

# # warnings.filterwarnings("ignore", category=UserWarning)


# # def _draw_distraction_label(
# #     frame: np.ndarray,
# #     bbox: tuple,
# #     distraction_type: str,
# #     timer_val: float,
# #     color: tuple = (0, 0, 255),
# # ) -> None:
# #     if bbox is None:
# #         return
# #     x1, y1, x2, y2 = bbox
# #     label = f"{distraction_type}  {timer_val:.1f}s"
# #     font       = cv2.FONT_HERSHEY_DUPLEX
# #     font_scale = 0.52
# #     thickness  = 1
# #     (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
# #     pad    = 4
# #     tag_y2 = max(y1, th + pad * 2)
# #     tag_y1 = tag_y2 - th - pad * 2
# #     tag_x2 = x1 + tw + pad * 2
# #     cv2.rectangle(frame, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
# #     cv2.putText(
# #         frame, label,
# #         (x1 + pad, tag_y2 - pad - baseline // 2),
# #         font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
# #     )


# # class GadgetDetectionPipeline:

# #     def __init__(
# #         self,
# #         source:          str | int,
# #         analysis_id:     Optional[str] = None,
# #         train_detail_id: int           = 0,
# #         save:            bool          = False,
# #         display:         bool          = False,
# #         time_offset:     float         = 0.0,
# #         frame_offset:    int           = 0,       # ← NEW: cumulative frame count before this video
# #         shared_vstore=None,
# #         original_filename: Optional[str] = None,
# #     ) -> None:
# #         self.source            = source
# #         self.train_detail_id   = train_detail_id
# #         self.save              = save
# #         self.display           = display
# #         self.time_offset       = time_offset      # cumulative seconds before this video
# #         self.frame_offset      = frame_offset     # cumulative frames before this video
# #         self.shared_vstore     = shared_vstore    # if set, use this instead of creating new one
# #         self.original_filename = original_filename  # real upload name, overrides tmp path basename

# #         if analysis_id:
# #             self.analysis_id = analysis_id
# #         elif (
# #             isinstance(source, str)
# #             and source not in ("0",)
# #             and os.path.isfile(source)
# #         ):
# #             stem             = os.path.splitext(os.path.basename(source))[0]
# #             self.analysis_id = re.sub(r"[^A-Za-z0-9_-]", "_", stem)
# #         else:
# #             self.analysis_id = uuid.uuid4().hex[:8]

# #         self.logger           = setup_logger()
# #         self.detector         = GadgetDetector()
# #         self.absence_detector = SeatAbsenceDetector()
# #         self.droop_detector   = HeadDroopDetector()
# #         self._writer:  Optional[cv2.VideoWriter] = None
# #         self.vstore:   Optional[ViolationStore]  = None

# #         # 3 workers: one per detector, no excess overhead
# #         self.executor = ThreadPoolExecutor(max_workers=3)

# #         self._prev_pilot_boxes      = []
# #         self._prev_frame_detections = None
# #         self._processed_frame_no    = 0   

# #         self._read_queue:  queue.Queue = queue.Queue(maxsize=READ_QUEUE_MAXSIZE)
# #         self._write_queue: queue.Queue = queue.Queue(maxsize=WRITE_QUEUE_MAXSIZE)

    
# #     # ENTRY POINT
    

# #     def run(self) -> tuple:
# #         """Returns (report_path, duration_seconds, total_frame_count)."""
# #         import time
# #         start_time = time.time()

# #         cap = cv2.VideoCapture(self.source)
# #         if not cap.isOpened():
# #             self.logger.error(f"Cannot open source: {self.source!r}")
# #             sys.exit(1)

# #         _raw_fps = cap.get(cv2.CAP_PROP_FPS)
# #         if not _raw_fps:
# #             print("[WARNING] FPS not detected — defaulting to 25.0")
# #         fps    = _raw_fps or 25.0
# #         width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# #         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# #         total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# #         print(f"Video      : {width}x{height} @ {fps:.1f}fps  "
# #               f"{total/fps:.1f}s  {total} frames")
# #         print(f"Analysis ID: {self.analysis_id}")
# #         print(f"Processing : every {RAW_FRAME_SKIP}rd raw frame  |  "
# #               f"YOLO every {RAW_FRAME_SKIP * GADGET_EVERY} raw frames  |  "
# #               f"Droop every {RAW_FRAME_SKIP * DROOP_EVERY} raw frames")

# #         source_str  = str(self.source)
# #         # Prefer the real uploaded filename passed in from api.py;
# #         # fall back to basename of the (temp) path for direct/CLI use.
# #         source_name = (
# #             self.original_filename
# #             if self.original_filename
# #             else (os.path.basename(source_str) if isinstance(self.source, str) else "webcam")
# #         )
# #         # Seek to end to get true duration — handles VFR and mismatched fps tags
# #         # (e.g. cabin_video.mp4 has container fps=30 but actual fps=6)
# #         cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
# #         duration_s = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 3)
# #         cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # rewind for processing
# #         if duration_s <= 0:
# #             # Fallback for sources that don't support seek-to-end
# #             duration_s = round(total / fps, 3) if total > 0 and fps > 0 else 0.0
# #         h, m, s    = (
# #             int(duration_s) // 3600,
# #             (int(duration_s) % 3600) // 60,
# #             int(duration_s) % 60,
# #         )
# #         size_mb = (
# #             round(os.path.getsize(source_str) / 1_000_000, 2)
# #             if isinstance(self.source, str) and os.path.isfile(source_str) else 0
# #         )

# #         video_info = {
# #             "filename":          source_name,
# #             "videoPath":         source_str,
# #             "durationSeconds":   duration_s,
# #             "durationFormatted": f"{h}:{m:02d}:{s:02d}",
# #             "resolution":        f"{width}x{height}",
# #             "fps":               round(fps, 3),
# #             "totalFrames":       total,
# #             "sizeMb":            size_mb,
# #         }

# #         if self.shared_vstore is not None:
# #             # Batch mode: attach this video's metadata to the shared store
# #             self.vstore = self.shared_vstore
# #             self.vstore.add_video_info(video_info)
# #         else:
# #             self.vstore = ViolationStore(
# #                 analysis_id     = self.analysis_id,
# #                 train_detail_id = self.train_detail_id,
# #                 video_info      = video_info,
# #             )
# #         self._print_banner(fps, width, height, total)

# #         if self.save:
# #             os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
# #             self._writer = cv2.VideoWriter(
# #                 OUTPUT_PATH,
# #                 cv2.VideoWriter_fourcc(*"mp4v"),
# #                 fps,
# #                 (width, height),
# #             )

# #         raw_frame_no = 0
# #         report_path  = ""

# #         reader_thread = threading.Thread(
# #             target=self._reader_loop, args=(cap,),
# #             daemon=True, name="FrameReader",
# #         )
# #         writer_thread = threading.Thread(
# #             target=self._writer_loop,
# #             daemon=True, name="FrameWriter",
# #         )
# #         reader_thread.start()
# #         writer_thread.start()

# #         try:
# #             while True:
# #                 item = self._read_queue.get()
# #                 if item is _STOP:
# #                     break

# #                 raw_frame, raw_frame_no, video_time = item

# #                 # ── Skip most raw frames — pass through as-is ─────
# #                 # raw_frame_no here is already globally offset so the
# #                 # modulo cadence is kept consistent across videos.
# #                 if raw_frame_no % RAW_FRAME_SKIP != 0:
# #                     self._write_queue.put(raw_frame)
# #                     continue

# #                 # ── Process this frame ────────────────────────────
# #                 self._processed_frame_no += 1
# #                 annotated = self._process_frame(
# #                     raw_frame, video_time, raw_frame_no, self._processed_frame_no
# #                 )
# #                 self._write_queue.put(annotated)

# #                 if self.display:
# #                     show = annotated
# #                     if DISPLAY_SCALE != 1.0:
# #                         show = cv2.resize(
# #                             annotated,
# #                             (int(width * DISPLAY_SCALE), int(height * DISPLAY_SCALE)),
# #                         )
# #                     cv2.imshow(WINDOW_NAME, show)
# #                     key = cv2.waitKey(1) & 0xFF
# #                     if key in (ord("q"), 27):
# #                         self.logger.info("Quit by user.")
# #                         break

# #         except KeyboardInterrupt:
# #             self.logger.info("\nInterrupted by user.")
# #         except Exception:
# #             self.logger.error("Unexpected error:\n" + traceback.format_exc())
# #         finally:
# #             self._write_queue.put(_STOP)
# #             writer_thread.join(timeout=30)
# #             cap.release()
# #             if self._writer:
# #                 self._writer.release()
# #             if self.display:
# #                 cv2.destroyAllWindows()

# #             processing_time = round(time.time() - start_time, 3)
# #             self._print_summary(raw_frame_no, processing_time)
# #             finalize_report()
# #             # In batch mode (shared_vstore) we do NOT finalize here —
# #             # api.py finalizes the shared store once after ALL videos are done.
# #             if self.shared_vstore is None:
# #                 report_path = self.vstore.finalize(processing_time=processing_time)
# #             else:
# #                 report_path = ""   # will be set by api.py after last video

# #         actual_fps = raw_frame_no / processing_time if processing_time > 0 else 0
# #         print(f"\nTotal Time : {processing_time:.2f}s   FPS : {actual_fps:.2f}")
# #         # Return duration_s and total so api.py can accumulate offsets
# #         # without re-opening the (already-deleted) temp file.
# #         return report_path, duration_s, total

 
# #     # READER THREAD
   

# #     def _reader_loop(self, cap: cv2.VideoCapture) -> None:
# #         # frame_no counts frames within THIS video (1-based).
# #         # We add self.frame_offset so every frame has a globally unique
# #         # index across the entire batch — prevents deduplication collisions
# #         # in ViolationStore._seen_frames when two videos share the same
# #         # local frame numbers.
# #         frame_no = 0
# #         try:
# #             while True:
# #                 ret, frame = cap.read()
# #                 if not ret:
# #                     break
# #                 frame_no  += 1
# #                 global_frame_no = frame_no + self.frame_offset          # ← CHANGED
# #                 video_time      = (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0) + self.time_offset
# #                 self._read_queue.put((frame, global_frame_no, video_time))
# #         except Exception:
# #             self.logger.error("Reader error:\n" + traceback.format_exc())
# #         finally:
# #             self._read_queue.put(_STOP)

    
# #     # WRITER THREAD
    

# #     def _writer_loop(self) -> None:
# #         try:
# #             while True:
# #                 item = self._write_queue.get()
# #                 if item is _STOP:
# #                     break
# #                 if self._writer:
# #                     self._writer.write(item)
# #         except Exception:
# #             self.logger.error("Writer error:\n" + traceback.format_exc())

    
# #     # PER-FRAME PROCESSING
    

# #     def _process_frame(
# #         self,
# #         frame:              np.ndarray,
# #         video_time:         float,
# #         raw_frame_no:       int,
# #         processed_frame_no: int,
# #     ) -> np.ndarray:
# #         annotated = frame

# #         # Cadences are relative to processed_frame_no so that the effective ML frequency is consistent regardless of RAW_FRAME_SKIP.
# #         run_gadget  = (processed_frame_no % GADGET_EVERY  == 0)
# #         run_absence = (processed_frame_no % ABSENCE_EVERY == 0)
# #         run_droop   = (processed_frame_no % DROOP_EVERY   == 0)

# #         prev_pilot_boxes     = self._prev_pilot_boxes
# #         prev_frame_detection = self._prev_frame_detections

# #         future_gadget = (
# #             self.executor.submit(self.detector.process, frame, round(video_time, 0))
# #             if run_gadget else None
# #         )
# #         future_absence = (
# #             self.executor.submit(
# #                 self.absence_detector.process,
# #                 prev_pilot_boxes, video_time, frame.shape[1], frame.shape[0],
# #             )
# #             if run_absence else None
# #         )
# #         future_droop = (
# #             self.executor.submit(
# #                 self.droop_detector.process,
# #                 frame, video_time, prev_frame_detection,
# #             )
# #             if run_droop else None
# #         )

# #         results,         log_events         = [], []
# #         absence_results, absence_log_events = [], []
# #         droop_results,   droop_log_events   = [], []

# #         try:
# #             if future_gadget is not None:
# #                 results, log_events = future_gadget.result()
# #         except Exception as exc:
# #             self.logger.error(f"Gadget error frame {raw_frame_no}: {exc}", exc_info=True)

# #         try:
# #             if future_absence is not None:
# #                 absence_results, absence_log_events = future_absence.result()
# #         except Exception as exc:
# #             self.logger.error(f"Absence error frame {raw_frame_no}: {exc}", exc_info=True)

# #         try:
# #             if future_droop is not None:
# #                 droop_results, droop_log_events = future_droop.result()
# #         except Exception as exc:
# #             self.logger.error(f"Droop error frame {raw_frame_no}: {exc}", exc_info=True)

# #         if run_gadget:
# #             self._prev_pilot_boxes      = [(r.pilot_id, r.bbox) for r in results]
# #             self._prev_frame_detections = self.detector.last_frame_detections

# #         #  Draw (skipped entirely when DRAW=False
# #         if DRAW:
# #             for g in self.detector.last_gadget_hits:
# #                 draw_gadget_box(annotated, g.bbox, g.class_name, g.confidence)
# #             for ar in absence_results:
# #                 if ar.calibrated and ar.seat_zone is not None:
# #                     draw_seat_zone(annotated, ar.seat_zone, ar.pilot_id)

# #         any_gadget_distracted = False
# #         last_gadget_pilot     = None
# #         last_gadget_name      = ""

# #         for r in results:
# #             gadget_names = [g.class_name for g in r.gadgets]
# #             if DRAW:
# #                 draw_pilot_box(annotated, r.bbox, r.pilot_id, r.distracted, [])
# #             if r.distracted:
# #                 any_gadget_distracted = True
# #                 last_gadget_pilot     = r.pilot_id
# #                 last_gadget_name      = gadget_names[0] if gadget_names else "gadget"
# #                 if DRAW:
# #                     _draw_distraction_label(annotated, r.bbox, "Phone Usage",
# #                                             r.timer_value, color=(0, 0, 220))

# #         any_absence_distracted = False
# #         last_absent_pilot      = None
# #         last_absent_duration   = 0.0

# #         for ar in absence_results:
# #             current_bbox = next(
# #                 (r.bbox for r in results if r.pilot_id == ar.pilot_id), None
# #             )
# #             if DRAW:
# #                 draw_absence_overlay(
# #                     frame=annotated, bbox=current_bbox, pilot_id=ar.pilot_id,
# #                     absent=ar.absent, timer_val=ar.timer_value,
# #                     calibrated=ar.calibrated,
# #                 )
# #             if ar.absent:
# #                 any_absence_distracted = True
# #                 last_absent_pilot      = ar.pilot_id
# #                 last_absent_duration   = ar.timer_value
# #                 if DRAW:
# #                     _draw_distraction_label(annotated, current_bbox, "Away From Seat",
# #                                             ar.timer_value, color=(0, 140, 255))

# #         any_droop_distracted = False
# #         last_droop_pilot     = None
# #         last_droop_duration  = 0.0
# #         last_droop_severity  = "DROWSINESS"
# #         bbox_by_pid          = {}

# #         if droop_results:
# #             bbox_by_pid = {r.pilot_id: r.bbox for r in results}

# #         for dr in droop_results:
# #             current_bbox = bbox_by_pid.get(dr.pilot_id)
# #             if not dr.is_seated:
# #                 if DRAW:
# #                     draw_standing_label(annotated, dr.pilot_id, current_bbox)
# #                 continue
# #             if hasattr(dr, "keypoints") and dr.keypoints:
# #                 if DRAW:
# #                     draw_droop_keypoints(
# #                         frame=annotated, keypoints=dr.keypoints,
# #                         pilot_id=dr.pilot_id, drooping=dr.drooping,
# #                         angle=getattr(dr, "angle", 0.0),
# #                     )
# #             if DRAW:
# #                 draw_droop_overlay(
# #                     frame=annotated, pilot_id=dr.pilot_id,
# #                     drooping=dr.drooping, timer_val=dr.timer_value,
# #                     bbox=current_bbox,
# #                     severity=getattr(dr, "severity", "DROWSINESS"),
# #                 )
# #             if dr.drooping:
# #                 any_droop_distracted = True
# #                 last_droop_pilot     = dr.pilot_id
# #                 last_droop_duration  = dr.timer_value
# #                 last_droop_severity  = getattr(dr, "severity", "DROWSINESS")
# #                 display_secs         = dr.timer_value * (38 / 25.0)
# #                 if DRAW:
# #                     _draw_distraction_label(
# #                         annotated, current_bbox, last_droop_severity, display_secs,
# #                         color=(0, 200, 255)
# #                         if last_droop_severity == "DROWSINESS" else (0, 80, 200),
# #                     )

# #         if DRAW:
# #             if any_gadget_distracted and last_gadget_pilot is not None:
# #                 draw_alert_banner(annotated, last_gadget_pilot, last_gadget_name)
# #             if any_absence_distracted and last_absent_pilot is not None:
# #                 draw_absence_banner(annotated, last_absent_pilot, last_absent_duration)
# #             if any_droop_distracted and last_droop_pilot is not None:
# #                 draw_droop_banner(annotated, last_droop_pilot, last_droop_duration,
# #                                   severity=last_droop_severity)
# #             for dr in droop_results:
# #                 if not dr.drooping:
# #                     continue
# #                 if any(ar.absent and ar.pilot_id == dr.pilot_id
# #                        for ar in absence_results):
# #                     cb = bbox_by_pid.get(dr.pilot_id)
# #                     _draw_distraction_label(annotated, cb, "SLEEPING / ABSENT",
# #                                             dr.timer_value, color=(0, 50, 200))
# #             draw_hud(annotated, video_time, raw_frame_no, len(results))

# #         #  Log + store violations
# #         if log_events:
# #             r_ref = next((r for r in results if r.distracted), None)
# #             conf  = r_ref.gadgets[0].confidence if (r_ref and r_ref.gadgets) else 0.9
# #             dur   = r_ref.timer_value if r_ref else 0.0
# #             # Clamp to time_offset floor so we never produce a timestamp
# #             # earlier than the start of this video in the combined timeline.
# #             event_time = max(self.time_offset, video_time - GADGET_ALLOWED_DURATION)  # ← CHANGED
# #             self.vstore.record_violation(
# #                 annotated_frame=annotated, original_frame=frame,
# #                 video_time=event_time, frame_index=raw_frame_no,
# #                 event_type="phone_use", severity="CRITICAL",
# #                 confidence=conf, risk_score=80, risk_level="CRITICAL",
# #                 factors=["phone_use"], duration=dur,
# #             )
# #             log_distraction(self.logger, event_time,
# #                             event="One of the pilots is using a mobile phone",
# #                             severity="CRITICAL", frame=annotated)

# #         if absence_log_events:
# #             ar_ref  = next((ar for ar in absence_results if ar.absent), None)
# #             dur_abs = ar_ref.timer_value if ar_ref else 0.0
# #             event_time = max(self.time_offset, video_time - ABSENCE_ALLOWED_DURATION)  # ← CHANGED
# #             self.vstore.record_violation(
# #                 annotated_frame=annotated, original_frame=frame,
# #                 video_time=event_time, frame_index=raw_frame_no,
# #                 event_type="seat_absence", severity="CRITICAL",
# #                 confidence=1.0, risk_score=70, risk_level="CRITICAL",
# #                 factors=["seat_absence"], duration=dur_abs,
# #             )
# #             log_distraction(self.logger, event_time,
# #                             event="One of the pilots is away from the seat",
# #                             severity="CRITICAL", frame=annotated)

# #         if droop_log_events:
# #             severities  = [e[1] for e in droop_log_events]
# #             is_sleeping = any("SLEEPING" in s for s in severities)
# #             droop_pids  = {e[0] for e in droop_log_events}
# #             absent_pids = {ar.pilot_id for ar in absence_results if ar.absent}
# #             also_absent = bool(droop_pids & absent_pids)

# #             if also_absent:
# #                 event_msg = "One of the pilots is sleeping / slumped in seat"
# #                 etype     = "sleeping_absent"
# #             elif is_sleeping:
# #                 event_msg = "One of the pilots is sleeping"
# #                 etype     = "sleeping"
# #             else:
# #                 event_msg = "One of the pilots is drowsy"
# #                 etype     = "drowsy"

# #             dr_ref  = next((dr for dr in droop_results if dr.drooping), None)
# #             dur_drp = dr_ref.timer_value if dr_ref else 0.0
# #             event_time = max(self.time_offset, video_time - HEAD_DROP_DURATION)  # ← CHANGED
# #             self.vstore.record_violation(
# #                 annotated_frame=annotated, original_frame=frame,
# #                 video_time=event_time, frame_index=raw_frame_no,
# #                 event_type=etype, severity="CRITICAL",
# #                 confidence=0.9, risk_score=75, risk_level="HIGH",
# #                 factors=["drowsy", "head_droop"], duration=dur_drp,
# #             )
# #             log_distraction(self.logger, event_time, event=event_msg,
# #                             severity="CRITICAL", frame=annotated)

# #         return annotated

    
# #     # HELPERS
    

# #     def _print_banner(self, fps: float, w: int, h: int, total: int) -> None:
# #         self.logger.info(
# #             f"\n{'='*60}\n"
# #             f"  LOCO PILOT DISTRACTION DETECTION\n"
# #             f"  Analysis ID : {self.analysis_id}\n"
# #             f"  Source      : {self.source}\n"
# #             f"  Video       : {w}x{h} @ {fps:.1f} fps ({total} frames)\n"
# #             f"  Output      : {OUTPUT_PATH if self.save else 'disabled'}\n"
# #             f"{'='*60}\n"
# #         )

# #     def _print_summary(self, frame_no: int, processing_time: float) -> None:
# #         self.logger.info(
# #             f"\n{'='*60}\n"
# #             f"  Processing complete\n"
# #             f"  Raw frames  : {frame_no}\n"
# #             f"  Processed   : {self._processed_frame_no} "
# #             f"(1 in every {RAW_FRAME_SKIP})\n"
# #             f"  Time        : {processing_time:.2f}s\n"
# #             f"  Report : outputs/{self.analysis_id}/analysis_report.json\n"
# #             f"  Frames : outputs/{self.analysis_id}/frames/\n"
# #             f"{'='*60}\n"
# #         )



# # # CLI


# # def parse_args() -> argparse.Namespace:
# #     p = argparse.ArgumentParser(description="Loco Pilot Distraction Detection")
# #     p.add_argument("--source",          default=0,
# #                    help="Video file path or camera index (default: 0 = webcam)")
# #     p.add_argument("--analysis-id",     default=None)
# #     p.add_argument("--train-detail-id", default=0, type=int)
# #     p.add_argument("--no-display",      action="store_true")
# #     p.add_argument("--no-save",         action="store_true")
# #     return p.parse_args()


# # if __name__ == "__main__":
# #     args   = parse_args()
# #     source = args.source
# #     if isinstance(source, str) and source.isdigit():
# #         source = int(source)

# #     GadgetDetectionPipeline(
# #         source          = source,
# #         analysis_id     = args.analysis_id,
# #         train_detail_id = args.train_detail_id,
# #         save            = not args.no_save,
# #         display         = False,
# #     ).run()



# from __future__ import annotations

# import argparse
# import os
# import queue
# import re
# import sys
# import threading
# import traceback
# import uuid
# from typing import Optional
# from concurrent.futures import ThreadPoolExecutor
# import warnings
# import cv2
# import numpy as np


# DRAW           = False  # set True only for visual debug
# RAW_FRAME_SKIP = 3      # process 1 in every N raw frames
# GADGET_EVERY   = 6      # YOLO  every Nth processed frame
# ABSENCE_EVERY  = 4      # absence every Nth processed frame
# DROOP_EVERY    = 15     # droop every Nth processed frame


# from config.settings import (
#     OUTPUT_PATH, WINDOW_NAME, DISPLAY_SCALE,
#     GADGET_ALLOWED_DURATION, ABSENCE_ALLOWED_DURATION, HEAD_DROP_DURATION,
# )
# from utils.logger import setup_logger, log_distraction, finalize_report
# from utils.violation_store import ViolationStore
# from utils.draw import (
#     draw_pilot_box, draw_gadget_box, draw_hud, draw_alert_banner,
#     draw_seat_zone, draw_absence_overlay, draw_absence_banner,
#     draw_droop_keypoints, draw_droop_overlay, draw_droop_banner,
#     draw_standing_label,
# )
# from detector.gadget_detector import GadgetDetector
# from detector.seat_absence_detector import SeatAbsenceDetector
# from detector.head_drop_detector import HeadDroopDetector

# _STOP = object()

# READ_QUEUE_MAXSIZE  = 8
# WRITE_QUEUE_MAXSIZE = 8

# warnings.filterwarnings("ignore", category=UserWarning)


# def _draw_distraction_label(
#     frame: np.ndarray,
#     bbox: tuple,
#     distraction_type: str,
#     timer_val: float,
#     color: tuple = (0, 0, 255),
# ) -> None:
#     if bbox is None:
#         return
#     x1, y1, x2, y2 = bbox
#     label = f"{distraction_type}  {timer_val:.1f}s"
#     font       = cv2.FONT_HERSHEY_DUPLEX
#     font_scale = 0.52
#     thickness  = 1
#     (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
#     pad    = 4
#     tag_y2 = max(y1, th + pad * 2)
#     tag_y1 = tag_y2 - th - pad * 2
#     tag_x2 = x1 + tw + pad * 2
#     cv2.rectangle(frame, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
#     cv2.putText(
#         frame, label,
#         (x1 + pad, tag_y2 - pad - baseline // 2),
#         font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
#     )


# class GadgetDetectionPipeline:

#     def __init__(
#         self,
#         source:            str | int,
#         analysis_id:       Optional[str] = None,
#         train_detail_id:   int           = 0,
#         save:              bool          = False,
#         display:           bool          = False,
#         time_offset:       float         = 0.0,
#         frame_offset:      int           = 0,
#         shared_vstore                    = None,
#         original_filename: Optional[str] = None,
#     ) -> None:
#         self.source            = source
#         self.train_detail_id   = train_detail_id
#         self.save              = save
#         self.display           = display
#         self.time_offset       = time_offset       # cumulative seconds before this video
#         self.frame_offset      = frame_offset      # cumulative frames before this video
#         self.shared_vstore     = shared_vstore
#         self.original_filename = original_filename # real upload filename, not the tmp path

#         if analysis_id:
#             self.analysis_id = analysis_id
#         elif (
#             isinstance(source, str)
#             and source not in ("0",)
#             and os.path.isfile(source)
#         ):
#             stem             = os.path.splitext(os.path.basename(source))[0]
#             self.analysis_id = re.sub(r"[^A-Za-z0-9_-]", "_", stem)
#         else:
#             self.analysis_id = uuid.uuid4().hex[:8]

#         self.logger           = setup_logger()
#         self.detector         = GadgetDetector()
#         self.absence_detector = SeatAbsenceDetector()
#         self.droop_detector   = HeadDroopDetector()
#         self._writer: Optional[cv2.VideoWriter] = None
#         self.vstore: Optional[ViolationStore]   = None

#         self.executor = ThreadPoolExecutor(max_workers=3)

#         self._prev_pilot_boxes      = []
#         self._prev_frame_detections = None
#         self._processed_frame_no    = 0

#         self._read_queue:  queue.Queue = queue.Queue(maxsize=READ_QUEUE_MAXSIZE)
#         self._write_queue: queue.Queue = queue.Queue(maxsize=WRITE_QUEUE_MAXSIZE)


#     # ── ENTRY POINT ───────────────────────────────────────────────

#     def run(self) -> tuple:
#         """Returns (report_path, duration_seconds, total_frame_count)."""
#         import time
#         start_time = time.time()

#         cap = cv2.VideoCapture(self.source)
#         if not cap.isOpened():
#             self.logger.error(f"Cannot open source: {self.source!r}")
#             sys.exit(1)

#         _raw_fps = cap.get(cv2.CAP_PROP_FPS)
#         if not _raw_fps:
#             print("[WARNING] FPS not detected — defaulting to 25.0")
#         fps    = _raw_fps or 25.0
#         width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#         print(f"Video      : {width}x{height} @ {fps:.1f}fps  "
#               f"{total/fps:.1f}s  {total} frames")
#         print(f"Analysis ID: {self.analysis_id}")
#         print(f"Processing : every {RAW_FRAME_SKIP}rd raw frame  |  "
#               f"YOLO every {RAW_FRAME_SKIP * GADGET_EVERY} raw frames  |  "
#               f"Droop every {RAW_FRAME_SKIP * DROOP_EVERY} raw frames")

#         source_str  = str(self.source)
#         # Use real uploaded filename if provided; fall back to temp file basename
#         source_name = (
#             self.original_filename
#             if self.original_filename
#             else (os.path.basename(source_str) if isinstance(self.source, str) else "webcam")
#         )

#         # Seek to end for accurate duration (handles VFR / mismatched fps tags)
#         cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
#         duration_s = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 3)
#         cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # rewind
#         if duration_s <= 0:
#             duration_s = round(total / fps, 3) if total > 0 and fps > 0 else 0.0

#         h = int(duration_s) // 3600
#         m = (int(duration_s) % 3600) // 60
#         s = int(duration_s) % 60
#         size_mb = (
#             round(os.path.getsize(source_str) / 1_000_000, 2)
#             if isinstance(self.source, str) and os.path.isfile(source_str) else 0
#         )

#         video_info = {
#             "filename":          source_name,
#             "videoPath":         source_str,
#             "durationSeconds":   duration_s,
#             "durationFormatted": f"{h}:{m:02d}:{s:02d}",
#             "resolution":        f"{width}x{height}",
#             "fps":               round(fps, 3),
#             "totalFrames":       total,
#             "sizeMb":            size_mb,
#         }

#         if self.shared_vstore is not None:
#             # Batch mode: attach this video's metadata to the shared store
#             self.vstore = self.shared_vstore
#             self.vstore.add_video_info(video_info)
#         else:
#             self.vstore = ViolationStore(
#                 analysis_id     = self.analysis_id,
#                 train_detail_id = self.train_detail_id,
#                 video_info      = video_info,
#             )

#         self._print_banner(fps, width, height, total)

#         if self.save:
#             os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
#             self._writer = cv2.VideoWriter(
#                 OUTPUT_PATH,
#                 cv2.VideoWriter_fourcc(*"mp4v"),
#                 fps,
#                 (width, height),
#             )

#         raw_frame_no = 0
#         report_path  = ""

#         reader_thread = threading.Thread(
#             target=self._reader_loop, args=(cap,),
#             daemon=True, name="FrameReader",
#         )
#         writer_thread = threading.Thread(
#             target=self._writer_loop,
#             daemon=True, name="FrameWriter",
#         )
#         reader_thread.start()
#         writer_thread.start()

#         try:
#             while True:
#                 item = self._read_queue.get()
#                 if item is _STOP:
#                     break

#                 raw_frame, raw_frame_no, video_time = item

#                 if raw_frame_no % RAW_FRAME_SKIP != 0:
#                     self._write_queue.put(raw_frame)
#                     continue

#                 self._processed_frame_no += 1
#                 annotated = self._process_frame(
#                     raw_frame, video_time, raw_frame_no, self._processed_frame_no
#                 )
#                 self._write_queue.put(annotated)

#                 if self.display:
#                     show = annotated
#                     if DISPLAY_SCALE != 1.0:
#                         show = cv2.resize(
#                             annotated,
#                             (int(width * DISPLAY_SCALE), int(height * DISPLAY_SCALE)),
#                         )
#                     cv2.imshow(WINDOW_NAME, show)
#                     key = cv2.waitKey(1) & 0xFF
#                     if key in (ord("q"), 27):
#                         self.logger.info("Quit by user.")
#                         break

#         except KeyboardInterrupt:
#             self.logger.info("\nInterrupted by user.")
#         except Exception:
#             self.logger.error("Unexpected error:\n" + traceback.format_exc())
#         finally:
#             self._write_queue.put(_STOP)
#             writer_thread.join(timeout=30)
#             cap.release()
#             if self._writer:
#                 self._writer.release()
#             if self.display:
#                 cv2.destroyAllWindows()

#             processing_time = round(time.time() - start_time, 3)
#             self._print_summary(raw_frame_no, processing_time)
#             finalize_report()

#             # In batch mode the caller (api.py) finalizes after ALL videos are done
#             if self.shared_vstore is None:
#                 report_path = self.vstore.finalize(processing_time=processing_time)
#             else:
#                 report_path = ""

#         actual_fps = raw_frame_no / processing_time if processing_time > 0 else 0
#         print(f"\nTotal Time : {processing_time:.2f}s   FPS : {actual_fps:.2f}")
#         return report_path, duration_s, total


#     # ── READER THREAD ─────────────────────────────────────────────

#     def _reader_loop(self, cap: cv2.VideoCapture) -> None:
#         # frame_no counts frames within THIS video (1-based).
#         # Adding self.frame_offset gives every frame a globally unique
#         # index across the entire batch — prevents dedup collisions in
#         # ViolationStore._seen_frames when two videos share local frame numbers.
#         frame_no = 0
#         try:
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
#                 frame_no        += 1
#                 global_frame_no  = frame_no + self.frame_offset
#                 video_time       = (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0) + self.time_offset
#                 self._read_queue.put((frame, global_frame_no, video_time))
#         except Exception:
#             self.logger.error("Reader error:\n" + traceback.format_exc())
#         finally:
#             self._read_queue.put(_STOP)


#     # ── WRITER THREAD ─────────────────────────────────────────────

#     def _writer_loop(self) -> None:
#         try:
#             while True:
#                 item = self._write_queue.get()
#                 if item is _STOP:
#                     break
#                 if self._writer:
#                     self._writer.write(item)
#         except Exception:
#             self.logger.error("Writer error:\n" + traceback.format_exc())


#     # ── PER-FRAME PROCESSING ──────────────────────────────────────

#     def _process_frame(
#         self,
#         frame:              np.ndarray,
#         video_time:         float,
#         raw_frame_no:       int,
#         processed_frame_no: int,
#     ) -> np.ndarray:
#         annotated = frame

#         run_gadget  = (processed_frame_no % GADGET_EVERY  == 0)
#         run_absence = (processed_frame_no % ABSENCE_EVERY == 0)
#         run_droop   = (processed_frame_no % DROOP_EVERY   == 0)

#         prev_pilot_boxes     = self._prev_pilot_boxes
#         prev_frame_detection = self._prev_frame_detections

#         future_gadget = (
#             self.executor.submit(self.detector.process, frame, round(video_time, 0))
#             if run_gadget else None
#         )
#         future_absence = (
#             self.executor.submit(
#                 self.absence_detector.process,
#                 prev_pilot_boxes, video_time, frame.shape[1], frame.shape[0],
#             )
#             if run_absence else None
#         )
#         future_droop = (
#             self.executor.submit(
#                 self.droop_detector.process,
#                 frame, video_time, prev_frame_detection,
#             )
#             if run_droop else None
#         )

#         results,         log_events         = [], []
#         absence_results, absence_log_events = [], []
#         droop_results,   droop_log_events   = [], []

#         try:
#             if future_gadget is not None:
#                 results, log_events = future_gadget.result()
#         except Exception as exc:
#             self.logger.error(f"Gadget error frame {raw_frame_no}: {exc}", exc_info=True)

#         try:
#             if future_absence is not None:
#                 absence_results, absence_log_events = future_absence.result()
#         except Exception as exc:
#             self.logger.error(f"Absence error frame {raw_frame_no}: {exc}", exc_info=True)

#         try:
#             if future_droop is not None:
#                 droop_results, droop_log_events = future_droop.result()
#         except Exception as exc:
#             self.logger.error(f"Droop error frame {raw_frame_no}: {exc}", exc_info=True)

#         if run_gadget:
#             self._prev_pilot_boxes      = [(r.pilot_id, r.bbox) for r in results]
#             self._prev_frame_detections = self.detector.last_frame_detections

#         # ── Draw (skipped entirely when DRAW=False) ───────────────
#         if DRAW:
#             for g in self.detector.last_gadget_hits:
#                 draw_gadget_box(annotated, g.bbox, g.class_name, g.confidence)
#             for ar in absence_results:
#                 if ar.calibrated and ar.seat_zone is not None:
#                     draw_seat_zone(annotated, ar.seat_zone, ar.pilot_id)

#         any_gadget_distracted = False
#         last_gadget_pilot     = None
#         last_gadget_name      = ""

#         for r in results:
#             gadget_names = [g.class_name for g in r.gadgets]
#             if DRAW:
#                 draw_pilot_box(annotated, r.bbox, r.pilot_id, r.distracted, [])
#             if r.distracted:
#                 any_gadget_distracted = True
#                 last_gadget_pilot     = r.pilot_id
#                 last_gadget_name      = gadget_names[0] if gadget_names else "gadget"
#                 if DRAW:
#                     _draw_distraction_label(annotated, r.bbox, "Phone Usage",
#                                             r.timer_value, color=(0, 0, 220))

#         any_absence_distracted = False
#         last_absent_pilot      = None
#         last_absent_duration   = 0.0

#         for ar in absence_results:
#             current_bbox = next(
#                 (r.bbox for r in results if r.pilot_id == ar.pilot_id), None
#             )
#             if DRAW:
#                 draw_absence_overlay(
#                     frame=annotated, bbox=current_bbox, pilot_id=ar.pilot_id,
#                     absent=ar.absent, timer_val=ar.timer_value,
#                     calibrated=ar.calibrated,
#                 )
#             if ar.absent:
#                 any_absence_distracted = True
#                 last_absent_pilot      = ar.pilot_id
#                 last_absent_duration   = ar.timer_value
#                 if DRAW:
#                     _draw_distraction_label(annotated, current_bbox, "Away From Seat",
#                                             ar.timer_value, color=(0, 140, 255))

#         any_droop_distracted = False
#         last_droop_pilot     = None
#         last_droop_duration  = 0.0
#         last_droop_severity  = "DROWSINESS"
#         bbox_by_pid          = {}

#         if droop_results:
#             bbox_by_pid = {r.pilot_id: r.bbox for r in results}

#         for dr in droop_results:
#             current_bbox = bbox_by_pid.get(dr.pilot_id)
#             if not dr.is_seated:
#                 if DRAW:
#                     draw_standing_label(annotated, dr.pilot_id, current_bbox)
#                 continue
#             if hasattr(dr, "keypoints") and dr.keypoints:
#                 if DRAW:
#                     draw_droop_keypoints(
#                         frame=annotated, keypoints=dr.keypoints,
#                         pilot_id=dr.pilot_id, drooping=dr.drooping,
#                         angle=getattr(dr, "angle", 0.0),
#                     )
#             if DRAW:
#                 draw_droop_overlay(
#                     frame=annotated, pilot_id=dr.pilot_id,
#                     drooping=dr.drooping, timer_val=dr.timer_value,
#                     bbox=current_bbox,
#                     severity=getattr(dr, "severity", "DROWSINESS"),
#                 )
#             if dr.drooping:
#                 any_droop_distracted = True
#                 last_droop_pilot     = dr.pilot_id
#                 last_droop_duration  = dr.timer_value
#                 last_droop_severity  = getattr(dr, "severity", "DROWSINESS")
#                 display_secs         = dr.timer_value * (38 / 25.0)
#                 if DRAW:
#                     _draw_distraction_label(
#                         annotated, current_bbox, last_droop_severity, display_secs,
#                         color=(0, 200, 255) if last_droop_severity == "DROWSINESS"
#                         else (0, 80, 200),
#                     )

#         if DRAW:
#             if any_gadget_distracted and last_gadget_pilot is not None:
#                 draw_alert_banner(annotated, last_gadget_pilot, last_gadget_name)
#             if any_absence_distracted and last_absent_pilot is not None:
#                 draw_absence_banner(annotated, last_absent_pilot, last_absent_duration)
#             if any_droop_distracted and last_droop_pilot is not None:
#                 draw_droop_banner(annotated, last_droop_pilot, last_droop_duration,
#                                   severity=last_droop_severity)
#             for dr in droop_results:
#                 if not dr.drooping:
#                     continue
#                 if any(ar.absent and ar.pilot_id == dr.pilot_id
#                        for ar in absence_results):
#                     cb = bbox_by_pid.get(dr.pilot_id)
#                     _draw_distraction_label(annotated, cb, "SLEEPING / ABSENT",
#                                             dr.timer_value, color=(0, 50, 200))
#             draw_hud(annotated, video_time, raw_frame_no, len(results))

#         # ── CHANGED: compute local_video_time so original_video_timestamp ──
#         # in the JSON report shows the correct time WITHIN the source file,
#         # not the global cumulative timeline position.
#         # source_filename is passed so the field reads e.g.:
#         #   "ch06_...mp4 00:00:17"  instead of  " 00:00:00"
#         local_video_time = video_time - self.time_offset   # time within THIS video
#         src_filename     = self.original_filename or ""    # real filename, not tmp path

#         # ── Log + store violations ────────────────────────────────

#         if log_events:
#             r_ref = next((r for r in results if r.distracted), None)
#             conf  = r_ref.gadgets[0].confidence if (r_ref and r_ref.gadgets) else 0.9
#             dur   = r_ref.timer_value if r_ref else 0.0
#             event_time = max(self.time_offset, video_time - GADGET_ALLOWED_DURATION)
#             self.vstore.record_violation(
#                 annotated_frame  = annotated,
#                 original_frame   = frame,
#                 video_time       = event_time,
#                 frame_index      = raw_frame_no,
#                 event_type       = "phone_use",
#                 severity         = "CRITICAL",
#                 confidence       = conf,
#                 risk_score       = 80,
#                 risk_level       = "CRITICAL",
#                 factors          = ["phone_use"],
#                 duration         = dur,
#                 source_filename  = src_filename,       # ← CHANGED
#                 local_video_time = local_video_time,   # ← CHANGED
#             )
#             log_distraction(self.logger, event_time,
#                             event="One of the pilots is using a mobile phone",
#                             severity="CRITICAL", frame=annotated)

#         if absence_log_events:
#             ar_ref  = next((ar for ar in absence_results if ar.absent), None)
#             dur_abs = ar_ref.timer_value if ar_ref else 0.0
#             event_time = max(self.time_offset, video_time - ABSENCE_ALLOWED_DURATION)
#             self.vstore.record_violation(
#                 annotated_frame  = annotated,
#                 original_frame   = frame,
#                 video_time       = event_time,
#                 frame_index      = raw_frame_no,
#                 event_type       = "seat_absence",
#                 severity         = "CRITICAL",
#                 confidence       = 1.0,
#                 risk_score       = 70,
#                 risk_level       = "CRITICAL",
#                 factors          = ["seat_absence"],
#                 duration         = dur_abs,
#                 source_filename  = src_filename,       # ← CHANGED
#                 local_video_time = local_video_time,   # ← CHANGED
#             )
#             log_distraction(self.logger, event_time,
#                             event="One of the pilots is away from the seat",
#                             severity="CRITICAL", frame=annotated)

#         if droop_log_events:
#             severities  = [e[1] for e in droop_log_events]
#             is_sleeping = any("SLEEPING" in s for s in severities)
#             droop_pids  = {e[0] for e in droop_log_events}
#             absent_pids = {ar.pilot_id for ar in absence_results if ar.absent}
#             also_absent = bool(droop_pids & absent_pids)

#             if also_absent:
#                 event_msg = "One of the pilots is sleeping / slumped in seat"
#                 etype     = "sleeping_absent"
#             elif is_sleeping:
#                 event_msg = "One of the pilots is sleeping"
#                 etype     = "sleeping"
#             else:
#                 event_msg = "One of the pilots is drowsy"
#                 etype     = "drowsy"

#             dr_ref  = next((dr for dr in droop_results if dr.drooping), None)
#             dur_drp = dr_ref.timer_value if dr_ref else 0.0
#             event_time = max(self.time_offset, video_time - HEAD_DROP_DURATION)
#             self.vstore.record_violation(
#                 annotated_frame  = annotated,
#                 original_frame   = frame,
#                 video_time       = event_time,
#                 frame_index      = raw_frame_no,
#                 event_type       = etype,
#                 severity         = "CRITICAL",
#                 confidence       = 0.9,
#                 risk_score       = 75,
#                 risk_level       = "HIGH",
#                 factors          = ["drowsy", "head_droop"],
#                 duration         = dur_drp,
#                 source_filename  = src_filename,       # ← CHANGED
#                 local_video_time = local_video_time,   # ← CHANGED
#             )
#             log_distraction(self.logger, event_time, event=event_msg,
#                             severity="CRITICAL", frame=annotated)

#         return annotated


#     # ── HELPERS ───────────────────────────────────────────────────

#     def _print_banner(self, fps: float, w: int, h: int, total: int) -> None:
#         self.logger.info(
#             f"\n{'='*60}\n"
#             f"  LOCO PILOT DISTRACTION DETECTION\n"
#             f"  Analysis ID : {self.analysis_id}\n"
#             f"  Source      : {self.source}\n"
#             f"  Video       : {w}x{h} @ {fps:.1f} fps ({total} frames)\n"
#             f"  Output      : {OUTPUT_PATH if self.save else 'disabled'}\n"
#             f"{'='*60}\n"
#         )

#     def _print_summary(self, frame_no: int, processing_time: float) -> None:
#         self.logger.info(
#             f"\n{'='*60}\n"
#             f"  Processing complete\n"
#             f"  Raw frames  : {frame_no}\n"
#             f"  Processed   : {self._processed_frame_no} "
#             f"(1 in every {RAW_FRAME_SKIP})\n"
#             f"  Time        : {processing_time:.2f}s\n"
#             f"  Report : outputs/{self.analysis_id}/analysis_report.json\n"
#             f"  Frames : outputs/{self.analysis_id}/frames/\n"
#             f"{'='*60}\n"
#         )


# # ── CLI ───────────────────────────────────────────────────────────

# def parse_args() -> argparse.Namespace:
#     p = argparse.ArgumentParser(description="Loco Pilot Distraction Detection")
#     p.add_argument("--source",          default=0,
#                    help="Video file path or camera index (default: 0 = webcam)")
#     p.add_argument("--analysis-id",     default=None)
#     p.add_argument("--train-detail-id", default=0, type=int)
#     p.add_argument("--no-display",      action="store_true")
#     p.add_argument("--no-save",         action="store_true")
#     return p.parse_args()


# if __name__ == "__main__":
#     args   = parse_args()
#     source = args.source
#     if isinstance(source, str) and source.isdigit():
#         source = int(source)

#     GadgetDetectionPipeline(
#         source          = source,
#         analysis_id     = args.analysis_id,
#         train_detail_id = args.train_detail_id,
#         save            = not args.no_save,
#         display         = False,
#     ).run()

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

print("[main] ✅ NEW main.py loaded — v5 (mediapipe ear-check, mouth-exclusion, wrist-confirm)")

DRAW           = False
RAW_FRAME_SKIP = 3
GADGET_EVERY   = 6
ABSENCE_EVERY  = 4
DROOP_EVERY    = 15

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
    ) -> None:
        self.source          = source
        self.train_detail_id = train_detail_id
        self.save            = save
        self.display         = display
        self.shared_vstore   = shared_vstore
        self.time_offset     = time_offset
        self.frame_offset    = frame_offset
        self.source_filename = source_filename

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

    def run(self) -> str:
        import time
        start_time = time.time()

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.logger.error(f"Cannot open source: {self.source!r}")
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

        prev_frame_detection = self._prev_frame_detections

        # ── Build per-pilot MediaPipe landmarks for gadget detector ───────────
        pose_landmarks_by_pilot = self._get_pose_landmarks(frame) if run_gadget else None

        future_gadget  = self.executor.submit(self.detector.process, frame, round(global_time, 3), pose_landmarks_by_pilot) if run_gadget  else None
        future_absence = self.executor.submit(self.absence_detector.process, self._absence_pilot_boxes, global_time, frame.shape[1], frame.shape[0]) if run_absence else None
        future_droop   = self.executor.submit(self.droop_detector.process, frame, global_time, prev_frame_detection) if run_droop else None

        results,         log_events         = [], []
        absence_results, absence_log_events = [], []
        droop_results,   droop_log_events   = [], []

        try:
            if future_gadget  is not None: results,         log_events         = future_gadget.result()
        except Exception as exc:
            self.logger.error(f"Gadget error frame {global_frame}: {exc}", exc_info=True)
        try:
            if future_absence is not None: absence_results, absence_log_events = future_absence.result()
        except Exception as exc:
            self.logger.error(f"Absence error frame {global_frame}: {exc}", exc_info=True)
        try:
            if future_droop   is not None: droop_results,   droop_log_events   = future_droop.result()
        except Exception as exc:
            self.logger.error(f"Droop error frame {global_frame}: {exc}", exc_info=True)

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

        # ── Draw (only when DRAW=True) ────────────────────────────────────────
        if DRAW:
            for g in self.detector.last_gadget_hits:
                draw_gadget_box(annotated, g.bbox, g.class_name, g.confidence)
            for ar in absence_results:
                if ar.calibrated and ar.seat_zone is not None:
                    draw_seat_zone(annotated, ar.seat_zone, ar.pilot_id)

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
            self.vstore.record_violation(
                annotated_frame=annotated, original_frame=frame,
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
            self.vstore.record_violation(
                annotated_frame=annotated, original_frame=frame,
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
            self.vstore.record_violation(
                annotated_frame=annotated, original_frame=frame,
                video_time=event_global, frame_index=global_frame,
                event_type=etype, severity="CRITICAL",
                confidence=0.9, risk_score=75, risk_level="HIGH",
                factors=["drowsy", "head_droop"], duration=dur_drp,
                source_filename=self.source_filename, local_video_time=event_local,
            )
            log_distraction(self.logger, event_global, event=event_msg, severity="CRITICAL", frame=annotated)

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