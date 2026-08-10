# # # # # from __future__ import annotations

# # # # # import json
# # # # # import os
# # # # # from dataclasses import dataclass
# # # # # from typing import Any, Dict, List, Optional

# # # # # import cv2
# # # # # import numpy as np

# # # # # OUTPUTS_ROOT = "outputs"
# # # # # MERGE_WINDOW = 2.0


# # # # # @dataclass
# # # # # class _Violation:
# # # # #     timestamp:       float
# # # # #     time_str:        str
# # # # #     frame_index:     int
# # # # #     type:            str
# # # # #     events:          List[str]
# # # # #     severity:        str
# # # # #     duration:        float
# # # # #     risk_score:      int
# # # # #     risk_level:      str
# # # # #     confidence:      float
# # # # #     factors:         List[str]
# # # # #     source_filename: str                   = ""    # e.g. "ax.mp4"
# # # # #     local_time_str:  str                   = ""    # local time within that file e.g. "00:00:18"
# # # # #     frame_path:      Optional[str]         = None
# # # # #     annotated_frame: Optional[np.ndarray]  = None


# # # # # class ViolationStore:

# # # # #     def __init__(self, analysis_id: str, train_detail_id: int, video_info: Optional[Dict[str, Any]] = None):
# # # # #         self.analysis_id     = analysis_id
# # # # #         self.train_detail_id = train_detail_id
# # # # #         # video_infos is always a list — 1 entry for single-video, N entries for batch
# # # # #         self.video_infos: List[Dict[str, Any]] = [video_info] if video_info is not None else []

# # # # #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# # # # #         self.frames_dir = os.path.join(self.output_dir, "frames")
# # # # #         os.makedirs(self.frames_dir, exist_ok=True)

# # # # #         self._violations: List[_Violation] = []
# # # # #         self._seen_frames: set             = set()
# # # # #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# # # # #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# # # # #         """
# # # # #         Register one video's metadata into the shared store.
# # # # #         Called once per video in batch mode (api.py passes shared_vstore).
# # # # #         """
# # # # #         self.video_infos.append(video_info)

# # # # #     def record_violation(
# # # # #         self,
# # # # #         annotated_frame: np.ndarray,
# # # # #         video_time:      float,
# # # # #         frame_index:     int,
# # # # #         event_type:      str,
# # # # #         original_frame:  Optional[np.ndarray] = None,
# # # # #         severity:        str   = "CRITICAL",
# # # # #         confidence:      float = 0.9,
# # # # #         risk_score:      int   = 80,
# # # # #         risk_level:      str   = "CRITICAL",
# # # # #         factors:         Optional[List[str]] = None,
# # # # #         duration:        float = 0.0,
# # # # #         source_filename: str   = "",   # original upload filename e.g. "ax.mp4"
# # # # #         local_video_time: float = -1.0, # raw video_time before offset; -1 = same as video_time
# # # # #     ):
# # # # #         # Deduplicate on (frame_index, event_type) so that:
# # # # #         #  • the same violation type on the same frame is recorded only once
# # # # #         #  • different violation types on the same frame are each recorded
# # # # #         #  • frame numbers from different videos never collide (frame_offset
# # # # #         #    in main.py makes every global frame_index unique across the batch)
# # # # #         dedup_key = (frame_index, event_type)
# # # # #         if dedup_key in self._seen_frames:
# # # # #             return
# # # # #         self._seen_frames.add(dedup_key)
# # # # #         factors   = factors or []
# # # # #         t         = int(round(video_time))
# # # # #         time_str  = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"
# # # # #         # Build the per-file local timestamp (time within the source video)
# # # # #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# # # # #         local_str = f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
# # # # #         self._violations.append(_Violation(
# # # # #             timestamp        = video_time,
# # # # #             time_str         = time_str,
# # # # #             frame_index      = frame_index,
# # # # #             type             = event_type,
# # # # #             events           = [event_type],
# # # # #             severity         = severity,
# # # # #             duration         = round(duration, 2),
# # # # #             risk_score       = risk_score,
# # # # #             risk_level       = risk_level,
# # # # #             confidence       = round(confidence, 3),
# # # # #             factors          = list(factors),
# # # # #             source_filename  = source_filename,
# # # # #             local_time_str   = local_str,
# # # # #             annotated_frame  = annotated_frame.copy() if annotated_frame is not None else None,
# # # # #         ))

# # # # #     def _deduplicate_by_frame(self):
# # # # #         unique: Dict[int, _Violation] = {}
# # # # #         for v in self._violations:
# # # # #             if v.frame_index not in unique:
# # # # #                 unique[v.frame_index] = v
# # # # #             else:
# # # # #                 ex = unique[v.frame_index]
# # # # #                 ex.events  = list(set(ex.events  + v.events))
# # # # #                 ex.factors = list(set(ex.factors + v.factors))
# # # # #                 if v.risk_score > ex.risk_score:
# # # # #                     ex.risk_score = v.risk_score
# # # # #                     ex.risk_level = v.risk_level
# # # # #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# # # # #                     ex.annotated_frame = v.annotated_frame
# # # # #         self._violations = list(unique.values())

# # # # #     def _merge_by_time_window(self):
# # # # #         if not self._violations:
# # # # #             return
# # # # #         self._violations.sort(key=lambda x: x.timestamp)
# # # # #         merged = []
# # # # #         group  = [self._violations[0]]
# # # # #         for v in self._violations[1:]:
# # # # #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# # # # #                 group.append(v)
# # # # #             else:
# # # # #                 merged.append(self._merge_group(group))
# # # # #                 group = [v]
# # # # #         merged.append(self._merge_group(group))
# # # # #         self._violations = merged

# # # # #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# # # # #         base            = group[0]
# # # # #         events, factors = [], []
# # # # #         max_risk        = base.risk_score
# # # # #         risk_level      = base.risk_level
# # # # #         best_frame      = base.annotated_frame
# # # # #         for v in group:
# # # # #             events.extend(v.events)
# # # # #             factors.extend(v.factors)
# # # # #             if v.risk_score > max_risk:
# # # # #                 max_risk, risk_level = v.risk_score, v.risk_level
# # # # #             if best_frame is None and v.annotated_frame is not None:
# # # # #                 best_frame = v.annotated_frame
# # # # #         return _Violation(
# # # # #             timestamp        = base.timestamp,
# # # # #             time_str         = base.time_str,
# # # # #             frame_index      = base.frame_index,
# # # # #             type             = base.type,
# # # # #             events           = list(set(events)),
# # # # #             severity         = base.severity,
# # # # #             duration         = base.duration,
# # # # #             risk_score       = max_risk,
# # # # #             risk_level       = risk_level,
# # # # #             confidence       = base.confidence,
# # # # #             factors          = list(set(factors)),
# # # # #             source_filename  = base.source_filename,
# # # # #             local_time_str   = base.local_time_str,
# # # # #             annotated_frame  = best_frame,
# # # # #         )

# # # # #     def extract_violation_frames(self, video_path: str):
# # # # #         print("[ViolationStore] Saving frames...")
# # # # #         need_video = [v for v in self._violations if v.annotated_frame is None]
# # # # #         saved = 0
# # # # #         for v in self._violations:
# # # # #             if v.annotated_frame is not None:
# # # # #                 v.frame_path      = self._save_frame(v.annotated_frame, v.events, v.time_str)
# # # # #                 v.annotated_frame = None
# # # # #                 saved += 1
# # # # #         if need_video:
# # # # #             cap = cv2.VideoCapture(video_path)
# # # # #             if not cap.isOpened():
# # # # #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# # # # #             else:
# # # # #                 seen: set = set()
# # # # #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# # # # #                     if v.frame_index in seen:
# # # # #                         continue
# # # # #                     seen.add(v.frame_index)
# # # # #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# # # # #                     ret, frame = cap.read()
# # # # #                     if not ret:
# # # # #                         continue
# # # # #                     v.frame_path = self._save_frame(frame, v.events, v.time_str)
# # # # #                     saved += 1
# # # # #                 cap.release()
# # # # #         print(f"[ViolationStore] {saved} frames saved")

# # # # #     def _save_frame(self, frame: np.ndarray, events: List[str], time_str: str) -> str:
# # # # #         distraction   = "_".join(events)
# # # # #         filename_time = time_str.replace(":", "-")
# # # # #         filename      = f"{distraction}_{filename_time}.jpg"
# # # # #         path          = os.path.join(self.frames_dir, filename)
# # # # #         ok = cv2.imwrite(path, cv2.resize(frame, (640, 360)),
# # # # #                          [cv2.IMWRITE_JPEG_QUALITY, 85])
# # # # #         if not ok:
# # # # #             print(f"[ViolationStore] imwrite failed: {path}")
# # # # #         return os.path.join(self.analysis_id, "frames", filename)

# # # # #     def _build_report(self, processing_time: float = 0.0) -> dict:
# # # # #         return {
# # # # #             "analysis_id":     self.analysis_id,
# # # # #             "train_detail_id": self.train_detail_id,
# # # # #             "processing_time": round(processing_time, 3),
# # # # #             # Single video → keep as dict for backwards compat; batch → list
# # # # #             "video_info": self.video_infos[0] if len(self.video_infos) == 1
# # # # #                           else self.video_infos,
# # # # #             "violations": [
# # # # #                 {
# # # # #                     "timestamp":   v.time_str,
# # # # #                     "frame_index": v.frame_index,
# # # # #                     "events":      v.events,
# # # # #                     "severity":    v.severity,
# # # # #                     "duration":    v.duration,
# # # # #                     "risk_score":  v.risk_score,
# # # # #                     "risk_level":  v.risk_level,
# # # # #                     "confidence":  v.confidence,
# # # # #                     "factors":     v.factors,
# # # # #                     "original_video_timestamp": f"{v.source_filename} {v.local_time_str}",
# # # # #                     "frame_path":  v.frame_path,
# # # # #                 }
# # # # #                 for v in self._violations
# # # # #             ],
# # # # #         }

# # # # #     def finalize(self, processing_time: float = 0.0) -> str:
# # # # #         self._deduplicate_by_frame()
# # # # #         self._merge_by_time_window()
# # # # #         # Extract frames from every video in the batch (or the single video)
# # # # #         for vi in self.video_infos:
# # # # #             if vi and vi.get("videoPath"):
# # # # #                 self.extract_violation_frames(vi["videoPath"])
# # # # #         report   = self._build_report(processing_time=processing_time)
# # # # #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# # # # #         with open(out_path, "w", encoding="utf-8") as f:
# # # # #             json.dump(report, f, indent=2)
# # # # #         print(f"[ViolationStore] JSON report     : {out_path}")
# # # # #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# # # # #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# # # # #         try:
# # # # #             from utils.db_s3_uploader import finalize_and_upload
# # # # #             finalize_and_upload(
# # # # #                 report_path     = out_path,
# # # # #                 analysis_id     = self.analysis_id,
# # # # #                 train_detail_id = self.train_detail_id,
# # # # #             )
# # # # #         except Exception as exc:
# # # # #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# # # # #         return out_path



# # # # # utils/violation_store.py
# # # # # ─────────────────────────────────────────────────────────────────
# # # # # Change from original:
# # # # #   _save_frame() now appends _f{frame_index} to the filename so
# # # # #   two violations of the same type at the same timestamp never
# # # # #   silently overwrite each other.
# # # # #   e.g.  phone_use_00-13-16_f6762.jpg
# # # # #         seat_absence_00-00-17_f516.jpg
# # # # # ─────────────────────────────────────────────────────────────────

# # # # from __future__ import annotations

# # # # import json
# # # # import os
# # # # from dataclasses import dataclass
# # # # from typing import Any, Dict, List, Optional

# # # # import cv2
# # # # import numpy as np

# # # # OUTPUTS_ROOT = "outputs"
# # # # MERGE_WINDOW = 2.0


# # # # @dataclass
# # # # class _Violation:
# # # #     timestamp:       float
# # # #     time_str:        str
# # # #     frame_index:     int
# # # #     type:            str
# # # #     events:          List[str]
# # # #     severity:        str
# # # #     duration:        float
# # # #     risk_score:      int
# # # #     risk_level:      str
# # # #     confidence:      float
# # # #     factors:         List[str]
# # # #     source_filename: str                  = ""
# # # #     local_time_str:  str                  = ""
# # # #     frame_path:      Optional[str]        = None
# # # #     annotated_frame: Optional[np.ndarray] = None


# # # # class ViolationStore:

# # # #     def __init__(
# # # #         self,
# # # #         analysis_id:     str,
# # # #         train_detail_id: int,
# # # #         video_info:      Optional[Dict[str, Any]] = None,
# # # #     ):
# # # #         self.analysis_id     = analysis_id
# # # #         self.train_detail_id = train_detail_id
# # # #         self.video_infos: List[Dict[str, Any]] = (
# # # #             [video_info] if video_info is not None else []
# # # #         )

# # # #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# # # #         self.frames_dir = os.path.join(self.output_dir, "frames")
# # # #         os.makedirs(self.frames_dir, exist_ok=True)

# # # #         self._violations: List[_Violation] = []
# # # #         self._seen_frames: set             = set()
# # # #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# # # #     # ── called once per video in batch mode ──────────────────────
# # # #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# # # #         self.video_infos.append(video_info)

# # # #     # ── called from main.py for every detected violation ─────────
# # # #     def record_violation(
# # # #         self,
# # # #         annotated_frame:  np.ndarray,
# # # #         video_time:       float,
# # # #         frame_index:      int,
# # # #         event_type:       str,
# # # #         original_frame:   Optional[np.ndarray] = None,
# # # #         severity:         str   = "CRITICAL",
# # # #         confidence:       float = 0.9,
# # # #         risk_score:       int   = 80,
# # # #         risk_level:       str   = "CRITICAL",
# # # #         factors:          Optional[List[str]] = None,
# # # #         duration:         float = 0.0,
# # # #         source_filename:  str   = "",
# # # #         local_video_time: float = -1.0,
# # # #     ):
# # # #         # Deduplicate: same frame + same event type recorded only once.
# # # #         # Different event types on the same frame are each recorded.
# # # #         # frame_index is globally unique across batch (frame_offset in main.py).
# # # #         dedup_key = (frame_index, event_type)
# # # #         if dedup_key in self._seen_frames:
# # # #             return
# # # #         self._seen_frames.add(dedup_key)

# # # #         factors  = factors or []
# # # #         t        = int(round(video_time))
# # # #         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

# # # #         # Local timestamp = time within the source video file (before cumulative offset)
# # # #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# # # #         local_str = f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"

# # # #         self._violations.append(_Violation(
# # # #             timestamp        = video_time,
# # # #             time_str         = time_str,
# # # #             frame_index      = frame_index,
# # # #             type             = event_type,
# # # #             events           = [event_type],
# # # #             severity         = severity,
# # # #             duration         = round(duration, 2),
# # # #             risk_score       = risk_score,
# # # #             risk_level       = risk_level,
# # # #             confidence       = round(confidence, 3),
# # # #             factors          = list(factors),
# # # #             source_filename  = source_filename,
# # # #             local_time_str   = local_str,
# # # #             annotated_frame  = annotated_frame.copy() if annotated_frame is not None else None,
# # # #         ))

# # # #     # ─────────────────────────────────────────────────────────────

# # # #     def _deduplicate_by_frame(self):
# # # #         unique: Dict[int, _Violation] = {}
# # # #         for v in self._violations:
# # # #             if v.frame_index not in unique:
# # # #                 unique[v.frame_index] = v
# # # #             else:
# # # #                 ex = unique[v.frame_index]
# # # #                 ex.events  = list(set(ex.events  + v.events))
# # # #                 ex.factors = list(set(ex.factors + v.factors))
# # # #                 if v.risk_score > ex.risk_score:
# # # #                     ex.risk_score = v.risk_score
# # # #                     ex.risk_level = v.risk_level
# # # #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# # # #                     ex.annotated_frame = v.annotated_frame
# # # #         self._violations = list(unique.values())

# # # #     def _merge_by_time_window(self):
# # # #         if not self._violations:
# # # #             return
# # # #         self._violations.sort(key=lambda x: x.timestamp)
# # # #         merged = []
# # # #         group  = [self._violations[0]]
# # # #         for v in self._violations[1:]:
# # # #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# # # #                 group.append(v)
# # # #             else:
# # # #                 merged.append(self._merge_group(group))
# # # #                 group = [v]
# # # #         merged.append(self._merge_group(group))
# # # #         self._violations = merged

# # # #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# # # #         base            = group[0]
# # # #         events, factors = [], []
# # # #         max_risk        = base.risk_score
# # # #         risk_level      = base.risk_level
# # # #         best_frame      = base.annotated_frame
# # # #         for v in group:
# # # #             events.extend(v.events)
# # # #             factors.extend(v.factors)
# # # #             if v.risk_score > max_risk:
# # # #                 max_risk, risk_level = v.risk_score, v.risk_level
# # # #             if best_frame is None and v.annotated_frame is not None:
# # # #                 best_frame = v.annotated_frame
# # # #         return _Violation(
# # # #             timestamp        = base.timestamp,
# # # #             time_str         = base.time_str,
# # # #             frame_index      = base.frame_index,
# # # #             type             = base.type,
# # # #             events           = list(set(events)),
# # # #             severity         = base.severity,
# # # #             duration         = base.duration,
# # # #             risk_score       = max_risk,
# # # #             risk_level       = risk_level,
# # # #             confidence       = base.confidence,
# # # #             factors          = list(set(factors)),
# # # #             source_filename  = base.source_filename,
# # # #             local_time_str   = base.local_time_str,
# # # #             annotated_frame  = best_frame,
# # # #         )

# # # #     # ─────────────────────────────────────────────────────────────

# # # #     def extract_violation_frames(self, video_path: str):
# # # #         print("[ViolationStore] Saving frames...")
# # # #         need_video = [v for v in self._violations if v.annotated_frame is None]
# # # #         saved = 0

# # # #         for v in self._violations:
# # # #             if v.annotated_frame is not None:
# # # #                 v.frame_path      = self._save_frame(
# # # #                     v.annotated_frame, v.events, v.time_str, v.frame_index
# # # #                 )
# # # #                 v.annotated_frame = None
# # # #                 saved += 1

# # # #         if need_video:
# # # #             cap = cv2.VideoCapture(video_path)
# # # #             if not cap.isOpened():
# # # #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# # # #             else:
# # # #                 seen: set = set()
# # # #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# # # #                     if v.frame_index in seen:
# # # #                         continue
# # # #                     seen.add(v.frame_index)
# # # #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# # # #                     ret, frame = cap.read()
# # # #                     if not ret:
# # # #                         continue
# # # #                     v.frame_path = self._save_frame(
# # # #                         frame, v.events, v.time_str, v.frame_index
# # # #                     )
# # # #                     saved += 1
# # # #                 cap.release()

# # # #         print(f"[ViolationStore] {saved} frames saved")

# # # #     def _save_frame(
# # # #         self,
# # # #         frame:       np.ndarray,
# # # #         events:      List[str],
# # # #         time_str:    str,
# # # #         frame_index: int,            # ← ADDED: makes filename globally unique
# # # #     ) -> str:
# # # #         """
# # # #         Save one violation frame as JPEG.

# # # #         Filename: {event_types}_{hh-mm-ss}_f{frame_index}.jpg
# # # #         Example:  phone_use_00-13-16_f6762.jpg
# # # #                   seat_absence_00-00-17_f516.jpg

# # # #         frame_index prevents two violations of the same type at the
# # # #         same timestamp from silently overwriting each other.
# # # #         """
# # # #         distraction   = "_".join(events)
# # # #         filename_time = time_str.replace(":", "-")
# # # #         filename      = f"{distraction}_{filename_time}_f{frame_index}.jpg"  # ← CHANGED
# # # #         path          = os.path.join(self.frames_dir, filename)
# # # #         ok = cv2.imwrite(
# # # #             path,
# # # #             cv2.resize(frame, (640, 360)),
# # # #             [cv2.IMWRITE_JPEG_QUALITY, 85],
# # # #         )
# # # #         if not ok:
# # # #             print(f"[ViolationStore] imwrite failed: {path}")
# # # #         return os.path.join(self.analysis_id, "frames", filename)

# # # #     # ─────────────────────────────────────────────────────────────

# # # #     def _build_report(self, processing_time: float = 0.0) -> dict:
# # # #         return {
# # # #             "analysis_id":     self.analysis_id,
# # # #             "train_detail_id": self.train_detail_id,
# # # #             "processing_time": round(processing_time, 3),
# # # #             "video_info": (
# # # #                 self.video_infos[0] if len(self.video_infos) == 1
# # # #                 else self.video_infos
# # # #             ),
# # # #             "violations": [
# # # #                 {
# # # #                     "timestamp":   v.time_str,
# # # #                     "frame_index": v.frame_index,
# # # #                     "events":      v.events,
# # # #                     "severity":    v.severity,
# # # #                     "duration":    v.duration,
# # # #                     "risk_score":  v.risk_score,
# # # #                     "risk_level":  v.risk_level,
# # # #                     "confidence":  v.confidence,
# # # #                     "factors":     v.factors,
# # # #                     "original_video_timestamp": f"{v.source_filename} {v.local_time_str}",
# # # #                     "frame_path":  v.frame_path,
# # # #                 }
# # # #                 for v in self._violations
# # # #             ],
# # # #         }

# # # #     def finalize(self, processing_time: float = 0.0) -> str:
# # # #         self._deduplicate_by_frame()
# # # #         self._merge_by_time_window()

# # # #         # Save frames — each video's temp path is in video_infos
# # # #         for vi in self.video_infos:
# # # #             if vi and vi.get("videoPath"):
# # # #                 self.extract_violation_frames(vi["videoPath"])

# # # #         report   = self._build_report(processing_time=processing_time)
# # # #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# # # #         with open(out_path, "w", encoding="utf-8") as f:
# # # #             json.dump(report, f, indent=2)

# # # #         print(f"[ViolationStore] JSON report     : {out_path}")
# # # #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# # # #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# # # #         # Upload results to S3 and update DB result_s3_path
# # # #         try:
# # # #             from utils.db_s3_uploader import finalize_and_upload
# # # #             finalize_and_upload(
# # # #                 report_path     = out_path,
# # # #                 analysis_id     = self.analysis_id,
# # # #                 train_detail_id = self.train_detail_id,
# # # #             )
# # # #         except Exception as exc:
# # # #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# # # #         return out_path

# # # # # # from __future__ import annotations

# # # # # # import json
# # # # # # import os
# # # # # # from dataclasses import dataclass
# # # # # # from typing import Any, Dict, List, Optional

# # # # # # import cv2
# # # # # # import numpy as np

# # # # # # OUTPUTS_ROOT = "outputs"
# # # # # # MERGE_WINDOW = 2.0


# # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # INTERNAL DATA CLASS
# # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # @dataclass
# # # # # # class _Violation:
# # # # # #     timestamp:       float
# # # # # #     time_str:        str
# # # # # #     frame_index:     int
# # # # # #     type:            str
# # # # # #     events:          List[str]
# # # # # #     severity:        str
# # # # # #     duration:        float
# # # # # #     risk_score:      int
# # # # # #     risk_level:      str
# # # # # #     confidence:      float
# # # # # #     factors:         List[str]
# # # # # #     source_filename: str                  = ""    # e.g. "ch01.mp4"
# # # # # #     local_time_str:  str                  = ""    # local time within that file e.g. "00:00:18"
# # # # # #     frame_path:      Optional[str]        = None
# # # # # #     annotated_frame: Optional[np.ndarray] = None


# # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # VIOLATION STORE
# # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # class ViolationStore:
# # # # # #     """
# # # # # #     Accumulates all violations found across one analysis run (single video
# # # # # #     or a multi-video batch that shares the same analysis_id / folder_name).

# # # # # #     Usage
# # # # # #     ─────
# # # # # #     1. Construct once per analysis run.
# # # # # #     2. Call record_violation() from the pipeline for every distraction event.
# # # # # #     3. Call finalize() at the very end — it deduplicates, merges, saves frames,
# # # # # #        writes the JSON report, then calls db_s3_uploader.finalize_and_upload()
# # # # # #        to push everything to S3 and record results in the DB.
# # # # # #     """

# # # # # #     def __init__(
# # # # # #         self,
# # # # # #         analysis_id:     str,
# # # # # #         train_detail_id: int,
# # # # # #         video_info:      Optional[Dict[str, Any]] = None,
# # # # # #     ):
# # # # # #         self.analysis_id     = analysis_id
# # # # # #         self.train_detail_id = train_detail_id
# # # # # #         # video_infos is always a list:
# # # # # #         #   • 1 entry for single-video runs
# # # # # #         #   • N entries for batch runs (add_video_info called per video)
# # # # # #         self.video_infos: List[Dict[str, Any]] = (
# # # # # #             [video_info] if video_info is not None else []
# # # # # #         )

# # # # # #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# # # # # #         self.frames_dir = os.path.join(self.output_dir, "frames")
# # # # # #         os.makedirs(self.frames_dir, exist_ok=True)

# # # # # #         self._violations:  List[_Violation] = []
# # # # # #         self._seen_frames: set              = set()
# # # # # #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# # # # # #     # ── Public helpers ────────────────────────────────────────────────────────

# # # # # #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# # # # # #         """
# # # # # #         Register one video's metadata into the shared store.
# # # # # #         Called once per video in batch mode.
# # # # # #         """
# # # # # #         self.video_infos.append(video_info)

# # # # # #     def record_violation(
# # # # # #         self,
# # # # # #         annotated_frame:  np.ndarray,
# # # # # #         video_time:       float,
# # # # # #         frame_index:      int,
# # # # # #         event_type:       str,
# # # # # #         original_frame:   Optional[np.ndarray] = None,
# # # # # #         severity:         str   = "CRITICAL",
# # # # # #         confidence:       float = 0.9,
# # # # # #         risk_score:       int   = 80,
# # # # # #         risk_level:       str   = "CRITICAL",
# # # # # #         factors:          Optional[List[str]] = None,
# # # # # #         duration:         float = 0.0,
# # # # # #         source_filename:  str   = "",    # original filename e.g. "ch01.mp4"
# # # # # #         local_video_time: float = -1.0,  # video_time before global offset; -1 = same as video_time
# # # # # #     ) -> None:
# # # # # #         """
# # # # # #         Record one distraction event.

# # # # # #         Deduplication key is (frame_index, event_type) so that:
# # # # # #           • the same violation type on the same frame is recorded only once
# # # # # #           • different violation types on the same frame are each recorded
# # # # # #           • frame numbers from different videos never collide because
# # # # # #             main.py applies a frame_offset to make every global frame_index
# # # # # #             unique across the batch
# # # # # #         """
# # # # # #         dedup_key = (frame_index, event_type)
# # # # # #         if dedup_key in self._seen_frames:
# # # # # #             return
# # # # # #         self._seen_frames.add(dedup_key)

# # # # # #         factors  = factors or []
# # # # # #         t        = int(round(video_time))
# # # # # #         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

# # # # # #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# # # # # #         local_str = (
# # # # # #             f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
# # # # # #         )

# # # # # #         self._violations.append(
# # # # # #             _Violation(
# # # # # #                 timestamp        = video_time,
# # # # # #                 time_str         = time_str,
# # # # # #                 frame_index      = frame_index,
# # # # # #                 type             = event_type,
# # # # # #                 events           = [event_type],
# # # # # #                 severity         = severity,
# # # # # #                 duration         = round(duration, 2),
# # # # # #                 risk_score       = risk_score,
# # # # # #                 risk_level       = risk_level,
# # # # # #                 confidence       = round(confidence, 3),
# # # # # #                 factors          = list(factors),
# # # # # #                 source_filename  = source_filename,
# # # # # #                 local_time_str   = local_str,
# # # # # #                 annotated_frame  = (
# # # # # #                     annotated_frame.copy() if annotated_frame is not None else None
# # # # # #                 ),
# # # # # #             )
# # # # # #         )

# # # # # #     # ── Finalize ──────────────────────────────────────────────────────────────

# # # # # #     def finalize(self, processing_time: float = 0.0) -> str:
# # # # # #         """
# # # # # #         1. Deduplicate violations that share the same frame.
# # # # # #         2. Merge violations that are within MERGE_WINDOW seconds of each other.
# # # # # #         3. Save annotated frame images to disk (falls back to re-reading from
# # # # # #            the source video when no annotated frame was captured).
# # # # # #         4. Write analysis_report.json.
# # # # # #         5. Upload frames + report to S3 and record in DB via db_s3_uploader.
# # # # # #         Returns the local path to analysis_report.json.
# # # # # #         """
# # # # # #         self._deduplicate_by_frame()
# # # # # #         self._merge_by_time_window()

# # # # # #         # Extract frames from every video in the batch (or the single video)
# # # # # #         for vi in self.video_infos:
# # # # # #             if vi and vi.get("videoPath"):
# # # # # #                 self.extract_violation_frames(vi["videoPath"])

# # # # # #         report   = self._build_report(processing_time=processing_time)
# # # # # #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# # # # # #         with open(out_path, "w", encoding="utf-8") as f:
# # # # # #             json.dump(report, f, indent=2)

# # # # # #         print(f"[ViolationStore] JSON report     : {out_path}")
# # # # # #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# # # # # #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# # # # # #         try:
# # # # # #             from utils.db_s3_uploader import finalize_and_upload
# # # # # #             finalize_and_upload(
# # # # # #                 report_path     = out_path,
# # # # # #                 analysis_id     = self.analysis_id,
# # # # # #                 train_detail_id = self.train_detail_id,
# # # # # #             )
# # # # # #         except Exception as exc:
# # # # # #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# # # # # #         return out_path

# # # # # #     # ── Private — deduplication & merging ────────────────────────────────────

# # # # # #     def _deduplicate_by_frame(self) -> None:
# # # # # #         unique: Dict[int, _Violation] = {}
# # # # # #         for v in self._violations:
# # # # # #             if v.frame_index not in unique:
# # # # # #                 unique[v.frame_index] = v
# # # # # #             else:
# # # # # #                 ex = unique[v.frame_index]
# # # # # #                 ex.events  = list(set(ex.events  + v.events))
# # # # # #                 ex.factors = list(set(ex.factors + v.factors))
# # # # # #                 if v.risk_score > ex.risk_score:
# # # # # #                     ex.risk_score = v.risk_score
# # # # # #                     ex.risk_level = v.risk_level
# # # # # #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# # # # # #                     ex.annotated_frame = v.annotated_frame
# # # # # #         self._violations = list(unique.values())

# # # # # #     def _merge_by_time_window(self) -> None:
# # # # # #         if not self._violations:
# # # # # #             return
# # # # # #         self._violations.sort(key=lambda x: x.timestamp)
# # # # # #         merged: List[_Violation] = []
# # # # # #         group  = [self._violations[0]]
# # # # # #         for v in self._violations[1:]:
# # # # # #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# # # # # #                 group.append(v)
# # # # # #             else:
# # # # # #                 merged.append(self._merge_group(group))
# # # # # #                 group = [v]
# # # # # #         merged.append(self._merge_group(group))
# # # # # #         self._violations = merged

# # # # # #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# # # # # #         base            = group[0]
# # # # # #         events: List[str]  = []
# # # # # #         factors: List[str] = []
# # # # # #         max_risk   = base.risk_score
# # # # # #         risk_level = base.risk_level
# # # # # #         best_frame = base.annotated_frame
# # # # # #         for v in group:
# # # # # #             events.extend(v.events)
# # # # # #             factors.extend(v.factors)
# # # # # #             if v.risk_score > max_risk:
# # # # # #                 max_risk, risk_level = v.risk_score, v.risk_level
# # # # # #             if best_frame is None and v.annotated_frame is not None:
# # # # # #                 best_frame = v.annotated_frame
# # # # # #         return _Violation(
# # # # # #             timestamp        = base.timestamp,
# # # # # #             time_str         = base.time_str,
# # # # # #             frame_index      = base.frame_index,
# # # # # #             type             = base.type,
# # # # # #             events           = list(set(events)),
# # # # # #             severity         = base.severity,
# # # # # #             duration         = base.duration,
# # # # # #             risk_score       = max_risk,
# # # # # #             risk_level       = risk_level,
# # # # # #             confidence       = base.confidence,
# # # # # #             factors          = list(set(factors)),
# # # # # #             source_filename  = base.source_filename,
# # # # # #             local_time_str   = base.local_time_str,
# # # # # #             annotated_frame  = best_frame,
# # # # # #         )

# # # # # #     # ── Private — frame extraction & saving ──────────────────────────────────

# # # # # #     def extract_violation_frames(self, video_path: str) -> None:
# # # # # #         print("[ViolationStore] Saving frames...")
# # # # # #         need_video = [v for v in self._violations if v.annotated_frame is None]
# # # # # #         saved = 0

# # # # # #         # First pass: save violations that already have an annotated frame
# # # # # #         for v in self._violations:
# # # # # #             if v.annotated_frame is not None:
# # # # # #                 v.frame_path      = self._save_frame(v.annotated_frame, v.events, v.time_str)
# # # # # #                 v.annotated_frame = None   # free memory
# # # # # #                 saved += 1

# # # # # #         # Second pass: re-read from the source video for any that are missing
# # # # # #         if need_video:
# # # # # #             cap = cv2.VideoCapture(video_path)
# # # # # #             if not cap.isOpened():
# # # # # #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# # # # # #             else:
# # # # # #                 seen: set = set()
# # # # # #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# # # # # #                     if v.frame_index in seen:
# # # # # #                         continue
# # # # # #                     seen.add(v.frame_index)
# # # # # #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# # # # # #                     ret, frame = cap.read()
# # # # # #                     if not ret:
# # # # # #                         continue
# # # # # #                     v.frame_path = self._save_frame(frame, v.events, v.time_str)
# # # # # #                     saved += 1
# # # # # #                 cap.release()

# # # # # #         print(f"[ViolationStore] {saved} frames saved")

# # # # # #     def _save_frame(
# # # # # #         self,
# # # # # #         frame:    np.ndarray,
# # # # # #         events:   List[str],
# # # # # #         time_str: str,
# # # # # #     ) -> str:
# # # # # #         distraction   = "_".join(events)
# # # # # #         filename_time = time_str.replace(":", "-")
# # # # # #         filename      = f"{distraction}_{filename_time}.jpg"
# # # # # #         path          = os.path.join(self.frames_dir, filename)
# # # # # #         ok = cv2.imwrite(
# # # # # #             path,
# # # # # #             cv2.resize(frame, (640, 360)),
# # # # # #             [cv2.IMWRITE_JPEG_QUALITY, 85],
# # # # # #         )
# # # # # #         if not ok:
# # # # # #             print(f"[ViolationStore] imwrite failed: {path}")
# # # # # #         return os.path.join(self.analysis_id, "frames", filename)

# # # # # #     # ── Private — report builder ──────────────────────────────────────────────

# # # # # #     def _build_report(self, processing_time: float = 0.0) -> dict:
# # # # # #         return {
# # # # # #             "analysis_id":     self.analysis_id,
# # # # # #             "train_detail_id": self.train_detail_id,
# # # # # #             "processing_time": round(processing_time, 3),
# # # # # #             # Single video → keep as dict for backwards compat; batch → list
# # # # # #             "video_info": (
# # # # # #                 self.video_infos[0]
# # # # # #                 if len(self.video_infos) == 1
# # # # # #                 else self.video_infos
# # # # # #             ),
# # # # # #             "violations": [
# # # # # #                 {
# # # # # #                     "timestamp":   v.time_str,
# # # # # #                     "frame_index": v.frame_index,
# # # # # #                     "events":      v.events,
# # # # # #                     "severity":    v.severity,
# # # # # #                     "duration":    v.duration,
# # # # # #                     "risk_score":  v.risk_score,
# # # # # #                     "risk_level":  v.risk_level,
# # # # # #                     "confidence":  v.confidence,
# # # # # #                     "factors":     v.factors,
# # # # # #                     "original_video_timestamp": (
# # # # # #                         f"{v.source_filename} {v.local_time_str}"
# # # # # #                     ),
# # # # # #                     "frame_path":  v.frame_path,
# # # # # #                 }
# # # # # #                 for v in self._violations
# # # # # #             ],
# # # # # #         }

# # # # # from __future__ import annotations

# # # # # import json
# # # # # import os
# # # # # from dataclasses import dataclass
# # # # # from typing import Any, Dict, List, Optional

# # # # # import cv2
# # # # # import numpy as np

# # # # # OUTPUTS_ROOT = "outputs"
# # # # # MERGE_WINDOW = 2.0


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # INTERNAL DATA CLASS
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # @dataclass
# # # # # class _Violation:
# # # # #     timestamp:       float
# # # # #     time_str:        str
# # # # #     frame_index:     int
# # # # #     type:            str
# # # # #     events:          List[str]
# # # # #     severity:        str
# # # # #     duration:        float
# # # # #     risk_score:      int
# # # # #     risk_level:      str
# # # # #     confidence:      float
# # # # #     factors:         List[str]
# # # # #     source_filename: str                  = ""    # e.g. "ch01.mp4"
# # # # #     local_time_str:  str                  = ""    # local time within that file e.g. "00:00:18"
# # # # #     frame_path:      Optional[str]        = None
# # # # #     annotated_frame: Optional[np.ndarray] = None


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # VIOLATION STORE
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # class ViolationStore:
# # # # #     """
# # # # #     Accumulates all violations found across one analysis run (single video
# # # # #     or a multi-video batch that shares the same analysis_id / folder_name).

# # # # #     Usage
# # # # #     ─────
# # # # #     1. Construct once per analysis run.
# # # # #     2. Call record_violation() from the pipeline for every distraction event.
# # # # #     3. Call finalize() at the very end — it deduplicates, merges, saves frames,
# # # # #        writes the JSON report, then calls db_s3_uploader.finalize_and_upload()
# # # # #        to push everything to S3 and record results in the DB.
# # # # #     """

# # # # #     def __init__(
# # # # #         self,
# # # # #         analysis_id:     str,
# # # # #         train_detail_id: int,
# # # # #         video_info:      Optional[Dict[str, Any]] = None,
# # # # #     ):
# # # # #         self.analysis_id     = analysis_id
# # # # #         self.train_detail_id = train_detail_id
# # # # #         # video_infos is always a list:
# # # # #         #   • 1 entry for single-video runs
# # # # #         #   • N entries for batch runs (add_video_info called per video)
# # # # #         self.video_infos: List[Dict[str, Any]] = (
# # # # #             [video_info] if video_info is not None else []
# # # # #         )

# # # # #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# # # # #         self.frames_dir = os.path.join(self.output_dir, "frames")
# # # # #         os.makedirs(self.frames_dir, exist_ok=True)

# # # # #         self._violations:  List[_Violation] = []
# # # # #         self._seen_frames: set              = set()
# # # # #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# # # # #     # ── Public helpers ────────────────────────────────────────────────────────

# # # # #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# # # # #         """
# # # # #         Register one video's metadata into the shared store.
# # # # #         Called once per video in batch mode.
# # # # #         """
# # # # #         self.video_infos.append(video_info)

# # # # #     def record_violation(
# # # # #         self,
# # # # #         annotated_frame:  np.ndarray,
# # # # #         video_time:       float,
# # # # #         frame_index:      int,
# # # # #         event_type:       str,
# # # # #         original_frame:   Optional[np.ndarray] = None,
# # # # #         severity:         str   = "CRITICAL",
# # # # #         confidence:       float = 0.9,
# # # # #         risk_score:       int   = 80,
# # # # #         risk_level:       str   = "CRITICAL",
# # # # #         factors:          Optional[List[str]] = None,
# # # # #         duration:         float = 0.0,
# # # # #         source_filename:  str   = "",    # original filename e.g. "ch01.mp4"
# # # # #         local_video_time: float = -1.0,  # video_time before global offset; -1 = same as video_time
# # # # #     ) -> None:
# # # # #         """
# # # # #         Record one distraction event.

# # # # #         Deduplication key is (frame_index, event_type) so that:
# # # # #           • the same violation type on the same frame is recorded only once
# # # # #           • different violation types on the same frame are each recorded
# # # # #           • frame numbers from different videos never collide because
# # # # #             main.py applies a frame_offset to make every global frame_index
# # # # #             unique across the batch
# # # # #         """
# # # # #         dedup_key = (frame_index, event_type)
# # # # #         if dedup_key in self._seen_frames:
# # # # #             return
# # # # #         self._seen_frames.add(dedup_key)

# # # # #         factors  = factors or []
# # # # #         t        = int(round(video_time))
# # # # #         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

# # # # #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# # # # #         local_str = (
# # # # #             f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
# # # # #         )

# # # # #         self._violations.append(
# # # # #             _Violation(
# # # # #                 timestamp        = video_time,
# # # # #                 time_str         = time_str,
# # # # #                 frame_index      = frame_index,
# # # # #                 type             = event_type,
# # # # #                 events           = [event_type],
# # # # #                 severity         = severity,
# # # # #                 duration         = round(duration, 2),
# # # # #                 risk_score       = risk_score,
# # # # #                 risk_level       = risk_level,
# # # # #                 confidence       = round(confidence, 3),
# # # # #                 factors          = list(factors),
# # # # #                 source_filename  = source_filename,
# # # # #                 local_time_str   = local_str,
# # # # #                 annotated_frame  = (
# # # # #                     annotated_frame.copy() if annotated_frame is not None else None
# # # # #                 ),
# # # # #             )
# # # # #         )

# # # # #     # ── Finalize ──────────────────────────────────────────────────────────────

# # # # #     def finalize(self, processing_time: float = 0.0) -> str:
# # # # #         """
# # # # #         1. Deduplicate violations that share the same frame.
# # # # #         2. Merge violations that are within MERGE_WINDOW seconds of each other.
# # # # #         3. Save annotated frame images to disk (falls back to re-reading from
# # # # #            the source video when no annotated frame was captured).
# # # # #         4. Write analysis_report.json.
# # # # #         5. Upload frames + report to S3 and record in DB via db_s3_uploader.
# # # # #         Returns the local path to analysis_report.json.
# # # # #         """
# # # # #         self._deduplicate_by_frame()
# # # # #         self._merge_by_time_window()

# # # # #         # Extract frames from every video in the batch (or the single video)
# # # # #         for vi in self.video_infos:
# # # # #             if vi and vi.get("videoPath"):
# # # # #                 self.extract_violation_frames(vi["videoPath"])

# # # # #         report   = self._build_report(processing_time=processing_time)
# # # # #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# # # # #         with open(out_path, "w", encoding="utf-8") as f:
# # # # #             json.dump(report, f, indent=2)

# # # # #         print(f"[ViolationStore] JSON report     : {out_path}")
# # # # #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# # # # #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# # # # #         try:
# # # # #             from utils.db_s3_uploader import finalize_and_upload
# # # # #             finalize_and_upload(
# # # # #                 report_path     = out_path,
# # # # #                 analysis_id     = self.analysis_id,
# # # # #                 train_detail_id = self.train_detail_id,
# # # # #             )
# # # # #         except Exception as exc:
# # # # #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# # # # #         return out_path

# # # # #     # ── Private — deduplication & merging ────────────────────────────────────

# # # # #     def _deduplicate_by_frame(self) -> None:
# # # # #         unique: Dict[int, _Violation] = {}
# # # # #         for v in self._violations:
# # # # #             if v.frame_index not in unique:
# # # # #                 unique[v.frame_index] = v
# # # # #             else:
# # # # #                 ex = unique[v.frame_index]
# # # # #                 ex.events  = list(set(ex.events  + v.events))
# # # # #                 ex.factors = list(set(ex.factors + v.factors))
# # # # #                 if v.risk_score > ex.risk_score:
# # # # #                     ex.risk_score = v.risk_score
# # # # #                     ex.risk_level = v.risk_level
# # # # #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# # # # #                     ex.annotated_frame = v.annotated_frame
# # # # #         self._violations = list(unique.values())

# # # # #     def _merge_by_time_window(self) -> None:
# # # # #         if not self._violations:
# # # # #             return
# # # # #         self._violations.sort(key=lambda x: x.timestamp)
# # # # #         merged: List[_Violation] = []
# # # # #         group  = [self._violations[0]]
# # # # #         for v in self._violations[1:]:
# # # # #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# # # # #                 group.append(v)
# # # # #             else:
# # # # #                 merged.append(self._merge_group(group))
# # # # #                 group = [v]
# # # # #         merged.append(self._merge_group(group))
# # # # #         self._violations = merged

# # # # #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# # # # #         base             = group[0]
# # # # #         events: List[str]  = []
# # # # #         factors: List[str] = []
# # # # #         max_risk   = base.risk_score
# # # # #         risk_level = base.risk_level
# # # # #         best_frame = base.annotated_frame
# # # # #         for v in group:
# # # # #             events.extend(v.events)
# # # # #             factors.extend(v.factors)
# # # # #             if v.risk_score > max_risk:
# # # # #                 max_risk, risk_level = v.risk_score, v.risk_level
# # # # #             if best_frame is None and v.annotated_frame is not None:
# # # # #                 best_frame = v.annotated_frame
# # # # #         return _Violation(
# # # # #             timestamp        = base.timestamp,
# # # # #             time_str         = base.time_str,
# # # # #             frame_index      = base.frame_index,
# # # # #             type             = base.type,
# # # # #             events           = list(set(events)),
# # # # #             severity         = base.severity,
# # # # #             duration         = base.duration,
# # # # #             risk_score       = max_risk,
# # # # #             risk_level       = risk_level,
# # # # #             confidence       = base.confidence,
# # # # #             factors          = list(set(factors)),
# # # # #             source_filename  = base.source_filename,
# # # # #             local_time_str   = base.local_time_str,
# # # # #             annotated_frame  = best_frame,
# # # # #         )

# # # # #     # ── Private — frame extraction & saving ──────────────────────────────────

# # # # #     def extract_violation_frames(self, video_path: str) -> None:
# # # # #         print("[ViolationStore] Saving frames...")
# # # # #         need_video = [v for v in self._violations if v.annotated_frame is None]
# # # # #         saved = 0

# # # # #         # First pass: save violations that already have an annotated frame
# # # # #         for v in self._violations:
# # # # #             if v.annotated_frame is not None:
# # # # #                 v.frame_path      = self._save_frame(
# # # # #                     v.annotated_frame, v.events, v.time_str, v.frame_index
# # # # #                 )
# # # # #                 v.annotated_frame = None   # free memory
# # # # #                 saved += 1

# # # # #         # Second pass: re-read from the source video for any that are missing
# # # # #         if need_video:
# # # # #             cap = cv2.VideoCapture(video_path)
# # # # #             if not cap.isOpened():
# # # # #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# # # # #             else:
# # # # #                 seen: set = set()
# # # # #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# # # # #                     if v.frame_index in seen:
# # # # #                         continue
# # # # #                     seen.add(v.frame_index)
# # # # #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# # # # #                     ret, frame = cap.read()
# # # # #                     if not ret:
# # # # #                         continue
# # # # #                     v.frame_path = self._save_frame(
# # # # #                         frame, v.events, v.time_str, v.frame_index
# # # # #                     )
# # # # #                     saved += 1
# # # # #                 cap.release()

# # # # #         print(f"[ViolationStore] {saved} frames saved")

# # # # #     def _save_frame(
# # # # #         self,
# # # # #         frame:       np.ndarray,
# # # # #         events:      List[str],
# # # # #         time_str:    str,
# # # # #         frame_index: int,
# # # # #     ) -> str:
# # # # #         """
# # # # #         Save a single violation frame as a JPEG.

# # # # #         Filename format:  <events>_<HH-MM-SS>_f<frame_index>.jpg
# # # # #         Example:          seat_absence_00-01-14_f2106.jpg
# # # # #                           seat_absence_drowsy_00-03-02_f5058.jpg
# # # # #         """
# # # # #         distraction   = "_".join(sorted(events))   # sorted for deterministic name
# # # # #         filename_time = time_str.replace(":", "-")
# # # # #         filename      = f"{distraction}_{filename_time}_f{frame_index}.jpg"
# # # # #         path          = os.path.join(self.frames_dir, filename)
# # # # #         ok = cv2.imwrite(
# # # # #             path,
# # # # #             cv2.resize(frame, (640, 360)),
# # # # #             [cv2.IMWRITE_JPEG_QUALITY, 85],
# # # # #         )
# # # # #         if not ok:
# # # # #             print(f"[ViolationStore] imwrite failed: {path}")
# # # # #         return os.path.join(self.analysis_id, "frames", filename)

# # # # #     # ── Private — report builder ──────────────────────────────────────────────

# # # # #     def _build_report(self, processing_time: float = 0.0) -> dict:
# # # # #         return {
# # # # #             "analysis_id":     self.analysis_id,
# # # # #             "train_detail_id": self.train_detail_id,
# # # # #             "processing_time": round(processing_time, 3),
# # # # #             # Single video → keep as dict for backwards compat; batch → list
# # # # #             "video_info": (
# # # # #                 self.video_infos[0]
# # # # #                 if len(self.video_infos) == 1
# # # # #                 else self.video_infos
# # # # #             ),
# # # # #             "violations": [
# # # # #                 {
# # # # #                     "timestamp":   v.time_str,
# # # # #                     "frame_index": v.frame_index,
# # # # #                     "events":      v.events,
# # # # #                     "severity":    v.severity,
# # # # #                     "duration":    v.duration,
# # # # #                     "risk_score":  v.risk_score,
# # # # #                     "risk_level":  v.risk_level,
# # # # #                     "confidence":  v.confidence,
# # # # #                     "factors":     v.factors,
# # # # #                     "original_video_timestamp": (
# # # # #                         f"{v.source_filename} {v.local_time_str}"
# # # # #                     ),
# # # # #                     "frame_path":  v.frame_path,
# # # # #                 }
# # # # #                 for v in self._violations
# # # # #             ],
# # # # #         }


# # # # from __future__ import annotations

# # # # import json
# # # # import os
# # # # from dataclasses import dataclass
# # # # from typing import Any, Dict, List, Optional

# # # # import cv2
# # # # import numpy as np

# # # # OUTPUTS_ROOT = "outputs"
# # # # MERGE_WINDOW = 2.0


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # INTERNAL DATA CLASS
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # @dataclass
# # # # class _Violation:
# # # #     timestamp:       float
# # # #     time_str:        str
# # # #     frame_index:     int
# # # #     type:            str
# # # #     events:          List[str]
# # # #     severity:        str
# # # #     duration:        float
# # # #     risk_score:      int
# # # #     risk_level:      str
# # # #     confidence:      float
# # # #     factors:         List[str]
# # # #     source_filename: str                  = ""    # DB filename e.g. "mobile.mp4"
# # # #     local_time_str:  str                  = ""    # local time within that file e.g. "00:00:23"
# # # #     frame_path:      Optional[str]        = None
# # # #     annotated_frame: Optional[np.ndarray] = None


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # VIOLATION STORE
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # class ViolationStore:
# # # #     """
# # # #     Accumulates all violations found across one analysis run (single video
# # # #     or a multi-video batch that shares the same analysis_id / folder_name).

# # # #     Batch mode usage (api.py)
# # # #     ─────────────────────────
# # # #     1. Construct ONCE for the whole folder (no video_info in __init__).
# # # #     2. Pass as shared_vstore= to each GadgetDetectionPipeline.
# # # #        The pipeline calls add_video_info() automatically.
# # # #     3. Call finalize() ONCE after all videos in the folder are done.

# # # #     Standalone mode usage (CLI / single video)
# # # #     ──────────────────────────────────────────
# # # #     1. Construct with video_info= for the single video.
# # # #     2. Pipeline calls finalize() automatically at the end of run().
# # # #     """

# # # #     def __init__(
# # # #         self,
# # # #         analysis_id:     str,
# # # #         train_detail_id: int,
# # # #         video_info:      Optional[Dict[str, Any]] = None,
# # # #     ):
# # # #         self.analysis_id     = analysis_id
# # # #         self.train_detail_id = train_detail_id
# # # #         # Always a list — 0 entries until add_video_info() is called (batch mode),
# # # #         # or 1 entry when video_info is provided (standalone mode).
# # # #         self.video_infos: List[Dict[str, Any]] = (
# # # #             [video_info] if video_info is not None else []
# # # #         )

# # # #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# # # #         self.frames_dir = os.path.join(self.output_dir, "frames")
# # # #         os.makedirs(self.frames_dir, exist_ok=True)

# # # #         self._violations:  List[_Violation] = []
# # # #         self._seen_frames: set              = set()
# # # #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# # # #     # ── Public helpers ────────────────────────────────────────────────────────

# # # #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# # # #         """Append one video's metadata. Called once per video in batch mode."""
# # # #         self.video_infos.append(video_info)

# # # #     def record_violation(
# # # #         self,
# # # #         annotated_frame:  np.ndarray,
# # # #         video_time:       float,          # global timestamp (offset-adjusted)
# # # #         frame_index:      int,            # global frame index (offset-adjusted)
# # # #         event_type:       str,
# # # #         original_frame:   Optional[np.ndarray] = None,
# # # #         severity:         str   = "CRITICAL",
# # # #         confidence:       float = 0.9,
# # # #         risk_score:       int   = 80,
# # # #         risk_level:       str   = "CRITICAL",
# # # #         factors:          Optional[List[str]] = None,
# # # #         duration:         float = 0.0,
# # # #         source_filename:  str   = "",     # DB filename shown in original_video_timestamp
# # # #         local_video_time: float = -1.0,   # local time within the source file; -1 = same as video_time
# # # #     ) -> None:
# # # #         """
# # # #         Record one distraction event.

# # # #         Deduplication key is (frame_index, event_type):
# # # #           • same event on the same global frame is recorded once
# # # #           • different events on the same global frame are each recorded
# # # #           • global frame_index is unique across videos (frame_offset applied in main.py)
# # # #         """
# # # #         dedup_key = (frame_index, event_type)
# # # #         if dedup_key in self._seen_frames:
# # # #             return
# # # #         self._seen_frames.add(dedup_key)

# # # #         factors  = factors or []
# # # #         t        = int(round(video_time))
# # # #         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

# # # #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# # # #         local_str = (
# # # #             f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
# # # #         )

# # # #         self._violations.append(
# # # #             _Violation(
# # # #                 timestamp        = video_time,
# # # #                 time_str         = time_str,
# # # #                 frame_index      = frame_index,
# # # #                 type             = event_type,
# # # #                 events           = [event_type],
# # # #                 severity         = severity,
# # # #                 duration         = round(duration, 2),
# # # #                 risk_score       = risk_score,
# # # #                 risk_level       = risk_level,
# # # #                 confidence       = round(confidence, 3),
# # # #                 factors          = list(factors),
# # # #                 source_filename  = source_filename,
# # # #                 local_time_str   = local_str,
# # # #                 annotated_frame  = (
# # # #                     annotated_frame.copy() if annotated_frame is not None else None
# # # #                 ),
# # # #             )
# # # #         )

# # # #     # ── Finalize ──────────────────────────────────────────────────────────────

# # # #     def finalize(self, processing_time: float = 0.0) -> str:
# # # #         """
# # # #         Deduplicate → merge → save frames → write JSON → upload to S3/DB.
# # # #         Returns the local path to analysis_report.json.
# # # #         """
# # # #         self._deduplicate_by_frame()
# # # #         self._merge_by_time_window()

# # # #         # Extract frames from every video in the batch (or the single video)
# # # #         for vi in self.video_infos:
# # # #             if vi and vi.get("videoPath"):
# # # #                 self.extract_violation_frames(vi["videoPath"])

# # # #         report   = self._build_report(processing_time=processing_time)
# # # #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# # # #         with open(out_path, "w", encoding="utf-8") as f:
# # # #             json.dump(report, f, indent=2)

# # # #         print(f"[ViolationStore] JSON report     : {out_path}")
# # # #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# # # #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# # # #         try:
# # # #             from utils.db_s3_uploader import finalize_and_upload
# # # #             finalize_and_upload(
# # # #                 report_path     = out_path,
# # # #                 analysis_id     = self.analysis_id,
# # # #                 train_detail_id = self.train_detail_id,
# # # #             )
# # # #         except Exception as exc:
# # # #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# # # #         return out_path

# # # #     # ── Private — deduplication & merging ────────────────────────────────────

# # # #     def _deduplicate_by_frame(self) -> None:
# # # #         unique: Dict[int, _Violation] = {}
# # # #         for v in self._violations:
# # # #             if v.frame_index not in unique:
# # # #                 unique[v.frame_index] = v
# # # #             else:
# # # #                 ex = unique[v.frame_index]
# # # #                 ex.events  = list(set(ex.events  + v.events))
# # # #                 ex.factors = list(set(ex.factors + v.factors))
# # # #                 if v.risk_score > ex.risk_score:
# # # #                     ex.risk_score = v.risk_score
# # # #                     ex.risk_level = v.risk_level
# # # #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# # # #                     ex.annotated_frame = v.annotated_frame
# # # #         self._violations = list(unique.values())

# # # #     def _merge_by_time_window(self) -> None:
# # # #         if not self._violations:
# # # #             return
# # # #         self._violations.sort(key=lambda x: x.timestamp)
# # # #         merged: List[_Violation] = []
# # # #         group  = [self._violations[0]]
# # # #         for v in self._violations[1:]:
# # # #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# # # #                 group.append(v)
# # # #             else:
# # # #                 merged.append(self._merge_group(group))
# # # #                 group = [v]
# # # #         merged.append(self._merge_group(group))
# # # #         self._violations = merged

# # # #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# # # #         base             = group[0]
# # # #         events: List[str]  = []
# # # #         factors: List[str] = []
# # # #         max_risk   = base.risk_score
# # # #         risk_level = base.risk_level
# # # #         best_frame = base.annotated_frame
# # # #         for v in group:
# # # #             events.extend(v.events)
# # # #             factors.extend(v.factors)
# # # #             if v.risk_score > max_risk:
# # # #                 max_risk, risk_level = v.risk_score, v.risk_level
# # # #             if best_frame is None and v.annotated_frame is not None:
# # # #                 best_frame = v.annotated_frame
# # # #         return _Violation(
# # # #             timestamp        = base.timestamp,
# # # #             time_str         = base.time_str,
# # # #             frame_index      = base.frame_index,
# # # #             type             = base.type,
# # # #             events           = list(set(events)),
# # # #             severity         = base.severity,
# # # #             duration         = base.duration,
# # # #             risk_score       = max_risk,
# # # #             risk_level       = risk_level,
# # # #             confidence       = base.confidence,
# # # #             factors          = list(set(factors)),
# # # #             source_filename  = base.source_filename,
# # # #             local_time_str   = base.local_time_str,
# # # #             annotated_frame  = best_frame,
# # # #         )

# # # #     # ── Private — frame extraction & saving ──────────────────────────────────

# # # #     def extract_violation_frames(self, video_path: str) -> None:
# # # #         print("[ViolationStore] Saving frames...")
# # # #         need_video = [v for v in self._violations if v.annotated_frame is None]
# # # #         saved = 0

# # # #         # First pass: save violations that already have an annotated frame
# # # #         for v in self._violations:
# # # #             if v.annotated_frame is not None:
# # # #                 v.frame_path      = self._save_frame(v.annotated_frame, v.events, v.time_str)
# # # #                 v.annotated_frame = None   # free memory
# # # #                 saved += 1

# # # #         # Second pass: re-read from the source video for any that are missing
# # # #         if need_video:
# # # #             cap = cv2.VideoCapture(video_path)
# # # #             if not cap.isOpened():
# # # #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# # # #             else:
# # # #                 seen: set = set()
# # # #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# # # #                     if v.frame_index in seen:
# # # #                         continue
# # # #                     seen.add(v.frame_index)
# # # #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# # # #                     ret, frame = cap.read()
# # # #                     if not ret:
# # # #                         continue
# # # #                     v.frame_path = self._save_frame(frame, v.events, v.time_str)
# # # #                     saved += 1
# # # #                 cap.release()

# # # #         print(f"[ViolationStore] {saved} frames saved")

# # # #     def _save_frame(
# # # #         self,
# # # #         frame:    np.ndarray,
# # # #         events:   List[str],
# # # #         time_str: str,
# # # #     ) -> str:
# # # #         """
# # # #         Save a single violation frame as JPEG.

# # # #         Filename format:  <events>_<HH-MM-SS>.jpg
# # # #         Example:          seat_absence_00-01-14.jpg
# # # #                           seat_absence_drowsy_00-03-02.jpg
# # # #                           phone_use_00-00-24.jpg
# # # #         """
# # # #         distraction   = "_".join(sorted(events))   # sorted for deterministic name
# # # #         filename_time = time_str.replace(":", "-")
# # # #         filename      = f"{distraction}_{filename_time}.jpg"
# # # #         path          = os.path.join(self.frames_dir, filename)
# # # #         ok = cv2.imwrite(
# # # #             path,
# # # #             cv2.resize(frame, (640, 360)),
# # # #             [cv2.IMWRITE_JPEG_QUALITY, 85],
# # # #         )
# # # #         if not ok:
# # # #             print(f"[ViolationStore] imwrite failed: {path}")
# # # #         return os.path.join(self.analysis_id, "frames", filename)

# # # #     # ── Private — report builder ──────────────────────────────────────────────

# # # #     def _build_report(self, processing_time: float = 0.0) -> dict:
# # # #         return {
# # # #             "analysis_id":     self.analysis_id,
# # # #             "train_detail_id": self.train_detail_id,
# # # #             "processing_time": round(processing_time, 3),
# # # #             # Single video → dict (backwards compat); batch → list
# # # #             "video_info": (
# # # #                 self.video_infos[0]
# # # #                 if len(self.video_infos) == 1
# # # #                 else self.video_infos
# # # #             ),
# # # #             "violations": [
# # # #                 {
# # # #                     "timestamp":   v.time_str,
# # # #                     "frame_index": v.frame_index,
# # # #                     "events":      v.events,
# # # #                     "severity":    v.severity,
# # # #                     "duration":    v.duration,
# # # #                     "risk_score":  v.risk_score,
# # # #                     "risk_level":  v.risk_level,
# # # #                     "confidence":  v.confidence,
# # # #                     "factors":     v.factors,
# # # #                     # "original_video_timestamp" = "<db_filename> <local_HH:MM:SS>"
# # # #                     # When timestamp == original (video 1), local_time_str == time_str
# # # #                     "original_video_timestamp": (
# # # #                         f"{v.source_filename} {v.local_time_str}"
# # # #                     ),
# # # #                     "frame_path":  v.frame_path,
# # # #                 }
# # # #                 for v in self._violations
# # # #             ],
# # # #         }


# # # from __future__ import annotations

# # # import json
# # # import os
# # # from dataclasses import dataclass
# # # from typing import Any, Dict, List, Optional

# # # import cv2
# # # import numpy as np

# # # OUTPUTS_ROOT = "outputs"
# # # MERGE_WINDOW = 2.0


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # # INTERNAL DATA CLASS
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # @dataclass
# # # class _Violation:
# # #     timestamp:       float
# # #     time_str:        str
# # #     frame_index:     int
# # #     type:            str
# # #     events:          List[str]
# # #     severity:        str
# # #     duration:        float
# # #     risk_score:      int
# # #     risk_level:      str
# # #     confidence:      float
# # #     factors:         List[str]
# # #     source_filename: str                  = ""    # DB filename e.g. "mobile.mp4"
# # #     local_time_str:  str                  = ""    # local time within that file e.g. "00:00:23"
# # #     frame_path:      Optional[str]        = None
# # #     annotated_frame: Optional[np.ndarray] = None


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # # VIOLATION STORE
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # class ViolationStore:
# # #     """
# # #     Accumulates all violations found across one analysis run (single video
# # #     or a multi-video batch that shares the same analysis_id / folder_name).

# # #     Batch mode usage (api.py)
# # #     ─────────────────────────
# # #     1. Construct ONCE for the whole folder (no video_info in __init__).
# # #     2. Pass as shared_vstore= to each GadgetDetectionPipeline.
# # #        The pipeline calls add_video_info() automatically.
# # #     3. Call finalize() ONCE after all videos in the folder are done.

# # #     Standalone mode usage (CLI / single video)
# # #     ──────────────────────────────────────────
# # #     1. Construct with video_info= for the single video.
# # #     2. Pipeline calls finalize() automatically at the end of run().
# # #     """

# # #     def __init__(
# # #         self,
# # #         analysis_id:     str,
# # #         train_detail_id: int,
# # #         video_info:      Optional[Dict[str, Any]] = None,
# # #     ):
# # #         self.analysis_id     = analysis_id
# # #         self.train_detail_id = train_detail_id
# # #         # Always a list — 0 entries until add_video_info() is called (batch mode),
# # #         # or 1 entry when video_info is provided (standalone mode).
# # #         self.video_infos: List[Dict[str, Any]] = (
# # #             [video_info] if video_info is not None else []
# # #         )

# # #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# # #         self.frames_dir = os.path.join(self.output_dir, "frames")
# # #         os.makedirs(self.frames_dir, exist_ok=True)

# # #         self._violations:  List[_Violation] = []
# # #         self._seen_frames: set              = set()
# # #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# # #     # ── Public helpers ────────────────────────────────────────────────────────

# # #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# # #         """Append one video's metadata. Called once per video in batch mode."""
# # #         self.video_infos.append(video_info)

# # #     def record_violation(
# # #         self,
# # #         annotated_frame:  np.ndarray,
# # #         video_time:       float,          # global timestamp (offset-adjusted)
# # #         frame_index:      int,            # global frame index (offset-adjusted)
# # #         event_type:       str,
# # #         original_frame:   Optional[np.ndarray] = None,
# # #         severity:         str   = "CRITICAL",
# # #         confidence:       float = 0.9,
# # #         risk_score:       int   = 80,
# # #         risk_level:       str   = "CRITICAL",
# # #         factors:          Optional[List[str]] = None,
# # #         duration:         float = 0.0,
# # #         source_filename:  str   = "",     # DB filename shown in original_video_timestamp
# # #         local_video_time: float = -1.0,   # local time within the source file; -1 = same as video_time
# # #     ) -> None:
# # #         """
# # #         Record one distraction event.

# # #         Deduplication key is (frame_index, event_type):
# # #           • same event on the same global frame is recorded once
# # #           • different events on the same global frame are each recorded
# # #           • global frame_index is unique across videos (frame_offset applied in main.py)
# # #         """
# # #         dedup_key = (frame_index, event_type)
# # #         if dedup_key in self._seen_frames:
# # #             return
# # #         self._seen_frames.add(dedup_key)

# # #         factors  = factors or []
# # #         t        = int(round(video_time))
# # #         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

# # #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# # #         local_str = (
# # #             f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
# # #         )

# # #         self._violations.append(
# # #             _Violation(
# # #                 timestamp        = video_time,
# # #                 time_str         = time_str,
# # #                 frame_index      = frame_index,
# # #                 type             = event_type,
# # #                 events           = [event_type],
# # #                 severity         = severity,
# # #                 duration         = round(duration, 2),
# # #                 risk_score       = risk_score,
# # #                 risk_level       = risk_level,
# # #                 confidence       = round(confidence, 3),
# # #                 factors          = list(factors),
# # #                 source_filename  = source_filename,
# # #                 local_time_str   = local_str,
# # #                 annotated_frame  = (
# # #                     annotated_frame.copy() if annotated_frame is not None else None
# # #                 ),
# # #             )
# # #         )

# # #     # ── Finalize ──────────────────────────────────────────────────────────────

# # #     def finalize(self, processing_time: float = 0.0) -> str:
# # #         """
# # #         Deduplicate → merge → save frames → write JSON → upload to S3/DB.
# # #         Returns the local path to analysis_report.json.
# # #         """
# # #         self._deduplicate_by_frame()
# # #         self._merge_by_time_window()

# # #         # Extract frames from every video in the batch (or the single video)
# # #         for vi in self.video_infos:
# # #             if vi and vi.get("videoPath"):
# # #                 self.extract_violation_frames(vi["videoPath"])

# # #         report   = self._build_report(processing_time=processing_time)
# # #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# # #         with open(out_path, "w", encoding="utf-8") as f:
# # #             json.dump(report, f, indent=2)

# # #         print(f"[ViolationStore] JSON report     : {out_path}")
# # #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# # #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# # #         try:
# # #             from utils.db_s3_uploader import finalize_and_upload
# # #             finalize_and_upload(
# # #                 report_path     = out_path,
# # #                 analysis_id     = self.analysis_id,
# # #                 train_detail_id = self.train_detail_id,
# # #             )
# # #         except Exception as exc:
# # #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# # #         return out_path

# # #     # ── Private — deduplication & merging ────────────────────────────────────

# # #     def _deduplicate_by_frame(self) -> None:
# # #         unique: Dict[int, _Violation] = {}
# # #         for v in self._violations:
# # #             if v.frame_index not in unique:
# # #                 unique[v.frame_index] = v
# # #             else:
# # #                 ex = unique[v.frame_index]
# # #                 ex.events  = list(set(ex.events  + v.events))
# # #                 ex.factors = list(set(ex.factors + v.factors))
# # #                 if v.risk_score > ex.risk_score:
# # #                     ex.risk_score = v.risk_score
# # #                     ex.risk_level = v.risk_level
# # #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# # #                     ex.annotated_frame = v.annotated_frame
# # #         self._violations = list(unique.values())

# # #     def _merge_by_time_window(self) -> None:
# # #         if not self._violations:
# # #             return
# # #         self._violations.sort(key=lambda x: x.timestamp)
# # #         merged: List[_Violation] = []
# # #         group  = [self._violations[0]]
# # #         for v in self._violations[1:]:
# # #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# # #                 group.append(v)
# # #             else:
# # #                 merged.append(self._merge_group(group))
# # #                 group = [v]
# # #         merged.append(self._merge_group(group))
# # #         self._violations = merged

# # #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# # #         base             = group[0]
# # #         events: List[str]  = []
# # #         factors: List[str] = []
# # #         max_risk   = base.risk_score
# # #         risk_level = base.risk_level
# # #         best_frame = base.annotated_frame
# # #         for v in group:
# # #             events.extend(v.events)
# # #             factors.extend(v.factors)
# # #             if v.risk_score > max_risk:
# # #                 max_risk, risk_level = v.risk_score, v.risk_level
# # #             if best_frame is None and v.annotated_frame is not None:
# # #                 best_frame = v.annotated_frame
# # #         return _Violation(
# # #             timestamp        = base.timestamp,
# # #             time_str         = base.time_str,
# # #             frame_index      = base.frame_index,
# # #             type             = base.type,
# # #             events           = list(set(events)),
# # #             severity         = base.severity,
# # #             duration         = base.duration,
# # #             risk_score       = max_risk,
# # #             risk_level       = risk_level,
# # #             confidence       = base.confidence,
# # #             factors          = list(set(factors)),
# # #             source_filename  = base.source_filename,
# # #             local_time_str   = base.local_time_str,
# # #             annotated_frame  = best_frame,
# # #         )

# # #     # ── Private — frame extraction & saving ──────────────────────────────────

# # #     def extract_violation_frames(self, video_path: str) -> None:
# # #         print("[ViolationStore] Saving frames...")
# # #         need_video = [v for v in self._violations if v.annotated_frame is None]
# # #         saved = 0

# # #         # First pass: save violations that already have an annotated frame
# # #         for v in self._violations:
# # #             if v.annotated_frame is not None:
# # #                 v.frame_path      = self._save_frame(v.annotated_frame, v.events, v.time_str)
# # #                 v.annotated_frame = None   # free memory
# # #                 saved += 1

# # #         # Second pass: re-read from the source video for any that are missing
# # #         if need_video:
# # #             cap = cv2.VideoCapture(video_path)
# # #             if not cap.isOpened():
# # #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# # #             else:
# # #                 seen: set = set()
# # #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# # #                     if v.frame_index in seen:
# # #                         continue
# # #                     seen.add(v.frame_index)
# # #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# # #                     ret, frame = cap.read()
# # #                     if not ret:
# # #                         continue
# # #                     v.frame_path = self._save_frame(frame, v.events, v.time_str)
# # #                     saved += 1
# # #                 cap.release()

# # #         print(f"[ViolationStore] {saved} frames saved")

# # #     def _save_frame(
# # #         self,
# # #         frame:    np.ndarray,
# # #         events:   List[str],
# # #         time_str: str,
# # #     ) -> str:
# # #         """
# # #         Save a single violation frame as JPEG.

# # #         Filename format:  <events>_<HH-MM-SS>.jpg
# # #         Example:          seat_absence_00-01-14.jpg
# # #                           seat_absence_drowsy_00-03-02.jpg
# # #                           phone_use_00-00-24.jpg
# # #         """
# # #         distraction   = "_".join(sorted(events))   # sorted for deterministic name
# # #         filename_time = time_str.replace(":", "-")
# # #         filename      = f"{distraction}_{filename_time}.jpg"
# # #         path          = os.path.join(self.frames_dir, filename)
# # #         ok = cv2.imwrite(
# # #             path,
# # #             cv2.resize(frame, (640, 360)),
# # #             [cv2.IMWRITE_JPEG_QUALITY, 85],
# # #         )
# # #         if not ok:
# # #             print(f"[ViolationStore] imwrite failed: {path}")
# # #         return os.path.join(self.analysis_id, "frames", filename)

# # #     # ── Private — report builder ──────────────────────────────────────────────

# # #     def _build_report(self, processing_time: float = 0.0) -> dict:
# # #         return {
# # #             "analysis_id":     self.analysis_id,
# # #             "train_detail_id": self.train_detail_id,
# # #             "processing_time": round(processing_time, 3),
# # #             # Single video → dict (backwards compat); batch → list
# # #             "video_info": (
# # #                 self.video_infos[0]
# # #                 if len(self.video_infos) == 1
# # #                 else self.video_infos
# # #             ),
# # #             "violations": [
# # #                 {
# # #                     "timestamp":   v.time_str,
# # #                     "frame_index": v.frame_index,
# # #                     "events":      v.events,
# # #                     "severity":    v.severity,
# # #                     "duration":    v.duration,
# # #                     "risk_score":  v.risk_score,
# # #                     "risk_level":  v.risk_level,
# # #                     "confidence":  v.confidence,
# # #                     "factors":     v.factors,
# # #                     # "original_video_timestamp" = "<db_filename> <local_HH:MM:SS>"
# # #                     # When timestamp == original (video 1), local_time_str == time_str
# # #                     "original_video_timestamp": (
# # #                         f"{v.source_filename} {v.local_time_str}"
# # #                     ),
# # #                     "frame_path":  v.frame_path,
# # #                 }
# # #                 for v in self._violations
# # #             ],
# # #         }


# # # # from __future__ import annotations

# # # # import json
# # # # import os
# # # # from dataclasses import dataclass
# # # # from typing import Any, Dict, List, Optional

# # # # import cv2
# # # # import numpy as np

# # # # OUTPUTS_ROOT = "outputs"
# # # # MERGE_WINDOW = 2.0


# # # # @dataclass
# # # # class _Violation:
# # # #     timestamp:       float
# # # #     time_str:        str
# # # #     frame_index:     int
# # # #     type:            str
# # # #     events:          List[str]
# # # #     severity:        str
# # # #     duration:        float
# # # #     risk_score:      int
# # # #     risk_level:      str
# # # #     confidence:      float
# # # #     factors:         List[str]
# # # #     source_filename: str                   = ""    # e.g. "ax.mp4"
# # # #     local_time_str:  str                   = ""    # local time within that file e.g. "00:00:18"
# # # #     frame_path:      Optional[str]         = None
# # # #     annotated_frame: Optional[np.ndarray]  = None


# # # # class ViolationStore:

# # # #     def __init__(self, analysis_id: str, train_detail_id: int, video_info: Optional[Dict[str, Any]] = None):
# # # #         self.analysis_id     = analysis_id
# # # #         self.train_detail_id = train_detail_id
# # # #         # video_infos is always a list — 1 entry for single-video, N entries for batch
# # # #         self.video_infos: List[Dict[str, Any]] = [video_info] if video_info is not None else []

# # # #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# # # #         self.frames_dir = os.path.join(self.output_dir, "frames")
# # # #         os.makedirs(self.frames_dir, exist_ok=True)

# # # #         self._violations: List[_Violation] = []
# # # #         self._seen_frames: set             = set()
# # # #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# # # #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# # # #         """
# # # #         Register one video's metadata into the shared store.
# # # #         Called once per video in batch mode (api.py passes shared_vstore).
# # # #         """
# # # #         self.video_infos.append(video_info)

# # # #     def record_violation(
# # # #         self,
# # # #         annotated_frame: np.ndarray,
# # # #         video_time:      float,
# # # #         frame_index:     int,
# # # #         event_type:      str,
# # # #         original_frame:  Optional[np.ndarray] = None,
# # # #         severity:        str   = "CRITICAL",
# # # #         confidence:      float = 0.9,
# # # #         risk_score:      int   = 80,
# # # #         risk_level:      str   = "CRITICAL",
# # # #         factors:         Optional[List[str]] = None,
# # # #         duration:        float = 0.0,
# # # #         source_filename: str   = "",   # original upload filename e.g. "ax.mp4"
# # # #         local_video_time: float = -1.0, # raw video_time before offset; -1 = same as video_time
# # # #     ):
# # # #         # Deduplicate on (frame_index, event_type) so that:
# # # #         #  • the same violation type on the same frame is recorded only once
# # # #         #  • different violation types on the same frame are each recorded
# # # #         #  • frame numbers from different videos never collide (frame_offset
# # # #         #    in main.py makes every global frame_index unique across the batch)
# # # #         dedup_key = (frame_index, event_type)
# # # #         if dedup_key in self._seen_frames:
# # # #             return
# # # #         self._seen_frames.add(dedup_key)
# # # #         factors   = factors or []
# # # #         t         = int(round(video_time))
# # # #         time_str  = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"
# # # #         # Build the per-file local timestamp (time within the source video)
# # # #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# # # #         local_str = f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
# # # #         self._violations.append(_Violation(
# # # #             timestamp        = video_time,
# # # #             time_str         = time_str,
# # # #             frame_index      = frame_index,
# # # #             type             = event_type,
# # # #             events           = [event_type],
# # # #             severity         = severity,
# # # #             duration         = round(duration, 2),
# # # #             risk_score       = risk_score,
# # # #             risk_level       = risk_level,
# # # #             confidence       = round(confidence, 3),
# # # #             factors          = list(factors),
# # # #             source_filename  = source_filename,
# # # #             local_time_str   = local_str,
# # # #             annotated_frame  = annotated_frame.copy() if annotated_frame is not None else None,
# # # #         ))

# # # #     def _deduplicate_by_frame(self):
# # # #         unique: Dict[int, _Violation] = {}
# # # #         for v in self._violations:
# # # #             if v.frame_index not in unique:
# # # #                 unique[v.frame_index] = v
# # # #             else:
# # # #                 ex = unique[v.frame_index]
# # # #                 ex.events  = list(set(ex.events  + v.events))
# # # #                 ex.factors = list(set(ex.factors + v.factors))
# # # #                 if v.risk_score > ex.risk_score:
# # # #                     ex.risk_score = v.risk_score
# # # #                     ex.risk_level = v.risk_level
# # # #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# # # #                     ex.annotated_frame = v.annotated_frame
# # # #         self._violations = list(unique.values())

# # # #     def _merge_by_time_window(self):
# # # #         if not self._violations:
# # # #             return
# # # #         self._violations.sort(key=lambda x: x.timestamp)
# # # #         merged = []
# # # #         group  = [self._violations[0]]
# # # #         for v in self._violations[1:]:
# # # #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# # # #                 group.append(v)
# # # #             else:
# # # #                 merged.append(self._merge_group(group))
# # # #                 group = [v]
# # # #         merged.append(self._merge_group(group))
# # # #         self._violations = merged

# # # #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# # # #         base            = group[0]
# # # #         events, factors = [], []
# # # #         max_risk        = base.risk_score
# # # #         risk_level      = base.risk_level
# # # #         best_frame      = base.annotated_frame
# # # #         for v in group:
# # # #             events.extend(v.events)
# # # #             factors.extend(v.factors)
# # # #             if v.risk_score > max_risk:
# # # #                 max_risk, risk_level = v.risk_score, v.risk_level
# # # #             if best_frame is None and v.annotated_frame is not None:
# # # #                 best_frame = v.annotated_frame
# # # #         return _Violation(
# # # #             timestamp        = base.timestamp,
# # # #             time_str         = base.time_str,
# # # #             frame_index      = base.frame_index,
# # # #             type             = base.type,
# # # #             events           = list(set(events)),
# # # #             severity         = base.severity,
# # # #             duration         = base.duration,
# # # #             risk_score       = max_risk,
# # # #             risk_level       = risk_level,
# # # #             confidence       = base.confidence,
# # # #             factors          = list(set(factors)),
# # # #             source_filename  = base.source_filename,
# # # #             local_time_str   = base.local_time_str,
# # # #             annotated_frame  = best_frame,
# # # #         )

# # # #     def extract_violation_frames(self, video_path: str):
# # # #         print("[ViolationStore] Saving frames...")
# # # #         need_video = [v for v in self._violations if v.annotated_frame is None]
# # # #         saved = 0
# # # #         for v in self._violations:
# # # #             if v.annotated_frame is not None:
# # # #                 v.frame_path      = self._save_frame(v.annotated_frame, v.events, v.time_str)
# # # #                 v.annotated_frame = None
# # # #                 saved += 1
# # # #         if need_video:
# # # #             cap = cv2.VideoCapture(video_path)
# # # #             if not cap.isOpened():
# # # #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# # # #             else:
# # # #                 seen: set = set()
# # # #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# # # #                     if v.frame_index in seen:
# # # #                         continue
# # # #                     seen.add(v.frame_index)
# # # #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# # # #                     ret, frame = cap.read()
# # # #                     if not ret:
# # # #                         continue
# # # #                     v.frame_path = self._save_frame(frame, v.events, v.time_str)
# # # #                     saved += 1
# # # #                 cap.release()
# # # #         print(f"[ViolationStore] {saved} frames saved")

# # # #     def _save_frame(self, frame: np.ndarray, events: List[str], time_str: str) -> str:
# # # #         distraction   = "_".join(events)
# # # #         filename_time = time_str.replace(":", "-")
# # # #         filename      = f"{distraction}_{filename_time}.jpg"
# # # #         path          = os.path.join(self.frames_dir, filename)
# # # #         ok = cv2.imwrite(path, cv2.resize(frame, (640, 360)),
# # # #                          [cv2.IMWRITE_JPEG_QUALITY, 85])
# # # #         if not ok:
# # # #             print(f"[ViolationStore] imwrite failed: {path}")
# # # #         return os.path.join(self.analysis_id, "frames", filename)

# # # #     def _build_report(self, processing_time: float = 0.0) -> dict:
# # # #         return {
# # # #             "analysis_id":     self.analysis_id,
# # # #             "train_detail_id": self.train_detail_id,
# # # #             "processing_time": round(processing_time, 3),
# # # #             # Single video → keep as dict for backwards compat; batch → list
# # # #             "video_info": self.video_infos[0] if len(self.video_infos) == 1
# # # #                           else self.video_infos,
# # # #             "violations": [
# # # #                 {
# # # #                     "timestamp":   v.time_str,
# # # #                     "frame_index": v.frame_index,
# # # #                     "events":      v.events,
# # # #                     "severity":    v.severity,
# # # #                     "duration":    v.duration,
# # # #                     "risk_score":  v.risk_score,
# # # #                     "risk_level":  v.risk_level,
# # # #                     "confidence":  v.confidence,
# # # #                     "factors":     v.factors,
# # # #                     "original_video_timestamp": f"{v.source_filename} {v.local_time_str}",
# # # #                     "frame_path":  v.frame_path,
# # # #                 }
# # # #                 for v in self._violations
# # # #             ],
# # # #         }

# # # #     def finalize(self, processing_time: float = 0.0) -> str:
# # # #         self._deduplicate_by_frame()
# # # #         self._merge_by_time_window()
# # # #         # Extract frames from every video in the batch (or the single video)
# # # #         for vi in self.video_infos:
# # # #             if vi and vi.get("videoPath"):
# # # #                 self.extract_violation_frames(vi["videoPath"])
# # # #         report   = self._build_report(processing_time=processing_time)
# # # #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# # # #         with open(out_path, "w", encoding="utf-8") as f:
# # # #             json.dump(report, f, indent=2)
# # # #         print(f"[ViolationStore] JSON report     : {out_path}")
# # # #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# # # #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# # # #         try:
# # # #             from utils.db_s3_uploader import finalize_and_upload
# # # #             finalize_and_upload(
# # # #                 report_path     = out_path,
# # # #                 analysis_id     = self.analysis_id,
# # # #                 train_detail_id = self.train_detail_id,
# # # #             )
# # # #         except Exception as exc:
# # # #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# # # #         return out_path



# # # # utils/violation_store.py
# # # # ─────────────────────────────────────────────────────────────────
# # # # Change from original:
# # # #   _save_frame() now appends _f{frame_index} to the filename so
# # # #   two violations of the same type at the same timestamp never
# # # #   silently overwrite each other.
# # # #   e.g.  phone_use_00-13-16_f6762.jpg
# # # #         seat_absence_00-00-17_f516.jpg
# # # # ─────────────────────────────────────────────────────────────────

# # # from __future__ import annotations

# # # import json
# # # import os
# # # from dataclasses import dataclass
# # # from typing import Any, Dict, List, Optional

# # # import cv2
# # # import numpy as np

# # # OUTPUTS_ROOT = "outputs"
# # # MERGE_WINDOW = 2.0


# # # @dataclass
# # # class _Violation:
# # #     timestamp:       float
# # #     time_str:        str
# # #     frame_index:     int
# # #     type:            str
# # #     events:          List[str]
# # #     severity:        str
# # #     duration:        float
# # #     risk_score:      int
# # #     risk_level:      str
# # #     confidence:      float
# # #     factors:         List[str]
# # #     source_filename: str                  = ""
# # #     local_time_str:  str                  = ""
# # #     frame_path:      Optional[str]        = None
# # #     annotated_frame: Optional[np.ndarray] = None


# # # class ViolationStore:

# # #     def __init__(
# # #         self,
# # #         analysis_id:     str,
# # #         train_detail_id: int,
# # #         video_info:      Optional[Dict[str, Any]] = None,
# # #     ):
# # #         self.analysis_id     = analysis_id
# # #         self.train_detail_id = train_detail_id
# # #         self.video_infos: List[Dict[str, Any]] = (
# # #             [video_info] if video_info is not None else []
# # #         )

# # #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# # #         self.frames_dir = os.path.join(self.output_dir, "frames")
# # #         os.makedirs(self.frames_dir, exist_ok=True)

# # #         self._violations: List[_Violation] = []
# # #         self._seen_frames: set             = set()
# # #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# # #     # ── called once per video in batch mode ──────────────────────
# # #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# # #         self.video_infos.append(video_info)

# # #     # ── called from main.py for every detected violation ─────────
# # #     def record_violation(
# # #         self,
# # #         annotated_frame:  np.ndarray,
# # #         video_time:       float,
# # #         frame_index:      int,
# # #         event_type:       str,
# # #         original_frame:   Optional[np.ndarray] = None,
# # #         severity:         str   = "CRITICAL",
# # #         confidence:       float = 0.9,
# # #         risk_score:       int   = 80,
# # #         risk_level:       str   = "CRITICAL",
# # #         factors:          Optional[List[str]] = None,
# # #         duration:         float = 0.0,
# # #         source_filename:  str   = "",
# # #         local_video_time: float = -1.0,
# # #     ):
# # #         # Deduplicate: same frame + same event type recorded only once.
# # #         # Different event types on the same frame are each recorded.
# # #         # frame_index is globally unique across batch (frame_offset in main.py).
# # #         dedup_key = (frame_index, event_type)
# # #         if dedup_key in self._seen_frames:
# # #             return
# # #         self._seen_frames.add(dedup_key)

# # #         factors  = factors or []
# # #         t        = int(round(video_time))
# # #         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

# # #         # Local timestamp = time within the source video file (before cumulative offset)
# # #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# # #         local_str = f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"

# # #         self._violations.append(_Violation(
# # #             timestamp        = video_time,
# # #             time_str         = time_str,
# # #             frame_index      = frame_index,
# # #             type             = event_type,
# # #             events           = [event_type],
# # #             severity         = severity,
# # #             duration         = round(duration, 2),
# # #             risk_score       = risk_score,
# # #             risk_level       = risk_level,
# # #             confidence       = round(confidence, 3),
# # #             factors          = list(factors),
# # #             source_filename  = source_filename,
# # #             local_time_str   = local_str,
# # #             annotated_frame  = annotated_frame.copy() if annotated_frame is not None else None,
# # #         ))

# # #     # ─────────────────────────────────────────────────────────────

# # #     def _deduplicate_by_frame(self):
# # #         unique: Dict[int, _Violation] = {}
# # #         for v in self._violations:
# # #             if v.frame_index not in unique:
# # #                 unique[v.frame_index] = v
# # #             else:
# # #                 ex = unique[v.frame_index]
# # #                 ex.events  = list(set(ex.events  + v.events))
# # #                 ex.factors = list(set(ex.factors + v.factors))
# # #                 if v.risk_score > ex.risk_score:
# # #                     ex.risk_score = v.risk_score
# # #                     ex.risk_level = v.risk_level
# # #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# # #                     ex.annotated_frame = v.annotated_frame
# # #         self._violations = list(unique.values())

# # #     def _merge_by_time_window(self):
# # #         if not self._violations:
# # #             return
# # #         self._violations.sort(key=lambda x: x.timestamp)
# # #         merged = []
# # #         group  = [self._violations[0]]
# # #         for v in self._violations[1:]:
# # #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# # #                 group.append(v)
# # #             else:
# # #                 merged.append(self._merge_group(group))
# # #                 group = [v]
# # #         merged.append(self._merge_group(group))
# # #         self._violations = merged

# # #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# # #         base            = group[0]
# # #         events, factors = [], []
# # #         max_risk        = base.risk_score
# # #         risk_level      = base.risk_level
# # #         best_frame      = base.annotated_frame
# # #         for v in group:
# # #             events.extend(v.events)
# # #             factors.extend(v.factors)
# # #             if v.risk_score > max_risk:
# # #                 max_risk, risk_level = v.risk_score, v.risk_level
# # #             if best_frame is None and v.annotated_frame is not None:
# # #                 best_frame = v.annotated_frame
# # #         return _Violation(
# # #             timestamp        = base.timestamp,
# # #             time_str         = base.time_str,
# # #             frame_index      = base.frame_index,
# # #             type             = base.type,
# # #             events           = list(set(events)),
# # #             severity         = base.severity,
# # #             duration         = base.duration,
# # #             risk_score       = max_risk,
# # #             risk_level       = risk_level,
# # #             confidence       = base.confidence,
# # #             factors          = list(set(factors)),
# # #             source_filename  = base.source_filename,
# # #             local_time_str   = base.local_time_str,
# # #             annotated_frame  = best_frame,
# # #         )

# # #     # ─────────────────────────────────────────────────────────────

# # #     def extract_violation_frames(self, video_path: str):
# # #         print("[ViolationStore] Saving frames...")
# # #         need_video = [v for v in self._violations if v.annotated_frame is None]
# # #         saved = 0

# # #         for v in self._violations:
# # #             if v.annotated_frame is not None:
# # #                 v.frame_path      = self._save_frame(
# # #                     v.annotated_frame, v.events, v.time_str, v.frame_index
# # #                 )
# # #                 v.annotated_frame = None
# # #                 saved += 1

# # #         if need_video:
# # #             cap = cv2.VideoCapture(video_path)
# # #             if not cap.isOpened():
# # #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# # #             else:
# # #                 seen: set = set()
# # #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# # #                     if v.frame_index in seen:
# # #                         continue
# # #                     seen.add(v.frame_index)
# # #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# # #                     ret, frame = cap.read()
# # #                     if not ret:
# # #                         continue
# # #                     v.frame_path = self._save_frame(
# # #                         frame, v.events, v.time_str, v.frame_index
# # #                     )
# # #                     saved += 1
# # #                 cap.release()

# # #         print(f"[ViolationStore] {saved} frames saved")

# # #     def _save_frame(
# # #         self,
# # #         frame:       np.ndarray,
# # #         events:      List[str],
# # #         time_str:    str,
# # #         frame_index: int,            # ← ADDED: makes filename globally unique
# # #     ) -> str:
# # #         """
# # #         Save one violation frame as JPEG.

# # #         Filename: {event_types}_{hh-mm-ss}_f{frame_index}.jpg
# # #         Example:  phone_use_00-13-16_f6762.jpg
# # #                   seat_absence_00-00-17_f516.jpg

# # #         frame_index prevents two violations of the same type at the
# # #         same timestamp from silently overwriting each other.
# # #         """
# # #         distraction   = "_".join(events)
# # #         filename_time = time_str.replace(":", "-")
# # #         filename      = f"{distraction}_{filename_time}_f{frame_index}.jpg"  # ← CHANGED
# # #         path          = os.path.join(self.frames_dir, filename)
# # #         ok = cv2.imwrite(
# # #             path,
# # #             cv2.resize(frame, (640, 360)),
# # #             [cv2.IMWRITE_JPEG_QUALITY, 85],
# # #         )
# # #         if not ok:
# # #             print(f"[ViolationStore] imwrite failed: {path}")
# # #         return os.path.join(self.analysis_id, "frames", filename)

# # #     # ─────────────────────────────────────────────────────────────

# # #     def _build_report(self, processing_time: float = 0.0) -> dict:
# # #         return {
# # #             "analysis_id":     self.analysis_id,
# # #             "train_detail_id": self.train_detail_id,
# # #             "processing_time": round(processing_time, 3),
# # #             "video_info": (
# # #                 self.video_infos[0] if len(self.video_infos) == 1
# # #                 else self.video_infos
# # #             ),
# # #             "violations": [
# # #                 {
# # #                     "timestamp":   v.time_str,
# # #                     "frame_index": v.frame_index,
# # #                     "events":      v.events,
# # #                     "severity":    v.severity,
# # #                     "duration":    v.duration,
# # #                     "risk_score":  v.risk_score,
# # #                     "risk_level":  v.risk_level,
# # #                     "confidence":  v.confidence,
# # #                     "factors":     v.factors,
# # #                     "original_video_timestamp": f"{v.source_filename} {v.local_time_str}",
# # #                     "frame_path":  v.frame_path,
# # #                 }
# # #                 for v in self._violations
# # #             ],
# # #         }

# # #     def finalize(self, processing_time: float = 0.0) -> str:
# # #         self._deduplicate_by_frame()
# # #         self._merge_by_time_window()

# # #         # Save frames — each video's temp path is in video_infos
# # #         for vi in self.video_infos:
# # #             if vi and vi.get("videoPath"):
# # #                 self.extract_violation_frames(vi["videoPath"])

# # #         report   = self._build_report(processing_time=processing_time)
# # #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# # #         with open(out_path, "w", encoding="utf-8") as f:
# # #             json.dump(report, f, indent=2)

# # #         print(f"[ViolationStore] JSON report     : {out_path}")
# # #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# # #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# # #         # Upload results to S3 and update DB result_s3_path
# # #         try:
# # #             from utils.db_s3_uploader import finalize_and_upload
# # #             finalize_and_upload(
# # #                 report_path     = out_path,
# # #                 analysis_id     = self.analysis_id,
# # #                 train_detail_id = self.train_detail_id,
# # #             )
# # #         except Exception as exc:
# # #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# # #         return out_path

# # # # # from __future__ import annotations

# # # # # import json
# # # # # import os
# # # # # from dataclasses import dataclass
# # # # # from typing import Any, Dict, List, Optional

# # # # # import cv2
# # # # # import numpy as np

# # # # # OUTPUTS_ROOT = "outputs"
# # # # # MERGE_WINDOW = 2.0


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # INTERNAL DATA CLASS
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # @dataclass
# # # # # class _Violation:
# # # # #     timestamp:       float
# # # # #     time_str:        str
# # # # #     frame_index:     int
# # # # #     type:            str
# # # # #     events:          List[str]
# # # # #     severity:        str
# # # # #     duration:        float
# # # # #     risk_score:      int
# # # # #     risk_level:      str
# # # # #     confidence:      float
# # # # #     factors:         List[str]
# # # # #     source_filename: str                  = ""    # e.g. "ch01.mp4"
# # # # #     local_time_str:  str                  = ""    # local time within that file e.g. "00:00:18"
# # # # #     frame_path:      Optional[str]        = None
# # # # #     annotated_frame: Optional[np.ndarray] = None


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # VIOLATION STORE
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # class ViolationStore:
# # # # #     """
# # # # #     Accumulates all violations found across one analysis run (single video
# # # # #     or a multi-video batch that shares the same analysis_id / folder_name).

# # # # #     Usage
# # # # #     ─────
# # # # #     1. Construct once per analysis run.
# # # # #     2. Call record_violation() from the pipeline for every distraction event.
# # # # #     3. Call finalize() at the very end — it deduplicates, merges, saves frames,
# # # # #        writes the JSON report, then calls db_s3_uploader.finalize_and_upload()
# # # # #        to push everything to S3 and record results in the DB.
# # # # #     """

# # # # #     def __init__(
# # # # #         self,
# # # # #         analysis_id:     str,
# # # # #         train_detail_id: int,
# # # # #         video_info:      Optional[Dict[str, Any]] = None,
# # # # #     ):
# # # # #         self.analysis_id     = analysis_id
# # # # #         self.train_detail_id = train_detail_id
# # # # #         # video_infos is always a list:
# # # # #         #   • 1 entry for single-video runs
# # # # #         #   • N entries for batch runs (add_video_info called per video)
# # # # #         self.video_infos: List[Dict[str, Any]] = (
# # # # #             [video_info] if video_info is not None else []
# # # # #         )

# # # # #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# # # # #         self.frames_dir = os.path.join(self.output_dir, "frames")
# # # # #         os.makedirs(self.frames_dir, exist_ok=True)

# # # # #         self._violations:  List[_Violation] = []
# # # # #         self._seen_frames: set              = set()
# # # # #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# # # # #     # ── Public helpers ────────────────────────────────────────────────────────

# # # # #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# # # # #         """
# # # # #         Register one video's metadata into the shared store.
# # # # #         Called once per video in batch mode.
# # # # #         """
# # # # #         self.video_infos.append(video_info)

# # # # #     def record_violation(
# # # # #         self,
# # # # #         annotated_frame:  np.ndarray,
# # # # #         video_time:       float,
# # # # #         frame_index:      int,
# # # # #         event_type:       str,
# # # # #         original_frame:   Optional[np.ndarray] = None,
# # # # #         severity:         str   = "CRITICAL",
# # # # #         confidence:       float = 0.9,
# # # # #         risk_score:       int   = 80,
# # # # #         risk_level:       str   = "CRITICAL",
# # # # #         factors:          Optional[List[str]] = None,
# # # # #         duration:         float = 0.0,
# # # # #         source_filename:  str   = "",    # original filename e.g. "ch01.mp4"
# # # # #         local_video_time: float = -1.0,  # video_time before global offset; -1 = same as video_time
# # # # #     ) -> None:
# # # # #         """
# # # # #         Record one distraction event.

# # # # #         Deduplication key is (frame_index, event_type) so that:
# # # # #           • the same violation type on the same frame is recorded only once
# # # # #           • different violation types on the same frame are each recorded
# # # # #           • frame numbers from different videos never collide because
# # # # #             main.py applies a frame_offset to make every global frame_index
# # # # #             unique across the batch
# # # # #         """
# # # # #         dedup_key = (frame_index, event_type)
# # # # #         if dedup_key in self._seen_frames:
# # # # #             return
# # # # #         self._seen_frames.add(dedup_key)

# # # # #         factors  = factors or []
# # # # #         t        = int(round(video_time))
# # # # #         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

# # # # #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# # # # #         local_str = (
# # # # #             f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
# # # # #         )

# # # # #         self._violations.append(
# # # # #             _Violation(
# # # # #                 timestamp        = video_time,
# # # # #                 time_str         = time_str,
# # # # #                 frame_index      = frame_index,
# # # # #                 type             = event_type,
# # # # #                 events           = [event_type],
# # # # #                 severity         = severity,
# # # # #                 duration         = round(duration, 2),
# # # # #                 risk_score       = risk_score,
# # # # #                 risk_level       = risk_level,
# # # # #                 confidence       = round(confidence, 3),
# # # # #                 factors          = list(factors),
# # # # #                 source_filename  = source_filename,
# # # # #                 local_time_str   = local_str,
# # # # #                 annotated_frame  = (
# # # # #                     annotated_frame.copy() if annotated_frame is not None else None
# # # # #                 ),
# # # # #             )
# # # # #         )

# # # # #     # ── Finalize ──────────────────────────────────────────────────────────────

# # # # #     def finalize(self, processing_time: float = 0.0) -> str:
# # # # #         """
# # # # #         1. Deduplicate violations that share the same frame.
# # # # #         2. Merge violations that are within MERGE_WINDOW seconds of each other.
# # # # #         3. Save annotated frame images to disk (falls back to re-reading from
# # # # #            the source video when no annotated frame was captured).
# # # # #         4. Write analysis_report.json.
# # # # #         5. Upload frames + report to S3 and record in DB via db_s3_uploader.
# # # # #         Returns the local path to analysis_report.json.
# # # # #         """
# # # # #         self._deduplicate_by_frame()
# # # # #         self._merge_by_time_window()

# # # # #         # Extract frames from every video in the batch (or the single video)
# # # # #         for vi in self.video_infos:
# # # # #             if vi and vi.get("videoPath"):
# # # # #                 self.extract_violation_frames(vi["videoPath"])

# # # # #         report   = self._build_report(processing_time=processing_time)
# # # # #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# # # # #         with open(out_path, "w", encoding="utf-8") as f:
# # # # #             json.dump(report, f, indent=2)

# # # # #         print(f"[ViolationStore] JSON report     : {out_path}")
# # # # #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# # # # #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# # # # #         try:
# # # # #             from utils.db_s3_uploader import finalize_and_upload
# # # # #             finalize_and_upload(
# # # # #                 report_path     = out_path,
# # # # #                 analysis_id     = self.analysis_id,
# # # # #                 train_detail_id = self.train_detail_id,
# # # # #             )
# # # # #         except Exception as exc:
# # # # #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# # # # #         return out_path

# # # # #     # ── Private — deduplication & merging ────────────────────────────────────

# # # # #     def _deduplicate_by_frame(self) -> None:
# # # # #         unique: Dict[int, _Violation] = {}
# # # # #         for v in self._violations:
# # # # #             if v.frame_index not in unique:
# # # # #                 unique[v.frame_index] = v
# # # # #             else:
# # # # #                 ex = unique[v.frame_index]
# # # # #                 ex.events  = list(set(ex.events  + v.events))
# # # # #                 ex.factors = list(set(ex.factors + v.factors))
# # # # #                 if v.risk_score > ex.risk_score:
# # # # #                     ex.risk_score = v.risk_score
# # # # #                     ex.risk_level = v.risk_level
# # # # #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# # # # #                     ex.annotated_frame = v.annotated_frame
# # # # #         self._violations = list(unique.values())

# # # # #     def _merge_by_time_window(self) -> None:
# # # # #         if not self._violations:
# # # # #             return
# # # # #         self._violations.sort(key=lambda x: x.timestamp)
# # # # #         merged: List[_Violation] = []
# # # # #         group  = [self._violations[0]]
# # # # #         for v in self._violations[1:]:
# # # # #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# # # # #                 group.append(v)
# # # # #             else:
# # # # #                 merged.append(self._merge_group(group))
# # # # #                 group = [v]
# # # # #         merged.append(self._merge_group(group))
# # # # #         self._violations = merged

# # # # #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# # # # #         base            = group[0]
# # # # #         events: List[str]  = []
# # # # #         factors: List[str] = []
# # # # #         max_risk   = base.risk_score
# # # # #         risk_level = base.risk_level
# # # # #         best_frame = base.annotated_frame
# # # # #         for v in group:
# # # # #             events.extend(v.events)
# # # # #             factors.extend(v.factors)
# # # # #             if v.risk_score > max_risk:
# # # # #                 max_risk, risk_level = v.risk_score, v.risk_level
# # # # #             if best_frame is None and v.annotated_frame is not None:
# # # # #                 best_frame = v.annotated_frame
# # # # #         return _Violation(
# # # # #             timestamp        = base.timestamp,
# # # # #             time_str         = base.time_str,
# # # # #             frame_index      = base.frame_index,
# # # # #             type             = base.type,
# # # # #             events           = list(set(events)),
# # # # #             severity         = base.severity,
# # # # #             duration         = base.duration,
# # # # #             risk_score       = max_risk,
# # # # #             risk_level       = risk_level,
# # # # #             confidence       = base.confidence,
# # # # #             factors          = list(set(factors)),
# # # # #             source_filename  = base.source_filename,
# # # # #             local_time_str   = base.local_time_str,
# # # # #             annotated_frame  = best_frame,
# # # # #         )

# # # # #     # ── Private — frame extraction & saving ──────────────────────────────────

# # # # #     def extract_violation_frames(self, video_path: str) -> None:
# # # # #         print("[ViolationStore] Saving frames...")
# # # # #         need_video = [v for v in self._violations if v.annotated_frame is None]
# # # # #         saved = 0

# # # # #         # First pass: save violations that already have an annotated frame
# # # # #         for v in self._violations:
# # # # #             if v.annotated_frame is not None:
# # # # #                 v.frame_path      = self._save_frame(v.annotated_frame, v.events, v.time_str)
# # # # #                 v.annotated_frame = None   # free memory
# # # # #                 saved += 1

# # # # #         # Second pass: re-read from the source video for any that are missing
# # # # #         if need_video:
# # # # #             cap = cv2.VideoCapture(video_path)
# # # # #             if not cap.isOpened():
# # # # #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# # # # #             else:
# # # # #                 seen: set = set()
# # # # #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# # # # #                     if v.frame_index in seen:
# # # # #                         continue
# # # # #                     seen.add(v.frame_index)
# # # # #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# # # # #                     ret, frame = cap.read()
# # # # #                     if not ret:
# # # # #                         continue
# # # # #                     v.frame_path = self._save_frame(frame, v.events, v.time_str)
# # # # #                     saved += 1
# # # # #                 cap.release()

# # # # #         print(f"[ViolationStore] {saved} frames saved")

# # # # #     def _save_frame(
# # # # #         self,
# # # # #         frame:    np.ndarray,
# # # # #         events:   List[str],
# # # # #         time_str: str,
# # # # #     ) -> str:
# # # # #         distraction   = "_".join(events)
# # # # #         filename_time = time_str.replace(":", "-")
# # # # #         filename      = f"{distraction}_{filename_time}.jpg"
# # # # #         path          = os.path.join(self.frames_dir, filename)
# # # # #         ok = cv2.imwrite(
# # # # #             path,
# # # # #             cv2.resize(frame, (640, 360)),
# # # # #             [cv2.IMWRITE_JPEG_QUALITY, 85],
# # # # #         )
# # # # #         if not ok:
# # # # #             print(f"[ViolationStore] imwrite failed: {path}")
# # # # #         return os.path.join(self.analysis_id, "frames", filename)

# # # # #     # ── Private — report builder ──────────────────────────────────────────────

# # # # #     def _build_report(self, processing_time: float = 0.0) -> dict:
# # # # #         return {
# # # # #             "analysis_id":     self.analysis_id,
# # # # #             "train_detail_id": self.train_detail_id,
# # # # #             "processing_time": round(processing_time, 3),
# # # # #             # Single video → keep as dict for backwards compat; batch → list
# # # # #             "video_info": (
# # # # #                 self.video_infos[0]
# # # # #                 if len(self.video_infos) == 1
# # # # #                 else self.video_infos
# # # # #             ),
# # # # #             "violations": [
# # # # #                 {
# # # # #                     "timestamp":   v.time_str,
# # # # #                     "frame_index": v.frame_index,
# # # # #                     "events":      v.events,
# # # # #                     "severity":    v.severity,
# # # # #                     "duration":    v.duration,
# # # # #                     "risk_score":  v.risk_score,
# # # # #                     "risk_level":  v.risk_level,
# # # # #                     "confidence":  v.confidence,
# # # # #                     "factors":     v.factors,
# # # # #                     "original_video_timestamp": (
# # # # #                         f"{v.source_filename} {v.local_time_str}"
# # # # #                     ),
# # # # #                     "frame_path":  v.frame_path,
# # # # #                 }
# # # # #                 for v in self._violations
# # # # #             ],
# # # # #         }

# # # # from __future__ import annotations

# # # # import json
# # # # import os
# # # # from dataclasses import dataclass
# # # # from typing import Any, Dict, List, Optional

# # # # import cv2
# # # # import numpy as np

# # # # OUTPUTS_ROOT = "outputs"
# # # # MERGE_WINDOW = 2.0


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # INTERNAL DATA CLASS
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # @dataclass
# # # # class _Violation:
# # # #     timestamp:       float
# # # #     time_str:        str
# # # #     frame_index:     int
# # # #     type:            str
# # # #     events:          List[str]
# # # #     severity:        str
# # # #     duration:        float
# # # #     risk_score:      int
# # # #     risk_level:      str
# # # #     confidence:      float
# # # #     factors:         List[str]
# # # #     source_filename: str                  = ""    # e.g. "ch01.mp4"
# # # #     local_time_str:  str                  = ""    # local time within that file e.g. "00:00:18"
# # # #     frame_path:      Optional[str]        = None
# # # #     annotated_frame: Optional[np.ndarray] = None


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # VIOLATION STORE
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # class ViolationStore:
# # # #     """
# # # #     Accumulates all violations found across one analysis run (single video
# # # #     or a multi-video batch that shares the same analysis_id / folder_name).

# # # #     Usage
# # # #     ─────
# # # #     1. Construct once per analysis run.
# # # #     2. Call record_violation() from the pipeline for every distraction event.
# # # #     3. Call finalize() at the very end — it deduplicates, merges, saves frames,
# # # #        writes the JSON report, then calls db_s3_uploader.finalize_and_upload()
# # # #        to push everything to S3 and record results in the DB.
# # # #     """

# # # #     def __init__(
# # # #         self,
# # # #         analysis_id:     str,
# # # #         train_detail_id: int,
# # # #         video_info:      Optional[Dict[str, Any]] = None,
# # # #     ):
# # # #         self.analysis_id     = analysis_id
# # # #         self.train_detail_id = train_detail_id
# # # #         # video_infos is always a list:
# # # #         #   • 1 entry for single-video runs
# # # #         #   • N entries for batch runs (add_video_info called per video)
# # # #         self.video_infos: List[Dict[str, Any]] = (
# # # #             [video_info] if video_info is not None else []
# # # #         )

# # # #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# # # #         self.frames_dir = os.path.join(self.output_dir, "frames")
# # # #         os.makedirs(self.frames_dir, exist_ok=True)

# # # #         self._violations:  List[_Violation] = []
# # # #         self._seen_frames: set              = set()
# # # #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# # # #     # ── Public helpers ────────────────────────────────────────────────────────

# # # #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# # # #         """
# # # #         Register one video's metadata into the shared store.
# # # #         Called once per video in batch mode.
# # # #         """
# # # #         self.video_infos.append(video_info)

# # # #     def record_violation(
# # # #         self,
# # # #         annotated_frame:  np.ndarray,
# # # #         video_time:       float,
# # # #         frame_index:      int,
# # # #         event_type:       str,
# # # #         original_frame:   Optional[np.ndarray] = None,
# # # #         severity:         str   = "CRITICAL",
# # # #         confidence:       float = 0.9,
# # # #         risk_score:       int   = 80,
# # # #         risk_level:       str   = "CRITICAL",
# # # #         factors:          Optional[List[str]] = None,
# # # #         duration:         float = 0.0,
# # # #         source_filename:  str   = "",    # original filename e.g. "ch01.mp4"
# # # #         local_video_time: float = -1.0,  # video_time before global offset; -1 = same as video_time
# # # #     ) -> None:
# # # #         """
# # # #         Record one distraction event.

# # # #         Deduplication key is (frame_index, event_type) so that:
# # # #           • the same violation type on the same frame is recorded only once
# # # #           • different violation types on the same frame are each recorded
# # # #           • frame numbers from different videos never collide because
# # # #             main.py applies a frame_offset to make every global frame_index
# # # #             unique across the batch
# # # #         """
# # # #         dedup_key = (frame_index, event_type)
# # # #         if dedup_key in self._seen_frames:
# # # #             return
# # # #         self._seen_frames.add(dedup_key)

# # # #         factors  = factors or []
# # # #         t        = int(round(video_time))
# # # #         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

# # # #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# # # #         local_str = (
# # # #             f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
# # # #         )

# # # #         self._violations.append(
# # # #             _Violation(
# # # #                 timestamp        = video_time,
# # # #                 time_str         = time_str,
# # # #                 frame_index      = frame_index,
# # # #                 type             = event_type,
# # # #                 events           = [event_type],
# # # #                 severity         = severity,
# # # #                 duration         = round(duration, 2),
# # # #                 risk_score       = risk_score,
# # # #                 risk_level       = risk_level,
# # # #                 confidence       = round(confidence, 3),
# # # #                 factors          = list(factors),
# # # #                 source_filename  = source_filename,
# # # #                 local_time_str   = local_str,
# # # #                 annotated_frame  = (
# # # #                     annotated_frame.copy() if annotated_frame is not None else None
# # # #                 ),
# # # #             )
# # # #         )

# # # #     # ── Finalize ──────────────────────────────────────────────────────────────

# # # #     def finalize(self, processing_time: float = 0.0) -> str:
# # # #         """
# # # #         1. Deduplicate violations that share the same frame.
# # # #         2. Merge violations that are within MERGE_WINDOW seconds of each other.
# # # #         3. Save annotated frame images to disk (falls back to re-reading from
# # # #            the source video when no annotated frame was captured).
# # # #         4. Write analysis_report.json.
# # # #         5. Upload frames + report to S3 and record in DB via db_s3_uploader.
# # # #         Returns the local path to analysis_report.json.
# # # #         """
# # # #         self._deduplicate_by_frame()
# # # #         self._merge_by_time_window()

# # # #         # Extract frames from every video in the batch (or the single video)
# # # #         for vi in self.video_infos:
# # # #             if vi and vi.get("videoPath"):
# # # #                 self.extract_violation_frames(vi["videoPath"])

# # # #         report   = self._build_report(processing_time=processing_time)
# # # #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# # # #         with open(out_path, "w", encoding="utf-8") as f:
# # # #             json.dump(report, f, indent=2)

# # # #         print(f"[ViolationStore] JSON report     : {out_path}")
# # # #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# # # #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# # # #         try:
# # # #             from utils.db_s3_uploader import finalize_and_upload
# # # #             finalize_and_upload(
# # # #                 report_path     = out_path,
# # # #                 analysis_id     = self.analysis_id,
# # # #                 train_detail_id = self.train_detail_id,
# # # #             )
# # # #         except Exception as exc:
# # # #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# # # #         return out_path

# # # #     # ── Private — deduplication & merging ────────────────────────────────────

# # # #     def _deduplicate_by_frame(self) -> None:
# # # #         unique: Dict[int, _Violation] = {}
# # # #         for v in self._violations:
# # # #             if v.frame_index not in unique:
# # # #                 unique[v.frame_index] = v
# # # #             else:
# # # #                 ex = unique[v.frame_index]
# # # #                 ex.events  = list(set(ex.events  + v.events))
# # # #                 ex.factors = list(set(ex.factors + v.factors))
# # # #                 if v.risk_score > ex.risk_score:
# # # #                     ex.risk_score = v.risk_score
# # # #                     ex.risk_level = v.risk_level
# # # #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# # # #                     ex.annotated_frame = v.annotated_frame
# # # #         self._violations = list(unique.values())

# # # #     def _merge_by_time_window(self) -> None:
# # # #         if not self._violations:
# # # #             return
# # # #         self._violations.sort(key=lambda x: x.timestamp)
# # # #         merged: List[_Violation] = []
# # # #         group  = [self._violations[0]]
# # # #         for v in self._violations[1:]:
# # # #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# # # #                 group.append(v)
# # # #             else:
# # # #                 merged.append(self._merge_group(group))
# # # #                 group = [v]
# # # #         merged.append(self._merge_group(group))
# # # #         self._violations = merged

# # # #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# # # #         base             = group[0]
# # # #         events: List[str]  = []
# # # #         factors: List[str] = []
# # # #         max_risk   = base.risk_score
# # # #         risk_level = base.risk_level
# # # #         best_frame = base.annotated_frame
# # # #         for v in group:
# # # #             events.extend(v.events)
# # # #             factors.extend(v.factors)
# # # #             if v.risk_score > max_risk:
# # # #                 max_risk, risk_level = v.risk_score, v.risk_level
# # # #             if best_frame is None and v.annotated_frame is not None:
# # # #                 best_frame = v.annotated_frame
# # # #         return _Violation(
# # # #             timestamp        = base.timestamp,
# # # #             time_str         = base.time_str,
# # # #             frame_index      = base.frame_index,
# # # #             type             = base.type,
# # # #             events           = list(set(events)),
# # # #             severity         = base.severity,
# # # #             duration         = base.duration,
# # # #             risk_score       = max_risk,
# # # #             risk_level       = risk_level,
# # # #             confidence       = base.confidence,
# # # #             factors          = list(set(factors)),
# # # #             source_filename  = base.source_filename,
# # # #             local_time_str   = base.local_time_str,
# # # #             annotated_frame  = best_frame,
# # # #         )

# # # #     # ── Private — frame extraction & saving ──────────────────────────────────

# # # #     def extract_violation_frames(self, video_path: str) -> None:
# # # #         print("[ViolationStore] Saving frames...")
# # # #         need_video = [v for v in self._violations if v.annotated_frame is None]
# # # #         saved = 0

# # # #         # First pass: save violations that already have an annotated frame
# # # #         for v in self._violations:
# # # #             if v.annotated_frame is not None:
# # # #                 v.frame_path      = self._save_frame(
# # # #                     v.annotated_frame, v.events, v.time_str, v.frame_index
# # # #                 )
# # # #                 v.annotated_frame = None   # free memory
# # # #                 saved += 1

# # # #         # Second pass: re-read from the source video for any that are missing
# # # #         if need_video:
# # # #             cap = cv2.VideoCapture(video_path)
# # # #             if not cap.isOpened():
# # # #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# # # #             else:
# # # #                 seen: set = set()
# # # #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# # # #                     if v.frame_index in seen:
# # # #                         continue
# # # #                     seen.add(v.frame_index)
# # # #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# # # #                     ret, frame = cap.read()
# # # #                     if not ret:
# # # #                         continue
# # # #                     v.frame_path = self._save_frame(
# # # #                         frame, v.events, v.time_str, v.frame_index
# # # #                     )
# # # #                     saved += 1
# # # #                 cap.release()

# # # #         print(f"[ViolationStore] {saved} frames saved")

# # # #     def _save_frame(
# # # #         self,
# # # #         frame:       np.ndarray,
# # # #         events:      List[str],
# # # #         time_str:    str,
# # # #         frame_index: int,
# # # #     ) -> str:
# # # #         """
# # # #         Save a single violation frame as a JPEG.

# # # #         Filename format:  <events>_<HH-MM-SS>_f<frame_index>.jpg
# # # #         Example:          seat_absence_00-01-14_f2106.jpg
# # # #                           seat_absence_drowsy_00-03-02_f5058.jpg
# # # #         """
# # # #         distraction   = "_".join(sorted(events))   # sorted for deterministic name
# # # #         filename_time = time_str.replace(":", "-")
# # # #         filename      = f"{distraction}_{filename_time}_f{frame_index}.jpg"
# # # #         path          = os.path.join(self.frames_dir, filename)
# # # #         ok = cv2.imwrite(
# # # #             path,
# # # #             cv2.resize(frame, (640, 360)),
# # # #             [cv2.IMWRITE_JPEG_QUALITY, 85],
# # # #         )
# # # #         if not ok:
# # # #             print(f"[ViolationStore] imwrite failed: {path}")
# # # #         return os.path.join(self.analysis_id, "frames", filename)

# # # #     # ── Private — report builder ──────────────────────────────────────────────

# # # #     def _build_report(self, processing_time: float = 0.0) -> dict:
# # # #         return {
# # # #             "analysis_id":     self.analysis_id,
# # # #             "train_detail_id": self.train_detail_id,
# # # #             "processing_time": round(processing_time, 3),
# # # #             # Single video → keep as dict for backwards compat; batch → list
# # # #             "video_info": (
# # # #                 self.video_infos[0]
# # # #                 if len(self.video_infos) == 1
# # # #                 else self.video_infos
# # # #             ),
# # # #             "violations": [
# # # #                 {
# # # #                     "timestamp":   v.time_str,
# # # #                     "frame_index": v.frame_index,
# # # #                     "events":      v.events,
# # # #                     "severity":    v.severity,
# # # #                     "duration":    v.duration,
# # # #                     "risk_score":  v.risk_score,
# # # #                     "risk_level":  v.risk_level,
# # # #                     "confidence":  v.confidence,
# # # #                     "factors":     v.factors,
# # # #                     "original_video_timestamp": (
# # # #                         f"{v.source_filename} {v.local_time_str}"
# # # #                     ),
# # # #                     "frame_path":  v.frame_path,
# # # #                 }
# # # #                 for v in self._violations
# # # #             ],
# # # #         }


# # # from __future__ import annotations

# # # import json
# # # import os
# # # from dataclasses import dataclass
# # # from typing import Any, Dict, List, Optional

# # # import cv2
# # # import numpy as np

# # # OUTPUTS_ROOT = "outputs"
# # # MERGE_WINDOW = 2.0


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # # INTERNAL DATA CLASS
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # @dataclass
# # # class _Violation:
# # #     timestamp:       float
# # #     time_str:        str
# # #     frame_index:     int
# # #     type:            str
# # #     events:          List[str]
# # #     severity:        str
# # #     duration:        float
# # #     risk_score:      int
# # #     risk_level:      str
# # #     confidence:      float
# # #     factors:         List[str]
# # #     source_filename: str                  = ""    # DB filename e.g. "mobile.mp4"
# # #     local_time_str:  str                  = ""    # local time within that file e.g. "00:00:23"
# # #     frame_path:      Optional[str]        = None
# # #     annotated_frame: Optional[np.ndarray] = None


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # # VIOLATION STORE
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # class ViolationStore:
# # #     """
# # #     Accumulates all violations found across one analysis run (single video
# # #     or a multi-video batch that shares the same analysis_id / folder_name).

# # #     Batch mode usage (api.py)
# # #     ─────────────────────────
# # #     1. Construct ONCE for the whole folder (no video_info in __init__).
# # #     2. Pass as shared_vstore= to each GadgetDetectionPipeline.
# # #        The pipeline calls add_video_info() automatically.
# # #     3. Call finalize() ONCE after all videos in the folder are done.

# # #     Standalone mode usage (CLI / single video)
# # #     ──────────────────────────────────────────
# # #     1. Construct with video_info= for the single video.
# # #     2. Pipeline calls finalize() automatically at the end of run().
# # #     """

# # #     def __init__(
# # #         self,
# # #         analysis_id:     str,
# # #         train_detail_id: int,
# # #         video_info:      Optional[Dict[str, Any]] = None,
# # #     ):
# # #         self.analysis_id     = analysis_id
# # #         self.train_detail_id = train_detail_id
# # #         # Always a list — 0 entries until add_video_info() is called (batch mode),
# # #         # or 1 entry when video_info is provided (standalone mode).
# # #         self.video_infos: List[Dict[str, Any]] = (
# # #             [video_info] if video_info is not None else []
# # #         )

# # #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# # #         self.frames_dir = os.path.join(self.output_dir, "frames")
# # #         os.makedirs(self.frames_dir, exist_ok=True)

# # #         self._violations:  List[_Violation] = []
# # #         self._seen_frames: set              = set()
# # #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# # #     # ── Public helpers ────────────────────────────────────────────────────────

# # #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# # #         """Append one video's metadata. Called once per video in batch mode."""
# # #         self.video_infos.append(video_info)

# # #     def record_violation(
# # #         self,
# # #         annotated_frame:  np.ndarray,
# # #         video_time:       float,          # global timestamp (offset-adjusted)
# # #         frame_index:      int,            # global frame index (offset-adjusted)
# # #         event_type:       str,
# # #         original_frame:   Optional[np.ndarray] = None,
# # #         severity:         str   = "CRITICAL",
# # #         confidence:       float = 0.9,
# # #         risk_score:       int   = 80,
# # #         risk_level:       str   = "CRITICAL",
# # #         factors:          Optional[List[str]] = None,
# # #         duration:         float = 0.0,
# # #         source_filename:  str   = "",     # DB filename shown in original_video_timestamp
# # #         local_video_time: float = -1.0,   # local time within the source file; -1 = same as video_time
# # #     ) -> None:
# # #         """
# # #         Record one distraction event.

# # #         Deduplication key is (frame_index, event_type):
# # #           • same event on the same global frame is recorded once
# # #           • different events on the same global frame are each recorded
# # #           • global frame_index is unique across videos (frame_offset applied in main.py)
# # #         """
# # #         dedup_key = (frame_index, event_type)
# # #         if dedup_key in self._seen_frames:
# # #             return
# # #         self._seen_frames.add(dedup_key)

# # #         factors  = factors or []
# # #         t        = int(round(video_time))
# # #         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

# # #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# # #         local_str = (
# # #             f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
# # #         )

# # #         self._violations.append(
# # #             _Violation(
# # #                 timestamp        = video_time,
# # #                 time_str         = time_str,
# # #                 frame_index      = frame_index,
# # #                 type             = event_type,
# # #                 events           = [event_type],
# # #                 severity         = severity,
# # #                 duration         = round(duration, 2),
# # #                 risk_score       = risk_score,
# # #                 risk_level       = risk_level,
# # #                 confidence       = round(confidence, 3),
# # #                 factors          = list(factors),
# # #                 source_filename  = source_filename,
# # #                 local_time_str   = local_str,
# # #                 annotated_frame  = (
# # #                     annotated_frame.copy() if annotated_frame is not None else None
# # #                 ),
# # #             )
# # #         )

# # #     # ── Finalize ──────────────────────────────────────────────────────────────

# # #     def finalize(self, processing_time: float = 0.0) -> str:
# # #         """
# # #         Deduplicate → merge → save frames → write JSON → upload to S3/DB.
# # #         Returns the local path to analysis_report.json.
# # #         """
# # #         self._deduplicate_by_frame()
# # #         self._merge_by_time_window()

# # #         # Extract frames from every video in the batch (or the single video)
# # #         for vi in self.video_infos:
# # #             if vi and vi.get("videoPath"):
# # #                 self.extract_violation_frames(vi["videoPath"])

# # #         report   = self._build_report(processing_time=processing_time)
# # #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# # #         with open(out_path, "w", encoding="utf-8") as f:
# # #             json.dump(report, f, indent=2)

# # #         print(f"[ViolationStore] JSON report     : {out_path}")
# # #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# # #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# # #         try:
# # #             from utils.db_s3_uploader import finalize_and_upload
# # #             finalize_and_upload(
# # #                 report_path     = out_path,
# # #                 analysis_id     = self.analysis_id,
# # #                 train_detail_id = self.train_detail_id,
# # #             )
# # #         except Exception as exc:
# # #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# # #         return out_path

# # #     # ── Private — deduplication & merging ────────────────────────────────────

# # #     def _deduplicate_by_frame(self) -> None:
# # #         unique: Dict[int, _Violation] = {}
# # #         for v in self._violations:
# # #             if v.frame_index not in unique:
# # #                 unique[v.frame_index] = v
# # #             else:
# # #                 ex = unique[v.frame_index]
# # #                 ex.events  = list(set(ex.events  + v.events))
# # #                 ex.factors = list(set(ex.factors + v.factors))
# # #                 if v.risk_score > ex.risk_score:
# # #                     ex.risk_score = v.risk_score
# # #                     ex.risk_level = v.risk_level
# # #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# # #                     ex.annotated_frame = v.annotated_frame
# # #         self._violations = list(unique.values())

# # #     def _merge_by_time_window(self) -> None:
# # #         if not self._violations:
# # #             return
# # #         self._violations.sort(key=lambda x: x.timestamp)
# # #         merged: List[_Violation] = []
# # #         group  = [self._violations[0]]
# # #         for v in self._violations[1:]:
# # #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# # #                 group.append(v)
# # #             else:
# # #                 merged.append(self._merge_group(group))
# # #                 group = [v]
# # #         merged.append(self._merge_group(group))
# # #         self._violations = merged

# # #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# # #         base             = group[0]
# # #         events: List[str]  = []
# # #         factors: List[str] = []
# # #         max_risk   = base.risk_score
# # #         risk_level = base.risk_level
# # #         best_frame = base.annotated_frame
# # #         for v in group:
# # #             events.extend(v.events)
# # #             factors.extend(v.factors)
# # #             if v.risk_score > max_risk:
# # #                 max_risk, risk_level = v.risk_score, v.risk_level
# # #             if best_frame is None and v.annotated_frame is not None:
# # #                 best_frame = v.annotated_frame
# # #         return _Violation(
# # #             timestamp        = base.timestamp,
# # #             time_str         = base.time_str,
# # #             frame_index      = base.frame_index,
# # #             type             = base.type,
# # #             events           = list(set(events)),
# # #             severity         = base.severity,
# # #             duration         = base.duration,
# # #             risk_score       = max_risk,
# # #             risk_level       = risk_level,
# # #             confidence       = base.confidence,
# # #             factors          = list(set(factors)),
# # #             source_filename  = base.source_filename,
# # #             local_time_str   = base.local_time_str,
# # #             annotated_frame  = best_frame,
# # #         )

# # #     # ── Private — frame extraction & saving ──────────────────────────────────

# # #     def extract_violation_frames(self, video_path: str) -> None:
# # #         print("[ViolationStore] Saving frames...")
# # #         need_video = [v for v in self._violations if v.annotated_frame is None]
# # #         saved = 0

# # #         # First pass: save violations that already have an annotated frame
# # #         for v in self._violations:
# # #             if v.annotated_frame is not None:
# # #                 v.frame_path      = self._save_frame(v.annotated_frame, v.events, v.time_str)
# # #                 v.annotated_frame = None   # free memory
# # #                 saved += 1

# # #         # Second pass: re-read from the source video for any that are missing
# # #         if need_video:
# # #             cap = cv2.VideoCapture(video_path)
# # #             if not cap.isOpened():
# # #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# # #             else:
# # #                 seen: set = set()
# # #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# # #                     if v.frame_index in seen:
# # #                         continue
# # #                     seen.add(v.frame_index)
# # #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# # #                     ret, frame = cap.read()
# # #                     if not ret:
# # #                         continue
# # #                     v.frame_path = self._save_frame(frame, v.events, v.time_str)
# # #                     saved += 1
# # #                 cap.release()

# # #         print(f"[ViolationStore] {saved} frames saved")

# # #     def _save_frame(
# # #         self,
# # #         frame:    np.ndarray,
# # #         events:   List[str],
# # #         time_str: str,
# # #     ) -> str:
# # #         """
# # #         Save a single violation frame as JPEG.

# # #         Filename format:  <events>_<HH-MM-SS>.jpg
# # #         Example:          seat_absence_00-01-14.jpg
# # #                           seat_absence_drowsy_00-03-02.jpg
# # #                           phone_use_00-00-24.jpg
# # #         """
# # #         distraction   = "_".join(sorted(events))   # sorted for deterministic name
# # #         filename_time = time_str.replace(":", "-")
# # #         filename      = f"{distraction}_{filename_time}.jpg"
# # #         path          = os.path.join(self.frames_dir, filename)
# # #         ok = cv2.imwrite(
# # #             path,
# # #             cv2.resize(frame, (640, 360)),
# # #             [cv2.IMWRITE_JPEG_QUALITY, 85],
# # #         )
# # #         if not ok:
# # #             print(f"[ViolationStore] imwrite failed: {path}")
# # #         return os.path.join(self.analysis_id, "frames", filename)

# # #     # ── Private — report builder ──────────────────────────────────────────────

# # #     def _build_report(self, processing_time: float = 0.0) -> dict:
# # #         return {
# # #             "analysis_id":     self.analysis_id,
# # #             "train_detail_id": self.train_detail_id,
# # #             "processing_time": round(processing_time, 3),
# # #             # Single video → dict (backwards compat); batch → list
# # #             "video_info": (
# # #                 self.video_infos[0]
# # #                 if len(self.video_infos) == 1
# # #                 else self.video_infos
# # #             ),
# # #             "violations": [
# # #                 {
# # #                     "timestamp":   v.time_str,
# # #                     "frame_index": v.frame_index,
# # #                     "events":      v.events,
# # #                     "severity":    v.severity,
# # #                     "duration":    v.duration,
# # #                     "risk_score":  v.risk_score,
# # #                     "risk_level":  v.risk_level,
# # #                     "confidence":  v.confidence,
# # #                     "factors":     v.factors,
# # #                     # "original_video_timestamp" = "<db_filename> <local_HH:MM:SS>"
# # #                     # When timestamp == original (video 1), local_time_str == time_str
# # #                     "original_video_timestamp": (
# # #                         f"{v.source_filename} {v.local_time_str}"
# # #                     ),
# # #                     "frame_path":  v.frame_path,
# # #                 }
# # #                 for v in self._violations
# # #             ],
# # #         }


# # from __future__ import annotations

# # import json
# # import os
# # from dataclasses import dataclass
# # from typing import Any, Dict, List, Optional

# # import cv2
# # import numpy as np

# # OUTPUTS_ROOT = "outputs"
# # MERGE_WINDOW = 2.0


# # # ══════════════════════════════════════════════════════════════════════════════
# # # INTERNAL DATA CLASS
# # # ══════════════════════════════════════════════════════════════════════════════

# # @dataclass
# # class _Violation:
# #     timestamp:       float
# #     time_str:        str
# #     frame_index:     int
# #     type:            str
# #     events:          List[str]
# #     severity:        str
# #     duration:        float
# #     risk_score:      int
# #     risk_level:      str
# #     confidence:      float
# #     factors:         List[str]
# #     source_filename: str                  = ""    # DB filename e.g. "mobile.mp4"
# #     local_time_str:  str                  = ""    # local time within that file e.g. "00:00:23"
# #     frame_path:      Optional[str]        = None
# #     annotated_frame: Optional[np.ndarray] = None


# # # ══════════════════════════════════════════════════════════════════════════════
# # # VIOLATION STORE
# # # ══════════════════════════════════════════════════════════════════════════════

# # class ViolationStore:
# #     """
# #     Accumulates all violations found across one analysis run (single video
# #     or a multi-video batch that shares the same analysis_id / folder_name).

# #     Batch mode usage (api.py)
# #     ─────────────────────────
# #     1. Construct ONCE for the whole folder (no video_info in __init__).
# #     2. Pass as shared_vstore= to each GadgetDetectionPipeline.
# #        The pipeline calls add_video_info() automatically.
# #     3. Call finalize() ONCE after all videos in the folder are done.

# #     Standalone mode usage (CLI / single video)
# #     ──────────────────────────────────────────
# #     1. Construct with video_info= for the single video.
# #     2. Pipeline calls finalize() automatically at the end of run().
# #     """

# #     def __init__(
# #         self,
# #         analysis_id:     str,
# #         train_detail_id: int,
# #         video_info:      Optional[Dict[str, Any]] = None,
# #     ):
# #         self.analysis_id     = analysis_id
# #         self.train_detail_id = train_detail_id
# #         # Always a list — 0 entries until add_video_info() is called (batch mode),
# #         # or 1 entry when video_info is provided (standalone mode).
# #         self.video_infos: List[Dict[str, Any]] = (
# #             [video_info] if video_info is not None else []
# #         )

# #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# #         self.frames_dir = os.path.join(self.output_dir, "frames")
# #         os.makedirs(self.frames_dir, exist_ok=True)

# #         self._violations:  List[_Violation] = []
# #         self._seen_frames: set              = set()
# #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# #     # ── Public helpers ────────────────────────────────────────────────────────

# #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# #         """Append one video's metadata. Called once per video in batch mode."""
# #         self.video_infos.append(video_info)

# #     def record_violation(
# #         self,
# #         annotated_frame:  np.ndarray,
# #         video_time:       float,          # global timestamp (offset-adjusted)
# #         frame_index:      int,            # global frame index (offset-adjusted)
# #         event_type:       str,
# #         original_frame:   Optional[np.ndarray] = None,
# #         severity:         str   = "CRITICAL",
# #         confidence:       float = 0.9,
# #         risk_score:       int   = 80,
# #         risk_level:       str   = "CRITICAL",
# #         factors:          Optional[List[str]] = None,
# #         duration:         float = 0.0,
# #         source_filename:  str   = "",     # DB filename shown in original_video_timestamp
# #         local_video_time: float = -1.0,   # local time within the source file; -1 = same as video_time
# #     ) -> None:
# #         """
# #         Record one distraction event.

# #         Deduplication key is (frame_index, event_type):
# #           • same event on the same global frame is recorded once
# #           • different events on the same global frame are each recorded
# #           • global frame_index is unique across videos (frame_offset applied in main.py)
# #         """
# #         dedup_key = (frame_index, event_type)
# #         if dedup_key in self._seen_frames:
# #             return
# #         self._seen_frames.add(dedup_key)

# #         factors  = factors or []
# #         t        = int(round(video_time))
# #         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

# #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# #         local_str = (
# #             f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
# #         )

# #         self._violations.append(
# #             _Violation(
# #                 timestamp        = video_time,
# #                 time_str         = time_str,
# #                 frame_index      = frame_index,
# #                 type             = event_type,
# #                 events           = [event_type],
# #                 severity         = severity,
# #                 duration         = round(duration, 2),
# #                 risk_score       = risk_score,
# #                 risk_level       = risk_level,
# #                 confidence       = round(confidence, 3),
# #                 factors          = list(factors),
# #                 source_filename  = source_filename,
# #                 local_time_str   = local_str,
# #                 annotated_frame  = (
# #                     annotated_frame.copy() if annotated_frame is not None else None
# #                 ),
# #             )
# #         )

# #     # ── Finalize ──────────────────────────────────────────────────────────────

# #     def write_report(self, processing_time: float = 0.0) -> str:
# #         """
# #         Build analysis_report.json from the CURRENT in-memory violations and
# #         write it to disk at  outputs/<analysis_id>/analysis_report.json
# #         (i.e. as a SIBLING of the frames/ folder, not inside it).

# #         Unlike finalize(), this does NOT touch S3 or the legacy DB uploader —
# #         it is safe to call from the journey/batch pipeline (analyzer.py),
# #         which has its own separate callback-based completion flow.

# #         Does NOT run dedup/merge/extract_violation_frames — call those first
# #         (analyze_journey() already does, via the shared ViolationStore) if
# #         you need them. Safe to call multiple times; it always overwrites.

# #         Returns the local path to analysis_report.json.
# #         """
# #         report   = self._build_report(processing_time=processing_time)
# #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# #         os.makedirs(self.output_dir, exist_ok=True)
# #         with open(out_path, "w", encoding="utf-8") as f:
# #             json.dump(report, f, indent=2)

# #         print(f"[ViolationStore] JSON report     : {out_path}")
# #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# #         return out_path

# #     def finalize(self, processing_time: float = 0.0) -> str:
# #         """
# #         Deduplicate → merge → save frames → write JSON → upload to S3/DB.
# #         Returns the local path to analysis_report.json.

# #         Standalone/CLI mode only. Journey/batch mode (analyzer.py) should call
# #         write_report() directly instead, after its own dedup/merge/extract
# #         steps, since it has its own callback-based upload path and does not
# #         want the legacy db_s3_uploader to also run.
# #         """
# #         self._deduplicate_by_frame()
# #         self._merge_by_time_window()

# #         # Extract frames from every video in the batch (or the single video)
# #         for vi in self.video_infos:
# #             if vi and vi.get("videoPath"):
# #                 self.extract_violation_frames(vi["videoPath"])

# #         out_path = self.write_report(processing_time=processing_time)

# #         try:
# #             from utils.db_s3_uploader import finalize_and_upload
# #             finalize_and_upload(
# #                 report_path     = out_path,
# #                 analysis_id     = self.analysis_id,
# #                 train_detail_id = self.train_detail_id,
# #             )
# #         except Exception as exc:
# #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# #         return out_path

# #     # ── Private — deduplication & merging ────────────────────────────────────

# #     def _deduplicate_by_frame(self) -> None:
# #         unique: Dict[int, _Violation] = {}
# #         for v in self._violations:
# #             if v.frame_index not in unique:
# #                 unique[v.frame_index] = v
# #             else:
# #                 ex = unique[v.frame_index]
# #                 ex.events  = list(set(ex.events  + v.events))
# #                 ex.factors = list(set(ex.factors + v.factors))
# #                 if v.risk_score > ex.risk_score:
# #                     ex.risk_score = v.risk_score
# #                     ex.risk_level = v.risk_level
# #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# #                     ex.annotated_frame = v.annotated_frame
# #         self._violations = list(unique.values())

# #     def _merge_by_time_window(self) -> None:
# #         if not self._violations:
# #             return
# #         self._violations.sort(key=lambda x: x.timestamp)
# #         merged: List[_Violation] = []
# #         group  = [self._violations[0]]
# #         for v in self._violations[1:]:
# #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# #                 group.append(v)
# #             else:
# #                 merged.append(self._merge_group(group))
# #                 group = [v]
# #         merged.append(self._merge_group(group))
# #         self._violations = merged

# #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# #         base             = group[0]
# #         events: List[str]  = []
# #         factors: List[str] = []
# #         max_risk   = base.risk_score
# #         risk_level = base.risk_level
# #         best_frame = base.annotated_frame
# #         for v in group:
# #             events.extend(v.events)
# #             factors.extend(v.factors)
# #             if v.risk_score > max_risk:
# #                 max_risk, risk_level = v.risk_score, v.risk_level
# #             if best_frame is None and v.annotated_frame is not None:
# #                 best_frame = v.annotated_frame
# #         return _Violation(
# #             timestamp        = base.timestamp,
# #             time_str         = base.time_str,
# #             frame_index      = base.frame_index,
# #             type             = base.type,
# #             events           = list(set(events)),
# #             severity         = base.severity,
# #             duration         = base.duration,
# #             risk_score       = max_risk,
# #             risk_level       = risk_level,
# #             confidence       = base.confidence,
# #             factors          = list(set(factors)),
# #             source_filename  = base.source_filename,
# #             local_time_str   = base.local_time_str,
# #             annotated_frame  = best_frame,
# #         )

# #     # ── Private — frame extraction & saving ──────────────────────────────────

# #     def extract_violation_frames(self, video_path: str) -> None:
# #         print("[ViolationStore] Saving frames...")
# #         need_video = [v for v in self._violations if v.annotated_frame is None]
# #         saved = 0

# #         # First pass: save violations that already have an annotated frame
# #         for v in self._violations:
# #             if v.annotated_frame is not None:
# #                 v.frame_path      = self._save_frame(v.annotated_frame, v.events, v.time_str)
# #                 v.annotated_frame = None   # free memory
# #                 saved += 1

# #         # Second pass: re-read from the source video for any that are missing
# #         if need_video:
# #             cap = cv2.VideoCapture(video_path)
# #             if not cap.isOpened():
# #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# #             else:
# #                 seen: set = set()
# #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# #                     if v.frame_index in seen:
# #                         continue
# #                     seen.add(v.frame_index)
# #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# #                     ret, frame = cap.read()
# #                     if not ret:
# #                         continue
# #                     v.frame_path = self._save_frame(frame, v.events, v.time_str)
# #                     saved += 1
# #                 cap.release()

# #         print(f"[ViolationStore] {saved} frames saved")

# #     def _save_frame(
# #         self,
# #         frame:    np.ndarray,
# #         events:   List[str],
# #         time_str: str,
# #     ) -> str:
# #         """
# #         Save a single violation frame as JPEG.

# #         Filename format:  <events>_<HH-MM-SS>.jpg
# #         Example:          seat_absence_00-01-14.jpg
# #                           seat_absence_drowsy_00-03-02.jpg
# #                           phone_use_00-00-24.jpg
# #         """
# #         distraction   = "_".join(sorted(events))   # sorted for deterministic name
# #         filename_time = time_str.replace(":", "-")
# #         filename      = f"{distraction}_{filename_time}.jpg"
# #         path          = os.path.join(self.frames_dir, filename)
# #         ok = cv2.imwrite(
# #             path,
# #             cv2.resize(frame, (640, 360)),
# #             [cv2.IMWRITE_JPEG_QUALITY, 85],
# #         )
# #         if not ok:
# #             print(f"[ViolationStore] imwrite failed: {path}")
# #         return os.path.join(self.analysis_id, "frames", filename)

# #     # ── Private — report builder ──────────────────────────────────────────────

# #     def _build_report(self, processing_time: float = 0.0) -> dict:
# #         return {
# #             "analysis_id":     self.analysis_id,
# #             "train_detail_id": self.train_detail_id,
# #             "processing_time": round(processing_time, 3),
# #             # Single video → dict (backwards compat); batch → list
# #             "video_info": (
# #                 self.video_infos[0]
# #                 if len(self.video_infos) == 1
# #                 else self.video_infos
# #             ),
# #             "violations": [
# #                 {
# #                     "timestamp":   v.time_str,
# #                     "frame_index": v.frame_index,
# #                     "events":      v.events,
# #                     "severity":    v.severity,
# #                     "duration":    v.duration,
# #                     "risk_score":  v.risk_score,
# #                     "risk_level":  v.risk_level,
# #                     "confidence":  v.confidence,
# #                     "factors":     v.factors,
# #                     # "original_video_timestamp" = "<db_filename> <local_HH:MM:SS>"
# #                     # When timestamp == original (video 1), local_time_str == time_str
# #                     "original_video_timestamp": (
# #                         f"{v.source_filename} {v.local_time_str}"
# #                     ),
# #                     "frame_path":  v.frame_path,
# #                 }
# #                 for v in self._violations
# #             ],
# #         }




# # # # from __future__ import annotations

# # # # import json
# # # # import os
# # # # from dataclasses import dataclass
# # # # from typing import Any, Dict, List, Optional

# # # # import cv2
# # # # import numpy as np

# # # # OUTPUTS_ROOT = "outputs"
# # # # MERGE_WINDOW = 2.0


# # # # @dataclass
# # # # class _Violation:
# # # #     timestamp:       float
# # # #     time_str:        str
# # # #     frame_index:     int
# # # #     type:            str
# # # #     events:          List[str]
# # # #     severity:        str
# # # #     duration:        float
# # # #     risk_score:      int
# # # #     risk_level:      str
# # # #     confidence:      float
# # # #     factors:         List[str]
# # # #     source_filename: str                   = ""    # e.g. "ax.mp4"
# # # #     local_time_str:  str                   = ""    # local time within that file e.g. "00:00:18"
# # # #     frame_path:      Optional[str]         = None
# # # #     annotated_frame: Optional[np.ndarray]  = None


# # # # class ViolationStore:

# # # #     def __init__(self, analysis_id: str, train_detail_id: int, video_info: Optional[Dict[str, Any]] = None):
# # # #         self.analysis_id     = analysis_id
# # # #         self.train_detail_id = train_detail_id
# # # #         # video_infos is always a list — 1 entry for single-video, N entries for batch
# # # #         self.video_infos: List[Dict[str, Any]] = [video_info] if video_info is not None else []

# # # #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# # # #         self.frames_dir = os.path.join(self.output_dir, "frames")
# # # #         os.makedirs(self.frames_dir, exist_ok=True)

# # # #         self._violations: List[_Violation] = []
# # # #         self._seen_frames: set             = set()
# # # #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# # # #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# # # #         """
# # # #         Register one video's metadata into the shared store.
# # # #         Called once per video in batch mode (api.py passes shared_vstore).
# # # #         """
# # # #         self.video_infos.append(video_info)

# # # #     def record_violation(
# # # #         self,
# # # #         annotated_frame: np.ndarray,
# # # #         video_time:      float,
# # # #         frame_index:     int,
# # # #         event_type:      str,
# # # #         original_frame:  Optional[np.ndarray] = None,
# # # #         severity:        str   = "CRITICAL",
# # # #         confidence:      float = 0.9,
# # # #         risk_score:      int   = 80,
# # # #         risk_level:      str   = "CRITICAL",
# # # #         factors:         Optional[List[str]] = None,
# # # #         duration:        float = 0.0,
# # # #         source_filename: str   = "",   # original upload filename e.g. "ax.mp4"
# # # #         local_video_time: float = -1.0, # raw video_time before offset; -1 = same as video_time
# # # #     ):
# # # #         # Deduplicate on (frame_index, event_type) so that:
# # # #         #  • the same violation type on the same frame is recorded only once
# # # #         #  • different violation types on the same frame are each recorded
# # # #         #  • frame numbers from different videos never collide (frame_offset
# # # #         #    in main.py makes every global frame_index unique across the batch)
# # # #         dedup_key = (frame_index, event_type)
# # # #         if dedup_key in self._seen_frames:
# # # #             return
# # # #         self._seen_frames.add(dedup_key)
# # # #         factors   = factors or []
# # # #         t         = int(round(video_time))
# # # #         time_str  = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"
# # # #         # Build the per-file local timestamp (time within the source video)
# # # #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# # # #         local_str = f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
# # # #         self._violations.append(_Violation(
# # # #             timestamp        = video_time,
# # # #             time_str         = time_str,
# # # #             frame_index      = frame_index,
# # # #             type             = event_type,
# # # #             events           = [event_type],
# # # #             severity         = severity,
# # # #             duration         = round(duration, 2),
# # # #             risk_score       = risk_score,
# # # #             risk_level       = risk_level,
# # # #             confidence       = round(confidence, 3),
# # # #             factors          = list(factors),
# # # #             source_filename  = source_filename,
# # # #             local_time_str   = local_str,
# # # #             annotated_frame  = annotated_frame.copy() if annotated_frame is not None else None,
# # # #         ))

# # # #     def _deduplicate_by_frame(self):
# # # #         unique: Dict[int, _Violation] = {}
# # # #         for v in self._violations:
# # # #             if v.frame_index not in unique:
# # # #                 unique[v.frame_index] = v
# # # #             else:
# # # #                 ex = unique[v.frame_index]
# # # #                 ex.events  = list(set(ex.events  + v.events))
# # # #                 ex.factors = list(set(ex.factors + v.factors))
# # # #                 if v.risk_score > ex.risk_score:
# # # #                     ex.risk_score = v.risk_score
# # # #                     ex.risk_level = v.risk_level
# # # #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# # # #                     ex.annotated_frame = v.annotated_frame
# # # #         self._violations = list(unique.values())

# # # #     def _merge_by_time_window(self):
# # # #         if not self._violations:
# # # #             return
# # # #         self._violations.sort(key=lambda x: x.timestamp)
# # # #         merged = []
# # # #         group  = [self._violations[0]]
# # # #         for v in self._violations[1:]:
# # # #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# # # #                 group.append(v)
# # # #             else:
# # # #                 merged.append(self._merge_group(group))
# # # #                 group = [v]
# # # #         merged.append(self._merge_group(group))
# # # #         self._violations = merged

# # # #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# # # #         base            = group[0]
# # # #         events, factors = [], []
# # # #         max_risk        = base.risk_score
# # # #         risk_level      = base.risk_level
# # # #         best_frame      = base.annotated_frame
# # # #         for v in group:
# # # #             events.extend(v.events)
# # # #             factors.extend(v.factors)
# # # #             if v.risk_score > max_risk:
# # # #                 max_risk, risk_level = v.risk_score, v.risk_level
# # # #             if best_frame is None and v.annotated_frame is not None:
# # # #                 best_frame = v.annotated_frame
# # # #         return _Violation(
# # # #             timestamp        = base.timestamp,
# # # #             time_str         = base.time_str,
# # # #             frame_index      = base.frame_index,
# # # #             type             = base.type,
# # # #             events           = list(set(events)),
# # # #             severity         = base.severity,
# # # #             duration         = base.duration,
# # # #             risk_score       = max_risk,
# # # #             risk_level       = risk_level,
# # # #             confidence       = base.confidence,
# # # #             factors          = list(set(factors)),
# # # #             source_filename  = base.source_filename,
# # # #             local_time_str   = base.local_time_str,
# # # #             annotated_frame  = best_frame,
# # # #         )

# # # #     def extract_violation_frames(self, video_path: str):
# # # #         print("[ViolationStore] Saving frames...")
# # # #         need_video = [v for v in self._violations if v.annotated_frame is None]
# # # #         saved = 0
# # # #         for v in self._violations:
# # # #             if v.annotated_frame is not None:
# # # #                 v.frame_path      = self._save_frame(v.annotated_frame, v.events, v.time_str)
# # # #                 v.annotated_frame = None
# # # #                 saved += 1
# # # #         if need_video:
# # # #             cap = cv2.VideoCapture(video_path)
# # # #             if not cap.isOpened():
# # # #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# # # #             else:
# # # #                 seen: set = set()
# # # #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# # # #                     if v.frame_index in seen:
# # # #                         continue
# # # #                     seen.add(v.frame_index)
# # # #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# # # #                     ret, frame = cap.read()
# # # #                     if not ret:
# # # #                         continue
# # # #                     v.frame_path = self._save_frame(frame, v.events, v.time_str)
# # # #                     saved += 1
# # # #                 cap.release()
# # # #         print(f"[ViolationStore] {saved} frames saved")

# # # #     def _save_frame(self, frame: np.ndarray, events: List[str], time_str: str) -> str:
# # # #         distraction   = "_".join(events)
# # # #         filename_time = time_str.replace(":", "-")
# # # #         filename      = f"{distraction}_{filename_time}.jpg"
# # # #         path          = os.path.join(self.frames_dir, filename)
# # # #         ok = cv2.imwrite(path, cv2.resize(frame, (640, 360)),
# # # #                          [cv2.IMWRITE_JPEG_QUALITY, 85])
# # # #         if not ok:
# # # #             print(f"[ViolationStore] imwrite failed: {path}")
# # # #         return os.path.join(self.analysis_id, "frames", filename)

# # # #     def _build_report(self, processing_time: float = 0.0) -> dict:
# # # #         return {
# # # #             "analysis_id":     self.analysis_id,
# # # #             "train_detail_id": self.train_detail_id,
# # # #             "processing_time": round(processing_time, 3),
# # # #             # Single video → keep as dict for backwards compat; batch → list
# # # #             "video_info": self.video_infos[0] if len(self.video_infos) == 1
# # # #                           else self.video_infos,
# # # #             "violations": [
# # # #                 {
# # # #                     "timestamp":   v.time_str,
# # # #                     "frame_index": v.frame_index,
# # # #                     "events":      v.events,
# # # #                     "severity":    v.severity,
# # # #                     "duration":    v.duration,
# # # #                     "risk_score":  v.risk_score,
# # # #                     "risk_level":  v.risk_level,
# # # #                     "confidence":  v.confidence,
# # # #                     "factors":     v.factors,
# # # #                     "original_video_timestamp": f"{v.source_filename} {v.local_time_str}",
# # # #                     "frame_path":  v.frame_path,
# # # #                 }
# # # #                 for v in self._violations
# # # #             ],
# # # #         }

# # # #     def finalize(self, processing_time: float = 0.0) -> str:
# # # #         self._deduplicate_by_frame()
# # # #         self._merge_by_time_window()
# # # #         # Extract frames from every video in the batch (or the single video)
# # # #         for vi in self.video_infos:
# # # #             if vi and vi.get("videoPath"):
# # # #                 self.extract_violation_frames(vi["videoPath"])
# # # #         report   = self._build_report(processing_time=processing_time)
# # # #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# # # #         with open(out_path, "w", encoding="utf-8") as f:
# # # #             json.dump(report, f, indent=2)
# # # #         print(f"[ViolationStore] JSON report     : {out_path}")
# # # #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# # # #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# # # #         try:
# # # #             from utils.db_s3_uploader import finalize_and_upload
# # # #             finalize_and_upload(
# # # #                 report_path     = out_path,
# # # #                 analysis_id     = self.analysis_id,
# # # #                 train_detail_id = self.train_detail_id,
# # # #             )
# # # #         except Exception as exc:
# # # #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# # # #         return out_path



# # # # utils/violation_store.py
# # # # ─────────────────────────────────────────────────────────────────
# # # # Change from original:
# # # #   _save_frame() now appends _f{frame_index} to the filename so
# # # #   two violations of the same type at the same timestamp never
# # # #   silently overwrite each other.
# # # #   e.g.  phone_use_00-13-16_f6762.jpg
# # # #         seat_absence_00-00-17_f516.jpg
# # # # ─────────────────────────────────────────────────────────────────

# # # from __future__ import annotations

# # # import json
# # # import os
# # # from dataclasses import dataclass
# # # from typing import Any, Dict, List, Optional

# # # import cv2
# # # import numpy as np

# # # OUTPUTS_ROOT = "outputs"
# # # MERGE_WINDOW = 2.0


# # # @dataclass
# # # class _Violation:
# # #     timestamp:       float
# # #     time_str:        str
# # #     frame_index:     int
# # #     type:            str
# # #     events:          List[str]
# # #     severity:        str
# # #     duration:        float
# # #     risk_score:      int
# # #     risk_level:      str
# # #     confidence:      float
# # #     factors:         List[str]
# # #     source_filename: str                  = ""
# # #     local_time_str:  str                  = ""
# # #     frame_path:      Optional[str]        = None
# # #     annotated_frame: Optional[np.ndarray] = None


# # # class ViolationStore:

# # #     def __init__(
# # #         self,
# # #         analysis_id:     str,
# # #         train_detail_id: int,
# # #         video_info:      Optional[Dict[str, Any]] = None,
# # #     ):
# # #         self.analysis_id     = analysis_id
# # #         self.train_detail_id = train_detail_id
# # #         self.video_infos: List[Dict[str, Any]] = (
# # #             [video_info] if video_info is not None else []
# # #         )

# # #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# # #         self.frames_dir = os.path.join(self.output_dir, "frames")
# # #         os.makedirs(self.frames_dir, exist_ok=True)

# # #         self._violations: List[_Violation] = []
# # #         self._seen_frames: set             = set()
# # #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# # #     # ── called once per video in batch mode ──────────────────────
# # #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# # #         self.video_infos.append(video_info)

# # #     # ── called from main.py for every detected violation ─────────
# # #     def record_violation(
# # #         self,
# # #         annotated_frame:  np.ndarray,
# # #         video_time:       float,
# # #         frame_index:      int,
# # #         event_type:       str,
# # #         original_frame:   Optional[np.ndarray] = None,
# # #         severity:         str   = "CRITICAL",
# # #         confidence:       float = 0.9,
# # #         risk_score:       int   = 80,
# # #         risk_level:       str   = "CRITICAL",
# # #         factors:          Optional[List[str]] = None,
# # #         duration:         float = 0.0,
# # #         source_filename:  str   = "",
# # #         local_video_time: float = -1.0,
# # #     ):
# # #         # Deduplicate: same frame + same event type recorded only once.
# # #         # Different event types on the same frame are each recorded.
# # #         # frame_index is globally unique across batch (frame_offset in main.py).
# # #         dedup_key = (frame_index, event_type)
# # #         if dedup_key in self._seen_frames:
# # #             return
# # #         self._seen_frames.add(dedup_key)

# # #         factors  = factors or []
# # #         t        = int(round(video_time))
# # #         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

# # #         # Local timestamp = time within the source video file (before cumulative offset)
# # #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# # #         local_str = f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"

# # #         self._violations.append(_Violation(
# # #             timestamp        = video_time,
# # #             time_str         = time_str,
# # #             frame_index      = frame_index,
# # #             type             = event_type,
# # #             events           = [event_type],
# # #             severity         = severity,
# # #             duration         = round(duration, 2),
# # #             risk_score       = risk_score,
# # #             risk_level       = risk_level,
# # #             confidence       = round(confidence, 3),
# # #             factors          = list(factors),
# # #             source_filename  = source_filename,
# # #             local_time_str   = local_str,
# # #             annotated_frame  = annotated_frame.copy() if annotated_frame is not None else None,
# # #         ))

# # #     # ─────────────────────────────────────────────────────────────

# # #     def _deduplicate_by_frame(self):
# # #         unique: Dict[int, _Violation] = {}
# # #         for v in self._violations:
# # #             if v.frame_index not in unique:
# # #                 unique[v.frame_index] = v
# # #             else:
# # #                 ex = unique[v.frame_index]
# # #                 ex.events  = list(set(ex.events  + v.events))
# # #                 ex.factors = list(set(ex.factors + v.factors))
# # #                 if v.risk_score > ex.risk_score:
# # #                     ex.risk_score = v.risk_score
# # #                     ex.risk_level = v.risk_level
# # #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# # #                     ex.annotated_frame = v.annotated_frame
# # #         self._violations = list(unique.values())

# # #     def _merge_by_time_window(self):
# # #         if not self._violations:
# # #             return
# # #         self._violations.sort(key=lambda x: x.timestamp)
# # #         merged = []
# # #         group  = [self._violations[0]]
# # #         for v in self._violations[1:]:
# # #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# # #                 group.append(v)
# # #             else:
# # #                 merged.append(self._merge_group(group))
# # #                 group = [v]
# # #         merged.append(self._merge_group(group))
# # #         self._violations = merged

# # #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# # #         base            = group[0]
# # #         events, factors = [], []
# # #         max_risk        = base.risk_score
# # #         risk_level      = base.risk_level
# # #         best_frame      = base.annotated_frame
# # #         for v in group:
# # #             events.extend(v.events)
# # #             factors.extend(v.factors)
# # #             if v.risk_score > max_risk:
# # #                 max_risk, risk_level = v.risk_score, v.risk_level
# # #             if best_frame is None and v.annotated_frame is not None:
# # #                 best_frame = v.annotated_frame
# # #         return _Violation(
# # #             timestamp        = base.timestamp,
# # #             time_str         = base.time_str,
# # #             frame_index      = base.frame_index,
# # #             type             = base.type,
# # #             events           = list(set(events)),
# # #             severity         = base.severity,
# # #             duration         = base.duration,
# # #             risk_score       = max_risk,
# # #             risk_level       = risk_level,
# # #             confidence       = base.confidence,
# # #             factors          = list(set(factors)),
# # #             source_filename  = base.source_filename,
# # #             local_time_str   = base.local_time_str,
# # #             annotated_frame  = best_frame,
# # #         )

# # #     # ─────────────────────────────────────────────────────────────

# # #     def extract_violation_frames(self, video_path: str):
# # #         print("[ViolationStore] Saving frames...")
# # #         need_video = [v for v in self._violations if v.annotated_frame is None]
# # #         saved = 0

# # #         for v in self._violations:
# # #             if v.annotated_frame is not None:
# # #                 v.frame_path      = self._save_frame(
# # #                     v.annotated_frame, v.events, v.time_str, v.frame_index
# # #                 )
# # #                 v.annotated_frame = None
# # #                 saved += 1

# # #         if need_video:
# # #             cap = cv2.VideoCapture(video_path)
# # #             if not cap.isOpened():
# # #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# # #             else:
# # #                 seen: set = set()
# # #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# # #                     if v.frame_index in seen:
# # #                         continue
# # #                     seen.add(v.frame_index)
# # #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# # #                     ret, frame = cap.read()
# # #                     if not ret:
# # #                         continue
# # #                     v.frame_path = self._save_frame(
# # #                         frame, v.events, v.time_str, v.frame_index
# # #                     )
# # #                     saved += 1
# # #                 cap.release()

# # #         print(f"[ViolationStore] {saved} frames saved")

# # #     def _save_frame(
# # #         self,
# # #         frame:       np.ndarray,
# # #         events:      List[str],
# # #         time_str:    str,
# # #         frame_index: int,            # ← ADDED: makes filename globally unique
# # #     ) -> str:
# # #         """
# # #         Save one violation frame as JPEG.

# # #         Filename: {event_types}_{hh-mm-ss}_f{frame_index}.jpg
# # #         Example:  phone_use_00-13-16_f6762.jpg
# # #                   seat_absence_00-00-17_f516.jpg

# # #         frame_index prevents two violations of the same type at the
# # #         same timestamp from silently overwriting each other.
# # #         """
# # #         distraction   = "_".join(events)
# # #         filename_time = time_str.replace(":", "-")
# # #         filename      = f"{distraction}_{filename_time}_f{frame_index}.jpg"  # ← CHANGED
# # #         path          = os.path.join(self.frames_dir, filename)
# # #         ok = cv2.imwrite(
# # #             path,
# # #             cv2.resize(frame, (640, 360)),
# # #             [cv2.IMWRITE_JPEG_QUALITY, 85],
# # #         )
# # #         if not ok:
# # #             print(f"[ViolationStore] imwrite failed: {path}")
# # #         return os.path.join(self.analysis_id, "frames", filename)

# # #     # ─────────────────────────────────────────────────────────────

# # #     def _build_report(self, processing_time: float = 0.0) -> dict:
# # #         return {
# # #             "analysis_id":     self.analysis_id,
# # #             "train_detail_id": self.train_detail_id,
# # #             "processing_time": round(processing_time, 3),
# # #             "video_info": (
# # #                 self.video_infos[0] if len(self.video_infos) == 1
# # #                 else self.video_infos
# # #             ),
# # #             "violations": [
# # #                 {
# # #                     "timestamp":   v.time_str,
# # #                     "frame_index": v.frame_index,
# # #                     "events":      v.events,
# # #                     "severity":    v.severity,
# # #                     "duration":    v.duration,
# # #                     "risk_score":  v.risk_score,
# # #                     "risk_level":  v.risk_level,
# # #                     "confidence":  v.confidence,
# # #                     "factors":     v.factors,
# # #                     "original_video_timestamp": f"{v.source_filename} {v.local_time_str}",
# # #                     "frame_path":  v.frame_path,
# # #                 }
# # #                 for v in self._violations
# # #             ],
# # #         }

# # #     def finalize(self, processing_time: float = 0.0) -> str:
# # #         self._deduplicate_by_frame()
# # #         self._merge_by_time_window()

# # #         # Save frames — each video's temp path is in video_infos
# # #         for vi in self.video_infos:
# # #             if vi and vi.get("videoPath"):
# # #                 self.extract_violation_frames(vi["videoPath"])

# # #         report   = self._build_report(processing_time=processing_time)
# # #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# # #         with open(out_path, "w", encoding="utf-8") as f:
# # #             json.dump(report, f, indent=2)

# # #         print(f"[ViolationStore] JSON report     : {out_path}")
# # #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# # #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# # #         # Upload results to S3 and update DB result_s3_path
# # #         try:
# # #             from utils.db_s3_uploader import finalize_and_upload
# # #             finalize_and_upload(
# # #                 report_path     = out_path,
# # #                 analysis_id     = self.analysis_id,
# # #                 train_detail_id = self.train_detail_id,
# # #             )
# # #         except Exception as exc:
# # #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# # #         return out_path

# # # # # from __future__ import annotations

# # # # # import json
# # # # # import os
# # # # # from dataclasses import dataclass
# # # # # from typing import Any, Dict, List, Optional

# # # # # import cv2
# # # # # import numpy as np

# # # # # OUTPUTS_ROOT = "outputs"
# # # # # MERGE_WINDOW = 2.0


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # INTERNAL DATA CLASS
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # @dataclass
# # # # # class _Violation:
# # # # #     timestamp:       float
# # # # #     time_str:        str
# # # # #     frame_index:     int
# # # # #     type:            str
# # # # #     events:          List[str]
# # # # #     severity:        str
# # # # #     duration:        float
# # # # #     risk_score:      int
# # # # #     risk_level:      str
# # # # #     confidence:      float
# # # # #     factors:         List[str]
# # # # #     source_filename: str                  = ""    # e.g. "ch01.mp4"
# # # # #     local_time_str:  str                  = ""    # local time within that file e.g. "00:00:18"
# # # # #     frame_path:      Optional[str]        = None
# # # # #     annotated_frame: Optional[np.ndarray] = None


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # VIOLATION STORE
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # class ViolationStore:
# # # # #     """
# # # # #     Accumulates all violations found across one analysis run (single video
# # # # #     or a multi-video batch that shares the same analysis_id / folder_name).

# # # # #     Usage
# # # # #     ─────
# # # # #     1. Construct once per analysis run.
# # # # #     2. Call record_violation() from the pipeline for every distraction event.
# # # # #     3. Call finalize() at the very end — it deduplicates, merges, saves frames,
# # # # #        writes the JSON report, then calls db_s3_uploader.finalize_and_upload()
# # # # #        to push everything to S3 and record results in the DB.
# # # # #     """

# # # # #     def __init__(
# # # # #         self,
# # # # #         analysis_id:     str,
# # # # #         train_detail_id: int,
# # # # #         video_info:      Optional[Dict[str, Any]] = None,
# # # # #     ):
# # # # #         self.analysis_id     = analysis_id
# # # # #         self.train_detail_id = train_detail_id
# # # # #         # video_infos is always a list:
# # # # #         #   • 1 entry for single-video runs
# # # # #         #   • N entries for batch runs (add_video_info called per video)
# # # # #         self.video_infos: List[Dict[str, Any]] = (
# # # # #             [video_info] if video_info is not None else []
# # # # #         )

# # # # #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# # # # #         self.frames_dir = os.path.join(self.output_dir, "frames")
# # # # #         os.makedirs(self.frames_dir, exist_ok=True)

# # # # #         self._violations:  List[_Violation] = []
# # # # #         self._seen_frames: set              = set()
# # # # #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# # # # #     # ── Public helpers ────────────────────────────────────────────────────────

# # # # #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# # # # #         """
# # # # #         Register one video's metadata into the shared store.
# # # # #         Called once per video in batch mode.
# # # # #         """
# # # # #         self.video_infos.append(video_info)

# # # # #     def record_violation(
# # # # #         self,
# # # # #         annotated_frame:  np.ndarray,
# # # # #         video_time:       float,
# # # # #         frame_index:      int,
# # # # #         event_type:       str,
# # # # #         original_frame:   Optional[np.ndarray] = None,
# # # # #         severity:         str   = "CRITICAL",
# # # # #         confidence:       float = 0.9,
# # # # #         risk_score:       int   = 80,
# # # # #         risk_level:       str   = "CRITICAL",
# # # # #         factors:          Optional[List[str]] = None,
# # # # #         duration:         float = 0.0,
# # # # #         source_filename:  str   = "",    # original filename e.g. "ch01.mp4"
# # # # #         local_video_time: float = -1.0,  # video_time before global offset; -1 = same as video_time
# # # # #     ) -> None:
# # # # #         """
# # # # #         Record one distraction event.

# # # # #         Deduplication key is (frame_index, event_type) so that:
# # # # #           • the same violation type on the same frame is recorded only once
# # # # #           • different violation types on the same frame are each recorded
# # # # #           • frame numbers from different videos never collide because
# # # # #             main.py applies a frame_offset to make every global frame_index
# # # # #             unique across the batch
# # # # #         """
# # # # #         dedup_key = (frame_index, event_type)
# # # # #         if dedup_key in self._seen_frames:
# # # # #             return
# # # # #         self._seen_frames.add(dedup_key)

# # # # #         factors  = factors or []
# # # # #         t        = int(round(video_time))
# # # # #         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

# # # # #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# # # # #         local_str = (
# # # # #             f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
# # # # #         )

# # # # #         self._violations.append(
# # # # #             _Violation(
# # # # #                 timestamp        = video_time,
# # # # #                 time_str         = time_str,
# # # # #                 frame_index      = frame_index,
# # # # #                 type             = event_type,
# # # # #                 events           = [event_type],
# # # # #                 severity         = severity,
# # # # #                 duration         = round(duration, 2),
# # # # #                 risk_score       = risk_score,
# # # # #                 risk_level       = risk_level,
# # # # #                 confidence       = round(confidence, 3),
# # # # #                 factors          = list(factors),
# # # # #                 source_filename  = source_filename,
# # # # #                 local_time_str   = local_str,
# # # # #                 annotated_frame  = (
# # # # #                     annotated_frame.copy() if annotated_frame is not None else None
# # # # #                 ),
# # # # #             )
# # # # #         )

# # # # #     # ── Finalize ──────────────────────────────────────────────────────────────

# # # # #     def finalize(self, processing_time: float = 0.0) -> str:
# # # # #         """
# # # # #         1. Deduplicate violations that share the same frame.
# # # # #         2. Merge violations that are within MERGE_WINDOW seconds of each other.
# # # # #         3. Save annotated frame images to disk (falls back to re-reading from
# # # # #            the source video when no annotated frame was captured).
# # # # #         4. Write analysis_report.json.
# # # # #         5. Upload frames + report to S3 and record in DB via db_s3_uploader.
# # # # #         Returns the local path to analysis_report.json.
# # # # #         """
# # # # #         self._deduplicate_by_frame()
# # # # #         self._merge_by_time_window()

# # # # #         # Extract frames from every video in the batch (or the single video)
# # # # #         for vi in self.video_infos:
# # # # #             if vi and vi.get("videoPath"):
# # # # #                 self.extract_violation_frames(vi["videoPath"])

# # # # #         report   = self._build_report(processing_time=processing_time)
# # # # #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# # # # #         with open(out_path, "w", encoding="utf-8") as f:
# # # # #             json.dump(report, f, indent=2)

# # # # #         print(f"[ViolationStore] JSON report     : {out_path}")
# # # # #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# # # # #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# # # # #         try:
# # # # #             from utils.db_s3_uploader import finalize_and_upload
# # # # #             finalize_and_upload(
# # # # #                 report_path     = out_path,
# # # # #                 analysis_id     = self.analysis_id,
# # # # #                 train_detail_id = self.train_detail_id,
# # # # #             )
# # # # #         except Exception as exc:
# # # # #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# # # # #         return out_path

# # # # #     # ── Private — deduplication & merging ────────────────────────────────────

# # # # #     def _deduplicate_by_frame(self) -> None:
# # # # #         unique: Dict[int, _Violation] = {}
# # # # #         for v in self._violations:
# # # # #             if v.frame_index not in unique:
# # # # #                 unique[v.frame_index] = v
# # # # #             else:
# # # # #                 ex = unique[v.frame_index]
# # # # #                 ex.events  = list(set(ex.events  + v.events))
# # # # #                 ex.factors = list(set(ex.factors + v.factors))
# # # # #                 if v.risk_score > ex.risk_score:
# # # # #                     ex.risk_score = v.risk_score
# # # # #                     ex.risk_level = v.risk_level
# # # # #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# # # # #                     ex.annotated_frame = v.annotated_frame
# # # # #         self._violations = list(unique.values())

# # # # #     def _merge_by_time_window(self) -> None:
# # # # #         if not self._violations:
# # # # #             return
# # # # #         self._violations.sort(key=lambda x: x.timestamp)
# # # # #         merged: List[_Violation] = []
# # # # #         group  = [self._violations[0]]
# # # # #         for v in self._violations[1:]:
# # # # #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# # # # #                 group.append(v)
# # # # #             else:
# # # # #                 merged.append(self._merge_group(group))
# # # # #                 group = [v]
# # # # #         merged.append(self._merge_group(group))
# # # # #         self._violations = merged

# # # # #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# # # # #         base            = group[0]
# # # # #         events: List[str]  = []
# # # # #         factors: List[str] = []
# # # # #         max_risk   = base.risk_score
# # # # #         risk_level = base.risk_level
# # # # #         best_frame = base.annotated_frame
# # # # #         for v in group:
# # # # #             events.extend(v.events)
# # # # #             factors.extend(v.factors)
# # # # #             if v.risk_score > max_risk:
# # # # #                 max_risk, risk_level = v.risk_score, v.risk_level
# # # # #             if best_frame is None and v.annotated_frame is not None:
# # # # #                 best_frame = v.annotated_frame
# # # # #         return _Violation(
# # # # #             timestamp        = base.timestamp,
# # # # #             time_str         = base.time_str,
# # # # #             frame_index      = base.frame_index,
# # # # #             type             = base.type,
# # # # #             events           = list(set(events)),
# # # # #             severity         = base.severity,
# # # # #             duration         = base.duration,
# # # # #             risk_score       = max_risk,
# # # # #             risk_level       = risk_level,
# # # # #             confidence       = base.confidence,
# # # # #             factors          = list(set(factors)),
# # # # #             source_filename  = base.source_filename,
# # # # #             local_time_str   = base.local_time_str,
# # # # #             annotated_frame  = best_frame,
# # # # #         )

# # # # #     # ── Private — frame extraction & saving ──────────────────────────────────

# # # # #     def extract_violation_frames(self, video_path: str) -> None:
# # # # #         print("[ViolationStore] Saving frames...")
# # # # #         need_video = [v for v in self._violations if v.annotated_frame is None]
# # # # #         saved = 0

# # # # #         # First pass: save violations that already have an annotated frame
# # # # #         for v in self._violations:
# # # # #             if v.annotated_frame is not None:
# # # # #                 v.frame_path      = self._save_frame(v.annotated_frame, v.events, v.time_str)
# # # # #                 v.annotated_frame = None   # free memory
# # # # #                 saved += 1

# # # # #         # Second pass: re-read from the source video for any that are missing
# # # # #         if need_video:
# # # # #             cap = cv2.VideoCapture(video_path)
# # # # #             if not cap.isOpened():
# # # # #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# # # # #             else:
# # # # #                 seen: set = set()
# # # # #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# # # # #                     if v.frame_index in seen:
# # # # #                         continue
# # # # #                     seen.add(v.frame_index)
# # # # #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# # # # #                     ret, frame = cap.read()
# # # # #                     if not ret:
# # # # #                         continue
# # # # #                     v.frame_path = self._save_frame(frame, v.events, v.time_str)
# # # # #                     saved += 1
# # # # #                 cap.release()

# # # # #         print(f"[ViolationStore] {saved} frames saved")

# # # # #     def _save_frame(
# # # # #         self,
# # # # #         frame:    np.ndarray,
# # # # #         events:   List[str],
# # # # #         time_str: str,
# # # # #     ) -> str:
# # # # #         distraction   = "_".join(events)
# # # # #         filename_time = time_str.replace(":", "-")
# # # # #         filename      = f"{distraction}_{filename_time}.jpg"
# # # # #         path          = os.path.join(self.frames_dir, filename)
# # # # #         ok = cv2.imwrite(
# # # # #             path,
# # # # #             cv2.resize(frame, (640, 360)),
# # # # #             [cv2.IMWRITE_JPEG_QUALITY, 85],
# # # # #         )
# # # # #         if not ok:
# # # # #             print(f"[ViolationStore] imwrite failed: {path}")
# # # # #         return os.path.join(self.analysis_id, "frames", filename)

# # # # #     # ── Private — report builder ──────────────────────────────────────────────

# # # # #     def _build_report(self, processing_time: float = 0.0) -> dict:
# # # # #         return {
# # # # #             "analysis_id":     self.analysis_id,
# # # # #             "train_detail_id": self.train_detail_id,
# # # # #             "processing_time": round(processing_time, 3),
# # # # #             # Single video → keep as dict for backwards compat; batch → list
# # # # #             "video_info": (
# # # # #                 self.video_infos[0]
# # # # #                 if len(self.video_infos) == 1
# # # # #                 else self.video_infos
# # # # #             ),
# # # # #             "violations": [
# # # # #                 {
# # # # #                     "timestamp":   v.time_str,
# # # # #                     "frame_index": v.frame_index,
# # # # #                     "events":      v.events,
# # # # #                     "severity":    v.severity,
# # # # #                     "duration":    v.duration,
# # # # #                     "risk_score":  v.risk_score,
# # # # #                     "risk_level":  v.risk_level,
# # # # #                     "confidence":  v.confidence,
# # # # #                     "factors":     v.factors,
# # # # #                     "original_video_timestamp": (
# # # # #                         f"{v.source_filename} {v.local_time_str}"
# # # # #                     ),
# # # # #                     "frame_path":  v.frame_path,
# # # # #                 }
# # # # #                 for v in self._violations
# # # # #             ],
# # # # #         }

# # # # from __future__ import annotations

# # # # import json
# # # # import os
# # # # from dataclasses import dataclass
# # # # from typing import Any, Dict, List, Optional

# # # # import cv2
# # # # import numpy as np

# # # # OUTPUTS_ROOT = "outputs"
# # # # MERGE_WINDOW = 2.0


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # INTERNAL DATA CLASS
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # @dataclass
# # # # class _Violation:
# # # #     timestamp:       float
# # # #     time_str:        str
# # # #     frame_index:     int
# # # #     type:            str
# # # #     events:          List[str]
# # # #     severity:        str
# # # #     duration:        float
# # # #     risk_score:      int
# # # #     risk_level:      str
# # # #     confidence:      float
# # # #     factors:         List[str]
# # # #     source_filename: str                  = ""    # e.g. "ch01.mp4"
# # # #     local_time_str:  str                  = ""    # local time within that file e.g. "00:00:18"
# # # #     frame_path:      Optional[str]        = None
# # # #     annotated_frame: Optional[np.ndarray] = None


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # VIOLATION STORE
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # class ViolationStore:
# # # #     """
# # # #     Accumulates all violations found across one analysis run (single video
# # # #     or a multi-video batch that shares the same analysis_id / folder_name).

# # # #     Usage
# # # #     ─────
# # # #     1. Construct once per analysis run.
# # # #     2. Call record_violation() from the pipeline for every distraction event.
# # # #     3. Call finalize() at the very end — it deduplicates, merges, saves frames,
# # # #        writes the JSON report, then calls db_s3_uploader.finalize_and_upload()
# # # #        to push everything to S3 and record results in the DB.
# # # #     """

# # # #     def __init__(
# # # #         self,
# # # #         analysis_id:     str,
# # # #         train_detail_id: int,
# # # #         video_info:      Optional[Dict[str, Any]] = None,
# # # #     ):
# # # #         self.analysis_id     = analysis_id
# # # #         self.train_detail_id = train_detail_id
# # # #         # video_infos is always a list:
# # # #         #   • 1 entry for single-video runs
# # # #         #   • N entries for batch runs (add_video_info called per video)
# # # #         self.video_infos: List[Dict[str, Any]] = (
# # # #             [video_info] if video_info is not None else []
# # # #         )

# # # #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# # # #         self.frames_dir = os.path.join(self.output_dir, "frames")
# # # #         os.makedirs(self.frames_dir, exist_ok=True)

# # # #         self._violations:  List[_Violation] = []
# # # #         self._seen_frames: set              = set()
# # # #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# # # #     # ── Public helpers ────────────────────────────────────────────────────────

# # # #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# # # #         """
# # # #         Register one video's metadata into the shared store.
# # # #         Called once per video in batch mode.
# # # #         """
# # # #         self.video_infos.append(video_info)

# # # #     def record_violation(
# # # #         self,
# # # #         annotated_frame:  np.ndarray,
# # # #         video_time:       float,
# # # #         frame_index:      int,
# # # #         event_type:       str,
# # # #         original_frame:   Optional[np.ndarray] = None,
# # # #         severity:         str   = "CRITICAL",
# # # #         confidence:       float = 0.9,
# # # #         risk_score:       int   = 80,
# # # #         risk_level:       str   = "CRITICAL",
# # # #         factors:          Optional[List[str]] = None,
# # # #         duration:         float = 0.0,
# # # #         source_filename:  str   = "",    # original filename e.g. "ch01.mp4"
# # # #         local_video_time: float = -1.0,  # video_time before global offset; -1 = same as video_time
# # # #     ) -> None:
# # # #         """
# # # #         Record one distraction event.

# # # #         Deduplication key is (frame_index, event_type) so that:
# # # #           • the same violation type on the same frame is recorded only once
# # # #           • different violation types on the same frame are each recorded
# # # #           • frame numbers from different videos never collide because
# # # #             main.py applies a frame_offset to make every global frame_index
# # # #             unique across the batch
# # # #         """
# # # #         dedup_key = (frame_index, event_type)
# # # #         if dedup_key in self._seen_frames:
# # # #             return
# # # #         self._seen_frames.add(dedup_key)

# # # #         factors  = factors or []
# # # #         t        = int(round(video_time))
# # # #         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

# # # #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# # # #         local_str = (
# # # #             f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
# # # #         )

# # # #         self._violations.append(
# # # #             _Violation(
# # # #                 timestamp        = video_time,
# # # #                 time_str         = time_str,
# # # #                 frame_index      = frame_index,
# # # #                 type             = event_type,
# # # #                 events           = [event_type],
# # # #                 severity         = severity,
# # # #                 duration         = round(duration, 2),
# # # #                 risk_score       = risk_score,
# # # #                 risk_level       = risk_level,
# # # #                 confidence       = round(confidence, 3),
# # # #                 factors          = list(factors),
# # # #                 source_filename  = source_filename,
# # # #                 local_time_str   = local_str,
# # # #                 annotated_frame  = (
# # # #                     annotated_frame.copy() if annotated_frame is not None else None
# # # #                 ),
# # # #             )
# # # #         )

# # # #     # ── Finalize ──────────────────────────────────────────────────────────────

# # # #     def finalize(self, processing_time: float = 0.0) -> str:
# # # #         """
# # # #         1. Deduplicate violations that share the same frame.
# # # #         2. Merge violations that are within MERGE_WINDOW seconds of each other.
# # # #         3. Save annotated frame images to disk (falls back to re-reading from
# # # #            the source video when no annotated frame was captured).
# # # #         4. Write analysis_report.json.
# # # #         5. Upload frames + report to S3 and record in DB via db_s3_uploader.
# # # #         Returns the local path to analysis_report.json.
# # # #         """
# # # #         self._deduplicate_by_frame()
# # # #         self._merge_by_time_window()

# # # #         # Extract frames from every video in the batch (or the single video)
# # # #         for vi in self.video_infos:
# # # #             if vi and vi.get("videoPath"):
# # # #                 self.extract_violation_frames(vi["videoPath"])

# # # #         report   = self._build_report(processing_time=processing_time)
# # # #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# # # #         with open(out_path, "w", encoding="utf-8") as f:
# # # #             json.dump(report, f, indent=2)

# # # #         print(f"[ViolationStore] JSON report     : {out_path}")
# # # #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# # # #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# # # #         try:
# # # #             from utils.db_s3_uploader import finalize_and_upload
# # # #             finalize_and_upload(
# # # #                 report_path     = out_path,
# # # #                 analysis_id     = self.analysis_id,
# # # #                 train_detail_id = self.train_detail_id,
# # # #             )
# # # #         except Exception as exc:
# # # #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# # # #         return out_path

# # # #     # ── Private — deduplication & merging ────────────────────────────────────

# # # #     def _deduplicate_by_frame(self) -> None:
# # # #         unique: Dict[int, _Violation] = {}
# # # #         for v in self._violations:
# # # #             if v.frame_index not in unique:
# # # #                 unique[v.frame_index] = v
# # # #             else:
# # # #                 ex = unique[v.frame_index]
# # # #                 ex.events  = list(set(ex.events  + v.events))
# # # #                 ex.factors = list(set(ex.factors + v.factors))
# # # #                 if v.risk_score > ex.risk_score:
# # # #                     ex.risk_score = v.risk_score
# # # #                     ex.risk_level = v.risk_level
# # # #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# # # #                     ex.annotated_frame = v.annotated_frame
# # # #         self._violations = list(unique.values())

# # # #     def _merge_by_time_window(self) -> None:
# # # #         if not self._violations:
# # # #             return
# # # #         self._violations.sort(key=lambda x: x.timestamp)
# # # #         merged: List[_Violation] = []
# # # #         group  = [self._violations[0]]
# # # #         for v in self._violations[1:]:
# # # #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# # # #                 group.append(v)
# # # #             else:
# # # #                 merged.append(self._merge_group(group))
# # # #                 group = [v]
# # # #         merged.append(self._merge_group(group))
# # # #         self._violations = merged

# # # #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# # # #         base             = group[0]
# # # #         events: List[str]  = []
# # # #         factors: List[str] = []
# # # #         max_risk   = base.risk_score
# # # #         risk_level = base.risk_level
# # # #         best_frame = base.annotated_frame
# # # #         for v in group:
# # # #             events.extend(v.events)
# # # #             factors.extend(v.factors)
# # # #             if v.risk_score > max_risk:
# # # #                 max_risk, risk_level = v.risk_score, v.risk_level
# # # #             if best_frame is None and v.annotated_frame is not None:
# # # #                 best_frame = v.annotated_frame
# # # #         return _Violation(
# # # #             timestamp        = base.timestamp,
# # # #             time_str         = base.time_str,
# # # #             frame_index      = base.frame_index,
# # # #             type             = base.type,
# # # #             events           = list(set(events)),
# # # #             severity         = base.severity,
# # # #             duration         = base.duration,
# # # #             risk_score       = max_risk,
# # # #             risk_level       = risk_level,
# # # #             confidence       = base.confidence,
# # # #             factors          = list(set(factors)),
# # # #             source_filename  = base.source_filename,
# # # #             local_time_str   = base.local_time_str,
# # # #             annotated_frame  = best_frame,
# # # #         )

# # # #     # ── Private — frame extraction & saving ──────────────────────────────────

# # # #     def extract_violation_frames(self, video_path: str) -> None:
# # # #         print("[ViolationStore] Saving frames...")
# # # #         need_video = [v for v in self._violations if v.annotated_frame is None]
# # # #         saved = 0

# # # #         # First pass: save violations that already have an annotated frame
# # # #         for v in self._violations:
# # # #             if v.annotated_frame is not None:
# # # #                 v.frame_path      = self._save_frame(
# # # #                     v.annotated_frame, v.events, v.time_str, v.frame_index
# # # #                 )
# # # #                 v.annotated_frame = None   # free memory
# # # #                 saved += 1

# # # #         # Second pass: re-read from the source video for any that are missing
# # # #         if need_video:
# # # #             cap = cv2.VideoCapture(video_path)
# # # #             if not cap.isOpened():
# # # #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# # # #             else:
# # # #                 seen: set = set()
# # # #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# # # #                     if v.frame_index in seen:
# # # #                         continue
# # # #                     seen.add(v.frame_index)
# # # #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# # # #                     ret, frame = cap.read()
# # # #                     if not ret:
# # # #                         continue
# # # #                     v.frame_path = self._save_frame(
# # # #                         frame, v.events, v.time_str, v.frame_index
# # # #                     )
# # # #                     saved += 1
# # # #                 cap.release()

# # # #         print(f"[ViolationStore] {saved} frames saved")

# # # #     def _save_frame(
# # # #         self,
# # # #         frame:       np.ndarray,
# # # #         events:      List[str],
# # # #         time_str:    str,
# # # #         frame_index: int,
# # # #     ) -> str:
# # # #         """
# # # #         Save a single violation frame as a JPEG.

# # # #         Filename format:  <events>_<HH-MM-SS>_f<frame_index>.jpg
# # # #         Example:          seat_absence_00-01-14_f2106.jpg
# # # #                           seat_absence_drowsy_00-03-02_f5058.jpg
# # # #         """
# # # #         distraction   = "_".join(sorted(events))   # sorted for deterministic name
# # # #         filename_time = time_str.replace(":", "-")
# # # #         filename      = f"{distraction}_{filename_time}_f{frame_index}.jpg"
# # # #         path          = os.path.join(self.frames_dir, filename)
# # # #         ok = cv2.imwrite(
# # # #             path,
# # # #             cv2.resize(frame, (640, 360)),
# # # #             [cv2.IMWRITE_JPEG_QUALITY, 85],
# # # #         )
# # # #         if not ok:
# # # #             print(f"[ViolationStore] imwrite failed: {path}")
# # # #         return os.path.join(self.analysis_id, "frames", filename)

# # # #     # ── Private — report builder ──────────────────────────────────────────────

# # # #     def _build_report(self, processing_time: float = 0.0) -> dict:
# # # #         return {
# # # #             "analysis_id":     self.analysis_id,
# # # #             "train_detail_id": self.train_detail_id,
# # # #             "processing_time": round(processing_time, 3),
# # # #             # Single video → keep as dict for backwards compat; batch → list
# # # #             "video_info": (
# # # #                 self.video_infos[0]
# # # #                 if len(self.video_infos) == 1
# # # #                 else self.video_infos
# # # #             ),
# # # #             "violations": [
# # # #                 {
# # # #                     "timestamp":   v.time_str,
# # # #                     "frame_index": v.frame_index,
# # # #                     "events":      v.events,
# # # #                     "severity":    v.severity,
# # # #                     "duration":    v.duration,
# # # #                     "risk_score":  v.risk_score,
# # # #                     "risk_level":  v.risk_level,
# # # #                     "confidence":  v.confidence,
# # # #                     "factors":     v.factors,
# # # #                     "original_video_timestamp": (
# # # #                         f"{v.source_filename} {v.local_time_str}"
# # # #                     ),
# # # #                     "frame_path":  v.frame_path,
# # # #                 }
# # # #                 for v in self._violations
# # # #             ],
# # # #         }


# # # from __future__ import annotations

# # # import json
# # # import os
# # # from dataclasses import dataclass
# # # from typing import Any, Dict, List, Optional

# # # import cv2
# # # import numpy as np

# # # OUTPUTS_ROOT = "outputs"
# # # MERGE_WINDOW = 2.0


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # # INTERNAL DATA CLASS
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # @dataclass
# # # class _Violation:
# # #     timestamp:       float
# # #     time_str:        str
# # #     frame_index:     int
# # #     type:            str
# # #     events:          List[str]
# # #     severity:        str
# # #     duration:        float
# # #     risk_score:      int
# # #     risk_level:      str
# # #     confidence:      float
# # #     factors:         List[str]
# # #     source_filename: str                  = ""    # DB filename e.g. "mobile.mp4"
# # #     local_time_str:  str                  = ""    # local time within that file e.g. "00:00:23"
# # #     frame_path:      Optional[str]        = None
# # #     annotated_frame: Optional[np.ndarray] = None


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # # VIOLATION STORE
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # class ViolationStore:
# # #     """
# # #     Accumulates all violations found across one analysis run (single video
# # #     or a multi-video batch that shares the same analysis_id / folder_name).

# # #     Batch mode usage (api.py)
# # #     ─────────────────────────
# # #     1. Construct ONCE for the whole folder (no video_info in __init__).
# # #     2. Pass as shared_vstore= to each GadgetDetectionPipeline.
# # #        The pipeline calls add_video_info() automatically.
# # #     3. Call finalize() ONCE after all videos in the folder are done.

# # #     Standalone mode usage (CLI / single video)
# # #     ──────────────────────────────────────────
# # #     1. Construct with video_info= for the single video.
# # #     2. Pipeline calls finalize() automatically at the end of run().
# # #     """

# # #     def __init__(
# # #         self,
# # #         analysis_id:     str,
# # #         train_detail_id: int,
# # #         video_info:      Optional[Dict[str, Any]] = None,
# # #     ):
# # #         self.analysis_id     = analysis_id
# # #         self.train_detail_id = train_detail_id
# # #         # Always a list — 0 entries until add_video_info() is called (batch mode),
# # #         # or 1 entry when video_info is provided (standalone mode).
# # #         self.video_infos: List[Dict[str, Any]] = (
# # #             [video_info] if video_info is not None else []
# # #         )

# # #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# # #         self.frames_dir = os.path.join(self.output_dir, "frames")
# # #         os.makedirs(self.frames_dir, exist_ok=True)

# # #         self._violations:  List[_Violation] = []
# # #         self._seen_frames: set              = set()
# # #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# # #     # ── Public helpers ────────────────────────────────────────────────────────

# # #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# # #         """Append one video's metadata. Called once per video in batch mode."""
# # #         self.video_infos.append(video_info)

# # #     def record_violation(
# # #         self,
# # #         annotated_frame:  np.ndarray,
# # #         video_time:       float,          # global timestamp (offset-adjusted)
# # #         frame_index:      int,            # global frame index (offset-adjusted)
# # #         event_type:       str,
# # #         original_frame:   Optional[np.ndarray] = None,
# # #         severity:         str   = "CRITICAL",
# # #         confidence:       float = 0.9,
# # #         risk_score:       int   = 80,
# # #         risk_level:       str   = "CRITICAL",
# # #         factors:          Optional[List[str]] = None,
# # #         duration:         float = 0.0,
# # #         source_filename:  str   = "",     # DB filename shown in original_video_timestamp
# # #         local_video_time: float = -1.0,   # local time within the source file; -1 = same as video_time
# # #     ) -> None:
# # #         """
# # #         Record one distraction event.

# # #         Deduplication key is (frame_index, event_type):
# # #           • same event on the same global frame is recorded once
# # #           • different events on the same global frame are each recorded
# # #           • global frame_index is unique across videos (frame_offset applied in main.py)
# # #         """
# # #         dedup_key = (frame_index, event_type)
# # #         if dedup_key in self._seen_frames:
# # #             return
# # #         self._seen_frames.add(dedup_key)

# # #         factors  = factors or []
# # #         t        = int(round(video_time))
# # #         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

# # #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# # #         local_str = (
# # #             f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
# # #         )

# # #         self._violations.append(
# # #             _Violation(
# # #                 timestamp        = video_time,
# # #                 time_str         = time_str,
# # #                 frame_index      = frame_index,
# # #                 type             = event_type,
# # #                 events           = [event_type],
# # #                 severity         = severity,
# # #                 duration         = round(duration, 2),
# # #                 risk_score       = risk_score,
# # #                 risk_level       = risk_level,
# # #                 confidence       = round(confidence, 3),
# # #                 factors          = list(factors),
# # #                 source_filename  = source_filename,
# # #                 local_time_str   = local_str,
# # #                 annotated_frame  = (
# # #                     annotated_frame.copy() if annotated_frame is not None else None
# # #                 ),
# # #             )
# # #         )

# # #     # ── Finalize ──────────────────────────────────────────────────────────────

# # #     def finalize(self, processing_time: float = 0.0) -> str:
# # #         """
# # #         Deduplicate → merge → save frames → write JSON → upload to S3/DB.
# # #         Returns the local path to analysis_report.json.
# # #         """
# # #         self._deduplicate_by_frame()
# # #         self._merge_by_time_window()

# # #         # Extract frames from every video in the batch (or the single video)
# # #         for vi in self.video_infos:
# # #             if vi and vi.get("videoPath"):
# # #                 self.extract_violation_frames(vi["videoPath"])

# # #         report   = self._build_report(processing_time=processing_time)
# # #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# # #         with open(out_path, "w", encoding="utf-8") as f:
# # #             json.dump(report, f, indent=2)

# # #         print(f"[ViolationStore] JSON report     : {out_path}")
# # #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# # #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# # #         try:
# # #             from utils.db_s3_uploader import finalize_and_upload
# # #             finalize_and_upload(
# # #                 report_path     = out_path,
# # #                 analysis_id     = self.analysis_id,
# # #                 train_detail_id = self.train_detail_id,
# # #             )
# # #         except Exception as exc:
# # #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# # #         return out_path

# # #     # ── Private — deduplication & merging ────────────────────────────────────

# # #     def _deduplicate_by_frame(self) -> None:
# # #         unique: Dict[int, _Violation] = {}
# # #         for v in self._violations:
# # #             if v.frame_index not in unique:
# # #                 unique[v.frame_index] = v
# # #             else:
# # #                 ex = unique[v.frame_index]
# # #                 ex.events  = list(set(ex.events  + v.events))
# # #                 ex.factors = list(set(ex.factors + v.factors))
# # #                 if v.risk_score > ex.risk_score:
# # #                     ex.risk_score = v.risk_score
# # #                     ex.risk_level = v.risk_level
# # #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# # #                     ex.annotated_frame = v.annotated_frame
# # #         self._violations = list(unique.values())

# # #     def _merge_by_time_window(self) -> None:
# # #         if not self._violations:
# # #             return
# # #         self._violations.sort(key=lambda x: x.timestamp)
# # #         merged: List[_Violation] = []
# # #         group  = [self._violations[0]]
# # #         for v in self._violations[1:]:
# # #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# # #                 group.append(v)
# # #             else:
# # #                 merged.append(self._merge_group(group))
# # #                 group = [v]
# # #         merged.append(self._merge_group(group))
# # #         self._violations = merged

# # #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# # #         base             = group[0]
# # #         events: List[str]  = []
# # #         factors: List[str] = []
# # #         max_risk   = base.risk_score
# # #         risk_level = base.risk_level
# # #         best_frame = base.annotated_frame
# # #         for v in group:
# # #             events.extend(v.events)
# # #             factors.extend(v.factors)
# # #             if v.risk_score > max_risk:
# # #                 max_risk, risk_level = v.risk_score, v.risk_level
# # #             if best_frame is None and v.annotated_frame is not None:
# # #                 best_frame = v.annotated_frame
# # #         return _Violation(
# # #             timestamp        = base.timestamp,
# # #             time_str         = base.time_str,
# # #             frame_index      = base.frame_index,
# # #             type             = base.type,
# # #             events           = list(set(events)),
# # #             severity         = base.severity,
# # #             duration         = base.duration,
# # #             risk_score       = max_risk,
# # #             risk_level       = risk_level,
# # #             confidence       = base.confidence,
# # #             factors          = list(set(factors)),
# # #             source_filename  = base.source_filename,
# # #             local_time_str   = base.local_time_str,
# # #             annotated_frame  = best_frame,
# # #         )

# # #     # ── Private — frame extraction & saving ──────────────────────────────────

# # #     def extract_violation_frames(self, video_path: str) -> None:
# # #         print("[ViolationStore] Saving frames...")
# # #         need_video = [v for v in self._violations if v.annotated_frame is None]
# # #         saved = 0

# # #         # First pass: save violations that already have an annotated frame
# # #         for v in self._violations:
# # #             if v.annotated_frame is not None:
# # #                 v.frame_path      = self._save_frame(v.annotated_frame, v.events, v.time_str)
# # #                 v.annotated_frame = None   # free memory
# # #                 saved += 1

# # #         # Second pass: re-read from the source video for any that are missing
# # #         if need_video:
# # #             cap = cv2.VideoCapture(video_path)
# # #             if not cap.isOpened():
# # #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# # #             else:
# # #                 seen: set = set()
# # #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# # #                     if v.frame_index in seen:
# # #                         continue
# # #                     seen.add(v.frame_index)
# # #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# # #                     ret, frame = cap.read()
# # #                     if not ret:
# # #                         continue
# # #                     v.frame_path = self._save_frame(frame, v.events, v.time_str)
# # #                     saved += 1
# # #                 cap.release()

# # #         print(f"[ViolationStore] {saved} frames saved")

# # #     def _save_frame(
# # #         self,
# # #         frame:    np.ndarray,
# # #         events:   List[str],
# # #         time_str: str,
# # #     ) -> str:
# # #         """
# # #         Save a single violation frame as JPEG.

# # #         Filename format:  <events>_<HH-MM-SS>.jpg
# # #         Example:          seat_absence_00-01-14.jpg
# # #                           seat_absence_drowsy_00-03-02.jpg
# # #                           phone_use_00-00-24.jpg
# # #         """
# # #         distraction   = "_".join(sorted(events))   # sorted for deterministic name
# # #         filename_time = time_str.replace(":", "-")
# # #         filename      = f"{distraction}_{filename_time}.jpg"
# # #         path          = os.path.join(self.frames_dir, filename)
# # #         ok = cv2.imwrite(
# # #             path,
# # #             cv2.resize(frame, (640, 360)),
# # #             [cv2.IMWRITE_JPEG_QUALITY, 85],
# # #         )
# # #         if not ok:
# # #             print(f"[ViolationStore] imwrite failed: {path}")
# # #         return os.path.join(self.analysis_id, "frames", filename)

# # #     # ── Private — report builder ──────────────────────────────────────────────

# # #     def _build_report(self, processing_time: float = 0.0) -> dict:
# # #         return {
# # #             "analysis_id":     self.analysis_id,
# # #             "train_detail_id": self.train_detail_id,
# # #             "processing_time": round(processing_time, 3),
# # #             # Single video → dict (backwards compat); batch → list
# # #             "video_info": (
# # #                 self.video_infos[0]
# # #                 if len(self.video_infos) == 1
# # #                 else self.video_infos
# # #             ),
# # #             "violations": [
# # #                 {
# # #                     "timestamp":   v.time_str,
# # #                     "frame_index": v.frame_index,
# # #                     "events":      v.events,
# # #                     "severity":    v.severity,
# # #                     "duration":    v.duration,
# # #                     "risk_score":  v.risk_score,
# # #                     "risk_level":  v.risk_level,
# # #                     "confidence":  v.confidence,
# # #                     "factors":     v.factors,
# # #                     # "original_video_timestamp" = "<db_filename> <local_HH:MM:SS>"
# # #                     # When timestamp == original (video 1), local_time_str == time_str
# # #                     "original_video_timestamp": (
# # #                         f"{v.source_filename} {v.local_time_str}"
# # #                     ),
# # #                     "frame_path":  v.frame_path,
# # #                 }
# # #                 for v in self._violations
# # #             ],
# # #         }


# # from __future__ import annotations

# # import json
# # import os
# # from dataclasses import dataclass
# # from typing import Any, Dict, List, Optional

# # import cv2
# # import numpy as np

# # OUTPUTS_ROOT = "outputs"
# # MERGE_WINDOW = 2.0


# # # ══════════════════════════════════════════════════════════════════════════════
# # # INTERNAL DATA CLASS
# # # ══════════════════════════════════════════════════════════════════════════════

# # @dataclass
# # class _Violation:
# #     timestamp:       float
# #     time_str:        str
# #     frame_index:     int
# #     type:            str
# #     events:          List[str]
# #     severity:        str
# #     duration:        float
# #     risk_score:      int
# #     risk_level:      str
# #     confidence:      float
# #     factors:         List[str]
# #     source_filename: str                  = ""    # DB filename e.g. "mobile.mp4"
# #     local_time_str:  str                  = ""    # local time within that file e.g. "00:00:23"
# #     frame_path:      Optional[str]        = None
# #     annotated_frame: Optional[np.ndarray] = None


# # # ══════════════════════════════════════════════════════════════════════════════
# # # VIOLATION STORE
# # # ══════════════════════════════════════════════════════════════════════════════

# # class ViolationStore:
# #     """
# #     Accumulates all violations found across one analysis run (single video
# #     or a multi-video batch that shares the same analysis_id / folder_name).

# #     Batch mode usage (api.py)
# #     ─────────────────────────
# #     1. Construct ONCE for the whole folder (no video_info in __init__).
# #     2. Pass as shared_vstore= to each GadgetDetectionPipeline.
# #        The pipeline calls add_video_info() automatically.
# #     3. Call finalize() ONCE after all videos in the folder are done.

# #     Standalone mode usage (CLI / single video)
# #     ──────────────────────────────────────────
# #     1. Construct with video_info= for the single video.
# #     2. Pipeline calls finalize() automatically at the end of run().
# #     """

# #     def __init__(
# #         self,
# #         analysis_id:     str,
# #         train_detail_id: int,
# #         video_info:      Optional[Dict[str, Any]] = None,
# #     ):
# #         self.analysis_id     = analysis_id
# #         self.train_detail_id = train_detail_id
# #         # Always a list — 0 entries until add_video_info() is called (batch mode),
# #         # or 1 entry when video_info is provided (standalone mode).
# #         self.video_infos: List[Dict[str, Any]] = (
# #             [video_info] if video_info is not None else []
# #         )

# #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# #         self.frames_dir = os.path.join(self.output_dir, "frames")
# #         os.makedirs(self.frames_dir, exist_ok=True)

# #         self._violations:  List[_Violation] = []
# #         self._seen_frames: set              = set()
# #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# #     # ── Public helpers ────────────────────────────────────────────────────────

# #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# #         """Append one video's metadata. Called once per video in batch mode."""
# #         self.video_infos.append(video_info)

# #     def record_violation(
# #         self,
# #         annotated_frame:  np.ndarray,
# #         video_time:       float,          # global timestamp (offset-adjusted)
# #         frame_index:      int,            # global frame index (offset-adjusted)
# #         event_type:       str,
# #         original_frame:   Optional[np.ndarray] = None,
# #         severity:         str   = "CRITICAL",
# #         confidence:       float = 0.9,
# #         risk_score:       int   = 80,
# #         risk_level:       str   = "CRITICAL",
# #         factors:          Optional[List[str]] = None,
# #         duration:         float = 0.0,
# #         source_filename:  str   = "",     # DB filename shown in original_video_timestamp
# #         local_video_time: float = -1.0,   # local time within the source file; -1 = same as video_time
# #     ) -> None:
# #         """
# #         Record one distraction event.

# #         Deduplication key is (frame_index, event_type):
# #           • same event on the same global frame is recorded once
# #           • different events on the same global frame are each recorded
# #           • global frame_index is unique across videos (frame_offset applied in main.py)
# #         """
# #         dedup_key = (frame_index, event_type)
# #         if dedup_key in self._seen_frames:
# #             return
# #         self._seen_frames.add(dedup_key)

# #         factors  = factors or []
# #         t        = int(round(video_time))
# #         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

# #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# #         local_str = (
# #             f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
# #         )

# #         self._violations.append(
# #             _Violation(
# #                 timestamp        = video_time,
# #                 time_str         = time_str,
# #                 frame_index      = frame_index,
# #                 type             = event_type,
# #                 events           = [event_type],
# #                 severity         = severity,
# #                 duration         = round(duration, 2),
# #                 risk_score       = risk_score,
# #                 risk_level       = risk_level,
# #                 confidence       = round(confidence, 3),
# #                 factors          = list(factors),
# #                 source_filename  = source_filename,
# #                 local_time_str   = local_str,
# #                 annotated_frame  = (
# #                     annotated_frame.copy() if annotated_frame is not None else None
# #                 ),
# #             )
# #         )

# #     # ── Finalize ──────────────────────────────────────────────────────────────

# #     def finalize(self, processing_time: float = 0.0) -> str:
# #         """
# #         Deduplicate → merge → save frames → write JSON → upload to S3/DB.
# #         Returns the local path to analysis_report.json.
# #         """
# #         self._deduplicate_by_frame()
# #         self._merge_by_time_window()

# #         # Extract frames from every video in the batch (or the single video)
# #         for vi in self.video_infos:
# #             if vi and vi.get("videoPath"):
# #                 self.extract_violation_frames(vi["videoPath"])

# #         report   = self._build_report(processing_time=processing_time)
# #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# #         with open(out_path, "w", encoding="utf-8") as f:
# #             json.dump(report, f, indent=2)

# #         print(f"[ViolationStore] JSON report     : {out_path}")
# #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# #         try:
# #             from utils.db_s3_uploader import finalize_and_upload
# #             finalize_and_upload(
# #                 report_path     = out_path,
# #                 analysis_id     = self.analysis_id,
# #                 train_detail_id = self.train_detail_id,
# #             )
# #         except Exception as exc:
# #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# #         return out_path

# #     # ── Private — deduplication & merging ────────────────────────────────────

# #     def _deduplicate_by_frame(self) -> None:
# #         unique: Dict[int, _Violation] = {}
# #         for v in self._violations:
# #             if v.frame_index not in unique:
# #                 unique[v.frame_index] = v
# #             else:
# #                 ex = unique[v.frame_index]
# #                 ex.events  = list(set(ex.events  + v.events))
# #                 ex.factors = list(set(ex.factors + v.factors))
# #                 if v.risk_score > ex.risk_score:
# #                     ex.risk_score = v.risk_score
# #                     ex.risk_level = v.risk_level
# #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# #                     ex.annotated_frame = v.annotated_frame
# #         self._violations = list(unique.values())

# #     def _merge_by_time_window(self) -> None:
# #         if not self._violations:
# #             return
# #         self._violations.sort(key=lambda x: x.timestamp)
# #         merged: List[_Violation] = []
# #         group  = [self._violations[0]]
# #         for v in self._violations[1:]:
# #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# #                 group.append(v)
# #             else:
# #                 merged.append(self._merge_group(group))
# #                 group = [v]
# #         merged.append(self._merge_group(group))
# #         self._violations = merged

# #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# #         base             = group[0]
# #         events: List[str]  = []
# #         factors: List[str] = []
# #         max_risk   = base.risk_score
# #         risk_level = base.risk_level
# #         best_frame = base.annotated_frame
# #         for v in group:
# #             events.extend(v.events)
# #             factors.extend(v.factors)
# #             if v.risk_score > max_risk:
# #                 max_risk, risk_level = v.risk_score, v.risk_level
# #             if best_frame is None and v.annotated_frame is not None:
# #                 best_frame = v.annotated_frame
# #         return _Violation(
# #             timestamp        = base.timestamp,
# #             time_str         = base.time_str,
# #             frame_index      = base.frame_index,
# #             type             = base.type,
# #             events           = list(set(events)),
# #             severity         = base.severity,
# #             duration         = base.duration,
# #             risk_score       = max_risk,
# #             risk_level       = risk_level,
# #             confidence       = base.confidence,
# #             factors          = list(set(factors)),
# #             source_filename  = base.source_filename,
# #             local_time_str   = base.local_time_str,
# #             annotated_frame  = best_frame,
# #         )

# #     # ── Private — frame extraction & saving ──────────────────────────────────

# #     def extract_violation_frames(self, video_path: str) -> None:
# #         print("[ViolationStore] Saving frames...")
# #         need_video = [v for v in self._violations if v.annotated_frame is None]
# #         saved = 0

# #         # First pass: save violations that already have an annotated frame
# #         for v in self._violations:
# #             if v.annotated_frame is not None:
# #                 v.frame_path      = self._save_frame(v.annotated_frame, v.events, v.time_str)
# #                 v.annotated_frame = None   # free memory
# #                 saved += 1

# #         # Second pass: re-read from the source video for any that are missing
# #         if need_video:
# #             cap = cv2.VideoCapture(video_path)
# #             if not cap.isOpened():
# #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# #             else:
# #                 seen: set = set()
# #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# #                     if v.frame_index in seen:
# #                         continue
# #                     seen.add(v.frame_index)
# #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# #                     ret, frame = cap.read()
# #                     if not ret:
# #                         continue
# #                     v.frame_path = self._save_frame(frame, v.events, v.time_str)
# #                     saved += 1
# #                 cap.release()

# #         print(f"[ViolationStore] {saved} frames saved")

# #     def _save_frame(
# #         self,
# #         frame:    np.ndarray,
# #         events:   List[str],
# #         time_str: str,
# #     ) -> str:
# #         """
# #         Save a single violation frame as JPEG.

# #         Filename format:  <events>_<HH-MM-SS>.jpg
# #         Example:          seat_absence_00-01-14.jpg
# #                           seat_absence_drowsy_00-03-02.jpg
# #                           phone_use_00-00-24.jpg
# #         """
# #         distraction   = "_".join(sorted(events))   # sorted for deterministic name
# #         filename_time = time_str.replace(":", "-")
# #         filename      = f"{distraction}_{filename_time}.jpg"
# #         path          = os.path.join(self.frames_dir, filename)
# #         ok = cv2.imwrite(
# #             path,
# #             cv2.resize(frame, (640, 360)),
# #             [cv2.IMWRITE_JPEG_QUALITY, 85],
# #         )
# #         if not ok:
# #             print(f"[ViolationStore] imwrite failed: {path}")
# #         return os.path.join(self.analysis_id, "frames", filename)

# #     # ── Private — report builder ──────────────────────────────────────────────

# #     def _build_report(self, processing_time: float = 0.0) -> dict:
# #         return {
# #             "analysis_id":     self.analysis_id,
# #             "train_detail_id": self.train_detail_id,
# #             "processing_time": round(processing_time, 3),
# #             # Single video → dict (backwards compat); batch → list
# #             "video_info": (
# #                 self.video_infos[0]
# #                 if len(self.video_infos) == 1
# #                 else self.video_infos
# #             ),
# #             "violations": [
# #                 {
# #                     "timestamp":   v.time_str,
# #                     "frame_index": v.frame_index,
# #                     "events":      v.events,
# #                     "severity":    v.severity,
# #                     "duration":    v.duration,
# #                     "risk_score":  v.risk_score,
# #                     "risk_level":  v.risk_level,
# #                     "confidence":  v.confidence,
# #                     "factors":     v.factors,
# #                     # "original_video_timestamp" = "<db_filename> <local_HH:MM:SS>"
# #                     # When timestamp == original (video 1), local_time_str == time_str
# #                     "original_video_timestamp": (
# #                         f"{v.source_filename} {v.local_time_str}"
# #                     ),
# #                     "frame_path":  v.frame_path,
# #                 }
# #                 for v in self._violations
# #             ],
# #         }


# # # from __future__ import annotations

# # # import json
# # # import os
# # # from dataclasses import dataclass
# # # from typing import Any, Dict, List, Optional

# # # import cv2
# # # import numpy as np

# # # OUTPUTS_ROOT = "outputs"
# # # MERGE_WINDOW = 2.0


# # # @dataclass
# # # class _Violation:
# # #     timestamp:       float
# # #     time_str:        str
# # #     frame_index:     int
# # #     type:            str
# # #     events:          List[str]
# # #     severity:        str
# # #     duration:        float
# # #     risk_score:      int
# # #     risk_level:      str
# # #     confidence:      float
# # #     factors:         List[str]
# # #     source_filename: str                   = ""    # e.g. "ax.mp4"
# # #     local_time_str:  str                   = ""    # local time within that file e.g. "00:00:18"
# # #     frame_path:      Optional[str]         = None
# # #     annotated_frame: Optional[np.ndarray]  = None


# # # class ViolationStore:

# # #     def __init__(self, analysis_id: str, train_detail_id: int, video_info: Optional[Dict[str, Any]] = None):
# # #         self.analysis_id     = analysis_id
# # #         self.train_detail_id = train_detail_id
# # #         # video_infos is always a list — 1 entry for single-video, N entries for batch
# # #         self.video_infos: List[Dict[str, Any]] = [video_info] if video_info is not None else []

# # #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# # #         self.frames_dir = os.path.join(self.output_dir, "frames")
# # #         os.makedirs(self.frames_dir, exist_ok=True)

# # #         self._violations: List[_Violation] = []
# # #         self._seen_frames: set             = set()
# # #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# # #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# # #         """
# # #         Register one video's metadata into the shared store.
# # #         Called once per video in batch mode (api.py passes shared_vstore).
# # #         """
# # #         self.video_infos.append(video_info)

# # #     def record_violation(
# # #         self,
# # #         annotated_frame: np.ndarray,
# # #         video_time:      float,
# # #         frame_index:     int,
# # #         event_type:      str,
# # #         original_frame:  Optional[np.ndarray] = None,
# # #         severity:        str   = "CRITICAL",
# # #         confidence:      float = 0.9,
# # #         risk_score:      int   = 80,
# # #         risk_level:      str   = "CRITICAL",
# # #         factors:         Optional[List[str]] = None,
# # #         duration:        float = 0.0,
# # #         source_filename: str   = "",   # original upload filename e.g. "ax.mp4"
# # #         local_video_time: float = -1.0, # raw video_time before offset; -1 = same as video_time
# # #     ):
# # #         # Deduplicate on (frame_index, event_type) so that:
# # #         #  • the same violation type on the same frame is recorded only once
# # #         #  • different violation types on the same frame are each recorded
# # #         #  • frame numbers from different videos never collide (frame_offset
# # #         #    in main.py makes every global frame_index unique across the batch)
# # #         dedup_key = (frame_index, event_type)
# # #         if dedup_key in self._seen_frames:
# # #             return
# # #         self._seen_frames.add(dedup_key)
# # #         factors   = factors or []
# # #         t         = int(round(video_time))
# # #         time_str  = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"
# # #         # Build the per-file local timestamp (time within the source video)
# # #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# # #         local_str = f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
# # #         self._violations.append(_Violation(
# # #             timestamp        = video_time,
# # #             time_str         = time_str,
# # #             frame_index      = frame_index,
# # #             type             = event_type,
# # #             events           = [event_type],
# # #             severity         = severity,
# # #             duration         = round(duration, 2),
# # #             risk_score       = risk_score,
# # #             risk_level       = risk_level,
# # #             confidence       = round(confidence, 3),
# # #             factors          = list(factors),
# # #             source_filename  = source_filename,
# # #             local_time_str   = local_str,
# # #             annotated_frame  = annotated_frame.copy() if annotated_frame is not None else None,
# # #         ))

# # #     def _deduplicate_by_frame(self):
# # #         unique: Dict[int, _Violation] = {}
# # #         for v in self._violations:
# # #             if v.frame_index not in unique:
# # #                 unique[v.frame_index] = v
# # #             else:
# # #                 ex = unique[v.frame_index]
# # #                 ex.events  = list(set(ex.events  + v.events))
# # #                 ex.factors = list(set(ex.factors + v.factors))
# # #                 if v.risk_score > ex.risk_score:
# # #                     ex.risk_score = v.risk_score
# # #                     ex.risk_level = v.risk_level
# # #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# # #                     ex.annotated_frame = v.annotated_frame
# # #         self._violations = list(unique.values())

# # #     def _merge_by_time_window(self):
# # #         if not self._violations:
# # #             return
# # #         self._violations.sort(key=lambda x: x.timestamp)
# # #         merged = []
# # #         group  = [self._violations[0]]
# # #         for v in self._violations[1:]:
# # #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# # #                 group.append(v)
# # #             else:
# # #                 merged.append(self._merge_group(group))
# # #                 group = [v]
# # #         merged.append(self._merge_group(group))
# # #         self._violations = merged

# # #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# # #         base            = group[0]
# # #         events, factors = [], []
# # #         max_risk        = base.risk_score
# # #         risk_level      = base.risk_level
# # #         best_frame      = base.annotated_frame
# # #         for v in group:
# # #             events.extend(v.events)
# # #             factors.extend(v.factors)
# # #             if v.risk_score > max_risk:
# # #                 max_risk, risk_level = v.risk_score, v.risk_level
# # #             if best_frame is None and v.annotated_frame is not None:
# # #                 best_frame = v.annotated_frame
# # #         return _Violation(
# # #             timestamp        = base.timestamp,
# # #             time_str         = base.time_str,
# # #             frame_index      = base.frame_index,
# # #             type             = base.type,
# # #             events           = list(set(events)),
# # #             severity         = base.severity,
# # #             duration         = base.duration,
# # #             risk_score       = max_risk,
# # #             risk_level       = risk_level,
# # #             confidence       = base.confidence,
# # #             factors          = list(set(factors)),
# # #             source_filename  = base.source_filename,
# # #             local_time_str   = base.local_time_str,
# # #             annotated_frame  = best_frame,
# # #         )

# # #     def extract_violation_frames(self, video_path: str):
# # #         print("[ViolationStore] Saving frames...")
# # #         need_video = [v for v in self._violations if v.annotated_frame is None]
# # #         saved = 0
# # #         for v in self._violations:
# # #             if v.annotated_frame is not None:
# # #                 v.frame_path      = self._save_frame(v.annotated_frame, v.events, v.time_str)
# # #                 v.annotated_frame = None
# # #                 saved += 1
# # #         if need_video:
# # #             cap = cv2.VideoCapture(video_path)
# # #             if not cap.isOpened():
# # #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# # #             else:
# # #                 seen: set = set()
# # #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# # #                     if v.frame_index in seen:
# # #                         continue
# # #                     seen.add(v.frame_index)
# # #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# # #                     ret, frame = cap.read()
# # #                     if not ret:
# # #                         continue
# # #                     v.frame_path = self._save_frame(frame, v.events, v.time_str)
# # #                     saved += 1
# # #                 cap.release()
# # #         print(f"[ViolationStore] {saved} frames saved")

# # #     def _save_frame(self, frame: np.ndarray, events: List[str], time_str: str) -> str:
# # #         distraction   = "_".join(events)
# # #         filename_time = time_str.replace(":", "-")
# # #         filename      = f"{distraction}_{filename_time}.jpg"
# # #         path          = os.path.join(self.frames_dir, filename)
# # #         ok = cv2.imwrite(path, cv2.resize(frame, (640, 360)),
# # #                          [cv2.IMWRITE_JPEG_QUALITY, 85])
# # #         if not ok:
# # #             print(f"[ViolationStore] imwrite failed: {path}")
# # #         return os.path.join(self.analysis_id, "frames", filename)

# # #     def _build_report(self, processing_time: float = 0.0) -> dict:
# # #         return {
# # #             "analysis_id":     self.analysis_id,
# # #             "train_detail_id": self.train_detail_id,
# # #             "processing_time": round(processing_time, 3),
# # #             # Single video → keep as dict for backwards compat; batch → list
# # #             "video_info": self.video_infos[0] if len(self.video_infos) == 1
# # #                           else self.video_infos,
# # #             "violations": [
# # #                 {
# # #                     "timestamp":   v.time_str,
# # #                     "frame_index": v.frame_index,
# # #                     "events":      v.events,
# # #                     "severity":    v.severity,
# # #                     "duration":    v.duration,
# # #                     "risk_score":  v.risk_score,
# # #                     "risk_level":  v.risk_level,
# # #                     "confidence":  v.confidence,
# # #                     "factors":     v.factors,
# # #                     "original_video_timestamp": f"{v.source_filename} {v.local_time_str}",
# # #                     "frame_path":  v.frame_path,
# # #                 }
# # #                 for v in self._violations
# # #             ],
# # #         }

# # #     def finalize(self, processing_time: float = 0.0) -> str:
# # #         self._deduplicate_by_frame()
# # #         self._merge_by_time_window()
# # #         # Extract frames from every video in the batch (or the single video)
# # #         for vi in self.video_infos:
# # #             if vi and vi.get("videoPath"):
# # #                 self.extract_violation_frames(vi["videoPath"])
# # #         report   = self._build_report(processing_time=processing_time)
# # #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# # #         with open(out_path, "w", encoding="utf-8") as f:
# # #             json.dump(report, f, indent=2)
# # #         print(f"[ViolationStore] JSON report     : {out_path}")
# # #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# # #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# # #         try:
# # #             from utils.db_s3_uploader import finalize_and_upload
# # #             finalize_and_upload(
# # #                 report_path     = out_path,
# # #                 analysis_id     = self.analysis_id,
# # #                 train_detail_id = self.train_detail_id,
# # #             )
# # #         except Exception as exc:
# # #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# # #         return out_path



# # # utils/violation_store.py
# # # ─────────────────────────────────────────────────────────────────
# # # Change from original:
# # #   _save_frame() now appends _f{frame_index} to the filename so
# # #   two violations of the same type at the same timestamp never
# # #   silently overwrite each other.
# # #   e.g.  phone_use_00-13-16_f6762.jpg
# # #         seat_absence_00-00-17_f516.jpg
# # # ─────────────────────────────────────────────────────────────────

# # from __future__ import annotations

# # import json
# # import os
# # from dataclasses import dataclass
# # from typing import Any, Dict, List, Optional

# # import cv2
# # import numpy as np

# # OUTPUTS_ROOT = "outputs"
# # MERGE_WINDOW = 2.0


# # @dataclass
# # class _Violation:
# #     timestamp:       float
# #     time_str:        str
# #     frame_index:     int
# #     type:            str
# #     events:          List[str]
# #     severity:        str
# #     duration:        float
# #     risk_score:      int
# #     risk_level:      str
# #     confidence:      float
# #     factors:         List[str]
# #     source_filename: str                  = ""
# #     local_time_str:  str                  = ""
# #     frame_path:      Optional[str]        = None
# #     annotated_frame: Optional[np.ndarray] = None


# # class ViolationStore:

# #     def __init__(
# #         self,
# #         analysis_id:     str,
# #         train_detail_id: int,
# #         video_info:      Optional[Dict[str, Any]] = None,
# #     ):
# #         self.analysis_id     = analysis_id
# #         self.train_detail_id = train_detail_id
# #         self.video_infos: List[Dict[str, Any]] = (
# #             [video_info] if video_info is not None else []
# #         )

# #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# #         self.frames_dir = os.path.join(self.output_dir, "frames")
# #         os.makedirs(self.frames_dir, exist_ok=True)

# #         self._violations: List[_Violation] = []
# #         self._seen_frames: set             = set()
# #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# #     # ── called once per video in batch mode ──────────────────────
# #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# #         self.video_infos.append(video_info)

# #     # ── called from main.py for every detected violation ─────────
# #     def record_violation(
# #         self,
# #         annotated_frame:  np.ndarray,
# #         video_time:       float,
# #         frame_index:      int,
# #         event_type:       str,
# #         original_frame:   Optional[np.ndarray] = None,
# #         severity:         str   = "CRITICAL",
# #         confidence:       float = 0.9,
# #         risk_score:       int   = 80,
# #         risk_level:       str   = "CRITICAL",
# #         factors:          Optional[List[str]] = None,
# #         duration:         float = 0.0,
# #         source_filename:  str   = "",
# #         local_video_time: float = -1.0,
# #     ):
# #         # Deduplicate: same frame + same event type recorded only once.
# #         # Different event types on the same frame are each recorded.
# #         # frame_index is globally unique across batch (frame_offset in main.py).
# #         dedup_key = (frame_index, event_type)
# #         if dedup_key in self._seen_frames:
# #             return
# #         self._seen_frames.add(dedup_key)

# #         factors  = factors or []
# #         t        = int(round(video_time))
# #         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

# #         # Local timestamp = time within the source video file (before cumulative offset)
# #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# #         local_str = f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"

# #         self._violations.append(_Violation(
# #             timestamp        = video_time,
# #             time_str         = time_str,
# #             frame_index      = frame_index,
# #             type             = event_type,
# #             events           = [event_type],
# #             severity         = severity,
# #             duration         = round(duration, 2),
# #             risk_score       = risk_score,
# #             risk_level       = risk_level,
# #             confidence       = round(confidence, 3),
# #             factors          = list(factors),
# #             source_filename  = source_filename,
# #             local_time_str   = local_str,
# #             annotated_frame  = annotated_frame.copy() if annotated_frame is not None else None,
# #         ))

# #     # ─────────────────────────────────────────────────────────────

# #     def _deduplicate_by_frame(self):
# #         unique: Dict[int, _Violation] = {}
# #         for v in self._violations:
# #             if v.frame_index not in unique:
# #                 unique[v.frame_index] = v
# #             else:
# #                 ex = unique[v.frame_index]
# #                 ex.events  = list(set(ex.events  + v.events))
# #                 ex.factors = list(set(ex.factors + v.factors))
# #                 if v.risk_score > ex.risk_score:
# #                     ex.risk_score = v.risk_score
# #                     ex.risk_level = v.risk_level
# #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# #                     ex.annotated_frame = v.annotated_frame
# #         self._violations = list(unique.values())

# #     def _merge_by_time_window(self):
# #         if not self._violations:
# #             return
# #         self._violations.sort(key=lambda x: x.timestamp)
# #         merged = []
# #         group  = [self._violations[0]]
# #         for v in self._violations[1:]:
# #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# #                 group.append(v)
# #             else:
# #                 merged.append(self._merge_group(group))
# #                 group = [v]
# #         merged.append(self._merge_group(group))
# #         self._violations = merged

# #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# #         base            = group[0]
# #         events, factors = [], []
# #         max_risk        = base.risk_score
# #         risk_level      = base.risk_level
# #         best_frame      = base.annotated_frame
# #         for v in group:
# #             events.extend(v.events)
# #             factors.extend(v.factors)
# #             if v.risk_score > max_risk:
# #                 max_risk, risk_level = v.risk_score, v.risk_level
# #             if best_frame is None and v.annotated_frame is not None:
# #                 best_frame = v.annotated_frame
# #         return _Violation(
# #             timestamp        = base.timestamp,
# #             time_str         = base.time_str,
# #             frame_index      = base.frame_index,
# #             type             = base.type,
# #             events           = list(set(events)),
# #             severity         = base.severity,
# #             duration         = base.duration,
# #             risk_score       = max_risk,
# #             risk_level       = risk_level,
# #             confidence       = base.confidence,
# #             factors          = list(set(factors)),
# #             source_filename  = base.source_filename,
# #             local_time_str   = base.local_time_str,
# #             annotated_frame  = best_frame,
# #         )

# #     # ─────────────────────────────────────────────────────────────

# #     def extract_violation_frames(self, video_path: str):
# #         print("[ViolationStore] Saving frames...")
# #         need_video = [v for v in self._violations if v.annotated_frame is None]
# #         saved = 0

# #         for v in self._violations:
# #             if v.annotated_frame is not None:
# #                 v.frame_path      = self._save_frame(
# #                     v.annotated_frame, v.events, v.time_str, v.frame_index
# #                 )
# #                 v.annotated_frame = None
# #                 saved += 1

# #         if need_video:
# #             cap = cv2.VideoCapture(video_path)
# #             if not cap.isOpened():
# #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# #             else:
# #                 seen: set = set()
# #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# #                     if v.frame_index in seen:
# #                         continue
# #                     seen.add(v.frame_index)
# #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# #                     ret, frame = cap.read()
# #                     if not ret:
# #                         continue
# #                     v.frame_path = self._save_frame(
# #                         frame, v.events, v.time_str, v.frame_index
# #                     )
# #                     saved += 1
# #                 cap.release()

# #         print(f"[ViolationStore] {saved} frames saved")

# #     def _save_frame(
# #         self,
# #         frame:       np.ndarray,
# #         events:      List[str],
# #         time_str:    str,
# #         frame_index: int,            # ← ADDED: makes filename globally unique
# #     ) -> str:
# #         """
# #         Save one violation frame as JPEG.

# #         Filename: {event_types}_{hh-mm-ss}_f{frame_index}.jpg
# #         Example:  phone_use_00-13-16_f6762.jpg
# #                   seat_absence_00-00-17_f516.jpg

# #         frame_index prevents two violations of the same type at the
# #         same timestamp from silently overwriting each other.
# #         """
# #         distraction   = "_".join(events)
# #         filename_time = time_str.replace(":", "-")
# #         filename      = f"{distraction}_{filename_time}_f{frame_index}.jpg"  # ← CHANGED
# #         path          = os.path.join(self.frames_dir, filename)
# #         ok = cv2.imwrite(
# #             path,
# #             cv2.resize(frame, (640, 360)),
# #             [cv2.IMWRITE_JPEG_QUALITY, 85],
# #         )
# #         if not ok:
# #             print(f"[ViolationStore] imwrite failed: {path}")
# #         return os.path.join(self.analysis_id, "frames", filename)

# #     # ─────────────────────────────────────────────────────────────

# #     def _build_report(self, processing_time: float = 0.0) -> dict:
# #         return {
# #             "analysis_id":     self.analysis_id,
# #             "train_detail_id": self.train_detail_id,
# #             "processing_time": round(processing_time, 3),
# #             "video_info": (
# #                 self.video_infos[0] if len(self.video_infos) == 1
# #                 else self.video_infos
# #             ),
# #             "violations": [
# #                 {
# #                     "timestamp":   v.time_str,
# #                     "frame_index": v.frame_index,
# #                     "events":      v.events,
# #                     "severity":    v.severity,
# #                     "duration":    v.duration,
# #                     "risk_score":  v.risk_score,
# #                     "risk_level":  v.risk_level,
# #                     "confidence":  v.confidence,
# #                     "factors":     v.factors,
# #                     "original_video_timestamp": f"{v.source_filename} {v.local_time_str}",
# #                     "frame_path":  v.frame_path,
# #                 }
# #                 for v in self._violations
# #             ],
# #         }

# #     def finalize(self, processing_time: float = 0.0) -> str:
# #         self._deduplicate_by_frame()
# #         self._merge_by_time_window()

# #         # Save frames — each video's temp path is in video_infos
# #         for vi in self.video_infos:
# #             if vi and vi.get("videoPath"):
# #                 self.extract_violation_frames(vi["videoPath"])

# #         report   = self._build_report(processing_time=processing_time)
# #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# #         with open(out_path, "w", encoding="utf-8") as f:
# #             json.dump(report, f, indent=2)

# #         print(f"[ViolationStore] JSON report     : {out_path}")
# #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# #         # Upload results to S3 and update DB result_s3_path
# #         try:
# #             from utils.db_s3_uploader import finalize_and_upload
# #             finalize_and_upload(
# #                 report_path     = out_path,
# #                 analysis_id     = self.analysis_id,
# #                 train_detail_id = self.train_detail_id,
# #             )
# #         except Exception as exc:
# #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# #         return out_path

# # # # from __future__ import annotations

# # # # import json
# # # # import os
# # # # from dataclasses import dataclass
# # # # from typing import Any, Dict, List, Optional

# # # # import cv2
# # # # import numpy as np

# # # # OUTPUTS_ROOT = "outputs"
# # # # MERGE_WINDOW = 2.0


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # INTERNAL DATA CLASS
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # @dataclass
# # # # class _Violation:
# # # #     timestamp:       float
# # # #     time_str:        str
# # # #     frame_index:     int
# # # #     type:            str
# # # #     events:          List[str]
# # # #     severity:        str
# # # #     duration:        float
# # # #     risk_score:      int
# # # #     risk_level:      str
# # # #     confidence:      float
# # # #     factors:         List[str]
# # # #     source_filename: str                  = ""    # e.g. "ch01.mp4"
# # # #     local_time_str:  str                  = ""    # local time within that file e.g. "00:00:18"
# # # #     frame_path:      Optional[str]        = None
# # # #     annotated_frame: Optional[np.ndarray] = None


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # VIOLATION STORE
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # class ViolationStore:
# # # #     """
# # # #     Accumulates all violations found across one analysis run (single video
# # # #     or a multi-video batch that shares the same analysis_id / folder_name).

# # # #     Usage
# # # #     ─────
# # # #     1. Construct once per analysis run.
# # # #     2. Call record_violation() from the pipeline for every distraction event.
# # # #     3. Call finalize() at the very end — it deduplicates, merges, saves frames,
# # # #        writes the JSON report, then calls db_s3_uploader.finalize_and_upload()
# # # #        to push everything to S3 and record results in the DB.
# # # #     """

# # # #     def __init__(
# # # #         self,
# # # #         analysis_id:     str,
# # # #         train_detail_id: int,
# # # #         video_info:      Optional[Dict[str, Any]] = None,
# # # #     ):
# # # #         self.analysis_id     = analysis_id
# # # #         self.train_detail_id = train_detail_id
# # # #         # video_infos is always a list:
# # # #         #   • 1 entry for single-video runs
# # # #         #   • N entries for batch runs (add_video_info called per video)
# # # #         self.video_infos: List[Dict[str, Any]] = (
# # # #             [video_info] if video_info is not None else []
# # # #         )

# # # #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# # # #         self.frames_dir = os.path.join(self.output_dir, "frames")
# # # #         os.makedirs(self.frames_dir, exist_ok=True)

# # # #         self._violations:  List[_Violation] = []
# # # #         self._seen_frames: set              = set()
# # # #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# # # #     # ── Public helpers ────────────────────────────────────────────────────────

# # # #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# # # #         """
# # # #         Register one video's metadata into the shared store.
# # # #         Called once per video in batch mode.
# # # #         """
# # # #         self.video_infos.append(video_info)

# # # #     def record_violation(
# # # #         self,
# # # #         annotated_frame:  np.ndarray,
# # # #         video_time:       float,
# # # #         frame_index:      int,
# # # #         event_type:       str,
# # # #         original_frame:   Optional[np.ndarray] = None,
# # # #         severity:         str   = "CRITICAL",
# # # #         confidence:       float = 0.9,
# # # #         risk_score:       int   = 80,
# # # #         risk_level:       str   = "CRITICAL",
# # # #         factors:          Optional[List[str]] = None,
# # # #         duration:         float = 0.0,
# # # #         source_filename:  str   = "",    # original filename e.g. "ch01.mp4"
# # # #         local_video_time: float = -1.0,  # video_time before global offset; -1 = same as video_time
# # # #     ) -> None:
# # # #         """
# # # #         Record one distraction event.

# # # #         Deduplication key is (frame_index, event_type) so that:
# # # #           • the same violation type on the same frame is recorded only once
# # # #           • different violation types on the same frame are each recorded
# # # #           • frame numbers from different videos never collide because
# # # #             main.py applies a frame_offset to make every global frame_index
# # # #             unique across the batch
# # # #         """
# # # #         dedup_key = (frame_index, event_type)
# # # #         if dedup_key in self._seen_frames:
# # # #             return
# # # #         self._seen_frames.add(dedup_key)

# # # #         factors  = factors or []
# # # #         t        = int(round(video_time))
# # # #         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

# # # #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# # # #         local_str = (
# # # #             f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
# # # #         )

# # # #         self._violations.append(
# # # #             _Violation(
# # # #                 timestamp        = video_time,
# # # #                 time_str         = time_str,
# # # #                 frame_index      = frame_index,
# # # #                 type             = event_type,
# # # #                 events           = [event_type],
# # # #                 severity         = severity,
# # # #                 duration         = round(duration, 2),
# # # #                 risk_score       = risk_score,
# # # #                 risk_level       = risk_level,
# # # #                 confidence       = round(confidence, 3),
# # # #                 factors          = list(factors),
# # # #                 source_filename  = source_filename,
# # # #                 local_time_str   = local_str,
# # # #                 annotated_frame  = (
# # # #                     annotated_frame.copy() if annotated_frame is not None else None
# # # #                 ),
# # # #             )
# # # #         )

# # # #     # ── Finalize ──────────────────────────────────────────────────────────────

# # # #     def finalize(self, processing_time: float = 0.0) -> str:
# # # #         """
# # # #         1. Deduplicate violations that share the same frame.
# # # #         2. Merge violations that are within MERGE_WINDOW seconds of each other.
# # # #         3. Save annotated frame images to disk (falls back to re-reading from
# # # #            the source video when no annotated frame was captured).
# # # #         4. Write analysis_report.json.
# # # #         5. Upload frames + report to S3 and record in DB via db_s3_uploader.
# # # #         Returns the local path to analysis_report.json.
# # # #         """
# # # #         self._deduplicate_by_frame()
# # # #         self._merge_by_time_window()

# # # #         # Extract frames from every video in the batch (or the single video)
# # # #         for vi in self.video_infos:
# # # #             if vi and vi.get("videoPath"):
# # # #                 self.extract_violation_frames(vi["videoPath"])

# # # #         report   = self._build_report(processing_time=processing_time)
# # # #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# # # #         with open(out_path, "w", encoding="utf-8") as f:
# # # #             json.dump(report, f, indent=2)

# # # #         print(f"[ViolationStore] JSON report     : {out_path}")
# # # #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# # # #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# # # #         try:
# # # #             from utils.db_s3_uploader import finalize_and_upload
# # # #             finalize_and_upload(
# # # #                 report_path     = out_path,
# # # #                 analysis_id     = self.analysis_id,
# # # #                 train_detail_id = self.train_detail_id,
# # # #             )
# # # #         except Exception as exc:
# # # #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# # # #         return out_path

# # # #     # ── Private — deduplication & merging ────────────────────────────────────

# # # #     def _deduplicate_by_frame(self) -> None:
# # # #         unique: Dict[int, _Violation] = {}
# # # #         for v in self._violations:
# # # #             if v.frame_index not in unique:
# # # #                 unique[v.frame_index] = v
# # # #             else:
# # # #                 ex = unique[v.frame_index]
# # # #                 ex.events  = list(set(ex.events  + v.events))
# # # #                 ex.factors = list(set(ex.factors + v.factors))
# # # #                 if v.risk_score > ex.risk_score:
# # # #                     ex.risk_score = v.risk_score
# # # #                     ex.risk_level = v.risk_level
# # # #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# # # #                     ex.annotated_frame = v.annotated_frame
# # # #         self._violations = list(unique.values())

# # # #     def _merge_by_time_window(self) -> None:
# # # #         if not self._violations:
# # # #             return
# # # #         self._violations.sort(key=lambda x: x.timestamp)
# # # #         merged: List[_Violation] = []
# # # #         group  = [self._violations[0]]
# # # #         for v in self._violations[1:]:
# # # #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# # # #                 group.append(v)
# # # #             else:
# # # #                 merged.append(self._merge_group(group))
# # # #                 group = [v]
# # # #         merged.append(self._merge_group(group))
# # # #         self._violations = merged

# # # #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# # # #         base            = group[0]
# # # #         events: List[str]  = []
# # # #         factors: List[str] = []
# # # #         max_risk   = base.risk_score
# # # #         risk_level = base.risk_level
# # # #         best_frame = base.annotated_frame
# # # #         for v in group:
# # # #             events.extend(v.events)
# # # #             factors.extend(v.factors)
# # # #             if v.risk_score > max_risk:
# # # #                 max_risk, risk_level = v.risk_score, v.risk_level
# # # #             if best_frame is None and v.annotated_frame is not None:
# # # #                 best_frame = v.annotated_frame
# # # #         return _Violation(
# # # #             timestamp        = base.timestamp,
# # # #             time_str         = base.time_str,
# # # #             frame_index      = base.frame_index,
# # # #             type             = base.type,
# # # #             events           = list(set(events)),
# # # #             severity         = base.severity,
# # # #             duration         = base.duration,
# # # #             risk_score       = max_risk,
# # # #             risk_level       = risk_level,
# # # #             confidence       = base.confidence,
# # # #             factors          = list(set(factors)),
# # # #             source_filename  = base.source_filename,
# # # #             local_time_str   = base.local_time_str,
# # # #             annotated_frame  = best_frame,
# # # #         )

# # # #     # ── Private — frame extraction & saving ──────────────────────────────────

# # # #     def extract_violation_frames(self, video_path: str) -> None:
# # # #         print("[ViolationStore] Saving frames...")
# # # #         need_video = [v for v in self._violations if v.annotated_frame is None]
# # # #         saved = 0

# # # #         # First pass: save violations that already have an annotated frame
# # # #         for v in self._violations:
# # # #             if v.annotated_frame is not None:
# # # #                 v.frame_path      = self._save_frame(v.annotated_frame, v.events, v.time_str)
# # # #                 v.annotated_frame = None   # free memory
# # # #                 saved += 1

# # # #         # Second pass: re-read from the source video for any that are missing
# # # #         if need_video:
# # # #             cap = cv2.VideoCapture(video_path)
# # # #             if not cap.isOpened():
# # # #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# # # #             else:
# # # #                 seen: set = set()
# # # #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# # # #                     if v.frame_index in seen:
# # # #                         continue
# # # #                     seen.add(v.frame_index)
# # # #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# # # #                     ret, frame = cap.read()
# # # #                     if not ret:
# # # #                         continue
# # # #                     v.frame_path = self._save_frame(frame, v.events, v.time_str)
# # # #                     saved += 1
# # # #                 cap.release()

# # # #         print(f"[ViolationStore] {saved} frames saved")

# # # #     def _save_frame(
# # # #         self,
# # # #         frame:    np.ndarray,
# # # #         events:   List[str],
# # # #         time_str: str,
# # # #     ) -> str:
# # # #         distraction   = "_".join(events)
# # # #         filename_time = time_str.replace(":", "-")
# # # #         filename      = f"{distraction}_{filename_time}.jpg"
# # # #         path          = os.path.join(self.frames_dir, filename)
# # # #         ok = cv2.imwrite(
# # # #             path,
# # # #             cv2.resize(frame, (640, 360)),
# # # #             [cv2.IMWRITE_JPEG_QUALITY, 85],
# # # #         )
# # # #         if not ok:
# # # #             print(f"[ViolationStore] imwrite failed: {path}")
# # # #         return os.path.join(self.analysis_id, "frames", filename)

# # # #     # ── Private — report builder ──────────────────────────────────────────────

# # # #     def _build_report(self, processing_time: float = 0.0) -> dict:
# # # #         return {
# # # #             "analysis_id":     self.analysis_id,
# # # #             "train_detail_id": self.train_detail_id,
# # # #             "processing_time": round(processing_time, 3),
# # # #             # Single video → keep as dict for backwards compat; batch → list
# # # #             "video_info": (
# # # #                 self.video_infos[0]
# # # #                 if len(self.video_infos) == 1
# # # #                 else self.video_infos
# # # #             ),
# # # #             "violations": [
# # # #                 {
# # # #                     "timestamp":   v.time_str,
# # # #                     "frame_index": v.frame_index,
# # # #                     "events":      v.events,
# # # #                     "severity":    v.severity,
# # # #                     "duration":    v.duration,
# # # #                     "risk_score":  v.risk_score,
# # # #                     "risk_level":  v.risk_level,
# # # #                     "confidence":  v.confidence,
# # # #                     "factors":     v.factors,
# # # #                     "original_video_timestamp": (
# # # #                         f"{v.source_filename} {v.local_time_str}"
# # # #                     ),
# # # #                     "frame_path":  v.frame_path,
# # # #                 }
# # # #                 for v in self._violations
# # # #             ],
# # # #         }

# # # from __future__ import annotations

# # # import json
# # # import os
# # # from dataclasses import dataclass
# # # from typing import Any, Dict, List, Optional

# # # import cv2
# # # import numpy as np

# # # OUTPUTS_ROOT = "outputs"
# # # MERGE_WINDOW = 2.0


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # # INTERNAL DATA CLASS
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # @dataclass
# # # class _Violation:
# # #     timestamp:       float
# # #     time_str:        str
# # #     frame_index:     int
# # #     type:            str
# # #     events:          List[str]
# # #     severity:        str
# # #     duration:        float
# # #     risk_score:      int
# # #     risk_level:      str
# # #     confidence:      float
# # #     factors:         List[str]
# # #     source_filename: str                  = ""    # e.g. "ch01.mp4"
# # #     local_time_str:  str                  = ""    # local time within that file e.g. "00:00:18"
# # #     frame_path:      Optional[str]        = None
# # #     annotated_frame: Optional[np.ndarray] = None


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # # VIOLATION STORE
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # class ViolationStore:
# # #     """
# # #     Accumulates all violations found across one analysis run (single video
# # #     or a multi-video batch that shares the same analysis_id / folder_name).

# # #     Usage
# # #     ─────
# # #     1. Construct once per analysis run.
# # #     2. Call record_violation() from the pipeline for every distraction event.
# # #     3. Call finalize() at the very end — it deduplicates, merges, saves frames,
# # #        writes the JSON report, then calls db_s3_uploader.finalize_and_upload()
# # #        to push everything to S3 and record results in the DB.
# # #     """

# # #     def __init__(
# # #         self,
# # #         analysis_id:     str,
# # #         train_detail_id: int,
# # #         video_info:      Optional[Dict[str, Any]] = None,
# # #     ):
# # #         self.analysis_id     = analysis_id
# # #         self.train_detail_id = train_detail_id
# # #         # video_infos is always a list:
# # #         #   • 1 entry for single-video runs
# # #         #   • N entries for batch runs (add_video_info called per video)
# # #         self.video_infos: List[Dict[str, Any]] = (
# # #             [video_info] if video_info is not None else []
# # #         )

# # #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# # #         self.frames_dir = os.path.join(self.output_dir, "frames")
# # #         os.makedirs(self.frames_dir, exist_ok=True)

# # #         self._violations:  List[_Violation] = []
# # #         self._seen_frames: set              = set()
# # #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# # #     # ── Public helpers ────────────────────────────────────────────────────────

# # #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# # #         """
# # #         Register one video's metadata into the shared store.
# # #         Called once per video in batch mode.
# # #         """
# # #         self.video_infos.append(video_info)

# # #     def record_violation(
# # #         self,
# # #         annotated_frame:  np.ndarray,
# # #         video_time:       float,
# # #         frame_index:      int,
# # #         event_type:       str,
# # #         original_frame:   Optional[np.ndarray] = None,
# # #         severity:         str   = "CRITICAL",
# # #         confidence:       float = 0.9,
# # #         risk_score:       int   = 80,
# # #         risk_level:       str   = "CRITICAL",
# # #         factors:          Optional[List[str]] = None,
# # #         duration:         float = 0.0,
# # #         source_filename:  str   = "",    # original filename e.g. "ch01.mp4"
# # #         local_video_time: float = -1.0,  # video_time before global offset; -1 = same as video_time
# # #     ) -> None:
# # #         """
# # #         Record one distraction event.

# # #         Deduplication key is (frame_index, event_type) so that:
# # #           • the same violation type on the same frame is recorded only once
# # #           • different violation types on the same frame are each recorded
# # #           • frame numbers from different videos never collide because
# # #             main.py applies a frame_offset to make every global frame_index
# # #             unique across the batch
# # #         """
# # #         dedup_key = (frame_index, event_type)
# # #         if dedup_key in self._seen_frames:
# # #             return
# # #         self._seen_frames.add(dedup_key)

# # #         factors  = factors or []
# # #         t        = int(round(video_time))
# # #         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

# # #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# # #         local_str = (
# # #             f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
# # #         )

# # #         self._violations.append(
# # #             _Violation(
# # #                 timestamp        = video_time,
# # #                 time_str         = time_str,
# # #                 frame_index      = frame_index,
# # #                 type             = event_type,
# # #                 events           = [event_type],
# # #                 severity         = severity,
# # #                 duration         = round(duration, 2),
# # #                 risk_score       = risk_score,
# # #                 risk_level       = risk_level,
# # #                 confidence       = round(confidence, 3),
# # #                 factors          = list(factors),
# # #                 source_filename  = source_filename,
# # #                 local_time_str   = local_str,
# # #                 annotated_frame  = (
# # #                     annotated_frame.copy() if annotated_frame is not None else None
# # #                 ),
# # #             )
# # #         )

# # #     # ── Finalize ──────────────────────────────────────────────────────────────

# # #     def finalize(self, processing_time: float = 0.0) -> str:
# # #         """
# # #         1. Deduplicate violations that share the same frame.
# # #         2. Merge violations that are within MERGE_WINDOW seconds of each other.
# # #         3. Save annotated frame images to disk (falls back to re-reading from
# # #            the source video when no annotated frame was captured).
# # #         4. Write analysis_report.json.
# # #         5. Upload frames + report to S3 and record in DB via db_s3_uploader.
# # #         Returns the local path to analysis_report.json.
# # #         """
# # #         self._deduplicate_by_frame()
# # #         self._merge_by_time_window()

# # #         # Extract frames from every video in the batch (or the single video)
# # #         for vi in self.video_infos:
# # #             if vi and vi.get("videoPath"):
# # #                 self.extract_violation_frames(vi["videoPath"])

# # #         report   = self._build_report(processing_time=processing_time)
# # #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# # #         with open(out_path, "w", encoding="utf-8") as f:
# # #             json.dump(report, f, indent=2)

# # #         print(f"[ViolationStore] JSON report     : {out_path}")
# # #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# # #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# # #         try:
# # #             from utils.db_s3_uploader import finalize_and_upload
# # #             finalize_and_upload(
# # #                 report_path     = out_path,
# # #                 analysis_id     = self.analysis_id,
# # #                 train_detail_id = self.train_detail_id,
# # #             )
# # #         except Exception as exc:
# # #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# # #         return out_path

# # #     # ── Private — deduplication & merging ────────────────────────────────────

# # #     def _deduplicate_by_frame(self) -> None:
# # #         unique: Dict[int, _Violation] = {}
# # #         for v in self._violations:
# # #             if v.frame_index not in unique:
# # #                 unique[v.frame_index] = v
# # #             else:
# # #                 ex = unique[v.frame_index]
# # #                 ex.events  = list(set(ex.events  + v.events))
# # #                 ex.factors = list(set(ex.factors + v.factors))
# # #                 if v.risk_score > ex.risk_score:
# # #                     ex.risk_score = v.risk_score
# # #                     ex.risk_level = v.risk_level
# # #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# # #                     ex.annotated_frame = v.annotated_frame
# # #         self._violations = list(unique.values())

# # #     def _merge_by_time_window(self) -> None:
# # #         if not self._violations:
# # #             return
# # #         self._violations.sort(key=lambda x: x.timestamp)
# # #         merged: List[_Violation] = []
# # #         group  = [self._violations[0]]
# # #         for v in self._violations[1:]:
# # #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# # #                 group.append(v)
# # #             else:
# # #                 merged.append(self._merge_group(group))
# # #                 group = [v]
# # #         merged.append(self._merge_group(group))
# # #         self._violations = merged

# # #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# # #         base             = group[0]
# # #         events: List[str]  = []
# # #         factors: List[str] = []
# # #         max_risk   = base.risk_score
# # #         risk_level = base.risk_level
# # #         best_frame = base.annotated_frame
# # #         for v in group:
# # #             events.extend(v.events)
# # #             factors.extend(v.factors)
# # #             if v.risk_score > max_risk:
# # #                 max_risk, risk_level = v.risk_score, v.risk_level
# # #             if best_frame is None and v.annotated_frame is not None:
# # #                 best_frame = v.annotated_frame
# # #         return _Violation(
# # #             timestamp        = base.timestamp,
# # #             time_str         = base.time_str,
# # #             frame_index      = base.frame_index,
# # #             type             = base.type,
# # #             events           = list(set(events)),
# # #             severity         = base.severity,
# # #             duration         = base.duration,
# # #             risk_score       = max_risk,
# # #             risk_level       = risk_level,
# # #             confidence       = base.confidence,
# # #             factors          = list(set(factors)),
# # #             source_filename  = base.source_filename,
# # #             local_time_str   = base.local_time_str,
# # #             annotated_frame  = best_frame,
# # #         )

# # #     # ── Private — frame extraction & saving ──────────────────────────────────

# # #     def extract_violation_frames(self, video_path: str) -> None:
# # #         print("[ViolationStore] Saving frames...")
# # #         need_video = [v for v in self._violations if v.annotated_frame is None]
# # #         saved = 0

# # #         # First pass: save violations that already have an annotated frame
# # #         for v in self._violations:
# # #             if v.annotated_frame is not None:
# # #                 v.frame_path      = self._save_frame(
# # #                     v.annotated_frame, v.events, v.time_str, v.frame_index
# # #                 )
# # #                 v.annotated_frame = None   # free memory
# # #                 saved += 1

# # #         # Second pass: re-read from the source video for any that are missing
# # #         if need_video:
# # #             cap = cv2.VideoCapture(video_path)
# # #             if not cap.isOpened():
# # #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# # #             else:
# # #                 seen: set = set()
# # #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# # #                     if v.frame_index in seen:
# # #                         continue
# # #                     seen.add(v.frame_index)
# # #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# # #                     ret, frame = cap.read()
# # #                     if not ret:
# # #                         continue
# # #                     v.frame_path = self._save_frame(
# # #                         frame, v.events, v.time_str, v.frame_index
# # #                     )
# # #                     saved += 1
# # #                 cap.release()

# # #         print(f"[ViolationStore] {saved} frames saved")

# # #     def _save_frame(
# # #         self,
# # #         frame:       np.ndarray,
# # #         events:      List[str],
# # #         time_str:    str,
# # #         frame_index: int,
# # #     ) -> str:
# # #         """
# # #         Save a single violation frame as a JPEG.

# # #         Filename format:  <events>_<HH-MM-SS>_f<frame_index>.jpg
# # #         Example:          seat_absence_00-01-14_f2106.jpg
# # #                           seat_absence_drowsy_00-03-02_f5058.jpg
# # #         """
# # #         distraction   = "_".join(sorted(events))   # sorted for deterministic name
# # #         filename_time = time_str.replace(":", "-")
# # #         filename      = f"{distraction}_{filename_time}_f{frame_index}.jpg"
# # #         path          = os.path.join(self.frames_dir, filename)
# # #         ok = cv2.imwrite(
# # #             path,
# # #             cv2.resize(frame, (640, 360)),
# # #             [cv2.IMWRITE_JPEG_QUALITY, 85],
# # #         )
# # #         if not ok:
# # #             print(f"[ViolationStore] imwrite failed: {path}")
# # #         return os.path.join(self.analysis_id, "frames", filename)

# # #     # ── Private — report builder ──────────────────────────────────────────────

# # #     def _build_report(self, processing_time: float = 0.0) -> dict:
# # #         return {
# # #             "analysis_id":     self.analysis_id,
# # #             "train_detail_id": self.train_detail_id,
# # #             "processing_time": round(processing_time, 3),
# # #             # Single video → keep as dict for backwards compat; batch → list
# # #             "video_info": (
# # #                 self.video_infos[0]
# # #                 if len(self.video_infos) == 1
# # #                 else self.video_infos
# # #             ),
# # #             "violations": [
# # #                 {
# # #                     "timestamp":   v.time_str,
# # #                     "frame_index": v.frame_index,
# # #                     "events":      v.events,
# # #                     "severity":    v.severity,
# # #                     "duration":    v.duration,
# # #                     "risk_score":  v.risk_score,
# # #                     "risk_level":  v.risk_level,
# # #                     "confidence":  v.confidence,
# # #                     "factors":     v.factors,
# # #                     "original_video_timestamp": (
# # #                         f"{v.source_filename} {v.local_time_str}"
# # #                     ),
# # #                     "frame_path":  v.frame_path,
# # #                 }
# # #                 for v in self._violations
# # #             ],
# # #         }


# # from __future__ import annotations

# # import json
# # import os
# # from dataclasses import dataclass
# # from typing import Any, Dict, List, Optional

# # import cv2
# # import numpy as np

# # OUTPUTS_ROOT = "outputs"
# # MERGE_WINDOW = 2.0


# # # ══════════════════════════════════════════════════════════════════════════════
# # # INTERNAL DATA CLASS
# # # ══════════════════════════════════════════════════════════════════════════════

# # @dataclass
# # class _Violation:
# #     timestamp:       float
# #     time_str:        str
# #     frame_index:     int
# #     type:            str
# #     events:          List[str]
# #     severity:        str
# #     duration:        float
# #     risk_score:      int
# #     risk_level:      str
# #     confidence:      float
# #     factors:         List[str]
# #     source_filename: str                  = ""    # DB filename e.g. "mobile.mp4"
# #     local_time_str:  str                  = ""    # local time within that file e.g. "00:00:23"
# #     frame_path:      Optional[str]        = None
# #     annotated_frame: Optional[np.ndarray] = None


# # # ══════════════════════════════════════════════════════════════════════════════
# # # VIOLATION STORE
# # # ══════════════════════════════════════════════════════════════════════════════

# # class ViolationStore:
# #     """
# #     Accumulates all violations found across one analysis run (single video
# #     or a multi-video batch that shares the same analysis_id / folder_name).

# #     Batch mode usage (api.py)
# #     ─────────────────────────
# #     1. Construct ONCE for the whole folder (no video_info in __init__).
# #     2. Pass as shared_vstore= to each GadgetDetectionPipeline.
# #        The pipeline calls add_video_info() automatically.
# #     3. Call finalize() ONCE after all videos in the folder are done.

# #     Standalone mode usage (CLI / single video)
# #     ──────────────────────────────────────────
# #     1. Construct with video_info= for the single video.
# #     2. Pipeline calls finalize() automatically at the end of run().
# #     """

# #     def __init__(
# #         self,
# #         analysis_id:     str,
# #         train_detail_id: int,
# #         video_info:      Optional[Dict[str, Any]] = None,
# #     ):
# #         self.analysis_id     = analysis_id
# #         self.train_detail_id = train_detail_id
# #         # Always a list — 0 entries until add_video_info() is called (batch mode),
# #         # or 1 entry when video_info is provided (standalone mode).
# #         self.video_infos: List[Dict[str, Any]] = (
# #             [video_info] if video_info is not None else []
# #         )

# #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# #         self.frames_dir = os.path.join(self.output_dir, "frames")
# #         os.makedirs(self.frames_dir, exist_ok=True)

# #         self._violations:  List[_Violation] = []
# #         self._seen_frames: set              = set()
# #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# #     # ── Public helpers ────────────────────────────────────────────────────────

# #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# #         """Append one video's metadata. Called once per video in batch mode."""
# #         self.video_infos.append(video_info)

# #     def record_violation(
# #         self,
# #         annotated_frame:  np.ndarray,
# #         video_time:       float,          # global timestamp (offset-adjusted)
# #         frame_index:      int,            # global frame index (offset-adjusted)
# #         event_type:       str,
# #         original_frame:   Optional[np.ndarray] = None,
# #         severity:         str   = "CRITICAL",
# #         confidence:       float = 0.9,
# #         risk_score:       int   = 80,
# #         risk_level:       str   = "CRITICAL",
# #         factors:          Optional[List[str]] = None,
# #         duration:         float = 0.0,
# #         source_filename:  str   = "",     # DB filename shown in original_video_timestamp
# #         local_video_time: float = -1.0,   # local time within the source file; -1 = same as video_time
# #     ) -> None:
# #         """
# #         Record one distraction event.

# #         Deduplication key is (frame_index, event_type):
# #           • same event on the same global frame is recorded once
# #           • different events on the same global frame are each recorded
# #           • global frame_index is unique across videos (frame_offset applied in main.py)
# #         """
# #         dedup_key = (frame_index, event_type)
# #         if dedup_key in self._seen_frames:
# #             return
# #         self._seen_frames.add(dedup_key)

# #         factors  = factors or []
# #         t        = int(round(video_time))
# #         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

# #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# #         local_str = (
# #             f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
# #         )

# #         self._violations.append(
# #             _Violation(
# #                 timestamp        = video_time,
# #                 time_str         = time_str,
# #                 frame_index      = frame_index,
# #                 type             = event_type,
# #                 events           = [event_type],
# #                 severity         = severity,
# #                 duration         = round(duration, 2),
# #                 risk_score       = risk_score,
# #                 risk_level       = risk_level,
# #                 confidence       = round(confidence, 3),
# #                 factors          = list(factors),
# #                 source_filename  = source_filename,
# #                 local_time_str   = local_str,
# #                 annotated_frame  = (
# #                     annotated_frame.copy() if annotated_frame is not None else None
# #                 ),
# #             )
# #         )

# #     # ── Finalize ──────────────────────────────────────────────────────────────

# #     def finalize(self, processing_time: float = 0.0) -> str:
# #         """
# #         Deduplicate → merge → save frames → write JSON → upload to S3/DB.
# #         Returns the local path to analysis_report.json.
# #         """
# #         self._deduplicate_by_frame()
# #         self._merge_by_time_window()

# #         # Extract frames from every video in the batch (or the single video)
# #         for vi in self.video_infos:
# #             if vi and vi.get("videoPath"):
# #                 self.extract_violation_frames(vi["videoPath"])

# #         report   = self._build_report(processing_time=processing_time)
# #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# #         with open(out_path, "w", encoding="utf-8") as f:
# #             json.dump(report, f, indent=2)

# #         print(f"[ViolationStore] JSON report     : {out_path}")
# #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# #         try:
# #             from utils.db_s3_uploader import finalize_and_upload
# #             finalize_and_upload(
# #                 report_path     = out_path,
# #                 analysis_id     = self.analysis_id,
# #                 train_detail_id = self.train_detail_id,
# #             )
# #         except Exception as exc:
# #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# #         return out_path

# #     # ── Private — deduplication & merging ────────────────────────────────────

# #     def _deduplicate_by_frame(self) -> None:
# #         unique: Dict[int, _Violation] = {}
# #         for v in self._violations:
# #             if v.frame_index not in unique:
# #                 unique[v.frame_index] = v
# #             else:
# #                 ex = unique[v.frame_index]
# #                 ex.events  = list(set(ex.events  + v.events))
# #                 ex.factors = list(set(ex.factors + v.factors))
# #                 if v.risk_score > ex.risk_score:
# #                     ex.risk_score = v.risk_score
# #                     ex.risk_level = v.risk_level
# #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# #                     ex.annotated_frame = v.annotated_frame
# #         self._violations = list(unique.values())

# #     def _merge_by_time_window(self) -> None:
# #         if not self._violations:
# #             return
# #         self._violations.sort(key=lambda x: x.timestamp)
# #         merged: List[_Violation] = []
# #         group  = [self._violations[0]]
# #         for v in self._violations[1:]:
# #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# #                 group.append(v)
# #             else:
# #                 merged.append(self._merge_group(group))
# #                 group = [v]
# #         merged.append(self._merge_group(group))
# #         self._violations = merged

# #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# #         base             = group[0]
# #         events: List[str]  = []
# #         factors: List[str] = []
# #         max_risk   = base.risk_score
# #         risk_level = base.risk_level
# #         best_frame = base.annotated_frame
# #         for v in group:
# #             events.extend(v.events)
# #             factors.extend(v.factors)
# #             if v.risk_score > max_risk:
# #                 max_risk, risk_level = v.risk_score, v.risk_level
# #             if best_frame is None and v.annotated_frame is not None:
# #                 best_frame = v.annotated_frame
# #         return _Violation(
# #             timestamp        = base.timestamp,
# #             time_str         = base.time_str,
# #             frame_index      = base.frame_index,
# #             type             = base.type,
# #             events           = list(set(events)),
# #             severity         = base.severity,
# #             duration         = base.duration,
# #             risk_score       = max_risk,
# #             risk_level       = risk_level,
# #             confidence       = base.confidence,
# #             factors          = list(set(factors)),
# #             source_filename  = base.source_filename,
# #             local_time_str   = base.local_time_str,
# #             annotated_frame  = best_frame,
# #         )

# #     # ── Private — frame extraction & saving ──────────────────────────────────

# #     def extract_violation_frames(self, video_path: str) -> None:
# #         print("[ViolationStore] Saving frames...")
# #         need_video = [v for v in self._violations if v.annotated_frame is None]
# #         saved = 0

# #         # First pass: save violations that already have an annotated frame
# #         for v in self._violations:
# #             if v.annotated_frame is not None:
# #                 v.frame_path      = self._save_frame(v.annotated_frame, v.events, v.time_str)
# #                 v.annotated_frame = None   # free memory
# #                 saved += 1

# #         # Second pass: re-read from the source video for any that are missing
# #         if need_video:
# #             cap = cv2.VideoCapture(video_path)
# #             if not cap.isOpened():
# #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# #             else:
# #                 seen: set = set()
# #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# #                     if v.frame_index in seen:
# #                         continue
# #                     seen.add(v.frame_index)
# #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# #                     ret, frame = cap.read()
# #                     if not ret:
# #                         continue
# #                     v.frame_path = self._save_frame(frame, v.events, v.time_str)
# #                     saved += 1
# #                 cap.release()

# #         print(f"[ViolationStore] {saved} frames saved")

# #     def _save_frame(
# #         self,
# #         frame:    np.ndarray,
# #         events:   List[str],
# #         time_str: str,
# #     ) -> str:
# #         """
# #         Save a single violation frame as JPEG.

# #         Filename format:  <events>_<HH-MM-SS>.jpg
# #         Example:          seat_absence_00-01-14.jpg
# #                           seat_absence_drowsy_00-03-02.jpg
# #                           phone_use_00-00-24.jpg
# #         """
# #         distraction   = "_".join(sorted(events))   # sorted for deterministic name
# #         filename_time = time_str.replace(":", "-")
# #         filename      = f"{distraction}_{filename_time}.jpg"
# #         path          = os.path.join(self.frames_dir, filename)
# #         ok = cv2.imwrite(
# #             path,
# #             cv2.resize(frame, (640, 360)),
# #             [cv2.IMWRITE_JPEG_QUALITY, 85],
# #         )
# #         if not ok:
# #             print(f"[ViolationStore] imwrite failed: {path}")
# #         return os.path.join(self.analysis_id, "frames", filename)

# #     # ── Private — report builder ──────────────────────────────────────────────

# #     def _build_report(self, processing_time: float = 0.0) -> dict:
# #         return {
# #             "analysis_id":     self.analysis_id,
# #             "train_detail_id": self.train_detail_id,
# #             "processing_time": round(processing_time, 3),
# #             # Single video → dict (backwards compat); batch → list
# #             "video_info": (
# #                 self.video_infos[0]
# #                 if len(self.video_infos) == 1
# #                 else self.video_infos
# #             ),
# #             "violations": [
# #                 {
# #                     "timestamp":   v.time_str,
# #                     "frame_index": v.frame_index,
# #                     "events":      v.events,
# #                     "severity":    v.severity,
# #                     "duration":    v.duration,
# #                     "risk_score":  v.risk_score,
# #                     "risk_level":  v.risk_level,
# #                     "confidence":  v.confidence,
# #                     "factors":     v.factors,
# #                     # "original_video_timestamp" = "<db_filename> <local_HH:MM:SS>"
# #                     # When timestamp == original (video 1), local_time_str == time_str
# #                     "original_video_timestamp": (
# #                         f"{v.source_filename} {v.local_time_str}"
# #                     ),
# #                     "frame_path":  v.frame_path,
# #                 }
# #                 for v in self._violations
# #             ],
# #         }


# from __future__ import annotations

# import json
# import os
# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional

# import cv2
# import numpy as np

# OUTPUTS_ROOT = "outputs"
# MERGE_WINDOW = 2.0


# # ══════════════════════════════════════════════════════════════════════════════
# # INTERNAL DATA CLASS
# # ══════════════════════════════════════════════════════════════════════════════

# @dataclass
# class _Violation:
#     timestamp:       float
#     time_str:        str
#     frame_index:     int
#     type:            str
#     events:          List[str]
#     severity:        str
#     duration:        float
#     risk_score:      int
#     risk_level:      str
#     confidence:      float
#     factors:         List[str]
#     source_filename: str                  = ""    # DB filename e.g. "mobile.mp4"
#     local_time_str:  str                  = ""    # local time within that file e.g. "00:00:23"
#     frame_path:      Optional[str]        = None
#     annotated_frame: Optional[np.ndarray] = None


# # ══════════════════════════════════════════════════════════════════════════════
# # VIOLATION STORE
# # ══════════════════════════════════════════════════════════════════════════════

# class ViolationStore:
#     """
#     Accumulates all violations found across one analysis run (single video
#     or a multi-video batch that shares the same analysis_id / folder_name).

#     Batch mode usage (api.py)
#     ─────────────────────────
#     1. Construct ONCE for the whole folder (no video_info in __init__).
#     2. Pass as shared_vstore= to each GadgetDetectionPipeline.
#        The pipeline calls add_video_info() automatically.
#     3. Call finalize() ONCE after all videos in the folder are done.

#     Standalone mode usage (CLI / single video)
#     ──────────────────────────────────────────
#     1. Construct with video_info= for the single video.
#     2. Pipeline calls finalize() automatically at the end of run().
#     """

#     def __init__(
#         self,
#         analysis_id:     str,
#         train_detail_id: int,
#         video_info:      Optional[Dict[str, Any]] = None,
#     ):
#         self.analysis_id     = analysis_id
#         self.train_detail_id = train_detail_id
#         # Always a list — 0 entries until add_video_info() is called (batch mode),
#         # or 1 entry when video_info is provided (standalone mode).
#         self.video_infos: List[Dict[str, Any]] = (
#             [video_info] if video_info is not None else []
#         )

#         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
#         self.frames_dir = os.path.join(self.output_dir, "frames")
#         os.makedirs(self.frames_dir, exist_ok=True)

#         self._violations:  List[_Violation] = []
#         self._seen_frames: set              = set()
#         print(f"[ViolationStore] Output dir : {self.output_dir}")

#     # ── Public helpers ────────────────────────────────────────────────────────

#     def add_video_info(self, video_info: Dict[str, Any]) -> None:
#         """Append one video's metadata. Called once per video in batch mode."""
#         self.video_infos.append(video_info)

#     def record_violation(
#         self,
#         annotated_frame:  np.ndarray,
#         video_time:       float,          # global timestamp (offset-adjusted)
#         frame_index:      int,            # global frame index (offset-adjusted)
#         event_type:       str,
#         original_frame:   Optional[np.ndarray] = None,
#         severity:         str   = "CRITICAL",
#         confidence:       float = 0.9,
#         risk_score:       int   = 80,
#         risk_level:       str   = "CRITICAL",
#         factors:          Optional[List[str]] = None,
#         duration:         float = 0.0,
#         source_filename:  str   = "",     # DB filename shown in original_video_timestamp
#         local_video_time: float = -1.0,   # local time within the source file; -1 = same as video_time
#     ) -> None:
#         """
#         Record one distraction event.

#         Deduplication key is (frame_index, event_type):
#           • same event on the same global frame is recorded once
#           • different events on the same global frame are each recorded
#           • global frame_index is unique across videos (frame_offset applied in main.py)
#         """
#         dedup_key = (frame_index, event_type)
#         if dedup_key in self._seen_frames:
#             return
#         self._seen_frames.add(dedup_key)

#         factors  = factors or []
#         t        = int(round(video_time))
#         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

#         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
#         local_str = (
#             f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
#         )

#         self._violations.append(
#             _Violation(
#                 timestamp        = video_time,
#                 time_str         = time_str,
#                 frame_index      = frame_index,
#                 type             = event_type,
#                 events           = [event_type],
#                 severity         = severity,
#                 duration         = round(duration, 2),
#                 risk_score       = risk_score,
#                 risk_level       = risk_level,
#                 confidence       = round(confidence, 3),
#                 factors          = list(factors),
#                 source_filename  = source_filename,
#                 local_time_str   = local_str,
#                 annotated_frame  = (
#                     annotated_frame.copy() if annotated_frame is not None else None
#                 ),
#             )
#         )

#     # ── Finalize ──────────────────────────────────────────────────────────────

#     def write_report(self, processing_time: float = 0.0) -> str:
#         """
#         Build analysis_report.json from the CURRENT in-memory violations and
#         write it to disk at  outputs/<analysis_id>/analysis_report.json
#         (i.e. as a SIBLING of the frames/ folder, not inside it).

#         Unlike finalize(), this does NOT touch S3 or the legacy DB uploader —
#         it is safe to call from the journey/batch pipeline (analyzer.py),
#         which has its own separate callback-based completion flow.

#         Does NOT run dedup/merge/extract_violation_frames — call those first
#         (analyze_journey() already does, via the shared ViolationStore) if
#         you need them. Safe to call multiple times; it always overwrites.

#         Returns the local path to analysis_report.json.
#         """
#         report   = self._build_report(processing_time=processing_time)
#         out_path = os.path.join(self.output_dir, "analysis_report.json")
#         os.makedirs(self.output_dir, exist_ok=True)
#         with open(out_path, "w", encoding="utf-8") as f:
#             json.dump(report, f, indent=2)

#         print(f"[ViolationStore] JSON report     : {out_path}")
#         print(f"[ViolationStore] Violations      : {len(self._violations)}")
#         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

#         return out_path

#     def finalize(self, processing_time: float = 0.0) -> str:
#         """
#         Deduplicate → merge → save frames → write JSON → upload to S3/DB.
#         Returns the local path to analysis_report.json.

#         Standalone/CLI mode only. Journey/batch mode (analyzer.py) should call
#         write_report() directly instead, after its own dedup/merge/extract
#         steps, since it has its own callback-based upload path and does not
#         want the legacy db_s3_uploader to also run.
#         """
#         self._deduplicate_by_frame()
#         self._merge_by_time_window()

#         # Extract frames from every video in the batch (or the single video)
#         for vi in self.video_infos:
#             if vi and vi.get("videoPath"):
#                 self.extract_violation_frames(vi["videoPath"])

#         out_path = self.write_report(processing_time=processing_time)

#         try:
#             from utils.db_s3_uploader import finalize_and_upload
#             finalize_and_upload(
#                 report_path     = out_path,
#                 analysis_id     = self.analysis_id,
#                 train_detail_id = self.train_detail_id,
#             )
#         except Exception as exc:
#             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

#         return out_path

#     # ── Private — deduplication & merging ────────────────────────────────────

#     def _deduplicate_by_frame(self) -> None:
#         # FIX (Multi-video dedup collision):
#         # The old key was frame_index alone. In a multi-video journey, Video 1
#         # frame 500 and Video 2 frame 500 share the same frame_index value
#         # (frame_offset makes them globally unique across the batch, but after
#         # merging by time window the stored frame_index is the base's global
#         # index). The safe dedup key must include source_filename so violations
#         # from different source files never collide.
#         unique: Dict[tuple, _Violation] = {}
#         for v in self._violations:
#             key = (v.source_filename, v.frame_index)
#             if key not in unique:
#                 unique[key] = v
#             else:
#                 ex = unique[key]
#                 ex.events  = list(set(ex.events  + v.events))
#                 ex.factors = list(set(ex.factors + v.factors))
#                 if v.risk_score > ex.risk_score:
#                     ex.risk_score = v.risk_score
#                     ex.risk_level = v.risk_level
#                 if ex.annotated_frame is None and v.annotated_frame is not None:
#                     ex.annotated_frame = v.annotated_frame
#         self._violations = list(unique.values())

#     def _merge_by_time_window(self) -> None:
#         if not self._violations:
#             return
#         self._violations.sort(key=lambda x: x.timestamp)
#         merged: List[_Violation] = []
#         group  = [self._violations[0]]
#         for v in self._violations[1:]:
#             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
#                 group.append(v)
#             else:
#                 merged.append(self._merge_group(group))
#                 group = [v]
#         merged.append(self._merge_group(group))
#         self._violations = merged

#     def _merge_group(self, group: List[_Violation]) -> _Violation:
#         base             = group[0]
#         events: List[str]  = []
#         factors: List[str] = []
#         max_risk   = base.risk_score
#         risk_level = base.risk_level
#         best_frame = base.annotated_frame
#         for v in group:
#             events.extend(v.events)
#             factors.extend(v.factors)
#             if v.risk_score > max_risk:
#                 max_risk, risk_level = v.risk_score, v.risk_level
#             if best_frame is None and v.annotated_frame is not None:
#                 best_frame = v.annotated_frame
#         return _Violation(
#             timestamp        = base.timestamp,
#             time_str         = base.time_str,
#             frame_index      = base.frame_index,
#             type             = base.type,
#             events           = list(set(events)),
#             severity         = base.severity,
#             duration         = base.duration,
#             risk_score       = max_risk,
#             risk_level       = risk_level,
#             confidence       = base.confidence,
#             factors          = list(set(factors)),
#             source_filename  = base.source_filename,
#             local_time_str   = base.local_time_str,
#             annotated_frame  = best_frame,
#         )

#     # ── Private — frame extraction & saving ──────────────────────────────────

#     def extract_violation_frames(self, video_path: str) -> None:
#         """
#         Extract and save one frame image per violation.

#         FIX (Wrong frames in multi-video journeys):
#         ────────────────────────────────────────────
#         The old implementation had two bugs when called in a loop over multiple
#         video files (as analyzer.py does for batch journeys):

#         BUG A — Global frame_index used to seek into per-video files.
#           v.frame_index is a GLOBAL frame number that accumulates across all
#           videos in a journey (set by frame_offset in main.py).  Seeking to
#           frame_index 4500 in video_2 lands on a completely unrelated frame if
#           video_1 contained frames 0-5000.  The resulting evidence image is
#           from the wrong video entirely.

#         BUG B — First pass re-ran for every video in the loop.
#           Violations that already had annotated_frame were saved to disk on the
#           first call (video_1), then overwritten on the second call (video_2)
#           because the first-pass loop had no guard for frame_path already set.

#         FIX:
#           1. Filter by source_filename so this call only processes violations
#              that belong to the video file at video_path.
#           2. Seek using local_time_str (seconds within this specific file)
#              via CAP_PROP_POS_MSEC — always correct regardless of how many
#              videos precede this one in the journey.
#           3. Guard the first pass with `v.frame_path is None` so already-saved
#              violations are never re-processed on subsequent calls.
#         """
#         import os as _os
#         src_filename = _os.path.basename(video_path)
#         print(f"[ViolationStore] Saving frames for {src_filename!r}...")

#         # Violations that belong to this source file and haven't been saved yet
#         mine = [
#             v for v in self._violations
#             if _os.path.basename(getattr(v, "source_filename", "")) == src_filename
#                and v.frame_path is None
#         ]

#         saved = 0

#         # Pass 1: violations with an annotated frame already in memory
#         need_video = []
#         for v in mine:
#             if v.annotated_frame is not None:
#                 v.frame_path      = self._save_frame(v.annotated_frame, v.events, v.time_str)
#                 v.annotated_frame = None   # free memory
#                 saved += 1
#             else:
#                 need_video.append(v)

#         # Pass 2: re-read from the source video using LOCAL time (not global frame_index)
#         if need_video:
#             cap = cv2.VideoCapture(video_path)
#             if not cap.isOpened():
#                 print(f"[ViolationStore] Cannot open video: {video_path}")
#             else:
#                 fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
#                 total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#                 # Sort by local time for efficient sequential seeking
#                 for v in sorted(need_video, key=lambda x: getattr(x, "local_time_str", "0:00:00")):
#                     local_str = getattr(v, "local_time_str", "0:00:00")
#                     # Parse "HH:MM:SS" → seconds
#                     try:
#                         parts = local_str.strip().split(":")
#                         local_secs = int(parts[0])*3600 + int(parts[1])*60 + float(parts[2])
#                     except Exception:
#                         local_secs = 0.0

#                     # Seek by milliseconds (more reliable than frame index)
#                     cap.set(cv2.CAP_PROP_POS_MSEC, local_secs * 1000.0)
#                     ret, frame = cap.read()
#                     if not ret:
#                         # Fallback: seek by local frame number
#                         local_frame = min(int(local_secs * fps), max(0, total - 1))
#                         cap.set(cv2.CAP_PROP_POS_FRAMES, local_frame)
#                         ret, frame = cap.read()
#                     if not ret:
#                         print(f"[ViolationStore] Seek failed for {src_filename} "
#                               f"@ local_time={local_str}")
#                         continue
#                     v.frame_path = self._save_frame(frame, v.events, v.time_str)
#                     saved += 1
#                 cap.release()

#         print(f"[ViolationStore] {saved} frames saved for {src_filename!r}")

#     def _save_frame(
#         self,
#         frame:    np.ndarray,
#         events:   List[str],
#         time_str: str,
#     ) -> str:
#         """
#         Save a single violation frame as JPEG.

#         Filename format:  <events>_<HH-MM-SS>.jpg
#         Example:          seat_absence_00-01-14.jpg
#                           seat_absence_drowsy_00-03-02.jpg
#                           phone_use_00-00-24.jpg
#         """
#         distraction   = "_".join(sorted(events))   # sorted for deterministic name
#         filename_time = time_str.replace(":", "-")
#         filename      = f"{distraction}_{filename_time}.jpg"
#         path          = os.path.join(self.frames_dir, filename)
#         ok = cv2.imwrite(
#             path,
#             cv2.resize(frame, (640, 360)),
#             [cv2.IMWRITE_JPEG_QUALITY, 85],
#         )
#         if not ok:
#             print(f"[ViolationStore] imwrite failed: {path}")
#         return os.path.join(self.analysis_id, "frames", filename)

#     # ── Private — report builder ──────────────────────────────────────────────

#     def _build_report(self, processing_time: float = 0.0) -> dict:
#         return {
#             "analysis_id":     self.analysis_id,
#             "train_detail_id": self.train_detail_id,
#             "processing_time": round(processing_time, 3),
#             # Single video → dict (backwards compat); batch → list
#             "video_info": (
#                 self.video_infos[0]
#                 if len(self.video_infos) == 1
#                 else self.video_infos
#             ),
#             "violations": [
#                 {
#                     "timestamp":   v.time_str,
#                     "frame_index": v.frame_index,
#                     "events":      v.events,
#                     "severity":    v.severity,
#                     "duration":    v.duration,
#                     "risk_score":  v.risk_score,
#                     "risk_level":  v.risk_level,
#                     "confidence":  v.confidence,
#                     "factors":     v.factors,
#                     # "original_video_timestamp" = "<db_filename> <local_HH:MM:SS>"
#                     # When timestamp == original (video 1), local_time_str == time_str
#                     "original_video_timestamp": (
#                         f"{v.source_filename} {v.local_time_str}"
#                     ),
#                     "frame_path":  v.frame_path,
#                 }
#                 for v in self._violations
#             ],
#         }

# # # # from __future__ import annotations

# # # # import json
# # # # import os
# # # # from dataclasses import dataclass
# # # # from typing import Any, Dict, List, Optional

# # # # import cv2
# # # # import numpy as np

# # # # OUTPUTS_ROOT = "outputs"
# # # # MERGE_WINDOW = 2.0


# # # # @dataclass
# # # # class _Violation:
# # # #     timestamp:       float
# # # #     time_str:        str
# # # #     frame_index:     int
# # # #     type:            str
# # # #     events:          List[str]
# # # #     severity:        str
# # # #     duration:        float
# # # #     risk_score:      int
# # # #     risk_level:      str
# # # #     confidence:      float
# # # #     factors:         List[str]
# # # #     source_filename: str                   = ""    # e.g. "ax.mp4"
# # # #     local_time_str:  str                   = ""    # local time within that file e.g. "00:00:18"
# # # #     frame_path:      Optional[str]         = None
# # # #     annotated_frame: Optional[np.ndarray]  = None


# # # # class ViolationStore:

# # # #     def __init__(self, analysis_id: str, train_detail_id: int, video_info: Optional[Dict[str, Any]] = None):
# # # #         self.analysis_id     = analysis_id
# # # #         self.train_detail_id = train_detail_id
# # # #         # video_infos is always a list — 1 entry for single-video, N entries for batch
# # # #         self.video_infos: List[Dict[str, Any]] = [video_info] if video_info is not None else []

# # # #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# # # #         self.frames_dir = os.path.join(self.output_dir, "frames")
# # # #         os.makedirs(self.frames_dir, exist_ok=True)

# # # #         self._violations: List[_Violation] = []
# # # #         self._seen_frames: set             = set()
# # # #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# # # #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# # # #         """
# # # #         Register one video's metadata into the shared store.
# # # #         Called once per video in batch mode (api.py passes shared_vstore).
# # # #         """
# # # #         self.video_infos.append(video_info)

# # # #     def record_violation(
# # # #         self,
# # # #         annotated_frame: np.ndarray,
# # # #         video_time:      float,
# # # #         frame_index:     int,
# # # #         event_type:      str,
# # # #         original_frame:  Optional[np.ndarray] = None,
# # # #         severity:        str   = "CRITICAL",
# # # #         confidence:      float = 0.9,
# # # #         risk_score:      int   = 80,
# # # #         risk_level:      str   = "CRITICAL",
# # # #         factors:         Optional[List[str]] = None,
# # # #         duration:        float = 0.0,
# # # #         source_filename: str   = "",   # original upload filename e.g. "ax.mp4"
# # # #         local_video_time: float = -1.0, # raw video_time before offset; -1 = same as video_time
# # # #     ):
# # # #         # Deduplicate on (frame_index, event_type) so that:
# # # #         #  • the same violation type on the same frame is recorded only once
# # # #         #  • different violation types on the same frame are each recorded
# # # #         #  • frame numbers from different videos never collide (frame_offset
# # # #         #    in main.py makes every global frame_index unique across the batch)
# # # #         dedup_key = (frame_index, event_type)
# # # #         if dedup_key in self._seen_frames:
# # # #             return
# # # #         self._seen_frames.add(dedup_key)
# # # #         factors   = factors or []
# # # #         t         = int(round(video_time))
# # # #         time_str  = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"
# # # #         # Build the per-file local timestamp (time within the source video)
# # # #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# # # #         local_str = f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
# # # #         self._violations.append(_Violation(
# # # #             timestamp        = video_time,
# # # #             time_str         = time_str,
# # # #             frame_index      = frame_index,
# # # #             type             = event_type,
# # # #             events           = [event_type],
# # # #             severity         = severity,
# # # #             duration         = round(duration, 2),
# # # #             risk_score       = risk_score,
# # # #             risk_level       = risk_level,
# # # #             confidence       = round(confidence, 3),
# # # #             factors          = list(factors),
# # # #             source_filename  = source_filename,
# # # #             local_time_str   = local_str,
# # # #             annotated_frame  = annotated_frame.copy() if annotated_frame is not None else None,
# # # #         ))

# # # #     def _deduplicate_by_frame(self):
# # # #         unique: Dict[int, _Violation] = {}
# # # #         for v in self._violations:
# # # #             if v.frame_index not in unique:
# # # #                 unique[v.frame_index] = v
# # # #             else:
# # # #                 ex = unique[v.frame_index]
# # # #                 ex.events  = list(set(ex.events  + v.events))
# # # #                 ex.factors = list(set(ex.factors + v.factors))
# # # #                 if v.risk_score > ex.risk_score:
# # # #                     ex.risk_score = v.risk_score
# # # #                     ex.risk_level = v.risk_level
# # # #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# # # #                     ex.annotated_frame = v.annotated_frame
# # # #         self._violations = list(unique.values())

# # # #     def _merge_by_time_window(self):
# # # #         if not self._violations:
# # # #             return
# # # #         self._violations.sort(key=lambda x: x.timestamp)
# # # #         merged = []
# # # #         group  = [self._violations[0]]
# # # #         for v in self._violations[1:]:
# # # #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# # # #                 group.append(v)
# # # #             else:
# # # #                 merged.append(self._merge_group(group))
# # # #                 group = [v]
# # # #         merged.append(self._merge_group(group))
# # # #         self._violations = merged

# # # #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# # # #         base            = group[0]
# # # #         events, factors = [], []
# # # #         max_risk        = base.risk_score
# # # #         risk_level      = base.risk_level
# # # #         best_frame      = base.annotated_frame
# # # #         for v in group:
# # # #             events.extend(v.events)
# # # #             factors.extend(v.factors)
# # # #             if v.risk_score > max_risk:
# # # #                 max_risk, risk_level = v.risk_score, v.risk_level
# # # #             if best_frame is None and v.annotated_frame is not None:
# # # #                 best_frame = v.annotated_frame
# # # #         return _Violation(
# # # #             timestamp        = base.timestamp,
# # # #             time_str         = base.time_str,
# # # #             frame_index      = base.frame_index,
# # # #             type             = base.type,
# # # #             events           = list(set(events)),
# # # #             severity         = base.severity,
# # # #             duration         = base.duration,
# # # #             risk_score       = max_risk,
# # # #             risk_level       = risk_level,
# # # #             confidence       = base.confidence,
# # # #             factors          = list(set(factors)),
# # # #             source_filename  = base.source_filename,
# # # #             local_time_str   = base.local_time_str,
# # # #             annotated_frame  = best_frame,
# # # #         )

# # # #     def extract_violation_frames(self, video_path: str):
# # # #         print("[ViolationStore] Saving frames...")
# # # #         need_video = [v for v in self._violations if v.annotated_frame is None]
# # # #         saved = 0
# # # #         for v in self._violations:
# # # #             if v.annotated_frame is not None:
# # # #                 v.frame_path      = self._save_frame(v.annotated_frame, v.events, v.time_str)
# # # #                 v.annotated_frame = None
# # # #                 saved += 1
# # # #         if need_video:
# # # #             cap = cv2.VideoCapture(video_path)
# # # #             if not cap.isOpened():
# # # #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# # # #             else:
# # # #                 seen: set = set()
# # # #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# # # #                     if v.frame_index in seen:
# # # #                         continue
# # # #                     seen.add(v.frame_index)
# # # #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# # # #                     ret, frame = cap.read()
# # # #                     if not ret:
# # # #                         continue
# # # #                     v.frame_path = self._save_frame(frame, v.events, v.time_str)
# # # #                     saved += 1
# # # #                 cap.release()
# # # #         print(f"[ViolationStore] {saved} frames saved")

# # # #     def _save_frame(self, frame: np.ndarray, events: List[str], time_str: str) -> str:
# # # #         distraction   = "_".join(events)
# # # #         filename_time = time_str.replace(":", "-")
# # # #         filename      = f"{distraction}_{filename_time}.jpg"
# # # #         path          = os.path.join(self.frames_dir, filename)
# # # #         ok = cv2.imwrite(path, cv2.resize(frame, (640, 360)),
# # # #                          [cv2.IMWRITE_JPEG_QUALITY, 85])
# # # #         if not ok:
# # # #             print(f"[ViolationStore] imwrite failed: {path}")
# # # #         return os.path.join(self.analysis_id, "frames", filename)

# # # #     def _build_report(self, processing_time: float = 0.0) -> dict:
# # # #         return {
# # # #             "analysis_id":     self.analysis_id,
# # # #             "train_detail_id": self.train_detail_id,
# # # #             "processing_time": round(processing_time, 3),
# # # #             # Single video → keep as dict for backwards compat; batch → list
# # # #             "video_info": self.video_infos[0] if len(self.video_infos) == 1
# # # #                           else self.video_infos,
# # # #             "violations": [
# # # #                 {
# # # #                     "timestamp":   v.time_str,
# # # #                     "frame_index": v.frame_index,
# # # #                     "events":      v.events,
# # # #                     "severity":    v.severity,
# # # #                     "duration":    v.duration,
# # # #                     "risk_score":  v.risk_score,
# # # #                     "risk_level":  v.risk_level,
# # # #                     "confidence":  v.confidence,
# # # #                     "factors":     v.factors,
# # # #                     "original_video_timestamp": f"{v.source_filename} {v.local_time_str}",
# # # #                     "frame_path":  v.frame_path,
# # # #                 }
# # # #                 for v in self._violations
# # # #             ],
# # # #         }

# # # #     def finalize(self, processing_time: float = 0.0) -> str:
# # # #         self._deduplicate_by_frame()
# # # #         self._merge_by_time_window()
# # # #         # Extract frames from every video in the batch (or the single video)
# # # #         for vi in self.video_infos:
# # # #             if vi and vi.get("videoPath"):
# # # #                 self.extract_violation_frames(vi["videoPath"])
# # # #         report   = self._build_report(processing_time=processing_time)
# # # #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# # # #         with open(out_path, "w", encoding="utf-8") as f:
# # # #             json.dump(report, f, indent=2)
# # # #         print(f"[ViolationStore] JSON report     : {out_path}")
# # # #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# # # #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# # # #         try:
# # # #             from utils.db_s3_uploader import finalize_and_upload
# # # #             finalize_and_upload(
# # # #                 report_path     = out_path,
# # # #                 analysis_id     = self.analysis_id,
# # # #                 train_detail_id = self.train_detail_id,
# # # #             )
# # # #         except Exception as exc:
# # # #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# # # #         return out_path



# # # # utils/violation_store.py
# # # # ─────────────────────────────────────────────────────────────────
# # # # Change from original:
# # # #   _save_frame() now appends _f{frame_index} to the filename so
# # # #   two violations of the same type at the same timestamp never
# # # #   silently overwrite each other.
# # # #   e.g.  phone_use_00-13-16_f6762.jpg
# # # #         seat_absence_00-00-17_f516.jpg
# # # # ─────────────────────────────────────────────────────────────────

# # # from __future__ import annotations

# # # import json
# # # import os
# # # from dataclasses import dataclass
# # # from typing import Any, Dict, List, Optional

# # # import cv2
# # # import numpy as np

# # # OUTPUTS_ROOT = "outputs"
# # # MERGE_WINDOW = 2.0


# # # @dataclass
# # # class _Violation:
# # #     timestamp:       float
# # #     time_str:        str
# # #     frame_index:     int
# # #     type:            str
# # #     events:          List[str]
# # #     severity:        str
# # #     duration:        float
# # #     risk_score:      int
# # #     risk_level:      str
# # #     confidence:      float
# # #     factors:         List[str]
# # #     source_filename: str                  = ""
# # #     local_time_str:  str                  = ""
# # #     frame_path:      Optional[str]        = None
# # #     annotated_frame: Optional[np.ndarray] = None


# # # class ViolationStore:

# # #     def __init__(
# # #         self,
# # #         analysis_id:     str,
# # #         train_detail_id: int,
# # #         video_info:      Optional[Dict[str, Any]] = None,
# # #     ):
# # #         self.analysis_id     = analysis_id
# # #         self.train_detail_id = train_detail_id
# # #         self.video_infos: List[Dict[str, Any]] = (
# # #             [video_info] if video_info is not None else []
# # #         )

# # #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# # #         self.frames_dir = os.path.join(self.output_dir, "frames")
# # #         os.makedirs(self.frames_dir, exist_ok=True)

# # #         self._violations: List[_Violation] = []
# # #         self._seen_frames: set             = set()
# # #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# # #     # ── called once per video in batch mode ──────────────────────
# # #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# # #         self.video_infos.append(video_info)

# # #     # ── called from main.py for every detected violation ─────────
# # #     def record_violation(
# # #         self,
# # #         annotated_frame:  np.ndarray,
# # #         video_time:       float,
# # #         frame_index:      int,
# # #         event_type:       str,
# # #         original_frame:   Optional[np.ndarray] = None,
# # #         severity:         str   = "CRITICAL",
# # #         confidence:       float = 0.9,
# # #         risk_score:       int   = 80,
# # #         risk_level:       str   = "CRITICAL",
# # #         factors:          Optional[List[str]] = None,
# # #         duration:         float = 0.0,
# # #         source_filename:  str   = "",
# # #         local_video_time: float = -1.0,
# # #     ):
# # #         # Deduplicate: same frame + same event type recorded only once.
# # #         # Different event types on the same frame are each recorded.
# # #         # frame_index is globally unique across batch (frame_offset in main.py).
# # #         dedup_key = (frame_index, event_type)
# # #         if dedup_key in self._seen_frames:
# # #             return
# # #         self._seen_frames.add(dedup_key)

# # #         factors  = factors or []
# # #         t        = int(round(video_time))
# # #         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

# # #         # Local timestamp = time within the source video file (before cumulative offset)
# # #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# # #         local_str = f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"

# # #         self._violations.append(_Violation(
# # #             timestamp        = video_time,
# # #             time_str         = time_str,
# # #             frame_index      = frame_index,
# # #             type             = event_type,
# # #             events           = [event_type],
# # #             severity         = severity,
# # #             duration         = round(duration, 2),
# # #             risk_score       = risk_score,
# # #             risk_level       = risk_level,
# # #             confidence       = round(confidence, 3),
# # #             factors          = list(factors),
# # #             source_filename  = source_filename,
# # #             local_time_str   = local_str,
# # #             annotated_frame  = annotated_frame.copy() if annotated_frame is not None else None,
# # #         ))

# # #     # ─────────────────────────────────────────────────────────────

# # #     def _deduplicate_by_frame(self):
# # #         unique: Dict[int, _Violation] = {}
# # #         for v in self._violations:
# # #             if v.frame_index not in unique:
# # #                 unique[v.frame_index] = v
# # #             else:
# # #                 ex = unique[v.frame_index]
# # #                 ex.events  = list(set(ex.events  + v.events))
# # #                 ex.factors = list(set(ex.factors + v.factors))
# # #                 if v.risk_score > ex.risk_score:
# # #                     ex.risk_score = v.risk_score
# # #                     ex.risk_level = v.risk_level
# # #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# # #                     ex.annotated_frame = v.annotated_frame
# # #         self._violations = list(unique.values())

# # #     def _merge_by_time_window(self):
# # #         if not self._violations:
# # #             return
# # #         self._violations.sort(key=lambda x: x.timestamp)
# # #         merged = []
# # #         group  = [self._violations[0]]
# # #         for v in self._violations[1:]:
# # #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# # #                 group.append(v)
# # #             else:
# # #                 merged.append(self._merge_group(group))
# # #                 group = [v]
# # #         merged.append(self._merge_group(group))
# # #         self._violations = merged

# # #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# # #         base            = group[0]
# # #         events, factors = [], []
# # #         max_risk        = base.risk_score
# # #         risk_level      = base.risk_level
# # #         best_frame      = base.annotated_frame
# # #         for v in group:
# # #             events.extend(v.events)
# # #             factors.extend(v.factors)
# # #             if v.risk_score > max_risk:
# # #                 max_risk, risk_level = v.risk_score, v.risk_level
# # #             if best_frame is None and v.annotated_frame is not None:
# # #                 best_frame = v.annotated_frame
# # #         return _Violation(
# # #             timestamp        = base.timestamp,
# # #             time_str         = base.time_str,
# # #             frame_index      = base.frame_index,
# # #             type             = base.type,
# # #             events           = list(set(events)),
# # #             severity         = base.severity,
# # #             duration         = base.duration,
# # #             risk_score       = max_risk,
# # #             risk_level       = risk_level,
# # #             confidence       = base.confidence,
# # #             factors          = list(set(factors)),
# # #             source_filename  = base.source_filename,
# # #             local_time_str   = base.local_time_str,
# # #             annotated_frame  = best_frame,
# # #         )

# # #     # ─────────────────────────────────────────────────────────────

# # #     def extract_violation_frames(self, video_path: str):
# # #         print("[ViolationStore] Saving frames...")
# # #         need_video = [v for v in self._violations if v.annotated_frame is None]
# # #         saved = 0

# # #         for v in self._violations:
# # #             if v.annotated_frame is not None:
# # #                 v.frame_path      = self._save_frame(
# # #                     v.annotated_frame, v.events, v.time_str, v.frame_index
# # #                 )
# # #                 v.annotated_frame = None
# # #                 saved += 1

# # #         if need_video:
# # #             cap = cv2.VideoCapture(video_path)
# # #             if not cap.isOpened():
# # #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# # #             else:
# # #                 seen: set = set()
# # #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# # #                     if v.frame_index in seen:
# # #                         continue
# # #                     seen.add(v.frame_index)
# # #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# # #                     ret, frame = cap.read()
# # #                     if not ret:
# # #                         continue
# # #                     v.frame_path = self._save_frame(
# # #                         frame, v.events, v.time_str, v.frame_index
# # #                     )
# # #                     saved += 1
# # #                 cap.release()

# # #         print(f"[ViolationStore] {saved} frames saved")

# # #     def _save_frame(
# # #         self,
# # #         frame:       np.ndarray,
# # #         events:      List[str],
# # #         time_str:    str,
# # #         frame_index: int,            # ← ADDED: makes filename globally unique
# # #     ) -> str:
# # #         """
# # #         Save one violation frame as JPEG.

# # #         Filename: {event_types}_{hh-mm-ss}_f{frame_index}.jpg
# # #         Example:  phone_use_00-13-16_f6762.jpg
# # #                   seat_absence_00-00-17_f516.jpg

# # #         frame_index prevents two violations of the same type at the
# # #         same timestamp from silently overwriting each other.
# # #         """
# # #         distraction   = "_".join(events)
# # #         filename_time = time_str.replace(":", "-")
# # #         filename      = f"{distraction}_{filename_time}_f{frame_index}.jpg"  # ← CHANGED
# # #         path          = os.path.join(self.frames_dir, filename)
# # #         ok = cv2.imwrite(
# # #             path,
# # #             cv2.resize(frame, (640, 360)),
# # #             [cv2.IMWRITE_JPEG_QUALITY, 85],
# # #         )
# # #         if not ok:
# # #             print(f"[ViolationStore] imwrite failed: {path}")
# # #         return os.path.join(self.analysis_id, "frames", filename)

# # #     # ─────────────────────────────────────────────────────────────

# # #     def _build_report(self, processing_time: float = 0.0) -> dict:
# # #         return {
# # #             "analysis_id":     self.analysis_id,
# # #             "train_detail_id": self.train_detail_id,
# # #             "processing_time": round(processing_time, 3),
# # #             "video_info": (
# # #                 self.video_infos[0] if len(self.video_infos) == 1
# # #                 else self.video_infos
# # #             ),
# # #             "violations": [
# # #                 {
# # #                     "timestamp":   v.time_str,
# # #                     "frame_index": v.frame_index,
# # #                     "events":      v.events,
# # #                     "severity":    v.severity,
# # #                     "duration":    v.duration,
# # #                     "risk_score":  v.risk_score,
# # #                     "risk_level":  v.risk_level,
# # #                     "confidence":  v.confidence,
# # #                     "factors":     v.factors,
# # #                     "original_video_timestamp": f"{v.source_filename} {v.local_time_str}",
# # #                     "frame_path":  v.frame_path,
# # #                 }
# # #                 for v in self._violations
# # #             ],
# # #         }

# # #     def finalize(self, processing_time: float = 0.0) -> str:
# # #         self._deduplicate_by_frame()
# # #         self._merge_by_time_window()

# # #         # Save frames — each video's temp path is in video_infos
# # #         for vi in self.video_infos:
# # #             if vi and vi.get("videoPath"):
# # #                 self.extract_violation_frames(vi["videoPath"])

# # #         report   = self._build_report(processing_time=processing_time)
# # #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# # #         with open(out_path, "w", encoding="utf-8") as f:
# # #             json.dump(report, f, indent=2)

# # #         print(f"[ViolationStore] JSON report     : {out_path}")
# # #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# # #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# # #         # Upload results to S3 and update DB result_s3_path
# # #         try:
# # #             from utils.db_s3_uploader import finalize_and_upload
# # #             finalize_and_upload(
# # #                 report_path     = out_path,
# # #                 analysis_id     = self.analysis_id,
# # #                 train_detail_id = self.train_detail_id,
# # #             )
# # #         except Exception as exc:
# # #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# # #         return out_path

# # # # # from __future__ import annotations

# # # # # import json
# # # # # import os
# # # # # from dataclasses import dataclass
# # # # # from typing import Any, Dict, List, Optional

# # # # # import cv2
# # # # # import numpy as np

# # # # # OUTPUTS_ROOT = "outputs"
# # # # # MERGE_WINDOW = 2.0


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # INTERNAL DATA CLASS
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # @dataclass
# # # # # class _Violation:
# # # # #     timestamp:       float
# # # # #     time_str:        str
# # # # #     frame_index:     int
# # # # #     type:            str
# # # # #     events:          List[str]
# # # # #     severity:        str
# # # # #     duration:        float
# # # # #     risk_score:      int
# # # # #     risk_level:      str
# # # # #     confidence:      float
# # # # #     factors:         List[str]
# # # # #     source_filename: str                  = ""    # e.g. "ch01.mp4"
# # # # #     local_time_str:  str                  = ""    # local time within that file e.g. "00:00:18"
# # # # #     frame_path:      Optional[str]        = None
# # # # #     annotated_frame: Optional[np.ndarray] = None


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # VIOLATION STORE
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # class ViolationStore:
# # # # #     """
# # # # #     Accumulates all violations found across one analysis run (single video
# # # # #     or a multi-video batch that shares the same analysis_id / folder_name).

# # # # #     Usage
# # # # #     ─────
# # # # #     1. Construct once per analysis run.
# # # # #     2. Call record_violation() from the pipeline for every distraction event.
# # # # #     3. Call finalize() at the very end — it deduplicates, merges, saves frames,
# # # # #        writes the JSON report, then calls db_s3_uploader.finalize_and_upload()
# # # # #        to push everything to S3 and record results in the DB.
# # # # #     """

# # # # #     def __init__(
# # # # #         self,
# # # # #         analysis_id:     str,
# # # # #         train_detail_id: int,
# # # # #         video_info:      Optional[Dict[str, Any]] = None,
# # # # #     ):
# # # # #         self.analysis_id     = analysis_id
# # # # #         self.train_detail_id = train_detail_id
# # # # #         # video_infos is always a list:
# # # # #         #   • 1 entry for single-video runs
# # # # #         #   • N entries for batch runs (add_video_info called per video)
# # # # #         self.video_infos: List[Dict[str, Any]] = (
# # # # #             [video_info] if video_info is not None else []
# # # # #         )

# # # # #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# # # # #         self.frames_dir = os.path.join(self.output_dir, "frames")
# # # # #         os.makedirs(self.frames_dir, exist_ok=True)

# # # # #         self._violations:  List[_Violation] = []
# # # # #         self._seen_frames: set              = set()
# # # # #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# # # # #     # ── Public helpers ────────────────────────────────────────────────────────

# # # # #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# # # # #         """
# # # # #         Register one video's metadata into the shared store.
# # # # #         Called once per video in batch mode.
# # # # #         """
# # # # #         self.video_infos.append(video_info)

# # # # #     def record_violation(
# # # # #         self,
# # # # #         annotated_frame:  np.ndarray,
# # # # #         video_time:       float,
# # # # #         frame_index:      int,
# # # # #         event_type:       str,
# # # # #         original_frame:   Optional[np.ndarray] = None,
# # # # #         severity:         str   = "CRITICAL",
# # # # #         confidence:       float = 0.9,
# # # # #         risk_score:       int   = 80,
# # # # #         risk_level:       str   = "CRITICAL",
# # # # #         factors:          Optional[List[str]] = None,
# # # # #         duration:         float = 0.0,
# # # # #         source_filename:  str   = "",    # original filename e.g. "ch01.mp4"
# # # # #         local_video_time: float = -1.0,  # video_time before global offset; -1 = same as video_time
# # # # #     ) -> None:
# # # # #         """
# # # # #         Record one distraction event.

# # # # #         Deduplication key is (frame_index, event_type) so that:
# # # # #           • the same violation type on the same frame is recorded only once
# # # # #           • different violation types on the same frame are each recorded
# # # # #           • frame numbers from different videos never collide because
# # # # #             main.py applies a frame_offset to make every global frame_index
# # # # #             unique across the batch
# # # # #         """
# # # # #         dedup_key = (frame_index, event_type)
# # # # #         if dedup_key in self._seen_frames:
# # # # #             return
# # # # #         self._seen_frames.add(dedup_key)

# # # # #         factors  = factors or []
# # # # #         t        = int(round(video_time))
# # # # #         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

# # # # #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# # # # #         local_str = (
# # # # #             f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
# # # # #         )

# # # # #         self._violations.append(
# # # # #             _Violation(
# # # # #                 timestamp        = video_time,
# # # # #                 time_str         = time_str,
# # # # #                 frame_index      = frame_index,
# # # # #                 type             = event_type,
# # # # #                 events           = [event_type],
# # # # #                 severity         = severity,
# # # # #                 duration         = round(duration, 2),
# # # # #                 risk_score       = risk_score,
# # # # #                 risk_level       = risk_level,
# # # # #                 confidence       = round(confidence, 3),
# # # # #                 factors          = list(factors),
# # # # #                 source_filename  = source_filename,
# # # # #                 local_time_str   = local_str,
# # # # #                 annotated_frame  = (
# # # # #                     annotated_frame.copy() if annotated_frame is not None else None
# # # # #                 ),
# # # # #             )
# # # # #         )

# # # # #     # ── Finalize ──────────────────────────────────────────────────────────────

# # # # #     def finalize(self, processing_time: float = 0.0) -> str:
# # # # #         """
# # # # #         1. Deduplicate violations that share the same frame.
# # # # #         2. Merge violations that are within MERGE_WINDOW seconds of each other.
# # # # #         3. Save annotated frame images to disk (falls back to re-reading from
# # # # #            the source video when no annotated frame was captured).
# # # # #         4. Write analysis_report.json.
# # # # #         5. Upload frames + report to S3 and record in DB via db_s3_uploader.
# # # # #         Returns the local path to analysis_report.json.
# # # # #         """
# # # # #         self._deduplicate_by_frame()
# # # # #         self._merge_by_time_window()

# # # # #         # Extract frames from every video in the batch (or the single video)
# # # # #         for vi in self.video_infos:
# # # # #             if vi and vi.get("videoPath"):
# # # # #                 self.extract_violation_frames(vi["videoPath"])

# # # # #         report   = self._build_report(processing_time=processing_time)
# # # # #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# # # # #         with open(out_path, "w", encoding="utf-8") as f:
# # # # #             json.dump(report, f, indent=2)

# # # # #         print(f"[ViolationStore] JSON report     : {out_path}")
# # # # #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# # # # #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# # # # #         try:
# # # # #             from utils.db_s3_uploader import finalize_and_upload
# # # # #             finalize_and_upload(
# # # # #                 report_path     = out_path,
# # # # #                 analysis_id     = self.analysis_id,
# # # # #                 train_detail_id = self.train_detail_id,
# # # # #             )
# # # # #         except Exception as exc:
# # # # #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# # # # #         return out_path

# # # # #     # ── Private — deduplication & merging ────────────────────────────────────

# # # # #     def _deduplicate_by_frame(self) -> None:
# # # # #         unique: Dict[int, _Violation] = {}
# # # # #         for v in self._violations:
# # # # #             if v.frame_index not in unique:
# # # # #                 unique[v.frame_index] = v
# # # # #             else:
# # # # #                 ex = unique[v.frame_index]
# # # # #                 ex.events  = list(set(ex.events  + v.events))
# # # # #                 ex.factors = list(set(ex.factors + v.factors))
# # # # #                 if v.risk_score > ex.risk_score:
# # # # #                     ex.risk_score = v.risk_score
# # # # #                     ex.risk_level = v.risk_level
# # # # #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# # # # #                     ex.annotated_frame = v.annotated_frame
# # # # #         self._violations = list(unique.values())

# # # # #     def _merge_by_time_window(self) -> None:
# # # # #         if not self._violations:
# # # # #             return
# # # # #         self._violations.sort(key=lambda x: x.timestamp)
# # # # #         merged: List[_Violation] = []
# # # # #         group  = [self._violations[0]]
# # # # #         for v in self._violations[1:]:
# # # # #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# # # # #                 group.append(v)
# # # # #             else:
# # # # #                 merged.append(self._merge_group(group))
# # # # #                 group = [v]
# # # # #         merged.append(self._merge_group(group))
# # # # #         self._violations = merged

# # # # #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# # # # #         base            = group[0]
# # # # #         events: List[str]  = []
# # # # #         factors: List[str] = []
# # # # #         max_risk   = base.risk_score
# # # # #         risk_level = base.risk_level
# # # # #         best_frame = base.annotated_frame
# # # # #         for v in group:
# # # # #             events.extend(v.events)
# # # # #             factors.extend(v.factors)
# # # # #             if v.risk_score > max_risk:
# # # # #                 max_risk, risk_level = v.risk_score, v.risk_level
# # # # #             if best_frame is None and v.annotated_frame is not None:
# # # # #                 best_frame = v.annotated_frame
# # # # #         return _Violation(
# # # # #             timestamp        = base.timestamp,
# # # # #             time_str         = base.time_str,
# # # # #             frame_index      = base.frame_index,
# # # # #             type             = base.type,
# # # # #             events           = list(set(events)),
# # # # #             severity         = base.severity,
# # # # #             duration         = base.duration,
# # # # #             risk_score       = max_risk,
# # # # #             risk_level       = risk_level,
# # # # #             confidence       = base.confidence,
# # # # #             factors          = list(set(factors)),
# # # # #             source_filename  = base.source_filename,
# # # # #             local_time_str   = base.local_time_str,
# # # # #             annotated_frame  = best_frame,
# # # # #         )

# # # # #     # ── Private — frame extraction & saving ──────────────────────────────────

# # # # #     def extract_violation_frames(self, video_path: str) -> None:
# # # # #         print("[ViolationStore] Saving frames...")
# # # # #         need_video = [v for v in self._violations if v.annotated_frame is None]
# # # # #         saved = 0

# # # # #         # First pass: save violations that already have an annotated frame
# # # # #         for v in self._violations:
# # # # #             if v.annotated_frame is not None:
# # # # #                 v.frame_path      = self._save_frame(v.annotated_frame, v.events, v.time_str)
# # # # #                 v.annotated_frame = None   # free memory
# # # # #                 saved += 1

# # # # #         # Second pass: re-read from the source video for any that are missing
# # # # #         if need_video:
# # # # #             cap = cv2.VideoCapture(video_path)
# # # # #             if not cap.isOpened():
# # # # #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# # # # #             else:
# # # # #                 seen: set = set()
# # # # #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# # # # #                     if v.frame_index in seen:
# # # # #                         continue
# # # # #                     seen.add(v.frame_index)
# # # # #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# # # # #                     ret, frame = cap.read()
# # # # #                     if not ret:
# # # # #                         continue
# # # # #                     v.frame_path = self._save_frame(frame, v.events, v.time_str)
# # # # #                     saved += 1
# # # # #                 cap.release()

# # # # #         print(f"[ViolationStore] {saved} frames saved")

# # # # #     def _save_frame(
# # # # #         self,
# # # # #         frame:    np.ndarray,
# # # # #         events:   List[str],
# # # # #         time_str: str,
# # # # #     ) -> str:
# # # # #         distraction   = "_".join(events)
# # # # #         filename_time = time_str.replace(":", "-")
# # # # #         filename      = f"{distraction}_{filename_time}.jpg"
# # # # #         path          = os.path.join(self.frames_dir, filename)
# # # # #         ok = cv2.imwrite(
# # # # #             path,
# # # # #             cv2.resize(frame, (640, 360)),
# # # # #             [cv2.IMWRITE_JPEG_QUALITY, 85],
# # # # #         )
# # # # #         if not ok:
# # # # #             print(f"[ViolationStore] imwrite failed: {path}")
# # # # #         return os.path.join(self.analysis_id, "frames", filename)

# # # # #     # ── Private — report builder ──────────────────────────────────────────────

# # # # #     def _build_report(self, processing_time: float = 0.0) -> dict:
# # # # #         return {
# # # # #             "analysis_id":     self.analysis_id,
# # # # #             "train_detail_id": self.train_detail_id,
# # # # #             "processing_time": round(processing_time, 3),
# # # # #             # Single video → keep as dict for backwards compat; batch → list
# # # # #             "video_info": (
# # # # #                 self.video_infos[0]
# # # # #                 if len(self.video_infos) == 1
# # # # #                 else self.video_infos
# # # # #             ),
# # # # #             "violations": [
# # # # #                 {
# # # # #                     "timestamp":   v.time_str,
# # # # #                     "frame_index": v.frame_index,
# # # # #                     "events":      v.events,
# # # # #                     "severity":    v.severity,
# # # # #                     "duration":    v.duration,
# # # # #                     "risk_score":  v.risk_score,
# # # # #                     "risk_level":  v.risk_level,
# # # # #                     "confidence":  v.confidence,
# # # # #                     "factors":     v.factors,
# # # # #                     "original_video_timestamp": (
# # # # #                         f"{v.source_filename} {v.local_time_str}"
# # # # #                     ),
# # # # #                     "frame_path":  v.frame_path,
# # # # #                 }
# # # # #                 for v in self._violations
# # # # #             ],
# # # # #         }

# # # # from __future__ import annotations

# # # # import json
# # # # import os
# # # # from dataclasses import dataclass
# # # # from typing import Any, Dict, List, Optional

# # # # import cv2
# # # # import numpy as np

# # # # OUTPUTS_ROOT = "outputs"
# # # # MERGE_WINDOW = 2.0


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # INTERNAL DATA CLASS
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # @dataclass
# # # # class _Violation:
# # # #     timestamp:       float
# # # #     time_str:        str
# # # #     frame_index:     int
# # # #     type:            str
# # # #     events:          List[str]
# # # #     severity:        str
# # # #     duration:        float
# # # #     risk_score:      int
# # # #     risk_level:      str
# # # #     confidence:      float
# # # #     factors:         List[str]
# # # #     source_filename: str                  = ""    # e.g. "ch01.mp4"
# # # #     local_time_str:  str                  = ""    # local time within that file e.g. "00:00:18"
# # # #     frame_path:      Optional[str]        = None
# # # #     annotated_frame: Optional[np.ndarray] = None


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # VIOLATION STORE
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # class ViolationStore:
# # # #     """
# # # #     Accumulates all violations found across one analysis run (single video
# # # #     or a multi-video batch that shares the same analysis_id / folder_name).

# # # #     Usage
# # # #     ─────
# # # #     1. Construct once per analysis run.
# # # #     2. Call record_violation() from the pipeline for every distraction event.
# # # #     3. Call finalize() at the very end — it deduplicates, merges, saves frames,
# # # #        writes the JSON report, then calls db_s3_uploader.finalize_and_upload()
# # # #        to push everything to S3 and record results in the DB.
# # # #     """

# # # #     def __init__(
# # # #         self,
# # # #         analysis_id:     str,
# # # #         train_detail_id: int,
# # # #         video_info:      Optional[Dict[str, Any]] = None,
# # # #     ):
# # # #         self.analysis_id     = analysis_id
# # # #         self.train_detail_id = train_detail_id
# # # #         # video_infos is always a list:
# # # #         #   • 1 entry for single-video runs
# # # #         #   • N entries for batch runs (add_video_info called per video)
# # # #         self.video_infos: List[Dict[str, Any]] = (
# # # #             [video_info] if video_info is not None else []
# # # #         )

# # # #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# # # #         self.frames_dir = os.path.join(self.output_dir, "frames")
# # # #         os.makedirs(self.frames_dir, exist_ok=True)

# # # #         self._violations:  List[_Violation] = []
# # # #         self._seen_frames: set              = set()
# # # #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# # # #     # ── Public helpers ────────────────────────────────────────────────────────

# # # #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# # # #         """
# # # #         Register one video's metadata into the shared store.
# # # #         Called once per video in batch mode.
# # # #         """
# # # #         self.video_infos.append(video_info)

# # # #     def record_violation(
# # # #         self,
# # # #         annotated_frame:  np.ndarray,
# # # #         video_time:       float,
# # # #         frame_index:      int,
# # # #         event_type:       str,
# # # #         original_frame:   Optional[np.ndarray] = None,
# # # #         severity:         str   = "CRITICAL",
# # # #         confidence:       float = 0.9,
# # # #         risk_score:       int   = 80,
# # # #         risk_level:       str   = "CRITICAL",
# # # #         factors:          Optional[List[str]] = None,
# # # #         duration:         float = 0.0,
# # # #         source_filename:  str   = "",    # original filename e.g. "ch01.mp4"
# # # #         local_video_time: float = -1.0,  # video_time before global offset; -1 = same as video_time
# # # #     ) -> None:
# # # #         """
# # # #         Record one distraction event.

# # # #         Deduplication key is (frame_index, event_type) so that:
# # # #           • the same violation type on the same frame is recorded only once
# # # #           • different violation types on the same frame are each recorded
# # # #           • frame numbers from different videos never collide because
# # # #             main.py applies a frame_offset to make every global frame_index
# # # #             unique across the batch
# # # #         """
# # # #         dedup_key = (frame_index, event_type)
# # # #         if dedup_key in self._seen_frames:
# # # #             return
# # # #         self._seen_frames.add(dedup_key)

# # # #         factors  = factors or []
# # # #         t        = int(round(video_time))
# # # #         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

# # # #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# # # #         local_str = (
# # # #             f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
# # # #         )

# # # #         self._violations.append(
# # # #             _Violation(
# # # #                 timestamp        = video_time,
# # # #                 time_str         = time_str,
# # # #                 frame_index      = frame_index,
# # # #                 type             = event_type,
# # # #                 events           = [event_type],
# # # #                 severity         = severity,
# # # #                 duration         = round(duration, 2),
# # # #                 risk_score       = risk_score,
# # # #                 risk_level       = risk_level,
# # # #                 confidence       = round(confidence, 3),
# # # #                 factors          = list(factors),
# # # #                 source_filename  = source_filename,
# # # #                 local_time_str   = local_str,
# # # #                 annotated_frame  = (
# # # #                     annotated_frame.copy() if annotated_frame is not None else None
# # # #                 ),
# # # #             )
# # # #         )

# # # #     # ── Finalize ──────────────────────────────────────────────────────────────

# # # #     def finalize(self, processing_time: float = 0.0) -> str:
# # # #         """
# # # #         1. Deduplicate violations that share the same frame.
# # # #         2. Merge violations that are within MERGE_WINDOW seconds of each other.
# # # #         3. Save annotated frame images to disk (falls back to re-reading from
# # # #            the source video when no annotated frame was captured).
# # # #         4. Write analysis_report.json.
# # # #         5. Upload frames + report to S3 and record in DB via db_s3_uploader.
# # # #         Returns the local path to analysis_report.json.
# # # #         """
# # # #         self._deduplicate_by_frame()
# # # #         self._merge_by_time_window()

# # # #         # Extract frames from every video in the batch (or the single video)
# # # #         for vi in self.video_infos:
# # # #             if vi and vi.get("videoPath"):
# # # #                 self.extract_violation_frames(vi["videoPath"])

# # # #         report   = self._build_report(processing_time=processing_time)
# # # #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# # # #         with open(out_path, "w", encoding="utf-8") as f:
# # # #             json.dump(report, f, indent=2)

# # # #         print(f"[ViolationStore] JSON report     : {out_path}")
# # # #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# # # #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# # # #         try:
# # # #             from utils.db_s3_uploader import finalize_and_upload
# # # #             finalize_and_upload(
# # # #                 report_path     = out_path,
# # # #                 analysis_id     = self.analysis_id,
# # # #                 train_detail_id = self.train_detail_id,
# # # #             )
# # # #         except Exception as exc:
# # # #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# # # #         return out_path

# # # #     # ── Private — deduplication & merging ────────────────────────────────────

# # # #     def _deduplicate_by_frame(self) -> None:
# # # #         unique: Dict[int, _Violation] = {}
# # # #         for v in self._violations:
# # # #             if v.frame_index not in unique:
# # # #                 unique[v.frame_index] = v
# # # #             else:
# # # #                 ex = unique[v.frame_index]
# # # #                 ex.events  = list(set(ex.events  + v.events))
# # # #                 ex.factors = list(set(ex.factors + v.factors))
# # # #                 if v.risk_score > ex.risk_score:
# # # #                     ex.risk_score = v.risk_score
# # # #                     ex.risk_level = v.risk_level
# # # #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# # # #                     ex.annotated_frame = v.annotated_frame
# # # #         self._violations = list(unique.values())

# # # #     def _merge_by_time_window(self) -> None:
# # # #         if not self._violations:
# # # #             return
# # # #         self._violations.sort(key=lambda x: x.timestamp)
# # # #         merged: List[_Violation] = []
# # # #         group  = [self._violations[0]]
# # # #         for v in self._violations[1:]:
# # # #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# # # #                 group.append(v)
# # # #             else:
# # # #                 merged.append(self._merge_group(group))
# # # #                 group = [v]
# # # #         merged.append(self._merge_group(group))
# # # #         self._violations = merged

# # # #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# # # #         base             = group[0]
# # # #         events: List[str]  = []
# # # #         factors: List[str] = []
# # # #         max_risk   = base.risk_score
# # # #         risk_level = base.risk_level
# # # #         best_frame = base.annotated_frame
# # # #         for v in group:
# # # #             events.extend(v.events)
# # # #             factors.extend(v.factors)
# # # #             if v.risk_score > max_risk:
# # # #                 max_risk, risk_level = v.risk_score, v.risk_level
# # # #             if best_frame is None and v.annotated_frame is not None:
# # # #                 best_frame = v.annotated_frame
# # # #         return _Violation(
# # # #             timestamp        = base.timestamp,
# # # #             time_str         = base.time_str,
# # # #             frame_index      = base.frame_index,
# # # #             type             = base.type,
# # # #             events           = list(set(events)),
# # # #             severity         = base.severity,
# # # #             duration         = base.duration,
# # # #             risk_score       = max_risk,
# # # #             risk_level       = risk_level,
# # # #             confidence       = base.confidence,
# # # #             factors          = list(set(factors)),
# # # #             source_filename  = base.source_filename,
# # # #             local_time_str   = base.local_time_str,
# # # #             annotated_frame  = best_frame,
# # # #         )

# # # #     # ── Private — frame extraction & saving ──────────────────────────────────

# # # #     def extract_violation_frames(self, video_path: str) -> None:
# # # #         print("[ViolationStore] Saving frames...")
# # # #         need_video = [v for v in self._violations if v.annotated_frame is None]
# # # #         saved = 0

# # # #         # First pass: save violations that already have an annotated frame
# # # #         for v in self._violations:
# # # #             if v.annotated_frame is not None:
# # # #                 v.frame_path      = self._save_frame(
# # # #                     v.annotated_frame, v.events, v.time_str, v.frame_index
# # # #                 )
# # # #                 v.annotated_frame = None   # free memory
# # # #                 saved += 1

# # # #         # Second pass: re-read from the source video for any that are missing
# # # #         if need_video:
# # # #             cap = cv2.VideoCapture(video_path)
# # # #             if not cap.isOpened():
# # # #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# # # #             else:
# # # #                 seen: set = set()
# # # #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# # # #                     if v.frame_index in seen:
# # # #                         continue
# # # #                     seen.add(v.frame_index)
# # # #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# # # #                     ret, frame = cap.read()
# # # #                     if not ret:
# # # #                         continue
# # # #                     v.frame_path = self._save_frame(
# # # #                         frame, v.events, v.time_str, v.frame_index
# # # #                     )
# # # #                     saved += 1
# # # #                 cap.release()

# # # #         print(f"[ViolationStore] {saved} frames saved")

# # # #     def _save_frame(
# # # #         self,
# # # #         frame:       np.ndarray,
# # # #         events:      List[str],
# # # #         time_str:    str,
# # # #         frame_index: int,
# # # #     ) -> str:
# # # #         """
# # # #         Save a single violation frame as a JPEG.

# # # #         Filename format:  <events>_<HH-MM-SS>_f<frame_index>.jpg
# # # #         Example:          seat_absence_00-01-14_f2106.jpg
# # # #                           seat_absence_drowsy_00-03-02_f5058.jpg
# # # #         """
# # # #         distraction   = "_".join(sorted(events))   # sorted for deterministic name
# # # #         filename_time = time_str.replace(":", "-")
# # # #         filename      = f"{distraction}_{filename_time}_f{frame_index}.jpg"
# # # #         path          = os.path.join(self.frames_dir, filename)
# # # #         ok = cv2.imwrite(
# # # #             path,
# # # #             cv2.resize(frame, (640, 360)),
# # # #             [cv2.IMWRITE_JPEG_QUALITY, 85],
# # # #         )
# # # #         if not ok:
# # # #             print(f"[ViolationStore] imwrite failed: {path}")
# # # #         return os.path.join(self.analysis_id, "frames", filename)

# # # #     # ── Private — report builder ──────────────────────────────────────────────

# # # #     def _build_report(self, processing_time: float = 0.0) -> dict:
# # # #         return {
# # # #             "analysis_id":     self.analysis_id,
# # # #             "train_detail_id": self.train_detail_id,
# # # #             "processing_time": round(processing_time, 3),
# # # #             # Single video → keep as dict for backwards compat; batch → list
# # # #             "video_info": (
# # # #                 self.video_infos[0]
# # # #                 if len(self.video_infos) == 1
# # # #                 else self.video_infos
# # # #             ),
# # # #             "violations": [
# # # #                 {
# # # #                     "timestamp":   v.time_str,
# # # #                     "frame_index": v.frame_index,
# # # #                     "events":      v.events,
# # # #                     "severity":    v.severity,
# # # #                     "duration":    v.duration,
# # # #                     "risk_score":  v.risk_score,
# # # #                     "risk_level":  v.risk_level,
# # # #                     "confidence":  v.confidence,
# # # #                     "factors":     v.factors,
# # # #                     "original_video_timestamp": (
# # # #                         f"{v.source_filename} {v.local_time_str}"
# # # #                     ),
# # # #                     "frame_path":  v.frame_path,
# # # #                 }
# # # #                 for v in self._violations
# # # #             ],
# # # #         }


# # # from __future__ import annotations

# # # import json
# # # import os
# # # from dataclasses import dataclass
# # # from typing import Any, Dict, List, Optional

# # # import cv2
# # # import numpy as np

# # # OUTPUTS_ROOT = "outputs"
# # # MERGE_WINDOW = 2.0


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # # INTERNAL DATA CLASS
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # @dataclass
# # # class _Violation:
# # #     timestamp:       float
# # #     time_str:        str
# # #     frame_index:     int
# # #     type:            str
# # #     events:          List[str]
# # #     severity:        str
# # #     duration:        float
# # #     risk_score:      int
# # #     risk_level:      str
# # #     confidence:      float
# # #     factors:         List[str]
# # #     source_filename: str                  = ""    # DB filename e.g. "mobile.mp4"
# # #     local_time_str:  str                  = ""    # local time within that file e.g. "00:00:23"
# # #     frame_path:      Optional[str]        = None
# # #     annotated_frame: Optional[np.ndarray] = None


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # # VIOLATION STORE
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # class ViolationStore:
# # #     """
# # #     Accumulates all violations found across one analysis run (single video
# # #     or a multi-video batch that shares the same analysis_id / folder_name).

# # #     Batch mode usage (api.py)
# # #     ─────────────────────────
# # #     1. Construct ONCE for the whole folder (no video_info in __init__).
# # #     2. Pass as shared_vstore= to each GadgetDetectionPipeline.
# # #        The pipeline calls add_video_info() automatically.
# # #     3. Call finalize() ONCE after all videos in the folder are done.

# # #     Standalone mode usage (CLI / single video)
# # #     ──────────────────────────────────────────
# # #     1. Construct with video_info= for the single video.
# # #     2. Pipeline calls finalize() automatically at the end of run().
# # #     """

# # #     def __init__(
# # #         self,
# # #         analysis_id:     str,
# # #         train_detail_id: int,
# # #         video_info:      Optional[Dict[str, Any]] = None,
# # #     ):
# # #         self.analysis_id     = analysis_id
# # #         self.train_detail_id = train_detail_id
# # #         # Always a list — 0 entries until add_video_info() is called (batch mode),
# # #         # or 1 entry when video_info is provided (standalone mode).
# # #         self.video_infos: List[Dict[str, Any]] = (
# # #             [video_info] if video_info is not None else []
# # #         )

# # #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# # #         self.frames_dir = os.path.join(self.output_dir, "frames")
# # #         os.makedirs(self.frames_dir, exist_ok=True)

# # #         self._violations:  List[_Violation] = []
# # #         self._seen_frames: set              = set()
# # #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# # #     # ── Public helpers ────────────────────────────────────────────────────────

# # #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# # #         """Append one video's metadata. Called once per video in batch mode."""
# # #         self.video_infos.append(video_info)

# # #     def record_violation(
# # #         self,
# # #         annotated_frame:  np.ndarray,
# # #         video_time:       float,          # global timestamp (offset-adjusted)
# # #         frame_index:      int,            # global frame index (offset-adjusted)
# # #         event_type:       str,
# # #         original_frame:   Optional[np.ndarray] = None,
# # #         severity:         str   = "CRITICAL",
# # #         confidence:       float = 0.9,
# # #         risk_score:       int   = 80,
# # #         risk_level:       str   = "CRITICAL",
# # #         factors:          Optional[List[str]] = None,
# # #         duration:         float = 0.0,
# # #         source_filename:  str   = "",     # DB filename shown in original_video_timestamp
# # #         local_video_time: float = -1.0,   # local time within the source file; -1 = same as video_time
# # #     ) -> None:
# # #         """
# # #         Record one distraction event.

# # #         Deduplication key is (frame_index, event_type):
# # #           • same event on the same global frame is recorded once
# # #           • different events on the same global frame are each recorded
# # #           • global frame_index is unique across videos (frame_offset applied in main.py)
# # #         """
# # #         dedup_key = (frame_index, event_type)
# # #         if dedup_key in self._seen_frames:
# # #             return
# # #         self._seen_frames.add(dedup_key)

# # #         factors  = factors or []
# # #         t        = int(round(video_time))
# # #         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

# # #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# # #         local_str = (
# # #             f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
# # #         )

# # #         self._violations.append(
# # #             _Violation(
# # #                 timestamp        = video_time,
# # #                 time_str         = time_str,
# # #                 frame_index      = frame_index,
# # #                 type             = event_type,
# # #                 events           = [event_type],
# # #                 severity         = severity,
# # #                 duration         = round(duration, 2),
# # #                 risk_score       = risk_score,
# # #                 risk_level       = risk_level,
# # #                 confidence       = round(confidence, 3),
# # #                 factors          = list(factors),
# # #                 source_filename  = source_filename,
# # #                 local_time_str   = local_str,
# # #                 annotated_frame  = (
# # #                     annotated_frame.copy() if annotated_frame is not None else None
# # #                 ),
# # #             )
# # #         )

# # #     # ── Finalize ──────────────────────────────────────────────────────────────

# # #     def finalize(self, processing_time: float = 0.0) -> str:
# # #         """
# # #         Deduplicate → merge → save frames → write JSON → upload to S3/DB.
# # #         Returns the local path to analysis_report.json.
# # #         """
# # #         self._deduplicate_by_frame()
# # #         self._merge_by_time_window()

# # #         # Extract frames from every video in the batch (or the single video)
# # #         for vi in self.video_infos:
# # #             if vi and vi.get("videoPath"):
# # #                 self.extract_violation_frames(vi["videoPath"])

# # #         report   = self._build_report(processing_time=processing_time)
# # #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# # #         with open(out_path, "w", encoding="utf-8") as f:
# # #             json.dump(report, f, indent=2)

# # #         print(f"[ViolationStore] JSON report     : {out_path}")
# # #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# # #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# # #         try:
# # #             from utils.db_s3_uploader import finalize_and_upload
# # #             finalize_and_upload(
# # #                 report_path     = out_path,
# # #                 analysis_id     = self.analysis_id,
# # #                 train_detail_id = self.train_detail_id,
# # #             )
# # #         except Exception as exc:
# # #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# # #         return out_path

# # #     # ── Private — deduplication & merging ────────────────────────────────────

# # #     def _deduplicate_by_frame(self) -> None:
# # #         unique: Dict[int, _Violation] = {}
# # #         for v in self._violations:
# # #             if v.frame_index not in unique:
# # #                 unique[v.frame_index] = v
# # #             else:
# # #                 ex = unique[v.frame_index]
# # #                 ex.events  = list(set(ex.events  + v.events))
# # #                 ex.factors = list(set(ex.factors + v.factors))
# # #                 if v.risk_score > ex.risk_score:
# # #                     ex.risk_score = v.risk_score
# # #                     ex.risk_level = v.risk_level
# # #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# # #                     ex.annotated_frame = v.annotated_frame
# # #         self._violations = list(unique.values())

# # #     def _merge_by_time_window(self) -> None:
# # #         if not self._violations:
# # #             return
# # #         self._violations.sort(key=lambda x: x.timestamp)
# # #         merged: List[_Violation] = []
# # #         group  = [self._violations[0]]
# # #         for v in self._violations[1:]:
# # #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# # #                 group.append(v)
# # #             else:
# # #                 merged.append(self._merge_group(group))
# # #                 group = [v]
# # #         merged.append(self._merge_group(group))
# # #         self._violations = merged

# # #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# # #         base             = group[0]
# # #         events: List[str]  = []
# # #         factors: List[str] = []
# # #         max_risk   = base.risk_score
# # #         risk_level = base.risk_level
# # #         best_frame = base.annotated_frame
# # #         for v in group:
# # #             events.extend(v.events)
# # #             factors.extend(v.factors)
# # #             if v.risk_score > max_risk:
# # #                 max_risk, risk_level = v.risk_score, v.risk_level
# # #             if best_frame is None and v.annotated_frame is not None:
# # #                 best_frame = v.annotated_frame
# # #         return _Violation(
# # #             timestamp        = base.timestamp,
# # #             time_str         = base.time_str,
# # #             frame_index      = base.frame_index,
# # #             type             = base.type,
# # #             events           = list(set(events)),
# # #             severity         = base.severity,
# # #             duration         = base.duration,
# # #             risk_score       = max_risk,
# # #             risk_level       = risk_level,
# # #             confidence       = base.confidence,
# # #             factors          = list(set(factors)),
# # #             source_filename  = base.source_filename,
# # #             local_time_str   = base.local_time_str,
# # #             annotated_frame  = best_frame,
# # #         )

# # #     # ── Private — frame extraction & saving ──────────────────────────────────

# # #     def extract_violation_frames(self, video_path: str) -> None:
# # #         print("[ViolationStore] Saving frames...")
# # #         need_video = [v for v in self._violations if v.annotated_frame is None]
# # #         saved = 0

# # #         # First pass: save violations that already have an annotated frame
# # #         for v in self._violations:
# # #             if v.annotated_frame is not None:
# # #                 v.frame_path      = self._save_frame(v.annotated_frame, v.events, v.time_str)
# # #                 v.annotated_frame = None   # free memory
# # #                 saved += 1

# # #         # Second pass: re-read from the source video for any that are missing
# # #         if need_video:
# # #             cap = cv2.VideoCapture(video_path)
# # #             if not cap.isOpened():
# # #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# # #             else:
# # #                 seen: set = set()
# # #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# # #                     if v.frame_index in seen:
# # #                         continue
# # #                     seen.add(v.frame_index)
# # #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# # #                     ret, frame = cap.read()
# # #                     if not ret:
# # #                         continue
# # #                     v.frame_path = self._save_frame(frame, v.events, v.time_str)
# # #                     saved += 1
# # #                 cap.release()

# # #         print(f"[ViolationStore] {saved} frames saved")

# # #     def _save_frame(
# # #         self,
# # #         frame:    np.ndarray,
# # #         events:   List[str],
# # #         time_str: str,
# # #     ) -> str:
# # #         """
# # #         Save a single violation frame as JPEG.

# # #         Filename format:  <events>_<HH-MM-SS>.jpg
# # #         Example:          seat_absence_00-01-14.jpg
# # #                           seat_absence_drowsy_00-03-02.jpg
# # #                           phone_use_00-00-24.jpg
# # #         """
# # #         distraction   = "_".join(sorted(events))   # sorted for deterministic name
# # #         filename_time = time_str.replace(":", "-")
# # #         filename      = f"{distraction}_{filename_time}.jpg"
# # #         path          = os.path.join(self.frames_dir, filename)
# # #         ok = cv2.imwrite(
# # #             path,
# # #             cv2.resize(frame, (640, 360)),
# # #             [cv2.IMWRITE_JPEG_QUALITY, 85],
# # #         )
# # #         if not ok:
# # #             print(f"[ViolationStore] imwrite failed: {path}")
# # #         return os.path.join(self.analysis_id, "frames", filename)

# # #     # ── Private — report builder ──────────────────────────────────────────────

# # #     def _build_report(self, processing_time: float = 0.0) -> dict:
# # #         return {
# # #             "analysis_id":     self.analysis_id,
# # #             "train_detail_id": self.train_detail_id,
# # #             "processing_time": round(processing_time, 3),
# # #             # Single video → dict (backwards compat); batch → list
# # #             "video_info": (
# # #                 self.video_infos[0]
# # #                 if len(self.video_infos) == 1
# # #                 else self.video_infos
# # #             ),
# # #             "violations": [
# # #                 {
# # #                     "timestamp":   v.time_str,
# # #                     "frame_index": v.frame_index,
# # #                     "events":      v.events,
# # #                     "severity":    v.severity,
# # #                     "duration":    v.duration,
# # #                     "risk_score":  v.risk_score,
# # #                     "risk_level":  v.risk_level,
# # #                     "confidence":  v.confidence,
# # #                     "factors":     v.factors,
# # #                     # "original_video_timestamp" = "<db_filename> <local_HH:MM:SS>"
# # #                     # When timestamp == original (video 1), local_time_str == time_str
# # #                     "original_video_timestamp": (
# # #                         f"{v.source_filename} {v.local_time_str}"
# # #                     ),
# # #                     "frame_path":  v.frame_path,
# # #                 }
# # #                 for v in self._violations
# # #             ],
# # #         }


# # from __future__ import annotations

# # import json
# # import os
# # from dataclasses import dataclass
# # from typing import Any, Dict, List, Optional

# # import cv2
# # import numpy as np

# # OUTPUTS_ROOT = "outputs"
# # MERGE_WINDOW = 2.0


# # # ══════════════════════════════════════════════════════════════════════════════
# # # INTERNAL DATA CLASS
# # # ══════════════════════════════════════════════════════════════════════════════

# # @dataclass
# # class _Violation:
# #     timestamp:       float
# #     time_str:        str
# #     frame_index:     int
# #     type:            str
# #     events:          List[str]
# #     severity:        str
# #     duration:        float
# #     risk_score:      int
# #     risk_level:      str
# #     confidence:      float
# #     factors:         List[str]
# #     source_filename: str                  = ""    # DB filename e.g. "mobile.mp4"
# #     local_time_str:  str                  = ""    # local time within that file e.g. "00:00:23"
# #     frame_path:      Optional[str]        = None
# #     annotated_frame: Optional[np.ndarray] = None


# # # ══════════════════════════════════════════════════════════════════════════════
# # # VIOLATION STORE
# # # ══════════════════════════════════════════════════════════════════════════════

# # class ViolationStore:
# #     """
# #     Accumulates all violations found across one analysis run (single video
# #     or a multi-video batch that shares the same analysis_id / folder_name).

# #     Batch mode usage (api.py)
# #     ─────────────────────────
# #     1. Construct ONCE for the whole folder (no video_info in __init__).
# #     2. Pass as shared_vstore= to each GadgetDetectionPipeline.
# #        The pipeline calls add_video_info() automatically.
# #     3. Call finalize() ONCE after all videos in the folder are done.

# #     Standalone mode usage (CLI / single video)
# #     ──────────────────────────────────────────
# #     1. Construct with video_info= for the single video.
# #     2. Pipeline calls finalize() automatically at the end of run().
# #     """

# #     def __init__(
# #         self,
# #         analysis_id:     str,
# #         train_detail_id: int,
# #         video_info:      Optional[Dict[str, Any]] = None,
# #     ):
# #         self.analysis_id     = analysis_id
# #         self.train_detail_id = train_detail_id
# #         # Always a list — 0 entries until add_video_info() is called (batch mode),
# #         # or 1 entry when video_info is provided (standalone mode).
# #         self.video_infos: List[Dict[str, Any]] = (
# #             [video_info] if video_info is not None else []
# #         )

# #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# #         self.frames_dir = os.path.join(self.output_dir, "frames")
# #         os.makedirs(self.frames_dir, exist_ok=True)

# #         self._violations:  List[_Violation] = []
# #         self._seen_frames: set              = set()
# #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# #     # ── Public helpers ────────────────────────────────────────────────────────

# #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# #         """Append one video's metadata. Called once per video in batch mode."""
# #         self.video_infos.append(video_info)

# #     def record_violation(
# #         self,
# #         annotated_frame:  np.ndarray,
# #         video_time:       float,          # global timestamp (offset-adjusted)
# #         frame_index:      int,            # global frame index (offset-adjusted)
# #         event_type:       str,
# #         original_frame:   Optional[np.ndarray] = None,
# #         severity:         str   = "CRITICAL",
# #         confidence:       float = 0.9,
# #         risk_score:       int   = 80,
# #         risk_level:       str   = "CRITICAL",
# #         factors:          Optional[List[str]] = None,
# #         duration:         float = 0.0,
# #         source_filename:  str   = "",     # DB filename shown in original_video_timestamp
# #         local_video_time: float = -1.0,   # local time within the source file; -1 = same as video_time
# #     ) -> None:
# #         """
# #         Record one distraction event.

# #         Deduplication key is (frame_index, event_type):
# #           • same event on the same global frame is recorded once
# #           • different events on the same global frame are each recorded
# #           • global frame_index is unique across videos (frame_offset applied in main.py)
# #         """
# #         dedup_key = (frame_index, event_type)
# #         if dedup_key in self._seen_frames:
# #             return
# #         self._seen_frames.add(dedup_key)

# #         factors  = factors or []
# #         t        = int(round(video_time))
# #         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

# #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# #         local_str = (
# #             f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
# #         )

# #         self._violations.append(
# #             _Violation(
# #                 timestamp        = video_time,
# #                 time_str         = time_str,
# #                 frame_index      = frame_index,
# #                 type             = event_type,
# #                 events           = [event_type],
# #                 severity         = severity,
# #                 duration         = round(duration, 2),
# #                 risk_score       = risk_score,
# #                 risk_level       = risk_level,
# #                 confidence       = round(confidence, 3),
# #                 factors          = list(factors),
# #                 source_filename  = source_filename,
# #                 local_time_str   = local_str,
# #                 annotated_frame  = (
# #                     annotated_frame.copy() if annotated_frame is not None else None
# #                 ),
# #             )
# #         )

# #     # ── Finalize ──────────────────────────────────────────────────────────────

# #     def finalize(self, processing_time: float = 0.0) -> str:
# #         """
# #         Deduplicate → merge → save frames → write JSON → upload to S3/DB.
# #         Returns the local path to analysis_report.json.
# #         """
# #         self._deduplicate_by_frame()
# #         self._merge_by_time_window()

# #         # Extract frames from every video in the batch (or the single video)
# #         for vi in self.video_infos:
# #             if vi and vi.get("videoPath"):
# #                 self.extract_violation_frames(vi["videoPath"])

# #         report   = self._build_report(processing_time=processing_time)
# #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# #         with open(out_path, "w", encoding="utf-8") as f:
# #             json.dump(report, f, indent=2)

# #         print(f"[ViolationStore] JSON report     : {out_path}")
# #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# #         try:
# #             from utils.db_s3_uploader import finalize_and_upload
# #             finalize_and_upload(
# #                 report_path     = out_path,
# #                 analysis_id     = self.analysis_id,
# #                 train_detail_id = self.train_detail_id,
# #             )
# #         except Exception as exc:
# #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# #         return out_path

# #     # ── Private — deduplication & merging ────────────────────────────────────

# #     def _deduplicate_by_frame(self) -> None:
# #         unique: Dict[int, _Violation] = {}
# #         for v in self._violations:
# #             if v.frame_index not in unique:
# #                 unique[v.frame_index] = v
# #             else:
# #                 ex = unique[v.frame_index]
# #                 ex.events  = list(set(ex.events  + v.events))
# #                 ex.factors = list(set(ex.factors + v.factors))
# #                 if v.risk_score > ex.risk_score:
# #                     ex.risk_score = v.risk_score
# #                     ex.risk_level = v.risk_level
# #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# #                     ex.annotated_frame = v.annotated_frame
# #         self._violations = list(unique.values())

# #     def _merge_by_time_window(self) -> None:
# #         if not self._violations:
# #             return
# #         self._violations.sort(key=lambda x: x.timestamp)
# #         merged: List[_Violation] = []
# #         group  = [self._violations[0]]
# #         for v in self._violations[1:]:
# #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# #                 group.append(v)
# #             else:
# #                 merged.append(self._merge_group(group))
# #                 group = [v]
# #         merged.append(self._merge_group(group))
# #         self._violations = merged

# #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# #         base             = group[0]
# #         events: List[str]  = []
# #         factors: List[str] = []
# #         max_risk   = base.risk_score
# #         risk_level = base.risk_level
# #         best_frame = base.annotated_frame
# #         for v in group:
# #             events.extend(v.events)
# #             factors.extend(v.factors)
# #             if v.risk_score > max_risk:
# #                 max_risk, risk_level = v.risk_score, v.risk_level
# #             if best_frame is None and v.annotated_frame is not None:
# #                 best_frame = v.annotated_frame
# #         return _Violation(
# #             timestamp        = base.timestamp,
# #             time_str         = base.time_str,
# #             frame_index      = base.frame_index,
# #             type             = base.type,
# #             events           = list(set(events)),
# #             severity         = base.severity,
# #             duration         = base.duration,
# #             risk_score       = max_risk,
# #             risk_level       = risk_level,
# #             confidence       = base.confidence,
# #             factors          = list(set(factors)),
# #             source_filename  = base.source_filename,
# #             local_time_str   = base.local_time_str,
# #             annotated_frame  = best_frame,
# #         )

# #     # ── Private — frame extraction & saving ──────────────────────────────────

# #     def extract_violation_frames(self, video_path: str) -> None:
# #         print("[ViolationStore] Saving frames...")
# #         need_video = [v for v in self._violations if v.annotated_frame is None]
# #         saved = 0

# #         # First pass: save violations that already have an annotated frame
# #         for v in self._violations:
# #             if v.annotated_frame is not None:
# #                 v.frame_path      = self._save_frame(v.annotated_frame, v.events, v.time_str)
# #                 v.annotated_frame = None   # free memory
# #                 saved += 1

# #         # Second pass: re-read from the source video for any that are missing
# #         if need_video:
# #             cap = cv2.VideoCapture(video_path)
# #             if not cap.isOpened():
# #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# #             else:
# #                 seen: set = set()
# #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# #                     if v.frame_index in seen:
# #                         continue
# #                     seen.add(v.frame_index)
# #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# #                     ret, frame = cap.read()
# #                     if not ret:
# #                         continue
# #                     v.frame_path = self._save_frame(frame, v.events, v.time_str)
# #                     saved += 1
# #                 cap.release()

# #         print(f"[ViolationStore] {saved} frames saved")

# #     def _save_frame(
# #         self,
# #         frame:    np.ndarray,
# #         events:   List[str],
# #         time_str: str,
# #     ) -> str:
# #         """
# #         Save a single violation frame as JPEG.

# #         Filename format:  <events>_<HH-MM-SS>.jpg
# #         Example:          seat_absence_00-01-14.jpg
# #                           seat_absence_drowsy_00-03-02.jpg
# #                           phone_use_00-00-24.jpg
# #         """
# #         distraction   = "_".join(sorted(events))   # sorted for deterministic name
# #         filename_time = time_str.replace(":", "-")
# #         filename      = f"{distraction}_{filename_time}.jpg"
# #         path          = os.path.join(self.frames_dir, filename)
# #         ok = cv2.imwrite(
# #             path,
# #             cv2.resize(frame, (640, 360)),
# #             [cv2.IMWRITE_JPEG_QUALITY, 85],
# #         )
# #         if not ok:
# #             print(f"[ViolationStore] imwrite failed: {path}")
# #         return os.path.join(self.analysis_id, "frames", filename)

# #     # ── Private — report builder ──────────────────────────────────────────────

# #     def _build_report(self, processing_time: float = 0.0) -> dict:
# #         return {
# #             "analysis_id":     self.analysis_id,
# #             "train_detail_id": self.train_detail_id,
# #             "processing_time": round(processing_time, 3),
# #             # Single video → dict (backwards compat); batch → list
# #             "video_info": (
# #                 self.video_infos[0]
# #                 if len(self.video_infos) == 1
# #                 else self.video_infos
# #             ),
# #             "violations": [
# #                 {
# #                     "timestamp":   v.time_str,
# #                     "frame_index": v.frame_index,
# #                     "events":      v.events,
# #                     "severity":    v.severity,
# #                     "duration":    v.duration,
# #                     "risk_score":  v.risk_score,
# #                     "risk_level":  v.risk_level,
# #                     "confidence":  v.confidence,
# #                     "factors":     v.factors,
# #                     # "original_video_timestamp" = "<db_filename> <local_HH:MM:SS>"
# #                     # When timestamp == original (video 1), local_time_str == time_str
# #                     "original_video_timestamp": (
# #                         f"{v.source_filename} {v.local_time_str}"
# #                     ),
# #                     "frame_path":  v.frame_path,
# #                 }
# #                 for v in self._violations
# #             ],
# #         }


# # # from __future__ import annotations

# # # import json
# # # import os
# # # from dataclasses import dataclass
# # # from typing import Any, Dict, List, Optional

# # # import cv2
# # # import numpy as np

# # # OUTPUTS_ROOT = "outputs"
# # # MERGE_WINDOW = 2.0


# # # @dataclass
# # # class _Violation:
# # #     timestamp:       float
# # #     time_str:        str
# # #     frame_index:     int
# # #     type:            str
# # #     events:          List[str]
# # #     severity:        str
# # #     duration:        float
# # #     risk_score:      int
# # #     risk_level:      str
# # #     confidence:      float
# # #     factors:         List[str]
# # #     source_filename: str                   = ""    # e.g. "ax.mp4"
# # #     local_time_str:  str                   = ""    # local time within that file e.g. "00:00:18"
# # #     frame_path:      Optional[str]         = None
# # #     annotated_frame: Optional[np.ndarray]  = None


# # # class ViolationStore:

# # #     def __init__(self, analysis_id: str, train_detail_id: int, video_info: Optional[Dict[str, Any]] = None):
# # #         self.analysis_id     = analysis_id
# # #         self.train_detail_id = train_detail_id
# # #         # video_infos is always a list — 1 entry for single-video, N entries for batch
# # #         self.video_infos: List[Dict[str, Any]] = [video_info] if video_info is not None else []

# # #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# # #         self.frames_dir = os.path.join(self.output_dir, "frames")
# # #         os.makedirs(self.frames_dir, exist_ok=True)

# # #         self._violations: List[_Violation] = []
# # #         self._seen_frames: set             = set()
# # #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# # #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# # #         """
# # #         Register one video's metadata into the shared store.
# # #         Called once per video in batch mode (api.py passes shared_vstore).
# # #         """
# # #         self.video_infos.append(video_info)

# # #     def record_violation(
# # #         self,
# # #         annotated_frame: np.ndarray,
# # #         video_time:      float,
# # #         frame_index:     int,
# # #         event_type:      str,
# # #         original_frame:  Optional[np.ndarray] = None,
# # #         severity:        str   = "CRITICAL",
# # #         confidence:      float = 0.9,
# # #         risk_score:      int   = 80,
# # #         risk_level:      str   = "CRITICAL",
# # #         factors:         Optional[List[str]] = None,
# # #         duration:        float = 0.0,
# # #         source_filename: str   = "",   # original upload filename e.g. "ax.mp4"
# # #         local_video_time: float = -1.0, # raw video_time before offset; -1 = same as video_time
# # #     ):
# # #         # Deduplicate on (frame_index, event_type) so that:
# # #         #  • the same violation type on the same frame is recorded only once
# # #         #  • different violation types on the same frame are each recorded
# # #         #  • frame numbers from different videos never collide (frame_offset
# # #         #    in main.py makes every global frame_index unique across the batch)
# # #         dedup_key = (frame_index, event_type)
# # #         if dedup_key in self._seen_frames:
# # #             return
# # #         self._seen_frames.add(dedup_key)
# # #         factors   = factors or []
# # #         t         = int(round(video_time))
# # #         time_str  = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"
# # #         # Build the per-file local timestamp (time within the source video)
# # #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# # #         local_str = f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
# # #         self._violations.append(_Violation(
# # #             timestamp        = video_time,
# # #             time_str         = time_str,
# # #             frame_index      = frame_index,
# # #             type             = event_type,
# # #             events           = [event_type],
# # #             severity         = severity,
# # #             duration         = round(duration, 2),
# # #             risk_score       = risk_score,
# # #             risk_level       = risk_level,
# # #             confidence       = round(confidence, 3),
# # #             factors          = list(factors),
# # #             source_filename  = source_filename,
# # #             local_time_str   = local_str,
# # #             annotated_frame  = annotated_frame.copy() if annotated_frame is not None else None,
# # #         ))

# # #     def _deduplicate_by_frame(self):
# # #         unique: Dict[int, _Violation] = {}
# # #         for v in self._violations:
# # #             if v.frame_index not in unique:
# # #                 unique[v.frame_index] = v
# # #             else:
# # #                 ex = unique[v.frame_index]
# # #                 ex.events  = list(set(ex.events  + v.events))
# # #                 ex.factors = list(set(ex.factors + v.factors))
# # #                 if v.risk_score > ex.risk_score:
# # #                     ex.risk_score = v.risk_score
# # #                     ex.risk_level = v.risk_level
# # #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# # #                     ex.annotated_frame = v.annotated_frame
# # #         self._violations = list(unique.values())

# # #     def _merge_by_time_window(self):
# # #         if not self._violations:
# # #             return
# # #         self._violations.sort(key=lambda x: x.timestamp)
# # #         merged = []
# # #         group  = [self._violations[0]]
# # #         for v in self._violations[1:]:
# # #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# # #                 group.append(v)
# # #             else:
# # #                 merged.append(self._merge_group(group))
# # #                 group = [v]
# # #         merged.append(self._merge_group(group))
# # #         self._violations = merged

# # #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# # #         base            = group[0]
# # #         events, factors = [], []
# # #         max_risk        = base.risk_score
# # #         risk_level      = base.risk_level
# # #         best_frame      = base.annotated_frame
# # #         for v in group:
# # #             events.extend(v.events)
# # #             factors.extend(v.factors)
# # #             if v.risk_score > max_risk:
# # #                 max_risk, risk_level = v.risk_score, v.risk_level
# # #             if best_frame is None and v.annotated_frame is not None:
# # #                 best_frame = v.annotated_frame
# # #         return _Violation(
# # #             timestamp        = base.timestamp,
# # #             time_str         = base.time_str,
# # #             frame_index      = base.frame_index,
# # #             type             = base.type,
# # #             events           = list(set(events)),
# # #             severity         = base.severity,
# # #             duration         = base.duration,
# # #             risk_score       = max_risk,
# # #             risk_level       = risk_level,
# # #             confidence       = base.confidence,
# # #             factors          = list(set(factors)),
# # #             source_filename  = base.source_filename,
# # #             local_time_str   = base.local_time_str,
# # #             annotated_frame  = best_frame,
# # #         )

# # #     def extract_violation_frames(self, video_path: str):
# # #         print("[ViolationStore] Saving frames...")
# # #         need_video = [v for v in self._violations if v.annotated_frame is None]
# # #         saved = 0
# # #         for v in self._violations:
# # #             if v.annotated_frame is not None:
# # #                 v.frame_path      = self._save_frame(v.annotated_frame, v.events, v.time_str)
# # #                 v.annotated_frame = None
# # #                 saved += 1
# # #         if need_video:
# # #             cap = cv2.VideoCapture(video_path)
# # #             if not cap.isOpened():
# # #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# # #             else:
# # #                 seen: set = set()
# # #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# # #                     if v.frame_index in seen:
# # #                         continue
# # #                     seen.add(v.frame_index)
# # #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# # #                     ret, frame = cap.read()
# # #                     if not ret:
# # #                         continue
# # #                     v.frame_path = self._save_frame(frame, v.events, v.time_str)
# # #                     saved += 1
# # #                 cap.release()
# # #         print(f"[ViolationStore] {saved} frames saved")

# # #     def _save_frame(self, frame: np.ndarray, events: List[str], time_str: str) -> str:
# # #         distraction   = "_".join(events)
# # #         filename_time = time_str.replace(":", "-")
# # #         filename      = f"{distraction}_{filename_time}.jpg"
# # #         path          = os.path.join(self.frames_dir, filename)
# # #         ok = cv2.imwrite(path, cv2.resize(frame, (640, 360)),
# # #                          [cv2.IMWRITE_JPEG_QUALITY, 85])
# # #         if not ok:
# # #             print(f"[ViolationStore] imwrite failed: {path}")
# # #         return os.path.join(self.analysis_id, "frames", filename)

# # #     def _build_report(self, processing_time: float = 0.0) -> dict:
# # #         return {
# # #             "analysis_id":     self.analysis_id,
# # #             "train_detail_id": self.train_detail_id,
# # #             "processing_time": round(processing_time, 3),
# # #             # Single video → keep as dict for backwards compat; batch → list
# # #             "video_info": self.video_infos[0] if len(self.video_infos) == 1
# # #                           else self.video_infos,
# # #             "violations": [
# # #                 {
# # #                     "timestamp":   v.time_str,
# # #                     "frame_index": v.frame_index,
# # #                     "events":      v.events,
# # #                     "severity":    v.severity,
# # #                     "duration":    v.duration,
# # #                     "risk_score":  v.risk_score,
# # #                     "risk_level":  v.risk_level,
# # #                     "confidence":  v.confidence,
# # #                     "factors":     v.factors,
# # #                     "original_video_timestamp": f"{v.source_filename} {v.local_time_str}",
# # #                     "frame_path":  v.frame_path,
# # #                 }
# # #                 for v in self._violations
# # #             ],
# # #         }

# # #     def finalize(self, processing_time: float = 0.0) -> str:
# # #         self._deduplicate_by_frame()
# # #         self._merge_by_time_window()
# # #         # Extract frames from every video in the batch (or the single video)
# # #         for vi in self.video_infos:
# # #             if vi and vi.get("videoPath"):
# # #                 self.extract_violation_frames(vi["videoPath"])
# # #         report   = self._build_report(processing_time=processing_time)
# # #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# # #         with open(out_path, "w", encoding="utf-8") as f:
# # #             json.dump(report, f, indent=2)
# # #         print(f"[ViolationStore] JSON report     : {out_path}")
# # #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# # #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# # #         try:
# # #             from utils.db_s3_uploader import finalize_and_upload
# # #             finalize_and_upload(
# # #                 report_path     = out_path,
# # #                 analysis_id     = self.analysis_id,
# # #                 train_detail_id = self.train_detail_id,
# # #             )
# # #         except Exception as exc:
# # #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# # #         return out_path



# # # utils/violation_store.py
# # # ─────────────────────────────────────────────────────────────────
# # # Change from original:
# # #   _save_frame() now appends _f{frame_index} to the filename so
# # #   two violations of the same type at the same timestamp never
# # #   silently overwrite each other.
# # #   e.g.  phone_use_00-13-16_f6762.jpg
# # #         seat_absence_00-00-17_f516.jpg
# # # ─────────────────────────────────────────────────────────────────

# # from __future__ import annotations

# # import json
# # import os
# # from dataclasses import dataclass
# # from typing import Any, Dict, List, Optional

# # import cv2
# # import numpy as np

# # OUTPUTS_ROOT = "outputs"
# # MERGE_WINDOW = 2.0


# # @dataclass
# # class _Violation:
# #     timestamp:       float
# #     time_str:        str
# #     frame_index:     int
# #     type:            str
# #     events:          List[str]
# #     severity:        str
# #     duration:        float
# #     risk_score:      int
# #     risk_level:      str
# #     confidence:      float
# #     factors:         List[str]
# #     source_filename: str                  = ""
# #     local_time_str:  str                  = ""
# #     frame_path:      Optional[str]        = None
# #     annotated_frame: Optional[np.ndarray] = None


# # class ViolationStore:

# #     def __init__(
# #         self,
# #         analysis_id:     str,
# #         train_detail_id: int,
# #         video_info:      Optional[Dict[str, Any]] = None,
# #     ):
# #         self.analysis_id     = analysis_id
# #         self.train_detail_id = train_detail_id
# #         self.video_infos: List[Dict[str, Any]] = (
# #             [video_info] if video_info is not None else []
# #         )

# #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# #         self.frames_dir = os.path.join(self.output_dir, "frames")
# #         os.makedirs(self.frames_dir, exist_ok=True)

# #         self._violations: List[_Violation] = []
# #         self._seen_frames: set             = set()
# #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# #     # ── called once per video in batch mode ──────────────────────
# #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# #         self.video_infos.append(video_info)

# #     # ── called from main.py for every detected violation ─────────
# #     def record_violation(
# #         self,
# #         annotated_frame:  np.ndarray,
# #         video_time:       float,
# #         frame_index:      int,
# #         event_type:       str,
# #         original_frame:   Optional[np.ndarray] = None,
# #         severity:         str   = "CRITICAL",
# #         confidence:       float = 0.9,
# #         risk_score:       int   = 80,
# #         risk_level:       str   = "CRITICAL",
# #         factors:          Optional[List[str]] = None,
# #         duration:         float = 0.0,
# #         source_filename:  str   = "",
# #         local_video_time: float = -1.0,
# #     ):
# #         # Deduplicate: same frame + same event type recorded only once.
# #         # Different event types on the same frame are each recorded.
# #         # frame_index is globally unique across batch (frame_offset in main.py).
# #         dedup_key = (frame_index, event_type)
# #         if dedup_key in self._seen_frames:
# #             return
# #         self._seen_frames.add(dedup_key)

# #         factors  = factors or []
# #         t        = int(round(video_time))
# #         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

# #         # Local timestamp = time within the source video file (before cumulative offset)
# #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# #         local_str = f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"

# #         self._violations.append(_Violation(
# #             timestamp        = video_time,
# #             time_str         = time_str,
# #             frame_index      = frame_index,
# #             type             = event_type,
# #             events           = [event_type],
# #             severity         = severity,
# #             duration         = round(duration, 2),
# #             risk_score       = risk_score,
# #             risk_level       = risk_level,
# #             confidence       = round(confidence, 3),
# #             factors          = list(factors),
# #             source_filename  = source_filename,
# #             local_time_str   = local_str,
# #             annotated_frame  = annotated_frame.copy() if annotated_frame is not None else None,
# #         ))

# #     # ─────────────────────────────────────────────────────────────

# #     def _deduplicate_by_frame(self):
# #         unique: Dict[int, _Violation] = {}
# #         for v in self._violations:
# #             if v.frame_index not in unique:
# #                 unique[v.frame_index] = v
# #             else:
# #                 ex = unique[v.frame_index]
# #                 ex.events  = list(set(ex.events  + v.events))
# #                 ex.factors = list(set(ex.factors + v.factors))
# #                 if v.risk_score > ex.risk_score:
# #                     ex.risk_score = v.risk_score
# #                     ex.risk_level = v.risk_level
# #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# #                     ex.annotated_frame = v.annotated_frame
# #         self._violations = list(unique.values())

# #     def _merge_by_time_window(self):
# #         if not self._violations:
# #             return
# #         self._violations.sort(key=lambda x: x.timestamp)
# #         merged = []
# #         group  = [self._violations[0]]
# #         for v in self._violations[1:]:
# #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# #                 group.append(v)
# #             else:
# #                 merged.append(self._merge_group(group))
# #                 group = [v]
# #         merged.append(self._merge_group(group))
# #         self._violations = merged

# #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# #         base            = group[0]
# #         events, factors = [], []
# #         max_risk        = base.risk_score
# #         risk_level      = base.risk_level
# #         best_frame      = base.annotated_frame
# #         for v in group:
# #             events.extend(v.events)
# #             factors.extend(v.factors)
# #             if v.risk_score > max_risk:
# #                 max_risk, risk_level = v.risk_score, v.risk_level
# #             if best_frame is None and v.annotated_frame is not None:
# #                 best_frame = v.annotated_frame
# #         return _Violation(
# #             timestamp        = base.timestamp,
# #             time_str         = base.time_str,
# #             frame_index      = base.frame_index,
# #             type             = base.type,
# #             events           = list(set(events)),
# #             severity         = base.severity,
# #             duration         = base.duration,
# #             risk_score       = max_risk,
# #             risk_level       = risk_level,
# #             confidence       = base.confidence,
# #             factors          = list(set(factors)),
# #             source_filename  = base.source_filename,
# #             local_time_str   = base.local_time_str,
# #             annotated_frame  = best_frame,
# #         )

# #     # ─────────────────────────────────────────────────────────────

# #     def extract_violation_frames(self, video_path: str):
# #         print("[ViolationStore] Saving frames...")
# #         need_video = [v for v in self._violations if v.annotated_frame is None]
# #         saved = 0

# #         for v in self._violations:
# #             if v.annotated_frame is not None:
# #                 v.frame_path      = self._save_frame(
# #                     v.annotated_frame, v.events, v.time_str, v.frame_index
# #                 )
# #                 v.annotated_frame = None
# #                 saved += 1

# #         if need_video:
# #             cap = cv2.VideoCapture(video_path)
# #             if not cap.isOpened():
# #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# #             else:
# #                 seen: set = set()
# #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# #                     if v.frame_index in seen:
# #                         continue
# #                     seen.add(v.frame_index)
# #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# #                     ret, frame = cap.read()
# #                     if not ret:
# #                         continue
# #                     v.frame_path = self._save_frame(
# #                         frame, v.events, v.time_str, v.frame_index
# #                     )
# #                     saved += 1
# #                 cap.release()

# #         print(f"[ViolationStore] {saved} frames saved")

# #     def _save_frame(
# #         self,
# #         frame:       np.ndarray,
# #         events:      List[str],
# #         time_str:    str,
# #         frame_index: int,            # ← ADDED: makes filename globally unique
# #     ) -> str:
# #         """
# #         Save one violation frame as JPEG.

# #         Filename: {event_types}_{hh-mm-ss}_f{frame_index}.jpg
# #         Example:  phone_use_00-13-16_f6762.jpg
# #                   seat_absence_00-00-17_f516.jpg

# #         frame_index prevents two violations of the same type at the
# #         same timestamp from silently overwriting each other.
# #         """
# #         distraction   = "_".join(events)
# #         filename_time = time_str.replace(":", "-")
# #         filename      = f"{distraction}_{filename_time}_f{frame_index}.jpg"  # ← CHANGED
# #         path          = os.path.join(self.frames_dir, filename)
# #         ok = cv2.imwrite(
# #             path,
# #             cv2.resize(frame, (640, 360)),
# #             [cv2.IMWRITE_JPEG_QUALITY, 85],
# #         )
# #         if not ok:
# #             print(f"[ViolationStore] imwrite failed: {path}")
# #         return os.path.join(self.analysis_id, "frames", filename)

# #     # ─────────────────────────────────────────────────────────────

# #     def _build_report(self, processing_time: float = 0.0) -> dict:
# #         return {
# #             "analysis_id":     self.analysis_id,
# #             "train_detail_id": self.train_detail_id,
# #             "processing_time": round(processing_time, 3),
# #             "video_info": (
# #                 self.video_infos[0] if len(self.video_infos) == 1
# #                 else self.video_infos
# #             ),
# #             "violations": [
# #                 {
# #                     "timestamp":   v.time_str,
# #                     "frame_index": v.frame_index,
# #                     "events":      v.events,
# #                     "severity":    v.severity,
# #                     "duration":    v.duration,
# #                     "risk_score":  v.risk_score,
# #                     "risk_level":  v.risk_level,
# #                     "confidence":  v.confidence,
# #                     "factors":     v.factors,
# #                     "original_video_timestamp": f"{v.source_filename} {v.local_time_str}",
# #                     "frame_path":  v.frame_path,
# #                 }
# #                 for v in self._violations
# #             ],
# #         }

# #     def finalize(self, processing_time: float = 0.0) -> str:
# #         self._deduplicate_by_frame()
# #         self._merge_by_time_window()

# #         # Save frames — each video's temp path is in video_infos
# #         for vi in self.video_infos:
# #             if vi and vi.get("videoPath"):
# #                 self.extract_violation_frames(vi["videoPath"])

# #         report   = self._build_report(processing_time=processing_time)
# #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# #         with open(out_path, "w", encoding="utf-8") as f:
# #             json.dump(report, f, indent=2)

# #         print(f"[ViolationStore] JSON report     : {out_path}")
# #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# #         # Upload results to S3 and update DB result_s3_path
# #         try:
# #             from utils.db_s3_uploader import finalize_and_upload
# #             finalize_and_upload(
# #                 report_path     = out_path,
# #                 analysis_id     = self.analysis_id,
# #                 train_detail_id = self.train_detail_id,
# #             )
# #         except Exception as exc:
# #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# #         return out_path

# # # # from __future__ import annotations

# # # # import json
# # # # import os
# # # # from dataclasses import dataclass
# # # # from typing import Any, Dict, List, Optional

# # # # import cv2
# # # # import numpy as np

# # # # OUTPUTS_ROOT = "outputs"
# # # # MERGE_WINDOW = 2.0


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # INTERNAL DATA CLASS
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # @dataclass
# # # # class _Violation:
# # # #     timestamp:       float
# # # #     time_str:        str
# # # #     frame_index:     int
# # # #     type:            str
# # # #     events:          List[str]
# # # #     severity:        str
# # # #     duration:        float
# # # #     risk_score:      int
# # # #     risk_level:      str
# # # #     confidence:      float
# # # #     factors:         List[str]
# # # #     source_filename: str                  = ""    # e.g. "ch01.mp4"
# # # #     local_time_str:  str                  = ""    # local time within that file e.g. "00:00:18"
# # # #     frame_path:      Optional[str]        = None
# # # #     annotated_frame: Optional[np.ndarray] = None


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # VIOLATION STORE
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # class ViolationStore:
# # # #     """
# # # #     Accumulates all violations found across one analysis run (single video
# # # #     or a multi-video batch that shares the same analysis_id / folder_name).

# # # #     Usage
# # # #     ─────
# # # #     1. Construct once per analysis run.
# # # #     2. Call record_violation() from the pipeline for every distraction event.
# # # #     3. Call finalize() at the very end — it deduplicates, merges, saves frames,
# # # #        writes the JSON report, then calls db_s3_uploader.finalize_and_upload()
# # # #        to push everything to S3 and record results in the DB.
# # # #     """

# # # #     def __init__(
# # # #         self,
# # # #         analysis_id:     str,
# # # #         train_detail_id: int,
# # # #         video_info:      Optional[Dict[str, Any]] = None,
# # # #     ):
# # # #         self.analysis_id     = analysis_id
# # # #         self.train_detail_id = train_detail_id
# # # #         # video_infos is always a list:
# # # #         #   • 1 entry for single-video runs
# # # #         #   • N entries for batch runs (add_video_info called per video)
# # # #         self.video_infos: List[Dict[str, Any]] = (
# # # #             [video_info] if video_info is not None else []
# # # #         )

# # # #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# # # #         self.frames_dir = os.path.join(self.output_dir, "frames")
# # # #         os.makedirs(self.frames_dir, exist_ok=True)

# # # #         self._violations:  List[_Violation] = []
# # # #         self._seen_frames: set              = set()
# # # #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# # # #     # ── Public helpers ────────────────────────────────────────────────────────

# # # #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# # # #         """
# # # #         Register one video's metadata into the shared store.
# # # #         Called once per video in batch mode.
# # # #         """
# # # #         self.video_infos.append(video_info)

# # # #     def record_violation(
# # # #         self,
# # # #         annotated_frame:  np.ndarray,
# # # #         video_time:       float,
# # # #         frame_index:      int,
# # # #         event_type:       str,
# # # #         original_frame:   Optional[np.ndarray] = None,
# # # #         severity:         str   = "CRITICAL",
# # # #         confidence:       float = 0.9,
# # # #         risk_score:       int   = 80,
# # # #         risk_level:       str   = "CRITICAL",
# # # #         factors:          Optional[List[str]] = None,
# # # #         duration:         float = 0.0,
# # # #         source_filename:  str   = "",    # original filename e.g. "ch01.mp4"
# # # #         local_video_time: float = -1.0,  # video_time before global offset; -1 = same as video_time
# # # #     ) -> None:
# # # #         """
# # # #         Record one distraction event.

# # # #         Deduplication key is (frame_index, event_type) so that:
# # # #           • the same violation type on the same frame is recorded only once
# # # #           • different violation types on the same frame are each recorded
# # # #           • frame numbers from different videos never collide because
# # # #             main.py applies a frame_offset to make every global frame_index
# # # #             unique across the batch
# # # #         """
# # # #         dedup_key = (frame_index, event_type)
# # # #         if dedup_key in self._seen_frames:
# # # #             return
# # # #         self._seen_frames.add(dedup_key)

# # # #         factors  = factors or []
# # # #         t        = int(round(video_time))
# # # #         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

# # # #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# # # #         local_str = (
# # # #             f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
# # # #         )

# # # #         self._violations.append(
# # # #             _Violation(
# # # #                 timestamp        = video_time,
# # # #                 time_str         = time_str,
# # # #                 frame_index      = frame_index,
# # # #                 type             = event_type,
# # # #                 events           = [event_type],
# # # #                 severity         = severity,
# # # #                 duration         = round(duration, 2),
# # # #                 risk_score       = risk_score,
# # # #                 risk_level       = risk_level,
# # # #                 confidence       = round(confidence, 3),
# # # #                 factors          = list(factors),
# # # #                 source_filename  = source_filename,
# # # #                 local_time_str   = local_str,
# # # #                 annotated_frame  = (
# # # #                     annotated_frame.copy() if annotated_frame is not None else None
# # # #                 ),
# # # #             )
# # # #         )

# # # #     # ── Finalize ──────────────────────────────────────────────────────────────

# # # #     def finalize(self, processing_time: float = 0.0) -> str:
# # # #         """
# # # #         1. Deduplicate violations that share the same frame.
# # # #         2. Merge violations that are within MERGE_WINDOW seconds of each other.
# # # #         3. Save annotated frame images to disk (falls back to re-reading from
# # # #            the source video when no annotated frame was captured).
# # # #         4. Write analysis_report.json.
# # # #         5. Upload frames + report to S3 and record in DB via db_s3_uploader.
# # # #         Returns the local path to analysis_report.json.
# # # #         """
# # # #         self._deduplicate_by_frame()
# # # #         self._merge_by_time_window()

# # # #         # Extract frames from every video in the batch (or the single video)
# # # #         for vi in self.video_infos:
# # # #             if vi and vi.get("videoPath"):
# # # #                 self.extract_violation_frames(vi["videoPath"])

# # # #         report   = self._build_report(processing_time=processing_time)
# # # #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# # # #         with open(out_path, "w", encoding="utf-8") as f:
# # # #             json.dump(report, f, indent=2)

# # # #         print(f"[ViolationStore] JSON report     : {out_path}")
# # # #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# # # #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# # # #         try:
# # # #             from utils.db_s3_uploader import finalize_and_upload
# # # #             finalize_and_upload(
# # # #                 report_path     = out_path,
# # # #                 analysis_id     = self.analysis_id,
# # # #                 train_detail_id = self.train_detail_id,
# # # #             )
# # # #         except Exception as exc:
# # # #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# # # #         return out_path

# # # #     # ── Private — deduplication & merging ────────────────────────────────────

# # # #     def _deduplicate_by_frame(self) -> None:
# # # #         unique: Dict[int, _Violation] = {}
# # # #         for v in self._violations:
# # # #             if v.frame_index not in unique:
# # # #                 unique[v.frame_index] = v
# # # #             else:
# # # #                 ex = unique[v.frame_index]
# # # #                 ex.events  = list(set(ex.events  + v.events))
# # # #                 ex.factors = list(set(ex.factors + v.factors))
# # # #                 if v.risk_score > ex.risk_score:
# # # #                     ex.risk_score = v.risk_score
# # # #                     ex.risk_level = v.risk_level
# # # #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# # # #                     ex.annotated_frame = v.annotated_frame
# # # #         self._violations = list(unique.values())

# # # #     def _merge_by_time_window(self) -> None:
# # # #         if not self._violations:
# # # #             return
# # # #         self._violations.sort(key=lambda x: x.timestamp)
# # # #         merged: List[_Violation] = []
# # # #         group  = [self._violations[0]]
# # # #         for v in self._violations[1:]:
# # # #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# # # #                 group.append(v)
# # # #             else:
# # # #                 merged.append(self._merge_group(group))
# # # #                 group = [v]
# # # #         merged.append(self._merge_group(group))
# # # #         self._violations = merged

# # # #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# # # #         base            = group[0]
# # # #         events: List[str]  = []
# # # #         factors: List[str] = []
# # # #         max_risk   = base.risk_score
# # # #         risk_level = base.risk_level
# # # #         best_frame = base.annotated_frame
# # # #         for v in group:
# # # #             events.extend(v.events)
# # # #             factors.extend(v.factors)
# # # #             if v.risk_score > max_risk:
# # # #                 max_risk, risk_level = v.risk_score, v.risk_level
# # # #             if best_frame is None and v.annotated_frame is not None:
# # # #                 best_frame = v.annotated_frame
# # # #         return _Violation(
# # # #             timestamp        = base.timestamp,
# # # #             time_str         = base.time_str,
# # # #             frame_index      = base.frame_index,
# # # #             type             = base.type,
# # # #             events           = list(set(events)),
# # # #             severity         = base.severity,
# # # #             duration         = base.duration,
# # # #             risk_score       = max_risk,
# # # #             risk_level       = risk_level,
# # # #             confidence       = base.confidence,
# # # #             factors          = list(set(factors)),
# # # #             source_filename  = base.source_filename,
# # # #             local_time_str   = base.local_time_str,
# # # #             annotated_frame  = best_frame,
# # # #         )

# # # #     # ── Private — frame extraction & saving ──────────────────────────────────

# # # #     def extract_violation_frames(self, video_path: str) -> None:
# # # #         print("[ViolationStore] Saving frames...")
# # # #         need_video = [v for v in self._violations if v.annotated_frame is None]
# # # #         saved = 0

# # # #         # First pass: save violations that already have an annotated frame
# # # #         for v in self._violations:
# # # #             if v.annotated_frame is not None:
# # # #                 v.frame_path      = self._save_frame(v.annotated_frame, v.events, v.time_str)
# # # #                 v.annotated_frame = None   # free memory
# # # #                 saved += 1

# # # #         # Second pass: re-read from the source video for any that are missing
# # # #         if need_video:
# # # #             cap = cv2.VideoCapture(video_path)
# # # #             if not cap.isOpened():
# # # #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# # # #             else:
# # # #                 seen: set = set()
# # # #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# # # #                     if v.frame_index in seen:
# # # #                         continue
# # # #                     seen.add(v.frame_index)
# # # #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# # # #                     ret, frame = cap.read()
# # # #                     if not ret:
# # # #                         continue
# # # #                     v.frame_path = self._save_frame(frame, v.events, v.time_str)
# # # #                     saved += 1
# # # #                 cap.release()

# # # #         print(f"[ViolationStore] {saved} frames saved")

# # # #     def _save_frame(
# # # #         self,
# # # #         frame:    np.ndarray,
# # # #         events:   List[str],
# # # #         time_str: str,
# # # #     ) -> str:
# # # #         distraction   = "_".join(events)
# # # #         filename_time = time_str.replace(":", "-")
# # # #         filename      = f"{distraction}_{filename_time}.jpg"
# # # #         path          = os.path.join(self.frames_dir, filename)
# # # #         ok = cv2.imwrite(
# # # #             path,
# # # #             cv2.resize(frame, (640, 360)),
# # # #             [cv2.IMWRITE_JPEG_QUALITY, 85],
# # # #         )
# # # #         if not ok:
# # # #             print(f"[ViolationStore] imwrite failed: {path}")
# # # #         return os.path.join(self.analysis_id, "frames", filename)

# # # #     # ── Private — report builder ──────────────────────────────────────────────

# # # #     def _build_report(self, processing_time: float = 0.0) -> dict:
# # # #         return {
# # # #             "analysis_id":     self.analysis_id,
# # # #             "train_detail_id": self.train_detail_id,
# # # #             "processing_time": round(processing_time, 3),
# # # #             # Single video → keep as dict for backwards compat; batch → list
# # # #             "video_info": (
# # # #                 self.video_infos[0]
# # # #                 if len(self.video_infos) == 1
# # # #                 else self.video_infos
# # # #             ),
# # # #             "violations": [
# # # #                 {
# # # #                     "timestamp":   v.time_str,
# # # #                     "frame_index": v.frame_index,
# # # #                     "events":      v.events,
# # # #                     "severity":    v.severity,
# # # #                     "duration":    v.duration,
# # # #                     "risk_score":  v.risk_score,
# # # #                     "risk_level":  v.risk_level,
# # # #                     "confidence":  v.confidence,
# # # #                     "factors":     v.factors,
# # # #                     "original_video_timestamp": (
# # # #                         f"{v.source_filename} {v.local_time_str}"
# # # #                     ),
# # # #                     "frame_path":  v.frame_path,
# # # #                 }
# # # #                 for v in self._violations
# # # #             ],
# # # #         }

# # # from __future__ import annotations

# # # import json
# # # import os
# # # from dataclasses import dataclass
# # # from typing import Any, Dict, List, Optional

# # # import cv2
# # # import numpy as np

# # # OUTPUTS_ROOT = "outputs"
# # # MERGE_WINDOW = 2.0


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # # INTERNAL DATA CLASS
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # @dataclass
# # # class _Violation:
# # #     timestamp:       float
# # #     time_str:        str
# # #     frame_index:     int
# # #     type:            str
# # #     events:          List[str]
# # #     severity:        str
# # #     duration:        float
# # #     risk_score:      int
# # #     risk_level:      str
# # #     confidence:      float
# # #     factors:         List[str]
# # #     source_filename: str                  = ""    # e.g. "ch01.mp4"
# # #     local_time_str:  str                  = ""    # local time within that file e.g. "00:00:18"
# # #     frame_path:      Optional[str]        = None
# # #     annotated_frame: Optional[np.ndarray] = None


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # # VIOLATION STORE
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # class ViolationStore:
# # #     """
# # #     Accumulates all violations found across one analysis run (single video
# # #     or a multi-video batch that shares the same analysis_id / folder_name).

# # #     Usage
# # #     ─────
# # #     1. Construct once per analysis run.
# # #     2. Call record_violation() from the pipeline for every distraction event.
# # #     3. Call finalize() at the very end — it deduplicates, merges, saves frames,
# # #        writes the JSON report, then calls db_s3_uploader.finalize_and_upload()
# # #        to push everything to S3 and record results in the DB.
# # #     """

# # #     def __init__(
# # #         self,
# # #         analysis_id:     str,
# # #         train_detail_id: int,
# # #         video_info:      Optional[Dict[str, Any]] = None,
# # #     ):
# # #         self.analysis_id     = analysis_id
# # #         self.train_detail_id = train_detail_id
# # #         # video_infos is always a list:
# # #         #   • 1 entry for single-video runs
# # #         #   • N entries for batch runs (add_video_info called per video)
# # #         self.video_infos: List[Dict[str, Any]] = (
# # #             [video_info] if video_info is not None else []
# # #         )

# # #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# # #         self.frames_dir = os.path.join(self.output_dir, "frames")
# # #         os.makedirs(self.frames_dir, exist_ok=True)

# # #         self._violations:  List[_Violation] = []
# # #         self._seen_frames: set              = set()
# # #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# # #     # ── Public helpers ────────────────────────────────────────────────────────

# # #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# # #         """
# # #         Register one video's metadata into the shared store.
# # #         Called once per video in batch mode.
# # #         """
# # #         self.video_infos.append(video_info)

# # #     def record_violation(
# # #         self,
# # #         annotated_frame:  np.ndarray,
# # #         video_time:       float,
# # #         frame_index:      int,
# # #         event_type:       str,
# # #         original_frame:   Optional[np.ndarray] = None,
# # #         severity:         str   = "CRITICAL",
# # #         confidence:       float = 0.9,
# # #         risk_score:       int   = 80,
# # #         risk_level:       str   = "CRITICAL",
# # #         factors:          Optional[List[str]] = None,
# # #         duration:         float = 0.0,
# # #         source_filename:  str   = "",    # original filename e.g. "ch01.mp4"
# # #         local_video_time: float = -1.0,  # video_time before global offset; -1 = same as video_time
# # #     ) -> None:
# # #         """
# # #         Record one distraction event.

# # #         Deduplication key is (frame_index, event_type) so that:
# # #           • the same violation type on the same frame is recorded only once
# # #           • different violation types on the same frame are each recorded
# # #           • frame numbers from different videos never collide because
# # #             main.py applies a frame_offset to make every global frame_index
# # #             unique across the batch
# # #         """
# # #         dedup_key = (frame_index, event_type)
# # #         if dedup_key in self._seen_frames:
# # #             return
# # #         self._seen_frames.add(dedup_key)

# # #         factors  = factors or []
# # #         t        = int(round(video_time))
# # #         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

# # #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# # #         local_str = (
# # #             f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
# # #         )

# # #         self._violations.append(
# # #             _Violation(
# # #                 timestamp        = video_time,
# # #                 time_str         = time_str,
# # #                 frame_index      = frame_index,
# # #                 type             = event_type,
# # #                 events           = [event_type],
# # #                 severity         = severity,
# # #                 duration         = round(duration, 2),
# # #                 risk_score       = risk_score,
# # #                 risk_level       = risk_level,
# # #                 confidence       = round(confidence, 3),
# # #                 factors          = list(factors),
# # #                 source_filename  = source_filename,
# # #                 local_time_str   = local_str,
# # #                 annotated_frame  = (
# # #                     annotated_frame.copy() if annotated_frame is not None else None
# # #                 ),
# # #             )
# # #         )

# # #     # ── Finalize ──────────────────────────────────────────────────────────────

# # #     def finalize(self, processing_time: float = 0.0) -> str:
# # #         """
# # #         1. Deduplicate violations that share the same frame.
# # #         2. Merge violations that are within MERGE_WINDOW seconds of each other.
# # #         3. Save annotated frame images to disk (falls back to re-reading from
# # #            the source video when no annotated frame was captured).
# # #         4. Write analysis_report.json.
# # #         5. Upload frames + report to S3 and record in DB via db_s3_uploader.
# # #         Returns the local path to analysis_report.json.
# # #         """
# # #         self._deduplicate_by_frame()
# # #         self._merge_by_time_window()

# # #         # Extract frames from every video in the batch (or the single video)
# # #         for vi in self.video_infos:
# # #             if vi and vi.get("videoPath"):
# # #                 self.extract_violation_frames(vi["videoPath"])

# # #         report   = self._build_report(processing_time=processing_time)
# # #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# # #         with open(out_path, "w", encoding="utf-8") as f:
# # #             json.dump(report, f, indent=2)

# # #         print(f"[ViolationStore] JSON report     : {out_path}")
# # #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# # #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# # #         try:
# # #             from utils.db_s3_uploader import finalize_and_upload
# # #             finalize_and_upload(
# # #                 report_path     = out_path,
# # #                 analysis_id     = self.analysis_id,
# # #                 train_detail_id = self.train_detail_id,
# # #             )
# # #         except Exception as exc:
# # #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# # #         return out_path

# # #     # ── Private — deduplication & merging ────────────────────────────────────

# # #     def _deduplicate_by_frame(self) -> None:
# # #         unique: Dict[int, _Violation] = {}
# # #         for v in self._violations:
# # #             if v.frame_index not in unique:
# # #                 unique[v.frame_index] = v
# # #             else:
# # #                 ex = unique[v.frame_index]
# # #                 ex.events  = list(set(ex.events  + v.events))
# # #                 ex.factors = list(set(ex.factors + v.factors))
# # #                 if v.risk_score > ex.risk_score:
# # #                     ex.risk_score = v.risk_score
# # #                     ex.risk_level = v.risk_level
# # #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# # #                     ex.annotated_frame = v.annotated_frame
# # #         self._violations = list(unique.values())

# # #     def _merge_by_time_window(self) -> None:
# # #         if not self._violations:
# # #             return
# # #         self._violations.sort(key=lambda x: x.timestamp)
# # #         merged: List[_Violation] = []
# # #         group  = [self._violations[0]]
# # #         for v in self._violations[1:]:
# # #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# # #                 group.append(v)
# # #             else:
# # #                 merged.append(self._merge_group(group))
# # #                 group = [v]
# # #         merged.append(self._merge_group(group))
# # #         self._violations = merged

# # #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# # #         base             = group[0]
# # #         events: List[str]  = []
# # #         factors: List[str] = []
# # #         max_risk   = base.risk_score
# # #         risk_level = base.risk_level
# # #         best_frame = base.annotated_frame
# # #         for v in group:
# # #             events.extend(v.events)
# # #             factors.extend(v.factors)
# # #             if v.risk_score > max_risk:
# # #                 max_risk, risk_level = v.risk_score, v.risk_level
# # #             if best_frame is None and v.annotated_frame is not None:
# # #                 best_frame = v.annotated_frame
# # #         return _Violation(
# # #             timestamp        = base.timestamp,
# # #             time_str         = base.time_str,
# # #             frame_index      = base.frame_index,
# # #             type             = base.type,
# # #             events           = list(set(events)),
# # #             severity         = base.severity,
# # #             duration         = base.duration,
# # #             risk_score       = max_risk,
# # #             risk_level       = risk_level,
# # #             confidence       = base.confidence,
# # #             factors          = list(set(factors)),
# # #             source_filename  = base.source_filename,
# # #             local_time_str   = base.local_time_str,
# # #             annotated_frame  = best_frame,
# # #         )

# # #     # ── Private — frame extraction & saving ──────────────────────────────────

# # #     def extract_violation_frames(self, video_path: str) -> None:
# # #         print("[ViolationStore] Saving frames...")
# # #         need_video = [v for v in self._violations if v.annotated_frame is None]
# # #         saved = 0

# # #         # First pass: save violations that already have an annotated frame
# # #         for v in self._violations:
# # #             if v.annotated_frame is not None:
# # #                 v.frame_path      = self._save_frame(
# # #                     v.annotated_frame, v.events, v.time_str, v.frame_index
# # #                 )
# # #                 v.annotated_frame = None   # free memory
# # #                 saved += 1

# # #         # Second pass: re-read from the source video for any that are missing
# # #         if need_video:
# # #             cap = cv2.VideoCapture(video_path)
# # #             if not cap.isOpened():
# # #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# # #             else:
# # #                 seen: set = set()
# # #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# # #                     if v.frame_index in seen:
# # #                         continue
# # #                     seen.add(v.frame_index)
# # #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# # #                     ret, frame = cap.read()
# # #                     if not ret:
# # #                         continue
# # #                     v.frame_path = self._save_frame(
# # #                         frame, v.events, v.time_str, v.frame_index
# # #                     )
# # #                     saved += 1
# # #                 cap.release()

# # #         print(f"[ViolationStore] {saved} frames saved")

# # #     def _save_frame(
# # #         self,
# # #         frame:       np.ndarray,
# # #         events:      List[str],
# # #         time_str:    str,
# # #         frame_index: int,
# # #     ) -> str:
# # #         """
# # #         Save a single violation frame as a JPEG.

# # #         Filename format:  <events>_<HH-MM-SS>_f<frame_index>.jpg
# # #         Example:          seat_absence_00-01-14_f2106.jpg
# # #                           seat_absence_drowsy_00-03-02_f5058.jpg
# # #         """
# # #         distraction   = "_".join(sorted(events))   # sorted for deterministic name
# # #         filename_time = time_str.replace(":", "-")
# # #         filename      = f"{distraction}_{filename_time}_f{frame_index}.jpg"
# # #         path          = os.path.join(self.frames_dir, filename)
# # #         ok = cv2.imwrite(
# # #             path,
# # #             cv2.resize(frame, (640, 360)),
# # #             [cv2.IMWRITE_JPEG_QUALITY, 85],
# # #         )
# # #         if not ok:
# # #             print(f"[ViolationStore] imwrite failed: {path}")
# # #         return os.path.join(self.analysis_id, "frames", filename)

# # #     # ── Private — report builder ──────────────────────────────────────────────

# # #     def _build_report(self, processing_time: float = 0.0) -> dict:
# # #         return {
# # #             "analysis_id":     self.analysis_id,
# # #             "train_detail_id": self.train_detail_id,
# # #             "processing_time": round(processing_time, 3),
# # #             # Single video → keep as dict for backwards compat; batch → list
# # #             "video_info": (
# # #                 self.video_infos[0]
# # #                 if len(self.video_infos) == 1
# # #                 else self.video_infos
# # #             ),
# # #             "violations": [
# # #                 {
# # #                     "timestamp":   v.time_str,
# # #                     "frame_index": v.frame_index,
# # #                     "events":      v.events,
# # #                     "severity":    v.severity,
# # #                     "duration":    v.duration,
# # #                     "risk_score":  v.risk_score,
# # #                     "risk_level":  v.risk_level,
# # #                     "confidence":  v.confidence,
# # #                     "factors":     v.factors,
# # #                     "original_video_timestamp": (
# # #                         f"{v.source_filename} {v.local_time_str}"
# # #                     ),
# # #                     "frame_path":  v.frame_path,
# # #                 }
# # #                 for v in self._violations
# # #             ],
# # #         }


# # from __future__ import annotations

# # import json
# # import os
# # from dataclasses import dataclass
# # from typing import Any, Dict, List, Optional

# # import cv2
# # import numpy as np

# # OUTPUTS_ROOT = "outputs"
# # MERGE_WINDOW = 2.0


# # # ══════════════════════════════════════════════════════════════════════════════
# # # INTERNAL DATA CLASS
# # # ══════════════════════════════════════════════════════════════════════════════

# # @dataclass
# # class _Violation:
# #     timestamp:       float
# #     time_str:        str
# #     frame_index:     int
# #     type:            str
# #     events:          List[str]
# #     severity:        str
# #     duration:        float
# #     risk_score:      int
# #     risk_level:      str
# #     confidence:      float
# #     factors:         List[str]
# #     source_filename: str                  = ""    # DB filename e.g. "mobile.mp4"
# #     local_time_str:  str                  = ""    # local time within that file e.g. "00:00:23"
# #     frame_path:      Optional[str]        = None
# #     annotated_frame: Optional[np.ndarray] = None


# # # ══════════════════════════════════════════════════════════════════════════════
# # # VIOLATION STORE
# # # ══════════════════════════════════════════════════════════════════════════════

# # class ViolationStore:
# #     """
# #     Accumulates all violations found across one analysis run (single video
# #     or a multi-video batch that shares the same analysis_id / folder_name).

# #     Batch mode usage (api.py)
# #     ─────────────────────────
# #     1. Construct ONCE for the whole folder (no video_info in __init__).
# #     2. Pass as shared_vstore= to each GadgetDetectionPipeline.
# #        The pipeline calls add_video_info() automatically.
# #     3. Call finalize() ONCE after all videos in the folder are done.

# #     Standalone mode usage (CLI / single video)
# #     ──────────────────────────────────────────
# #     1. Construct with video_info= for the single video.
# #     2. Pipeline calls finalize() automatically at the end of run().
# #     """

# #     def __init__(
# #         self,
# #         analysis_id:     str,
# #         train_detail_id: int,
# #         video_info:      Optional[Dict[str, Any]] = None,
# #     ):
# #         self.analysis_id     = analysis_id
# #         self.train_detail_id = train_detail_id
# #         # Always a list — 0 entries until add_video_info() is called (batch mode),
# #         # or 1 entry when video_info is provided (standalone mode).
# #         self.video_infos: List[Dict[str, Any]] = (
# #             [video_info] if video_info is not None else []
# #         )

# #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# #         self.frames_dir = os.path.join(self.output_dir, "frames")
# #         os.makedirs(self.frames_dir, exist_ok=True)

# #         self._violations:  List[_Violation] = []
# #         self._seen_frames: set              = set()
# #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# #     # ── Public helpers ────────────────────────────────────────────────────────

# #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# #         """Append one video's metadata. Called once per video in batch mode."""
# #         self.video_infos.append(video_info)

# #     def record_violation(
# #         self,
# #         annotated_frame:  np.ndarray,
# #         video_time:       float,          # global timestamp (offset-adjusted)
# #         frame_index:      int,            # global frame index (offset-adjusted)
# #         event_type:       str,
# #         original_frame:   Optional[np.ndarray] = None,
# #         severity:         str   = "CRITICAL",
# #         confidence:       float = 0.9,
# #         risk_score:       int   = 80,
# #         risk_level:       str   = "CRITICAL",
# #         factors:          Optional[List[str]] = None,
# #         duration:         float = 0.0,
# #         source_filename:  str   = "",     # DB filename shown in original_video_timestamp
# #         local_video_time: float = -1.0,   # local time within the source file; -1 = same as video_time
# #     ) -> None:
# #         """
# #         Record one distraction event.

# #         Deduplication key is (frame_index, event_type):
# #           • same event on the same global frame is recorded once
# #           • different events on the same global frame are each recorded
# #           • global frame_index is unique across videos (frame_offset applied in main.py)
# #         """
# #         dedup_key = (frame_index, event_type)
# #         if dedup_key in self._seen_frames:
# #             return
# #         self._seen_frames.add(dedup_key)

# #         factors  = factors or []
# #         t        = int(round(video_time))
# #         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

# #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# #         local_str = (
# #             f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
# #         )

# #         self._violations.append(
# #             _Violation(
# #                 timestamp        = video_time,
# #                 time_str         = time_str,
# #                 frame_index      = frame_index,
# #                 type             = event_type,
# #                 events           = [event_type],
# #                 severity         = severity,
# #                 duration         = round(duration, 2),
# #                 risk_score       = risk_score,
# #                 risk_level       = risk_level,
# #                 confidence       = round(confidence, 3),
# #                 factors          = list(factors),
# #                 source_filename  = source_filename,
# #                 local_time_str   = local_str,
# #                 annotated_frame  = (
# #                     annotated_frame.copy() if annotated_frame is not None else None
# #                 ),
# #             )
# #         )

# #     # ── Finalize ──────────────────────────────────────────────────────────────

# #     def finalize(self, processing_time: float = 0.0) -> str:
# #         """
# #         Deduplicate → merge → save frames → write JSON → upload to S3/DB.
# #         Returns the local path to analysis_report.json.
# #         """
# #         self._deduplicate_by_frame()
# #         self._merge_by_time_window()

# #         # Extract frames from every video in the batch (or the single video)
# #         for vi in self.video_infos:
# #             if vi and vi.get("videoPath"):
# #                 self.extract_violation_frames(vi["videoPath"])

# #         report   = self._build_report(processing_time=processing_time)
# #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# #         with open(out_path, "w", encoding="utf-8") as f:
# #             json.dump(report, f, indent=2)

# #         print(f"[ViolationStore] JSON report     : {out_path}")
# #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# #         try:
# #             from utils.db_s3_uploader import finalize_and_upload
# #             finalize_and_upload(
# #                 report_path     = out_path,
# #                 analysis_id     = self.analysis_id,
# #                 train_detail_id = self.train_detail_id,
# #             )
# #         except Exception as exc:
# #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# #         return out_path

# #     # ── Private — deduplication & merging ────────────────────────────────────

# #     def _deduplicate_by_frame(self) -> None:
# #         unique: Dict[int, _Violation] = {}
# #         for v in self._violations:
# #             if v.frame_index not in unique:
# #                 unique[v.frame_index] = v
# #             else:
# #                 ex = unique[v.frame_index]
# #                 ex.events  = list(set(ex.events  + v.events))
# #                 ex.factors = list(set(ex.factors + v.factors))
# #                 if v.risk_score > ex.risk_score:
# #                     ex.risk_score = v.risk_score
# #                     ex.risk_level = v.risk_level
# #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# #                     ex.annotated_frame = v.annotated_frame
# #         self._violations = list(unique.values())

# #     def _merge_by_time_window(self) -> None:
# #         if not self._violations:
# #             return
# #         self._violations.sort(key=lambda x: x.timestamp)
# #         merged: List[_Violation] = []
# #         group  = [self._violations[0]]
# #         for v in self._violations[1:]:
# #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# #                 group.append(v)
# #             else:
# #                 merged.append(self._merge_group(group))
# #                 group = [v]
# #         merged.append(self._merge_group(group))
# #         self._violations = merged

# #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# #         base             = group[0]
# #         events: List[str]  = []
# #         factors: List[str] = []
# #         max_risk   = base.risk_score
# #         risk_level = base.risk_level
# #         best_frame = base.annotated_frame
# #         for v in group:
# #             events.extend(v.events)
# #             factors.extend(v.factors)
# #             if v.risk_score > max_risk:
# #                 max_risk, risk_level = v.risk_score, v.risk_level
# #             if best_frame is None and v.annotated_frame is not None:
# #                 best_frame = v.annotated_frame
# #         return _Violation(
# #             timestamp        = base.timestamp,
# #             time_str         = base.time_str,
# #             frame_index      = base.frame_index,
# #             type             = base.type,
# #             events           = list(set(events)),
# #             severity         = base.severity,
# #             duration         = base.duration,
# #             risk_score       = max_risk,
# #             risk_level       = risk_level,
# #             confidence       = base.confidence,
# #             factors          = list(set(factors)),
# #             source_filename  = base.source_filename,
# #             local_time_str   = base.local_time_str,
# #             annotated_frame  = best_frame,
# #         )

# #     # ── Private — frame extraction & saving ──────────────────────────────────

# #     def extract_violation_frames(self, video_path: str) -> None:
# #         print("[ViolationStore] Saving frames...")
# #         need_video = [v for v in self._violations if v.annotated_frame is None]
# #         saved = 0

# #         # First pass: save violations that already have an annotated frame
# #         for v in self._violations:
# #             if v.annotated_frame is not None:
# #                 v.frame_path      = self._save_frame(v.annotated_frame, v.events, v.time_str)
# #                 v.annotated_frame = None   # free memory
# #                 saved += 1

# #         # Second pass: re-read from the source video for any that are missing
# #         if need_video:
# #             cap = cv2.VideoCapture(video_path)
# #             if not cap.isOpened():
# #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# #             else:
# #                 seen: set = set()
# #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# #                     if v.frame_index in seen:
# #                         continue
# #                     seen.add(v.frame_index)
# #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# #                     ret, frame = cap.read()
# #                     if not ret:
# #                         continue
# #                     v.frame_path = self._save_frame(frame, v.events, v.time_str)
# #                     saved += 1
# #                 cap.release()

# #         print(f"[ViolationStore] {saved} frames saved")

# #     def _save_frame(
# #         self,
# #         frame:    np.ndarray,
# #         events:   List[str],
# #         time_str: str,
# #     ) -> str:
# #         """
# #         Save a single violation frame as JPEG.

# #         Filename format:  <events>_<HH-MM-SS>.jpg
# #         Example:          seat_absence_00-01-14.jpg
# #                           seat_absence_drowsy_00-03-02.jpg
# #                           phone_use_00-00-24.jpg
# #         """
# #         distraction   = "_".join(sorted(events))   # sorted for deterministic name
# #         filename_time = time_str.replace(":", "-")
# #         filename      = f"{distraction}_{filename_time}.jpg"
# #         path          = os.path.join(self.frames_dir, filename)
# #         ok = cv2.imwrite(
# #             path,
# #             cv2.resize(frame, (640, 360)),
# #             [cv2.IMWRITE_JPEG_QUALITY, 85],
# #         )
# #         if not ok:
# #             print(f"[ViolationStore] imwrite failed: {path}")
# #         return os.path.join(self.analysis_id, "frames", filename)

# #     # ── Private — report builder ──────────────────────────────────────────────

# #     def _build_report(self, processing_time: float = 0.0) -> dict:
# #         return {
# #             "analysis_id":     self.analysis_id,
# #             "train_detail_id": self.train_detail_id,
# #             "processing_time": round(processing_time, 3),
# #             # Single video → dict (backwards compat); batch → list
# #             "video_info": (
# #                 self.video_infos[0]
# #                 if len(self.video_infos) == 1
# #                 else self.video_infos
# #             ),
# #             "violations": [
# #                 {
# #                     "timestamp":   v.time_str,
# #                     "frame_index": v.frame_index,
# #                     "events":      v.events,
# #                     "severity":    v.severity,
# #                     "duration":    v.duration,
# #                     "risk_score":  v.risk_score,
# #                     "risk_level":  v.risk_level,
# #                     "confidence":  v.confidence,
# #                     "factors":     v.factors,
# #                     # "original_video_timestamp" = "<db_filename> <local_HH:MM:SS>"
# #                     # When timestamp == original (video 1), local_time_str == time_str
# #                     "original_video_timestamp": (
# #                         f"{v.source_filename} {v.local_time_str}"
# #                     ),
# #                     "frame_path":  v.frame_path,
# #                 }
# #                 for v in self._violations
# #             ],
# #         }


# from __future__ import annotations

# import json
# import os
# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional

# import cv2
# import numpy as np

# OUTPUTS_ROOT = "outputs"
# MERGE_WINDOW = 2.0


# # ══════════════════════════════════════════════════════════════════════════════
# # INTERNAL DATA CLASS
# # ══════════════════════════════════════════════════════════════════════════════

# @dataclass
# class _Violation:
#     timestamp:       float
#     time_str:        str
#     frame_index:     int
#     type:            str
#     events:          List[str]
#     severity:        str
#     duration:        float
#     risk_score:      int
#     risk_level:      str
#     confidence:      float
#     factors:         List[str]
#     source_filename: str                  = ""    # DB filename e.g. "mobile.mp4"
#     local_time_str:  str                  = ""    # local time within that file e.g. "00:00:23"
#     frame_path:      Optional[str]        = None
#     annotated_frame: Optional[np.ndarray] = None


# # ══════════════════════════════════════════════════════════════════════════════
# # VIOLATION STORE
# # ══════════════════════════════════════════════════════════════════════════════

# class ViolationStore:
#     """
#     Accumulates all violations found across one analysis run (single video
#     or a multi-video batch that shares the same analysis_id / folder_name).

#     Batch mode usage (api.py)
#     ─────────────────────────
#     1. Construct ONCE for the whole folder (no video_info in __init__).
#     2. Pass as shared_vstore= to each GadgetDetectionPipeline.
#        The pipeline calls add_video_info() automatically.
#     3. Call finalize() ONCE after all videos in the folder are done.

#     Standalone mode usage (CLI / single video)
#     ──────────────────────────────────────────
#     1. Construct with video_info= for the single video.
#     2. Pipeline calls finalize() automatically at the end of run().
#     """

#     def __init__(
#         self,
#         analysis_id:     str,
#         train_detail_id: int,
#         video_info:      Optional[Dict[str, Any]] = None,
#     ):
#         self.analysis_id     = analysis_id
#         self.train_detail_id = train_detail_id
#         # Always a list — 0 entries until add_video_info() is called (batch mode),
#         # or 1 entry when video_info is provided (standalone mode).
#         self.video_infos: List[Dict[str, Any]] = (
#             [video_info] if video_info is not None else []
#         )

#         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
#         self.frames_dir = os.path.join(self.output_dir, "frames")
#         os.makedirs(self.frames_dir, exist_ok=True)

#         self._violations:  List[_Violation] = []
#         self._seen_frames: set              = set()
#         print(f"[ViolationStore] Output dir : {self.output_dir}")

#     # ── Public helpers ────────────────────────────────────────────────────────

#     def add_video_info(self, video_info: Dict[str, Any]) -> None:
#         """Append one video's metadata. Called once per video in batch mode."""
#         self.video_infos.append(video_info)

#     def record_violation(
#         self,
#         annotated_frame:  np.ndarray,
#         video_time:       float,          # global timestamp (offset-adjusted)
#         frame_index:      int,            # global frame index (offset-adjusted)
#         event_type:       str,
#         original_frame:   Optional[np.ndarray] = None,
#         severity:         str   = "CRITICAL",
#         confidence:       float = 0.9,
#         risk_score:       int   = 80,
#         risk_level:       str   = "CRITICAL",
#         factors:          Optional[List[str]] = None,
#         duration:         float = 0.0,
#         source_filename:  str   = "",     # DB filename shown in original_video_timestamp
#         local_video_time: float = -1.0,   # local time within the source file; -1 = same as video_time
#     ) -> None:
#         """
#         Record one distraction event.

#         Deduplication key is (frame_index, event_type):
#           • same event on the same global frame is recorded once
#           • different events on the same global frame are each recorded
#           • global frame_index is unique across videos (frame_offset applied in main.py)
#         """
#         dedup_key = (frame_index, event_type)
#         if dedup_key in self._seen_frames:
#             return
#         self._seen_frames.add(dedup_key)

#         factors  = factors or []
#         t        = int(round(video_time))
#         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

#         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
#         local_str = (
#             f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
#         )

#         self._violations.append(
#             _Violation(
#                 timestamp        = video_time,
#                 time_str         = time_str,
#                 frame_index      = frame_index,
#                 type             = event_type,
#                 events           = [event_type],
#                 severity         = severity,
#                 duration         = round(duration, 2),
#                 risk_score       = risk_score,
#                 risk_level       = risk_level,
#                 confidence       = round(confidence, 3),
#                 factors          = list(factors),
#                 source_filename  = source_filename,
#                 local_time_str   = local_str,
#                 annotated_frame  = (
#                     annotated_frame.copy() if annotated_frame is not None else None
#                 ),
#             )
#         )

#     # ── Finalize ──────────────────────────────────────────────────────────────

#     def write_report(self, processing_time: float = 0.0) -> str:
#         """
#         Build analysis_report.json from the CURRENT in-memory violations and
#         write it to disk at  outputs/<analysis_id>/analysis_report.json
#         (i.e. as a SIBLING of the frames/ folder, not inside it).

#         Unlike finalize(), this does NOT touch S3 or the legacy DB uploader —
#         it is safe to call from the journey/batch pipeline (analyzer.py),
#         which has its own separate callback-based completion flow.

#         Does NOT run dedup/merge/extract_violation_frames — call those first
#         (analyze_journey() already does, via the shared ViolationStore) if
#         you need them. Safe to call multiple times; it always overwrites.

#         Returns the local path to analysis_report.json.
#         """
#         report   = self._build_report(processing_time=processing_time)
#         out_path = os.path.join(self.output_dir, "analysis_report.json")
#         os.makedirs(self.output_dir, exist_ok=True)
#         with open(out_path, "w", encoding="utf-8") as f:
#             json.dump(report, f, indent=2)

#         print(f"[ViolationStore] JSON report     : {out_path}")
#         print(f"[ViolationStore] Violations      : {len(self._violations)}")
#         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

#         return out_path

#     def finalize(self, processing_time: float = 0.0) -> str:
#         """
#         Deduplicate → merge → save frames → write JSON → upload to S3/DB.
#         Returns the local path to analysis_report.json.

#         Standalone/CLI mode only. Journey/batch mode (analyzer.py) should call
#         write_report() directly instead, after its own dedup/merge/extract
#         steps, since it has its own callback-based upload path and does not
#         want the legacy db_s3_uploader to also run.
#         """
#         self._deduplicate_by_frame()
#         self._merge_by_time_window()

#         # Extract frames from every video in the batch (or the single video)
#         for vi in self.video_infos:
#             if vi and vi.get("videoPath"):
#                 self.extract_violation_frames(vi["videoPath"])

#         out_path = self.write_report(processing_time=processing_time)

#         try:
#             from utils.db_s3_uploader import finalize_and_upload
#             finalize_and_upload(
#                 report_path     = out_path,
#                 analysis_id     = self.analysis_id,
#                 train_detail_id = self.train_detail_id,
#             )
#         except Exception as exc:
#             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

#         return out_path

#     # ── Private — deduplication & merging ────────────────────────────────────

#     def _deduplicate_by_frame(self) -> None:
#         unique: Dict[int, _Violation] = {}
#         for v in self._violations:
#             if v.frame_index not in unique:
#                 unique[v.frame_index] = v
#             else:
#                 ex = unique[v.frame_index]
#                 ex.events  = list(set(ex.events  + v.events))
#                 ex.factors = list(set(ex.factors + v.factors))
#                 if v.risk_score > ex.risk_score:
#                     ex.risk_score = v.risk_score
#                     ex.risk_level = v.risk_level
#                 if ex.annotated_frame is None and v.annotated_frame is not None:
#                     ex.annotated_frame = v.annotated_frame
#         self._violations = list(unique.values())

#     def _merge_by_time_window(self) -> None:
#         if not self._violations:
#             return
#         self._violations.sort(key=lambda x: x.timestamp)
#         merged: List[_Violation] = []
#         group  = [self._violations[0]]
#         for v in self._violations[1:]:
#             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
#                 group.append(v)
#             else:
#                 merged.append(self._merge_group(group))
#                 group = [v]
#         merged.append(self._merge_group(group))
#         self._violations = merged

#     def _merge_group(self, group: List[_Violation]) -> _Violation:
#         base             = group[0]
#         events: List[str]  = []
#         factors: List[str] = []
#         max_risk   = base.risk_score
#         risk_level = base.risk_level
#         best_frame = base.annotated_frame
#         for v in group:
#             events.extend(v.events)
#             factors.extend(v.factors)
#             if v.risk_score > max_risk:
#                 max_risk, risk_level = v.risk_score, v.risk_level
#             if best_frame is None and v.annotated_frame is not None:
#                 best_frame = v.annotated_frame
#         return _Violation(
#             timestamp        = base.timestamp,
#             time_str         = base.time_str,
#             frame_index      = base.frame_index,
#             type             = base.type,
#             events           = list(set(events)),
#             severity         = base.severity,
#             duration         = base.duration,
#             risk_score       = max_risk,
#             risk_level       = risk_level,
#             confidence       = base.confidence,
#             factors          = list(set(factors)),
#             source_filename  = base.source_filename,
#             local_time_str   = base.local_time_str,
#             annotated_frame  = best_frame,
#         )

#     # ── Private — frame extraction & saving ──────────────────────────────────

#     def extract_violation_frames(self, video_path: str) -> None:
#         print("[ViolationStore] Saving frames...")
#         need_video = [v for v in self._violations if v.annotated_frame is None]
#         saved = 0

#         # First pass: save violations that already have an annotated frame
#         for v in self._violations:
#             if v.annotated_frame is not None:
#                 v.frame_path      = self._save_frame(v.annotated_frame, v.events, v.time_str)
#                 v.annotated_frame = None   # free memory
#                 saved += 1

#         # Second pass: re-read from the source video for any that are missing
#         if need_video:
#             cap = cv2.VideoCapture(video_path)
#             if not cap.isOpened():
#                 print(f"[ViolationStore] Cannot open video: {video_path}")
#             else:
#                 seen: set = set()
#                 for v in sorted(need_video, key=lambda x: x.frame_index):
#                     if v.frame_index in seen:
#                         continue
#                     seen.add(v.frame_index)
#                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
#                     ret, frame = cap.read()
#                     if not ret:
#                         continue
#                     v.frame_path = self._save_frame(frame, v.events, v.time_str)
#                     saved += 1
#                 cap.release()

#         print(f"[ViolationStore] {saved} frames saved")

#     def _save_frame(
#         self,
#         frame:    np.ndarray,
#         events:   List[str],
#         time_str: str,
#     ) -> str:
#         """
#         Save a single violation frame as JPEG.

#         Filename format:  <events>_<HH-MM-SS>.jpg
#         Example:          seat_absence_00-01-14.jpg
#                           seat_absence_drowsy_00-03-02.jpg
#                           phone_use_00-00-24.jpg
#         """
#         distraction   = "_".join(sorted(events))   # sorted for deterministic name
#         filename_time = time_str.replace(":", "-")
#         filename      = f"{distraction}_{filename_time}.jpg"
#         path          = os.path.join(self.frames_dir, filename)
#         ok = cv2.imwrite(
#             path,
#             cv2.resize(frame, (640, 360)),
#             [cv2.IMWRITE_JPEG_QUALITY, 85],
#         )
#         if not ok:
#             print(f"[ViolationStore] imwrite failed: {path}")
#         return os.path.join(self.analysis_id, "frames", filename)

#     # ── Private — report builder ──────────────────────────────────────────────

#     def _build_report(self, processing_time: float = 0.0) -> dict:
#         return {
#             "analysis_id":     self.analysis_id,
#             "train_detail_id": self.train_detail_id,
#             "processing_time": round(processing_time, 3),
#             # Single video → dict (backwards compat); batch → list
#             "video_info": (
#                 self.video_infos[0]
#                 if len(self.video_infos) == 1
#                 else self.video_infos
#             ),
#             "violations": [
#                 {
#                     "timestamp":   v.time_str,
#                     "frame_index": v.frame_index,
#                     "events":      v.events,
#                     "severity":    v.severity,
#                     "duration":    v.duration,
#                     "risk_score":  v.risk_score,
#                     "risk_level":  v.risk_level,
#                     "confidence":  v.confidence,
#                     "factors":     v.factors,
#                     # "original_video_timestamp" = "<db_filename> <local_HH:MM:SS>"
#                     # When timestamp == original (video 1), local_time_str == time_str
#                     "original_video_timestamp": (
#                         f"{v.source_filename} {v.local_time_str}"
#                     ),
#                     "frame_path":  v.frame_path,
#                 }
#                 for v in self._violations
#             ],
#         }




# # # from __future__ import annotations

# # # import json
# # # import os
# # # from dataclasses import dataclass
# # # from typing import Any, Dict, List, Optional

# # # import cv2
# # # import numpy as np

# # # OUTPUTS_ROOT = "outputs"
# # # MERGE_WINDOW = 2.0


# # # @dataclass
# # # class _Violation:
# # #     timestamp:       float
# # #     time_str:        str
# # #     frame_index:     int
# # #     type:            str
# # #     events:          List[str]
# # #     severity:        str
# # #     duration:        float
# # #     risk_score:      int
# # #     risk_level:      str
# # #     confidence:      float
# # #     factors:         List[str]
# # #     source_filename: str                   = ""    # e.g. "ax.mp4"
# # #     local_time_str:  str                   = ""    # local time within that file e.g. "00:00:18"
# # #     frame_path:      Optional[str]         = None
# # #     annotated_frame: Optional[np.ndarray]  = None


# # # class ViolationStore:

# # #     def __init__(self, analysis_id: str, train_detail_id: int, video_info: Optional[Dict[str, Any]] = None):
# # #         self.analysis_id     = analysis_id
# # #         self.train_detail_id = train_detail_id
# # #         # video_infos is always a list — 1 entry for single-video, N entries for batch
# # #         self.video_infos: List[Dict[str, Any]] = [video_info] if video_info is not None else []

# # #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# # #         self.frames_dir = os.path.join(self.output_dir, "frames")
# # #         os.makedirs(self.frames_dir, exist_ok=True)

# # #         self._violations: List[_Violation] = []
# # #         self._seen_frames: set             = set()
# # #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# # #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# # #         """
# # #         Register one video's metadata into the shared store.
# # #         Called once per video in batch mode (api.py passes shared_vstore).
# # #         """
# # #         self.video_infos.append(video_info)

# # #     def record_violation(
# # #         self,
# # #         annotated_frame: np.ndarray,
# # #         video_time:      float,
# # #         frame_index:     int,
# # #         event_type:      str,
# # #         original_frame:  Optional[np.ndarray] = None,
# # #         severity:        str   = "CRITICAL",
# # #         confidence:      float = 0.9,
# # #         risk_score:      int   = 80,
# # #         risk_level:      str   = "CRITICAL",
# # #         factors:         Optional[List[str]] = None,
# # #         duration:        float = 0.0,
# # #         source_filename: str   = "",   # original upload filename e.g. "ax.mp4"
# # #         local_video_time: float = -1.0, # raw video_time before offset; -1 = same as video_time
# # #     ):
# # #         # Deduplicate on (frame_index, event_type) so that:
# # #         #  • the same violation type on the same frame is recorded only once
# # #         #  • different violation types on the same frame are each recorded
# # #         #  • frame numbers from different videos never collide (frame_offset
# # #         #    in main.py makes every global frame_index unique across the batch)
# # #         dedup_key = (frame_index, event_type)
# # #         if dedup_key in self._seen_frames:
# # #             return
# # #         self._seen_frames.add(dedup_key)
# # #         factors   = factors or []
# # #         t         = int(round(video_time))
# # #         time_str  = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"
# # #         # Build the per-file local timestamp (time within the source video)
# # #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# # #         local_str = f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
# # #         self._violations.append(_Violation(
# # #             timestamp        = video_time,
# # #             time_str         = time_str,
# # #             frame_index      = frame_index,
# # #             type             = event_type,
# # #             events           = [event_type],
# # #             severity         = severity,
# # #             duration         = round(duration, 2),
# # #             risk_score       = risk_score,
# # #             risk_level       = risk_level,
# # #             confidence       = round(confidence, 3),
# # #             factors          = list(factors),
# # #             source_filename  = source_filename,
# # #             local_time_str   = local_str,
# # #             annotated_frame  = annotated_frame.copy() if annotated_frame is not None else None,
# # #         ))

# # #     def _deduplicate_by_frame(self):
# # #         unique: Dict[int, _Violation] = {}
# # #         for v in self._violations:
# # #             if v.frame_index not in unique:
# # #                 unique[v.frame_index] = v
# # #             else:
# # #                 ex = unique[v.frame_index]
# # #                 ex.events  = list(set(ex.events  + v.events))
# # #                 ex.factors = list(set(ex.factors + v.factors))
# # #                 if v.risk_score > ex.risk_score:
# # #                     ex.risk_score = v.risk_score
# # #                     ex.risk_level = v.risk_level
# # #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# # #                     ex.annotated_frame = v.annotated_frame
# # #         self._violations = list(unique.values())

# # #     def _merge_by_time_window(self):
# # #         if not self._violations:
# # #             return
# # #         self._violations.sort(key=lambda x: x.timestamp)
# # #         merged = []
# # #         group  = [self._violations[0]]
# # #         for v in self._violations[1:]:
# # #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# # #                 group.append(v)
# # #             else:
# # #                 merged.append(self._merge_group(group))
# # #                 group = [v]
# # #         merged.append(self._merge_group(group))
# # #         self._violations = merged

# # #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# # #         base            = group[0]
# # #         events, factors = [], []
# # #         max_risk        = base.risk_score
# # #         risk_level      = base.risk_level
# # #         best_frame      = base.annotated_frame
# # #         for v in group:
# # #             events.extend(v.events)
# # #             factors.extend(v.factors)
# # #             if v.risk_score > max_risk:
# # #                 max_risk, risk_level = v.risk_score, v.risk_level
# # #             if best_frame is None and v.annotated_frame is not None:
# # #                 best_frame = v.annotated_frame
# # #         return _Violation(
# # #             timestamp        = base.timestamp,
# # #             time_str         = base.time_str,
# # #             frame_index      = base.frame_index,
# # #             type             = base.type,
# # #             events           = list(set(events)),
# # #             severity         = base.severity,
# # #             duration         = base.duration,
# # #             risk_score       = max_risk,
# # #             risk_level       = risk_level,
# # #             confidence       = base.confidence,
# # #             factors          = list(set(factors)),
# # #             source_filename  = base.source_filename,
# # #             local_time_str   = base.local_time_str,
# # #             annotated_frame  = best_frame,
# # #         )

# # #     def extract_violation_frames(self, video_path: str):
# # #         print("[ViolationStore] Saving frames...")
# # #         need_video = [v for v in self._violations if v.annotated_frame is None]
# # #         saved = 0
# # #         for v in self._violations:
# # #             if v.annotated_frame is not None:
# # #                 v.frame_path      = self._save_frame(v.annotated_frame, v.events, v.time_str)
# # #                 v.annotated_frame = None
# # #                 saved += 1
# # #         if need_video:
# # #             cap = cv2.VideoCapture(video_path)
# # #             if not cap.isOpened():
# # #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# # #             else:
# # #                 seen: set = set()
# # #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# # #                     if v.frame_index in seen:
# # #                         continue
# # #                     seen.add(v.frame_index)
# # #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# # #                     ret, frame = cap.read()
# # #                     if not ret:
# # #                         continue
# # #                     v.frame_path = self._save_frame(frame, v.events, v.time_str)
# # #                     saved += 1
# # #                 cap.release()
# # #         print(f"[ViolationStore] {saved} frames saved")

# # #     def _save_frame(self, frame: np.ndarray, events: List[str], time_str: str) -> str:
# # #         distraction   = "_".join(events)
# # #         filename_time = time_str.replace(":", "-")
# # #         filename      = f"{distraction}_{filename_time}.jpg"
# # #         path          = os.path.join(self.frames_dir, filename)
# # #         ok = cv2.imwrite(path, cv2.resize(frame, (640, 360)),
# # #                          [cv2.IMWRITE_JPEG_QUALITY, 85])
# # #         if not ok:
# # #             print(f"[ViolationStore] imwrite failed: {path}")
# # #         return os.path.join(self.analysis_id, "frames", filename)

# # #     def _build_report(self, processing_time: float = 0.0) -> dict:
# # #         return {
# # #             "analysis_id":     self.analysis_id,
# # #             "train_detail_id": self.train_detail_id,
# # #             "processing_time": round(processing_time, 3),
# # #             # Single video → keep as dict for backwards compat; batch → list
# # #             "video_info": self.video_infos[0] if len(self.video_infos) == 1
# # #                           else self.video_infos,
# # #             "violations": [
# # #                 {
# # #                     "timestamp":   v.time_str,
# # #                     "frame_index": v.frame_index,
# # #                     "events":      v.events,
# # #                     "severity":    v.severity,
# # #                     "duration":    v.duration,
# # #                     "risk_score":  v.risk_score,
# # #                     "risk_level":  v.risk_level,
# # #                     "confidence":  v.confidence,
# # #                     "factors":     v.factors,
# # #                     "original_video_timestamp": f"{v.source_filename} {v.local_time_str}",
# # #                     "frame_path":  v.frame_path,
# # #                 }
# # #                 for v in self._violations
# # #             ],
# # #         }

# # #     def finalize(self, processing_time: float = 0.0) -> str:
# # #         self._deduplicate_by_frame()
# # #         self._merge_by_time_window()
# # #         # Extract frames from every video in the batch (or the single video)
# # #         for vi in self.video_infos:
# # #             if vi and vi.get("videoPath"):
# # #                 self.extract_violation_frames(vi["videoPath"])
# # #         report   = self._build_report(processing_time=processing_time)
# # #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# # #         with open(out_path, "w", encoding="utf-8") as f:
# # #             json.dump(report, f, indent=2)
# # #         print(f"[ViolationStore] JSON report     : {out_path}")
# # #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# # #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# # #         try:
# # #             from utils.db_s3_uploader import finalize_and_upload
# # #             finalize_and_upload(
# # #                 report_path     = out_path,
# # #                 analysis_id     = self.analysis_id,
# # #                 train_detail_id = self.train_detail_id,
# # #             )
# # #         except Exception as exc:
# # #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# # #         return out_path



# # # utils/violation_store.py
# # # ─────────────────────────────────────────────────────────────────
# # # Change from original:
# # #   _save_frame() now appends _f{frame_index} to the filename so
# # #   two violations of the same type at the same timestamp never
# # #   silently overwrite each other.
# # #   e.g.  phone_use_00-13-16_f6762.jpg
# # #         seat_absence_00-00-17_f516.jpg
# # # ─────────────────────────────────────────────────────────────────

# # from __future__ import annotations

# # import json
# # import os
# # from dataclasses import dataclass
# # from typing import Any, Dict, List, Optional

# # import cv2
# # import numpy as np

# # OUTPUTS_ROOT = "outputs"
# # MERGE_WINDOW = 2.0


# # @dataclass
# # class _Violation:
# #     timestamp:       float
# #     time_str:        str
# #     frame_index:     int
# #     type:            str
# #     events:          List[str]
# #     severity:        str
# #     duration:        float
# #     risk_score:      int
# #     risk_level:      str
# #     confidence:      float
# #     factors:         List[str]
# #     source_filename: str                  = ""
# #     local_time_str:  str                  = ""
# #     frame_path:      Optional[str]        = None
# #     annotated_frame: Optional[np.ndarray] = None


# # class ViolationStore:

# #     def __init__(
# #         self,
# #         analysis_id:     str,
# #         train_detail_id: int,
# #         video_info:      Optional[Dict[str, Any]] = None,
# #     ):
# #         self.analysis_id     = analysis_id
# #         self.train_detail_id = train_detail_id
# #         self.video_infos: List[Dict[str, Any]] = (
# #             [video_info] if video_info is not None else []
# #         )

# #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# #         self.frames_dir = os.path.join(self.output_dir, "frames")
# #         os.makedirs(self.frames_dir, exist_ok=True)

# #         self._violations: List[_Violation] = []
# #         self._seen_frames: set             = set()
# #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# #     # ── called once per video in batch mode ──────────────────────
# #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# #         self.video_infos.append(video_info)

# #     # ── called from main.py for every detected violation ─────────
# #     def record_violation(
# #         self,
# #         annotated_frame:  np.ndarray,
# #         video_time:       float,
# #         frame_index:      int,
# #         event_type:       str,
# #         original_frame:   Optional[np.ndarray] = None,
# #         severity:         str   = "CRITICAL",
# #         confidence:       float = 0.9,
# #         risk_score:       int   = 80,
# #         risk_level:       str   = "CRITICAL",
# #         factors:          Optional[List[str]] = None,
# #         duration:         float = 0.0,
# #         source_filename:  str   = "",
# #         local_video_time: float = -1.0,
# #     ):
# #         # Deduplicate: same frame + same event type recorded only once.
# #         # Different event types on the same frame are each recorded.
# #         # frame_index is globally unique across batch (frame_offset in main.py).
# #         dedup_key = (frame_index, event_type)
# #         if dedup_key in self._seen_frames:
# #             return
# #         self._seen_frames.add(dedup_key)

# #         factors  = factors or []
# #         t        = int(round(video_time))
# #         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

# #         # Local timestamp = time within the source video file (before cumulative offset)
# #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# #         local_str = f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"

# #         self._violations.append(_Violation(
# #             timestamp        = video_time,
# #             time_str         = time_str,
# #             frame_index      = frame_index,
# #             type             = event_type,
# #             events           = [event_type],
# #             severity         = severity,
# #             duration         = round(duration, 2),
# #             risk_score       = risk_score,
# #             risk_level       = risk_level,
# #             confidence       = round(confidence, 3),
# #             factors          = list(factors),
# #             source_filename  = source_filename,
# #             local_time_str   = local_str,
# #             annotated_frame  = annotated_frame.copy() if annotated_frame is not None else None,
# #         ))

# #     # ─────────────────────────────────────────────────────────────

# #     def _deduplicate_by_frame(self):
# #         unique: Dict[int, _Violation] = {}
# #         for v in self._violations:
# #             if v.frame_index not in unique:
# #                 unique[v.frame_index] = v
# #             else:
# #                 ex = unique[v.frame_index]
# #                 ex.events  = list(set(ex.events  + v.events))
# #                 ex.factors = list(set(ex.factors + v.factors))
# #                 if v.risk_score > ex.risk_score:
# #                     ex.risk_score = v.risk_score
# #                     ex.risk_level = v.risk_level
# #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# #                     ex.annotated_frame = v.annotated_frame
# #         self._violations = list(unique.values())

# #     def _merge_by_time_window(self):
# #         if not self._violations:
# #             return
# #         self._violations.sort(key=lambda x: x.timestamp)
# #         merged = []
# #         group  = [self._violations[0]]
# #         for v in self._violations[1:]:
# #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# #                 group.append(v)
# #             else:
# #                 merged.append(self._merge_group(group))
# #                 group = [v]
# #         merged.append(self._merge_group(group))
# #         self._violations = merged

# #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# #         base            = group[0]
# #         events, factors = [], []
# #         max_risk        = base.risk_score
# #         risk_level      = base.risk_level
# #         best_frame      = base.annotated_frame
# #         for v in group:
# #             events.extend(v.events)
# #             factors.extend(v.factors)
# #             if v.risk_score > max_risk:
# #                 max_risk, risk_level = v.risk_score, v.risk_level
# #             if best_frame is None and v.annotated_frame is not None:
# #                 best_frame = v.annotated_frame
# #         return _Violation(
# #             timestamp        = base.timestamp,
# #             time_str         = base.time_str,
# #             frame_index      = base.frame_index,
# #             type             = base.type,
# #             events           = list(set(events)),
# #             severity         = base.severity,
# #             duration         = base.duration,
# #             risk_score       = max_risk,
# #             risk_level       = risk_level,
# #             confidence       = base.confidence,
# #             factors          = list(set(factors)),
# #             source_filename  = base.source_filename,
# #             local_time_str   = base.local_time_str,
# #             annotated_frame  = best_frame,
# #         )

# #     # ─────────────────────────────────────────────────────────────

# #     def extract_violation_frames(self, video_path: str):
# #         print("[ViolationStore] Saving frames...")
# #         need_video = [v for v in self._violations if v.annotated_frame is None]
# #         saved = 0

# #         for v in self._violations:
# #             if v.annotated_frame is not None:
# #                 v.frame_path      = self._save_frame(
# #                     v.annotated_frame, v.events, v.time_str, v.frame_index
# #                 )
# #                 v.annotated_frame = None
# #                 saved += 1

# #         if need_video:
# #             cap = cv2.VideoCapture(video_path)
# #             if not cap.isOpened():
# #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# #             else:
# #                 seen: set = set()
# #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# #                     if v.frame_index in seen:
# #                         continue
# #                     seen.add(v.frame_index)
# #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# #                     ret, frame = cap.read()
# #                     if not ret:
# #                         continue
# #                     v.frame_path = self._save_frame(
# #                         frame, v.events, v.time_str, v.frame_index
# #                     )
# #                     saved += 1
# #                 cap.release()

# #         print(f"[ViolationStore] {saved} frames saved")

# #     def _save_frame(
# #         self,
# #         frame:       np.ndarray,
# #         events:      List[str],
# #         time_str:    str,
# #         frame_index: int,            # ← ADDED: makes filename globally unique
# #     ) -> str:
# #         """
# #         Save one violation frame as JPEG.

# #         Filename: {event_types}_{hh-mm-ss}_f{frame_index}.jpg
# #         Example:  phone_use_00-13-16_f6762.jpg
# #                   seat_absence_00-00-17_f516.jpg

# #         frame_index prevents two violations of the same type at the
# #         same timestamp from silently overwriting each other.
# #         """
# #         distraction   = "_".join(events)
# #         filename_time = time_str.replace(":", "-")
# #         filename      = f"{distraction}_{filename_time}_f{frame_index}.jpg"  # ← CHANGED
# #         path          = os.path.join(self.frames_dir, filename)
# #         ok = cv2.imwrite(
# #             path,
# #             cv2.resize(frame, (640, 360)),
# #             [cv2.IMWRITE_JPEG_QUALITY, 85],
# #         )
# #         if not ok:
# #             print(f"[ViolationStore] imwrite failed: {path}")
# #         return os.path.join(self.analysis_id, "frames", filename)

# #     # ─────────────────────────────────────────────────────────────

# #     def _build_report(self, processing_time: float = 0.0) -> dict:
# #         return {
# #             "analysis_id":     self.analysis_id,
# #             "train_detail_id": self.train_detail_id,
# #             "processing_time": round(processing_time, 3),
# #             "video_info": (
# #                 self.video_infos[0] if len(self.video_infos) == 1
# #                 else self.video_infos
# #             ),
# #             "violations": [
# #                 {
# #                     "timestamp":   v.time_str,
# #                     "frame_index": v.frame_index,
# #                     "events":      v.events,
# #                     "severity":    v.severity,
# #                     "duration":    v.duration,
# #                     "risk_score":  v.risk_score,
# #                     "risk_level":  v.risk_level,
# #                     "confidence":  v.confidence,
# #                     "factors":     v.factors,
# #                     "original_video_timestamp": f"{v.source_filename} {v.local_time_str}",
# #                     "frame_path":  v.frame_path,
# #                 }
# #                 for v in self._violations
# #             ],
# #         }

# #     def finalize(self, processing_time: float = 0.0) -> str:
# #         self._deduplicate_by_frame()
# #         self._merge_by_time_window()

# #         # Save frames — each video's temp path is in video_infos
# #         for vi in self.video_infos:
# #             if vi and vi.get("videoPath"):
# #                 self.extract_violation_frames(vi["videoPath"])

# #         report   = self._build_report(processing_time=processing_time)
# #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# #         with open(out_path, "w", encoding="utf-8") as f:
# #             json.dump(report, f, indent=2)

# #         print(f"[ViolationStore] JSON report     : {out_path}")
# #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# #         # Upload results to S3 and update DB result_s3_path
# #         try:
# #             from utils.db_s3_uploader import finalize_and_upload
# #             finalize_and_upload(
# #                 report_path     = out_path,
# #                 analysis_id     = self.analysis_id,
# #                 train_detail_id = self.train_detail_id,
# #             )
# #         except Exception as exc:
# #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# #         return out_path

# # # # from __future__ import annotations

# # # # import json
# # # # import os
# # # # from dataclasses import dataclass
# # # # from typing import Any, Dict, List, Optional

# # # # import cv2
# # # # import numpy as np

# # # # OUTPUTS_ROOT = "outputs"
# # # # MERGE_WINDOW = 2.0


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # INTERNAL DATA CLASS
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # @dataclass
# # # # class _Violation:
# # # #     timestamp:       float
# # # #     time_str:        str
# # # #     frame_index:     int
# # # #     type:            str
# # # #     events:          List[str]
# # # #     severity:        str
# # # #     duration:        float
# # # #     risk_score:      int
# # # #     risk_level:      str
# # # #     confidence:      float
# # # #     factors:         List[str]
# # # #     source_filename: str                  = ""    # e.g. "ch01.mp4"
# # # #     local_time_str:  str                  = ""    # local time within that file e.g. "00:00:18"
# # # #     frame_path:      Optional[str]        = None
# # # #     annotated_frame: Optional[np.ndarray] = None


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # VIOLATION STORE
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # class ViolationStore:
# # # #     """
# # # #     Accumulates all violations found across one analysis run (single video
# # # #     or a multi-video batch that shares the same analysis_id / folder_name).

# # # #     Usage
# # # #     ─────
# # # #     1. Construct once per analysis run.
# # # #     2. Call record_violation() from the pipeline for every distraction event.
# # # #     3. Call finalize() at the very end — it deduplicates, merges, saves frames,
# # # #        writes the JSON report, then calls db_s3_uploader.finalize_and_upload()
# # # #        to push everything to S3 and record results in the DB.
# # # #     """

# # # #     def __init__(
# # # #         self,
# # # #         analysis_id:     str,
# # # #         train_detail_id: int,
# # # #         video_info:      Optional[Dict[str, Any]] = None,
# # # #     ):
# # # #         self.analysis_id     = analysis_id
# # # #         self.train_detail_id = train_detail_id
# # # #         # video_infos is always a list:
# # # #         #   • 1 entry for single-video runs
# # # #         #   • N entries for batch runs (add_video_info called per video)
# # # #         self.video_infos: List[Dict[str, Any]] = (
# # # #             [video_info] if video_info is not None else []
# # # #         )

# # # #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# # # #         self.frames_dir = os.path.join(self.output_dir, "frames")
# # # #         os.makedirs(self.frames_dir, exist_ok=True)

# # # #         self._violations:  List[_Violation] = []
# # # #         self._seen_frames: set              = set()
# # # #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# # # #     # ── Public helpers ────────────────────────────────────────────────────────

# # # #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# # # #         """
# # # #         Register one video's metadata into the shared store.
# # # #         Called once per video in batch mode.
# # # #         """
# # # #         self.video_infos.append(video_info)

# # # #     def record_violation(
# # # #         self,
# # # #         annotated_frame:  np.ndarray,
# # # #         video_time:       float,
# # # #         frame_index:      int,
# # # #         event_type:       str,
# # # #         original_frame:   Optional[np.ndarray] = None,
# # # #         severity:         str   = "CRITICAL",
# # # #         confidence:       float = 0.9,
# # # #         risk_score:       int   = 80,
# # # #         risk_level:       str   = "CRITICAL",
# # # #         factors:          Optional[List[str]] = None,
# # # #         duration:         float = 0.0,
# # # #         source_filename:  str   = "",    # original filename e.g. "ch01.mp4"
# # # #         local_video_time: float = -1.0,  # video_time before global offset; -1 = same as video_time
# # # #     ) -> None:
# # # #         """
# # # #         Record one distraction event.

# # # #         Deduplication key is (frame_index, event_type) so that:
# # # #           • the same violation type on the same frame is recorded only once
# # # #           • different violation types on the same frame are each recorded
# # # #           • frame numbers from different videos never collide because
# # # #             main.py applies a frame_offset to make every global frame_index
# # # #             unique across the batch
# # # #         """
# # # #         dedup_key = (frame_index, event_type)
# # # #         if dedup_key in self._seen_frames:
# # # #             return
# # # #         self._seen_frames.add(dedup_key)

# # # #         factors  = factors or []
# # # #         t        = int(round(video_time))
# # # #         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

# # # #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# # # #         local_str = (
# # # #             f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
# # # #         )

# # # #         self._violations.append(
# # # #             _Violation(
# # # #                 timestamp        = video_time,
# # # #                 time_str         = time_str,
# # # #                 frame_index      = frame_index,
# # # #                 type             = event_type,
# # # #                 events           = [event_type],
# # # #                 severity         = severity,
# # # #                 duration         = round(duration, 2),
# # # #                 risk_score       = risk_score,
# # # #                 risk_level       = risk_level,
# # # #                 confidence       = round(confidence, 3),
# # # #                 factors          = list(factors),
# # # #                 source_filename  = source_filename,
# # # #                 local_time_str   = local_str,
# # # #                 annotated_frame  = (
# # # #                     annotated_frame.copy() if annotated_frame is not None else None
# # # #                 ),
# # # #             )
# # # #         )

# # # #     # ── Finalize ──────────────────────────────────────────────────────────────

# # # #     def finalize(self, processing_time: float = 0.0) -> str:
# # # #         """
# # # #         1. Deduplicate violations that share the same frame.
# # # #         2. Merge violations that are within MERGE_WINDOW seconds of each other.
# # # #         3. Save annotated frame images to disk (falls back to re-reading from
# # # #            the source video when no annotated frame was captured).
# # # #         4. Write analysis_report.json.
# # # #         5. Upload frames + report to S3 and record in DB via db_s3_uploader.
# # # #         Returns the local path to analysis_report.json.
# # # #         """
# # # #         self._deduplicate_by_frame()
# # # #         self._merge_by_time_window()

# # # #         # Extract frames from every video in the batch (or the single video)
# # # #         for vi in self.video_infos:
# # # #             if vi and vi.get("videoPath"):
# # # #                 self.extract_violation_frames(vi["videoPath"])

# # # #         report   = self._build_report(processing_time=processing_time)
# # # #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# # # #         with open(out_path, "w", encoding="utf-8") as f:
# # # #             json.dump(report, f, indent=2)

# # # #         print(f"[ViolationStore] JSON report     : {out_path}")
# # # #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# # # #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# # # #         try:
# # # #             from utils.db_s3_uploader import finalize_and_upload
# # # #             finalize_and_upload(
# # # #                 report_path     = out_path,
# # # #                 analysis_id     = self.analysis_id,
# # # #                 train_detail_id = self.train_detail_id,
# # # #             )
# # # #         except Exception as exc:
# # # #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# # # #         return out_path

# # # #     # ── Private — deduplication & merging ────────────────────────────────────

# # # #     def _deduplicate_by_frame(self) -> None:
# # # #         unique: Dict[int, _Violation] = {}
# # # #         for v in self._violations:
# # # #             if v.frame_index not in unique:
# # # #                 unique[v.frame_index] = v
# # # #             else:
# # # #                 ex = unique[v.frame_index]
# # # #                 ex.events  = list(set(ex.events  + v.events))
# # # #                 ex.factors = list(set(ex.factors + v.factors))
# # # #                 if v.risk_score > ex.risk_score:
# # # #                     ex.risk_score = v.risk_score
# # # #                     ex.risk_level = v.risk_level
# # # #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# # # #                     ex.annotated_frame = v.annotated_frame
# # # #         self._violations = list(unique.values())

# # # #     def _merge_by_time_window(self) -> None:
# # # #         if not self._violations:
# # # #             return
# # # #         self._violations.sort(key=lambda x: x.timestamp)
# # # #         merged: List[_Violation] = []
# # # #         group  = [self._violations[0]]
# # # #         for v in self._violations[1:]:
# # # #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# # # #                 group.append(v)
# # # #             else:
# # # #                 merged.append(self._merge_group(group))
# # # #                 group = [v]
# # # #         merged.append(self._merge_group(group))
# # # #         self._violations = merged

# # # #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# # # #         base            = group[0]
# # # #         events: List[str]  = []
# # # #         factors: List[str] = []
# # # #         max_risk   = base.risk_score
# # # #         risk_level = base.risk_level
# # # #         best_frame = base.annotated_frame
# # # #         for v in group:
# # # #             events.extend(v.events)
# # # #             factors.extend(v.factors)
# # # #             if v.risk_score > max_risk:
# # # #                 max_risk, risk_level = v.risk_score, v.risk_level
# # # #             if best_frame is None and v.annotated_frame is not None:
# # # #                 best_frame = v.annotated_frame
# # # #         return _Violation(
# # # #             timestamp        = base.timestamp,
# # # #             time_str         = base.time_str,
# # # #             frame_index      = base.frame_index,
# # # #             type             = base.type,
# # # #             events           = list(set(events)),
# # # #             severity         = base.severity,
# # # #             duration         = base.duration,
# # # #             risk_score       = max_risk,
# # # #             risk_level       = risk_level,
# # # #             confidence       = base.confidence,
# # # #             factors          = list(set(factors)),
# # # #             source_filename  = base.source_filename,
# # # #             local_time_str   = base.local_time_str,
# # # #             annotated_frame  = best_frame,
# # # #         )

# # # #     # ── Private — frame extraction & saving ──────────────────────────────────

# # # #     def extract_violation_frames(self, video_path: str) -> None:
# # # #         print("[ViolationStore] Saving frames...")
# # # #         need_video = [v for v in self._violations if v.annotated_frame is None]
# # # #         saved = 0

# # # #         # First pass: save violations that already have an annotated frame
# # # #         for v in self._violations:
# # # #             if v.annotated_frame is not None:
# # # #                 v.frame_path      = self._save_frame(v.annotated_frame, v.events, v.time_str)
# # # #                 v.annotated_frame = None   # free memory
# # # #                 saved += 1

# # # #         # Second pass: re-read from the source video for any that are missing
# # # #         if need_video:
# # # #             cap = cv2.VideoCapture(video_path)
# # # #             if not cap.isOpened():
# # # #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# # # #             else:
# # # #                 seen: set = set()
# # # #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# # # #                     if v.frame_index in seen:
# # # #                         continue
# # # #                     seen.add(v.frame_index)
# # # #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# # # #                     ret, frame = cap.read()
# # # #                     if not ret:
# # # #                         continue
# # # #                     v.frame_path = self._save_frame(frame, v.events, v.time_str)
# # # #                     saved += 1
# # # #                 cap.release()

# # # #         print(f"[ViolationStore] {saved} frames saved")

# # # #     def _save_frame(
# # # #         self,
# # # #         frame:    np.ndarray,
# # # #         events:   List[str],
# # # #         time_str: str,
# # # #     ) -> str:
# # # #         distraction   = "_".join(events)
# # # #         filename_time = time_str.replace(":", "-")
# # # #         filename      = f"{distraction}_{filename_time}.jpg"
# # # #         path          = os.path.join(self.frames_dir, filename)
# # # #         ok = cv2.imwrite(
# # # #             path,
# # # #             cv2.resize(frame, (640, 360)),
# # # #             [cv2.IMWRITE_JPEG_QUALITY, 85],
# # # #         )
# # # #         if not ok:
# # # #             print(f"[ViolationStore] imwrite failed: {path}")
# # # #         return os.path.join(self.analysis_id, "frames", filename)

# # # #     # ── Private — report builder ──────────────────────────────────────────────

# # # #     def _build_report(self, processing_time: float = 0.0) -> dict:
# # # #         return {
# # # #             "analysis_id":     self.analysis_id,
# # # #             "train_detail_id": self.train_detail_id,
# # # #             "processing_time": round(processing_time, 3),
# # # #             # Single video → keep as dict for backwards compat; batch → list
# # # #             "video_info": (
# # # #                 self.video_infos[0]
# # # #                 if len(self.video_infos) == 1
# # # #                 else self.video_infos
# # # #             ),
# # # #             "violations": [
# # # #                 {
# # # #                     "timestamp":   v.time_str,
# # # #                     "frame_index": v.frame_index,
# # # #                     "events":      v.events,
# # # #                     "severity":    v.severity,
# # # #                     "duration":    v.duration,
# # # #                     "risk_score":  v.risk_score,
# # # #                     "risk_level":  v.risk_level,
# # # #                     "confidence":  v.confidence,
# # # #                     "factors":     v.factors,
# # # #                     "original_video_timestamp": (
# # # #                         f"{v.source_filename} {v.local_time_str}"
# # # #                     ),
# # # #                     "frame_path":  v.frame_path,
# # # #                 }
# # # #                 for v in self._violations
# # # #             ],
# # # #         }

# # # from __future__ import annotations

# # # import json
# # # import os
# # # from dataclasses import dataclass
# # # from typing import Any, Dict, List, Optional

# # # import cv2
# # # import numpy as np

# # # OUTPUTS_ROOT = "outputs"
# # # MERGE_WINDOW = 2.0


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # # INTERNAL DATA CLASS
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # @dataclass
# # # class _Violation:
# # #     timestamp:       float
# # #     time_str:        str
# # #     frame_index:     int
# # #     type:            str
# # #     events:          List[str]
# # #     severity:        str
# # #     duration:        float
# # #     risk_score:      int
# # #     risk_level:      str
# # #     confidence:      float
# # #     factors:         List[str]
# # #     source_filename: str                  = ""    # e.g. "ch01.mp4"
# # #     local_time_str:  str                  = ""    # local time within that file e.g. "00:00:18"
# # #     frame_path:      Optional[str]        = None
# # #     annotated_frame: Optional[np.ndarray] = None


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # # VIOLATION STORE
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # class ViolationStore:
# # #     """
# # #     Accumulates all violations found across one analysis run (single video
# # #     or a multi-video batch that shares the same analysis_id / folder_name).

# # #     Usage
# # #     ─────
# # #     1. Construct once per analysis run.
# # #     2. Call record_violation() from the pipeline for every distraction event.
# # #     3. Call finalize() at the very end — it deduplicates, merges, saves frames,
# # #        writes the JSON report, then calls db_s3_uploader.finalize_and_upload()
# # #        to push everything to S3 and record results in the DB.
# # #     """

# # #     def __init__(
# # #         self,
# # #         analysis_id:     str,
# # #         train_detail_id: int,
# # #         video_info:      Optional[Dict[str, Any]] = None,
# # #     ):
# # #         self.analysis_id     = analysis_id
# # #         self.train_detail_id = train_detail_id
# # #         # video_infos is always a list:
# # #         #   • 1 entry for single-video runs
# # #         #   • N entries for batch runs (add_video_info called per video)
# # #         self.video_infos: List[Dict[str, Any]] = (
# # #             [video_info] if video_info is not None else []
# # #         )

# # #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# # #         self.frames_dir = os.path.join(self.output_dir, "frames")
# # #         os.makedirs(self.frames_dir, exist_ok=True)

# # #         self._violations:  List[_Violation] = []
# # #         self._seen_frames: set              = set()
# # #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# # #     # ── Public helpers ────────────────────────────────────────────────────────

# # #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# # #         """
# # #         Register one video's metadata into the shared store.
# # #         Called once per video in batch mode.
# # #         """
# # #         self.video_infos.append(video_info)

# # #     def record_violation(
# # #         self,
# # #         annotated_frame:  np.ndarray,
# # #         video_time:       float,
# # #         frame_index:      int,
# # #         event_type:       str,
# # #         original_frame:   Optional[np.ndarray] = None,
# # #         severity:         str   = "CRITICAL",
# # #         confidence:       float = 0.9,
# # #         risk_score:       int   = 80,
# # #         risk_level:       str   = "CRITICAL",
# # #         factors:          Optional[List[str]] = None,
# # #         duration:         float = 0.0,
# # #         source_filename:  str   = "",    # original filename e.g. "ch01.mp4"
# # #         local_video_time: float = -1.0,  # video_time before global offset; -1 = same as video_time
# # #     ) -> None:
# # #         """
# # #         Record one distraction event.

# # #         Deduplication key is (frame_index, event_type) so that:
# # #           • the same violation type on the same frame is recorded only once
# # #           • different violation types on the same frame are each recorded
# # #           • frame numbers from different videos never collide because
# # #             main.py applies a frame_offset to make every global frame_index
# # #             unique across the batch
# # #         """
# # #         dedup_key = (frame_index, event_type)
# # #         if dedup_key in self._seen_frames:
# # #             return
# # #         self._seen_frames.add(dedup_key)

# # #         factors  = factors or []
# # #         t        = int(round(video_time))
# # #         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

# # #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# # #         local_str = (
# # #             f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
# # #         )

# # #         self._violations.append(
# # #             _Violation(
# # #                 timestamp        = video_time,
# # #                 time_str         = time_str,
# # #                 frame_index      = frame_index,
# # #                 type             = event_type,
# # #                 events           = [event_type],
# # #                 severity         = severity,
# # #                 duration         = round(duration, 2),
# # #                 risk_score       = risk_score,
# # #                 risk_level       = risk_level,
# # #                 confidence       = round(confidence, 3),
# # #                 factors          = list(factors),
# # #                 source_filename  = source_filename,
# # #                 local_time_str   = local_str,
# # #                 annotated_frame  = (
# # #                     annotated_frame.copy() if annotated_frame is not None else None
# # #                 ),
# # #             )
# # #         )

# # #     # ── Finalize ──────────────────────────────────────────────────────────────

# # #     def finalize(self, processing_time: float = 0.0) -> str:
# # #         """
# # #         1. Deduplicate violations that share the same frame.
# # #         2. Merge violations that are within MERGE_WINDOW seconds of each other.
# # #         3. Save annotated frame images to disk (falls back to re-reading from
# # #            the source video when no annotated frame was captured).
# # #         4. Write analysis_report.json.
# # #         5. Upload frames + report to S3 and record in DB via db_s3_uploader.
# # #         Returns the local path to analysis_report.json.
# # #         """
# # #         self._deduplicate_by_frame()
# # #         self._merge_by_time_window()

# # #         # Extract frames from every video in the batch (or the single video)
# # #         for vi in self.video_infos:
# # #             if vi and vi.get("videoPath"):
# # #                 self.extract_violation_frames(vi["videoPath"])

# # #         report   = self._build_report(processing_time=processing_time)
# # #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# # #         with open(out_path, "w", encoding="utf-8") as f:
# # #             json.dump(report, f, indent=2)

# # #         print(f"[ViolationStore] JSON report     : {out_path}")
# # #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# # #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# # #         try:
# # #             from utils.db_s3_uploader import finalize_and_upload
# # #             finalize_and_upload(
# # #                 report_path     = out_path,
# # #                 analysis_id     = self.analysis_id,
# # #                 train_detail_id = self.train_detail_id,
# # #             )
# # #         except Exception as exc:
# # #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# # #         return out_path

# # #     # ── Private — deduplication & merging ────────────────────────────────────

# # #     def _deduplicate_by_frame(self) -> None:
# # #         unique: Dict[int, _Violation] = {}
# # #         for v in self._violations:
# # #             if v.frame_index not in unique:
# # #                 unique[v.frame_index] = v
# # #             else:
# # #                 ex = unique[v.frame_index]
# # #                 ex.events  = list(set(ex.events  + v.events))
# # #                 ex.factors = list(set(ex.factors + v.factors))
# # #                 if v.risk_score > ex.risk_score:
# # #                     ex.risk_score = v.risk_score
# # #                     ex.risk_level = v.risk_level
# # #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# # #                     ex.annotated_frame = v.annotated_frame
# # #         self._violations = list(unique.values())

# # #     def _merge_by_time_window(self) -> None:
# # #         if not self._violations:
# # #             return
# # #         self._violations.sort(key=lambda x: x.timestamp)
# # #         merged: List[_Violation] = []
# # #         group  = [self._violations[0]]
# # #         for v in self._violations[1:]:
# # #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# # #                 group.append(v)
# # #             else:
# # #                 merged.append(self._merge_group(group))
# # #                 group = [v]
# # #         merged.append(self._merge_group(group))
# # #         self._violations = merged

# # #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# # #         base             = group[0]
# # #         events: List[str]  = []
# # #         factors: List[str] = []
# # #         max_risk   = base.risk_score
# # #         risk_level = base.risk_level
# # #         best_frame = base.annotated_frame
# # #         for v in group:
# # #             events.extend(v.events)
# # #             factors.extend(v.factors)
# # #             if v.risk_score > max_risk:
# # #                 max_risk, risk_level = v.risk_score, v.risk_level
# # #             if best_frame is None and v.annotated_frame is not None:
# # #                 best_frame = v.annotated_frame
# # #         return _Violation(
# # #             timestamp        = base.timestamp,
# # #             time_str         = base.time_str,
# # #             frame_index      = base.frame_index,
# # #             type             = base.type,
# # #             events           = list(set(events)),
# # #             severity         = base.severity,
# # #             duration         = base.duration,
# # #             risk_score       = max_risk,
# # #             risk_level       = risk_level,
# # #             confidence       = base.confidence,
# # #             factors          = list(set(factors)),
# # #             source_filename  = base.source_filename,
# # #             local_time_str   = base.local_time_str,
# # #             annotated_frame  = best_frame,
# # #         )

# # #     # ── Private — frame extraction & saving ──────────────────────────────────

# # #     def extract_violation_frames(self, video_path: str) -> None:
# # #         print("[ViolationStore] Saving frames...")
# # #         need_video = [v for v in self._violations if v.annotated_frame is None]
# # #         saved = 0

# # #         # First pass: save violations that already have an annotated frame
# # #         for v in self._violations:
# # #             if v.annotated_frame is not None:
# # #                 v.frame_path      = self._save_frame(
# # #                     v.annotated_frame, v.events, v.time_str, v.frame_index
# # #                 )
# # #                 v.annotated_frame = None   # free memory
# # #                 saved += 1

# # #         # Second pass: re-read from the source video for any that are missing
# # #         if need_video:
# # #             cap = cv2.VideoCapture(video_path)
# # #             if not cap.isOpened():
# # #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# # #             else:
# # #                 seen: set = set()
# # #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# # #                     if v.frame_index in seen:
# # #                         continue
# # #                     seen.add(v.frame_index)
# # #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# # #                     ret, frame = cap.read()
# # #                     if not ret:
# # #                         continue
# # #                     v.frame_path = self._save_frame(
# # #                         frame, v.events, v.time_str, v.frame_index
# # #                     )
# # #                     saved += 1
# # #                 cap.release()

# # #         print(f"[ViolationStore] {saved} frames saved")

# # #     def _save_frame(
# # #         self,
# # #         frame:       np.ndarray,
# # #         events:      List[str],
# # #         time_str:    str,
# # #         frame_index: int,
# # #     ) -> str:
# # #         """
# # #         Save a single violation frame as a JPEG.

# # #         Filename format:  <events>_<HH-MM-SS>_f<frame_index>.jpg
# # #         Example:          seat_absence_00-01-14_f2106.jpg
# # #                           seat_absence_drowsy_00-03-02_f5058.jpg
# # #         """
# # #         distraction   = "_".join(sorted(events))   # sorted for deterministic name
# # #         filename_time = time_str.replace(":", "-")
# # #         filename      = f"{distraction}_{filename_time}_f{frame_index}.jpg"
# # #         path          = os.path.join(self.frames_dir, filename)
# # #         ok = cv2.imwrite(
# # #             path,
# # #             cv2.resize(frame, (640, 360)),
# # #             [cv2.IMWRITE_JPEG_QUALITY, 85],
# # #         )
# # #         if not ok:
# # #             print(f"[ViolationStore] imwrite failed: {path}")
# # #         return os.path.join(self.analysis_id, "frames", filename)

# # #     # ── Private — report builder ──────────────────────────────────────────────

# # #     def _build_report(self, processing_time: float = 0.0) -> dict:
# # #         return {
# # #             "analysis_id":     self.analysis_id,
# # #             "train_detail_id": self.train_detail_id,
# # #             "processing_time": round(processing_time, 3),
# # #             # Single video → keep as dict for backwards compat; batch → list
# # #             "video_info": (
# # #                 self.video_infos[0]
# # #                 if len(self.video_infos) == 1
# # #                 else self.video_infos
# # #             ),
# # #             "violations": [
# # #                 {
# # #                     "timestamp":   v.time_str,
# # #                     "frame_index": v.frame_index,
# # #                     "events":      v.events,
# # #                     "severity":    v.severity,
# # #                     "duration":    v.duration,
# # #                     "risk_score":  v.risk_score,
# # #                     "risk_level":  v.risk_level,
# # #                     "confidence":  v.confidence,
# # #                     "factors":     v.factors,
# # #                     "original_video_timestamp": (
# # #                         f"{v.source_filename} {v.local_time_str}"
# # #                     ),
# # #                     "frame_path":  v.frame_path,
# # #                 }
# # #                 for v in self._violations
# # #             ],
# # #         }


# # from __future__ import annotations

# # import json
# # import os
# # from dataclasses import dataclass
# # from typing import Any, Dict, List, Optional

# # import cv2
# # import numpy as np

# # OUTPUTS_ROOT = "outputs"
# # MERGE_WINDOW = 2.0


# # # ══════════════════════════════════════════════════════════════════════════════
# # # INTERNAL DATA CLASS
# # # ══════════════════════════════════════════════════════════════════════════════

# # @dataclass
# # class _Violation:
# #     timestamp:       float
# #     time_str:        str
# #     frame_index:     int
# #     type:            str
# #     events:          List[str]
# #     severity:        str
# #     duration:        float
# #     risk_score:      int
# #     risk_level:      str
# #     confidence:      float
# #     factors:         List[str]
# #     source_filename: str                  = ""    # DB filename e.g. "mobile.mp4"
# #     local_time_str:  str                  = ""    # local time within that file e.g. "00:00:23"
# #     frame_path:      Optional[str]        = None
# #     annotated_frame: Optional[np.ndarray] = None


# # # ══════════════════════════════════════════════════════════════════════════════
# # # VIOLATION STORE
# # # ══════════════════════════════════════════════════════════════════════════════

# # class ViolationStore:
# #     """
# #     Accumulates all violations found across one analysis run (single video
# #     or a multi-video batch that shares the same analysis_id / folder_name).

# #     Batch mode usage (api.py)
# #     ─────────────────────────
# #     1. Construct ONCE for the whole folder (no video_info in __init__).
# #     2. Pass as shared_vstore= to each GadgetDetectionPipeline.
# #        The pipeline calls add_video_info() automatically.
# #     3. Call finalize() ONCE after all videos in the folder are done.

# #     Standalone mode usage (CLI / single video)
# #     ──────────────────────────────────────────
# #     1. Construct with video_info= for the single video.
# #     2. Pipeline calls finalize() automatically at the end of run().
# #     """

# #     def __init__(
# #         self,
# #         analysis_id:     str,
# #         train_detail_id: int,
# #         video_info:      Optional[Dict[str, Any]] = None,
# #     ):
# #         self.analysis_id     = analysis_id
# #         self.train_detail_id = train_detail_id
# #         # Always a list — 0 entries until add_video_info() is called (batch mode),
# #         # or 1 entry when video_info is provided (standalone mode).
# #         self.video_infos: List[Dict[str, Any]] = (
# #             [video_info] if video_info is not None else []
# #         )

# #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# #         self.frames_dir = os.path.join(self.output_dir, "frames")
# #         os.makedirs(self.frames_dir, exist_ok=True)

# #         self._violations:  List[_Violation] = []
# #         self._seen_frames: set              = set()
# #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# #     # ── Public helpers ────────────────────────────────────────────────────────

# #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# #         """Append one video's metadata. Called once per video in batch mode."""
# #         self.video_infos.append(video_info)

# #     def record_violation(
# #         self,
# #         annotated_frame:  np.ndarray,
# #         video_time:       float,          # global timestamp (offset-adjusted)
# #         frame_index:      int,            # global frame index (offset-adjusted)
# #         event_type:       str,
# #         original_frame:   Optional[np.ndarray] = None,
# #         severity:         str   = "CRITICAL",
# #         confidence:       float = 0.9,
# #         risk_score:       int   = 80,
# #         risk_level:       str   = "CRITICAL",
# #         factors:          Optional[List[str]] = None,
# #         duration:         float = 0.0,
# #         source_filename:  str   = "",     # DB filename shown in original_video_timestamp
# #         local_video_time: float = -1.0,   # local time within the source file; -1 = same as video_time
# #     ) -> None:
# #         """
# #         Record one distraction event.

# #         Deduplication key is (frame_index, event_type):
# #           • same event on the same global frame is recorded once
# #           • different events on the same global frame are each recorded
# #           • global frame_index is unique across videos (frame_offset applied in main.py)
# #         """
# #         dedup_key = (frame_index, event_type)
# #         if dedup_key in self._seen_frames:
# #             return
# #         self._seen_frames.add(dedup_key)

# #         factors  = factors or []
# #         t        = int(round(video_time))
# #         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

# #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# #         local_str = (
# #             f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
# #         )

# #         self._violations.append(
# #             _Violation(
# #                 timestamp        = video_time,
# #                 time_str         = time_str,
# #                 frame_index      = frame_index,
# #                 type             = event_type,
# #                 events           = [event_type],
# #                 severity         = severity,
# #                 duration         = round(duration, 2),
# #                 risk_score       = risk_score,
# #                 risk_level       = risk_level,
# #                 confidence       = round(confidence, 3),
# #                 factors          = list(factors),
# #                 source_filename  = source_filename,
# #                 local_time_str   = local_str,
# #                 annotated_frame  = (
# #                     annotated_frame.copy() if annotated_frame is not None else None
# #                 ),
# #             )
# #         )

# #     # ── Finalize ──────────────────────────────────────────────────────────────

# #     def finalize(self, processing_time: float = 0.0) -> str:
# #         """
# #         Deduplicate → merge → save frames → write JSON → upload to S3/DB.
# #         Returns the local path to analysis_report.json.
# #         """
# #         self._deduplicate_by_frame()
# #         self._merge_by_time_window()

# #         # Extract frames from every video in the batch (or the single video)
# #         for vi in self.video_infos:
# #             if vi and vi.get("videoPath"):
# #                 self.extract_violation_frames(vi["videoPath"])

# #         report   = self._build_report(processing_time=processing_time)
# #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# #         with open(out_path, "w", encoding="utf-8") as f:
# #             json.dump(report, f, indent=2)

# #         print(f"[ViolationStore] JSON report     : {out_path}")
# #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# #         try:
# #             from utils.db_s3_uploader import finalize_and_upload
# #             finalize_and_upload(
# #                 report_path     = out_path,
# #                 analysis_id     = self.analysis_id,
# #                 train_detail_id = self.train_detail_id,
# #             )
# #         except Exception as exc:
# #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# #         return out_path

# #     # ── Private — deduplication & merging ────────────────────────────────────

# #     def _deduplicate_by_frame(self) -> None:
# #         unique: Dict[int, _Violation] = {}
# #         for v in self._violations:
# #             if v.frame_index not in unique:
# #                 unique[v.frame_index] = v
# #             else:
# #                 ex = unique[v.frame_index]
# #                 ex.events  = list(set(ex.events  + v.events))
# #                 ex.factors = list(set(ex.factors + v.factors))
# #                 if v.risk_score > ex.risk_score:
# #                     ex.risk_score = v.risk_score
# #                     ex.risk_level = v.risk_level
# #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# #                     ex.annotated_frame = v.annotated_frame
# #         self._violations = list(unique.values())

# #     def _merge_by_time_window(self) -> None:
# #         if not self._violations:
# #             return
# #         self._violations.sort(key=lambda x: x.timestamp)
# #         merged: List[_Violation] = []
# #         group  = [self._violations[0]]
# #         for v in self._violations[1:]:
# #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# #                 group.append(v)
# #             else:
# #                 merged.append(self._merge_group(group))
# #                 group = [v]
# #         merged.append(self._merge_group(group))
# #         self._violations = merged

# #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# #         base             = group[0]
# #         events: List[str]  = []
# #         factors: List[str] = []
# #         max_risk   = base.risk_score
# #         risk_level = base.risk_level
# #         best_frame = base.annotated_frame
# #         for v in group:
# #             events.extend(v.events)
# #             factors.extend(v.factors)
# #             if v.risk_score > max_risk:
# #                 max_risk, risk_level = v.risk_score, v.risk_level
# #             if best_frame is None and v.annotated_frame is not None:
# #                 best_frame = v.annotated_frame
# #         return _Violation(
# #             timestamp        = base.timestamp,
# #             time_str         = base.time_str,
# #             frame_index      = base.frame_index,
# #             type             = base.type,
# #             events           = list(set(events)),
# #             severity         = base.severity,
# #             duration         = base.duration,
# #             risk_score       = max_risk,
# #             risk_level       = risk_level,
# #             confidence       = base.confidence,
# #             factors          = list(set(factors)),
# #             source_filename  = base.source_filename,
# #             local_time_str   = base.local_time_str,
# #             annotated_frame  = best_frame,
# #         )

# #     # ── Private — frame extraction & saving ──────────────────────────────────

# #     def extract_violation_frames(self, video_path: str) -> None:
# #         print("[ViolationStore] Saving frames...")
# #         need_video = [v for v in self._violations if v.annotated_frame is None]
# #         saved = 0

# #         # First pass: save violations that already have an annotated frame
# #         for v in self._violations:
# #             if v.annotated_frame is not None:
# #                 v.frame_path      = self._save_frame(v.annotated_frame, v.events, v.time_str)
# #                 v.annotated_frame = None   # free memory
# #                 saved += 1

# #         # Second pass: re-read from the source video for any that are missing
# #         if need_video:
# #             cap = cv2.VideoCapture(video_path)
# #             if not cap.isOpened():
# #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# #             else:
# #                 seen: set = set()
# #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# #                     if v.frame_index in seen:
# #                         continue
# #                     seen.add(v.frame_index)
# #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# #                     ret, frame = cap.read()
# #                     if not ret:
# #                         continue
# #                     v.frame_path = self._save_frame(frame, v.events, v.time_str)
# #                     saved += 1
# #                 cap.release()

# #         print(f"[ViolationStore] {saved} frames saved")

# #     def _save_frame(
# #         self,
# #         frame:    np.ndarray,
# #         events:   List[str],
# #         time_str: str,
# #     ) -> str:
# #         """
# #         Save a single violation frame as JPEG.

# #         Filename format:  <events>_<HH-MM-SS>.jpg
# #         Example:          seat_absence_00-01-14.jpg
# #                           seat_absence_drowsy_00-03-02.jpg
# #                           phone_use_00-00-24.jpg
# #         """
# #         distraction   = "_".join(sorted(events))   # sorted for deterministic name
# #         filename_time = time_str.replace(":", "-")
# #         filename      = f"{distraction}_{filename_time}.jpg"
# #         path          = os.path.join(self.frames_dir, filename)
# #         ok = cv2.imwrite(
# #             path,
# #             cv2.resize(frame, (640, 360)),
# #             [cv2.IMWRITE_JPEG_QUALITY, 85],
# #         )
# #         if not ok:
# #             print(f"[ViolationStore] imwrite failed: {path}")
# #         return os.path.join(self.analysis_id, "frames", filename)

# #     # ── Private — report builder ──────────────────────────────────────────────

# #     def _build_report(self, processing_time: float = 0.0) -> dict:
# #         return {
# #             "analysis_id":     self.analysis_id,
# #             "train_detail_id": self.train_detail_id,
# #             "processing_time": round(processing_time, 3),
# #             # Single video → dict (backwards compat); batch → list
# #             "video_info": (
# #                 self.video_infos[0]
# #                 if len(self.video_infos) == 1
# #                 else self.video_infos
# #             ),
# #             "violations": [
# #                 {
# #                     "timestamp":   v.time_str,
# #                     "frame_index": v.frame_index,
# #                     "events":      v.events,
# #                     "severity":    v.severity,
# #                     "duration":    v.duration,
# #                     "risk_score":  v.risk_score,
# #                     "risk_level":  v.risk_level,
# #                     "confidence":  v.confidence,
# #                     "factors":     v.factors,
# #                     # "original_video_timestamp" = "<db_filename> <local_HH:MM:SS>"
# #                     # When timestamp == original (video 1), local_time_str == time_str
# #                     "original_video_timestamp": (
# #                         f"{v.source_filename} {v.local_time_str}"
# #                     ),
# #                     "frame_path":  v.frame_path,
# #                 }
# #                 for v in self._violations
# #             ],
# #         }


# from __future__ import annotations

# import json
# import os
# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional

# import cv2
# import numpy as np

# OUTPUTS_ROOT = "outputs"
# MERGE_WINDOW = 2.0


# # ══════════════════════════════════════════════════════════════════════════════
# # INTERNAL DATA CLASS
# # ══════════════════════════════════════════════════════════════════════════════

# @dataclass
# class _Violation:
#     timestamp:       float
#     time_str:        str
#     frame_index:     int
#     type:            str
#     events:          List[str]
#     severity:        str
#     duration:        float
#     risk_score:      int
#     risk_level:      str
#     confidence:      float
#     factors:         List[str]
#     source_filename: str                  = ""    # DB filename e.g. "mobile.mp4"
#     local_time_str:  str                  = ""    # local time within that file e.g. "00:00:23"
#     frame_path:      Optional[str]        = None
#     annotated_frame: Optional[np.ndarray] = None


# # ══════════════════════════════════════════════════════════════════════════════
# # VIOLATION STORE
# # ══════════════════════════════════════════════════════════════════════════════

# class ViolationStore:
#     """
#     Accumulates all violations found across one analysis run (single video
#     or a multi-video batch that shares the same analysis_id / folder_name).

#     Batch mode usage (api.py)
#     ─────────────────────────
#     1. Construct ONCE for the whole folder (no video_info in __init__).
#     2. Pass as shared_vstore= to each GadgetDetectionPipeline.
#        The pipeline calls add_video_info() automatically.
#     3. Call finalize() ONCE after all videos in the folder are done.

#     Standalone mode usage (CLI / single video)
#     ──────────────────────────────────────────
#     1. Construct with video_info= for the single video.
#     2. Pipeline calls finalize() automatically at the end of run().
#     """

#     def __init__(
#         self,
#         analysis_id:     str,
#         train_detail_id: int,
#         video_info:      Optional[Dict[str, Any]] = None,
#     ):
#         self.analysis_id     = analysis_id
#         self.train_detail_id = train_detail_id
#         # Always a list — 0 entries until add_video_info() is called (batch mode),
#         # or 1 entry when video_info is provided (standalone mode).
#         self.video_infos: List[Dict[str, Any]] = (
#             [video_info] if video_info is not None else []
#         )

#         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
#         self.frames_dir = os.path.join(self.output_dir, "frames")
#         os.makedirs(self.frames_dir, exist_ok=True)

#         self._violations:  List[_Violation] = []
#         self._seen_frames: set              = set()
#         print(f"[ViolationStore] Output dir : {self.output_dir}")

#     # ── Public helpers ────────────────────────────────────────────────────────

#     def add_video_info(self, video_info: Dict[str, Any]) -> None:
#         """Append one video's metadata. Called once per video in batch mode."""
#         self.video_infos.append(video_info)

#     def record_violation(
#         self,
#         annotated_frame:  np.ndarray,
#         video_time:       float,          # global timestamp (offset-adjusted)
#         frame_index:      int,            # global frame index (offset-adjusted)
#         event_type:       str,
#         original_frame:   Optional[np.ndarray] = None,
#         severity:         str   = "CRITICAL",
#         confidence:       float = 0.9,
#         risk_score:       int   = 80,
#         risk_level:       str   = "CRITICAL",
#         factors:          Optional[List[str]] = None,
#         duration:         float = 0.0,
#         source_filename:  str   = "",     # DB filename shown in original_video_timestamp
#         local_video_time: float = -1.0,   # local time within the source file; -1 = same as video_time
#     ) -> None:
#         """
#         Record one distraction event.

#         Deduplication key is (frame_index, event_type):
#           • same event on the same global frame is recorded once
#           • different events on the same global frame are each recorded
#           • global frame_index is unique across videos (frame_offset applied in main.py)
#         """
#         dedup_key = (frame_index, event_type)
#         if dedup_key in self._seen_frames:
#             return
#         self._seen_frames.add(dedup_key)

#         factors  = factors or []
#         t        = int(round(video_time))
#         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

#         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
#         local_str = (
#             f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
#         )

#         self._violations.append(
#             _Violation(
#                 timestamp        = video_time,
#                 time_str         = time_str,
#                 frame_index      = frame_index,
#                 type             = event_type,
#                 events           = [event_type],
#                 severity         = severity,
#                 duration         = round(duration, 2),
#                 risk_score       = risk_score,
#                 risk_level       = risk_level,
#                 confidence       = round(confidence, 3),
#                 factors          = list(factors),
#                 source_filename  = source_filename,
#                 local_time_str   = local_str,
#                 annotated_frame  = (
#                     annotated_frame.copy() if annotated_frame is not None else None
#                 ),
#             )
#         )

#     # ── Finalize ──────────────────────────────────────────────────────────────

#     def finalize(self, processing_time: float = 0.0) -> str:
#         """
#         Deduplicate → merge → save frames → write JSON → upload to S3/DB.
#         Returns the local path to analysis_report.json.
#         """
#         self._deduplicate_by_frame()
#         self._merge_by_time_window()

#         # Extract frames from every video in the batch (or the single video)
#         for vi in self.video_infos:
#             if vi and vi.get("videoPath"):
#                 self.extract_violation_frames(vi["videoPath"])

#         report   = self._build_report(processing_time=processing_time)
#         out_path = os.path.join(self.output_dir, "analysis_report.json")
#         with open(out_path, "w", encoding="utf-8") as f:
#             json.dump(report, f, indent=2)

#         print(f"[ViolationStore] JSON report     : {out_path}")
#         print(f"[ViolationStore] Violations      : {len(self._violations)}")
#         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

#         try:
#             from utils.db_s3_uploader import finalize_and_upload
#             finalize_and_upload(
#                 report_path     = out_path,
#                 analysis_id     = self.analysis_id,
#                 train_detail_id = self.train_detail_id,
#             )
#         except Exception as exc:
#             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

#         return out_path

#     # ── Private — deduplication & merging ────────────────────────────────────

#     def _deduplicate_by_frame(self) -> None:
#         unique: Dict[int, _Violation] = {}
#         for v in self._violations:
#             if v.frame_index not in unique:
#                 unique[v.frame_index] = v
#             else:
#                 ex = unique[v.frame_index]
#                 ex.events  = list(set(ex.events  + v.events))
#                 ex.factors = list(set(ex.factors + v.factors))
#                 if v.risk_score > ex.risk_score:
#                     ex.risk_score = v.risk_score
#                     ex.risk_level = v.risk_level
#                 if ex.annotated_frame is None and v.annotated_frame is not None:
#                     ex.annotated_frame = v.annotated_frame
#         self._violations = list(unique.values())

#     def _merge_by_time_window(self) -> None:
#         if not self._violations:
#             return
#         self._violations.sort(key=lambda x: x.timestamp)
#         merged: List[_Violation] = []
#         group  = [self._violations[0]]
#         for v in self._violations[1:]:
#             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
#                 group.append(v)
#             else:
#                 merged.append(self._merge_group(group))
#                 group = [v]
#         merged.append(self._merge_group(group))
#         self._violations = merged

#     def _merge_group(self, group: List[_Violation]) -> _Violation:
#         base             = group[0]
#         events: List[str]  = []
#         factors: List[str] = []
#         max_risk   = base.risk_score
#         risk_level = base.risk_level
#         best_frame = base.annotated_frame
#         for v in group:
#             events.extend(v.events)
#             factors.extend(v.factors)
#             if v.risk_score > max_risk:
#                 max_risk, risk_level = v.risk_score, v.risk_level
#             if best_frame is None and v.annotated_frame is not None:
#                 best_frame = v.annotated_frame
#         return _Violation(
#             timestamp        = base.timestamp,
#             time_str         = base.time_str,
#             frame_index      = base.frame_index,
#             type             = base.type,
#             events           = list(set(events)),
#             severity         = base.severity,
#             duration         = base.duration,
#             risk_score       = max_risk,
#             risk_level       = risk_level,
#             confidence       = base.confidence,
#             factors          = list(set(factors)),
#             source_filename  = base.source_filename,
#             local_time_str   = base.local_time_str,
#             annotated_frame  = best_frame,
#         )

#     # ── Private — frame extraction & saving ──────────────────────────────────

#     def extract_violation_frames(self, video_path: str) -> None:
#         print("[ViolationStore] Saving frames...")
#         need_video = [v for v in self._violations if v.annotated_frame is None]
#         saved = 0

#         # First pass: save violations that already have an annotated frame
#         for v in self._violations:
#             if v.annotated_frame is not None:
#                 v.frame_path      = self._save_frame(v.annotated_frame, v.events, v.time_str)
#                 v.annotated_frame = None   # free memory
#                 saved += 1

#         # Second pass: re-read from the source video for any that are missing
#         if need_video:
#             cap = cv2.VideoCapture(video_path)
#             if not cap.isOpened():
#                 print(f"[ViolationStore] Cannot open video: {video_path}")
#             else:
#                 seen: set = set()
#                 for v in sorted(need_video, key=lambda x: x.frame_index):
#                     if v.frame_index in seen:
#                         continue
#                     seen.add(v.frame_index)
#                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
#                     ret, frame = cap.read()
#                     if not ret:
#                         continue
#                     v.frame_path = self._save_frame(frame, v.events, v.time_str)
#                     saved += 1
#                 cap.release()

#         print(f"[ViolationStore] {saved} frames saved")

#     def _save_frame(
#         self,
#         frame:    np.ndarray,
#         events:   List[str],
#         time_str: str,
#     ) -> str:
#         """
#         Save a single violation frame as JPEG.

#         Filename format:  <events>_<HH-MM-SS>.jpg
#         Example:          seat_absence_00-01-14.jpg
#                           seat_absence_drowsy_00-03-02.jpg
#                           phone_use_00-00-24.jpg
#         """
#         distraction   = "_".join(sorted(events))   # sorted for deterministic name
#         filename_time = time_str.replace(":", "-")
#         filename      = f"{distraction}_{filename_time}.jpg"
#         path          = os.path.join(self.frames_dir, filename)
#         ok = cv2.imwrite(
#             path,
#             cv2.resize(frame, (640, 360)),
#             [cv2.IMWRITE_JPEG_QUALITY, 85],
#         )
#         if not ok:
#             print(f"[ViolationStore] imwrite failed: {path}")
#         return os.path.join(self.analysis_id, "frames", filename)

#     # ── Private — report builder ──────────────────────────────────────────────

#     def _build_report(self, processing_time: float = 0.0) -> dict:
#         return {
#             "analysis_id":     self.analysis_id,
#             "train_detail_id": self.train_detail_id,
#             "processing_time": round(processing_time, 3),
#             # Single video → dict (backwards compat); batch → list
#             "video_info": (
#                 self.video_infos[0]
#                 if len(self.video_infos) == 1
#                 else self.video_infos
#             ),
#             "violations": [
#                 {
#                     "timestamp":   v.time_str,
#                     "frame_index": v.frame_index,
#                     "events":      v.events,
#                     "severity":    v.severity,
#                     "duration":    v.duration,
#                     "risk_score":  v.risk_score,
#                     "risk_level":  v.risk_level,
#                     "confidence":  v.confidence,
#                     "factors":     v.factors,
#                     # "original_video_timestamp" = "<db_filename> <local_HH:MM:SS>"
#                     # When timestamp == original (video 1), local_time_str == time_str
#                     "original_video_timestamp": (
#                         f"{v.source_filename} {v.local_time_str}"
#                     ),
#                     "frame_path":  v.frame_path,
#                 }
#                 for v in self._violations
#             ],
#         }


# # from __future__ import annotations

# # import json
# # import os
# # from dataclasses import dataclass
# # from typing import Any, Dict, List, Optional

# # import cv2
# # import numpy as np

# # OUTPUTS_ROOT = "outputs"
# # MERGE_WINDOW = 2.0


# # @dataclass
# # class _Violation:
# #     timestamp:       float
# #     time_str:        str
# #     frame_index:     int
# #     type:            str
# #     events:          List[str]
# #     severity:        str
# #     duration:        float
# #     risk_score:      int
# #     risk_level:      str
# #     confidence:      float
# #     factors:         List[str]
# #     source_filename: str                   = ""    # e.g. "ax.mp4"
# #     local_time_str:  str                   = ""    # local time within that file e.g. "00:00:18"
# #     frame_path:      Optional[str]         = None
# #     annotated_frame: Optional[np.ndarray]  = None


# # class ViolationStore:

# #     def __init__(self, analysis_id: str, train_detail_id: int, video_info: Optional[Dict[str, Any]] = None):
# #         self.analysis_id     = analysis_id
# #         self.train_detail_id = train_detail_id
# #         # video_infos is always a list — 1 entry for single-video, N entries for batch
# #         self.video_infos: List[Dict[str, Any]] = [video_info] if video_info is not None else []

# #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# #         self.frames_dir = os.path.join(self.output_dir, "frames")
# #         os.makedirs(self.frames_dir, exist_ok=True)

# #         self._violations: List[_Violation] = []
# #         self._seen_frames: set             = set()
# #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# #         """
# #         Register one video's metadata into the shared store.
# #         Called once per video in batch mode (api.py passes shared_vstore).
# #         """
# #         self.video_infos.append(video_info)

# #     def record_violation(
# #         self,
# #         annotated_frame: np.ndarray,
# #         video_time:      float,
# #         frame_index:     int,
# #         event_type:      str,
# #         original_frame:  Optional[np.ndarray] = None,
# #         severity:        str   = "CRITICAL",
# #         confidence:      float = 0.9,
# #         risk_score:      int   = 80,
# #         risk_level:      str   = "CRITICAL",
# #         factors:         Optional[List[str]] = None,
# #         duration:        float = 0.0,
# #         source_filename: str   = "",   # original upload filename e.g. "ax.mp4"
# #         local_video_time: float = -1.0, # raw video_time before offset; -1 = same as video_time
# #     ):
# #         # Deduplicate on (frame_index, event_type) so that:
# #         #  • the same violation type on the same frame is recorded only once
# #         #  • different violation types on the same frame are each recorded
# #         #  • frame numbers from different videos never collide (frame_offset
# #         #    in main.py makes every global frame_index unique across the batch)
# #         dedup_key = (frame_index, event_type)
# #         if dedup_key in self._seen_frames:
# #             return
# #         self._seen_frames.add(dedup_key)
# #         factors   = factors or []
# #         t         = int(round(video_time))
# #         time_str  = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"
# #         # Build the per-file local timestamp (time within the source video)
# #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# #         local_str = f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
# #         self._violations.append(_Violation(
# #             timestamp        = video_time,
# #             time_str         = time_str,
# #             frame_index      = frame_index,
# #             type             = event_type,
# #             events           = [event_type],
# #             severity         = severity,
# #             duration         = round(duration, 2),
# #             risk_score       = risk_score,
# #             risk_level       = risk_level,
# #             confidence       = round(confidence, 3),
# #             factors          = list(factors),
# #             source_filename  = source_filename,
# #             local_time_str   = local_str,
# #             annotated_frame  = annotated_frame.copy() if annotated_frame is not None else None,
# #         ))

# #     def _deduplicate_by_frame(self):
# #         unique: Dict[int, _Violation] = {}
# #         for v in self._violations:
# #             if v.frame_index not in unique:
# #                 unique[v.frame_index] = v
# #             else:
# #                 ex = unique[v.frame_index]
# #                 ex.events  = list(set(ex.events  + v.events))
# #                 ex.factors = list(set(ex.factors + v.factors))
# #                 if v.risk_score > ex.risk_score:
# #                     ex.risk_score = v.risk_score
# #                     ex.risk_level = v.risk_level
# #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# #                     ex.annotated_frame = v.annotated_frame
# #         self._violations = list(unique.values())

# #     def _merge_by_time_window(self):
# #         if not self._violations:
# #             return
# #         self._violations.sort(key=lambda x: x.timestamp)
# #         merged = []
# #         group  = [self._violations[0]]
# #         for v in self._violations[1:]:
# #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# #                 group.append(v)
# #             else:
# #                 merged.append(self._merge_group(group))
# #                 group = [v]
# #         merged.append(self._merge_group(group))
# #         self._violations = merged

# #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# #         base            = group[0]
# #         events, factors = [], []
# #         max_risk        = base.risk_score
# #         risk_level      = base.risk_level
# #         best_frame      = base.annotated_frame
# #         for v in group:
# #             events.extend(v.events)
# #             factors.extend(v.factors)
# #             if v.risk_score > max_risk:
# #                 max_risk, risk_level = v.risk_score, v.risk_level
# #             if best_frame is None and v.annotated_frame is not None:
# #                 best_frame = v.annotated_frame
# #         return _Violation(
# #             timestamp        = base.timestamp,
# #             time_str         = base.time_str,
# #             frame_index      = base.frame_index,
# #             type             = base.type,
# #             events           = list(set(events)),
# #             severity         = base.severity,
# #             duration         = base.duration,
# #             risk_score       = max_risk,
# #             risk_level       = risk_level,
# #             confidence       = base.confidence,
# #             factors          = list(set(factors)),
# #             source_filename  = base.source_filename,
# #             local_time_str   = base.local_time_str,
# #             annotated_frame  = best_frame,
# #         )

# #     def extract_violation_frames(self, video_path: str):
# #         print("[ViolationStore] Saving frames...")
# #         need_video = [v for v in self._violations if v.annotated_frame is None]
# #         saved = 0
# #         for v in self._violations:
# #             if v.annotated_frame is not None:
# #                 v.frame_path      = self._save_frame(v.annotated_frame, v.events, v.time_str)
# #                 v.annotated_frame = None
# #                 saved += 1
# #         if need_video:
# #             cap = cv2.VideoCapture(video_path)
# #             if not cap.isOpened():
# #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# #             else:
# #                 seen: set = set()
# #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# #                     if v.frame_index in seen:
# #                         continue
# #                     seen.add(v.frame_index)
# #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# #                     ret, frame = cap.read()
# #                     if not ret:
# #                         continue
# #                     v.frame_path = self._save_frame(frame, v.events, v.time_str)
# #                     saved += 1
# #                 cap.release()
# #         print(f"[ViolationStore] {saved} frames saved")

# #     def _save_frame(self, frame: np.ndarray, events: List[str], time_str: str) -> str:
# #         distraction   = "_".join(events)
# #         filename_time = time_str.replace(":", "-")
# #         filename      = f"{distraction}_{filename_time}.jpg"
# #         path          = os.path.join(self.frames_dir, filename)
# #         ok = cv2.imwrite(path, cv2.resize(frame, (640, 360)),
# #                          [cv2.IMWRITE_JPEG_QUALITY, 85])
# #         if not ok:
# #             print(f"[ViolationStore] imwrite failed: {path}")
# #         return os.path.join(self.analysis_id, "frames", filename)

# #     def _build_report(self, processing_time: float = 0.0) -> dict:
# #         return {
# #             "analysis_id":     self.analysis_id,
# #             "train_detail_id": self.train_detail_id,
# #             "processing_time": round(processing_time, 3),
# #             # Single video → keep as dict for backwards compat; batch → list
# #             "video_info": self.video_infos[0] if len(self.video_infos) == 1
# #                           else self.video_infos,
# #             "violations": [
# #                 {
# #                     "timestamp":   v.time_str,
# #                     "frame_index": v.frame_index,
# #                     "events":      v.events,
# #                     "severity":    v.severity,
# #                     "duration":    v.duration,
# #                     "risk_score":  v.risk_score,
# #                     "risk_level":  v.risk_level,
# #                     "confidence":  v.confidence,
# #                     "factors":     v.factors,
# #                     "original_video_timestamp": f"{v.source_filename} {v.local_time_str}",
# #                     "frame_path":  v.frame_path,
# #                 }
# #                 for v in self._violations
# #             ],
# #         }

# #     def finalize(self, processing_time: float = 0.0) -> str:
# #         self._deduplicate_by_frame()
# #         self._merge_by_time_window()
# #         # Extract frames from every video in the batch (or the single video)
# #         for vi in self.video_infos:
# #             if vi and vi.get("videoPath"):
# #                 self.extract_violation_frames(vi["videoPath"])
# #         report   = self._build_report(processing_time=processing_time)
# #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# #         with open(out_path, "w", encoding="utf-8") as f:
# #             json.dump(report, f, indent=2)
# #         print(f"[ViolationStore] JSON report     : {out_path}")
# #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# #         try:
# #             from utils.db_s3_uploader import finalize_and_upload
# #             finalize_and_upload(
# #                 report_path     = out_path,
# #                 analysis_id     = self.analysis_id,
# #                 train_detail_id = self.train_detail_id,
# #             )
# #         except Exception as exc:
# #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# #         return out_path



# # utils/violation_store.py
# # ─────────────────────────────────────────────────────────────────
# # Change from original:
# #   _save_frame() now appends _f{frame_index} to the filename so
# #   two violations of the same type at the same timestamp never
# #   silently overwrite each other.
# #   e.g.  phone_use_00-13-16_f6762.jpg
# #         seat_absence_00-00-17_f516.jpg
# # ─────────────────────────────────────────────────────────────────

# from __future__ import annotations

# import json
# import os
# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional

# import cv2
# import numpy as np

# OUTPUTS_ROOT = "outputs"
# MERGE_WINDOW = 2.0


# @dataclass
# class _Violation:
#     timestamp:       float
#     time_str:        str
#     frame_index:     int
#     type:            str
#     events:          List[str]
#     severity:        str
#     duration:        float
#     risk_score:      int
#     risk_level:      str
#     confidence:      float
#     factors:         List[str]
#     source_filename: str                  = ""
#     local_time_str:  str                  = ""
#     frame_path:      Optional[str]        = None
#     annotated_frame: Optional[np.ndarray] = None


# class ViolationStore:

#     def __init__(
#         self,
#         analysis_id:     str,
#         train_detail_id: int,
#         video_info:      Optional[Dict[str, Any]] = None,
#     ):
#         self.analysis_id     = analysis_id
#         self.train_detail_id = train_detail_id
#         self.video_infos: List[Dict[str, Any]] = (
#             [video_info] if video_info is not None else []
#         )

#         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
#         self.frames_dir = os.path.join(self.output_dir, "frames")
#         os.makedirs(self.frames_dir, exist_ok=True)

#         self._violations: List[_Violation] = []
#         self._seen_frames: set             = set()
#         print(f"[ViolationStore] Output dir : {self.output_dir}")

#     # ── called once per video in batch mode ──────────────────────
#     def add_video_info(self, video_info: Dict[str, Any]) -> None:
#         self.video_infos.append(video_info)

#     # ── called from main.py for every detected violation ─────────
#     def record_violation(
#         self,
#         annotated_frame:  np.ndarray,
#         video_time:       float,
#         frame_index:      int,
#         event_type:       str,
#         original_frame:   Optional[np.ndarray] = None,
#         severity:         str   = "CRITICAL",
#         confidence:       float = 0.9,
#         risk_score:       int   = 80,
#         risk_level:       str   = "CRITICAL",
#         factors:          Optional[List[str]] = None,
#         duration:         float = 0.0,
#         source_filename:  str   = "",
#         local_video_time: float = -1.0,
#     ):
#         # Deduplicate: same frame + same event type recorded only once.
#         # Different event types on the same frame are each recorded.
#         # frame_index is globally unique across batch (frame_offset in main.py).
#         dedup_key = (frame_index, event_type)
#         if dedup_key in self._seen_frames:
#             return
#         self._seen_frames.add(dedup_key)

#         factors  = factors or []
#         t        = int(round(video_time))
#         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

#         # Local timestamp = time within the source video file (before cumulative offset)
#         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
#         local_str = f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"

#         self._violations.append(_Violation(
#             timestamp        = video_time,
#             time_str         = time_str,
#             frame_index      = frame_index,
#             type             = event_type,
#             events           = [event_type],
#             severity         = severity,
#             duration         = round(duration, 2),
#             risk_score       = risk_score,
#             risk_level       = risk_level,
#             confidence       = round(confidence, 3),
#             factors          = list(factors),
#             source_filename  = source_filename,
#             local_time_str   = local_str,
#             annotated_frame  = annotated_frame.copy() if annotated_frame is not None else None,
#         ))

#     # ─────────────────────────────────────────────────────────────

#     def _deduplicate_by_frame(self):
#         unique: Dict[int, _Violation] = {}
#         for v in self._violations:
#             if v.frame_index not in unique:
#                 unique[v.frame_index] = v
#             else:
#                 ex = unique[v.frame_index]
#                 ex.events  = list(set(ex.events  + v.events))
#                 ex.factors = list(set(ex.factors + v.factors))
#                 if v.risk_score > ex.risk_score:
#                     ex.risk_score = v.risk_score
#                     ex.risk_level = v.risk_level
#                 if ex.annotated_frame is None and v.annotated_frame is not None:
#                     ex.annotated_frame = v.annotated_frame
#         self._violations = list(unique.values())

#     def _merge_by_time_window(self):
#         if not self._violations:
#             return
#         self._violations.sort(key=lambda x: x.timestamp)
#         merged = []
#         group  = [self._violations[0]]
#         for v in self._violations[1:]:
#             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
#                 group.append(v)
#             else:
#                 merged.append(self._merge_group(group))
#                 group = [v]
#         merged.append(self._merge_group(group))
#         self._violations = merged

#     def _merge_group(self, group: List[_Violation]) -> _Violation:
#         base            = group[0]
#         events, factors = [], []
#         max_risk        = base.risk_score
#         risk_level      = base.risk_level
#         best_frame      = base.annotated_frame
#         for v in group:
#             events.extend(v.events)
#             factors.extend(v.factors)
#             if v.risk_score > max_risk:
#                 max_risk, risk_level = v.risk_score, v.risk_level
#             if best_frame is None and v.annotated_frame is not None:
#                 best_frame = v.annotated_frame
#         return _Violation(
#             timestamp        = base.timestamp,
#             time_str         = base.time_str,
#             frame_index      = base.frame_index,
#             type             = base.type,
#             events           = list(set(events)),
#             severity         = base.severity,
#             duration         = base.duration,
#             risk_score       = max_risk,
#             risk_level       = risk_level,
#             confidence       = base.confidence,
#             factors          = list(set(factors)),
#             source_filename  = base.source_filename,
#             local_time_str   = base.local_time_str,
#             annotated_frame  = best_frame,
#         )

#     # ─────────────────────────────────────────────────────────────

#     def extract_violation_frames(self, video_path: str):
#         print("[ViolationStore] Saving frames...")
#         need_video = [v for v in self._violations if v.annotated_frame is None]
#         saved = 0

#         for v in self._violations:
#             if v.annotated_frame is not None:
#                 v.frame_path      = self._save_frame(
#                     v.annotated_frame, v.events, v.time_str, v.frame_index
#                 )
#                 v.annotated_frame = None
#                 saved += 1

#         if need_video:
#             cap = cv2.VideoCapture(video_path)
#             if not cap.isOpened():
#                 print(f"[ViolationStore] Cannot open video: {video_path}")
#             else:
#                 seen: set = set()
#                 for v in sorted(need_video, key=lambda x: x.frame_index):
#                     if v.frame_index in seen:
#                         continue
#                     seen.add(v.frame_index)
#                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
#                     ret, frame = cap.read()
#                     if not ret:
#                         continue
#                     v.frame_path = self._save_frame(
#                         frame, v.events, v.time_str, v.frame_index
#                     )
#                     saved += 1
#                 cap.release()

#         print(f"[ViolationStore] {saved} frames saved")

#     def _save_frame(
#         self,
#         frame:       np.ndarray,
#         events:      List[str],
#         time_str:    str,
#         frame_index: int,            # ← ADDED: makes filename globally unique
#     ) -> str:
#         """
#         Save one violation frame as JPEG.

#         Filename: {event_types}_{hh-mm-ss}_f{frame_index}.jpg
#         Example:  phone_use_00-13-16_f6762.jpg
#                   seat_absence_00-00-17_f516.jpg

#         frame_index prevents two violations of the same type at the
#         same timestamp from silently overwriting each other.
#         """
#         distraction   = "_".join(events)
#         filename_time = time_str.replace(":", "-")
#         filename      = f"{distraction}_{filename_time}_f{frame_index}.jpg"  # ← CHANGED
#         path          = os.path.join(self.frames_dir, filename)
#         ok = cv2.imwrite(
#             path,
#             cv2.resize(frame, (640, 360)),
#             [cv2.IMWRITE_JPEG_QUALITY, 85],
#         )
#         if not ok:
#             print(f"[ViolationStore] imwrite failed: {path}")
#         return os.path.join(self.analysis_id, "frames", filename)

#     # ─────────────────────────────────────────────────────────────

#     def _build_report(self, processing_time: float = 0.0) -> dict:
#         return {
#             "analysis_id":     self.analysis_id,
#             "train_detail_id": self.train_detail_id,
#             "processing_time": round(processing_time, 3),
#             "video_info": (
#                 self.video_infos[0] if len(self.video_infos) == 1
#                 else self.video_infos
#             ),
#             "violations": [
#                 {
#                     "timestamp":   v.time_str,
#                     "frame_index": v.frame_index,
#                     "events":      v.events,
#                     "severity":    v.severity,
#                     "duration":    v.duration,
#                     "risk_score":  v.risk_score,
#                     "risk_level":  v.risk_level,
#                     "confidence":  v.confidence,
#                     "factors":     v.factors,
#                     "original_video_timestamp": f"{v.source_filename} {v.local_time_str}",
#                     "frame_path":  v.frame_path,
#                 }
#                 for v in self._violations
#             ],
#         }

#     def finalize(self, processing_time: float = 0.0) -> str:
#         self._deduplicate_by_frame()
#         self._merge_by_time_window()

#         # Save frames — each video's temp path is in video_infos
#         for vi in self.video_infos:
#             if vi and vi.get("videoPath"):
#                 self.extract_violation_frames(vi["videoPath"])

#         report   = self._build_report(processing_time=processing_time)
#         out_path = os.path.join(self.output_dir, "analysis_report.json")
#         with open(out_path, "w", encoding="utf-8") as f:
#             json.dump(report, f, indent=2)

#         print(f"[ViolationStore] JSON report     : {out_path}")
#         print(f"[ViolationStore] Violations      : {len(self._violations)}")
#         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

#         # Upload results to S3 and update DB result_s3_path
#         try:
#             from utils.db_s3_uploader import finalize_and_upload
#             finalize_and_upload(
#                 report_path     = out_path,
#                 analysis_id     = self.analysis_id,
#                 train_detail_id = self.train_detail_id,
#             )
#         except Exception as exc:
#             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

#         return out_path

# # # from __future__ import annotations

# # # import json
# # # import os
# # # from dataclasses import dataclass
# # # from typing import Any, Dict, List, Optional

# # # import cv2
# # # import numpy as np

# # # OUTPUTS_ROOT = "outputs"
# # # MERGE_WINDOW = 2.0


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # # INTERNAL DATA CLASS
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # @dataclass
# # # class _Violation:
# # #     timestamp:       float
# # #     time_str:        str
# # #     frame_index:     int
# # #     type:            str
# # #     events:          List[str]
# # #     severity:        str
# # #     duration:        float
# # #     risk_score:      int
# # #     risk_level:      str
# # #     confidence:      float
# # #     factors:         List[str]
# # #     source_filename: str                  = ""    # e.g. "ch01.mp4"
# # #     local_time_str:  str                  = ""    # local time within that file e.g. "00:00:18"
# # #     frame_path:      Optional[str]        = None
# # #     annotated_frame: Optional[np.ndarray] = None


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # # VIOLATION STORE
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # class ViolationStore:
# # #     """
# # #     Accumulates all violations found across one analysis run (single video
# # #     or a multi-video batch that shares the same analysis_id / folder_name).

# # #     Usage
# # #     ─────
# # #     1. Construct once per analysis run.
# # #     2. Call record_violation() from the pipeline for every distraction event.
# # #     3. Call finalize() at the very end — it deduplicates, merges, saves frames,
# # #        writes the JSON report, then calls db_s3_uploader.finalize_and_upload()
# # #        to push everything to S3 and record results in the DB.
# # #     """

# # #     def __init__(
# # #         self,
# # #         analysis_id:     str,
# # #         train_detail_id: int,
# # #         video_info:      Optional[Dict[str, Any]] = None,
# # #     ):
# # #         self.analysis_id     = analysis_id
# # #         self.train_detail_id = train_detail_id
# # #         # video_infos is always a list:
# # #         #   • 1 entry for single-video runs
# # #         #   • N entries for batch runs (add_video_info called per video)
# # #         self.video_infos: List[Dict[str, Any]] = (
# # #             [video_info] if video_info is not None else []
# # #         )

# # #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# # #         self.frames_dir = os.path.join(self.output_dir, "frames")
# # #         os.makedirs(self.frames_dir, exist_ok=True)

# # #         self._violations:  List[_Violation] = []
# # #         self._seen_frames: set              = set()
# # #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# # #     # ── Public helpers ────────────────────────────────────────────────────────

# # #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# # #         """
# # #         Register one video's metadata into the shared store.
# # #         Called once per video in batch mode.
# # #         """
# # #         self.video_infos.append(video_info)

# # #     def record_violation(
# # #         self,
# # #         annotated_frame:  np.ndarray,
# # #         video_time:       float,
# # #         frame_index:      int,
# # #         event_type:       str,
# # #         original_frame:   Optional[np.ndarray] = None,
# # #         severity:         str   = "CRITICAL",
# # #         confidence:       float = 0.9,
# # #         risk_score:       int   = 80,
# # #         risk_level:       str   = "CRITICAL",
# # #         factors:          Optional[List[str]] = None,
# # #         duration:         float = 0.0,
# # #         source_filename:  str   = "",    # original filename e.g. "ch01.mp4"
# # #         local_video_time: float = -1.0,  # video_time before global offset; -1 = same as video_time
# # #     ) -> None:
# # #         """
# # #         Record one distraction event.

# # #         Deduplication key is (frame_index, event_type) so that:
# # #           • the same violation type on the same frame is recorded only once
# # #           • different violation types on the same frame are each recorded
# # #           • frame numbers from different videos never collide because
# # #             main.py applies a frame_offset to make every global frame_index
# # #             unique across the batch
# # #         """
# # #         dedup_key = (frame_index, event_type)
# # #         if dedup_key in self._seen_frames:
# # #             return
# # #         self._seen_frames.add(dedup_key)

# # #         factors  = factors or []
# # #         t        = int(round(video_time))
# # #         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

# # #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# # #         local_str = (
# # #             f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
# # #         )

# # #         self._violations.append(
# # #             _Violation(
# # #                 timestamp        = video_time,
# # #                 time_str         = time_str,
# # #                 frame_index      = frame_index,
# # #                 type             = event_type,
# # #                 events           = [event_type],
# # #                 severity         = severity,
# # #                 duration         = round(duration, 2),
# # #                 risk_score       = risk_score,
# # #                 risk_level       = risk_level,
# # #                 confidence       = round(confidence, 3),
# # #                 factors          = list(factors),
# # #                 source_filename  = source_filename,
# # #                 local_time_str   = local_str,
# # #                 annotated_frame  = (
# # #                     annotated_frame.copy() if annotated_frame is not None else None
# # #                 ),
# # #             )
# # #         )

# # #     # ── Finalize ──────────────────────────────────────────────────────────────

# # #     def finalize(self, processing_time: float = 0.0) -> str:
# # #         """
# # #         1. Deduplicate violations that share the same frame.
# # #         2. Merge violations that are within MERGE_WINDOW seconds of each other.
# # #         3. Save annotated frame images to disk (falls back to re-reading from
# # #            the source video when no annotated frame was captured).
# # #         4. Write analysis_report.json.
# # #         5. Upload frames + report to S3 and record in DB via db_s3_uploader.
# # #         Returns the local path to analysis_report.json.
# # #         """
# # #         self._deduplicate_by_frame()
# # #         self._merge_by_time_window()

# # #         # Extract frames from every video in the batch (or the single video)
# # #         for vi in self.video_infos:
# # #             if vi and vi.get("videoPath"):
# # #                 self.extract_violation_frames(vi["videoPath"])

# # #         report   = self._build_report(processing_time=processing_time)
# # #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# # #         with open(out_path, "w", encoding="utf-8") as f:
# # #             json.dump(report, f, indent=2)

# # #         print(f"[ViolationStore] JSON report     : {out_path}")
# # #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# # #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# # #         try:
# # #             from utils.db_s3_uploader import finalize_and_upload
# # #             finalize_and_upload(
# # #                 report_path     = out_path,
# # #                 analysis_id     = self.analysis_id,
# # #                 train_detail_id = self.train_detail_id,
# # #             )
# # #         except Exception as exc:
# # #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# # #         return out_path

# # #     # ── Private — deduplication & merging ────────────────────────────────────

# # #     def _deduplicate_by_frame(self) -> None:
# # #         unique: Dict[int, _Violation] = {}
# # #         for v in self._violations:
# # #             if v.frame_index not in unique:
# # #                 unique[v.frame_index] = v
# # #             else:
# # #                 ex = unique[v.frame_index]
# # #                 ex.events  = list(set(ex.events  + v.events))
# # #                 ex.factors = list(set(ex.factors + v.factors))
# # #                 if v.risk_score > ex.risk_score:
# # #                     ex.risk_score = v.risk_score
# # #                     ex.risk_level = v.risk_level
# # #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# # #                     ex.annotated_frame = v.annotated_frame
# # #         self._violations = list(unique.values())

# # #     def _merge_by_time_window(self) -> None:
# # #         if not self._violations:
# # #             return
# # #         self._violations.sort(key=lambda x: x.timestamp)
# # #         merged: List[_Violation] = []
# # #         group  = [self._violations[0]]
# # #         for v in self._violations[1:]:
# # #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# # #                 group.append(v)
# # #             else:
# # #                 merged.append(self._merge_group(group))
# # #                 group = [v]
# # #         merged.append(self._merge_group(group))
# # #         self._violations = merged

# # #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# # #         base            = group[0]
# # #         events: List[str]  = []
# # #         factors: List[str] = []
# # #         max_risk   = base.risk_score
# # #         risk_level = base.risk_level
# # #         best_frame = base.annotated_frame
# # #         for v in group:
# # #             events.extend(v.events)
# # #             factors.extend(v.factors)
# # #             if v.risk_score > max_risk:
# # #                 max_risk, risk_level = v.risk_score, v.risk_level
# # #             if best_frame is None and v.annotated_frame is not None:
# # #                 best_frame = v.annotated_frame
# # #         return _Violation(
# # #             timestamp        = base.timestamp,
# # #             time_str         = base.time_str,
# # #             frame_index      = base.frame_index,
# # #             type             = base.type,
# # #             events           = list(set(events)),
# # #             severity         = base.severity,
# # #             duration         = base.duration,
# # #             risk_score       = max_risk,
# # #             risk_level       = risk_level,
# # #             confidence       = base.confidence,
# # #             factors          = list(set(factors)),
# # #             source_filename  = base.source_filename,
# # #             local_time_str   = base.local_time_str,
# # #             annotated_frame  = best_frame,
# # #         )

# # #     # ── Private — frame extraction & saving ──────────────────────────────────

# # #     def extract_violation_frames(self, video_path: str) -> None:
# # #         print("[ViolationStore] Saving frames...")
# # #         need_video = [v for v in self._violations if v.annotated_frame is None]
# # #         saved = 0

# # #         # First pass: save violations that already have an annotated frame
# # #         for v in self._violations:
# # #             if v.annotated_frame is not None:
# # #                 v.frame_path      = self._save_frame(v.annotated_frame, v.events, v.time_str)
# # #                 v.annotated_frame = None   # free memory
# # #                 saved += 1

# # #         # Second pass: re-read from the source video for any that are missing
# # #         if need_video:
# # #             cap = cv2.VideoCapture(video_path)
# # #             if not cap.isOpened():
# # #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# # #             else:
# # #                 seen: set = set()
# # #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# # #                     if v.frame_index in seen:
# # #                         continue
# # #                     seen.add(v.frame_index)
# # #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# # #                     ret, frame = cap.read()
# # #                     if not ret:
# # #                         continue
# # #                     v.frame_path = self._save_frame(frame, v.events, v.time_str)
# # #                     saved += 1
# # #                 cap.release()

# # #         print(f"[ViolationStore] {saved} frames saved")

# # #     def _save_frame(
# # #         self,
# # #         frame:    np.ndarray,
# # #         events:   List[str],
# # #         time_str: str,
# # #     ) -> str:
# # #         distraction   = "_".join(events)
# # #         filename_time = time_str.replace(":", "-")
# # #         filename      = f"{distraction}_{filename_time}.jpg"
# # #         path          = os.path.join(self.frames_dir, filename)
# # #         ok = cv2.imwrite(
# # #             path,
# # #             cv2.resize(frame, (640, 360)),
# # #             [cv2.IMWRITE_JPEG_QUALITY, 85],
# # #         )
# # #         if not ok:
# # #             print(f"[ViolationStore] imwrite failed: {path}")
# # #         return os.path.join(self.analysis_id, "frames", filename)

# # #     # ── Private — report builder ──────────────────────────────────────────────

# # #     def _build_report(self, processing_time: float = 0.0) -> dict:
# # #         return {
# # #             "analysis_id":     self.analysis_id,
# # #             "train_detail_id": self.train_detail_id,
# # #             "processing_time": round(processing_time, 3),
# # #             # Single video → keep as dict for backwards compat; batch → list
# # #             "video_info": (
# # #                 self.video_infos[0]
# # #                 if len(self.video_infos) == 1
# # #                 else self.video_infos
# # #             ),
# # #             "violations": [
# # #                 {
# # #                     "timestamp":   v.time_str,
# # #                     "frame_index": v.frame_index,
# # #                     "events":      v.events,
# # #                     "severity":    v.severity,
# # #                     "duration":    v.duration,
# # #                     "risk_score":  v.risk_score,
# # #                     "risk_level":  v.risk_level,
# # #                     "confidence":  v.confidence,
# # #                     "factors":     v.factors,
# # #                     "original_video_timestamp": (
# # #                         f"{v.source_filename} {v.local_time_str}"
# # #                     ),
# # #                     "frame_path":  v.frame_path,
# # #                 }
# # #                 for v in self._violations
# # #             ],
# # #         }

# # from __future__ import annotations

# # import json
# # import os
# # from dataclasses import dataclass
# # from typing import Any, Dict, List, Optional

# # import cv2
# # import numpy as np

# # OUTPUTS_ROOT = "outputs"
# # MERGE_WINDOW = 2.0


# # # ══════════════════════════════════════════════════════════════════════════════
# # # INTERNAL DATA CLASS
# # # ══════════════════════════════════════════════════════════════════════════════

# # @dataclass
# # class _Violation:
# #     timestamp:       float
# #     time_str:        str
# #     frame_index:     int
# #     type:            str
# #     events:          List[str]
# #     severity:        str
# #     duration:        float
# #     risk_score:      int
# #     risk_level:      str
# #     confidence:      float
# #     factors:         List[str]
# #     source_filename: str                  = ""    # e.g. "ch01.mp4"
# #     local_time_str:  str                  = ""    # local time within that file e.g. "00:00:18"
# #     frame_path:      Optional[str]        = None
# #     annotated_frame: Optional[np.ndarray] = None


# # # ══════════════════════════════════════════════════════════════════════════════
# # # VIOLATION STORE
# # # ══════════════════════════════════════════════════════════════════════════════

# # class ViolationStore:
# #     """
# #     Accumulates all violations found across one analysis run (single video
# #     or a multi-video batch that shares the same analysis_id / folder_name).

# #     Usage
# #     ─────
# #     1. Construct once per analysis run.
# #     2. Call record_violation() from the pipeline for every distraction event.
# #     3. Call finalize() at the very end — it deduplicates, merges, saves frames,
# #        writes the JSON report, then calls db_s3_uploader.finalize_and_upload()
# #        to push everything to S3 and record results in the DB.
# #     """

# #     def __init__(
# #         self,
# #         analysis_id:     str,
# #         train_detail_id: int,
# #         video_info:      Optional[Dict[str, Any]] = None,
# #     ):
# #         self.analysis_id     = analysis_id
# #         self.train_detail_id = train_detail_id
# #         # video_infos is always a list:
# #         #   • 1 entry for single-video runs
# #         #   • N entries for batch runs (add_video_info called per video)
# #         self.video_infos: List[Dict[str, Any]] = (
# #             [video_info] if video_info is not None else []
# #         )

# #         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
# #         self.frames_dir = os.path.join(self.output_dir, "frames")
# #         os.makedirs(self.frames_dir, exist_ok=True)

# #         self._violations:  List[_Violation] = []
# #         self._seen_frames: set              = set()
# #         print(f"[ViolationStore] Output dir : {self.output_dir}")

# #     # ── Public helpers ────────────────────────────────────────────────────────

# #     def add_video_info(self, video_info: Dict[str, Any]) -> None:
# #         """
# #         Register one video's metadata into the shared store.
# #         Called once per video in batch mode.
# #         """
# #         self.video_infos.append(video_info)

# #     def record_violation(
# #         self,
# #         annotated_frame:  np.ndarray,
# #         video_time:       float,
# #         frame_index:      int,
# #         event_type:       str,
# #         original_frame:   Optional[np.ndarray] = None,
# #         severity:         str   = "CRITICAL",
# #         confidence:       float = 0.9,
# #         risk_score:       int   = 80,
# #         risk_level:       str   = "CRITICAL",
# #         factors:          Optional[List[str]] = None,
# #         duration:         float = 0.0,
# #         source_filename:  str   = "",    # original filename e.g. "ch01.mp4"
# #         local_video_time: float = -1.0,  # video_time before global offset; -1 = same as video_time
# #     ) -> None:
# #         """
# #         Record one distraction event.

# #         Deduplication key is (frame_index, event_type) so that:
# #           • the same violation type on the same frame is recorded only once
# #           • different violation types on the same frame are each recorded
# #           • frame numbers from different videos never collide because
# #             main.py applies a frame_offset to make every global frame_index
# #             unique across the batch
# #         """
# #         dedup_key = (frame_index, event_type)
# #         if dedup_key in self._seen_frames:
# #             return
# #         self._seen_frames.add(dedup_key)

# #         factors  = factors or []
# #         t        = int(round(video_time))
# #         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

# #         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
# #         local_str = (
# #             f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
# #         )

# #         self._violations.append(
# #             _Violation(
# #                 timestamp        = video_time,
# #                 time_str         = time_str,
# #                 frame_index      = frame_index,
# #                 type             = event_type,
# #                 events           = [event_type],
# #                 severity         = severity,
# #                 duration         = round(duration, 2),
# #                 risk_score       = risk_score,
# #                 risk_level       = risk_level,
# #                 confidence       = round(confidence, 3),
# #                 factors          = list(factors),
# #                 source_filename  = source_filename,
# #                 local_time_str   = local_str,
# #                 annotated_frame  = (
# #                     annotated_frame.copy() if annotated_frame is not None else None
# #                 ),
# #             )
# #         )

# #     # ── Finalize ──────────────────────────────────────────────────────────────

# #     def finalize(self, processing_time: float = 0.0) -> str:
# #         """
# #         1. Deduplicate violations that share the same frame.
# #         2. Merge violations that are within MERGE_WINDOW seconds of each other.
# #         3. Save annotated frame images to disk (falls back to re-reading from
# #            the source video when no annotated frame was captured).
# #         4. Write analysis_report.json.
# #         5. Upload frames + report to S3 and record in DB via db_s3_uploader.
# #         Returns the local path to analysis_report.json.
# #         """
# #         self._deduplicate_by_frame()
# #         self._merge_by_time_window()

# #         # Extract frames from every video in the batch (or the single video)
# #         for vi in self.video_infos:
# #             if vi and vi.get("videoPath"):
# #                 self.extract_violation_frames(vi["videoPath"])

# #         report   = self._build_report(processing_time=processing_time)
# #         out_path = os.path.join(self.output_dir, "analysis_report.json")
# #         with open(out_path, "w", encoding="utf-8") as f:
# #             json.dump(report, f, indent=2)

# #         print(f"[ViolationStore] JSON report     : {out_path}")
# #         print(f"[ViolationStore] Violations      : {len(self._violations)}")
# #         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

# #         try:
# #             from utils.db_s3_uploader import finalize_and_upload
# #             finalize_and_upload(
# #                 report_path     = out_path,
# #                 analysis_id     = self.analysis_id,
# #                 train_detail_id = self.train_detail_id,
# #             )
# #         except Exception as exc:
# #             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

# #         return out_path

# #     # ── Private — deduplication & merging ────────────────────────────────────

# #     def _deduplicate_by_frame(self) -> None:
# #         unique: Dict[int, _Violation] = {}
# #         for v in self._violations:
# #             if v.frame_index not in unique:
# #                 unique[v.frame_index] = v
# #             else:
# #                 ex = unique[v.frame_index]
# #                 ex.events  = list(set(ex.events  + v.events))
# #                 ex.factors = list(set(ex.factors + v.factors))
# #                 if v.risk_score > ex.risk_score:
# #                     ex.risk_score = v.risk_score
# #                     ex.risk_level = v.risk_level
# #                 if ex.annotated_frame is None and v.annotated_frame is not None:
# #                     ex.annotated_frame = v.annotated_frame
# #         self._violations = list(unique.values())

# #     def _merge_by_time_window(self) -> None:
# #         if not self._violations:
# #             return
# #         self._violations.sort(key=lambda x: x.timestamp)
# #         merged: List[_Violation] = []
# #         group  = [self._violations[0]]
# #         for v in self._violations[1:]:
# #             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
# #                 group.append(v)
# #             else:
# #                 merged.append(self._merge_group(group))
# #                 group = [v]
# #         merged.append(self._merge_group(group))
# #         self._violations = merged

# #     def _merge_group(self, group: List[_Violation]) -> _Violation:
# #         base             = group[0]
# #         events: List[str]  = []
# #         factors: List[str] = []
# #         max_risk   = base.risk_score
# #         risk_level = base.risk_level
# #         best_frame = base.annotated_frame
# #         for v in group:
# #             events.extend(v.events)
# #             factors.extend(v.factors)
# #             if v.risk_score > max_risk:
# #                 max_risk, risk_level = v.risk_score, v.risk_level
# #             if best_frame is None and v.annotated_frame is not None:
# #                 best_frame = v.annotated_frame
# #         return _Violation(
# #             timestamp        = base.timestamp,
# #             time_str         = base.time_str,
# #             frame_index      = base.frame_index,
# #             type             = base.type,
# #             events           = list(set(events)),
# #             severity         = base.severity,
# #             duration         = base.duration,
# #             risk_score       = max_risk,
# #             risk_level       = risk_level,
# #             confidence       = base.confidence,
# #             factors          = list(set(factors)),
# #             source_filename  = base.source_filename,
# #             local_time_str   = base.local_time_str,
# #             annotated_frame  = best_frame,
# #         )

# #     # ── Private — frame extraction & saving ──────────────────────────────────

# #     def extract_violation_frames(self, video_path: str) -> None:
# #         print("[ViolationStore] Saving frames...")
# #         need_video = [v for v in self._violations if v.annotated_frame is None]
# #         saved = 0

# #         # First pass: save violations that already have an annotated frame
# #         for v in self._violations:
# #             if v.annotated_frame is not None:
# #                 v.frame_path      = self._save_frame(
# #                     v.annotated_frame, v.events, v.time_str, v.frame_index
# #                 )
# #                 v.annotated_frame = None   # free memory
# #                 saved += 1

# #         # Second pass: re-read from the source video for any that are missing
# #         if need_video:
# #             cap = cv2.VideoCapture(video_path)
# #             if not cap.isOpened():
# #                 print(f"[ViolationStore] Cannot open video: {video_path}")
# #             else:
# #                 seen: set = set()
# #                 for v in sorted(need_video, key=lambda x: x.frame_index):
# #                     if v.frame_index in seen:
# #                         continue
# #                     seen.add(v.frame_index)
# #                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
# #                     ret, frame = cap.read()
# #                     if not ret:
# #                         continue
# #                     v.frame_path = self._save_frame(
# #                         frame, v.events, v.time_str, v.frame_index
# #                     )
# #                     saved += 1
# #                 cap.release()

# #         print(f"[ViolationStore] {saved} frames saved")

# #     def _save_frame(
# #         self,
# #         frame:       np.ndarray,
# #         events:      List[str],
# #         time_str:    str,
# #         frame_index: int,
# #     ) -> str:
# #         """
# #         Save a single violation frame as a JPEG.

# #         Filename format:  <events>_<HH-MM-SS>_f<frame_index>.jpg
# #         Example:          seat_absence_00-01-14_f2106.jpg
# #                           seat_absence_drowsy_00-03-02_f5058.jpg
# #         """
# #         distraction   = "_".join(sorted(events))   # sorted for deterministic name
# #         filename_time = time_str.replace(":", "-")
# #         filename      = f"{distraction}_{filename_time}_f{frame_index}.jpg"
# #         path          = os.path.join(self.frames_dir, filename)
# #         ok = cv2.imwrite(
# #             path,
# #             cv2.resize(frame, (640, 360)),
# #             [cv2.IMWRITE_JPEG_QUALITY, 85],
# #         )
# #         if not ok:
# #             print(f"[ViolationStore] imwrite failed: {path}")
# #         return os.path.join(self.analysis_id, "frames", filename)

# #     # ── Private — report builder ──────────────────────────────────────────────

# #     def _build_report(self, processing_time: float = 0.0) -> dict:
# #         return {
# #             "analysis_id":     self.analysis_id,
# #             "train_detail_id": self.train_detail_id,
# #             "processing_time": round(processing_time, 3),
# #             # Single video → keep as dict for backwards compat; batch → list
# #             "video_info": (
# #                 self.video_infos[0]
# #                 if len(self.video_infos) == 1
# #                 else self.video_infos
# #             ),
# #             "violations": [
# #                 {
# #                     "timestamp":   v.time_str,
# #                     "frame_index": v.frame_index,
# #                     "events":      v.events,
# #                     "severity":    v.severity,
# #                     "duration":    v.duration,
# #                     "risk_score":  v.risk_score,
# #                     "risk_level":  v.risk_level,
# #                     "confidence":  v.confidence,
# #                     "factors":     v.factors,
# #                     "original_video_timestamp": (
# #                         f"{v.source_filename} {v.local_time_str}"
# #                     ),
# #                     "frame_path":  v.frame_path,
# #                 }
# #                 for v in self._violations
# #             ],
# #         }


# from __future__ import annotations

# import json
# import os
# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional

# import cv2
# import numpy as np

# OUTPUTS_ROOT = "outputs"
# MERGE_WINDOW = 2.0


# # ══════════════════════════════════════════════════════════════════════════════
# # INTERNAL DATA CLASS
# # ══════════════════════════════════════════════════════════════════════════════

# @dataclass
# class _Violation:
#     timestamp:       float
#     time_str:        str
#     frame_index:     int
#     type:            str
#     events:          List[str]
#     severity:        str
#     duration:        float
#     risk_score:      int
#     risk_level:      str
#     confidence:      float
#     factors:         List[str]
#     source_filename: str                  = ""    # DB filename e.g. "mobile.mp4"
#     local_time_str:  str                  = ""    # local time within that file e.g. "00:00:23"
#     frame_path:      Optional[str]        = None
#     annotated_frame: Optional[np.ndarray] = None


# # ══════════════════════════════════════════════════════════════════════════════
# # VIOLATION STORE
# # ══════════════════════════════════════════════════════════════════════════════

# class ViolationStore:
#     """
#     Accumulates all violations found across one analysis run (single video
#     or a multi-video batch that shares the same analysis_id / folder_name).

#     Batch mode usage (api.py)
#     ─────────────────────────
#     1. Construct ONCE for the whole folder (no video_info in __init__).
#     2. Pass as shared_vstore= to each GadgetDetectionPipeline.
#        The pipeline calls add_video_info() automatically.
#     3. Call finalize() ONCE after all videos in the folder are done.

#     Standalone mode usage (CLI / single video)
#     ──────────────────────────────────────────
#     1. Construct with video_info= for the single video.
#     2. Pipeline calls finalize() automatically at the end of run().
#     """

#     def __init__(
#         self,
#         analysis_id:     str,
#         train_detail_id: int,
#         video_info:      Optional[Dict[str, Any]] = None,
#     ):
#         self.analysis_id     = analysis_id
#         self.train_detail_id = train_detail_id
#         # Always a list — 0 entries until add_video_info() is called (batch mode),
#         # or 1 entry when video_info is provided (standalone mode).
#         self.video_infos: List[Dict[str, Any]] = (
#             [video_info] if video_info is not None else []
#         )

#         self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
#         self.frames_dir = os.path.join(self.output_dir, "frames")
#         os.makedirs(self.frames_dir, exist_ok=True)

#         self._violations:  List[_Violation] = []
#         self._seen_frames: set              = set()
#         print(f"[ViolationStore] Output dir : {self.output_dir}")

#     # ── Public helpers ────────────────────────────────────────────────────────

#     def add_video_info(self, video_info: Dict[str, Any]) -> None:
#         """Append one video's metadata. Called once per video in batch mode."""
#         self.video_infos.append(video_info)

#     def record_violation(
#         self,
#         annotated_frame:  np.ndarray,
#         video_time:       float,          # global timestamp (offset-adjusted)
#         frame_index:      int,            # global frame index (offset-adjusted)
#         event_type:       str,
#         original_frame:   Optional[np.ndarray] = None,
#         severity:         str   = "CRITICAL",
#         confidence:       float = 0.9,
#         risk_score:       int   = 80,
#         risk_level:       str   = "CRITICAL",
#         factors:          Optional[List[str]] = None,
#         duration:         float = 0.0,
#         source_filename:  str   = "",     # DB filename shown in original_video_timestamp
#         local_video_time: float = -1.0,   # local time within the source file; -1 = same as video_time
#     ) -> None:
#         """
#         Record one distraction event.

#         Deduplication key is (frame_index, event_type):
#           • same event on the same global frame is recorded once
#           • different events on the same global frame are each recorded
#           • global frame_index is unique across videos (frame_offset applied in main.py)
#         """
#         dedup_key = (frame_index, event_type)
#         if dedup_key in self._seen_frames:
#             return
#         self._seen_frames.add(dedup_key)

#         factors  = factors or []
#         t        = int(round(video_time))
#         time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

#         local_t   = int(round(local_video_time if local_video_time >= 0 else video_time))
#         local_str = (
#             f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
#         )

#         self._violations.append(
#             _Violation(
#                 timestamp        = video_time,
#                 time_str         = time_str,
#                 frame_index      = frame_index,
#                 type             = event_type,
#                 events           = [event_type],
#                 severity         = severity,
#                 duration         = round(duration, 2),
#                 risk_score       = risk_score,
#                 risk_level       = risk_level,
#                 confidence       = round(confidence, 3),
#                 factors          = list(factors),
#                 source_filename  = source_filename,
#                 local_time_str   = local_str,
#                 annotated_frame  = (
#                     annotated_frame.copy() if annotated_frame is not None else None
#                 ),
#             )
#         )

#     # ── Finalize ──────────────────────────────────────────────────────────────

#     def finalize(self, processing_time: float = 0.0) -> str:
#         """
#         Deduplicate → merge → save frames → write JSON → upload to S3/DB.
#         Returns the local path to analysis_report.json.
#         """
#         self._deduplicate_by_frame()
#         self._merge_by_time_window()

#         # Extract frames from every video in the batch (or the single video)
#         for vi in self.video_infos:
#             if vi and vi.get("videoPath"):
#                 self.extract_violation_frames(vi["videoPath"])

#         report   = self._build_report(processing_time=processing_time)
#         out_path = os.path.join(self.output_dir, "analysis_report.json")
#         with open(out_path, "w", encoding="utf-8") as f:
#             json.dump(report, f, indent=2)

#         print(f"[ViolationStore] JSON report     : {out_path}")
#         print(f"[ViolationStore] Violations      : {len(self._violations)}")
#         print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

#         try:
#             from utils.db_s3_uploader import finalize_and_upload
#             finalize_and_upload(
#                 report_path     = out_path,
#                 analysis_id     = self.analysis_id,
#                 train_detail_id = self.train_detail_id,
#             )
#         except Exception as exc:
#             print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

#         return out_path

#     # ── Private — deduplication & merging ────────────────────────────────────

#     def _deduplicate_by_frame(self) -> None:
#         unique: Dict[int, _Violation] = {}
#         for v in self._violations:
#             if v.frame_index not in unique:
#                 unique[v.frame_index] = v
#             else:
#                 ex = unique[v.frame_index]
#                 ex.events  = list(set(ex.events  + v.events))
#                 ex.factors = list(set(ex.factors + v.factors))
#                 if v.risk_score > ex.risk_score:
#                     ex.risk_score = v.risk_score
#                     ex.risk_level = v.risk_level
#                 if ex.annotated_frame is None and v.annotated_frame is not None:
#                     ex.annotated_frame = v.annotated_frame
#         self._violations = list(unique.values())

#     def _merge_by_time_window(self) -> None:
#         if not self._violations:
#             return
#         self._violations.sort(key=lambda x: x.timestamp)
#         merged: List[_Violation] = []
#         group  = [self._violations[0]]
#         for v in self._violations[1:]:
#             if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
#                 group.append(v)
#             else:
#                 merged.append(self._merge_group(group))
#                 group = [v]
#         merged.append(self._merge_group(group))
#         self._violations = merged

#     def _merge_group(self, group: List[_Violation]) -> _Violation:
#         base             = group[0]
#         events: List[str]  = []
#         factors: List[str] = []
#         max_risk   = base.risk_score
#         risk_level = base.risk_level
#         best_frame = base.annotated_frame
#         for v in group:
#             events.extend(v.events)
#             factors.extend(v.factors)
#             if v.risk_score > max_risk:
#                 max_risk, risk_level = v.risk_score, v.risk_level
#             if best_frame is None and v.annotated_frame is not None:
#                 best_frame = v.annotated_frame
#         return _Violation(
#             timestamp        = base.timestamp,
#             time_str         = base.time_str,
#             frame_index      = base.frame_index,
#             type             = base.type,
#             events           = list(set(events)),
#             severity         = base.severity,
#             duration         = base.duration,
#             risk_score       = max_risk,
#             risk_level       = risk_level,
#             confidence       = base.confidence,
#             factors          = list(set(factors)),
#             source_filename  = base.source_filename,
#             local_time_str   = base.local_time_str,
#             annotated_frame  = best_frame,
#         )

#     # ── Private — frame extraction & saving ──────────────────────────────────

#     def extract_violation_frames(self, video_path: str) -> None:
#         print("[ViolationStore] Saving frames...")
#         need_video = [v for v in self._violations if v.annotated_frame is None]
#         saved = 0

#         # First pass: save violations that already have an annotated frame
#         for v in self._violations:
#             if v.annotated_frame is not None:
#                 v.frame_path      = self._save_frame(v.annotated_frame, v.events, v.time_str)
#                 v.annotated_frame = None   # free memory
#                 saved += 1

#         # Second pass: re-read from the source video for any that are missing
#         if need_video:
#             cap = cv2.VideoCapture(video_path)
#             if not cap.isOpened():
#                 print(f"[ViolationStore] Cannot open video: {video_path}")
#             else:
#                 seen: set = set()
#                 for v in sorted(need_video, key=lambda x: x.frame_index):
#                     if v.frame_index in seen:
#                         continue
#                     seen.add(v.frame_index)
#                     cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
#                     ret, frame = cap.read()
#                     if not ret:
#                         continue
#                     v.frame_path = self._save_frame(frame, v.events, v.time_str)
#                     saved += 1
#                 cap.release()

#         print(f"[ViolationStore] {saved} frames saved")

#     def _save_frame(
#         self,
#         frame:    np.ndarray,
#         events:   List[str],
#         time_str: str,
#     ) -> str:
#         """
#         Save a single violation frame as JPEG.

#         Filename format:  <events>_<HH-MM-SS>.jpg
#         Example:          seat_absence_00-01-14.jpg
#                           seat_absence_drowsy_00-03-02.jpg
#                           phone_use_00-00-24.jpg
#         """
#         distraction   = "_".join(sorted(events))   # sorted for deterministic name
#         filename_time = time_str.replace(":", "-")
#         filename      = f"{distraction}_{filename_time}.jpg"
#         path          = os.path.join(self.frames_dir, filename)
#         ok = cv2.imwrite(
#             path,
#             cv2.resize(frame, (640, 360)),
#             [cv2.IMWRITE_JPEG_QUALITY, 85],
#         )
#         if not ok:
#             print(f"[ViolationStore] imwrite failed: {path}")
#         return os.path.join(self.analysis_id, "frames", filename)

#     # ── Private — report builder ──────────────────────────────────────────────

#     def _build_report(self, processing_time: float = 0.0) -> dict:
#         return {
#             "analysis_id":     self.analysis_id,
#             "train_detail_id": self.train_detail_id,
#             "processing_time": round(processing_time, 3),
#             # Single video → dict (backwards compat); batch → list
#             "video_info": (
#                 self.video_infos[0]
#                 if len(self.video_infos) == 1
#                 else self.video_infos
#             ),
#             "violations": [
#                 {
#                     "timestamp":   v.time_str,
#                     "frame_index": v.frame_index,
#                     "events":      v.events,
#                     "severity":    v.severity,
#                     "duration":    v.duration,
#                     "risk_score":  v.risk_score,
#                     "risk_level":  v.risk_level,
#                     "confidence":  v.confidence,
#                     "factors":     v.factors,
#                     # "original_video_timestamp" = "<db_filename> <local_HH:MM:SS>"
#                     # When timestamp == original (video 1), local_time_str == time_str
#                     "original_video_timestamp": (
#                         f"{v.source_filename} {v.local_time_str}"
#                     ),
#                     "frame_path":  v.frame_path,
#                 }
#                 for v in self._violations
#             ],
#         }


from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

OUTPUTS_ROOT = "outputs"
MERGE_WINDOW = 2.0


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL DATA CLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _Violation:
    timestamp:       float
    time_str:        str
    frame_index:     int
    type:            str
    events:          List[str]
    severity:        str
    duration:        float
    risk_score:      int
    risk_level:      str
    confidence:      float
    factors:         List[str]
    source_filename: str                  = ""    # DB filename e.g. "mobile.mp4"
    local_time_str:  str                  = ""    # local time within that file e.g. "00:00:23"
    frame_path:      Optional[str]        = None
    annotated_frame: Optional[np.ndarray] = None
    # NEW — additive, filled in later (if at all) by close_violation_episode().
    # True trigger→end span: true_end_timestamp - true_start_timestamp ==
    # true_duration. Stay None if the episode was still ongoing when the
    # video ended (never actually "closed"), in which case the existing
    # `duration` field above (threshold-snapshot value) is the only
    # duration info available for this violation — this behaviour is
    # unchanged.
    true_start_timestamp: Optional[float] = None
    true_end_timestamp:   Optional[float] = None
    true_duration:        Optional[float] = None
    # NEW — additive: filled in by the LLM verification step in analyzer.py
    # (llm_verifier.verify_frame()) after dedup/merge.
    #   status: "TRUE" (violation confirmed) | "FALSE" (LLM rejected it).
    #           Defaults to "TRUE" until verification actually runs.
    #   role:   "LP" | "ALP" | "BOTH" | "AMBIGUOUS" when status == "TRUE",
    #           always None when status == "FALSE". None until verified.
    status:                str            = "TRUE"
    role:                  Optional[str]  = None
    # NEW — additive: which crew member (1/2, same convention as the rest
    # of the pipeline) this specific violation instance belongs to, when
    # known. None for every event type that doesn't tag it (unchanged
    # behaviour for all of them). Populated for "hand_raise" so two
    # different pilots signaling in the same cycle/time-window are never
    # collapsed into a single violation — see record_violation(),
    # _deduplicate_by_frame(), _merge_by_time_window() below.
    pilot_id:              Optional[int]  = None
    # Kept for backward compatibility with any external code still reading
    # this attribute; no longer used to filter violations out of the
    # completion payload (rejected candidates are now kept, tagged
    # status="FALSE", instead of being dropped).
    llm_rejected:          bool           = False


# ══════════════════════════════════════════════════════════════════════════════
# VIOLATION STORE
# ══════════════════════════════════════════════════════════════════════════════

class ViolationStore:
    """
    Accumulates all violations found across one analysis run (single video
    or a multi-video batch that shares the same analysis_id / folder_name).

    Batch mode usage (api.py)
    ─────────────────────────
    1. Construct ONCE for the whole folder (no video_info in __init__).
    2. Pass as shared_vstore= to each GadgetDetectionPipeline.
       The pipeline calls add_video_info() automatically.
    3. Call finalize() ONCE after all videos in the folder are done.

    Standalone mode usage (CLI / single video)
    ──────────────────────────────────────────
    1. Construct with video_info= for the single video.
    2. Pipeline calls finalize() automatically at the end of run().
    """

    def __init__(
        self,
        analysis_id:     str,
        train_detail_id: int,
        video_info:      Optional[Dict[str, Any]] = None,
    ):
        self.analysis_id     = analysis_id
        self.train_detail_id = train_detail_id
        # Always a list — 0 entries until add_video_info() is called (batch mode),
        # or 1 entry when video_info is provided (standalone mode).
        self.video_infos: List[Dict[str, Any]] = (
            [video_info] if video_info is not None else []
        )

        self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
        self.frames_dir = os.path.join(self.output_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)

        self._violations:  List[_Violation] = []
        self._seen_frames: set              = set()
        print(f"[ViolationStore] Output dir : {self.output_dir}")

    # ── Public helpers ────────────────────────────────────────────────────────

    def add_video_info(self, video_info: Dict[str, Any]) -> None:
        """Append one video's metadata. Called once per video in batch mode."""
        self.video_infos.append(video_info)

    def record_violation(
        self,
        annotated_frame:  np.ndarray,
        video_time:       float,          # global timestamp (offset-adjusted)
        frame_index:      int,            # global frame index (offset-adjusted)
        event_type:       str,
        original_frame:   Optional[np.ndarray] = None,
        severity:         str   = "CRITICAL",
        confidence:       float = 0.9,
        risk_score:       int   = 80,
        risk_level:       str   = "CRITICAL",
        factors:          Optional[List[str]] = None,
        duration:         float = 0.0,
        source_filename:  str   = "",     # DB filename shown in original_video_timestamp
        local_video_time: float = -1.0,   # local time within the source file; -1 = same as video_time
        pilot_id:         Optional[int] = None,  # NEW — see _Violation.pilot_id
    ) -> None:
        """
        Record one distraction event.

        Deduplication key is (frame_index, event_type, pilot_id):
          • same event on the same global frame is recorded once
          • different events on the same global frame are each recorded
          • global frame_index is unique across videos (frame_offset applied in main.py)
          • NEW — pilot_id is part of the key so two DIFFERENT pilots
            triggering the SAME event_type on the SAME global frame (e.g.
            both LP and ALP raising a hand to acknowledge the same signal
            at once) are recorded as two separate violations instead of
            the second silently deduplicating away. Callers that don't
            pass pilot_id (every event type except hand_raise, unchanged)
            keep the exact old behaviour — pilot_id=None is part of the
            key for them too, so nothing changes for those event types.
        """
        dedup_key = (frame_index, event_type, pilot_id)
        if dedup_key in self._seen_frames:
            return
        self._seen_frames.add(dedup_key)

        factors  = factors or []

        # FIX — timestamp/filename drift (log says 00:00:21, saved frame
        # filename said 00-00-22 for the same event_global value).
        #
        # Root cause: this method used int(round(video_time)) here, while
        # log_distraction() (utils/logger.py) floors the same float with
        # int(video_time). round() rounds 21.6 up to 22; floor keeps it at
        # 21. Both consumers receive the exact same event_global float —
        # they just formatted it differently. Switched to int() (floor)
        # here so this method's HH:MM:SS string always matches the log
        # line and the JSON report for the same timestamp value.
        t        = int(video_time)
        time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"

        local_t   = int(local_video_time if local_video_time >= 0 else video_time)
        local_str = (
            f"{local_t // 3600:02d}:{(local_t % 3600) // 60:02d}:{local_t % 60:02d}"
        )

        self._violations.append(
            _Violation(
                timestamp        = video_time,
                time_str         = time_str,
                frame_index      = frame_index,
                type             = event_type,
                events           = [event_type],
                severity         = severity,
                duration         = round(duration, 2),
                risk_score       = risk_score,
                risk_level       = risk_level,
                confidence       = round(confidence, 3),
                factors          = list(factors),
                source_filename  = source_filename,
                local_time_str   = local_str,
                annotated_frame  = (
                    annotated_frame.copy() if annotated_frame is not None else None
                ),
                pilot_id         = pilot_id,
            )
        )

    # NEW — additive: attach the true trigger→end duration ────────────────────

    def close_violation_episode(
        self,
        source_filename:  str,
        event_type:       str,
        start_video_time: float,
        end_video_time:   float,
        duration:         float,
        pilot_id:         Optional[int] = None,  # NEW — see _Violation.pilot_id
    ) -> None:
        """
        Record that a previously-logged violation episode has genuinely
        ended, and how long it actually lasted from trigger to end (e.g.
        triggered at 10.33, ended at 15.33 → duration 5.0).

        Called from main.py whenever a detector reports a
        completed_events entry (see gadget_detector.py /
        seat_absence_detector.py / head_drop_detector.py). Finds the most
        recently recorded, not-yet-closed violation for this
        (source_filename, event_type[, pilot_id]) and fills in its
        true_start_timestamp / true_end_timestamp / true_duration fields
        ONLY — every other field on that _Violation, and every other
        violation in the store, is left completely untouched. If no
        matching open violation is found (e.g. it was never actually
        confirmed/logged, or the episode was already closed), this is a
        harmless no-op.

        NEW — pilot_id: when provided (currently only by the hand_raise
        completion path in main.py), only a violation with a matching
        pilot_id can be closed by this call. Without this, if both LP and
        ALP had simultaneous open hand_raise episodes on the same source
        file, this would always close whichever one was appended most
        recently — potentially attributing one pilot's true_duration to
        the other. Callers that don't pass pilot_id (every other event
        type, unchanged) keep the exact old behaviour: match on
        (source_filename, event_type) alone.
        """
        for v in reversed(self._violations):
            if (
                v.source_filename == source_filename
                and event_type in v.events
                and v.true_duration is None
                and (pilot_id is None or getattr(v, "pilot_id", None) == pilot_id)
            ):
                v.true_start_timestamp = round(start_video_time, 2)
                v.true_end_timestamp   = round(end_video_time, 2)
                v.true_duration        = round(max(0.0, duration), 2)
                print(f"[CLOSE-EPISODE] matched violation frame_index={v.frame_index} "
                      f"type={event_type!r} source={source_filename!r} "
                      f"pilot_id={pilot_id!r} -> true_duration={v.true_duration}")
                return
        print(f"[CLOSE-EPISODE] NO MATCH for source={source_filename!r} "
              f"event_type={event_type!r} pilot_id={pilot_id!r} (start={start_video_time:.2f} "
              f"end={end_video_time:.2f} dur={duration:.2f}) — no open violation found")

    # ── Finalize ──────────────────────────────────────────────────────────────

    def write_report(self, processing_time: float = 0.0) -> str:
        """
        Build analysis_report.json from the CURRENT in-memory violations and
        write it to disk at  outputs/<analysis_id>/analysis_report.json
        (i.e. as a SIBLING of the frames/ folder, not inside it).

        Unlike finalize(), this does NOT touch S3 or the legacy DB uploader —
        it is safe to call from the journey/batch pipeline (analyzer.py),
        which has its own separate callback-based completion flow.

        Does NOT run dedup/merge/extract_violation_frames — call those first
        (analyze_journey() already does, via the shared ViolationStore) if
        you need them. Safe to call multiple times; it always overwrites.

        Returns the local path to analysis_report.json.
        """
        report   = self._build_report(processing_time=processing_time)
        out_path = os.path.join(self.output_dir, "analysis_report.json")
        os.makedirs(self.output_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"[ViolationStore] JSON report     : {out_path}")
        print(f"[ViolationStore] Violations      : {len(self._violations)}")
        print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

        return out_path

    def finalize(self, processing_time: float = 0.0) -> str:
        """
        Deduplicate → merge → save frames → write JSON → upload to S3/DB.
        Returns the local path to analysis_report.json.

        Standalone/CLI mode only. Journey/batch mode (analyzer.py) should call
        write_report() directly instead, after its own dedup/merge/extract
        steps, since it has its own callback-based upload path and does not
        want the legacy db_s3_uploader to also run.
        """
        self._deduplicate_by_frame()
        self._merge_by_time_window()

        # NEW — additive: RSL Hand Brake post-verification (standalone/CLI
        # mode counterpart of the same step in analyzer.py::analyze_journey).
        # See detector/rsl_hand_brake_verifier.py for the full rationale.
        # Runs only here, after dedup/merge have produced the final
        # representative frame per confirmed hand_raise episode, and never
        # per-frame. Non-fatal — a failure here never blocks the existing
        # finalize() flow below.
        try:
            self._generate_rsl_hand_brake_violations()
        except Exception as exc:
            print(f"[ViolationStore] RSL Hand Brake verification step failed (non-fatal): {exc}")

        # Extract frames from every video in the batch (or the single video)
        for vi in self.video_infos:
            if vi and vi.get("videoPath"):
                self.extract_violation_frames(vi["videoPath"])

        out_path = self.write_report(processing_time=processing_time)

        try:
            from utils.db_s3_uploader import finalize_and_upload
            finalize_and_upload(
                report_path     = out_path,
                analysis_id     = self.analysis_id,
                train_detail_id = self.train_detail_id,
            )
        except Exception as exc:
            print(f"[ViolationStore] S3/DB upload failed (non-fatal): {exc}")

        return out_path

    # NEW — additive: RSL Hand Brake post-verification ────────────────────────

    def _generate_rsl_hand_brake_violations(self) -> None:
        """
        Standalone/CLI-mode counterpart of the RSL Hand Brake step in
        analyzer.py::analyze_journey(). Not a standalone detector and never
        called per-frame -- only from finalize(), after dedup/merge have
        already produced the final representative frame for every confirmed
        "hand_raise" violation.

        For each such violation: reuses the EXACT same frame already
        captured for it in memory (v.annotated_frame -- set at detection
        time in main.py, still populated at this point since this runs
        BEFORE extract_violation_frames() clears it) whenever available,
        so this step and the Hand Raising LLM check upstream are
        guaranteed to be looking at the identical pixels for the identical
        moment -- no second, independent re-seek of the video that could
        land on a slightly different frame. Falls back to re-seeking via
        extract_signal_frame() only if no in-memory frame is available
        (e.g. an older/foreign _Violation that never carried one). Sends
        that single frame to Qwen-VL (detector/rsl_hand_brake_verifier.py)
        -- and if confirmed, clones the exact same frame/timestamp/
        metadata into a new "rsl_hand_brake" violation appended to
        self._violations, with role forced to "ALP". The clone then flows
        through the existing extract_violation_frames() / finalize()
        S3+DB upload path completely unmodified, exactly like every other
        violation.
        """
        import dataclasses as _dc

        # Match on `events` (not just `type`) — a hand_raise violation can
        # end up merged with another violation type that happened within
        # the same MERGE_WINDOW, in which case v.type is whichever event
        # came first but "hand_raise" is still present in v.events.
        hand_raise_violations = [v for v in self._violations if "hand_raise" in v.events]
        if not hand_raise_violations:
            return

        # Map source_filename -> local video path from the video_infos this
        # store already knows about (standalone mode: one entry; batch mode
        # is handled instead by analyzer.py's own copy of this step).
        path_by_filename: Dict[str, str] = {}
        for vi in self.video_infos:
            vp = vi.get("videoPath") if vi else None
            if vp:
                path_by_filename[os.path.basename(vp)] = vp

        from detector.rsl_hand_brake_verifier import extract_signal_frame, verify_rsl_hand_brake

        new_violations: List[_Violation] = []

        for v in hand_raise_violations:
            src_file   = os.path.basename(getattr(v, "source_filename", "") or "")
            local_str  = getattr(v, "local_time_str", "0:00:00")
            try:
                parts      = local_str.strip().split(":")
                local_secs = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            except Exception:
                local_secs = 0.0

            if v.annotated_frame is not None:
                # Preferred path — the exact frame already captured for
                # this violation, no re-seek involved.
                signal_frame = v.annotated_frame
            else:
                video_path = path_by_filename.get(src_file)
                if not video_path or not os.path.isfile(video_path):
                    continue
                cap = cv2.VideoCapture(video_path)
                fps = (cap.get(cv2.CAP_PROP_FPS) or 25.0) if cap.isOpened() else 25.0
                cap.release()
                signal_frame = extract_signal_frame(video_path, local_secs, fps)

            if signal_frame is None:
                print(f"[ViolationStore] RSL verify: could not read the "
                      f"selected hand_raise frame for {src_file} @ {local_secs:.2f}s -- skipping")
                continue

            verdict = verify_rsl_hand_brake(
                signal_frame, log_label=f"rsl_hand_brake:{src_file}@{local_secs:.2f}s",
            )
            print(f"[ViolationStore] RSL verify for hand_raise "
                  f"frame_index={v.frame_index} src={src_file}: {verdict}")

            if not verdict["confirmed"] or verdict["best_frame"] is None:
                continue

            new_violations.append(_dc.replace(
                v,
                type            = "rsl_hand_brake",
                events          = ["rsl_hand_brake"],
                confidence      = round(float(verdict["confidence"]), 3),
                factors         = list(set(list(v.factors) + ["rsl_hand_brake", "opposite_hand_on_brake"])),
                annotated_frame = verdict["best_frame"].copy(),
                frame_path      = None,
                status          = "TRUE",
                role            = verdict["role"],
            ))

        if new_violations:
            self._violations.extend(new_violations)
            print(f"[ViolationStore] RSL Hand Brake: {len(new_violations)} "
                  f"additional violation(s) created from confirmed hand_raise frame(s).")

    # ── Private — deduplication & merging ────────────────────────────────────

    def _deduplicate_by_frame(self) -> None:
        # FIX (Multi-video dedup collision):
        # The old key was frame_index alone. In a multi-video journey, Video 1
        # frame 500 and Video 2 frame 500 share the same frame_index value
        # (frame_offset makes them globally unique across the batch, but after
        # merging by time window the stored frame_index is the base's global
        # index). The safe dedup key must include source_filename so violations
        # from different source files never collide.
        #
        # FIX (both LP and ALP signaling together only ever showed ONE of
        # them): the key must also include pilot_id — two DIFFERENT pilots'
        # violations that happen to share the same (source_filename,
        # frame_index) — e.g. both raising a hand in the same cycle — must
        # stay distinct here too, or this step alone would collapse them
        # right back into one even after record_violation()'s own dedup was
        # fixed to tell them apart. pilot_id is None for every other event
        # type, so this is a no-op change for all of them.
        unique: Dict[tuple, _Violation] = {}
        for v in self._violations:
            key = (v.source_filename, v.frame_index, getattr(v, "pilot_id", None))
            if key not in unique:
                unique[key] = v
            else:
                ex = unique[key]
                ex.events  = list(set(ex.events  + v.events))
                ex.factors = list(set(ex.factors + v.factors))
                if v.risk_score > ex.risk_score:
                    ex.risk_score = v.risk_score
                    ex.risk_level = v.risk_level
                if ex.annotated_frame is None and v.annotated_frame is not None:
                    ex.annotated_frame = v.annotated_frame
        self._violations = list(unique.values())

    def _merge_by_time_window(self) -> None:
        if not self._violations:
            return
        self._violations.sort(key=lambda x: x.timestamp)
        merged: List[_Violation] = []
        group  = [self._violations[0]]
        for v in self._violations[1:]:
            # FIX (both LP and ALP signaling together only ever showed ONE
            # of them): merging purely by elapsed time collapsed two
            # DIFFERENT pilots' simultaneous violations (e.g. both
            # acknowledging the same signal within MERGE_WINDOW seconds of
            # each other) into a single _Violation, and _merge_group() below
            # only ever kept ONE role — so the second pilot's event vanished
            # from the report entirely. A violation only extends the current
            # group now if it is untagged (pilot_id is None, i.e. every
            # event type except hand_raise — unchanged behaviour) or its
            # pilot_id matches every non-None pilot_id already in the group;
            # otherwise it starts a new group so each pilot's own instance
            # of the event survives as its own violation.
            group_pilot_ids = {g.pilot_id for g in group if g.pilot_id is not None}
            same_pilot_or_untagged = (
                v.pilot_id is None or not group_pilot_ids or v.pilot_id in group_pilot_ids
            )
            if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW and same_pilot_or_untagged:
                group.append(v)
            else:
                merged.append(self._merge_group(group))
                group = [v]
        merged.append(self._merge_group(group))
        self._violations = merged

    def _merge_group(self, group: List[_Violation]) -> _Violation:
        base             = group[0]
        events: List[str]  = []
        factors: List[str] = []
        max_risk   = base.risk_score
        risk_level = base.risk_level
        best_frame = base.annotated_frame
        for v in group:
            events.extend(v.events)
            factors.extend(v.factors)
            if v.risk_score > max_risk:
                max_risk, risk_level = v.risk_score, v.risk_level
            if best_frame is None and v.annotated_frame is not None:
                best_frame = v.annotated_frame

        # FIX — true_start_timestamp / true_end_timestamp / true_duration
        # (and frame_path) used to be dropped here because they weren't
        # passed into the rebuilt _Violation, silently resetting to the
        # dataclass default (None) on every merge. close_violation_episode()
        # runs during frame processing, BEFORE finalize() calls this, so
        # any true_duration it had already filled in was being wiped out
        # right before the report was built.
        before_durations = [v.true_duration for v in group]

        merged = _Violation(
            timestamp        = base.timestamp,
            time_str         = base.time_str,
            frame_index      = base.frame_index,
            type             = base.type,
            events           = list(set(events)),
            severity         = base.severity,
            duration         = base.duration,
            risk_score       = max_risk,
            risk_level       = risk_level,
            confidence       = base.confidence,
            factors          = list(set(factors)),
            source_filename  = base.source_filename,
            local_time_str   = base.local_time_str,
            annotated_frame  = best_frame,
            frame_path            = base.frame_path,
            true_start_timestamp  = base.true_start_timestamp,
            true_end_timestamp    = base.true_end_timestamp,
            true_duration         = base.true_duration,
            role                  = base.role,
            llm_rejected          = base.llm_rejected,
            pilot_id              = base.pilot_id,
        )
        print(f"[MERGE] frame_index={base.frame_index} true_durations "
              f"before={before_durations} -> after={merged.true_duration}")
        return merged

    # ── Private — frame extraction & saving ──────────────────────────────────

    def extract_violation_frames(self, video_path: str) -> None:
        """
        Extract and save one frame image per violation.

        FIX (Wrong frames in multi-video journeys):
        ────────────────────────────────────────────
        The old implementation had two bugs when called in a loop over multiple
        video files (as analyzer.py does for batch journeys):

        BUG A — Global frame_index used to seek into per-video files.
          v.frame_index is a GLOBAL frame number that accumulates across all
          videos in a journey (set by frame_offset in main.py).  Seeking to
          frame_index 4500 in video_2 lands on a completely unrelated frame if
          video_1 contained frames 0-5000.  The resulting evidence image is
          from the wrong video entirely.

        BUG B — First pass re-ran for every video in the loop.
          Violations that already had annotated_frame were saved to disk on the
          first call (video_1), then overwritten on the second call (video_2)
          because the first-pass loop had no guard for frame_path already set.

        FIX:
          1. Filter by source_filename so this call only processes violations
             that belong to the video file at video_path.
          2. Seek using local_time_str (seconds within this specific file)
             via CAP_PROP_POS_MSEC — always correct regardless of how many
             videos precede this one in the journey.
          3. Guard the first pass with `v.frame_path is None` so already-saved
             violations are never re-processed on subsequent calls.
        """
        import os as _os
        src_filename = _os.path.basename(video_path)
        print(f"[ViolationStore] Saving frames for {src_filename!r}...")

        # Violations that belong to this source file and haven't been saved yet
        mine = [
            v for v in self._violations
            if _os.path.basename(getattr(v, "source_filename", "")) == src_filename
               and v.frame_path is None
        ]

        saved = 0

        # Pass 1: violations with an annotated frame already in memory
        need_video = []
        for v in mine:
            if v.annotated_frame is not None:
                v.frame_path      = self._save_frame(v.annotated_frame, v.events, v.time_str, v.frame_index)
                v.annotated_frame = None   # free memory
                saved += 1
            else:
                need_video.append(v)

        # Pass 2: re-read from the source video using LOCAL time (not global frame_index)
        if need_video:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[ViolationStore] Cannot open video: {video_path}")
            else:
                fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Sort by local time for efficient sequential seeking
                for v in sorted(need_video, key=lambda x: getattr(x, "local_time_str", "0:00:00")):
                    local_str = getattr(v, "local_time_str", "0:00:00")
                    # Parse "HH:MM:SS" → seconds
                    try:
                        parts = local_str.strip().split(":")
                        local_secs = int(parts[0])*3600 + int(parts[1])*60 + float(parts[2])
                    except Exception:
                        local_secs = 0.0

                    # Seek by milliseconds (more reliable than frame index)
                    cap.set(cv2.CAP_PROP_POS_MSEC, local_secs * 1000.0)
                    ret, frame = cap.read()
                    if not ret:
                        # Fallback: seek by local frame number
                        local_frame = min(int(local_secs * fps), max(0, total - 1))
                        cap.set(cv2.CAP_PROP_POS_FRAMES, local_frame)
                        ret, frame = cap.read()
                    if not ret:
                        print(f"[ViolationStore] Seek failed for {src_filename} "
                              f"@ local_time={local_str}")
                        continue
                    v.frame_path = self._save_frame(frame, v.events, v.time_str, v.frame_index)
                    saved += 1
                cap.release()

        print(f"[ViolationStore] {saved} frames saved for {src_filename!r}")

    def _save_frame(
        self,
        frame:       np.ndarray,
        events:      List[str],
        time_str:    str,
        frame_index: int,
    ) -> str:
        """
        Save a single violation frame as JPEG.

        Filename format:  <events>_<HH-MM-SS>_<frame_index>.jpg
        Example:          seat_absence_00-01-14_1042.jpg
                          seat_absence_drowsy_00-03-02_5310.jpg
                          phone_use_00-00-24_612.jpg

        FIX — Wrong/stale frame evidence (filename collision):
        ────────────────────────────────────────────────────────
        The filename used to be built from event-type + time_str ALONE
        (floored to the whole second). Any two DIFFERENT violations that
        share the same event type and land in the same whole second of
        global journey time — easily possible for a multi-frame event,
        for videos that meet near a journey boundary, or simply on a
        rerun/reprocess of the same journey (same S3 folder reused) —
        produced the IDENTICAL filename/S3 key. Uploads overwrite
        unconditionally, so whichever violation was saved last silently
        replaced the other's evidence image, and a rerun could leave a
        violation pointing at a stale frame from a previous run.

        frame_index is the value already used as (part of) the
        record_violation() dedup key, so it is guaranteed unique per
        violation within a journey — including it here makes the
        filename/S3 key collision-proof.
        """
        distraction   = "_".join(sorted(events))   # sorted for deterministic name
        filename_time = time_str.replace(":", "-")
        filename      = f"{distraction}_{filename_time}_{frame_index}.jpg"
        path          = os.path.join(self.frames_dir, filename)
        ok = cv2.imwrite(
            path,
            cv2.resize(frame, (640, 360)),
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )
        if not ok:
            print(f"[ViolationStore] imwrite failed: {path}")
        return os.path.join(self.analysis_id, "frames", filename)

    # ── Private — report builder ──────────────────────────────────────────────

    def _build_report(self, processing_time: float = 0.0) -> dict:
        for v in self._violations:
            print(f"[REPORT] frame_index={v.frame_index} type={v.type} "
                  f"events={v.events} true_duration={v.true_duration}")
        return {
            "analysis_id":     self.analysis_id,
            "train_detail_id": self.train_detail_id,
            "processing_time": round(processing_time, 3),
            # Single video → dict (backwards compat); batch → list
            "video_info": (
                self.video_infos[0]
                if len(self.video_infos) == 1
                else self.video_infos
            ),
            "violations": [
                {
                    "timestamp":   v.time_str,
                    "frame_index": v.frame_index,
                    "events":      v.events,
                    "severity":    v.severity,
                    "duration":    v.duration,
                    # NEW — additive: true trigger→end duration (e.g.
                    # triggered at 10.33, ended at 15.33 → 5.0). None if
                    # the violation was still ongoing when the video
                    # ended (never actually closed) — "duration" above
                    # remains the only figure available in that case,
                    # unchanged from before.
                    "trigger_duration_seconds": v.true_duration,
                    "risk_score":  v.risk_score,
                    "risk_level":  v.risk_level,
                    "confidence":  v.confidence,
                    "factors":     v.factors,
                    # "original_video_timestamp" = "<db_filename> <local_HH:MM:SS>"
                    # When timestamp == original (video 1), local_time_str == time_str
                    "original_video_timestamp": (
                        f"{v.source_filename} {v.local_time_str}"
                    ),
                    "frame_path":  v.frame_path,
                }
                for v in self._violations
            ],
        }