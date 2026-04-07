from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

OUTPUTS_ROOT = "outputs"
MERGE_WINDOW = 2.0


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
    frame_path:      Optional[str]         = None
    annotated_frame: Optional[np.ndarray] = None


class ViolationStore:

    def __init__(self, analysis_id: str, train_detail_id: int, video_info: Dict[str, Any]):
        self.analysis_id     = analysis_id
        self.train_detail_id = train_detail_id
        self.video_info      = video_info

        self.output_dir = os.path.join(OUTPUTS_ROOT, analysis_id)
        self.frames_dir = os.path.join(self.output_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)

        self._violations: List[_Violation] = []
        self._seen_frames: set             = set()
        print(f"[ViolationStore] Output dir : {self.output_dir}")

    def record_violation(
        self,
        annotated_frame: np.ndarray,
        video_time:      float,
        frame_index:     int,
        event_type:      str,
        original_frame:  Optional[np.ndarray] = None,
        severity:        str   = "CRITICAL",
        confidence:      float = 0.9,
        risk_score:      int   = 80,
        risk_level:      str   = "CRITICAL",
        factors:         Optional[List[str]] = None,
        duration:        float = 0.0,
    ):
        if frame_index in self._seen_frames:
            return
        self._seen_frames.add(frame_index)
        factors  = factors or []
        t        = int(round(video_time))
        time_str = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"
        self._violations.append(_Violation(
            timestamp       = video_time,
            time_str        = time_str,
            frame_index     = frame_index,
            type            = event_type,
            events          = [event_type],
            severity        = severity,
            duration        = round(duration, 2),
            risk_score      = risk_score,
            risk_level      = risk_level,
            confidence      = round(confidence, 3),
            factors         = list(factors),
            annotated_frame = annotated_frame.copy() if annotated_frame is not None else None,
        ))

    def _deduplicate_by_frame(self):
        unique: Dict[int, _Violation] = {}
        for v in self._violations:
            if v.frame_index not in unique:
                unique[v.frame_index] = v
            else:
                ex = unique[v.frame_index]
                ex.events  = list(set(ex.events  + v.events))
                ex.factors = list(set(ex.factors + v.factors))
                if v.risk_score > ex.risk_score:
                    ex.risk_score = v.risk_score
                    ex.risk_level = v.risk_level
                if ex.annotated_frame is None and v.annotated_frame is not None:
                    ex.annotated_frame = v.annotated_frame
        self._violations = list(unique.values())

    def _merge_by_time_window(self):
        if not self._violations:
            return
        self._violations.sort(key=lambda x: x.timestamp)
        merged = []
        group  = [self._violations[0]]
        for v in self._violations[1:]:
            if abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW:
                group.append(v)
            else:
                merged.append(self._merge_group(group))
                group = [v]
        merged.append(self._merge_group(group))
        self._violations = merged

    def _merge_group(self, group: List[_Violation]) -> _Violation:
        base            = group[0]
        events, factors = [], []
        max_risk        = base.risk_score
        risk_level      = base.risk_level
        best_frame      = base.annotated_frame
        for v in group:
            events.extend(v.events)
            factors.extend(v.factors)
            if v.risk_score > max_risk:
                max_risk, risk_level = v.risk_score, v.risk_level
            if best_frame is None and v.annotated_frame is not None:
                best_frame = v.annotated_frame
        return _Violation(
            timestamp       = base.timestamp,
            time_str        = base.time_str,
            frame_index     = base.frame_index,
            type            = base.type,
            events          = list(set(events)),
            severity        = base.severity,
            duration        = base.duration,
            risk_score      = max_risk,
            risk_level      = risk_level,
            confidence      = base.confidence,
            factors         = list(set(factors)),
            annotated_frame = best_frame,
        )

    def extract_violation_frames(self, video_path: str):
        print("[ViolationStore] Saving frames...")
        need_video = [v for v in self._violations if v.annotated_frame is None]
        saved = 0
        for v in self._violations:
            if v.annotated_frame is not None:
                v.frame_path      = self._save_frame(v.annotated_frame, v.events, v.time_str)
                v.annotated_frame = None
                saved += 1
        if need_video:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[ViolationStore] Cannot open video: {video_path}")
            else:
                seen: set = set()
                for v in sorted(need_video, key=lambda x: x.frame_index):
                    if v.frame_index in seen:
                        continue
                    seen.add(v.frame_index)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, v.frame_index)
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    v.frame_path = self._save_frame(frame, v.events, v.time_str)
                    saved += 1
                cap.release()
        print(f"[ViolationStore] {saved} frames saved")

    def _save_frame(self, frame: np.ndarray, events: List[str], time_str: str) -> str:
        distraction   = "_".join(events)
        filename_time = time_str.replace(":", "-")
        filename      = f"{distraction}_{filename_time}.jpg"
        path          = os.path.join(self.frames_dir, filename)
        ok = cv2.imwrite(path, cv2.resize(frame, (640, 360)),
                         [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            print(f"[ViolationStore] imwrite failed: {path}")
        return os.path.join(self.analysis_id, "frames", filename)

    def _build_report(self, processing_time: float = 0.0) -> dict:
        return {
            "analysis_id":     self.analysis_id,
            "train_detail_id": self.train_detail_id,
            "processing_time": round(processing_time, 3),
            "video_info":      self.video_info,
            "violations": [
                {
                    "timestamp":   v.time_str,
                    "frame_index": v.frame_index,
                    "events":      v.events,
                    "severity":    v.severity,
                    "duration":    v.duration,
                    "risk_score":  v.risk_score,
                    "risk_level":  v.risk_level,
                    "confidence":  v.confidence,
                    "factors":     v.factors,
                    "frame_path":  v.frame_path,
                }
                for v in self._violations
            ],
        }

    def finalize(self, processing_time: float = 0.0) -> str:
        self._deduplicate_by_frame()
        self._merge_by_time_window()
        self.extract_violation_frames(self.video_info["videoPath"])
        report   = self._build_report(processing_time=processing_time)
        out_path = os.path.join(self.output_dir, "analysis_report.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[ViolationStore] JSON report     : {out_path}")
        print(f"[ViolationStore] Violations      : {len(self._violations)}")
        print(f"[ViolationStore] Processing time : {processing_time:.3f}s")

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