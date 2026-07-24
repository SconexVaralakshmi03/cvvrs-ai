# # # # """
# # # # models.py
# # # # ─────────
# # # # Dataclasses that represent the RabbitMQ job message and the response
# # # # structures expected by the Spring Boot completion / progress APIs.

# # # # These are pure data holders — no I/O, no ML logic.
# # # # """

# # # # from __future__ import annotations

# # # # from dataclasses import dataclass, field
# # # # from typing import List, Optional


# # # # # ── Inbound (from RabbitMQ) ──────────────────────────────────────────────────

# # # # @dataclass
# # # # class VideoJob:
# # # #     """One video entry inside an AnalysisJobMessage."""
# # # #     video_id:    int
# # # #     sequence_no: int
# # # #     s3_key:      str     # e.g. "journeys/101/original/video1.mp4"

# # # #     @classmethod
# # # #     def from_dict(cls, d: dict) -> "VideoJob":
# # # #         return cls(
# # # #             video_id    = int(d["videoId"]),
# # # #             sequence_no = int(d["sequenceNo"]),
# # # #             s3_key      = d["s3Key"],
# # # #         )


# # # # @dataclass
# # # # class AnalysisJobMessage:
# # # #     """
# # # #     RabbitMQ message consumed from the 'analysis.jobs' queue.

# # # #     Example JSON
# # # #     ────────────
# # # #     {
# # # #         "jobId":     "JOB123",
# # # #         "journeyId": 101,
# # # #         "videos": [
# # # #             {"videoId": 1001, "sequenceNo": 1, "s3Key": "journeys/101/original/video1.mp4"},
# # # #             {"videoId": 1002, "sequenceNo": 2, "s3Key": "journeys/101/original/video2.mp4"}
# # # #         ]
# # # #     }
# # # #     """
# # # #     job_id:     str
# # # #     journey_id: int
# # # #     videos:     List[VideoJob]

# # # #     @classmethod
# # # #     def from_dict(cls, d: dict) -> "AnalysisJobMessage":
# # # #         return cls(
# # # #             job_id     = d["jobId"],
# # # #             journey_id = int(d["journeyId"]),
# # # #             videos     = [VideoJob.from_dict(v) for v in d.get("videos", [])],
# # # #         )


# # # # # ── Outbound (to Spring Boot) ────────────────────────────────────────────────

# # # # @dataclass
# # # # class ViolationResult:
# # # #     """
# # # #     One violation event to be persisted as a ViolationEvent + AnalysisFrame
# # # #     by Spring Boot.

# # # #     Fields
# # # #     ──────
# # # #     violation_type  : canonical type string, e.g. "PHONE_USAGE", "SEAT_ABSENCE",
# # # #                       "DROWSINESS"
# # # #     severity        : "HIGH" | "MEDIUM" | "LOW"
# # # #     confidence      : 0.0 – 100.0  (percentage)
# # # #     risk_score      : 0 – 100
# # # #     timestamp       : HH:MM:SS display string  (global journey time)
# # # #     timestamp_seconds: int  — seconds from start of journey
# # # #     original_video_timestamp : "<filename> HH:MM:SS"  — local file + local time
# # # #     frame_paths     : list of S3 keys (NOT signed URLs)
# # # #     """
# # # #     violation_type:            str
# # # #     severity:                  str
# # # #     confidence:                float
# # # #     risk_score:                int
# # # #     timestamp:                 str
# # # #     timestamp_seconds:         int
# # # #     original_video_timestamp:  str
# # # #     frame_paths:               List[str] = field(default_factory=list)

# # # #     def to_dict(self) -> dict:
# # # #         return {
# # # #             "violationType":           self.violation_type,
# # # #             "severity":                self.severity,
# # # #             "confidence":              round(self.confidence, 2),
# # # #             "riskScore":               self.risk_score,
# # # #             "timestamp":               self.timestamp,
# # # #             "timestampSeconds":        self.timestamp_seconds,
# # # #             "original_video_timestamp": self.original_video_timestamp,
# # # #             "framePaths":              self.frame_paths,
# # # #         }


# # # # @dataclass
# # # # class VideoResult:
# # # #     """
# # # #     Per-video summary sent inside the completion payload.

# # # #     Fields
# # # #     ──────
# # # #     video_id          : mirrors VideoJob.video_id
# # # #     video_name        : display filename
# # # #     sequence_no       : mirrors VideoJob.sequence_no
# # # #     duration_seconds  : float — duration of this video file
# # # #     duration_formatted: "H:MM:SS"
# # # #     fps               : frames per second
# # # #     size_mb           : file size in MB
# # # #     violations        : list of ViolationResult
# # # #     """
# # # #     video_id:           int
# # # #     video_name:         str
# # # #     sequence_no:        int
# # # #     duration_seconds:   float
# # # #     duration_formatted: str
# # # #     fps:                float
# # # #     size_mb:            float
# # # #     violations:         List[ViolationResult] = field(default_factory=list)

# # # #     def to_dict(self) -> dict:
# # # #         return {
# # # #             "videoId":           self.video_id,
# # # #             "video_name":        self.video_name,
# # # #             "sequenceNo":        self.sequence_no,
# # # #             "durationSeconds":   round(self.duration_seconds, 3),
# # # #             "durationFormatted": self.duration_formatted,
# # # #             "fps":               round(self.fps, 3),
# # # #             "sizeMb":            round(self.size_mb, 2),
# # # #             "violations":        [v.to_dict() for v in self.violations],
# # # #         }


# # # # @dataclass
# # # # class CompletionPayload:
# # # #     """
# # # #     Full payload for POST /api/internal/analysis/completed.
# # # #     """
# # # #     job_id:          str
# # # #     journey_id:      int
# # # #     processing_time: int           # wall-clock milliseconds
# # # #     video_results:   List[VideoResult] = field(default_factory=list)

# # # #     def to_dict(self) -> dict:
# # # #         return {
# # # #             "jobId":          self.job_id,
# # # #             "journeyId":      self.journey_id,
# # # #             "processingTime": self.processing_time,
# # # #             "videoResults":   [vr.to_dict() for vr in self.video_results],
# # # #         }

# # # """
# # # models.py
# # # ─────────
# # # Dataclasses that represent the RabbitMQ job message and the response
# # # structures expected by the Spring Boot completion / progress APIs.

# # # Aligned with CVVRS API Documentation (Base: https://cvvrsrailway-api.sconexsoft.com/cvs)

# # # Changes from previous version
# # # ──────────────────────────────
# # # • VideoJob         — added originalFileName field (present in RabbitMQ message).
# # # • AnalysisJobMessage — added trainDetailId, folderName, priority fields.
# # # • ViolationResult  — timestamp / originalVideoTimestamp are now float (seconds),
# # #                      matching the API schema. The HH:MM:SS display string is dropped
# # #                      from the outbound payload (Spring Boot derives it from the float).
# # # • VideoResult      — added originalS3Key (required by /internal/analysis/completed).
# # #                      videoId is serialised as a STRING per the API spec.
# # # • CompletionPayload — added batchId, trainDetailId, folderName fields required by
# # #                       the /internal/analysis/completed endpoint.
# # # """

# # # from __future__ import annotations

# # # from dataclasses import dataclass, field
# # # from typing import List, Optional


# # # # ── Inbound (from RabbitMQ) ──────────────────────────────────────────────────

# # # @dataclass
# # # class VideoJob:
# # #     """One video entry inside an AnalysisJobMessage."""
# # #     video_id:          int
# # #     sequence_no:       int
# # #     s3_key:            str            # e.g. "journeys/1/2026-06-10/.../original/001_front_cabin.mp4"
# # #     original_filename: str = ""       # e.g. "front_cabin.mp4"  (added per API spec)

# # #     @classmethod
# # #     def from_dict(cls, d: dict) -> "VideoJob":
# # #         return cls(
# # #             video_id          = int(d["videoId"]),
# # #             sequence_no       = int(d["sequenceNo"]),
# # #             s3_key            = d["s3Key"],
# # #             original_filename = d.get("originalFileName", ""),
# # #         )


# # # @dataclass
# # # class AnalysisJobMessage:
# # #     """
# # #     RabbitMQ message consumed from the 'analysis.jobs' queue.

# # #     Queue    : analysis.jobs
# # #     Exchange : dev.analysis.exchange
# # #     Routing  : dev.analysis.jobs.created

# # #     Example JSON
# # #     ────────────
# # #     {
# # #         "jobId":         "JOB-ABC123XYZ",
# # #         "journeyId":     10,
# # #         "trainDetailId": 1,
# # #         "folderName":    "journeys/1/2026-06-10/JRN-20260610-1-ABC123",
# # #         "priority":      "NORMAL",
# # #         "videos": [
# # #             {
# # #                 "videoId":           1,
# # #                 "sequenceNo":        1,
# # #                 "originalFileName":  "front_cabin.mp4",
# # #                 "s3Key":             "journeys/1/2026-06-10/.../original/001_front_cabin.mp4"
# # #             }
# # #         ]
# # #     }
# # #     """
# # #     job_id:          str
# # #     journey_id:      int
# # #     train_detail_id: int              # NEW — needed in completion payload
# # #     folder_name:     str              # NEW — S3 folder prefix for this journey
# # #     videos:          List[VideoJob]
# # #     priority:        str = "NORMAL"   # NEW — NORMAL | HIGH

# # #     @classmethod
# # #     def from_dict(cls, d: dict) -> "AnalysisJobMessage":
# # #         return cls(
# # #             job_id          = d["jobId"],
# # #             journey_id      = int(d["journeyId"]),
# # #             train_detail_id = int(d.get("trainDetailId", 0)),
# # #             folder_name     = d.get("folderName", f"journeys/{d.get('journeyId', 0)}"),
# # #             videos          = [VideoJob.from_dict(v) for v in d.get("videos", [])],
# # #             priority        = d.get("priority", "NORMAL"),
# # #         )


# # # # ── Outbound (to Spring Boot) ────────────────────────────────────────────────

# # # @dataclass
# # # class ViolationResult:
# # #     """
# # #     One violation event.

# # #     API field mapping (POST /api/internal/analysis/completed)
# # #     ──────────────────────────────────────────────────────────
# # #     violationType         → canonical string: PHONE_USAGE | DROWSY | SEAT_ABSENCE …
# # #     severity              → CRITICAL | HIGH | MEDIUM | LOW
# # #     confidence            → 0.0 – 100.0 (percentage)
# # #     riskScore             → 0.0 – 100.0
# # #     timestamp             → float seconds from start of journey   ← was HH:MM:SS string
# # #     originalVideoTimestamp→ float seconds within the source video ← was "<file> HH:MM:SS"
# # #     framePaths            → list of S3 keys (NOT signed URLs)
# # #     """
# # #     violation_type:             str
# # #     severity:                   str
# # #     confidence:                 float
# # #     risk_score:                 float          # float per API spec (was int)
# # #     timestamp_seconds:          float          # journey-global seconds (float)
# # #     original_video_timestamp:   float          # local-video seconds (float)
# # #     frame_paths:                List[str] = field(default_factory=list)

# # #     def to_dict(self) -> dict:
# # #         return {
# # #             "violationType":          self.violation_type,
# # #             "severity":               self.severity,
# # #             "confidence":             round(self.confidence, 2),
# # #             "riskScore":              round(self.risk_score, 2),
# # #             "timestamp":              round(self.timestamp_seconds, 3),
# # #             "originalVideoTimestamp": round(self.original_video_timestamp, 3),
# # #             "framePaths":             self.frame_paths,
# # #         }


# # # @dataclass
# # # class VideoResult:
# # #     """
# # #     Per-video summary inside the completion payload.

# # #     API field mapping (POST /api/internal/analysis/completed)
# # #     ──────────────────────────────────────────────────────────
# # #     videoId        → STRING per API spec  (was int in previous version)
# # #     sequenceNo     → int
# # #     durationSeconds→ float
# # #     originalS3Key  → NEW — the s3Key from the inbound VideoJob
# # #     violations     → list of ViolationResult
# # #     """
# # #     video_id:         int
# # #     video_name:       str
# # #     sequence_no:      int
# # #     duration_seconds: float
# # #     original_s3_key:  str             # NEW — required by API
# # #     violations:       List[ViolationResult] = field(default_factory=list)

# # #     def to_dict(self) -> dict:
# # #         return {
# # #             "videoId":         str(self.video_id),   # API spec says STRING
# # #             "sequenceNo":      self.sequence_no,
# # #             "durationSeconds": round(self.duration_seconds, 3),
# # #             "originalS3Key":   self.original_s3_key,
# # #             "violations":      [v.to_dict() for v in self.violations],
# # #         }


# # # @dataclass
# # # class CompletionPayload:
# # #     """
# # #     Full payload for POST /api/internal/analysis/completed.

# # #     New required fields vs previous version
# # #     ────────────────────────────────────────
# # #     • batchId        — generated as "BATCH-<jobId>" if not supplied
# # #     • trainDetailId  — forwarded from the RabbitMQ message
# # #     • folderName     — forwarded from the RabbitMQ message
# # #     """
# # #     job_id:          str
# # #     journey_id:      int
# # #     train_detail_id: int
# # #     folder_name:     str
# # #     processing_time: int                         # wall-clock milliseconds
# # #     video_results:   List[VideoResult] = field(default_factory=list)
# # #     batch_id:        str = ""                    # auto-filled in to_dict if blank

# # #     def to_dict(self) -> dict:
# # #         return {
# # #             "jobId":          self.job_id,
# # #             "journeyId":      self.journey_id,
# # #             "batchId":        self.batch_id or f"BATCH-{self.job_id}",
# # #             "trainDetailId":  self.train_detail_id,
# # #             "folderName":     self.folder_name,
# # #             "processingTime": self.processing_time,
# # #             "videoResults":   [vr.to_dict() for vr in self.video_results],
# # #         }



# # """

# # models.py

# # ─────────

# # Dataclasses that represent the RabbitMQ job message and the response

# # structures expected by the Spring Boot completion / progress APIs.
 
# # Aligned with CVVRS API (Base: https://cvvrsrailway-api.sconexsoft.com/cvs)
 
# # Fixes from previous version

# # ─────────────────────────────

# # • AnalysisJobMessage — added callbackBaseUrl field (sent by Spring Boot,

# #   used by consumer.py to route callbacks to the correct server).

# # • VideoResult — restored durationFormatted, fps, sizeMb fields that were

# #   accidentally removed. Spring Boot expects all three in the completion payload.

# # • VideoResult.to_dict() — serializes all fields including the restored ones.

# # """
 
# # from __future__ import annotations
 
# # from dataclasses import dataclass, field

# # from typing import List, Optional
 
 
# # # ── Inbound (from RabbitMQ) ──────────────────────────────────────────────────
 
# # @dataclass

# # class VideoJob:

# #     """One video entry inside an AnalysisJobMessage."""

# #     video_id:          int

# #     sequence_no:       int

# #     s3_key:            str            # e.g. "journeys/1/2026-06-10/.../original/001_front_cabin.mp4"

# #     original_filename: str = ""       # e.g. "front_cabin.mp4"
 
# #     @classmethod

# #     def from_dict(cls, d: dict) -> "VideoJob":

# #         return cls(

# #             video_id          = int(d["videoId"]),

# #             sequence_no       = int(d["sequenceNo"]),

# #             s3_key            = d["s3Key"],

# #             original_filename = d.get("originalFileName", ""),

# #         )
 
 
# # @dataclass

# # class AnalysisJobMessage:

# #     """

# #     RabbitMQ message consumed from the 'analysis.jobs' queue.
 
# #     Queue    : analysis.jobs

# #     Exchange : dev.analysis.exchange

# #     Routing  : dev.analysis.jobs.created
 
# #     Example JSON

# #     ────────────

# #     {

# #         "jobId":           "JOB-ABC123XYZ",

# #         "journeyId":       10,

# #         "trainDetailId":   1,

# #         "folderName":      "journeys/1/2026-06-10/JRN-20260610-1-ABC123",

# #         "callbackBaseUrl": "https://cvvrsrailway-api.sconexsoft.com/cvs/api/internal/analysis",

# #         "priority":        "NORMAL",

# #         "videos": [

# #             {

# #                 "videoId":           1,

# #                 "sequenceNo":        1,

# #                 "originalFileName":  "front_cabin.mp4",

# #                 "s3Key":             "journeys/1/2026-06-10/.../original/001_front_cabin.mp4"

# #             }

# #         ]

# #     }

# #     """

# #     job_id:            str

# #     journey_id:        int

# #     train_detail_id:   int

# #     folder_name:       str

# #     videos:            List[VideoJob]

# #     callback_base_url: str = ""       # FIX: was missing — Spring Boot sends this

# #     priority:          str = "NORMAL"
 
# #     @classmethod

# #     def from_dict(cls, d: dict) -> "AnalysisJobMessage":

# #         return cls(

# #             job_id            = d["jobId"],

# #             journey_id        = int(d["journeyId"]),

# #             train_detail_id   = int(d.get("trainDetailId", 0)),

# #             folder_name       = d.get("folderName", f"journeys/{d.get('journeyId', 0)}"),

# #             callback_base_url = d.get("callbackBaseUrl", ""),   # FIX

# #             videos            = [VideoJob.from_dict(v) for v in d.get("videos", [])],

# #             priority          = d.get("priority", "NORMAL"),

# #         )
 
 
# # # ── Outbound (to Spring Boot) ────────────────────────────────────────────────
 
# # @dataclass

# # class ViolationResult:

# #     """

# #     One violation event.
 
# #     API field mapping (POST /api/internal/analysis/completed)

# #     ──────────────────────────────────────────────────────────

# #     violationType          → canonical string: PHONE_USAGE | DROWSY | SEAT_ABSENCE ...

# #     severity               → CRITICAL | HIGH | MEDIUM | LOW

# #     confidence             → 0.0 – 100.0 (percentage)

# #     riskScore              → 0.0 – 100.0

# #     timestamp              → float seconds from start of journey (global)

# #     originalVideoTimestamp → float seconds within the source video (local)

# #     framePaths             → list of S3 keys (NOT signed URLs)

# #     """

# #     violation_type:             str

# #     severity:                   str

# #     confidence:                 float

# #     risk_score:                 float

# #     timestamp_seconds:          float          # journey-global seconds

# #     original_video_timestamp:   float          # local-video seconds

# #     frame_paths:                List[str] = field(default_factory=list)
 
# #     def to_dict(self) -> dict:

# #         return {

# #             "violationType":          self.violation_type,

# #             "severity":               self.severity,

# #             "confidence":             round(self.confidence, 2),

# #             "riskScore":              round(self.risk_score, 2),

# #             "timestamp":              round(self.timestamp_seconds, 3),

# #             "originalVideoTimestamp": round(self.original_video_timestamp, 3),

# #             "framePaths":             self.frame_paths,

# #         }
 
 
# # @dataclass

# # class VideoResult:

# #     """

# #     Per-video summary inside the completion payload.
 
# #     API field mapping (POST /api/internal/analysis/completed)

# #     ──────────────────────────────────────────────────────────

# #     videoId           → STRING per API spec

# #     sequenceNo        → int

# #     videoName         → original file name e.g. "front_cabin.mp4"

# #     durationSeconds   → float

# #     durationFormatted → "H:MM:SS"   FIX: was missing

# #     fps               → float       FIX: was missing

# #     sizeMb            → float       FIX: was missing

# #     originalS3Key     → the s3Key from the inbound VideoJob

# #     violations        → list of ViolationResult

# #     """

# #     video_id:           int

# #     video_name:         str

# #     sequence_no:        int

# #     duration_seconds:   float

# #     original_s3_key:    str

# #     violations:         List[ViolationResult] = field(default_factory=list)
 
# #     # FIX: restored missing fields

# #     duration_formatted: str   = ""

# #     fps:                float = 0.0

# #     size_mb:            float = 0.0
 
# #     def to_dict(self) -> dict:

# #         return {

# #             "videoId":           str(self.video_id),   # API spec says STRING

# #             "sequenceNo":        self.sequence_no,

# #             "videoName":         self.video_name,

# #             "durationSeconds":   round(self.duration_seconds, 3),

# #             "durationFormatted": self.duration_formatted,

# #             "fps":               round(self.fps, 3),

# #             "sizeMb":            round(self.size_mb, 2),

# #             "originalS3Key":     self.original_s3_key,

# #             "violations":        [v.to_dict() for v in self.violations],

# #         }
 
 
# # @dataclass

# # class CompletionPayload:

# #     """

# #     Full payload for POST /api/internal/analysis/completed.

# #     """

# #     job_id:          str

# #     journey_id:      int

# #     train_detail_id: int

# #     folder_name:     str

# #     processing_time: int                         # wall-clock milliseconds

# #     video_results:   List[VideoResult] = field(default_factory=list)

# #     batch_id:        str = ""                    # auto-filled in to_dict if blank
 
# #     def to_dict(self) -> dict:

# #         return {

# #             "jobId":          self.job_id,

# #             "journeyId":      self.journey_id,

# #             "batchId":        self.batch_id or f"BATCH-{self.job_id}",

# #             "trainDetailId":  self.train_detail_id,

# #             "folderName":     self.folder_name,

# #             "processingTime": self.processing_time,

# #             "videoResults":   [vr.to_dict() for vr in self.video_results],

# #         }



# # # """
# # # models.py
# # # ─────────
# # # Dataclasses that represent the RabbitMQ job message and the response
# # # structures expected by the Spring Boot completion / progress APIs.

# # # These are pure data holders — no I/O, no ML logic.
# # # """

# # # from __future__ import annotations

# # # from dataclasses import dataclass, field
# # # from typing import List, Optional


# # # # ── Inbound (from RabbitMQ) ──────────────────────────────────────────────────

# # # @dataclass
# # # class VideoJob:
# # #     """One video entry inside an AnalysisJobMessage."""
# # #     video_id:    int
# # #     sequence_no: int
# # #     s3_key:      str     # e.g. "journeys/101/original/video1.mp4"

# # #     @classmethod
# # #     def from_dict(cls, d: dict) -> "VideoJob":
# # #         return cls(
# # #             video_id    = int(d["videoId"]),
# # #             sequence_no = int(d["sequenceNo"]),
# # #             s3_key      = d["s3Key"],
# # #         )


# # # @dataclass
# # # class AnalysisJobMessage:
# # #     """
# # #     RabbitMQ message consumed from the 'analysis.jobs' queue.

# # #     Example JSON
# # #     ────────────
# # #     {
# # #         "jobId":     "JOB123",
# # #         "journeyId": 101,
# # #         "videos": [
# # #             {"videoId": 1001, "sequenceNo": 1, "s3Key": "journeys/101/original/video1.mp4"},
# # #             {"videoId": 1002, "sequenceNo": 2, "s3Key": "journeys/101/original/video2.mp4"}
# # #         ]
# # #     }
# # #     """
# # #     job_id:     str
# # #     journey_id: int
# # #     videos:     List[VideoJob]

# # #     @classmethod
# # #     def from_dict(cls, d: dict) -> "AnalysisJobMessage":
# # #         return cls(
# # #             job_id     = d["jobId"],
# # #             journey_id = int(d["journeyId"]),
# # #             videos     = [VideoJob.from_dict(v) for v in d.get("videos", [])],
# # #         )


# # # # ── Outbound (to Spring Boot) ────────────────────────────────────────────────

# # # @dataclass
# # # class ViolationResult:
# # #     """
# # #     One violation event to be persisted as a ViolationEvent + AnalysisFrame
# # #     by Spring Boot.

# # #     Fields
# # #     ──────
# # #     violation_type  : canonical type string, e.g. "PHONE_USAGE", "SEAT_ABSENCE",
# # #                       "DROWSINESS"
# # #     severity        : "HIGH" | "MEDIUM" | "LOW"
# # #     confidence      : 0.0 – 100.0  (percentage)
# # #     risk_score      : 0 – 100
# # #     timestamp       : HH:MM:SS display string  (global journey time)
# # #     timestamp_seconds: int  — seconds from start of journey
# # #     original_video_timestamp : "<filename> HH:MM:SS"  — local file + local time
# # #     frame_paths     : list of S3 keys (NOT signed URLs)
# # #     """
# # #     violation_type:            str
# # #     severity:                  str
# # #     confidence:                float
# # #     risk_score:                int
# # #     timestamp:                 str
# # #     timestamp_seconds:         int
# # #     original_video_timestamp:  str
# # #     frame_paths:               List[str] = field(default_factory=list)

# # #     def to_dict(self) -> dict:
# # #         return {
# # #             "violationType":           self.violation_type,
# # #             "severity":                self.severity,
# # #             "confidence":              round(self.confidence, 2),
# # #             "riskScore":               self.risk_score,
# # #             "timestamp":               self.timestamp,
# # #             "timestampSeconds":        self.timestamp_seconds,
# # #             "original_video_timestamp": self.original_video_timestamp,
# # #             "framePaths":              self.frame_paths,
# # #         }


# # # @dataclass
# # # class VideoResult:
# # #     """
# # #     Per-video summary sent inside the completion payload.

# # #     Fields
# # #     ──────
# # #     video_id          : mirrors VideoJob.video_id
# # #     video_name        : display filename
# # #     sequence_no       : mirrors VideoJob.sequence_no
# # #     duration_seconds  : float — duration of this video file
# # #     duration_formatted: "H:MM:SS"
# # #     fps               : frames per second
# # #     size_mb           : file size in MB
# # #     violations        : list of ViolationResult
# # #     """
# # #     video_id:           int
# # #     video_name:         str
# # #     sequence_no:        int
# # #     duration_seconds:   float
# # #     duration_formatted: str
# # #     fps:                float
# # #     size_mb:            float
# # #     violations:         List[ViolationResult] = field(default_factory=list)

# # #     def to_dict(self) -> dict:
# # #         return {
# # #             "videoId":           self.video_id,
# # #             "video_name":        self.video_name,
# # #             "sequenceNo":        self.sequence_no,
# # #             "durationSeconds":   round(self.duration_seconds, 3),
# # #             "durationFormatted": self.duration_formatted,
# # #             "fps":               round(self.fps, 3),
# # #             "sizeMb":            round(self.size_mb, 2),
# # #             "violations":        [v.to_dict() for v in self.violations],
# # #         }


# # # @dataclass
# # # class CompletionPayload:
# # #     """
# # #     Full payload for POST /api/internal/analysis/completed.
# # #     """
# # #     job_id:          str
# # #     journey_id:      int
# # #     processing_time: int           # wall-clock milliseconds
# # #     video_results:   List[VideoResult] = field(default_factory=list)

# # #     def to_dict(self) -> dict:
# # #         return {
# # #             "jobId":          self.job_id,
# # #             "journeyId":      self.journey_id,
# # #             "processingTime": self.processing_time,
# # #             "videoResults":   [vr.to_dict() for vr in self.video_results],
# # #         }

# # """
# # models.py
# # ─────────
# # Dataclasses that represent the RabbitMQ job message and the response
# # structures expected by the Spring Boot completion / progress APIs.

# # Aligned with CVVRS API Documentation (Base: https://cvvrsrailway-api.sconexsoft.com/cvs)

# # Changes from previous version
# # ──────────────────────────────
# # • VideoJob         — added originalFileName field (present in RabbitMQ message).
# # • AnalysisJobMessage — added trainDetailId, folderName, priority fields.
# # • ViolationResult  — timestamp / originalVideoTimestamp are now float (seconds),
# #                      matching the API schema. The HH:MM:SS display string is dropped
# #                      from the outbound payload (Spring Boot derives it from the float).
# # • VideoResult      — added originalS3Key (required by /internal/analysis/completed).
# #                      videoId is serialised as a STRING per the API spec.
# # • CompletionPayload — added batchId, trainDetailId, folderName fields required by
# #                       the /internal/analysis/completed endpoint.
# # """

# # from __future__ import annotations

# # from dataclasses import dataclass, field
# # from typing import List, Optional


# # # ── Inbound (from RabbitMQ) ──────────────────────────────────────────────────

# # @dataclass
# # class VideoJob:
# #     """One video entry inside an AnalysisJobMessage."""
# #     video_id:          int
# #     sequence_no:       int
# #     s3_key:            str            # e.g. "journeys/1/2026-06-10/.../original/001_front_cabin.mp4"
# #     original_filename: str = ""       # e.g. "front_cabin.mp4"  (added per API spec)

# #     @classmethod
# #     def from_dict(cls, d: dict) -> "VideoJob":
# #         return cls(
# #             video_id          = int(d["videoId"]),
# #             sequence_no       = int(d["sequenceNo"]),
# #             s3_key            = d["s3Key"],
# #             original_filename = d.get("originalFileName", ""),
# #         )


# # @dataclass
# # class AnalysisJobMessage:
# #     """
# #     RabbitMQ message consumed from the 'analysis.jobs' queue.

# #     Queue    : analysis.jobs
# #     Exchange : dev.analysis.exchange
# #     Routing  : dev.analysis.jobs.created

# #     Example JSON
# #     ────────────
# #     {
# #         "jobId":         "JOB-ABC123XYZ",
# #         "journeyId":     10,
# #         "trainDetailId": 1,
# #         "folderName":    "journeys/1/2026-06-10/JRN-20260610-1-ABC123",
# #         "priority":      "NORMAL",
# #         "videos": [
# #             {
# #                 "videoId":           1,
# #                 "sequenceNo":        1,
# #                 "originalFileName":  "front_cabin.mp4",
# #                 "s3Key":             "journeys/1/2026-06-10/.../original/001_front_cabin.mp4"
# #             }
# #         ]
# #     }
# #     """
# #     job_id:          str
# #     journey_id:      int
# #     train_detail_id: int              # NEW — needed in completion payload
# #     folder_name:     str              # NEW — S3 folder prefix for this journey
# #     videos:          List[VideoJob]
# #     priority:        str = "NORMAL"   # NEW — NORMAL | HIGH

# #     @classmethod
# #     def from_dict(cls, d: dict) -> "AnalysisJobMessage":
# #         return cls(
# #             job_id          = d["jobId"],
# #             journey_id      = int(d["journeyId"]),
# #             train_detail_id = int(d.get("trainDetailId", 0)),
# #             folder_name     = d.get("folderName", f"journeys/{d.get('journeyId', 0)}"),
# #             videos          = [VideoJob.from_dict(v) for v in d.get("videos", [])],
# #             priority        = d.get("priority", "NORMAL"),
# #         )


# # # ── Outbound (to Spring Boot) ────────────────────────────────────────────────

# # @dataclass
# # class ViolationResult:
# #     """
# #     One violation event.

# #     API field mapping (POST /api/internal/analysis/completed)
# #     ──────────────────────────────────────────────────────────
# #     violationType         → canonical string: PHONE_USAGE | DROWSY | SEAT_ABSENCE …
# #     severity              → CRITICAL | HIGH | MEDIUM | LOW
# #     confidence            → 0.0 – 100.0 (percentage)
# #     riskScore             → 0.0 – 100.0
# #     timestamp             → float seconds from start of journey   ← was HH:MM:SS string
# #     originalVideoTimestamp→ float seconds within the source video ← was "<file> HH:MM:SS"
# #     framePaths            → list of S3 keys (NOT signed URLs)
# #     """
# #     violation_type:             str
# #     severity:                   str
# #     confidence:                 float
# #     risk_score:                 float          # float per API spec (was int)
# #     timestamp_seconds:          float          # journey-global seconds (float)
# #     original_video_timestamp:   float          # local-video seconds (float)
# #     frame_paths:                List[str] = field(default_factory=list)

# #     def to_dict(self) -> dict:
# #         return {
# #             "violationType":          self.violation_type,
# #             "severity":               self.severity,
# #             "confidence":             round(self.confidence, 2),
# #             "riskScore":              round(self.risk_score, 2),
# #             "timestamp":              round(self.timestamp_seconds, 3),
# #             "originalVideoTimestamp": round(self.original_video_timestamp, 3),
# #             "framePaths":             self.frame_paths,
# #         }


# # @dataclass
# # class VideoResult:
# #     """
# #     Per-video summary inside the completion payload.

# #     API field mapping (POST /api/internal/analysis/completed)
# #     ──────────────────────────────────────────────────────────
# #     videoId        → STRING per API spec  (was int in previous version)
# #     sequenceNo     → int
# #     durationSeconds→ float
# #     originalS3Key  → NEW — the s3Key from the inbound VideoJob
# #     violations     → list of ViolationResult
# #     """
# #     video_id:         int
# #     video_name:       str
# #     sequence_no:      int
# #     duration_seconds: float
# #     original_s3_key:  str             # NEW — required by API
# #     violations:       List[ViolationResult] = field(default_factory=list)

# #     def to_dict(self) -> dict:
# #         return {
# #             "videoId":         str(self.video_id),   # API spec says STRING
# #             "sequenceNo":      self.sequence_no,
# #             "durationSeconds": round(self.duration_seconds, 3),
# #             "originalS3Key":   self.original_s3_key,
# #             "violations":      [v.to_dict() for v in self.violations],
# #         }


# # @dataclass
# # class CompletionPayload:
# #     """
# #     Full payload for POST /api/internal/analysis/completed.

# #     New required fields vs previous version
# #     ────────────────────────────────────────
# #     • batchId        — generated as "BATCH-<jobId>" if not supplied
# #     • trainDetailId  — forwarded from the RabbitMQ message
# #     • folderName     — forwarded from the RabbitMQ message
# #     """
# #     job_id:          str
# #     journey_id:      int
# #     train_detail_id: int
# #     folder_name:     str
# #     processing_time: int                         # wall-clock milliseconds
# #     video_results:   List[VideoResult] = field(default_factory=list)
# #     batch_id:        str = ""                    # auto-filled in to_dict if blank

# #     def to_dict(self) -> dict:
# #         return {
# #             "jobId":          self.job_id,
# #             "journeyId":      self.journey_id,
# #             "batchId":        self.batch_id or f"BATCH-{self.job_id}",
# #             "trainDetailId":  self.train_detail_id,
# #             "folderName":     self.folder_name,
# #             "processingTime": self.processing_time,
# #             "videoResults":   [vr.to_dict() for vr in self.video_results],
# #         }



# """

# models.py

# ─────────

# Dataclasses that represent the RabbitMQ job message and the response

# structures expected by the Spring Boot completion / progress APIs.
 
# Aligned with CVVRS API (Base: https://cvvrsrailway-api.sconexsoft.com/cvs)
 
# Fixes from previous version

# ─────────────────────────────

# • AnalysisJobMessage — added callbackBaseUrl field (sent by Spring Boot,

#   used by consumer.py to route callbacks to the correct server).

# • VideoResult — restored durationFormatted, fps, sizeMb fields that were

#   accidentally removed. Spring Boot expects all three in the completion payload.

# • VideoResult.to_dict() — serializes all fields including the restored ones.

# """
 
# from __future__ import annotations
 
# from dataclasses import dataclass, field

# from typing import List, Optional
 

# # ── Helpers ──────────────────────────────────────────────────────────────────

# def _fmt_duration(seconds: float) -> str:
#     """Convert a float number of seconds to an 'H:MM:SS' display string."""
#     t  = int(seconds)
#     hh = t // 3600
#     mm = (t % 3600) // 60
#     ss = t % 60
#     return f"{hh}:{mm:02d}:{ss:02d}"


# # ── Inbound (from RabbitMQ) ──────────────────────────────────────────────────
 
# @dataclass

# class VideoJob:

#     """One video entry inside an AnalysisJobMessage."""

#     video_id:          int

#     sequence_no:       int

#     s3_key:            str            # e.g. "journeys/1/2026-06-10/.../original/001_front_cabin.mp4"

#     original_filename: str = ""       # e.g. "front_cabin.mp4"
 
#     @classmethod

#     def from_dict(cls, d: dict) -> "VideoJob":

#         return cls(

#             video_id          = int(d["videoId"]),

#             sequence_no       = int(d["sequenceNo"]),

#             s3_key            = d["s3Key"],

#             original_filename = d.get("originalFileName", ""),

#         )
 
 
# @dataclass

# class AnalysisJobMessage:

#     """

#     RabbitMQ message consumed from the 'analysis.jobs' queue.
 
#     Queue    : analysis.jobs

#     Exchange : dev.analysis.exchange

#     Routing  : dev.analysis.jobs.created
 
#     Example JSON

#     ────────────

#     {

#         "jobId":           "JOB-ABC123XYZ",

#         "journeyId":       10,

#         "trainDetailId":   1,

#         "folderName":      "journeys/1/2026-06-10/JRN-20260610-1-ABC123",

#         "callbackBaseUrl": "https://cvvrsrailway-api.sconexsoft.com/cvs/api/internal/analysis",

#         "priority":        "NORMAL",

#         "videos": [

#             {

#                 "videoId":           1,

#                 "sequenceNo":        1,

#                 "originalFileName":  "front_cabin.mp4",

#                 "s3Key":             "journeys/1/2026-06-10/.../original/001_front_cabin.mp4"

#             }

#         ]

#     }

#     """

#     job_id:            str

#     journey_id:        int

#     train_detail_id:   int

#     folder_name:       str

#     videos:            List[VideoJob]

#     callback_base_url: str = ""       # FIX: was missing — Spring Boot sends this

#     priority:          str = "NORMAL"
 
#     @classmethod

#     def from_dict(cls, d: dict) -> "AnalysisJobMessage":

#         return cls(

#             job_id            = d["jobId"],

#             journey_id        = int(d["journeyId"]),

#             train_detail_id   = int(d.get("trainDetailId", 0)),

#             folder_name       = d.get("folderName", f"journeys/{d.get('journeyId', 0)}"),

#             callback_base_url = d.get("callbackBaseUrl", ""),   # FIX

#             videos            = [VideoJob.from_dict(v) for v in d.get("videos", [])],

#             priority          = d.get("priority", "NORMAL"),

#         )
 
 
# # ── Outbound (to Spring Boot) ────────────────────────────────────────────────
 
# @dataclass

# class ViolationResult:

#     """

#     One violation event.
 
#     API field mapping (POST /api/internal/analysis/completed)

#     ──────────────────────────────────────────────────────────

#     violationType          → canonical string: PHONE_USAGE | DROWSY | SEAT_ABSENCE ...

#     severity               → CRITICAL | HIGH | MEDIUM | LOW

#     confidence             → 0.0 – 100.0 (percentage)

#     riskScore              → 0.0 – 100.0

#     timestamp              → float seconds from start of journey (global)

#     originalVideoTimestamp → float seconds within the source video (local)

#     framePaths             → list of S3 keys (NOT signed URLs)

#     """

#     violation_type:             str

#     severity:                   str

#     confidence:                 float

#     risk_score:                 float

#     timestamp_seconds:          float          # journey-global seconds

#     original_video_timestamp:   float          # local-video seconds
#     duration_seconds:           float = 0.0

#     frame_paths:                List[str] = field(default_factory=list)
 
#     def to_dict(self) -> dict:

#         return {

#             "violationType":          self.violation_type,

#             "severity":               self.severity,

#             "confidence":             round(self.confidence, 2),

#             "riskScore":              round(self.risk_score, 2),

#             "timestamp":              self.timestamp_seconds if isinstance(self.timestamp_seconds, str) else _fmt_duration(self.timestamp_seconds),

#             "originalVideoTimestamp": self.original_video_timestamp if isinstance(self.original_video_timestamp, str) else _fmt_duration(self.original_video_timestamp),

#             "framePaths":             self.frame_paths,
            

#         }
 
 
# @dataclass

# class VideoResult:

#     """

#     Per-video summary inside the completion payload.
 
#     API field mapping (POST /api/internal/analysis/completed)

#     ──────────────────────────────────────────────────────────

#     videoId           → STRING per API spec

#     sequenceNo        → int

#     videoName         → original file name e.g. "front_cabin.mp4"

#     durationSeconds   → float

#     durationFormatted → "H:MM:SS"   FIX: was missing

#     fps               → float       FIX: was missing

#     sizeMb            → float       FIX: was missing

#     originalS3Key     → the s3Key from the inbound VideoJob

#     violations        → list of ViolationResult

#     """

#     video_id:           int

#     video_name:         str

#     sequence_no:        int

#     duration_seconds:   float

#     original_s3_key:    str

#     violations:         List[ViolationResult] = field(default_factory=list)
 
#     # FIX: restored missing fields

#     duration_formatted: str   = ""

#     fps:                float = 0.0

#     size_mb:            float = 0.0
 
#     def to_dict(self) -> dict:

#         return {

#             "videoId":           str(self.video_id),   # API spec says STRING

#             "sequenceNo":        self.sequence_no,

#             "videoName":         self.video_name,

#             "durationSeconds":   round(self.duration_seconds, 3),

#             "durationFormatted": self.duration_formatted,

#             "fps":               round(self.fps, 3),

#             "sizeMb":            round(self.size_mb, 2),

#             "originalS3Key":     self.original_s3_key,

#             "violations":        [v.to_dict() for v in self.violations],

#         }
 
 
# @dataclass

# class CompletionPayload:

#     """

#     Full payload for POST /api/internal/analysis/completed.

#     """

#     job_id:          str

#     journey_id:      int

#     train_detail_id: int

#     folder_name:     str

#     processing_time: int                         # wall-clock milliseconds

#     video_results:   List[VideoResult] = field(default_factory=list)

#     batch_id:        str = ""                    # auto-filled in to_dict if blank
 
#     def to_dict(self) -> dict:

#         return {

#             "jobId":          self.job_id,

#             "journeyId":      self.journey_id,

#             "batchId":        self.batch_id or f"BATCH-{self.job_id}",

#             "trainDetailId":  self.train_detail_id,

#             "folderName":     self.folder_name,

#             "processingTime": self.processing_time,

#             "videoResults":   [vr.to_dict() for vr in self.video_results],

#         }
 # # # """
# # # models.py
# # # ─────────
# # # Dataclasses that represent the RabbitMQ job message and the response
# # # structures expected by the Spring Boot completion / progress APIs.

# # # These are pure data holders — no I/O, no ML logic.
# # # """

# # # from __future__ import annotations

# # # from dataclasses import dataclass, field
# # # from typing import List, Optional


# # # # ── Inbound (from RabbitMQ) ──────────────────────────────────────────────────

# # # @dataclass
# # # class VideoJob:
# # #     """One video entry inside an AnalysisJobMessage."""
# # #     video_id:    int
# # #     sequence_no: int
# # #     s3_key:      str     # e.g. "journeys/101/original/video1.mp4"

# # #     @classmethod
# # #     def from_dict(cls, d: dict) -> "VideoJob":
# # #         return cls(
# # #             video_id    = int(d["videoId"]),
# # #             sequence_no = int(d["sequenceNo"]),
# # #             s3_key      = d["s3Key"],
# # #         )


# # # @dataclass
# # # class AnalysisJobMessage:
# # #     """
# # #     RabbitMQ message consumed from the 'analysis.jobs' queue.

# # #     Example JSON
# # #     ────────────
# # #     {
# # #         "jobId":     "JOB123",
# # #         "journeyId": 101,
# # #         "videos": [
# # #             {"videoId": 1001, "sequenceNo": 1, "s3Key": "journeys/101/original/video1.mp4"},
# # #             {"videoId": 1002, "sequenceNo": 2, "s3Key": "journeys/101/original/video2.mp4"}
# # #         ]
# # #     }
# # #     """
# # #     job_id:     str
# # #     journey_id: int
# # #     videos:     List[VideoJob]

# # #     @classmethod
# # #     def from_dict(cls, d: dict) -> "AnalysisJobMessage":
# # #         return cls(
# # #             job_id     = d["jobId"],
# # #             journey_id = int(d["journeyId"]),
# # #             videos     = [VideoJob.from_dict(v) for v in d.get("videos", [])],
# # #         )


# # # # ── Outbound (to Spring Boot) ────────────────────────────────────────────────

# # # @dataclass
# # # class ViolationResult:
# # #     """
# # #     One violation event to be persisted as a ViolationEvent + AnalysisFrame
# # #     by Spring Boot.

# # #     Fields
# # #     ──────
# # #     violation_type  : canonical type string, e.g. "PHONE_USAGE", "SEAT_ABSENCE",
# # #                       "DROWSINESS"
# # #     severity        : "HIGH" | "MEDIUM" | "LOW"
# # #     confidence      : 0.0 – 100.0  (percentage)
# # #     risk_score      : 0 – 100
# # #     timestamp       : HH:MM:SS display string  (global journey time)
# # #     timestamp_seconds: int  — seconds from start of journey
# # #     original_video_timestamp : "<filename> HH:MM:SS"  — local file + local time
# # #     frame_paths     : list of S3 keys (NOT signed URLs)
# # #     """
# # #     violation_type:            str
# # #     severity:                  str
# # #     confidence:                float
# # #     risk_score:                int
# # #     timestamp:                 str
# # #     timestamp_seconds:         int
# # #     original_video_timestamp:  str
# # #     frame_paths:               List[str] = field(default_factory=list)

# # #     def to_dict(self) -> dict:
# # #         return {
# # #             "violationType":           self.violation_type,
# # #             "severity":                self.severity,
# # #             "confidence":              round(self.confidence, 2),
# # #             "riskScore":               self.risk_score,
# # #             "timestamp":               self.timestamp,
# # #             "timestampSeconds":        self.timestamp_seconds,
# # #             "original_video_timestamp": self.original_video_timestamp,
# # #             "framePaths":              self.frame_paths,
# # #         }


# # # @dataclass
# # # class VideoResult:
# # #     """
# # #     Per-video summary sent inside the completion payload.

# # #     Fields
# # #     ──────
# # #     video_id          : mirrors VideoJob.video_id
# # #     video_name        : display filename
# # #     sequence_no       : mirrors VideoJob.sequence_no
# # #     duration_seconds  : float — duration of this video file
# # #     duration_formatted: "H:MM:SS"
# # #     fps               : frames per second
# # #     size_mb           : file size in MB
# # #     violations        : list of ViolationResult
# # #     """
# # #     video_id:           int
# # #     video_name:         str
# # #     sequence_no:        int
# # #     duration_seconds:   float
# # #     duration_formatted: str
# # #     fps:                float
# # #     size_mb:            float
# # #     violations:         List[ViolationResult] = field(default_factory=list)

# # #     def to_dict(self) -> dict:
# # #         return {
# # #             "videoId":           self.video_id,
# # #             "video_name":        self.video_name,
# # #             "sequenceNo":        self.sequence_no,
# # #             "durationSeconds":   round(self.duration_seconds, 3),
# # #             "durationFormatted": self.duration_formatted,
# # #             "fps":               round(self.fps, 3),
# # #             "sizeMb":            round(self.size_mb, 2),
# # #             "violations":        [v.to_dict() for v in self.violations],
# # #         }


# # # @dataclass
# # # class CompletionPayload:
# # #     """
# # #     Full payload for POST /api/internal/analysis/completed.
# # #     """
# # #     job_id:          str
# # #     journey_id:      int
# # #     processing_time: int           # wall-clock milliseconds
# # #     video_results:   List[VideoResult] = field(default_factory=list)

# # #     def to_dict(self) -> dict:
# # #         return {
# # #             "jobId":          self.job_id,
# # #             "journeyId":      self.journey_id,
# # #             "processingTime": self.processing_time,
# # #             "videoResults":   [vr.to_dict() for vr in self.video_results],
# # #         }

# # """
# # models.py
# # ─────────
# # Dataclasses that represent the RabbitMQ job message and the response
# # structures expected by the Spring Boot completion / progress APIs.

# # Aligned with CVVRS API Documentation (Base: https://cvvrsrailway-api.sconexsoft.com/cvs)

# # Changes from previous version
# # ──────────────────────────────
# # • VideoJob         — added originalFileName field (present in RabbitMQ message).
# # • AnalysisJobMessage — added trainDetailId, folderName, priority fields.
# # • ViolationResult  — timestamp / originalVideoTimestamp are now float (seconds),
# #                      matching the API schema. The HH:MM:SS display string is dropped
# #                      from the outbound payload (Spring Boot derives it from the float).
# # • VideoResult      — added originalS3Key (required by /internal/analysis/completed).
# #                      videoId is serialised as a STRING per the API spec.
# # • CompletionPayload — added batchId, trainDetailId, folderName fields required by
# #                       the /internal/analysis/completed endpoint.
# # """

# # from __future__ import annotations

# # from dataclasses import dataclass, field
# # from typing import List, Optional


# # # ── Inbound (from RabbitMQ) ──────────────────────────────────────────────────

# # @dataclass
# # class VideoJob:
# #     """One video entry inside an AnalysisJobMessage."""
# #     video_id:          int
# #     sequence_no:       int
# #     s3_key:            str            # e.g. "journeys/1/2026-06-10/.../original/001_front_cabin.mp4"
# #     original_filename: str = ""       # e.g. "front_cabin.mp4"  (added per API spec)

# #     @classmethod
# #     def from_dict(cls, d: dict) -> "VideoJob":
# #         return cls(
# #             video_id          = int(d["videoId"]),
# #             sequence_no       = int(d["sequenceNo"]),
# #             s3_key            = d["s3Key"],
# #             original_filename = d.get("originalFileName", ""),
# #         )


# # @dataclass
# # class AnalysisJobMessage:
# #     """
# #     RabbitMQ message consumed from the 'analysis.jobs' queue.

# #     Queue    : analysis.jobs
# #     Exchange : dev.analysis.exchange
# #     Routing  : dev.analysis.jobs.created

# #     Example JSON
# #     ────────────
# #     {
# #         "jobId":         "JOB-ABC123XYZ",
# #         "journeyId":     10,
# #         "trainDetailId": 1,
# #         "folderName":    "journeys/1/2026-06-10/JRN-20260610-1-ABC123",
# #         "priority":      "NORMAL",
# #         "videos": [
# #             {
# #                 "videoId":           1,
# #                 "sequenceNo":        1,
# #                 "originalFileName":  "front_cabin.mp4",
# #                 "s3Key":             "journeys/1/2026-06-10/.../original/001_front_cabin.mp4"
# #             }
# #         ]
# #     }
# #     """
# #     job_id:          str
# #     journey_id:      int
# #     train_detail_id: int              # NEW — needed in completion payload
# #     folder_name:     str              # NEW — S3 folder prefix for this journey
# #     videos:          List[VideoJob]
# #     priority:        str = "NORMAL"   # NEW — NORMAL | HIGH

# #     @classmethod
# #     def from_dict(cls, d: dict) -> "AnalysisJobMessage":
# #         return cls(
# #             job_id          = d["jobId"],
# #             journey_id      = int(d["journeyId"]),
# #             train_detail_id = int(d.get("trainDetailId", 0)),
# #             folder_name     = d.get("folderName", f"journeys/{d.get('journeyId', 0)}"),
# #             videos          = [VideoJob.from_dict(v) for v in d.get("videos", [])],
# #             priority        = d.get("priority", "NORMAL"),
# #         )


# # # ── Outbound (to Spring Boot) ────────────────────────────────────────────────

# # @dataclass
# # class ViolationResult:
# #     """
# #     One violation event.

# #     API field mapping (POST /api/internal/analysis/completed)
# #     ──────────────────────────────────────────────────────────
# #     violationType         → canonical string: PHONE_USAGE | DROWSY | SEAT_ABSENCE …
# #     severity              → CRITICAL | HIGH | MEDIUM | LOW
# #     confidence            → 0.0 – 100.0 (percentage)
# #     riskScore             → 0.0 – 100.0
# #     timestamp             → float seconds from start of journey   ← was HH:MM:SS string
# #     originalVideoTimestamp→ float seconds within the source video ← was "<file> HH:MM:SS"
# #     framePaths            → list of S3 keys (NOT signed URLs)
# #     """
# #     violation_type:             str
# #     severity:                   str
# #     confidence:                 float
# #     risk_score:                 float          # float per API spec (was int)
# #     timestamp_seconds:          float          # journey-global seconds (float)
# #     original_video_timestamp:   float          # local-video seconds (float)
# #     frame_paths:                List[str] = field(default_factory=list)

# #     def to_dict(self) -> dict:
# #         return {
# #             "violationType":          self.violation_type,
# #             "severity":               self.severity,
# #             "confidence":             round(self.confidence, 2),
# #             "riskScore":              round(self.risk_score, 2),
# #             "timestamp":              round(self.timestamp_seconds, 3),
# #             "originalVideoTimestamp": round(self.original_video_timestamp, 3),
# #             "framePaths":             self.frame_paths,
# #         }


# # @dataclass
# # class VideoResult:
# #     """
# #     Per-video summary inside the completion payload.

# #     API field mapping (POST /api/internal/analysis/completed)
# #     ──────────────────────────────────────────────────────────
# #     videoId        → STRING per API spec  (was int in previous version)
# #     sequenceNo     → int
# #     durationSeconds→ float
# #     originalS3Key  → NEW — the s3Key from the inbound VideoJob
# #     violations     → list of ViolationResult
# #     """
# #     video_id:         int
# #     video_name:       str
# #     sequence_no:      int
# #     duration_seconds: float
# #     original_s3_key:  str             # NEW — required by API
# #     violations:       List[ViolationResult] = field(default_factory=list)

# #     def to_dict(self) -> dict:
# #         return {
# #             "videoId":         str(self.video_id),   # API spec says STRING
# #             "sequenceNo":      self.sequence_no,
# #             "durationSeconds": round(self.duration_seconds, 3),
# #             "originalS3Key":   self.original_s3_key,
# #             "violations":      [v.to_dict() for v in self.violations],
# #         }


# # @dataclass
# # class CompletionPayload:
# #     """
# #     Full payload for POST /api/internal/analysis/completed.

# #     New required fields vs previous version
# #     ────────────────────────────────────────
# #     • batchId        — generated as "BATCH-<jobId>" if not supplied
# #     • trainDetailId  — forwarded from the RabbitMQ message
# #     • folderName     — forwarded from the RabbitMQ message
# #     """
# #     job_id:          str
# #     journey_id:      int
# #     train_detail_id: int
# #     folder_name:     str
# #     processing_time: int                         # wall-clock milliseconds
# #     video_results:   List[VideoResult] = field(default_factory=list)
# #     batch_id:        str = ""                    # auto-filled in to_dict if blank

# #     def to_dict(self) -> dict:
# #         return {
# #             "jobId":          self.job_id,
# #             "journeyId":      self.journey_id,
# #             "batchId":        self.batch_id or f"BATCH-{self.job_id}",
# #             "trainDetailId":  self.train_detail_id,
# #             "folderName":     self.folder_name,
# #             "processingTime": self.processing_time,
# #             "videoResults":   [vr.to_dict() for vr in self.video_results],
# #         }



# """

# models.py

# ─────────

# Dataclasses that represent the RabbitMQ job message and the response

# structures expected by the Spring Boot completion / progress APIs.
 
# Aligned with CVVRS API (Base: https://cvvrsrailway-api.sconexsoft.com/cvs)
 
# Fixes from previous version

# ─────────────────────────────

# • AnalysisJobMessage — added callbackBaseUrl field (sent by Spring Boot,

#   used by consumer.py to route callbacks to the correct server).

# • VideoResult — restored durationFormatted, fps, sizeMb fields that were

#   accidentally removed. Spring Boot expects all three in the completion payload.

# • VideoResult.to_dict() — serializes all fields including the restored ones.

# """
 
# from __future__ import annotations
 
# from dataclasses import dataclass, field

# from typing import List, Optional
 
 
# # ── Inbound (from RabbitMQ) ──────────────────────────────────────────────────
 
# @dataclass

# class VideoJob:

#     """One video entry inside an AnalysisJobMessage."""

#     video_id:          int

#     sequence_no:       int

#     s3_key:            str            # e.g. "journeys/1/2026-06-10/.../original/001_front_cabin.mp4"

#     original_filename: str = ""       # e.g. "front_cabin.mp4"
 
#     @classmethod

#     def from_dict(cls, d: dict) -> "VideoJob":

#         return cls(

#             video_id          = int(d["videoId"]),

#             sequence_no       = int(d["sequenceNo"]),

#             s3_key            = d["s3Key"],

#             original_filename = d.get("originalFileName", ""),

#         )
 
 
# @dataclass

# class AnalysisJobMessage:

#     """

#     RabbitMQ message consumed from the 'analysis.jobs' queue.
 
#     Queue    : analysis.jobs

#     Exchange : dev.analysis.exchange

#     Routing  : dev.analysis.jobs.created
 
#     Example JSON

#     ────────────

#     {

#         "jobId":           "JOB-ABC123XYZ",

#         "journeyId":       10,

#         "trainDetailId":   1,

#         "folderName":      "journeys/1/2026-06-10/JRN-20260610-1-ABC123",

#         "callbackBaseUrl": "https://cvvrsrailway-api.sconexsoft.com/cvs/api/internal/analysis",

#         "priority":        "NORMAL",

#         "videos": [

#             {

#                 "videoId":           1,

#                 "sequenceNo":        1,

#                 "originalFileName":  "front_cabin.mp4",

#                 "s3Key":             "journeys/1/2026-06-10/.../original/001_front_cabin.mp4"

#             }

#         ]

#     }

#     """

#     job_id:            str

#     journey_id:        int

#     train_detail_id:   int

#     folder_name:       str

#     videos:            List[VideoJob]

#     callback_base_url: str = ""       # FIX: was missing — Spring Boot sends this

#     priority:          str = "NORMAL"
 
#     @classmethod

#     def from_dict(cls, d: dict) -> "AnalysisJobMessage":

#         return cls(

#             job_id            = d["jobId"],

#             journey_id        = int(d["journeyId"]),

#             train_detail_id   = int(d.get("trainDetailId", 0)),

#             folder_name       = d.get("folderName", f"journeys/{d.get('journeyId', 0)}"),

#             callback_base_url = d.get("callbackBaseUrl", ""),   # FIX

#             videos            = [VideoJob.from_dict(v) for v in d.get("videos", [])],

#             priority          = d.get("priority", "NORMAL"),

#         )
 
 
# # ── Outbound (to Spring Boot) ────────────────────────────────────────────────
 
# @dataclass

# class ViolationResult:

#     """

#     One violation event.
 
#     API field mapping (POST /api/internal/analysis/completed)

#     ──────────────────────────────────────────────────────────

#     violationType          → canonical string: PHONE_USAGE | DROWSY | SEAT_ABSENCE ...

#     severity               → CRITICAL | HIGH | MEDIUM | LOW

#     confidence             → 0.0 – 100.0 (percentage)

#     riskScore              → 0.0 – 100.0

#     timestamp              → float seconds from start of journey (global)

#     originalVideoTimestamp → float seconds within the source video (local)

#     framePaths             → list of S3 keys (NOT signed URLs)

#     """

#     violation_type:             str

#     severity:                   str

#     confidence:                 float

#     risk_score:                 float

#     timestamp_seconds:          float          # journey-global seconds

#     original_video_timestamp:   float          # local-video seconds

#     frame_paths:                List[str] = field(default_factory=list)
 
#     def to_dict(self) -> dict:

#         return {

#             "violationType":          self.violation_type,

#             "severity":               self.severity,

#             "confidence":             round(self.confidence, 2),

#             "riskScore":              round(self.risk_score, 2),

#             "timestamp":              round(self.timestamp_seconds, 3),

#             "originalVideoTimestamp": round(self.original_video_timestamp, 3),

#             "framePaths":             self.frame_paths,

#         }
 
 
# @dataclass

# class VideoResult:

#     """

#     Per-video summary inside the completion payload.
 
#     API field mapping (POST /api/internal/analysis/completed)

#     ──────────────────────────────────────────────────────────

#     videoId           → STRING per API spec

#     sequenceNo        → int

#     videoName         → original file name e.g. "front_cabin.mp4"

#     durationSeconds   → float

#     durationFormatted → "H:MM:SS"   FIX: was missing

#     fps               → float       FIX: was missing

#     sizeMb            → float       FIX: was missing

#     originalS3Key     → the s3Key from the inbound VideoJob

#     violations        → list of ViolationResult

#     """

#     video_id:           int

#     video_name:         str

#     sequence_no:        int

#     duration_seconds:   float

#     original_s3_key:    str

#     violations:         List[ViolationResult] = field(default_factory=list)
 
#     # FIX: restored missing fields

#     duration_formatted: str   = ""

#     fps:                float = 0.0

#     size_mb:            float = 0.0
 
#     def to_dict(self) -> dict:

#         return {

#             "videoId":           str(self.video_id),   # API spec says STRING

#             "sequenceNo":        self.sequence_no,

#             "videoName":         self.video_name,

#             "durationSeconds":   round(self.duration_seconds, 3),

#             "durationFormatted": self.duration_formatted,

#             "fps":               round(self.fps, 3),

#             "sizeMb":            round(self.size_mb, 2),

#             "originalS3Key":     self.original_s3_key,

#             "violations":        [v.to_dict() for v in self.violations],

#         }
 
 
# @dataclass

# class CompletionPayload:

#     """

#     Full payload for POST /api/internal/analysis/completed.

#     """

#     job_id:          str

#     journey_id:      int

#     train_detail_id: int

#     folder_name:     str

#     processing_time: int                         # wall-clock milliseconds

#     video_results:   List[VideoResult] = field(default_factory=list)

#     batch_id:        str = ""                    # auto-filled in to_dict if blank
 
#     def to_dict(self) -> dict:

#         return {

#             "jobId":          self.job_id,

#             "journeyId":      self.journey_id,

#             "batchId":        self.batch_id or f"BATCH-{self.job_id}",

#             "trainDetailId":  self.train_detail_id,

#             "folderName":     self.folder_name,

#             "processingTime": self.processing_time,

#             "videoResults":   [vr.to_dict() for vr in self.video_results],

#         }



# # """
# # models.py
# # ─────────
# # Dataclasses that represent the RabbitMQ job message and the response
# # structures expected by the Spring Boot completion / progress APIs.

# # These are pure data holders — no I/O, no ML logic.
# # """

# # from __future__ import annotations

# # from dataclasses import dataclass, field
# # from typing import List, Optional


# # # ── Inbound (from RabbitMQ) ──────────────────────────────────────────────────

# # @dataclass
# # class VideoJob:
# #     """One video entry inside an AnalysisJobMessage."""
# #     video_id:    int
# #     sequence_no: int
# #     s3_key:      str     # e.g. "journeys/101/original/video1.mp4"

# #     @classmethod
# #     def from_dict(cls, d: dict) -> "VideoJob":
# #         return cls(
# #             video_id    = int(d["videoId"]),
# #             sequence_no = int(d["sequenceNo"]),
# #             s3_key      = d["s3Key"],
# #         )


# # @dataclass
# # class AnalysisJobMessage:
# #     """
# #     RabbitMQ message consumed from the 'analysis.jobs' queue.

# #     Example JSON
# #     ────────────
# #     {
# #         "jobId":     "JOB123",
# #         "journeyId": 101,
# #         "videos": [
# #             {"videoId": 1001, "sequenceNo": 1, "s3Key": "journeys/101/original/video1.mp4"},
# #             {"videoId": 1002, "sequenceNo": 2, "s3Key": "journeys/101/original/video2.mp4"}
# #         ]
# #     }
# #     """
# #     job_id:     str
# #     journey_id: int
# #     videos:     List[VideoJob]

# #     @classmethod
# #     def from_dict(cls, d: dict) -> "AnalysisJobMessage":
# #         return cls(
# #             job_id     = d["jobId"],
# #             journey_id = int(d["journeyId"]),
# #             videos     = [VideoJob.from_dict(v) for v in d.get("videos", [])],
# #         )


# # # ── Outbound (to Spring Boot) ────────────────────────────────────────────────

# # @dataclass
# # class ViolationResult:
# #     """
# #     One violation event to be persisted as a ViolationEvent + AnalysisFrame
# #     by Spring Boot.

# #     Fields
# #     ──────
# #     violation_type  : canonical type string, e.g. "PHONE_USAGE", "SEAT_ABSENCE",
# #                       "DROWSINESS"
# #     severity        : "HIGH" | "MEDIUM" | "LOW"
# #     confidence      : 0.0 – 100.0  (percentage)
# #     risk_score      : 0 – 100
# #     timestamp       : HH:MM:SS display string  (global journey time)
# #     timestamp_seconds: int  — seconds from start of journey
# #     original_video_timestamp : "<filename> HH:MM:SS"  — local file + local time
# #     frame_paths     : list of S3 keys (NOT signed URLs)
# #     """
# #     violation_type:            str
# #     severity:                  str
# #     confidence:                float
# #     risk_score:                int
# #     timestamp:                 str
# #     timestamp_seconds:         int
# #     original_video_timestamp:  str
# #     frame_paths:               List[str] = field(default_factory=list)

# #     def to_dict(self) -> dict:
# #         return {
# #             "violationType":           self.violation_type,
# #             "severity":                self.severity,
# #             "confidence":              round(self.confidence, 2),
# #             "riskScore":               self.risk_score,
# #             "timestamp":               self.timestamp,
# #             "timestampSeconds":        self.timestamp_seconds,
# #             "original_video_timestamp": self.original_video_timestamp,
# #             "framePaths":              self.frame_paths,
# #         }


# # @dataclass
# # class VideoResult:
# #     """
# #     Per-video summary sent inside the completion payload.

# #     Fields
# #     ──────
# #     video_id          : mirrors VideoJob.video_id
# #     video_name        : display filename
# #     sequence_no       : mirrors VideoJob.sequence_no
# #     duration_seconds  : float — duration of this video file
# #     duration_formatted: "H:MM:SS"
# #     fps               : frames per second
# #     size_mb           : file size in MB
# #     violations        : list of ViolationResult
# #     """
# #     video_id:           int
# #     video_name:         str
# #     sequence_no:        int
# #     duration_seconds:   float
# #     duration_formatted: str
# #     fps:                float
# #     size_mb:            float
# #     violations:         List[ViolationResult] = field(default_factory=list)

# #     def to_dict(self) -> dict:
# #         return {
# #             "videoId":           self.video_id,
# #             "video_name":        self.video_name,
# #             "sequenceNo":        self.sequence_no,
# #             "durationSeconds":   round(self.duration_seconds, 3),
# #             "durationFormatted": self.duration_formatted,
# #             "fps":               round(self.fps, 3),
# #             "sizeMb":            round(self.size_mb, 2),
# #             "violations":        [v.to_dict() for v in self.violations],
# #         }


# # @dataclass
# # class CompletionPayload:
# #     """
# #     Full payload for POST /api/internal/analysis/completed.
# #     """
# #     job_id:          str
# #     journey_id:      int
# #     processing_time: int           # wall-clock milliseconds
# #     video_results:   List[VideoResult] = field(default_factory=list)

# #     def to_dict(self) -> dict:
# #         return {
# #             "jobId":          self.job_id,
# #             "journeyId":      self.journey_id,
# #             "processingTime": self.processing_time,
# #             "videoResults":   [vr.to_dict() for vr in self.video_results],
# #         }

# """
# models.py
# ─────────
# Dataclasses that represent the RabbitMQ job message and the response
# structures expected by the Spring Boot completion / progress APIs.

# Aligned with CVVRS API Documentation (Base: https://cvvrsrailway-api.sconexsoft.com/cvs)

# Changes from previous version
# ──────────────────────────────
# • VideoJob         — added originalFileName field (present in RabbitMQ message).
# • AnalysisJobMessage — added trainDetailId, folderName, priority fields.
# • ViolationResult  — timestamp / originalVideoTimestamp are now float (seconds),
#                      matching the API schema. The HH:MM:SS display string is dropped
#                      from the outbound payload (Spring Boot derives it from the float).
# • VideoResult      — added originalS3Key (required by /internal/analysis/completed).
#                      videoId is serialised as a STRING per the API spec.
# • CompletionPayload — added batchId, trainDetailId, folderName fields required by
#                       the /internal/analysis/completed endpoint.
# """

# from __future__ import annotations

# from dataclasses import dataclass, field
# from typing import List, Optional


# # ── Inbound (from RabbitMQ) ──────────────────────────────────────────────────

# @dataclass
# class VideoJob:
#     """One video entry inside an AnalysisJobMessage."""
#     video_id:          int
#     sequence_no:       int
#     s3_key:            str            # e.g. "journeys/1/2026-06-10/.../original/001_front_cabin.mp4"
#     original_filename: str = ""       # e.g. "front_cabin.mp4"  (added per API spec)

#     @classmethod
#     def from_dict(cls, d: dict) -> "VideoJob":
#         return cls(
#             video_id          = int(d["videoId"]),
#             sequence_no       = int(d["sequenceNo"]),
#             s3_key            = d["s3Key"],
#             original_filename = d.get("originalFileName", ""),
#         )


# @dataclass
# class AnalysisJobMessage:
#     """
#     RabbitMQ message consumed from the 'analysis.jobs' queue.

#     Queue    : analysis.jobs
#     Exchange : dev.analysis.exchange
#     Routing  : dev.analysis.jobs.created

#     Example JSON
#     ────────────
#     {
#         "jobId":         "JOB-ABC123XYZ",
#         "journeyId":     10,
#         "trainDetailId": 1,
#         "folderName":    "journeys/1/2026-06-10/JRN-20260610-1-ABC123",
#         "priority":      "NORMAL",
#         "videos": [
#             {
#                 "videoId":           1,
#                 "sequenceNo":        1,
#                 "originalFileName":  "front_cabin.mp4",
#                 "s3Key":             "journeys/1/2026-06-10/.../original/001_front_cabin.mp4"
#             }
#         ]
#     }
#     """
#     job_id:          str
#     journey_id:      int
#     train_detail_id: int              # NEW — needed in completion payload
#     folder_name:     str              # NEW — S3 folder prefix for this journey
#     videos:          List[VideoJob]
#     priority:        str = "NORMAL"   # NEW — NORMAL | HIGH

#     @classmethod
#     def from_dict(cls, d: dict) -> "AnalysisJobMessage":
#         return cls(
#             job_id          = d["jobId"],
#             journey_id      = int(d["journeyId"]),
#             train_detail_id = int(d.get("trainDetailId", 0)),
#             folder_name     = d.get("folderName", f"journeys/{d.get('journeyId', 0)}"),
#             videos          = [VideoJob.from_dict(v) for v in d.get("videos", [])],
#             priority        = d.get("priority", "NORMAL"),
#         )


# # ── Outbound (to Spring Boot) ────────────────────────────────────────────────

# @dataclass
# class ViolationResult:
#     """
#     One violation event.

#     API field mapping (POST /api/internal/analysis/completed)
#     ──────────────────────────────────────────────────────────
#     violationType         → canonical string: PHONE_USAGE | DROWSY | SEAT_ABSENCE …
#     severity              → CRITICAL | HIGH | MEDIUM | LOW
#     confidence            → 0.0 – 100.0 (percentage)
#     riskScore             → 0.0 – 100.0
#     timestamp             → float seconds from start of journey   ← was HH:MM:SS string
#     originalVideoTimestamp→ float seconds within the source video ← was "<file> HH:MM:SS"
#     framePaths            → list of S3 keys (NOT signed URLs)
#     """
#     violation_type:             str
#     severity:                   str
#     confidence:                 float
#     risk_score:                 float          # float per API spec (was int)
#     timestamp_seconds:          float          # journey-global seconds (float)
#     original_video_timestamp:   float          # local-video seconds (float)
#     frame_paths:                List[str] = field(default_factory=list)

#     def to_dict(self) -> dict:
#         return {
#             "violationType":          self.violation_type,
#             "severity":               self.severity,
#             "confidence":             round(self.confidence, 2),
#             "riskScore":              round(self.risk_score, 2),
#             "timestamp":              round(self.timestamp_seconds, 3),
#             "originalVideoTimestamp": round(self.original_video_timestamp, 3),
#             "framePaths":             self.frame_paths,
#         }


# @dataclass
# class VideoResult:
#     """
#     Per-video summary inside the completion payload.

#     API field mapping (POST /api/internal/analysis/completed)
#     ──────────────────────────────────────────────────────────
#     videoId        → STRING per API spec  (was int in previous version)
#     sequenceNo     → int
#     durationSeconds→ float
#     originalS3Key  → NEW — the s3Key from the inbound VideoJob
#     violations     → list of ViolationResult
#     """
#     video_id:         int
#     video_name:       str
#     sequence_no:      int
#     duration_seconds: float
#     original_s3_key:  str             # NEW — required by API
#     violations:       List[ViolationResult] = field(default_factory=list)

#     def to_dict(self) -> dict:
#         return {
#             "videoId":         str(self.video_id),   # API spec says STRING
#             "sequenceNo":      self.sequence_no,
#             "durationSeconds": round(self.duration_seconds, 3),
#             "originalS3Key":   self.original_s3_key,
#             "violations":      [v.to_dict() for v in self.violations],
#         }


# @dataclass
# class CompletionPayload:
#     """
#     Full payload for POST /api/internal/analysis/completed.

#     New required fields vs previous version
#     ────────────────────────────────────────
#     • batchId        — generated as "BATCH-<jobId>" if not supplied
#     • trainDetailId  — forwarded from the RabbitMQ message
#     • folderName     — forwarded from the RabbitMQ message
#     """
#     job_id:          str
#     journey_id:      int
#     train_detail_id: int
#     folder_name:     str
#     processing_time: int                         # wall-clock milliseconds
#     video_results:   List[VideoResult] = field(default_factory=list)
#     batch_id:        str = ""                    # auto-filled in to_dict if blank

#     def to_dict(self) -> dict:
#         return {
#             "jobId":          self.job_id,
#             "journeyId":      self.journey_id,
#             "batchId":        self.batch_id or f"BATCH-{self.job_id}",
#             "trainDetailId":  self.train_detail_id,
#             "folderName":     self.folder_name,
#             "processingTime": self.processing_time,
#             "videoResults":   [vr.to_dict() for vr in self.video_results],
#         }



"""

models.py

─────────

Dataclasses that represent the RabbitMQ job message and the response

structures expected by the Spring Boot completion / progress APIs.
 
Aligned with CVVRS API (Base: https://cvvrsrailway-api.sconexsoft.com/cvs)
 
Fixes from previous version

─────────────────────────────

• AnalysisJobMessage — added callbackBaseUrl field (sent by Spring Boot,

  used by consumer.py to route callbacks to the correct server).

• VideoResult — restored durationFormatted, fps, sizeMb fields that were

  accidentally removed. Spring Boot expects all three in the completion payload.

• VideoResult.to_dict() — serializes all fields including the restored ones.

"""
 
from __future__ import annotations
 
from dataclasses import dataclass, field

from typing import List, Optional
 

# ── Helpers ──────────────────────────────────────────────────────────────────

def _fmt_duration(seconds: float) -> str:
    """Convert a float number of seconds to an 'H:MM:SS' display string."""
    t  = int(seconds)
    hh = t // 3600
    mm = (t % 3600) // 60
    ss = t % 60
    return f"{hh}:{mm:02d}:{ss:02d}"


# ── Inbound (from RabbitMQ) ──────────────────────────────────────────────────
 
@dataclass

class VideoJob:

    """One video entry inside an AnalysisJobMessage."""

    video_id:          int

    sequence_no:       int

    s3_key:            str            # e.g. "journeys/1/2026-06-10/.../original/001_front_cabin.mp4"

    original_filename: str = ""       # e.g. "front_cabin.mp4"
 
    @classmethod

    def from_dict(cls, d: dict) -> "VideoJob":

        return cls(

            video_id          = int(d["videoId"]),

            sequence_no       = int(d["sequenceNo"]),

            s3_key            = d["s3Key"],

            original_filename = d.get("originalFileName", ""),

        )
 
 
@dataclass

class AnalysisJobMessage:

    """

    RabbitMQ message consumed from the 'analysis.jobs' queue.
 
    Queue    : analysis.jobs

    Exchange : analysis.exchange

    Routing  : analysis.jobs.created
 
    Example JSON

    ────────────

    {

        "jobId":           "JOB-ABC123XYZ",

        "journeyId":       10,

        "trainDetailId":   1,

        "folderName":      "journeys/1/2026-06-10/JRN-20260610-1-ABC123",

        "callbackBaseUrl": "https://cvvrsrailway-api.sconexsoft.com/cvs/api/internal/analysis",

        "priority":        "NORMAL",

        "videos": [

            {

                "videoId":           1,

                "sequenceNo":        1,

                "originalFileName":  "front_cabin.mp4",

                "s3Key":             "journeys/1/2026-06-10/.../original/001_front_cabin.mp4"

            }

        ]

    }

    """

    job_id:            str

    journey_id:        int

    train_detail_id:   int

    folder_name:       str

    videos:            List[VideoJob]

    callback_base_url: str = ""       # FIX: was missing — Spring Boot sends this

    priority:          str = "NORMAL"
 
    @classmethod

    def from_dict(cls, d: dict) -> "AnalysisJobMessage":

        return cls(

            job_id            = d["jobId"],

            journey_id        = int(d["journeyId"]),

            train_detail_id   = int(d.get("trainDetailId", 0)),

            folder_name       = d.get("folderName", f"journeys/{d.get('journeyId', 0)}"),

            callback_base_url = d.get("callbackBaseUrl", ""),   # FIX

            videos            = [VideoJob.from_dict(v) for v in d.get("videos", [])],

            priority          = d.get("priority", "NORMAL"),

        )
 
 
# ── Outbound (to Spring Boot) ────────────────────────────────────────────────
 
@dataclass

class ViolationResult:

    """

    One violation event.
 
    API field mapping (POST /api/internal/analysis/completed)

    ──────────────────────────────────────────────────────────

    violationType          → canonical string: PHONE_USAGE | DROWSY | SEAT_ABSENCE ...

    severity               → CRITICAL | HIGH | MEDIUM | LOW

    confidence             → 0.0 – 100.0 (percentage)

    riskScore              → 0.0 – 100.0

    timestamp              → float seconds from start of journey (global)

    originalVideoTimestamp → float seconds within the source video (local)

    framePaths             → list of S3 keys (NOT signed URLs)

    """

    violation_type:             str

    severity:                   str

    confidence:                 float

    risk_score:                 float

    timestamp_seconds:          float          # journey-global seconds

    original_video_timestamp:   float          # local-video seconds
    duration_seconds:           float = 0.0
    # NEW — additive: the TRUE trigger→end duration (e.g. triggered at
    # 10.33, ended at 15.33 → 5.0), only when the pipeline actually
    # observed the violation end within the video. None if it was still
    # ongoing when the video ended — duration_seconds above remains the
    # only duration figure available in that case, exactly as before.
    trigger_duration_seconds:   Optional[float] = None

    frame_paths:                List[str] = field(default_factory=list)
 
    def to_dict(self) -> dict:

        return {

            "violationType":          self.violation_type,

            "severity":               self.severity,

            "confidence":             round(self.confidence, 2),

            "riskScore":              round(self.risk_score, 2),

            "timestamp":              self.timestamp_seconds if isinstance(self.timestamp_seconds, str) else _fmt_duration(self.timestamp_seconds),

            "originalVideoTimestamp": self.original_video_timestamp if isinstance(self.original_video_timestamp, str) else _fmt_duration(self.original_video_timestamp),

            "durationSeconds":        round(self.duration_seconds, 2),

            "triggerDurationSeconds": (
                round(self.trigger_duration_seconds, 2)
                if self.trigger_duration_seconds is not None else None
            ),

            "framePaths":             self.frame_paths,
          

        }
 
 
@dataclass

class VideoResult:

    """

    Per-video summary inside the completion payload.
 
    API field mapping (POST /api/internal/analysis/completed)

    ──────────────────────────────────────────────────────────

    videoId           → STRING per API spec

    sequenceNo        → int

    videoName         → original file name e.g. "front_cabin.mp4"

    durationSeconds   → float

    durationFormatted → "H:MM:SS"   FIX: was missing

    fps               → float       FIX: was missing

    sizeMb            → float       FIX: was missing

    originalS3Key     → the s3Key from the inbound VideoJob

    violations        → list of ViolationResult

    """

    video_id:           int

    video_name:         str

    sequence_no:        int

    duration_seconds:   float

    original_s3_key:    str

    violations:         List[ViolationResult] = field(default_factory=list)
 
    # FIX: restored missing fields

    duration_formatted: str   = ""

    fps:                float = 0.0

    size_mb:            float = 0.0
 
    def to_dict(self) -> dict:

        return {

            "videoId":           str(self.video_id),   # API spec says STRING

            "sequenceNo":        self.sequence_no,

            "videoName":         self.video_name,

            "durationSeconds":   round(self.duration_seconds, 3),

            "durationFormatted": self.duration_formatted,

            "sizeMb":            round(self.size_mb, 2),

            "originalS3Key":     self.original_s3_key,

            "violations":        [v.to_dict() for v in self.violations],

        }
 
 
@dataclass

class CompletionPayload:

    """

    Full payload for POST /api/internal/analysis/completed.

    """

    job_id:          str

    journey_id:      int

    train_detail_id: int

    folder_name:     str

    processing_time: int                         # wall-clock milliseconds

    video_results:   List[VideoResult] = field(default_factory=list)

    batch_id:        str = ""                    # auto-filled in to_dict if blank
 
    def to_dict(self) -> dict:

        return {

            "jobId":          self.job_id,

            "journeyId":      self.journey_id,

            "batchId":        self.batch_id or f"BATCH-{self.job_id}",

            "trainDetailId":  self.train_detail_id,

            "folderName":     self.folder_name,

            "processingTime": self.processing_time,

            "videoResults":   [vr.to_dict() for vr in self.video_results],

        }