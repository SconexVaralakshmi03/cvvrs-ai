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

    Routing  : analysis.job.created
 
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

    frame_paths:                List[str] = field(default_factory=list)
 
    def to_dict(self) -> dict:

        return {

            "violationType":          self.violation_type,

            "severity":               self.severity,

            "confidence":             round(self.confidence, 2),

            "riskScore":              round(self.risk_score, 2),

            "timestamp":              self.timestamp_seconds if isinstance(self.timestamp_seconds, str) else _fmt_duration(self.timestamp_seconds),

            "originalVideoTimestamp": self.original_video_timestamp if isinstance(self.original_video_timestamp, str) else _fmt_duration(self.original_video_timestamp),

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

            "fps":               round(self.fps, 3),

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
 