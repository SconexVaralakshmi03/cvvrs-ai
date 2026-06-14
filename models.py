"""
models.py
─────────
Dataclasses that represent the RabbitMQ job message and the response
structures expected by the Spring Boot completion / progress APIs.

These are pure data holders — no I/O, no ML logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ── Inbound (from RabbitMQ) ──────────────────────────────────────────────────

@dataclass
class VideoJob:
    """One video entry inside an AnalysisJobMessage."""
    video_id:    int
    sequence_no: int
    s3_key:      str     # e.g. "journeys/101/original/video1.mp4"

    @classmethod
    def from_dict(cls, d: dict) -> "VideoJob":
        return cls(
            video_id    = int(d["videoId"]),
            sequence_no = int(d["sequenceNo"]),
            s3_key      = d["s3Key"],
        )


@dataclass
class AnalysisJobMessage:
    """
    RabbitMQ message consumed from the 'analysis.jobs' queue.

    Example JSON
    ────────────
    {
        "jobId":     "JOB123",
        "journeyId": 101,
        "videos": [
            {"videoId": 1001, "sequenceNo": 1, "s3Key": "journeys/101/original/video1.mp4"},
            {"videoId": 1002, "sequenceNo": 2, "s3Key": "journeys/101/original/video2.mp4"}
        ]
    }
    """
    job_id:     str
    journey_id: int
    videos:     List[VideoJob]

    @classmethod
    def from_dict(cls, d: dict) -> "AnalysisJobMessage":
        return cls(
            job_id     = d["jobId"],
            journey_id = int(d["journeyId"]),
            videos     = [VideoJob.from_dict(v) for v in d.get("videos", [])],
        )


# ── Outbound (to Spring Boot) ────────────────────────────────────────────────

@dataclass
class ViolationResult:
    """
    One violation event to be persisted as a ViolationEvent + AnalysisFrame
    by Spring Boot.

    Fields
    ──────
    violation_type  : canonical type string, e.g. "PHONE_USAGE", "SEAT_ABSENCE",
                      "DROWSINESS"
    severity        : "HIGH" | "MEDIUM" | "LOW"
    confidence      : 0.0 – 100.0  (percentage)
    risk_score      : 0 – 100
    timestamp       : HH:MM:SS display string  (global journey time)
    timestamp_seconds: int  — seconds from start of journey
    original_video_timestamp : "<filename> HH:MM:SS"  — local file + local time
    frame_paths     : list of S3 keys (NOT signed URLs)
    """
    violation_type:            str
    severity:                  str
    confidence:                float
    risk_score:                int
    timestamp:                 str
    timestamp_seconds:         int
    original_video_timestamp:  str
    frame_paths:               List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "violationType":           self.violation_type,
            "severity":                self.severity,
            "confidence":              round(self.confidence, 2),
            "riskScore":               self.risk_score,
            "timestamp":               self.timestamp,
            "timestampSeconds":        self.timestamp_seconds,
            "original_video_timestamp": self.original_video_timestamp,
            "framePaths":              self.frame_paths,
        }


@dataclass
class VideoResult:
    """
    Per-video summary sent inside the completion payload.

    Fields
    ──────
    video_id          : mirrors VideoJob.video_id
    video_name        : display filename
    sequence_no       : mirrors VideoJob.sequence_no
    duration_seconds  : float — duration of this video file
    duration_formatted: "H:MM:SS"
    fps               : frames per second
    size_mb           : file size in MB
    violations        : list of ViolationResult
    """
    video_id:           int
    video_name:         str
    sequence_no:        int
    duration_seconds:   float
    duration_formatted: str
    fps:                float
    size_mb:            float
    violations:         List[ViolationResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "videoId":           self.video_id,
            "video_name":        self.video_name,
            "sequenceNo":        self.sequence_no,
            "durationSeconds":   round(self.duration_seconds, 3),
            "durationFormatted": self.duration_formatted,
            "fps":               round(self.fps, 3),
            "sizeMb":            round(self.size_mb, 2),
            "violations":        [v.to_dict() for v in self.violations],
        }


@dataclass
class CompletionPayload:
    """
    Full payload for POST /api/internal/analysis/completed.
    """
    job_id:          str
    journey_id:      int
    processing_time: int           # wall-clock milliseconds
    video_results:   List[VideoResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "jobId":          self.job_id,
            "journeyId":      self.journey_id,
            "processingTime": self.processing_time,
            "videoResults":   [vr.to_dict() for vr in self.video_results],
        }