# """
# journey_log.py
# ──────────────
# Builds the human-readable "JOB ANALYSIS" .txt report for a journey and
# uploads it to S3 alongside the frames, under the same dynamic folder:

#     <folderName>/<jobId>.txt

# e.g. journeys/104/2026-06-19/JRN-20260619-104-011E57/JOB-262B2786AC81.txt

# Only fields that already exist on VideoResult / ViolationResult (see
# models.py) are used — no fabricated data (no raw/processed frame counts,
# no frame-skip, no report/frames paths, since those aren't tracked by the
# pipeline today).

# Timestamp rule
# ──────────────
# • Video 1 (first video in the journey): only the journey-global timestamp
#   is shown next to each violation, e.g.:
#       [00:00:21] MOBILE_PHONE_USAGE             Severity: CRITICAL

# • Video 2 onward: the original (local, within-that-video) timestamp is
#   shown in addition to the journey-global one, e.g.:
#       [00:12:15] (orig 00:00:01) SEAT_ABSENCE   Severity: CRITICAL

# Usage
# ─────
#     from journey_log import build_journey_log_text, upload_journey_log

#     text = build_journey_log_text(job_id, journey_id, video_results,
#                                    total_wall_seconds, started_at)
#     upload_journey_log(text, job_id, folder_name)
# """

# from __future__ import annotations

# import datetime as _dt
# from typing import List, Optional

# from models import VideoResult, ViolationResult

# LINE = "─" * 80
# DLINE = "═" * 80


# def _fmt_hms(value) -> str:
#     """
#     VideoResult/ViolationResult store timestamps as already-formatted
#     'H:MM:SS' strings (see analyzer.py / models.py to_dict()).
#     Accept either a string (pass through) or a float (format it).
#     """
#     if isinstance(value, str):
#         return value
#     t = int(value or 0)
#     hh = t // 3600
#     mm = (t % 3600) // 60
#     ss = t % 60
#     return f"{hh}:{mm:02d}:{ss:02d}"


# def _violation_summary(violations: List[ViolationResult]) -> dict:
#     counts = {"SEAT_ABSENCE": 0, "PHONE_USAGE": 0, "DROWSY": 0}
#     for v in violations:
#         if v.violation_type in counts:
#             counts[v.violation_type] += 1
#     return counts


# def _video_block(idx: int, total: int, vr: VideoResult, is_first: bool) -> str:
#     lines = []
#     width = 80
#     top    = "┌" + "─" * width + "┐"
#     bottom = "└" + "─" * width + "┘"
#     inner  = f" VIDEO {idx} OF {total}"
#     middle = "│" + inner + " " * (width - len(inner)) + "│"
#     lines.append(top)
#     lines.append(middle)
#     lines.append(bottom)
#     lines.append("")
#     lines.append(f"Video ID        : {vr.video_id}")
#     lines.append(f"Sequence        : {vr.sequence_no}")
#     lines.append(f"Filename        : {vr.video_name}")
#     lines.append(f"Duration        : {round(vr.duration_seconds, 1)} sec "
#                  f"({vr.duration_formatted or _fmt_hms(vr.duration_seconds)})")
#     if vr.fps:
#         lines.append(f"FPS             : {round(vr.fps, 1)}")
#     lines.append("")
#     lines.append("Status          : COMPLETED")
#     lines.append("")
#     lines.append("Detected Violations")
#     lines.append("-" * 80)

#     if vr.violations:
#         for v in v_sorted(vr.violations):
#             journey_ts = _fmt_hms(v.timestamp_seconds)
#             sev = v.severity
#             vtype = v.violation_type
#             if is_first:
#                 lines.append(f"[{journey_ts}] {vtype:<30} Severity: {sev}")
#             else:
#                 orig_ts = _fmt_hms(v.original_video_timestamp)
#                 lines.append(
#                     f"[{journey_ts}] (orig {orig_ts}) {vtype:<30} Severity: {sev}"
#                 )
#     else:
#         lines.append("(none)")
#     lines.append("-" * 80)
#     lines.append("")

#     counts = _violation_summary(vr.violations)
#     lines.append("Violation Summary")
#     lines.append("-" * 80)
#     lines.append(f"Seat Absence Events : {counts['SEAT_ABSENCE']}")
#     lines.append(f"Mobile Usage Events : {counts['PHONE_USAGE']}")
#     lines.append(f"Drowsiness Events   : {counts['DROWSY']}")
#     lines.append(f"Total Violations    : {len(vr.violations)}")
#     lines.append("")

#     return "\n".join(lines)


# def v_sorted(violations: List[ViolationResult]) -> List[ViolationResult]:
#     """Sort violations by journey-global timestamp for readable output."""
#     def _key(v: ViolationResult):
#         ts = v.timestamp_seconds
#         if isinstance(ts, str):
#             parts = ts.split(":")
#             try:
#                 if len(parts) == 3:
#                     return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
#                 return float(ts)
#             except Exception:
#                 return 0.0
#         return float(ts or 0.0)
#     return sorted(violations, key=_key)


# def build_journey_log_text(
#     job_id: str,
#     journey_id: int,
#     video_results: List[VideoResult],
#     total_wall_seconds: float,
#     started_at: Optional[_dt.datetime] = None,
# ) -> str:
#     """
#     Build the full text report for a journey, matching the
#     JOB ANALYSIS START / per-video blocks / JOURNEY SUMMARY format.
#     """
#     started_at = started_at or _dt.datetime.now()
#     n_videos = len(video_results)

#     out = []
#     out.append(DLINE)
#     out.append("JOB ANALYSIS START")
#     out.append(DLINE)
#     out.append(f"Job ID          : {job_id}")
#     out.append(f"Journey ID      : {journey_id}")
#     out.append(f"Analysis ID     : journey_{journey_id}")
#     out.append(f"Total Videos    : {n_videos}")
#     out.append(f"Started At      : {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
#     out.append(DLINE)
#     out.append("")
#     out.append("")

#     ordered = sorted(video_results, key=lambda v: v.sequence_no)
#     for idx, vr in enumerate(ordered, start=1):
#         is_first = (idx == 1)
#         out.append(_video_block(idx, n_videos, vr, is_first))
#         out.append("")

#     # ── Journey summary ──────────────────────────────────────────────────
#     all_violations = [v for vr in ordered for v in vr.violations]
#     totals = _violation_summary(all_violations)

#     out.append(DLINE)
#     out.append("JOURNEY SUMMARY")
#     out.append(DLINE)
#     out.append("")
#     out.append(f"Job ID              : {job_id}")
#     out.append(f"Journey ID          : {journey_id}")
#     out.append(f"Videos Processed    : {n_videos} / {n_videos}")
#     out.append("")
#     out.append("Violations")
#     out.append("-" * 80)
#     out.append(f"Mobile Usage        : {totals['PHONE_USAGE']}")
#     out.append(f"Seat Absence        : {totals['SEAT_ABSENCE']}")
#     out.append(f"Drowsiness          : {totals['DROWSY']}")
#     out.append(f"Total Violations    : {len(all_violations)}")
#     out.append("")
#     out.append("Performance")
#     out.append("-" * 80)
#     out.append(f"Total Runtime       : {round(total_wall_seconds, 2)} sec")
#     out.append("")
#     out.append("Final Status        : SUCCESS")
#     out.append("")
#     out.append(DLINE)
#     out.append("JOB ANALYSIS COMPLETED")
#     out.append(DLINE)

#     return "\n".join(out)


"""
journey_log.py
──────────────
Builds the human-readable "JOB ANALYSIS" .txt report for a journey and
uploads it to S3 alongside the frames, under the same dynamic folder:

    <folderName>/<jobId>.txt

e.g. journeys/104/2026-06-19/JRN-20260619-104-011E57/JOB-262B2786AC81.txt

Only fields that already exist on VideoResult / ViolationResult (see
models.py) are used — no fabricated data (no raw/processed frame counts,
no frame-skip, no report/frames paths, since those aren't tracked by the
pipeline today).

Timestamp rule
──────────────
• Video 1 (first video in the journey): only the journey-global timestamp
  is shown next to each violation, e.g.:
      [00:00:21] MOBILE_PHONE_USAGE             Severity: CRITICAL

• Video 2 onward: the original (local, within-that-video) timestamp is
  shown in addition to the journey-global one, e.g.:
      [00:12:15] (orig 00:00:01) SEAT_ABSENCE   Severity: CRITICAL

Usage
─────
    from journey_log import build_journey_log_text, upload_journey_log

    text = build_journey_log_text(job_id, journey_id, video_results,
                                   total_wall_seconds, started_at)
    upload_journey_log(text, job_id, folder_name)
"""

from __future__ import annotations

import datetime as _dt
from typing import List, Optional

from models import VideoResult, ViolationResult

LINE = "─" * 80
DLINE = "═" * 80


def _fmt_hms(value) -> str:
    """
    VideoResult/ViolationResult store timestamps as already-formatted
    'H:MM:SS' strings (see analyzer.py / models.py to_dict()).
    Accept either a string (pass through) or a float (format it).
    """
    if isinstance(value, str):
        return value
    t = int(value or 0)
    hh = t // 3600
    mm = (t % 3600) // 60
    ss = t % 60
    return f"{hh}:{mm:02d}:{ss:02d}"


def _violation_summary(violations: List[ViolationResult]) -> dict:
    counts = {"SEAT_ABSENCE": 0, "PHONE_USAGE": 0, "DROWSY": 0}
    for v in violations:
        if v.violation_type in counts:
            counts[v.violation_type] += 1
    return counts


def _video_block(idx: int, total: int, vr: VideoResult, is_first: bool,
                  failure_info: Optional[dict] = None) -> str:
    """
    failure_info, if provided, is a dict like:
        {"errorType": "RESOURCE_EXHAUSTION", "errorMessage": "...",
         "reason": "Not Processed - Worker Resource Exhaustion"}
    for this specific video, or None if the video succeeded.
    """
    lines = []
    width = 80
    top    = "┌" + "─" * width + "┐"
    bottom = "└" + "─" * width + "┘"
    inner  = f" VIDEO {idx} OF {total}"
    middle = "│" + inner + " " * (width - len(inner)) + "│"
    lines.append(top)
    lines.append(middle)
    lines.append(bottom)
    lines.append("")
    lines.append(f"Video ID        : {vr.video_id}")
    lines.append(f"Sequence        : {vr.sequence_no}")
    lines.append(f"Filename        : {vr.video_name}")
    lines.append(f"Duration        : {round(vr.duration_seconds, 1)} sec "
                 f"({vr.duration_formatted or _fmt_hms(vr.duration_seconds)})")
    if vr.fps:
        lines.append(f"FPS             : {round(vr.fps, 1)}")
    lines.append("")
    if failure_info:
        lines.append("Status          : FAILED")
        lines.append(f"Error Type      : {failure_info.get('errorType', 'UNKNOWN')}")
        lines.append(f"Error Message   : {failure_info.get('errorMessage', '')}")
        if failure_info.get("reason"):
            lines.append(f"Reason          : {failure_info['reason']}")
    else:
        lines.append("Status          : COMPLETED")
    lines.append("")
    lines.append("Detected Violations")
    lines.append("-" * 80)

    if vr.violations:
        for v in v_sorted(vr.violations):
            journey_ts = _fmt_hms(v.timestamp_seconds)
            sev = v.severity
            vtype = v.violation_type
            if is_first:
                lines.append(f"[{journey_ts}] {vtype:<30} Severity: {sev}")
            else:
                orig_ts = _fmt_hms(v.original_video_timestamp)
                lines.append(
                    f"[{journey_ts}] (orig {orig_ts}) {vtype:<30} Severity: {sev}"
                )
    else:
        lines.append("(none)")
    lines.append("-" * 80)
    lines.append("")

    counts = _violation_summary(vr.violations)
    lines.append("Violation Summary")
    lines.append("-" * 80)
    lines.append(f"Seat Absence Events : {counts['SEAT_ABSENCE']}")
    lines.append(f"Mobile Usage Events : {counts['PHONE_USAGE']}")
    lines.append(f"Drowsiness Events   : {counts['DROWSY']}")
    lines.append(f"Total Violations    : {len(vr.violations)}")
    lines.append("")

    return "\n".join(lines)


def v_sorted(violations: List[ViolationResult]) -> List[ViolationResult]:
    """Sort violations by journey-global timestamp for readable output."""
    def _key(v: ViolationResult):
        ts = v.timestamp_seconds
        if isinstance(ts, str):
            parts = ts.split(":")
            try:
                if len(parts) == 3:
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                return float(ts)
            except Exception:
                return 0.0
        return float(ts or 0.0)
    return sorted(violations, key=_key)


def build_journey_log_text(
    job_id: str,
    journey_id: int,
    video_results: List[VideoResult],
    total_wall_seconds: float,
    started_at: Optional[_dt.datetime] = None,
    failed_videos: Optional[dict] = None,
    video_error_details: Optional[dict] = None,
    journey_status: Optional[str] = None,
) -> str:
    """
    Build the full text report for a journey, matching the
    JOB ANALYSIS START / per-video blocks / JOURNEY SUMMARY format.

    failed_videos : Optional[Dict[video_id, error_message]] — same shape
                    analyze_journey() returns. When a video_id appears here,
                    its block is rendered with Status: FAILED instead of
                    COMPLETED, and the JOURNEY SUMMARY's Final Status
                    reflects partial/total failure instead of always SUCCESS.
    video_error_details : Optional[Dict[video_id, dict]] — richer detail per
                    failed video: {"errorType": ..., "reason": ...}. Falls
                    back to just the error_message from failed_videos if a
                    video_id has no entry here.
    journey_status : Optional explicit override for Final Status (e.g. when
                    consumer.py has already computed COMPLETED /
                    COMPLETED_WITH_ERRORS / FAILED via
                    callback_client.compute_journey_status). If omitted, it
                    is derived from failed_videos vs. video_results here.
    """
    started_at = started_at or _dt.datetime.now()
    n_videos = len(video_results)
    failed_videos = failed_videos or {}
    video_error_details = video_error_details or {}

    out = []
    out.append(DLINE)
    out.append("JOB ANALYSIS START")
    out.append(DLINE)
    out.append(f"Job ID          : {job_id}")
    out.append(f"Journey ID      : {journey_id}")
    out.append(f"Analysis ID     : journey_{journey_id}")
    out.append(f"Total Videos    : {n_videos}")
    out.append(f"Started At      : {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    out.append(DLINE)
    out.append("")
    out.append("")

    ordered = sorted(video_results, key=lambda v: v.sequence_no)
    for idx, vr in enumerate(ordered, start=1):
        is_first = (idx == 1)
        failure_info = None
        if vr.video_id in failed_videos:
            detail = video_error_details.get(vr.video_id, {})
            failure_info = {
                "errorType":    detail.get("errorType", "UNKNOWN"),
                "errorMessage": failed_videos[vr.video_id],
                "reason":       detail.get("reason"),
            }
        out.append(_video_block(idx, n_videos, vr, is_first, failure_info))
        out.append("")

    # ── Journey summary ──────────────────────────────────────────────────
    all_violations = [v for vr in ordered for v in vr.violations]
    totals = _violation_summary(all_violations)
    n_failed    = len(failed_videos)
    n_succeeded = n_videos - n_failed

    out.append(DLINE)
    out.append("JOURNEY SUMMARY")
    out.append(DLINE)
    out.append("")
    out.append(f"Job ID              : {job_id}")
    out.append(f"Journey ID          : {journey_id}")
    out.append(f"Videos Processed    : {n_succeeded} / {n_videos}")
    if n_failed:
        out.append(f"Videos Failed       : {n_failed} / {n_videos}")
    out.append("")
    out.append("Violations")
    out.append("-" * 80)
    out.append(f"Mobile Usage        : {totals['PHONE_USAGE']}")
    out.append(f"Seat Absence        : {totals['SEAT_ABSENCE']}")
    out.append(f"Drowsiness          : {totals['DROWSY']}")
    out.append(f"Total Violations    : {len(all_violations)}")
    out.append("")
    out.append("Performance")
    out.append("-" * 80)
    out.append(f"Total Runtime       : {round(total_wall_seconds, 2)} sec")
    out.append("")

    if journey_status is None:
        if n_failed == 0:
            journey_status = "SUCCESS"
        elif n_succeeded == 0:
            journey_status = "FAILED"
        else:
            journey_status = "COMPLETED_WITH_ERRORS"

    out.append(f"Final Status        : {journey_status}")
    out.append("")
    out.append(DLINE)
    out.append("JOB ANALYSIS COMPLETED")
    out.append(DLINE)

    return "\n".join(out)