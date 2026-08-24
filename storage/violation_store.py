

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

OUTPUTS_ROOT = "outputs"
MERGE_WINDOW = 2.0

# ── Merge/dedup family mapping ────────────────────────────────────────────
#
# FIX (unrelated violations silently absorbing each other's risk_score):
# _deduplicate_by_frame()/_merge_by_time_window() below used to group ANY
# violations that landed on the same frame_index / within MERGE_WINDOW
# seconds of each other, REGARDLESS of event type (only hand_raise had a
# pilot_id carve-out). That let e.g. a "curve_checking" snapshot (risk_score
# 0, LOW) sitting within 2s of an unrelated "drowsy"/"sleeping" detection
# (risk_score 75) absorb that risk_score and get reported as MEDIUM — while
# the drowsy candidate's own `type` was overwritten with "curve_checking"
# (group[0].type wins) and its evidence frame discarded, so it never even
# reached the LLM verification loop in analyzer.py as its own distinct
# "drowsy" candidate. Two genuinely unrelated detector outputs should never
# blend into one report line.
#
# Merging/dedup is now gated by "family": violations only ever
# merge/dedup with OTHER violations in the SAME family, never across
# families. Most event types are their own single-member family (the
# `.get(event_type, event_type)` fallback below). The one deliberate
# exception is "drowsy"/"sleeping"/"sleeping_absent" — these three are NOT
# unrelated detectors; they're the SAME head-droop/PERCLOS detector's
# evolving severity label for one continuous episode (main.py picks
# whichever of the three applies on a given cycle as the episode
# progresses — see the also_absent/is_sleeping branch there), so keeping
# them mergeable preserves the original "one sustained episode → one
# report line" behaviour for that specific case. Every other pair of event
# types (curve_checking, hand_raise, seat_absence, phone_use, drowsy-family)
# is always a different family and can never merge/dedup with each other,
# no matter how close in time or how identical the frame_index.
_MERGE_FAMILIES: Dict[str, str] = {
    "drowsy":          "drowsiness_family",
    "sleeping":        "drowsiness_family",
    "sleeping_absent": "drowsiness_family",
}


def _merge_family(event_type: str) -> str:
    """Family key used to gate BOTH _deduplicate_by_frame() and
    _merge_by_time_window() — see _MERGE_FAMILIES note above."""
    return _MERGE_FAMILIES.get(event_type, event_type)


# JPEG quality used for in-memory violation frame storage (see _encode_frame).
# 90 is visually near-lossless for evidence purposes while cutting a raw
# 1280x720 BGR frame (~2.76MB) down to roughly 150-300KB (~10-18x smaller).
_FRAME_JPEG_QUALITY = 90


# ── Memory-footprint fix ─────────────────────────────────────────────────────
# _Violation.annotated_frame used to hold a raw uncompressed numpy array for
# every candidate violation, for every video, for the ENTIRE journey (only
# freed once, in a pass that runs after ALL videos in the journey have been
# processed — see analyzer.py::analyze_journey / extract_violation_frames
# below). On a journey with many violations across several videos this meant
# peak RAM grew with the TOTAL violation count across the whole journey, not
# per-video — a real contributor to OOM/resource-exhaustion aborts on later
# videos in a journey (e.g. journey stops partway through, remaining videos
# marked "Not Processed - Worker Resource Exhaustion").
#
# Fix: store the JPEG-encoded bytes instead of the raw array. Encoding is
# cheap (a few ms) and the size reduction (~10-18x) directly shrinks the
# amount of RAM held per in-flight violation. Decode on demand only where a
# raw array is actually needed (upload, LLM verification, resize-before-save).
def _encode_frame(frame: "np.ndarray") -> bytes:
    """JPEG-encode a BGR frame for compact in-memory violation storage."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _FRAME_JPEG_QUALITY])
    if not ok:
        raise ValueError("Failed to JPEG-encode frame for violation storage")
    return buf.tobytes()


def _decode_frame(data: bytes) -> "np.ndarray":
    """Decode a JPEG byte buffer (from _encode_frame) back to a BGR numpy array."""
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


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
    # NOTE: holds JPEG-ENCODED BYTES (see _encode_frame/_decode_frame near the
    # top of this file), not a raw numpy array — this is the memory-footprint
    # fix (was ~2.76MB/frame raw, now ~150-300KB/frame). Decode with
    # _decode_frame() before any pixel operation (resize, upload, LLM call).
    annotated_frame: Optional[bytes]      = None
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
                # Encode to JPEG bytes rather than keep a raw array copy —
                # see _encode_frame() docstring near the top of this file.
                annotated_frame  = (
                    _encode_frame(annotated_frame) if annotated_frame is not None else None
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

        from detectors.rsl_hand_brake_verifier import extract_signal_frame, verify_rsl_hand_brake

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
                # this violation, no re-seek involved. annotated_frame is
                # stored as JPEG bytes (see _encode_frame) — decode before
                # handing pixels to verify_rsl_hand_brake().
                signal_frame = _decode_frame(v.annotated_frame)
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
                # Encode to JPEG bytes rather than store a raw array copy —
                # see _encode_frame() docstring near the top of this file.
                annotated_frame = _encode_frame(verdict["best_frame"]),
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
        #
        # FIX (unrelated violations on the exact same frame_index merging):
        # the key must also include the merge-family (see _merge_family()
        # above) — two DIFFERENT detectors that happen to both fire on the
        # exact same global frame_index (e.g. curve_checking and drowsy on
        # the same frame) must stay separate violations, not collapse into
        # one with a borrowed risk_score. Same-family events (e.g. two
        # "drowsy" cycles, or drowsy→sleeping transitions) still dedup
        # together as before.
        unique: Dict[tuple, _Violation] = {}
        for v in self._violations:
            key = (
                v.source_filename,
                v.frame_index,
                getattr(v, "pilot_id", None),
                _merge_family(v.type),
            )
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

            # FIX (unrelated violations absorbing each other's risk_score/
            # type/evidence frame): a violation only extends the current
            # group if it's also in the SAME merge-family (see
            # _merge_family()/_MERGE_FAMILIES above) as what's already in
            # the group. Two DIFFERENT detectors — e.g. curve_checking and
            # drowsy — landing within MERGE_WINDOW of each other used to
            # get folded into one violation, with the later/higher-risk one
            # silently overriding the earlier one's risk_score while its
            # own `type`/evidence frame were discarded entirely (it never
            # even reached the LLM verification loop as its own distinct
            # candidate). Now they always stay separate violations, each
            # independently verified. Same-family events (e.g. drowsy →
            # sleeping as one episode's severity escalates) still merge
            # together as before — that's the one deliberate exception, see
            # _MERGE_FAMILIES' docstring.
            same_family = _merge_family(v.type) == _merge_family(group[-1].type)

            if (
                same_family
                and same_pilot_or_untagged
                and abs(v.timestamp - group[-1].timestamp) <= MERGE_WINDOW
            ):
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
                # annotated_frame is JPEG bytes (see _encode_frame) —
                # decode before _save_frame(), which resizes/re-writes it.
                v.frame_path      = self._save_frame(_decode_frame(v.annotated_frame),
                                                      v.events, v.time_str, v.frame_index)
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