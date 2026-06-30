from __future__ import annotations
 
import json

from datetime import datetime
 
from flask import Flask, request, jsonify
 
app = Flask(__name__)
 
# ── Expected fields for validation ───────────────────────────────────────────

_REQUIRED_VIDEO_FIELDS     = ["videoId", "sequenceNo", "videoName",

                               "durationSeconds", "durationFormatted",

                               "fps", "sizeMb", "originalS3Key"]

_REQUIRED_VIOLATION_FIELDS = ["violationType", "severity", "confidence",

                               "riskScore", "timestamp",

                               "originalVideoTimestamp", "framePaths"]
 
 
def _ts() -> str:

    return datetime.now().strftime("%H:%M:%S")
 
 
def _validate_fields(obj: dict, required: list, label: str) -> None:

    """Print a warning for any missing field."""

    for field in required:

        if field not in obj or obj[field] is None:

            print(f"           ⚠️  MISSING FIELD [{label}]: {field}")
 
 
# ── Routes ────────────────────────────────────────────────────────────────────
 
@app.post("/api/internal/analysis/progress")

def progress():

    data = request.get_json()

    current_video = data.get("currentVideo", "")

    cv_str = f"  currentVideo={current_video}" if current_video else ""

    print(

        f"[{_ts()}] PROGRESS  "

        f"jobId={data.get('jobId')}  "

        f"journeyId={data.get('journeyId')}  "

        f"progress={data.get('progress')}%  "

        f"status={data.get('status')}"

        f"{cv_str}  "

        f"message={data.get('message', '')}"

    )

    return jsonify({"status": "ok"}), 200
 
 
@app.post("/api/internal/analysis/completed")

def completed():

    data    = request.get_json()

    job_id  = data.get("jobId", "unknown")

    results = data.get("videoResults", [])
 
    print(f"\n[{_ts()}] ── COMPLETED ──────────────────────────────────────────────")

    print(f"  jobId          = {job_id}")

    print(f"  journeyId      = {data.get('journeyId')}")

    print(f"  trainDetailId  = {data.get('trainDetailId')}")

    print(f"  folderName     = {data.get('folderName')}")

    print(f"  processingTime = {data.get('processingTime')} ms")

    print(f"  batchId        = {data.get('batchId')}")

    print(f"  videos         = {len(results)}")
 
    total_violations = 0
 
    for vr in results:

        print(f"\n  ── Video seq={vr.get('sequenceNo')} ─────────────────────────────")
 
        # Validate required video fields

        _validate_fields(vr, _REQUIRED_VIDEO_FIELDS, f"video seq={vr.get('sequenceNo')}")
 
        print(f"     videoId          = {vr.get('videoId')}")

        print(f"     videoName        = {vr.get('videoName')}")

        print(f"     durationSeconds  = {vr.get('durationSeconds')}")

        print(f"     durationFormatted= {vr.get('durationFormatted')}")

        print(f"     fps              = {vr.get('fps')}")

        print(f"     sizeMb           = {vr.get('sizeMb')}")

        print(f"     originalS3Key    = {vr.get('originalS3Key', '')[:60]}...")
 
        violations = vr.get("violations", [])

        total_violations += len(violations)

        print(f"     violations       = {len(violations)}")
 
        for i, v in enumerate(violations):

            # Validate required violation fields

            _validate_fields(v, _REQUIRED_VIOLATION_FIELDS,

                             f"video seq={vr.get('sequenceNo')} violation[{i}]")
 
            print(

                f"       [{i}] type={v.get('violationType')}  "

                f"severity={v.get('severity')}  "

                f"confidence={v.get('confidence')}  "

                f"riskScore={v.get('riskScore')}"

            )

            print(

                f"            timestamp={v.get('timestamp')}  "

                f"originalVideoTimestamp={v.get('originalVideoTimestamp')}"

            )

            frames = v.get("framePaths", [])

            print(f"            framePaths={len(frames)} frame(s)")

            for fp in frames:

                print(f"              → {fp}")
 
    print(f"\n  TOTAL violations = {total_violations}")

    print(f"─────────────────────────────────────────────────────────────────────\n")
 
    # Save full payload to disk for inspection

    out_file = f"mock_completion_{job_id}.json"

    with open(out_file, "w", encoding="utf-8") as f:

        json.dump(data, f, indent=2)

    print(f"  Full payload saved → {out_file}\n")
 
    return jsonify({"status": "ok"}), 200
 
 
@app.post("/api/internal/analysis/failed")

def failed():

    data = request.get_json()

    print(f"\n[{_ts()}] ── FAILED ───────────────────────────────────────────────────")

    print(f"  jobId      = {data.get('jobId')}")

    print(f"  error      = {data.get('errorMessage', '')[:500]}")

    print(f"─────────────────────────────────────────────────────────────────────\n")

    return jsonify({"status": "ok"}), 200
 
 
# ── Startup ───────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":

    print("=" * 70)

    print("  Mock Spring Boot server running on http://localhost:9000")

    print("  Endpoints:")

    print("    POST /api/internal/analysis/progress")

    print("    POST /api/internal/analysis/completed")

    print("    POST /api/internal/analysis/failed")

    print()

    print("  To use with consumer.py, set in config/credentials.env:")

    print("    SPRING_BOOT_BASE_URL=http://localhost:9000")

    print("=" * 70)

    app.run(host="0.0.0.0", port=9000, debug=False)
 