"""
mock_spring_boot.py
────────────────────
A tiny local stand-in for the Spring Boot backend's internal callback
endpoints. Use this to test consumer.py / analyzer.py end-to-end without
needing the real backend running.

Run:
    python mock_spring_boot.py

It listens on http://localhost:9000 and prints every payload it receives,
plus saves the completion payload to mock_completion_<jobId>.json so you
can inspect the full violation/frame results.
"""

from __future__ import annotations

import json
from datetime import datetime

from flask import Flask, request, jsonify

app = Flask(__name__)


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


@app.post("/api/internal/analysis/progress")
def progress():
    data = request.get_json()
    print(f"[{_ts()}] PROGRESS  {data}")
    return jsonify({"status": "ok"}), 200


@app.post("/api/internal/analysis/completed")
def completed():
    data = request.get_json()
    job_id = data.get("jobId", "unknown")
    print(f"[{_ts()}] COMPLETED  jobId={job_id}  journeyId={data.get('journeyId')}  "
          f"processingTime={data.get('processingTime')}ms")

    n_violations = sum(len(vr.get("violations", [])) for vr in data.get("videoResults", []))
    print(f"           videos={len(data.get('videoResults', []))}  total_violations={n_violations}")

    out_file = f"mock_completion_{job_id}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"           Full payload saved → {out_file}")

    return jsonify({"status": "ok"}), 200


@app.post("/api/internal/analysis/failed")
def failed():
    data = request.get_json()
    print(f"[{_ts()}] FAILED  jobId={data.get('jobId')}  journeyId={data.get('journeyId')}")
    print(f"           error: {data.get('errorMessage', '')[:500]}")
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    print("Mock Spring Boot server running on http://localhost:9000")
    print("Endpoints:")
    print("  POST /api/internal/analysis/progress")
    print("  POST /api/internal/analysis/completed")
    print("  POST /api/internal/analysis/failed")
    app.run(host="0.0.0.0", port=9000, debug=False)