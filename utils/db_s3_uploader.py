

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3

try:
    import psycopg2
    from psycopg2.extras import execute_values  # type: ignore[import]
except ImportError:
    psycopg2 = None        # type: ignore[assignment]
    execute_values = None  # type: ignore[assignment]


# LOAD CREDENTIALS FROM config/credentials.env


def _load_env(env_path: str) -> None:
    path = Path(env_path)
    if not path.is_file():
        raise FileNotFoundError(f"Credentials file not found: {env_path}")
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key   = key.strip()
            value = value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


_CREDENTIALS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "credentials.env",
)
_load_env(_CREDENTIALS_FILE)


def _db_config() -> Dict[str, Any]:
    return {
        "host":     os.environ["DB_HOST"],
        "port":     int(os.environ.get("DB_PORT", 5432)),
        "database": os.environ["DB_NAME"],
        "user":     os.environ["DB_USER"],
        "password": os.environ["DB_PASSWORD"],
    }


def _aws_config() -> Dict[str, str]:
    return {
        "aws_access_key_id":     os.environ["AWS_ACCESS_KEY_ID"],
        "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
        "region_name":           os.environ.get("AWS_REGION", "ap-south-1"),
    }


def _s3_bucket() -> str:
    return os.environ["S3_BUCKET"]


OUTPUTS_ROOT = "outputs"


# S3 UPLOADER

def _s3_client():
    return boto3.client("s3", **_aws_config())


def upload_frames_to_s3(
    frames_dir:  str,
    analysis_id: str,
) -> Dict[str, str]:
    client: Any = _s3_client()
    bucket      = _s3_bucket()
    aws         = _aws_config()
    url_map: Dict[str, str] = {}

    if not os.path.isdir(frames_dir):
        print(f"[S3] frames_dir not found: {frames_dir}")
        return url_map

    jpg_files = [f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")]
    print(f"[S3] Uploading {len(jpg_files)} frame(s) for analysis '{analysis_id}' ...")

    for fname in jpg_files:
        local_path = os.path.join(frames_dir, fname)
        s3_key     = f"violations/{analysis_id}/frames/{fname}"
        try:
            client.upload_file(
                local_path, bucket, s3_key,
                ExtraArgs={"ContentType": "image/jpeg"},
            )
            s3_url = (
                f"https://{bucket}.s3.{aws['region_name']}"
                f".amazonaws.com/{s3_key}"
            )
            rel_path          = os.path.join(analysis_id, "frames", fname)
            url_map[rel_path] = s3_url
            print(f"[S3]{fname}")
        except Exception as exc:
            print(f"[S3]{fname}  —  {exc}")

    print(f"[S3] Done. {len(url_map)}/{len(jpg_files)} uploaded.")
    return url_map


def upload_json_report_to_s3(report_path: str, analysis_id: str) -> Optional[str]:
    client = _s3_client()
    bucket = _s3_bucket()
    aws    = _aws_config()
    s3_key = f"violations/{analysis_id}/analysis_report.json"
    try:
        client.upload_file(
            report_path, bucket, s3_key,
            ExtraArgs={"ContentType": "application/json"},
        )
        url = (
            f"https://{bucket}.s3.{aws['region_name']}"
            f".amazonaws.com/{s3_key}"
        )
        print(f"[S3] JSON report uploaded → {url}")
        return url
    except Exception as exc:
        print(f"[S3] JSON report upload failed: {exc}")
        return None


# DATABASE WRITER

def _get_conn():
    if psycopg2 is None:
        raise ImportError("psycopg2 is not installed")
    return psycopg2.connect(**_db_config())


def save_analysis_to_db(
    report:     Dict[str, Any],
    s3_url_map: Dict[str, str],
) -> None:
    """
    Write into your 4 existing tables:
      public.video_info         <- one upsert per run (includes processing_time)
      public.violations         <- one row per violation
      public.violation_events   <- one row per event label
      public.violation_factors  <- one row per factor
    """
    conn = _get_conn()
    try:
        analysis_id      = report["analysis_id"]
        train_detail_id  = report.get("train_detail_id", 0)
        processing_time  = report.get("processing_time", 0.0)
        vi               = report.get("video_info", {})
        violations       = report.get("violations", [])

        with conn.cursor() as cur:

            # 1. Upsert into video_info (with processing_time) 
            cur.execute(
                """
                INSERT INTO public.video_info
                    (analysis_id, train_detail_id, filename, video_path,
                     duration_seconds, duration_formatted, resolution,
                     fps, total_frames, size_mb, processing_time)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (analysis_id) DO UPDATE SET
                    train_detail_id    = EXCLUDED.train_detail_id,
                    filename           = EXCLUDED.filename,
                    video_path         = EXCLUDED.video_path,
                    duration_seconds   = EXCLUDED.duration_seconds,
                    duration_formatted = EXCLUDED.duration_formatted,
                    resolution         = EXCLUDED.resolution,
                    fps                = EXCLUDED.fps,
                    total_frames       = EXCLUDED.total_frames,
                    size_mb            = EXCLUDED.size_mb,
                    processing_time    = EXCLUDED.processing_time;
                """,
                (
                    analysis_id,
                    train_detail_id,
                    vi.get("filename",          ""),
                    vi.get("videoPath",         ""),
                    vi.get("durationSeconds",   0),
                    vi.get("durationFormatted", "0:00:00"),
                    vi.get("resolution",        ""),
                    vi.get("fps",               0),
                    vi.get("totalFrames",       0),
                    vi.get("sizeMb",            0),
                    processing_time,                    # ← processing_time stored here
                ),
            )
            print(f"[DB] video_info upserted for '{analysis_id}'  "
                  f"(processing_time={processing_time:.3f}s)")

            # 2. Insert each violation + events + factors
            for v in violations:
                frame_local         = v.get("frame_path") or ""
                frame_s3            = s3_url_map.get(frame_local, "")
                frame_path_to_store = frame_s3 if frame_s3 else frame_local

                cur.execute(
                    """
                    INSERT INTO public.violations
                        (analysis_id, "timestamp", frame_index,
                         severity, duration, risk_score, risk_level,
                         confidence, frame_path)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (
                        analysis_id,
                        v.get("timestamp",   "00:00:00"),
                        v.get("frame_index", 0),
                        v.get("severity",    "CRITICAL"),
                        v.get("duration",    0),
                        v.get("risk_score",  0),
                        v.get("risk_level",  "CRITICAL"),
                        v.get("confidence",  0),
                        frame_path_to_store,
                    ),
                )
                violation_id = cur.fetchone()[0]

                # violation_events
                events: List[str] = v.get("events", [])
                if events:
                    execute_values(
                        cur,
                        "INSERT INTO public.violation_events "
                        "(violation_id, event) VALUES %s",
                        [(violation_id, e) for e in events],
                    )

                # violation_factors
                factors: List[str] = v.get("factors", [])
                if factors:
                    execute_values(
                        cur,
                        "INSERT INTO public.violation_factors "
                        "(violation_id, factor) VALUES %s",
                        [(violation_id, f) for f in factors],
                    )

            print(f"[DB] {len(violations)} violation(s) inserted.")

        conn.commit()
        print("[DB] Commit successful.")

    except Exception as exc:
        conn.rollback()
        print(f"[DB] ERROR — rolled back: {exc}")
        raise
    finally:
        conn.close()


# CONVENIENCE WRAPPER

def finalize_and_upload(
    report_path:     str,
    analysis_id:     str,
    train_detail_id: int = 0,
) -> None:
    print("\n" + "=" * 60)
    print("  POST-PROCESSING: S3 Upload + Database Save")
    print("=" * 60)

    if not os.path.isfile(report_path):
        print(f"[Upload] Report file not found: {report_path}")
        return

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    # 1. Upload frames to S3
    frames_dir = os.path.join(OUTPUTS_ROOT, analysis_id, "frames")
    s3_url_map = upload_frames_to_s3(frames_dir, analysis_id)

    # 2. Upload JSON report to S3
    upload_json_report_to_s3(report_path, analysis_id)

    # 3. Save to DB (processing_time comes from report JSON)
    save_analysis_to_db(report, s3_url_map)

    print("=" * 60)
    print("  Upload + DB save complete.")
    print("=" * 60 + "\n")