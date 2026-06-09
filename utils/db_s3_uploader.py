from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# ── Credentials ────────────────────────────────────────────────────────────────
_ENV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "config", "credentials.env",
)
load_dotenv(_ENV_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_db_connection():
    return psycopg2.connect(
        host     = os.environ["DB_HOST"],
        port     = int(os.environ.get("DB_PORT", 5432)),
        dbname   = os.environ["DB_NAME"],
        user     = os.environ["DB_USER"],
        password = os.environ["DB_PASSWORD"],
    )


def get_pending_videos() -> List[Dict[str, Any]]:
    """
    Return all videos belonging to the SINGLE oldest pending folder.

    Query 1 — find the one folder_name with the smallest upload_timestamp
              among all rows with process_flag = 'N'  (LIMIT 1 on the folder).
    Query 2 — fetch every video in that folder ordered by seq_no.

    One folder per call → api.py while-loop processes folders one at a time
    in upload_timestamp ASC order, each folder starting fresh with its own
    time/frame offsets (timestamps reset to 00:00:00 for each new folder).
    """
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

            # Query 1 — pick the single oldest pending folder
            cur.execute(
                """
                SELECT   folder_name
                FROM     video_files
                WHERE    process_flag = 'N'
                ORDER BY upload_timestamp ASC
                LIMIT    1
                """
            )
            row = cur.fetchone()
            if row is None:
                print("[DB] No pending folders found (process_flag='N')")
                return []

            target_folder = row["folder_name"]
            print(f"[DB] Oldest pending folder: '{target_folder}'")

            # Query 2 — all videos in that folder in seq_no order
            cur.execute(
                """
                SELECT id, train_detail_id, folder_name,
                       filename, s3_video_path, seq_no, upload_timestamp
                FROM   video_files
                WHERE  process_flag = 'N'
                  AND  folder_name  = %s
                ORDER  BY seq_no
                """,
                (target_folder,),
            )
            rows = cur.fetchall()
            print(f"[DB] Found {len(rows)} video(s) in folder '{target_folder}'")
            for r in rows:
                print(
                    f"     train={r['train_detail_id']}  "
                    f"folder={r['folder_name']}  "
                    f"seq={r['seq_no']}  "
                    f"file={r['filename']}  "
                    f"uploaded={r['upload_timestamp']}"
                )
            return [dict(r) for r in rows]
    finally:
        conn.close()


def set_process_flag(video_id: int, flag: str) -> None:
    """
    Update process_flag for a single video row.
      N = pending
      I = in-progress  (set before pipeline starts)
      Y = done         (set after pipeline completes successfully)

    On pipeline failure this is intentionally NOT called with 'Y' —
    the flag stays 'I' so operators can see which video failed.
    """
    assert flag in ("N", "I", "Y"), f"Invalid flag: {flag!r}"
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE video_files
                SET    process_flag = %s,
                       updated_at   = NOW()
                WHERE  id = %s
                """,
                (flag, video_id),
            )
        conn.commit()
        print(f"[DB] id={video_id}  process_flag → '{flag}'")
    except Exception as exc:
        conn.rollback()
        print(f"[DB] set_process_flag error (id={video_id}): {exc}")
        raise
    finally:
        conn.close()


def update_result_s3_path(folder_name: str, result_s3_path: str) -> None:
    """Store the S3 result path on every row in this folder after processing."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE video_files
                SET    result_s3_path = %s,
                       updated_at     = NOW()
                WHERE  folder_name    = %s
                """,
                (result_s3_path, folder_name),
            )
        conn.commit()
        print(f"[DB] result_s3_path saved for folder '{folder_name}'")
    except Exception as exc:
        conn.rollback()
        print(f"[DB] update_result_s3_path error: {exc}")
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# S3 HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id     = os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name           = os.environ.get("AWS_REGION", "ap-south-1"),
    )


def _bucket() -> str:
    return os.environ["S3_BUCKET"]


def _s3_key_from_uri(s3_uri: str) -> str:
    """s3://bucket/folder/videos/ch01.mp4  →  folder/videos/ch01.mp4"""
    parts = s3_uri.replace("s3://", "").split("/", 1)
    return parts[1] if len(parts) == 2 else parts[0]


def download_video_from_s3(s3_video_path: str, local_path: str) -> str:
    """Download a video from S3 to local_path and return local_path."""
    if not s3_video_path or not s3_video_path.strip():
        raise ValueError(
            "s3_video_path is empty in DB. "
            "The frontend must store the S3 path when uploading."
        )
    key = (
        _s3_key_from_uri(s3_video_path)
        if s3_video_path.startswith("s3://")
        else s3_video_path.strip()
    )
    bkt = _bucket()
    print(f"[S3] Downloading  s3://{bkt}/{key}")
    print(f"     →  {local_path}")
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    _s3_client().download_file(bkt, key, local_path)
    return local_path


def _upload_results_to_s3(output_dir: str, folder_name: str) -> str:
    """
    Upload the full output folder to S3 after the pipeline finishes.

    Local layout:   outputs/<folder_name>/analysis_report.json
                    outputs/<folder_name>/frames/*.jpg
    S3 layout:      <folder_name>/results/analysis_report.json
                    <folder_name>/results/frames/*.jpg

    Returns the S3 URI of analysis_report.json (empty string on failure).
    """
    s3        = _s3_client()
    bkt       = _bucket()
    root      = Path(output_dir)
    report_s3 = ""

    for local_file in sorted(root.rglob("*")):
        if not local_file.is_file():
            continue
        rel    = local_file.relative_to(root)
        s3_key = f"{folder_name}/results/{rel.as_posix()}"
        print(f"[S3] Uploading {local_file.name}  →  s3://{bkt}/{s3_key}")
        s3.upload_file(str(local_file), bkt, s3_key)
        if local_file.name == "analysis_report.json":
            report_s3 = f"s3://{bkt}/{s3_key}"

    print(f"[S3] Upload complete for folder '{folder_name}'")
    return report_s3


# ══════════════════════════════════════════════════════════════════════════════
# CALLED BY ViolationStore.finalize()
# ══════════════════════════════════════════════════════════════════════════════

def finalize_and_upload(
    report_path:     str,
    analysis_id:     str,
    train_detail_id: int,
) -> None:
    """
    Called automatically by ViolationStore.finalize() after the JSON
    report and frame images are written locally.
    Uploads every output file to S3 and records the result path in DB.
    """
    output_dir  = os.path.dirname(report_path)
    folder_name = analysis_id
    try:
        report_s3 = _upload_results_to_s3(output_dir, folder_name)
        if report_s3:
            update_result_s3_path(folder_name, report_s3)
    except Exception as exc:
        print(f"[S3/DB] finalize_and_upload failed (non-fatal): {exc}")