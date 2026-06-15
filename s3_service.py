# # """
# # s3_service.py
# # ─────────────
# # S3 helpers for the Journey-based workflow.

# # • download_video()  — download one video from S3 to a local temp file.
# # • upload_frame()    — upload a single violation frame JPEG and return its S3 key.

# # Reuses the same boto3 / env-var pattern as the legacy db_s3_uploader.py so no
# # new credentials are needed.

# # S3 frame key convention
# # ───────────────────────
# #     journeys/<journeyId>/frames/<filename>

# # Spring Boot will generate signed URLs from these keys later.
# # """

# # from __future__ import annotations

# # import io
# # import os
# # from typing import Optional

# # import boto3
# # import cv2
# # import numpy as np
# # from dotenv import load_dotenv

# # # ── Credentials ────────────────────────────────────────────────────────────────
# # _ENV_PATH = os.path.join(
# #     os.path.dirname(os.path.abspath(__file__)),
# #     "config", "credentials.env",
# # )
# # load_dotenv(_ENV_PATH)


# # # ── boto3 helpers (same pattern as legacy db_s3_uploader.py) ──────────────────

# # def _s3_client():
# #     return boto3.client(
# #         "s3",
# #         aws_access_key_id     = os.environ["AWS_ACCESS_KEY_ID"],
# #         aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"],
# #         region_name           = os.environ.get("AWS_REGION", "ap-south-1"),
# #     )


# # def _bucket() -> str:
# #     return os.environ["S3_BUCKET"]


# # def _strip_s3_uri(s3_path: str) -> str:
# #     """s3://bucket/a/b/c  →  a/b/c   |   a/b/c  →  a/b/c"""
# #     if s3_path.startswith("s3://"):
# #         parts = s3_path.replace("s3://", "").split("/", 1)
# #         return parts[1] if len(parts) == 2 else parts[0]
# #     return s3_path.strip()


# # # ── Public API ─────────────────────────────────────────────────────────────────

# # def download_video(s3_key: str, local_path: str) -> str:
# #     """
# #     Download a video file from S3.

# #     Parameters
# #     ──────────
# #     s3_key     : S3 key (or full s3:// URI) as provided in the RabbitMQ message.
# #     local_path : Absolute local path where the file will be written.

# #     Returns local_path on success; raises on failure.
# #     """
# #     key = _strip_s3_uri(s3_key)
# #     bkt = _bucket()
# #     os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
# #     print(f"[S3] Downloading  s3://{bkt}/{key}  →  {local_path}")
# #     _s3_client().download_file(bkt, key, local_path)
# #     return local_path


# # def upload_frame(
# #     frame:      np.ndarray,
# #     journey_id: int,
# #     filename:   str,
# #     jpeg_quality: int = 85,
# # ) -> str:
# #     """
# #     Encode a numpy frame as JPEG and upload it to S3.

# #     Parameters
# #     ──────────
# #     frame        : BGR numpy array (OpenCV format).
# #     journey_id   : used to build the S3 key prefix.
# #     filename     : e.g. "phone_use_00-00-24.jpg"
# #     jpeg_quality : JPEG compression quality (default 85).

# #     Returns the S3 key (NOT a signed URL).
# #     Example return value: "journeys/101/frames/phone_use_00-00-24.jpg"
# #     """
# #     s3_key = f"journeys/{journey_id}/frames/{filename}"
# #     bkt    = _bucket()

# #     # Resize to 640×360 (matches legacy ViolationStore._save_frame behaviour)
# #     resized = cv2.resize(frame, (640, 360))
# #     ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
# #     if not ok:
# #         raise RuntimeError(f"cv2.imencode failed for frame '{filename}'")

# #     print(f"[S3] Uploading frame  →  s3://{bkt}/{s3_key}")
# #     _s3_client().put_object(
# #         Bucket      = bkt,
# #         Key         = s3_key,
# #         Body        = io.BytesIO(buf.tobytes()),
# #         ContentType = "image/jpeg",
# #     )
# #     return s3_key


# # def upload_frame_from_path(local_path: str, journey_id: int, filename: Optional[str] = None) -> str:
# #     """
# #     Upload a JPEG frame that has already been saved to disk.

# #     Parameters
# #     ──────────
# #     local_path : Absolute path to the .jpg file on disk.
# #     journey_id : used to build the S3 key prefix.
# #     filename   : Override the S3 filename; defaults to os.path.basename(local_path).

# #     Returns the S3 key.
# #     """
# #     fname  = filename or os.path.basename(local_path)
# #     s3_key = f"journeys/{journey_id}/frames/{fname}"
# #     bkt    = _bucket()

# #     print(f"[S3] Uploading frame (from disk)  →  s3://{bkt}/{s3_key}")
# #     _s3_client().upload_file(local_path, bkt, s3_key)
# #     return s3_key


# """
# s3_service.py
# ─────────────
# S3 helpers for the Journey-based workflow.

# • download_video()          — download one video from S3 to a local temp file.
# • upload_frame()            — upload a numpy frame JPEG and return its S3 key.
# • upload_frame_from_path()  — upload an already-saved JPEG and return its S3 key.

# Changes from previous version
# ──────────────────────────────
# • upload_frame() and upload_frame_from_path() now accept an optional
#   `folder_name` parameter so frames are written to:
#       <folderName>/frames/<filename>
#   rather than the old hard-coded:
#       journeys/<journeyId>/frames/<filename>

#   When folder_name is omitted (or None) the old path is used as a fallback
#   so call sites that haven't been updated yet continue to work.

# S3 frame key convention (new)
# ─────────────────────────────
#     <folderName>/frames/<filename>

#   e.g. "journeys/1/2026-06-10/JRN-20260610-1-ABC123/frames/phone_use_00-00-24.jpg"

# Spring Boot generates signed URLs from these keys.
# """

# from __future__ import annotations

# import io
# import os
# from typing import Optional

# import boto3
# import cv2
# import numpy as np
# from dotenv import load_dotenv

# # ── Credentials ────────────────────────────────────────────────────────────────
# _ENV_PATH = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)),
#     "config", "credentials.env",
# )
# load_dotenv(_ENV_PATH)


# # ── boto3 helpers ──────────────────────────────────────────────────────────────

# def _s3_client():
#     return boto3.client(
#         "s3",
#         aws_access_key_id     = os.environ["AWS_ACCESS_KEY_ID"],
#         aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"],
#         region_name           = os.environ.get("AWS_REGION", "ap-south-1"),
#     )


# def _bucket() -> str:
#     return os.environ["S3_BUCKET"]


# def _strip_s3_uri(s3_path: str) -> str:
#     """s3://bucket/a/b/c  →  a/b/c   |   a/b/c  →  a/b/c"""
#     if s3_path.startswith("s3://"):
#         parts = s3_path.replace("s3://", "").split("/", 1)
#         return parts[1] if len(parts) == 2 else parts[0]
#     return s3_path.strip()


# def _frame_s3_key(
#     filename:   str,
#     journey_id: int,
#     folder_name: Optional[str],
# ) -> str:
#     """
#     Build the S3 key for a violation frame.

#     Preferred (folder_name provided):
#         <folderName>/frames/<filename>
#     Fallback (no folder_name):
#         journeys/<journeyId>/frames/<filename>
#     """
#     if folder_name:
#         return f"{folder_name.rstrip('/')}/frames/{filename}"
#     return f"journeys/{journey_id}/frames/{filename}"


# # ── Public API ─────────────────────────────────────────────────────────────────

# def download_video(s3_key: str, local_path: str) -> str:
#     """
#     Download a video file from S3.

#     Parameters
#     ──────────
#     s3_key     : S3 key (or full s3:// URI) as provided in the RabbitMQ message.
#     local_path : Absolute local path where the file will be written.

#     Returns local_path on success; raises on failure.
#     """
#     key = _strip_s3_uri(s3_key)
#     bkt = _bucket()
#     os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
#     print(f"[S3] Downloading  s3://{bkt}/{key}  →  {local_path}")
#     _s3_client().download_file(bkt, key, local_path)
#     return local_path


# def upload_frame(
#     frame:        np.ndarray,
#     journey_id:   int,
#     filename:     str,
#     jpeg_quality: int = 85,
#     folder_name:  Optional[str] = None,   # NEW — preferred over journey_id fallback
# ) -> str:
#     """
#     Encode a numpy frame as JPEG and upload it to S3.

#     Parameters
#     ──────────
#     frame        : BGR numpy array (OpenCV format).
#     journey_id   : used for fallback S3 key when folder_name is not given.
#     filename     : e.g. "phone_use_00-00-24.jpg"
#     jpeg_quality : JPEG compression quality (default 85).
#     folder_name  : Journey folder prefix, e.g.
#                    "journeys/1/2026-06-10/JRN-20260610-1-ABC123".
#                    When provided, frame is uploaded to
#                    "<folderName>/frames/<filename>".

#     Returns the S3 key (NOT a signed URL).
#     """
#     s3_key = _frame_s3_key(filename, journey_id, folder_name)
#     bkt    = _bucket()

#     # Resize to 640×360 (matches legacy ViolationStore._save_frame behaviour)
#     resized = cv2.resize(frame, (640, 360))
#     ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
#     if not ok:
#         raise RuntimeError(f"cv2.imencode failed for frame '{filename}'")

#     print(f"[S3] Uploading frame  →  s3://{bkt}/{s3_key}")
#     _s3_client().put_object(
#         Bucket      = bkt,
#         Key         = s3_key,
#         Body        = io.BytesIO(buf.tobytes()),
#         ContentType = "image/jpeg",
#     )
#     return s3_key


# def upload_frame_from_path(
#     local_path:  str,
#     journey_id:  int,
#     filename:    Optional[str] = None,
#     folder_name: Optional[str] = None,   # NEW — preferred over journey_id fallback
# ) -> str:
#     """
#     Upload a JPEG frame that has already been saved to disk.

#     Parameters
#     ──────────
#     local_path  : Absolute path to the .jpg file on disk.
#     journey_id  : used for fallback S3 key when folder_name is not given.
#     filename    : Override the S3 filename; defaults to os.path.basename(local_path).
#     folder_name : Journey folder prefix (see upload_frame docstring).

#     Returns the S3 key.
#     """
#     fname  = filename or os.path.basename(local_path)
#     s3_key = _frame_s3_key(fname, journey_id, folder_name)
#     bkt    = _bucket()

#     print(f"[S3] Uploading frame (from disk)  →  s3://{bkt}/{s3_key}")
#     _s3_client().upload_file(local_path, bkt, s3_key)
#     return s3_key


# """

# s3_service.py

# ─────────────

# S3 helpers for the Journey-based workflow.
 
# • download_video()  — download one video from S3 to a local temp file.

# • upload_frame()    — upload a single violation frame JPEG and return its S3 key.
 
# Reuses the same boto3 / env-var pattern as the legacy db_s3_uploader.py so no

# new credentials are needed.
 
# S3 frame key convention

# ───────────────────────

#     journeys/<journeyId>/frames/<filename>
 
# Spring Boot will generate signed URLs from these keys later.

# """
 
# from __future__ import annotations
 
# import io

# import os

# from typing import Optional
 
# import boto3

# import cv2

# import numpy as np

# from dotenv import load_dotenv
 
# # ── Credentials ────────────────────────────────────────────────────────────────

# _ENV_PATH = os.path.join(

#     os.path.dirname(os.path.abspath(__file__)),

#     "config", "credentials.env",

# )

# load_dotenv(_ENV_PATH)
 
 
# # ── boto3 helpers (same pattern as legacy db_s3_uploader.py) ──────────────────
 
# def _s3_client():

#     return boto3.client(

#         "s3",

#         aws_access_key_id     = os.environ["AWS_ACCESS_KEY_ID"],

#         aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"],

#         region_name           = os.environ.get("AWS_REGION", "ap-south-1"),

#     )
 
 
# def _bucket() -> str:

#     return os.environ["S3_BUCKET"]
 
 
# def _strip_s3_uri(s3_path: str) -> str:

#     """s3://bucket/a/b/c  →  a/b/c   |   a/b/c  →  a/b/c"""

#     if s3_path.startswith("s3://"):

#         parts = s3_path.replace("s3://", "").split("/", 1)

#         return parts[1] if len(parts) == 2 else parts[0]

#     return s3_path.strip()
 
 
# # ── Public API ─────────────────────────────────────────────────────────────────
 
# def download_video(s3_key: str, local_path: str) -> str:

#     """

#     Download a video file from S3.
 
#     Parameters

#     ──────────

#     s3_key     : S3 key (or full s3:// URI) as provided in the RabbitMQ message.

#     local_path : Absolute local path where the file will be written.
 
#     Returns local_path on success; raises on failure.

#     """

#     key = _strip_s3_uri(s3_key)

#     bkt = _bucket()

#     os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

#     print(f"[S3] Downloading  s3://{bkt}/{key}  →  {local_path}")

#     _s3_client().download_file(bkt, key, local_path)

#     return local_path
 
 
# def upload_frame(

#     frame:      np.ndarray,

#     journey_id: int,

#     filename:   str,

#     jpeg_quality: int = 85,

# ) -> str:

#     """

#     Encode a numpy frame as JPEG and upload it to S3.
 
#     Parameters

#     ──────────

#     frame        : BGR numpy array (OpenCV format).

#     journey_id   : used to build the S3 key prefix.

#     filename     : e.g. "phone_use_00-00-24.jpg"

#     jpeg_quality : JPEG compression quality (default 85).
 
#     Returns the S3 key (NOT a signed URL).

#     Example return value: "journeys/101/frames/phone_use_00-00-24.jpg"

#     """

#     s3_key = f"journeys/{journey_id}/frames/{filename}"

#     bkt    = _bucket()
 
#     # Resize to 640×360 (matches legacy ViolationStore._save_frame behaviour)

#     resized = cv2.resize(frame, (640, 360))

#     ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

#     if not ok:

#         raise RuntimeError(f"cv2.imencode failed for frame '{filename}'")
 
#     print(f"[S3] Uploading frame  →  s3://{bkt}/{s3_key}")

#     _s3_client().put_object(

#         Bucket      = bkt,

#         Key         = s3_key,

#         Body        = io.BytesIO(buf.tobytes()),

#         ContentType = "image/jpeg",

#     )

#     return s3_key
 
 
# def upload_frame_from_path(local_path: str, journey_id: int, filename: Optional[str] = None) -> str:

#     """

#     Upload a JPEG frame that has already been saved to disk.
 
#     Parameters

#     ──────────

#     local_path : Absolute path to the .jpg file on disk.

#     journey_id : used to build the S3 key prefix.

#     filename   : Override the S3 filename; defaults to os.path.basename(local_path).
 
#     Returns the S3 key.

#     """

#     fname  = filename or os.path.basename(local_path)

#     s3_key = f"journeys/{journey_id}/frames/{fname}"

#     bkt    = _bucket()
 
#     print(f"[S3] Uploading frame (from disk)  →  s3://{bkt}/{s3_key}")

#     _s3_client().upload_file(local_path, bkt, s3_key)

#     return s3_key
 
 
"""

s3_service.py

─────────────

S3 helpers for the Journey-based workflow.
 
• download_video()          — download one video from S3 to a local temp file.

• upload_frame()            — upload a numpy frame JPEG and return its S3 key.

• upload_frame_from_path()  — upload an already-saved JPEG and return its S3 key.
 
Changes from previous version

──────────────────────────────

• upload_frame() and upload_frame_from_path() now accept an optional

  `folder_name` parameter so frames are written to:
<folderName>/frames/<filename>

  rather than the old hard-coded:

      journeys/<journeyId>/frames/<filename>
 
  When folder_name is omitted (or None) the old path is used as a fallback

  so call sites that haven't been updated yet continue to work.
 
S3 frame key convention (new)

─────────────────────────────
<folderName>/frames/<filename>
 
  e.g. "journeys/1/2026-06-10/JRN-20260610-1-ABC123/frames/phone_use_00-00-24.jpg"
 
Spring Boot generates signed URLs from these keys.

"""
 
from __future__ import annotations
 
import io

import os

from typing import Optional
 
import boto3

import cv2

import numpy as np

from dotenv import load_dotenv
 
# ── Credentials ────────────────────────────────────────────────────────────────

_ENV_PATH = os.path.join(

    os.path.dirname(os.path.abspath(__file__)),

    "config", "credentials.env",

)

load_dotenv(_ENV_PATH)
 
 
# ── boto3 helpers ──────────────────────────────────────────────────────────────
 
def _s3_client():

    return boto3.client(

        "s3",

        aws_access_key_id     = os.environ["AWS_ACCESS_KEY_ID"],

        aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"],

        region_name           = os.environ.get("AWS_REGION", "ap-south-1"),

    )
 
 
def _bucket() -> str:

    return os.environ["S3_BUCKET"]
 
 
def _strip_s3_uri(s3_path: str) -> str:

    """s3://bucket/a/b/c  →  a/b/c   |   a/b/c  →  a/b/c"""

    if s3_path.startswith("s3://"):

        parts = s3_path.replace("s3://", "").split("/", 1)

        return parts[1] if len(parts) == 2 else parts[0]

    return s3_path.strip()
 
 
def _frame_s3_key(

    filename:   str,

    journey_id: int,

    folder_name: Optional[str],

) -> str:

    """

    Build the S3 key for a violation frame.
 
    Preferred (folder_name provided):
<folderName>/frames/<filename>

    Fallback (no folder_name):

        journeys/<journeyId>/frames/<filename>

    """

    if folder_name:

        return f"{folder_name.rstrip('/')}/frames/{filename}"

    return f"journeys/{journey_id}/frames/{filename}"
 
 
# ── Public API ─────────────────────────────────────────────────────────────────
 
def download_video(s3_key: str, local_path: str) -> str:

    """

    Download a video file from S3.
 
    Parameters

    ──────────

    s3_key     : S3 key (or full s3:// URI) as provided in the RabbitMQ message.

    local_path : Absolute local path where the file will be written.
 
    Returns local_path on success; raises on failure.

    """

    key = _strip_s3_uri(s3_key)

    bkt = _bucket()

    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

    print(f"[S3] Downloading  s3://{bkt}/{key}  →  {local_path}")

    _s3_client().download_file(bkt, key, local_path)

    return local_path
 
 
def upload_frame(

    frame:        np.ndarray,

    journey_id:   int,

    filename:     str,

    jpeg_quality: int = 85,

    folder_name:  Optional[str] = None,   # NEW — preferred over journey_id fallback

) -> str:

    """

    Encode a numpy frame as JPEG and upload it to S3.
 
    Parameters

    ──────────

    frame        : BGR numpy array (OpenCV format).

    journey_id   : used for fallback S3 key when folder_name is not given.

    filename     : e.g. "phone_use_00-00-24.jpg"

    jpeg_quality : JPEG compression quality (default 85).

    folder_name  : Journey folder prefix, e.g.

                   "journeys/1/2026-06-10/JRN-20260610-1-ABC123".

                   When provided, frame is uploaded to

                   "<folderName>/frames/<filename>".
 
    Returns the S3 key (NOT a signed URL).

    """

    s3_key = _frame_s3_key(filename, journey_id, folder_name)

    bkt    = _bucket()
 
    # Resize to 640×360 (matches legacy ViolationStore._save_frame behaviour)

    resized = cv2.resize(frame, (640, 360))

    ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

    if not ok:

        raise RuntimeError(f"cv2.imencode failed for frame '{filename}'")
 
    print(f"[S3] Uploading frame  →  s3://{bkt}/{s3_key}")

    _s3_client().put_object(

        Bucket      = bkt,

        Key         = s3_key,

        Body        = io.BytesIO(buf.tobytes()),

        ContentType = "image/jpeg",

    )

    return s3_key
 
 
def upload_frame_from_path(

    local_path:  str,

    journey_id:  int,

    filename:    Optional[str] = None,

    folder_name: Optional[str] = None,   # NEW — preferred over journey_id fallback

) -> str:

    """

    Upload a JPEG frame that has already been saved to disk.
 
    Parameters

    ──────────

    local_path  : Absolute path to the .jpg file on disk.

    journey_id  : used for fallback S3 key when folder_name is not given.

    filename    : Override the S3 filename; defaults to os.path.basename(local_path).

    folder_name : Journey folder prefix (see upload_frame docstring).
 
    Returns the S3 key.

    """

    fname  = filename or os.path.basename(local_path)

    s3_key = _frame_s3_key(fname, journey_id, folder_name)

    bkt    = _bucket()
 
    print(f"[S3] Uploading frame (from disk)  →  s3://{bkt}/{s3_key}")

    _s3_client().upload_file(local_path, bkt, s3_key)

    return s3_key
 