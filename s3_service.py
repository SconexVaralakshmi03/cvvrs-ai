# # # # """
# # # # s3_service.py
# # # # ─────────────
# # # # S3 helpers for the Journey-based workflow.

# # # # • download_video()  — download one video from S3 to a local temp file.
# # # # • upload_frame()    — upload a single violation frame JPEG and return its S3 key.

# # # # Reuses the same boto3 / env-var pattern as the legacy db_s3_uploader.py so no
# # # # new credentials are needed.

# # # # S3 frame key convention
# # # # ───────────────────────
# # # #     journeys/<journeyId>/frames/<filename>

# # # # Spring Boot will generate signed URLs from these keys later.
# # # # """

# # # # from __future__ import annotations

# # # # import io
# # # # import os
# # # # from typing import Optional

# # # # import boto3
# # # # import cv2
# # # # import numpy as np
# # # # from dotenv import load_dotenv

# # # # # ── Credentials ────────────────────────────────────────────────────────────────
# # # # _ENV_PATH = os.path.join(
# # # #     os.path.dirname(os.path.abspath(__file__)),
# # # #     "config", "credentials.env",
# # # # )
# # # # load_dotenv(_ENV_PATH)


# # # # # ── boto3 helpers (same pattern as legacy db_s3_uploader.py) ──────────────────

# # # # def _s3_client():
# # # #     return boto3.client(
# # # #         "s3",
# # # #         aws_access_key_id     = os.environ["AWS_ACCESS_KEY_ID"],
# # # #         aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"],
# # # #         region_name           = os.environ.get("AWS_REGION", "ap-south-1"),
# # # #     )


# # # # def _bucket() -> str:
# # # #     return os.environ["S3_BUCKET"]


# # # # def _strip_s3_uri(s3_path: str) -> str:
# # # #     """s3://bucket/a/b/c  →  a/b/c   |   a/b/c  →  a/b/c"""
# # # #     if s3_path.startswith("s3://"):
# # # #         parts = s3_path.replace("s3://", "").split("/", 1)
# # # #         return parts[1] if len(parts) == 2 else parts[0]
# # # #     return s3_path.strip()


# # # # # ── Public API ─────────────────────────────────────────────────────────────────

# # # # def download_video(s3_key: str, local_path: str) -> str:
# # # #     """
# # # #     Download a video file from S3.

# # # #     Parameters
# # # #     ──────────
# # # #     s3_key     : S3 key (or full s3:// URI) as provided in the RabbitMQ message.
# # # #     local_path : Absolute local path where the file will be written.

# # # #     Returns local_path on success; raises on failure.
# # # #     """
# # # #     key = _strip_s3_uri(s3_key)
# # # #     bkt = _bucket()
# # # #     os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
# # # #     print(f"[S3] Downloading  s3://{bkt}/{key}  →  {local_path}")
# # # #     _s3_client().download_file(bkt, key, local_path)
# # # #     return local_path


# # # # def upload_frame(
# # # #     frame:      np.ndarray,
# # # #     journey_id: int,
# # # #     filename:   str,
# # # #     jpeg_quality: int = 85,
# # # # ) -> str:
# # # #     """
# # # #     Encode a numpy frame as JPEG and upload it to S3.

# # # #     Parameters
# # # #     ──────────
# # # #     frame        : BGR numpy array (OpenCV format).
# # # #     journey_id   : used to build the S3 key prefix.
# # # #     filename     : e.g. "phone_use_00-00-24.jpg"
# # # #     jpeg_quality : JPEG compression quality (default 85).

# # # #     Returns the S3 key (NOT a signed URL).
# # # #     Example return value: "journeys/101/frames/phone_use_00-00-24.jpg"
# # # #     """
# # # #     s3_key = f"journeys/{journey_id}/frames/{filename}"
# # # #     bkt    = _bucket()

# # # #     # Resize to 640×360 (matches legacy ViolationStore._save_frame behaviour)
# # # #     resized = cv2.resize(frame, (640, 360))
# # # #     ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
# # # #     if not ok:
# # # #         raise RuntimeError(f"cv2.imencode failed for frame '{filename}'")

# # # #     print(f"[S3] Uploading frame  →  s3://{bkt}/{s3_key}")
# # # #     _s3_client().put_object(
# # # #         Bucket      = bkt,
# # # #         Key         = s3_key,
# # # #         Body        = io.BytesIO(buf.tobytes()),
# # # #         ContentType = "image/jpeg",
# # # #     )
# # # #     return s3_key


# # # # def upload_frame_from_path(local_path: str, journey_id: int, filename: Optional[str] = None) -> str:
# # # #     """
# # # #     Upload a JPEG frame that has already been saved to disk.

# # # #     Parameters
# # # #     ──────────
# # # #     local_path : Absolute path to the .jpg file on disk.
# # # #     journey_id : used to build the S3 key prefix.
# # # #     filename   : Override the S3 filename; defaults to os.path.basename(local_path).

# # # #     Returns the S3 key.
# # # #     """
# # # #     fname  = filename or os.path.basename(local_path)
# # # #     s3_key = f"journeys/{journey_id}/frames/{fname}"
# # # #     bkt    = _bucket()

# # # #     print(f"[S3] Uploading frame (from disk)  →  s3://{bkt}/{s3_key}")
# # # #     _s3_client().upload_file(local_path, bkt, s3_key)
# # # #     return s3_key


# # # """
# # # s3_service.py
# # # ─────────────
# # # S3 helpers for the Journey-based workflow.

# # # • download_video()          — download one video from S3 to a local temp file.
# # # • upload_frame()            — upload a numpy frame JPEG and return its S3 key.
# # # • upload_frame_from_path()  — upload an already-saved JPEG and return its S3 key.

# # # Changes from previous version
# # # ──────────────────────────────
# # # • upload_frame() and upload_frame_from_path() now accept an optional
# # #   `folder_name` parameter so frames are written to:
# # #       <folderName>/frames/<filename>
# # #   rather than the old hard-coded:
# # #       journeys/<journeyId>/frames/<filename>

# # #   When folder_name is omitted (or None) the old path is used as a fallback
# # #   so call sites that haven't been updated yet continue to work.

# # # S3 frame key convention (new)
# # # ─────────────────────────────
# # #     <folderName>/frames/<filename>

# # #   e.g. "journeys/1/2026-06-10/JRN-20260610-1-ABC123/frames/phone_use_00-00-24.jpg"

# # # Spring Boot generates signed URLs from these keys.
# # # """

# # # from __future__ import annotations

# # # import io
# # # import os
# # # from typing import Optional

# # # import boto3
# # # import cv2
# # # import numpy as np
# # # from dotenv import load_dotenv

# # # # ── Credentials ────────────────────────────────────────────────────────────────
# # # _ENV_PATH = os.path.join(
# # #     os.path.dirname(os.path.abspath(__file__)),
# # #     "config", "credentials.env",
# # # )
# # # load_dotenv(_ENV_PATH)


# # # # ── boto3 helpers ──────────────────────────────────────────────────────────────

# # # def _s3_client():
# # #     return boto3.client(
# # #         "s3",
# # #         aws_access_key_id     = os.environ["AWS_ACCESS_KEY_ID"],
# # #         aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"],
# # #         region_name           = os.environ.get("AWS_REGION", "ap-south-1"),
# # #     )


# # # def _bucket() -> str:
# # #     return os.environ["S3_BUCKET"]


# # # def _strip_s3_uri(s3_path: str) -> str:
# # #     """s3://bucket/a/b/c  →  a/b/c   |   a/b/c  →  a/b/c"""
# # #     if s3_path.startswith("s3://"):
# # #         parts = s3_path.replace("s3://", "").split("/", 1)
# # #         return parts[1] if len(parts) == 2 else parts[0]
# # #     return s3_path.strip()


# # # def _frame_s3_key(
# # #     filename:   str,
# # #     journey_id: int,
# # #     folder_name: Optional[str],
# # # ) -> str:
# # #     """
# # #     Build the S3 key for a violation frame.

# # #     Preferred (folder_name provided):
# # #         <folderName>/frames/<filename>
# # #     Fallback (no folder_name):
# # #         journeys/<journeyId>/frames/<filename>
# # #     """
# # #     if folder_name:
# # #         return f"{folder_name.rstrip('/')}/frames/{filename}"
# # #     return f"journeys/{journey_id}/frames/{filename}"


# # # # ── Public API ─────────────────────────────────────────────────────────────────

# # # def download_video(s3_key: str, local_path: str) -> str:
# # #     """
# # #     Download a video file from S3.

# # #     Parameters
# # #     ──────────
# # #     s3_key     : S3 key (or full s3:// URI) as provided in the RabbitMQ message.
# # #     local_path : Absolute local path where the file will be written.

# # #     Returns local_path on success; raises on failure.
# # #     """
# # #     key = _strip_s3_uri(s3_key)
# # #     bkt = _bucket()
# # #     os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
# # #     print(f"[S3] Downloading  s3://{bkt}/{key}  →  {local_path}")
# # #     _s3_client().download_file(bkt, key, local_path)
# # #     return local_path


# # # def upload_frame(
# # #     frame:        np.ndarray,
# # #     journey_id:   int,
# # #     filename:     str,
# # #     jpeg_quality: int = 85,
# # #     folder_name:  Optional[str] = None,   # NEW — preferred over journey_id fallback
# # # ) -> str:
# # #     """
# # #     Encode a numpy frame as JPEG and upload it to S3.

# # #     Parameters
# # #     ──────────
# # #     frame        : BGR numpy array (OpenCV format).
# # #     journey_id   : used for fallback S3 key when folder_name is not given.
# # #     filename     : e.g. "phone_use_00-00-24.jpg"
# # #     jpeg_quality : JPEG compression quality (default 85).
# # #     folder_name  : Journey folder prefix, e.g.
# # #                    "journeys/1/2026-06-10/JRN-20260610-1-ABC123".
# # #                    When provided, frame is uploaded to
# # #                    "<folderName>/frames/<filename>".

# # #     Returns the S3 key (NOT a signed URL).
# # #     """
# # #     s3_key = _frame_s3_key(filename, journey_id, folder_name)
# # #     bkt    = _bucket()

# # #     # Resize to 640×360 (matches legacy ViolationStore._save_frame behaviour)
# # #     resized = cv2.resize(frame, (640, 360))
# # #     ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
# # #     if not ok:
# # #         raise RuntimeError(f"cv2.imencode failed for frame '{filename}'")

# # #     print(f"[S3] Uploading frame  →  s3://{bkt}/{s3_key}")
# # #     _s3_client().put_object(
# # #         Bucket      = bkt,
# # #         Key         = s3_key,
# # #         Body        = io.BytesIO(buf.tobytes()),
# # #         ContentType = "image/jpeg",
# # #     )
# # #     return s3_key


# # # def upload_frame_from_path(
# # #     local_path:  str,
# # #     journey_id:  int,
# # #     filename:    Optional[str] = None,
# # #     folder_name: Optional[str] = None,   # NEW — preferred over journey_id fallback
# # # ) -> str:
# # #     """
# # #     Upload a JPEG frame that has already been saved to disk.

# # #     Parameters
# # #     ──────────
# # #     local_path  : Absolute path to the .jpg file on disk.
# # #     journey_id  : used for fallback S3 key when folder_name is not given.
# # #     filename    : Override the S3 filename; defaults to os.path.basename(local_path).
# # #     folder_name : Journey folder prefix (see upload_frame docstring).

# # #     Returns the S3 key.
# # #     """
# # #     fname  = filename or os.path.basename(local_path)
# # #     s3_key = _frame_s3_key(fname, journey_id, folder_name)
# # #     bkt    = _bucket()

# # #     print(f"[S3] Uploading frame (from disk)  →  s3://{bkt}/{s3_key}")
# # #     _s3_client().upload_file(local_path, bkt, s3_key)
# # #     return s3_key


# # # """

# # # s3_service.py

# # # ─────────────

# # # S3 helpers for the Journey-based workflow.
 
# # # • download_video()  — download one video from S3 to a local temp file.

# # # • upload_frame()    — upload a single violation frame JPEG and return its S3 key.
 
# # # Reuses the same boto3 / env-var pattern as the legacy db_s3_uploader.py so no

# # # new credentials are needed.
 
# # # S3 frame key convention

# # # ───────────────────────

# # #     journeys/<journeyId>/frames/<filename>
 
# # # Spring Boot will generate signed URLs from these keys later.

# # # """
 
# # # from __future__ import annotations
 
# # # import io

# # # import os

# # # from typing import Optional
 
# # # import boto3

# # # import cv2

# # # import numpy as np

# # # from dotenv import load_dotenv
 
# # # # ── Credentials ────────────────────────────────────────────────────────────────

# # # _ENV_PATH = os.path.join(

# # #     os.path.dirname(os.path.abspath(__file__)),

# # #     "config", "credentials.env",

# # # )

# # # load_dotenv(_ENV_PATH)
 
 
# # # # ── boto3 helpers (same pattern as legacy db_s3_uploader.py) ──────────────────
 
# # # def _s3_client():

# # #     return boto3.client(

# # #         "s3",

# # #         aws_access_key_id     = os.environ["AWS_ACCESS_KEY_ID"],

# # #         aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"],

# # #         region_name           = os.environ.get("AWS_REGION", "ap-south-1"),

# # #     )
 
 
# # # def _bucket() -> str:

# # #     return os.environ["S3_BUCKET"]
 
 
# # # def _strip_s3_uri(s3_path: str) -> str:

# # #     """s3://bucket/a/b/c  →  a/b/c   |   a/b/c  →  a/b/c"""

# # #     if s3_path.startswith("s3://"):

# # #         parts = s3_path.replace("s3://", "").split("/", 1)

# # #         return parts[1] if len(parts) == 2 else parts[0]

# # #     return s3_path.strip()
 
 
# # # # ── Public API ─────────────────────────────────────────────────────────────────
 
# # # def download_video(s3_key: str, local_path: str) -> str:

# # #     """

# # #     Download a video file from S3.
 
# # #     Parameters

# # #     ──────────

# # #     s3_key     : S3 key (or full s3:// URI) as provided in the RabbitMQ message.

# # #     local_path : Absolute local path where the file will be written.
 
# # #     Returns local_path on success; raises on failure.

# # #     """

# # #     key = _strip_s3_uri(s3_key)

# # #     bkt = _bucket()

# # #     os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

# # #     print(f"[S3] Downloading  s3://{bkt}/{key}  →  {local_path}")

# # #     _s3_client().download_file(bkt, key, local_path)

# # #     return local_path
 
 
# # # def upload_frame(

# # #     frame:      np.ndarray,

# # #     journey_id: int,

# # #     filename:   str,

# # #     jpeg_quality: int = 85,

# # # ) -> str:

# # #     """

# # #     Encode a numpy frame as JPEG and upload it to S3.
 
# # #     Parameters

# # #     ──────────

# # #     frame        : BGR numpy array (OpenCV format).

# # #     journey_id   : used to build the S3 key prefix.

# # #     filename     : e.g. "phone_use_00-00-24.jpg"

# # #     jpeg_quality : JPEG compression quality (default 85).
 
# # #     Returns the S3 key (NOT a signed URL).

# # #     Example return value: "journeys/101/frames/phone_use_00-00-24.jpg"

# # #     """

# # #     s3_key = f"journeys/{journey_id}/frames/{filename}"

# # #     bkt    = _bucket()
 
# # #     # Resize to 640×360 (matches legacy ViolationStore._save_frame behaviour)

# # #     resized = cv2.resize(frame, (640, 360))

# # #     ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

# # #     if not ok:

# # #         raise RuntimeError(f"cv2.imencode failed for frame '{filename}'")
 
# # #     print(f"[S3] Uploading frame  →  s3://{bkt}/{s3_key}")

# # #     _s3_client().put_object(

# # #         Bucket      = bkt,

# # #         Key         = s3_key,

# # #         Body        = io.BytesIO(buf.tobytes()),

# # #         ContentType = "image/jpeg",

# # #     )

# # #     return s3_key
 
 
# # # def upload_frame_from_path(local_path: str, journey_id: int, filename: Optional[str] = None) -> str:

# # #     """

# # #     Upload a JPEG frame that has already been saved to disk.
 
# # #     Parameters

# # #     ──────────

# # #     local_path : Absolute path to the .jpg file on disk.

# # #     journey_id : used to build the S3 key prefix.

# # #     filename   : Override the S3 filename; defaults to os.path.basename(local_path).
 
# # #     Returns the S3 key.

# # #     """

# # #     fname  = filename or os.path.basename(local_path)

# # #     s3_key = f"journeys/{journey_id}/frames/{fname}"

# # #     bkt    = _bucket()
 
# # #     print(f"[S3] Uploading frame (from disk)  →  s3://{bkt}/{s3_key}")

# # #     _s3_client().upload_file(local_path, bkt, s3_key)

# # #     return s3_key
 
 
# # """

# # s3_service.py

# # ─────────────

# # S3 helpers for the Journey-based workflow.
 
# # • download_video()          — download one video from S3 to a local temp file.

# # • upload_frame()            — upload a numpy frame JPEG and return its S3 key.

# # • upload_frame_from_path()  — upload an already-saved JPEG and return its S3 key.
 
# # Changes from previous version

# # ──────────────────────────────

# # • upload_frame() and upload_frame_from_path() now accept an optional

# #   `folder_name` parameter so frames are written to:
# # <folderName>/frames/<filename>

# #   rather than the old hard-coded:

# #       journeys/<journeyId>/frames/<filename>
 
# #   When folder_name is omitted (or None) the old path is used as a fallback

# #   so call sites that haven't been updated yet continue to work.
 
# # S3 frame key convention (new)

# # ─────────────────────────────
# # <folderName>/frames/<filename>
 
# #   e.g. "journeys/1/2026-06-10/JRN-20260610-1-ABC123/frames/phone_use_00-00-24.jpg"
 
# # Spring Boot generates signed URLs from these keys.

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
 
 
# # # ── boto3 helpers ──────────────────────────────────────────────────────────────
 
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
 
 
# # def _frame_s3_key(

# #     filename:   str,

# #     journey_id: int,

# #     folder_name: Optional[str],

# # ) -> str:

# #     """

# #     Build the S3 key for a violation frame.
 
# #     Preferred (folder_name provided):
# # <folderName>/frames/<filename>

# #     Fallback (no folder_name):

# #         journeys/<journeyId>/frames/<filename>

# #     """

# #     if folder_name:

# #         return f"{folder_name.rstrip('/')}/frames/{filename}"

# #     return f"journeys/{journey_id}/frames/{filename}"
 
 
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

# #     frame:        np.ndarray,

# #     journey_id:   int,

# #     filename:     str,

# #     jpeg_quality: int = 85,

# #     folder_name:  Optional[str] = None,   # NEW — preferred over journey_id fallback

# # ) -> str:

# #     """

# #     Encode a numpy frame as JPEG and upload it to S3.
 
# #     Parameters

# #     ──────────

# #     frame        : BGR numpy array (OpenCV format).

# #     journey_id   : used for fallback S3 key when folder_name is not given.

# #     filename     : e.g. "phone_use_00-00-24.jpg"

# #     jpeg_quality : JPEG compression quality (default 85).

# #     folder_name  : Journey folder prefix, e.g.

# #                    "journeys/1/2026-06-10/JRN-20260610-1-ABC123".

# #                    When provided, frame is uploaded to

# #                    "<folderName>/frames/<filename>".
 
# #     Returns the S3 key (NOT a signed URL).

# #     """

# #     s3_key = _frame_s3_key(filename, journey_id, folder_name)

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
 
 
# # def upload_frame_from_path(

# #     local_path:  str,

# #     journey_id:  int,

# #     filename:    Optional[str] = None,

# #     folder_name: Optional[str] = None,   # NEW — preferred over journey_id fallback

# # ) -> str:

# #     """

# #     Upload a JPEG frame that has already been saved to disk.
 
# #     Parameters

# #     ──────────

# #     local_path  : Absolute path to the .jpg file on disk.

# #     journey_id  : used for fallback S3 key when folder_name is not given.

# #     filename    : Override the S3 filename; defaults to os.path.basename(local_path).

# #     folder_name : Journey folder prefix (see upload_frame docstring).
 
# #     Returns the S3 key.

# #     """

# #     fname  = filename or os.path.basename(local_path)

# #     s3_key = _frame_s3_key(fname, journey_id, folder_name)

# #     bkt    = _bucket()
 
# #     print(f"[S3] Uploading frame (from disk)  →  s3://{bkt}/{s3_key}")

# #     _s3_client().upload_file(local_path, bkt, s3_key)

# #     return s3_key
 
 
#  # # """
# # # s3_service.py
# # # ─────────────
# # # S3 helpers for the Journey-based workflow.

# # # • download_video()  — download one video from S3 to a local temp file.
# # # • upload_frame()    — upload a single violation frame JPEG and return its S3 key.

# # # Reuses the same boto3 / env-var pattern as the legacy db_s3_uploader.py so no
# # # new credentials are needed.

# # # S3 frame key convention
# # # ───────────────────────
# # #     journeys/<journeyId>/frames/<filename>

# # # Spring Boot will generate signed URLs from these keys later.
# # # """

# # # from __future__ import annotations

# # # import io
# # # import os
# # # from typing import Optional

# # # import boto3
# # # import cv2
# # # import numpy as np
# # # from dotenv import load_dotenv

# # # # ── Credentials ────────────────────────────────────────────────────────────────
# # # _ENV_PATH = os.path.join(
# # #     os.path.dirname(os.path.abspath(__file__)),
# # #     "config", "credentials.env",
# # # )
# # # load_dotenv(_ENV_PATH)


# # # # ── boto3 helpers (same pattern as legacy db_s3_uploader.py) ──────────────────

# # # def _s3_client():
# # #     return boto3.client(
# # #         "s3",
# # #         aws_access_key_id     = os.environ["AWS_ACCESS_KEY_ID"],
# # #         aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"],
# # #         region_name           = os.environ.get("AWS_REGION", "ap-south-1"),
# # #     )


# # # def _bucket() -> str:
# # #     return os.environ["S3_BUCKET"]


# # # def _strip_s3_uri(s3_path: str) -> str:
# # #     """s3://bucket/a/b/c  →  a/b/c   |   a/b/c  →  a/b/c"""
# # #     if s3_path.startswith("s3://"):
# # #         parts = s3_path.replace("s3://", "").split("/", 1)
# # #         return parts[1] if len(parts) == 2 else parts[0]
# # #     return s3_path.strip()


# # # # ── Public API ─────────────────────────────────────────────────────────────────

# # # def download_video(s3_key: str, local_path: str) -> str:
# # #     """
# # #     Download a video file from S3.

# # #     Parameters
# # #     ──────────
# # #     s3_key     : S3 key (or full s3:// URI) as provided in the RabbitMQ message.
# # #     local_path : Absolute local path where the file will be written.

# # #     Returns local_path on success; raises on failure.
# # #     """
# # #     key = _strip_s3_uri(s3_key)
# # #     bkt = _bucket()
# # #     os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
# # #     print(f"[S3] Downloading  s3://{bkt}/{key}  →  {local_path}")
# # #     _s3_client().download_file(bkt, key, local_path)
# # #     return local_path


# # # def upload_frame(
# # #     frame:      np.ndarray,
# # #     journey_id: int,
# # #     filename:   str,
# # #     jpeg_quality: int = 85,
# # # ) -> str:
# # #     """
# # #     Encode a numpy frame as JPEG and upload it to S3.

# # #     Parameters
# # #     ──────────
# # #     frame        : BGR numpy array (OpenCV format).
# # #     journey_id   : used to build the S3 key prefix.
# # #     filename     : e.g. "phone_use_00-00-24.jpg"
# # #     jpeg_quality : JPEG compression quality (default 85).

# # #     Returns the S3 key (NOT a signed URL).
# # #     Example return value: "journeys/101/frames/phone_use_00-00-24.jpg"
# # #     """
# # #     s3_key = f"journeys/{journey_id}/frames/{filename}"
# # #     bkt    = _bucket()

# # #     # Resize to 640×360 (matches legacy ViolationStore._save_frame behaviour)
# # #     resized = cv2.resize(frame, (640, 360))
# # #     ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
# # #     if not ok:
# # #         raise RuntimeError(f"cv2.imencode failed for frame '{filename}'")

# # #     print(f"[S3] Uploading frame  →  s3://{bkt}/{s3_key}")
# # #     _s3_client().put_object(
# # #         Bucket      = bkt,
# # #         Key         = s3_key,
# # #         Body        = io.BytesIO(buf.tobytes()),
# # #         ContentType = "image/jpeg",
# # #     )
# # #     return s3_key


# # # def upload_frame_from_path(local_path: str, journey_id: int, filename: Optional[str] = None) -> str:
# # #     """
# # #     Upload a JPEG frame that has already been saved to disk.

# # #     Parameters
# # #     ──────────
# # #     local_path : Absolute path to the .jpg file on disk.
# # #     journey_id : used to build the S3 key prefix.
# # #     filename   : Override the S3 filename; defaults to os.path.basename(local_path).

# # #     Returns the S3 key.
# # #     """
# # #     fname  = filename or os.path.basename(local_path)
# # #     s3_key = f"journeys/{journey_id}/frames/{fname}"
# # #     bkt    = _bucket()

# # #     print(f"[S3] Uploading frame (from disk)  →  s3://{bkt}/{s3_key}")
# # #     _s3_client().upload_file(local_path, bkt, s3_key)
# # #     return s3_key


# # """
# # s3_service.py
# # ─────────────
# # S3 helpers for the Journey-based workflow.

# # • download_video()          — download one video from S3 to a local temp file.
# # • upload_frame()            — upload a numpy frame JPEG and return its S3 key.
# # • upload_frame_from_path()  — upload an already-saved JPEG and return its S3 key.

# # Changes from previous version
# # ──────────────────────────────
# # • upload_frame() and upload_frame_from_path() now accept an optional
# #   `folder_name` parameter so frames are written to:
# #       <folderName>/frames/<filename>
# #   rather than the old hard-coded:
# #       journeys/<journeyId>/frames/<filename>

# #   When folder_name is omitted (or None) the old path is used as a fallback
# #   so call sites that haven't been updated yet continue to work.

# # S3 frame key convention (new)
# # ─────────────────────────────
# #     <folderName>/frames/<filename>

# #   e.g. "journeys/1/2026-06-10/JRN-20260610-1-ABC123/frames/phone_use_00-00-24.jpg"

# # Spring Boot generates signed URLs from these keys.
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


# # # ── boto3 helpers ──────────────────────────────────────────────────────────────

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


# # def _frame_s3_key(
# #     filename:   str,
# #     journey_id: int,
# #     folder_name: Optional[str],
# # ) -> str:
# #     """
# #     Build the S3 key for a violation frame.

# #     Preferred (folder_name provided):
# #         <folderName>/frames/<filename>
# #     Fallback (no folder_name):
# #         journeys/<journeyId>/frames/<filename>
# #     """
# #     if folder_name:
# #         return f"{folder_name.rstrip('/')}/frames/{filename}"
# #     return f"journeys/{journey_id}/frames/{filename}"


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
# #     frame:        np.ndarray,
# #     journey_id:   int,
# #     filename:     str,
# #     jpeg_quality: int = 85,
# #     folder_name:  Optional[str] = None,   # NEW — preferred over journey_id fallback
# # ) -> str:
# #     """
# #     Encode a numpy frame as JPEG and upload it to S3.

# #     Parameters
# #     ──────────
# #     frame        : BGR numpy array (OpenCV format).
# #     journey_id   : used for fallback S3 key when folder_name is not given.
# #     filename     : e.g. "phone_use_00-00-24.jpg"
# #     jpeg_quality : JPEG compression quality (default 85).
# #     folder_name  : Journey folder prefix, e.g.
# #                    "journeys/1/2026-06-10/JRN-20260610-1-ABC123".
# #                    When provided, frame is uploaded to
# #                    "<folderName>/frames/<filename>".

# #     Returns the S3 key (NOT a signed URL).
# #     """
# #     s3_key = _frame_s3_key(filename, journey_id, folder_name)
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


# # def upload_frame_from_path(
# #     local_path:  str,
# #     journey_id:  int,
# #     filename:    Optional[str] = None,
# #     folder_name: Optional[str] = None,   # NEW — preferred over journey_id fallback
# # ) -> str:
# #     """
# #     Upload a JPEG frame that has already been saved to disk.

# #     Parameters
# #     ──────────
# #     local_path  : Absolute path to the .jpg file on disk.
# #     journey_id  : used for fallback S3 key when folder_name is not given.
# #     filename    : Override the S3 filename; defaults to os.path.basename(local_path).
# #     folder_name : Journey folder prefix (see upload_frame docstring).

# #     Returns the S3 key.
# #     """
# #     fname  = filename or os.path.basename(local_path)
# #     s3_key = _frame_s3_key(fname, journey_id, folder_name)
# #     bkt    = _bucket()

# #     print(f"[S3] Uploading frame (from disk)  →  s3://{bkt}/{s3_key}")
# #     _s3_client().upload_file(local_path, bkt, s3_key)
# #     return s3_key


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
# <folderName>/frames/<filename>

#   rather than the old hard-coded:

#       journeys/<journeyId>/frames/<filename>
 
#   When folder_name is omitted (or None) the old path is used as a fallback

#   so call sites that haven't been updated yet continue to work.
 
# S3 frame key convention (new)

# ─────────────────────────────
# <folderName>/frames/<filename>
 
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
# <folderName>/frames/<filename>

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


# def upload_text_log(
#     text:        str,
#     folder_name: str,
#     filename:    str,
# ) -> str:
#     """
#     Upload a plain-text log/report directly under the journey folder
#     (NOT under /frames), e.g.:

#         <folderName>/<filename>

#     e.g. "journeys/104/2026-06-19/JRN-20260619-104-011E57/JOB-262B2786AC81.txt"

#     Parameters
#     ──────────
#     text        : Full text content to write to the .txt file.
#     folder_name : Journey folder prefix, e.g.
#                   "journeys/104/2026-06-19/JRN-20260619-104-011E57".
#     filename    : e.g. "JOB-262B2786AC81.txt" (will have .txt appended
#                   if missing).

#     Returns the S3 key.
#     """
#     if not filename.lower().endswith(".txt"):
#         filename = f"{filename}.txt"

#     s3_key = f"{folder_name.rstrip('/')}/{filename}"
#     bkt    = _bucket()

#     print(f"[S3] Uploading log  →  s3://{bkt}/{s3_key}")
#     _s3_client().put_object(
#         Bucket      = bkt,
#         Key         = s3_key,
#         Body        = text.encode("utf-8"),
#         ContentType = "text/plain; charset=utf-8",
#     )
#     return s3_key



# # # """
# # # s3_service.py
# # # ─────────────
# # # S3 helpers for the Journey-based workflow.

# # # • download_video()  — download one video from S3 to a local temp file.
# # # • upload_frame()    — upload a single violation frame JPEG and return its S3 key.

# # # Reuses the same boto3 / env-var pattern as the legacy db_s3_uploader.py so no
# # # new credentials are needed.

# # # S3 frame key convention
# # # ───────────────────────
# # #     journeys/<journeyId>/frames/<filename>

# # # Spring Boot will generate signed URLs from these keys later.
# # # """

# # # from __future__ import annotations

# # # import io
# # # import os
# # # from typing import Optional

# # # import boto3
# # # import cv2
# # # import numpy as np
# # # from dotenv import load_dotenv

# # # # ── Credentials ────────────────────────────────────────────────────────────────
# # # _ENV_PATH = os.path.join(
# # #     os.path.dirname(os.path.abspath(__file__)),
# # #     "config", "credentials.env",
# # # )
# # # load_dotenv(_ENV_PATH)


# # # # ── boto3 helpers (same pattern as legacy db_s3_uploader.py) ──────────────────

# # # def _s3_client():
# # #     return boto3.client(
# # #         "s3",
# # #         aws_access_key_id     = os.environ["AWS_ACCESS_KEY_ID"],
# # #         aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"],
# # #         region_name           = os.environ.get("AWS_REGION", "ap-south-1"),
# # #     )


# # # def _bucket() -> str:
# # #     return os.environ["S3_BUCKET"]


# # # def _strip_s3_uri(s3_path: str) -> str:
# # #     """s3://bucket/a/b/c  →  a/b/c   |   a/b/c  →  a/b/c"""
# # #     if s3_path.startswith("s3://"):
# # #         parts = s3_path.replace("s3://", "").split("/", 1)
# # #         return parts[1] if len(parts) == 2 else parts[0]
# # #     return s3_path.strip()


# # # # ── Public API ─────────────────────────────────────────────────────────────────

# # # def download_video(s3_key: str, local_path: str) -> str:
# # #     """
# # #     Download a video file from S3.

# # #     Parameters
# # #     ──────────
# # #     s3_key     : S3 key (or full s3:// URI) as provided in the RabbitMQ message.
# # #     local_path : Absolute local path where the file will be written.

# # #     Returns local_path on success; raises on failure.
# # #     """
# # #     key = _strip_s3_uri(s3_key)
# # #     bkt = _bucket()
# # #     os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
# # #     print(f"[S3] Downloading  s3://{bkt}/{key}  →  {local_path}")
# # #     _s3_client().download_file(bkt, key, local_path)
# # #     return local_path


# # # def upload_frame(
# # #     frame:      np.ndarray,
# # #     journey_id: int,
# # #     filename:   str,
# # #     jpeg_quality: int = 85,
# # # ) -> str:
# # #     """
# # #     Encode a numpy frame as JPEG and upload it to S3.

# # #     Parameters
# # #     ──────────
# # #     frame        : BGR numpy array (OpenCV format).
# # #     journey_id   : used to build the S3 key prefix.
# # #     filename     : e.g. "phone_use_00-00-24.jpg"
# # #     jpeg_quality : JPEG compression quality (default 85).

# # #     Returns the S3 key (NOT a signed URL).
# # #     Example return value: "journeys/101/frames/phone_use_00-00-24.jpg"
# # #     """
# # #     s3_key = f"journeys/{journey_id}/frames/{filename}"
# # #     bkt    = _bucket()

# # #     # Resize to 640×360 (matches legacy ViolationStore._save_frame behaviour)
# # #     resized = cv2.resize(frame, (640, 360))
# # #     ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
# # #     if not ok:
# # #         raise RuntimeError(f"cv2.imencode failed for frame '{filename}'")

# # #     print(f"[S3] Uploading frame  →  s3://{bkt}/{s3_key}")
# # #     _s3_client().put_object(
# # #         Bucket      = bkt,
# # #         Key         = s3_key,
# # #         Body        = io.BytesIO(buf.tobytes()),
# # #         ContentType = "image/jpeg",
# # #     )
# # #     return s3_key


# # # def upload_frame_from_path(local_path: str, journey_id: int, filename: Optional[str] = None) -> str:
# # #     """
# # #     Upload a JPEG frame that has already been saved to disk.

# # #     Parameters
# # #     ──────────
# # #     local_path : Absolute path to the .jpg file on disk.
# # #     journey_id : used to build the S3 key prefix.
# # #     filename   : Override the S3 filename; defaults to os.path.basename(local_path).

# # #     Returns the S3 key.
# # #     """
# # #     fname  = filename or os.path.basename(local_path)
# # #     s3_key = f"journeys/{journey_id}/frames/{fname}"
# # #     bkt    = _bucket()

# # #     print(f"[S3] Uploading frame (from disk)  →  s3://{bkt}/{s3_key}")
# # #     _s3_client().upload_file(local_path, bkt, s3_key)
# # #     return s3_key


# # """
# # s3_service.py
# # ─────────────
# # S3 helpers for the Journey-based workflow.

# # • download_video()          — download one video from S3 to a local temp file.
# # • upload_frame()            — upload a numpy frame JPEG and return its S3 key.
# # • upload_frame_from_path()  — upload an already-saved JPEG and return its S3 key.

# # Changes from previous version
# # ──────────────────────────────
# # • upload_frame() and upload_frame_from_path() now accept an optional
# #   `folder_name` parameter so frames are written to:
# #       <folderName>/frames/<filename>
# #   rather than the old hard-coded:
# #       journeys/<journeyId>/frames/<filename>

# #   When folder_name is omitted (or None) the old path is used as a fallback
# #   so call sites that haven't been updated yet continue to work.

# # S3 frame key convention (new)
# # ─────────────────────────────
# #     <folderName>/frames/<filename>

# #   e.g. "journeys/1/2026-06-10/JRN-20260610-1-ABC123/frames/phone_use_00-00-24.jpg"

# # Spring Boot generates signed URLs from these keys.
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


# # # ── boto3 helpers ──────────────────────────────────────────────────────────────

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


# # def _frame_s3_key(
# #     filename:   str,
# #     journey_id: int,
# #     folder_name: Optional[str],
# # ) -> str:
# #     """
# #     Build the S3 key for a violation frame.

# #     Preferred (folder_name provided):
# #         <folderName>/frames/<filename>
# #     Fallback (no folder_name):
# #         journeys/<journeyId>/frames/<filename>
# #     """
# #     if folder_name:
# #         return f"{folder_name.rstrip('/')}/frames/{filename}"
# #     return f"journeys/{journey_id}/frames/{filename}"


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
# #     frame:        np.ndarray,
# #     journey_id:   int,
# #     filename:     str,
# #     jpeg_quality: int = 85,
# #     folder_name:  Optional[str] = None,   # NEW — preferred over journey_id fallback
# # ) -> str:
# #     """
# #     Encode a numpy frame as JPEG and upload it to S3.

# #     Parameters
# #     ──────────
# #     frame        : BGR numpy array (OpenCV format).
# #     journey_id   : used for fallback S3 key when folder_name is not given.
# #     filename     : e.g. "phone_use_00-00-24.jpg"
# #     jpeg_quality : JPEG compression quality (default 85).
# #     folder_name  : Journey folder prefix, e.g.
# #                    "journeys/1/2026-06-10/JRN-20260610-1-ABC123".
# #                    When provided, frame is uploaded to
# #                    "<folderName>/frames/<filename>".

# #     Returns the S3 key (NOT a signed URL).
# #     """
# #     s3_key = _frame_s3_key(filename, journey_id, folder_name)
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


# # def upload_frame_from_path(
# #     local_path:  str,
# #     journey_id:  int,
# #     filename:    Optional[str] = None,
# #     folder_name: Optional[str] = None,   # NEW — preferred over journey_id fallback
# # ) -> str:
# #     """
# #     Upload a JPEG frame that has already been saved to disk.

# #     Parameters
# #     ──────────
# #     local_path  : Absolute path to the .jpg file on disk.
# #     journey_id  : used for fallback S3 key when folder_name is not given.
# #     filename    : Override the S3 filename; defaults to os.path.basename(local_path).
# #     folder_name : Journey folder prefix (see upload_frame docstring).

# #     Returns the S3 key.
# #     """
# #     fname  = filename or os.path.basename(local_path)
# #     s3_key = _frame_s3_key(fname, journey_id, folder_name)
# #     bkt    = _bucket()

# #     print(f"[S3] Uploading frame (from disk)  →  s3://{bkt}/{s3_key}")
# #     _s3_client().upload_file(local_path, bkt, s3_key)
# #     return s3_key


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
# <folderName>/frames/<filename>

#   rather than the old hard-coded:

#       journeys/<journeyId>/frames/<filename>
 
#   When folder_name is omitted (or None) the old path is used as a fallback

#   so call sites that haven't been updated yet continue to work.
 
# S3 frame key convention (new)

# ─────────────────────────────
# <folderName>/frames/<filename>
 
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
# <folderName>/frames/<filename>

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

import json

import logging

import os

import threading

import time

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

log = logging.getLogger("s3_service")
 
 
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
 
# ── Download diagnostics ──────────────────────────────────────────────────────
#
# ROOT-CAUSE INSTRUMENTATION (temporary/diagnostic — not a behavior change):
# added to separate the possible explanations for why two journeys' download
# phases run at very different speeds:
#   (a) genuinely different file sizes                → size_bytes logged
#   (b) per-call S3 client construction overhead        → client_ms logged
#   (c) network stalls mid-transfer (connection hangs,   → max_stall_s logged,
#       then recovers)                                    WARNING if large
#   (d) botocore silently retrying failed part-requests  → retry_count logged
#   (e) plain bandwidth-limited transfer                 → throughput_MBps
# A single structured "[S3-DIAG]" line is emitted per download so all five
# can be read off directly and compared side-by-side across journeys/videos
# without guessing.
_STALL_WARN_THRESHOLD_SECONDS = 3.0


class _TransferProgress:
    """
    Passed as the `Callback` to boto3's download_file(). boto3/S3Transfer
    invokes this from its worker thread(s) every time a chunk of the file
    is written to disk, so consecutive calls to it are effectively a
    heartbeat of the transfer. A big gap between two consecutive calls
    means the transfer itself stalled (network hang, throttling, slow
    part) for that long — as opposed to being uniformly slow throughout.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._last_ts = time.perf_counter()
        self.bytes_transferred = 0
        self.chunk_count = 0
        self.max_gap_seconds = 0.0
        self.total_idle_seconds = 0.0

    def __call__(self, bytes_amount: int) -> None:
        now = time.perf_counter()
        with self._lock:
            gap = now - self._last_ts
            if gap > self.max_gap_seconds:
                self.max_gap_seconds = gap
            if gap > 0.25:  # ignore sub-250ms scheduling noise
                self.total_idle_seconds += gap
            self._last_ts = now
            self.bytes_transferred += bytes_amount
            self.chunk_count += 1


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

    wall_clock_start = time.time()
    t0 = time.perf_counter()
    client = _s3_client()
    t1 = time.perf_counter()
    client_construction_s = t1 - t0

    # Count every retried S3 API call botocore makes on THIS client for
    # THIS download (multipart transfers issue several GetObject calls
    # internally, each of which can be individually retried — this counts
    # across all of them). Scoped to this one client instance, which is
    # freshly constructed per call, so no manual deregistration needed.
    retry_count = {"n": 0}

    def _on_retry(**kwargs):
        retry_count["n"] += 1

    client.meta.events.register("needs-retry.s3.*", _on_retry)

    progress = _TransferProgress()
    transfer_error: Optional[BaseException] = None
    t2 = time.perf_counter()
    try:
        client.download_file(bkt, key, local_path, Callback=progress)
    except BaseException as exc:
        transfer_error = exc
        raise
    finally:
        t3 = time.perf_counter()
        transfer_s = t3 - t2
        total_s = t3 - t0
        size_bytes = 0
        try:
            size_bytes = os.path.getsize(local_path)
        except OSError:
            pass
        throughput_mbps = (size_bytes / (1024 * 1024) / transfer_s) if transfer_s > 0 else 0.0

        level = log.warning if progress.max_gap_seconds > _STALL_WARN_THRESHOLD_SECONDS else log.info
        level(
            "[S3-DIAG]  key=%s  wall_clock_start=%s  size_bytes=%d  "
            "client_construction_ms=%.0f  transfer_s=%.2f  total_s=%.2f  "
            "throughput_MBps=%.2f  chunks=%d  max_stall_s=%.2f  "
            "total_idle_s=%.2f  retry_count=%d  error=%s",
            key, time.strftime("%H:%M:%S", time.localtime(wall_clock_start)),
            size_bytes, client_construction_s * 1000, transfer_s, total_s,
            throughput_mbps, progress.chunk_count, progress.max_gap_seconds,
            progress.total_idle_seconds, retry_count["n"],
            repr(transfer_error) if transfer_error else "none",
        )

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


def download_frame(s3_key: str) -> np.ndarray:
    """
    Download a previously-uploaded evidence frame from S3 and decode it
    back into a BGR numpy array.

    Counterpart to upload_frame() / upload_frame_from_path() -- used by the
    journey-end LLM verification stage (analyzer.py::analyze_journey) to
    retrieve suspected evidence frames that were persisted to S3 during
    each video's own processing (see
    analyzer.py::_persist_and_release_video_evidence), instead of relying
    on in-memory frame data that used to be kept in the worker until the
    whole journey finished.

    Parameters
    ──────────
    s3_key : S3 key (or full s3:// URI) as returned by upload_frame() /
             upload_frame_from_path().

    Returns a BGR numpy array. Raises on failure (missing object, decode
    error, etc.) -- callers should catch and handle this non-fatally, the
    same way existing upload_frame()/download_video() call sites do.
    """
    key = _strip_s3_uri(s3_key)
    bkt = _bucket()

    print(f"[S3] Downloading frame  <-  s3://{bkt}/{key}")
    obj  = _s3_client().get_object(Bucket=bkt, Key=key)
    data = obj["Body"].read()

    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Failed to decode frame downloaded from s3://{bkt}/{key}")
    return frame


def upload_text_log(
    text:        str,
    folder_name: str,
    filename:    str,
) -> str:
    """
    Upload a plain-text log/report directly under the journey folder
    (NOT under /frames), e.g.:

        <folderName>/<filename>

    e.g. "journeys/104/2026-06-19/JRN-20260619-104-011E57/JOB-262B2786AC81.txt"

    Parameters
    ──────────
    text        : Full text content to write to the .txt file.
    folder_name : Journey folder prefix, e.g.
                  "journeys/104/2026-06-19/JRN-20260619-104-011E57".
    filename    : e.g. "JOB-262B2786AC81.txt" (will have .txt appended
                  if missing).

    Returns the S3 key.
    """
    if not filename.lower().endswith(".txt"):
        filename = f"{filename}.txt"

    s3_key = f"{folder_name.rstrip('/')}/{filename}"
    bkt    = _bucket()

    print(f"[S3] Uploading log  →  s3://{bkt}/{s3_key}")
    _s3_client().put_object(
        Bucket      = bkt,
        Key         = s3_key,
        Body        = text.encode("utf-8"),
        ContentType = "text/plain; charset=utf-8",
    )
    return s3_key


def upload_json_result(
    payload:     dict,
    folder_name: str,
    filename:    str,
) -> str:
    """
    Upload a JSON document (e.g. the full completion payload) directly
    under the journey folder (NOT under /frames), e.g.:

        <folderName>/<filename>

    e.g. "journeys/104/2026-06-19/JRN-20260619-104-011E57/JOB-262B2786AC81_result.json"

    Intended use: persist the exact JSON we are about to POST to the Java
    backend to S3 first, so there is always an authoritative, immutable
    record of what was sent — independent of whatever happens on the
    backend/DB side afterwards (deserialization bugs, backend downtime,
    retries that end up sending a mutated dict, etc).

    Parameters
    ──────────
    payload     : JSON-serializable dict (e.g. completion_dict, the exact
                  object about to be handed to send_completed()).
    folder_name : Journey folder prefix, e.g.
                  "journeys/104/2026-06-19/JRN-20260619-104-011E57".
    filename    : e.g. "JOB-262B2786AC81_result.json" (will have .json
                  appended if missing).

    Returns the S3 key. Raises on failure — callers should catch and log,
    since a failed upload here must never block the actual backend
    callback from being sent.
    """
    if not filename.lower().endswith(".json"):
        filename = f"{filename}.json"

    s3_key = f"{folder_name.rstrip('/')}/{filename}"
    bkt    = _bucket()

    body = json.dumps(payload, indent=2, default=str).encode("utf-8")

    print(f"[S3] Uploading JSON result  →  s3://{bkt}/{s3_key}  ({len(body)} bytes)")
    _s3_client().put_object(
        Bucket      = bkt,
        Key         = s3_key,
        Body        = body,
        ContentType = "application/json; charset=utf-8",
    )
    return s3_key