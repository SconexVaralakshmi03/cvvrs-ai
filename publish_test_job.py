# """
# publish_test_job.py
# ────────────────────
# Simulates Spring Boot publishing a message to the 'dev.dev.analysis.jobs' queue.

# Usage:
#     python publish_test_job.py

# Edit JOB_MESSAGE below before running — in particular set "s3Key" to a
# real object that exists in your S3 bucket (railway-cvvrs), e.g. by
# uploading your local xyz.mp4 to S3 first:

#     aws s3 cp xyz.mp4 s3://railway-cvvrs/journeys/101/original/video1.mp4

# Or, if you don't want to touch S3 yet, see the "LOCAL FILE TEST" note
# at the bottom of this file for a way to bypass S3 entirely.
# """

# from __future__ import annotations

# import json
# import os

# import pika
# from dotenv import load_dotenv

# _ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "credentials.env")
# load_dotenv(_ENV_PATH)

# RABBITMQ_URL = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
# QUEUE_NAME   = os.environ.get("ANALYSIS_QUEUE", "dev.dev.analysis.jobs")

# # ── Edit this to match your test scenario ────────────────────────────────────
# JOB_MESSAGE = {
#     "jobId":     "TEST-JOB-001",
#     "journeyId": 101,
#     "videos": [
#         {
#             "videoId":    1001,
#             "sequenceNo": 1,
#             "s3Key":      "journeys/101/original/video1.mp4",
#         },
#         # Add more videos here if testing multi-video offsets:
#         # {
#         #     "videoId":    1002,
#         #     "sequenceNo": 2,
#         #     "s3Key":      "journeys/101/original/video2.mp4",
#         # },
#     ],
# }


# def main() -> None:
#     params     = pika.URLParameters(RABBITMQ_URL)
#     connection = pika.BlockingConnection(params)
#     channel    = connection.channel()

#     channel.queue_declare(queue=QUEUE_NAME, durable=True)

#     body = json.dumps(JOB_MESSAGE)
#     channel.basic_publish(
#         exchange    = "",
#         routing_key = QUEUE_NAME,
#         body        = body,
#         properties  = pika.BasicProperties(
#             delivery_mode = 2,  # persistent
#             content_type  = "application/json",
#         ),
#     )

#     print(f"Published to queue '{QUEUE_NAME}':")
#     print(json.dumps(JOB_MESSAGE, indent=2))

#     connection.close()


# if __name__ == "__main__":
#     main()


"""
publish_test_job.py
────────────────────
Simulates Spring Boot publishing a message to the 'dev.dev.analysis.jobs' queue
via the 'dev.dev.analysis.exchange' exchange (routing key: dev.dev.dev.analysis.jobs.created).

Usage
─────
    python publish_test_job.py

Edit JOB_MESSAGE below to match your test scenario.
The s3Key for each video must exist in your S3 bucket, e.g.:

    aws s3 cp front_cabin.mp4 \\
        s3://railway-cvvrs/journeys/1/2026-06-10/JRN-TEST/original/001_front_cabin.mp4

Message shape (aligned with CVVRS API documentation)
─────────────────────────────────────────────────────
{
    "jobId":         str   — unique job identifier
    "journeyId":     int   — ID of the Journey record in Spring Boot
    "trainDetailId": int   — ID of the TrainDetail record
    "folderName":    str   — S3 folder prefix for this journey
    "priority":      str   — "NORMAL" | "HIGH"
    "videos": [
        {
            "videoId":          int  — ID of the JourneyVideo record
            "sequenceNo":       int  — 1-based ordering
            "originalFileName": str  — original upload filename
            "s3Key":            str  — full S3 key of the uploaded video
        }
    ]
}
"""

from __future__ import annotations

import json
import os

import pika
from dotenv import load_dotenv

_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "credentials.env")
load_dotenv(_ENV_PATH)

RABBITMQ_URL  = os.environ.get("RABBITMQ_URL",       "amqp://guest:guest@localhost:5672/")
QUEUE_NAME    = os.environ.get("ANALYSIS_QUEUE",    "dev.analysis.jobs")
EXCHANGE_NAME = os.environ.get("ANALYSIS_EXCHANGE", "dev.analysis.exchange")
ROUTING_KEY   = os.environ.get("ANALYSIS_ROUTING",  "dev.analysis.jobs.created")

# ── Edit this to match your test scenario ────────────────────────────────────
JOB_MESSAGE = {
    "jobId":         "TEST-JOB-0061",
    "journeyId":     21,
    "trainDetailId": 8,
    "folderName":    "journeys/5/2026-06-19/JRN-20260619-3-keshika",
    "priority":      "NORMAL",
    "videos": [
        {
            "videoId":          10010,
            "sequenceNo":       9,
            "originalFileName": "changescheck.mp4",
            "s3Key":            "journeys/91/2026-06-16/JRN-20260616-91-135C3C/original/001_xyz 1.mp4 ",
        },
        # Uncomment to test multi-video offset continuity:
        # {
        #     "videoId":          1002,
        #     "sequenceNo":       2,
        #     "originalFileName": "rear_cabin.mp4",
        #     "s3Key":            "journeys/1/2026-06-10/JRN-20260610-1-ABC123/original/002_rear_cabin.mp4",
        # },
    ],
}


def main() -> None:
    params     = pika.URLParameters(RABBITMQ_URL)
    connection = pika.BlockingConnection(params)
    channel    = connection.channel()

    # Declare exchange + queue + binding so the test works even without Spring Boot running
    channel.exchange_declare(
        exchange      = EXCHANGE_NAME,
        exchange_type = "direct",
        durable       = True,
    )
    #channel.queue_declare(queue=QUEUE_NAME, durable=True)
    channel.queue_bind(
        queue       = QUEUE_NAME,
        exchange    = EXCHANGE_NAME,
        routing_key = ROUTING_KEY,
    )

    body = json.dumps(JOB_MESSAGE)
    channel.basic_publish(
        exchange    = EXCHANGE_NAME,
        routing_key = ROUTING_KEY,
        body        = body,
        properties  = pika.BasicProperties(
            delivery_mode = 2,               # persistent
            content_type  = "application/json",
        ),
    )

    print(f"Published to exchange '{EXCHANGE_NAME}' / routing '{ROUTING_KEY}':")
    print(json.dumps(JOB_MESSAGE, indent=2))

    connection.close()


if __name__ == "__main__":
    main()