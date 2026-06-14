"""
publish_test_job.py
────────────────────
Simulates Spring Boot publishing a message to the 'analysis.jobs' queue.

Usage:
    python publish_test_job.py

Edit JOB_MESSAGE below before running — in particular set "s3Key" to a
real object that exists in your S3 bucket (railway-cvvrs), e.g. by
uploading your local xyz.mp4 to S3 first:

    aws s3 cp xyz.mp4 s3://railway-cvvrs/journeys/101/original/video1.mp4

Or, if you don't want to touch S3 yet, see the "LOCAL FILE TEST" note
at the bottom of this file for a way to bypass S3 entirely.
"""

from __future__ import annotations

import json
import os

import pika
from dotenv import load_dotenv

_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "credentials.env")
load_dotenv(_ENV_PATH)

RABBITMQ_URL = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
QUEUE_NAME   = os.environ.get("ANALYSIS_QUEUE", "analysis.jobs")

# ── Edit this to match your test scenario ────────────────────────────────────
JOB_MESSAGE = {
    "jobId":     "TEST-JOB-001",
    "journeyId": 101,
    "videos": [
        {
            "videoId":    1001,
            "sequenceNo": 1,
            "s3Key":      "journeys/101/original/video1.mp4",
        },
        # Add more videos here if testing multi-video offsets:
        # {
        #     "videoId":    1002,
        #     "sequenceNo": 2,
        #     "s3Key":      "journeys/101/original/video2.mp4",
        # },
    ],
}


def main() -> None:
    params     = pika.URLParameters(RABBITMQ_URL)
    connection = pika.BlockingConnection(params)
    channel    = connection.channel()

    channel.queue_declare(queue=QUEUE_NAME, durable=True)

    body = json.dumps(JOB_MESSAGE)
    channel.basic_publish(
        exchange    = "",
        routing_key = QUEUE_NAME,
        body        = body,
        properties  = pika.BasicProperties(
            delivery_mode = 2,  # persistent
            content_type  = "application/json",
        ),
    )

    print(f"Published to queue '{QUEUE_NAME}':")
    print(json.dumps(JOB_MESSAGE, indent=2))

    connection.close()


if __name__ == "__main__":
    main()