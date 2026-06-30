from __future__ import annotations

import json
import os

import pika
from dotenv import load_dotenv

_ENV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config", "credentials.env"
)
load_dotenv(_ENV_PATH)

RABBITMQ_URL = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
QUEUE_NAME = os.environ.get("ANALYSIS_QUEUE", "analysis.jobs")
EXCHANGE_NAME = os.environ.get("ANALYSIS_EXCHANGE", "analysis.exchange")
ROUTING_KEY = os.environ.get("ANALYSIS_ROUTING", "analysis.job.created")

# ── Edit this to match your test scenario ────────────────────────────────────
JOB_MESSAGE = {
    "jobId": "TEST-JOB-BATCH-NEW",
    "journeyId": 109,
    "trainDetailId": 1,
    "folderName": "journeys/109/2026-06-22/BATCH-NEW",
    "priority": "NORMAL",
    "videos": [
        {
            "videoId": 3001,
            "sequenceNo": 1,
            "originalFileName": "Copy of ch04_20250330013939_20250330022333.mp4",
            "s3Key": "D:/PROJECTS/Sconex Workspace/CVVRS/cvvrs-ai/test_videos/railway-20260610T090253Z-3-008/railway/332 LN-37073 LP-S R KAMBLE(DD) ALP-RAJESH KR(DD) TN-22172 SEC-MMR-PUNE  DATE 30.03.2025/Copy of ch04_20250330013939_20250330022333.mp4",
        },
        {
            "videoId": 3002,
            "sequenceNo": 2,
            "originalFileName": "Copy of ch05_20250330063056_20250330071342.mp4",
            "s3Key": "D:/PROJECTS/Sconex Workspace/CVVRS/cvvrs-ai/test_videos/railway-20260610T090253Z-3-008/railway/332 LN-37073 LP-S R KAMBLE(DD) ALP-RAJESH KR(DD) TN-22172 SEC-MMR-PUNE  DATE 30.03.2025/Copy of ch05_20250330063056_20250330071342.mp4",
        },
        {
            "videoId": 3003,
            "sequenceNo": 3,
            "originalFileName": "Copy of ch04_20250330022333_20250330030659.mp4",
            "s3Key": "D:/PROJECTS/Sconex Workspace/CVVRS/cvvrs-ai/test_videos/railway-20260610T090253Z-3-008/railway/332 LN-37073 LP-S R KAMBLE(DD) ALP-RAJESH KR(DD) TN-22172 SEC-MMR-PUNE  DATE 30.03.2025/Copy of ch04_20250330022333_20250330030659.mp4",
        },
    ],
}


def main() -> None:
    params = pika.URLParameters(RABBITMQ_URL)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()

    # Declare exchange + queue + binding so the test works even without Spring Boot running
    channel.exchange_declare(
        exchange=EXCHANGE_NAME,
        exchange_type="direct",
        durable=True,
    )
    channel.queue_declare(queue=QUEUE_NAME, durable=True)
    channel.queue_bind(
        queue=QUEUE_NAME,
        exchange=EXCHANGE_NAME,
        routing_key=ROUTING_KEY,
    )

    body = json.dumps(JOB_MESSAGE)
    channel.basic_publish(
        exchange=EXCHANGE_NAME,
        routing_key=ROUTING_KEY,
        body=body,
        properties=pika.BasicProperties(
            delivery_mode=2,  # persistent
            content_type="application/json",
        ),
    )

    print(f"Published to exchange '{EXCHANGE_NAME}' / routing '{ROUTING_KEY}':")
    print(json.dumps(JOB_MESSAGE, indent=2))

    connection.close()


if __name__ == "__main__":
    main()
