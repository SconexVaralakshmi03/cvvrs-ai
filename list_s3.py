import os
import boto3
from dotenv import load_dotenv

load_dotenv("config/credentials.env")

# Try to connect to S3
s3 = boto3.client('s3',
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    region_name=os.environ.get("AWS_REGION", "ap-south-1")
)

buckets_to_check = [
    os.environ.get("S3_BUCKET", "railway-cvvrs-violations"),
    "railway-cvvrs"  # Standard data bucket guess
]

print("🔍 Searching your AWS S3 account for .mp4 videos...\n")

for bucket in buckets_to_check:
    print(f"Checking bucket: '{bucket}'...")
    try:
        paginator = s3.get_paginator('list_objects_v2')
        found = False
        for page in paginator.paginate(Bucket=bucket):
            if 'Contents' in page:
                for obj in page['Contents']:
                    if obj['Key'].endswith('.mp4'):
                        print(f"  ✅ FOUND S3 KEY: {obj['Key']}")
                        found = True
        
        if not found:
            print("  ❌ No .mp4 files found in this bucket.")
    except Exception as e:
        print(f"  ⚠️ Could not access bucket (maybe it doesn't exist or permissions are restricted).")
    print("-" * 50)
