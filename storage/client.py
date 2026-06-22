import boto3
import os
from botocore.exceptions import ClientError

def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION")
    )

def ensure_bucket_exists():
    s3 = get_s3_client()
    bucket_name = os.getenv("S3_BUCKET_NAME")
    try:
        s3.head_bucket(Bucket=bucket_name)
    except ClientError:
        s3.create_bucket(Bucket=bucket_name)

def upload_to_s3(file_obj, object_name: str):
    s3 = get_s3_client()
    s3.upload_fileobj(file_obj, os.getenv("S3_BUCKET_NAME"), object_name)
    return object_name

def download_from_s3(object_name: str, download_path: str):
    s3 = get_s3_client()
    s3.download_file(os.getenv("S3_BUCKET_NAME"), object_name, download_path)
    return download_path
