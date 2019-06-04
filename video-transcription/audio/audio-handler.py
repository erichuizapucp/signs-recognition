import boto3
import botocore
import os


class AudioHandler:
    def __init__(self):
        self.bucketName = os.environ("S3_Bucket")
        self.S3 = boto3.resource("s3")
        self.Bucket = self.S3.Bucket(self.bucketName)

    def handle(self, video_id):
        self.Bucket.download_file(video_id, "")
        print("handle audio extraction from Bucket")
