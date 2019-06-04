import boto3
import botocore

class AudioHandler:
    BUCKET_NAME = ""

    def handle(self):
        print("handle audio extraction from Bucket")
