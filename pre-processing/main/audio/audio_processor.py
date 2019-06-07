import boto3
import os


class AudioProcessor:
    def __init__(self):
        self.s3 = boto3.client('s3')
        self.bucketName = os.environ('S3_Bucket')

    def process(self, payload):
        video_id = payload['id']
        video = self.bucket.get_object(self.bucketName, video_id)

        data = video['body'].read()

