import boto3
import os
from botocore.exceptions import ClientError


class Processor:
    def __init__(self):
        self._video_key = None
        self._video_file_path = None
        self._s3 = boto3.client('s3')
        self._bucketName = os.environ['S3_BUCKET']
        self._binary_folder = os.environ['BINARY_FOLDER']
        self._source_videos_base = 'not-annotated-videos'
        self._always_upload = os.environ['ALWAYS_UPLOAD_S3'] == 'True'

    def process(self, data):
        self._video_key = data['id']

    def _generate_video_file_path(self, video_key):
        return os.path.join(self._binary_folder, video_key)

    def upload_file(self, file_path, key):
        if self._always_upload or not self.s3_exist_object(key):
            self._s3.upload_file(file_path, self._bucketName, key)
            print('the file was uploaded to {}/{}'.format(self._bucketName, key))
        else:
            print('the file is already available at: {}'.format(key))

    def s3_exist_object(self, key):
        try:
            self._s3.head_object(Bucket=self._bucketName, Key=key)
        except ClientError as e:
            return int(e.response['Error']['Code']) != 404
        return True
