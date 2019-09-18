import boto3
import logging
import os
from common import io_utils

from botocore.exceptions import ClientError


class Processor:
    def __init__(self):
        self._video_key = None
        self._video_file_path = None
        self._s3 = boto3.client('s3')
        self._bucketName = os.getenv('S3_BUCKET', None)
        self._work_dir = os.getenv('WORK_DIR', './')
        self._binary_folder = os.path.join(self._work_dir, io_utils.binary_folder)
        self._source_videos_base = 'not-annotated-videos'
        self._always_upload = os.getenv('ALWAYS_UPLOAD_S3', 'True') == 'True'

        self.logger = logging.getLogger(__name__)

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
            self.logger.error(e)
            return False
        return True

    def s3_move_object(self, source_key, destination_key):
        self._s3.copy_object(Bucket=self._bucketName, Key=destination_key,
                             CopySource={'Bucket': self._bucketName, 'Key': source_key})

    def s3_delete_object(self, source_key):
        self._s3.delete_object(Bucket=self._bucketName, Key=source_key)

    def s3_batch_download_files(self, s3_folder_path, allowed_extensions):
        s3_files = self._s3.list_objects_v2(
            Bucket=self._bucketName,
            Prefix=s3_folder_path)

        for s3_file in s3_files['Contents']:
            file_key = s3_file['Key']

            if not io_utils.get_file_extension(file_key) in allowed_extensions:
                continue

            file_local_path = os.path.join(self._work_dir, file_key)
            io_utils.check_path_dir(file_local_path)
            self._s3.download_file(self._bucketName, file_key, file_local_path)
            self.logger.debug('%s was downloaded to %s', file_key, file_local_path)
