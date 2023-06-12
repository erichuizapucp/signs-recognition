import boto3
import logging
import os

from abc import ABC, abstractmethod
from pre_processing.common import io_utils
from botocore.exceptions import ClientError


class Processor(ABC):
    def __init__(self):
        self.video_file_path = None
        self.s3 = boto3.client('s3')
        self.bucketName = os.getenv('S3_BUCKET', None)
        self.work_dir = os.getenv('WORK_DIR', './')
        self.binary_folder = os.path.join(self.work_dir, io_utils.binary_folder)
        self.source_videos_base = 'not-annotated-videos'
        self.always_upload = os.getenv('ALWAYS_UPLOAD_S3', 'True') == 'True'

        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def process(self, data):
        raise NotImplementedError("Method is not implemented.")

    def generate_video_file_path(self, video_key):
        return os.path.join(self.binary_folder, video_key)

    def upload_file(self, file_path, key):
        if self.always_upload or not self.s3_exist_object(key):
            self.s3.upload_file(file_path, self.bucketName, key)
            print('the file was uploaded to {}/{}'.format(self.bucketName, key))
        else:
            print('the file is already available at: {}'.format(key))

    def download_file(self, key, file_path):
        self.s3.download_file(self.bucketName, key, file_path)
        self.logger.debug('%s was downloaded to %s', key, file_path)

    def s3_exist_object(self, key):
        try:
            self.s3.head_object(Bucket=self.bucketName, Key=key)
        except ClientError as e:
            self.logger.error(e)
            return False
        return True

    def s3_move_object(self, source_key, destination_key):
        self.s3.copy_object(Bucket=self.bucketName, Key=destination_key,
                            CopySource={'Bucket': self.bucketName, 'Key': source_key})

    def s3_delete_object(self, source_key):
        self.s3.delete_object(Bucket=self.bucketName, Key=source_key)

    def s3_batch_download_files(self, s3_folder_path, excluded_path, allowed_extensions):
        s3_files = self.s3.list_objects_v2(
            Bucket=self.bucketName,
            Prefix=s3_folder_path)

        for s3_file in s3_files['Contents']:
            file_key = s3_file['Key']

            if excluded_path in file_key:
                continue

            if not io_utils.get_file_extension(file_key) in allowed_extensions:
                continue

            file_local_path = os.path.join(self.work_dir, file_key)
            io_utils.check_path_file(file_local_path)
            self.s3.download_file(self.bucketName, file_key, file_local_path)
            self.logger.debug('%s was downloaded to %s', file_key, file_local_path)
