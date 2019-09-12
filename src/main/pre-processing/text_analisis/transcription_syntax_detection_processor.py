import logging
import json
import os

from processor import Processor
from common import io_utils


class TranscriptionSyntaxDetectionProcessor(Processor):
    __transcription_files_prefix = "transcriptions/"
    __nouns_file_name = "nouns-detection.txt"
    __numbers_file_name = "numbers-detection.txt"

    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger(__name__)

    def process(self, data):
        transcription_folders = self._s3.list_objects_v2(
            Bucket=self._bucketName,
            StartAfter=self.__transcription_files_prefix)

        local_path = data['localPath']
        s3_path = data['s3Path']

        nouns_local_file_path = "{}{}".format(local_path, self.__nouns_file_name)
        nouns_s3_key = "{}{}".format(s3_path, self.__nouns_file_name)
        self.__check_local_file(nouns_local_file_path)
        self.__check_s3_key(nouns_s3_key)

        numbers_local_file_path = "{}{}".format(local_path, self.__numbers_file_name)
        numbers_s3_key = "{}{}".format(s3_path, self.__numbers_file_name)
        self.__check_local_file(numbers_local_file_path)
        self.__check_s3_key(numbers_s3_key)

        for transcription_dict in transcription_folders['Contents']:
            transcription_key = transcription_dict['Key']
            if not transcription_key.endswith('.json'):
                continue

            self.logger.debug('Extracting audio transcription from: %s', transcription_key)

            file_obj = self._s3.get_object(Bucket=self._bucketName, Key=transcription_key)
            file_content = file_obj['Body'].read().decode('utf-8')
            json_content = json.loads(file_content)

            transcriptions = json_content['results']['transcripts']
            for transcription in transcriptions:
                with open(local_file_name, 'a+') as tf:
                    tf.write(transcription['transcript'] + '\n')

                file_size = os.path.getsize(local_file_name)
                if file_size >= 4500:
                    print()

        self.logger.debug('audio transcription transcription-syntax-detection is available locally at: %s', local_file_name)

        self.upload_file(local_file_name, s3_key)
        self.logger.debug('audio transcription transcription-syntax-detection uploaded to: %s', s3_key)

    def __check_local_file(self, local_file_path):
        if os.path.exists(local_file_path):
            self.logger.debug('local file %s exist', local_file_path)
            os.remove(local_file_path)
            self.logger.debug('local file %s was deleted', local_file_path)

        io_utils.check_path_dir(local_file_path)

    def __check_s3_key(self, s3_key):
        if self.s3_exist_object(s3_key):
            self.logger.debug('audio transcription transcription-syntax-detection already available in s3 at: %s',
                              s3_key)
            self.s3_delete_object(s3_key)
            self.logger.debug('audio transcription %s deleted', s3_key)
