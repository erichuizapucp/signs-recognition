import logging
import json
import os

from processor import Processor
from common import io_utils


class AudioTranscriptionAggregationProcessor(Processor):
    __transcription_files_prefix = "transcriptions/"
    __transcriptions_folder = "transcriptions"

    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger(__name__)

    def process(self, data):
        transcription_folders = self._s3.list_objects_v2(
            Bucket=self._bucketName,
            StartAfter=self.__transcription_files_prefix)

        local_file_name = data['localFileName']
        s3_key = data['s3Key']

        if self.s3_exist_object(s3_key):
            self.logger.debug('audio transcription aggregation already available in s3 at: %s', s3_key)
            return

        if os.path.exists(local_file_name):
            self.logger.debug('local file %s exist', local_file_name)
            os.remove(local_file_name)
            self.logger.debug('local file %s was deleted', local_file_name)

        io_utils.check_path_dir(local_file_name)

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

        self.logger.debug('audio transcription aggregation is available locally at: %s', local_file_name)

        self.upload_file(local_file_name, s3_key)
        self.logger.debug('audio transcription aggregation uploaded to: %s', s3_key)
