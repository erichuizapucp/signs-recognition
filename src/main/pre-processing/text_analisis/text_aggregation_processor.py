import logging
import json

from processor import Processor


class TextAggregationProcessor(Processor):
    __transcription_files_base = "transcriptions"

    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger(__name__)

    def process(self, data):
        transcription_folders = self._s3.list_objects_v2(
            Bucket=self._bucketName,
            Prefix=self.__transcription_files_base)

        for folder in transcription_folders['Contents']:
            folder_key = folder['Key'] + '/chunks/'
            transcription_files = self._s3.list_objects_v2(Bucket=self._bucketName, Prefix=folder_key)

            for file in transcription_files['Contents']:
                file_key = file['Key']
                file_obj = self._s3.get_object(Bucket=self._bucketName, Key=file_key)
                file_content = file_obj['Body'].read().decode('utf-8')
                json_content = json.loads(file_content)

                transcriptions = json_content['results']['transcripts']
                for transcription in transcriptions:
                    with open('/transcriptions/..') as tf:
                        tf.write(transcription['transcript'])
