import boto3
import os

from common import io_utils
from processor import Processor


class TranscriptionProcessor(Processor):
    def __init__(self):
        self._transcribe = boto3.client('transcribe')
        super().__init__()

    def process(self, data):
        super().process(data)

        prefix = io_utils.get_video_chunk_base_key(self._video_key)
        video_chunks = self._s3.list_objects_v2(Bucket=self._bucketName, Prefix=prefix)

        for video_chunk in video_chunks['Contents']:
            video_key = video_chunk['Key'];

            job_name = self.__generate_transcription_job_name(video_key)
            print(job_name)
            # response = self._transcribe.start_transcription_job(
            #     TranscriptionJobName=job_name,
            #     LanguageCode='es-ES',
            # )


    @staticmethod
    def __generate_transcription_job_name(video_key):
        return video_key
