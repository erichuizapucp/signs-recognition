import logging
import boto3
import datetime
import io_utils

from common.processor import Processor


class TranscriptionProcessor(Processor):
    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger(__name__)
        self._transcribe = boto3.client('transcribe')

    def process(self, data):
        super().process(data)

        prefix = io_utils.get_video_chunk_base_key(self._video_key)
        self.logger.debug('using %s for video chunks prefix', prefix)

        video_chunks = self._s3.list_objects_v2(Bucket=self._bucketName, Prefix=prefix)
        if not video_chunks or 'Contents' not in video_chunks:
            raise RuntimeError('There are not video chunks available for processing at the following location: {}'
                               .format(prefix))

        self.logger.debug('%d video chunks were found', len(video_chunks['Contents']))

        for video_chunk in video_chunks['Contents']:
            video_key = video_chunk['Key']
            self.logger.debug('Starting %s audio transcription', video_key)

            job_name = self.__generate_transcription_job_name(video_key)
            # response = self._transcribe.start_transcription_job(
            #     TranscriptionJobName=job_name,
            #     LanguageCode='es-ES',
            # )

    @staticmethod
    def __generate_transcription_job_name(video_key: str):
        str_timestamp = str(datetime.datetime.now().timestamp()).replace('.', '-')
        key_no_extension = io_utils.get_filename_without_extension(video_key)

        return "_".join([str_timestamp, key_no_extension])
