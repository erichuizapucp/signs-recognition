import logging
import boto3
import datetime
import time
import io_utils

from common.processor import Processor


class TranscriptionProcessor(Processor):
    preferred_language_code = 'es-ES'
    preferred_vocabulary_name = 'peruvian-expressions'
    preferred_max_speakers = 3

    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger(__name__)
        self._transcribe = boto3.client('transcribe')

    def process(self, data, max_no_jobs=1):
        super().process(data)

        prefix = io_utils.get_video_chunk_base_key(self._video_key)
        self.logger.debug('using %s for video chunks prefix', prefix)

        video_chunks = self._s3.list_objects_v2(Bucket=self._bucketName, Prefix=prefix)
        if not (video_chunks and 'Contents' in video_chunks):
            raise RuntimeError('There are not video chunks available for processing at the following location: {}'
                               .format(prefix))

        self.logger.debug('%d video chunks were found', len(video_chunks['Contents']))

        for index, video_chunk in enumerate(video_chunks['Contents']):
            try:
                if index >= max_no_jobs:
                    break

                video_key = video_chunk['Key']
                resp = self.__start_audio_transcription(video_key)

                job_name = resp['TranscriptionJob']['TranscriptionJobName']
                resp = self.__wait_for_job_completion(job_name)

                transcript_file_uri = resp['TranscriptionJob']['Transcript']['TranscriptFileUri']
                if self.s3_move_object(transcript_file_uri, '' + transcript_file_uri):
                    print('')

            except Exception as e:
                self.logger.error(e)

    def __start_audio_transcription(self, video_key):
        job_name = self.__generate_transcription_job_name(video_key)
        media_format = io_utils.get_file_extension(video_key)
        media_url = self.__get_media_file_uri(video_key)

        response = self._transcribe.start_transcription_job(
            TranscriptionJobName=job_name,
            LanguageCode=self.preferred_language_code,
            MediaFormat=media_format,
            Media={
                'MediaFileUri': media_url
            },
            OutputBucketName=self._bucketName,
            Settings={
                'VocabularyName': self.preferred_vocabulary_name,
                'ShowSpeakersLabels': True,
                'MaxSpeakersLabels': self.preferred_max_speakers,
                'ChannelIdentification': False
            }
        )

        self.logger.debug('A new audio transcription job was started: \tJob Name: %s \tMedia Format: %s \tMedia '
                          'File Url: %s', job_name, media_format, media_url)
        return response

    def __wait_for_job_completion(self, job_name):
        while True:
            resp = self._transcribe.get_transcription_job(TranscriptionJobName=job_name)
            status = resp['TranscriptionJob']['TranscriptionJobStatus']
            if status in ['COMPLETED', 'FAILED']:
                self.logger.debug('The transcription job %s has completed with status of: %s', status)
                if status == 'FAILED':
                    failed_reason = resp['TranscriptionJob']['TranscriptionJob']
                    raise RuntimeError(failed_reason)
                break
            self.logger.debug('The transcription job %s is not completed yet', job_name)
            time.sleep(5)

        return resp

    def __get_media_file_uri(self, video_key):
        return 'https://{}.s3.amazonaws.com/{}'.format(self._bucketName, video_key)

    @staticmethod
    def __generate_transcription_job_name(video_key: str):
        str_timestamp = str(datetime.datetime.now().timestamp()).replace('.', '-')
        key_no_extension = io_utils.get_filename_without_extension(video_key)

        return '_'.join([str_timestamp, key_no_extension])
