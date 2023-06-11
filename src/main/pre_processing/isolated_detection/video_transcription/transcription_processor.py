import logging
import boto3
import datetime
import time
import io_utils

from pre_processing.common.processor import Processor


class TranscriptionProcessor(Processor):
    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger(__name__)
        self.transcribe = boto3.client('transcribe')

        self.preferred_language_code = 'es-ES'
        self.preferred_vocabulary_name = 'peruvian-expressions'
        self.preferred_max_speakers = 3
        self.transcription_base = 'transcriptions/'

    def process(self, data, max_no_jobs=100):
        prefix = data['id']  # video dataset key

        self.logger.debug('using %s for samples_generation chunks prefix', prefix)

        videos = self.s3.list_objects_v2(Bucket=self.bucketName, Prefix=prefix)
        if not (videos and 'Contents' in videos):
            raise RuntimeError('There are not samples_generation videos available for processing at the following '
                               'location: {} '
                               .format(prefix))

        self.logger.debug('%d samples_generation videos were found', len(videos['Contents']))
        self.logger.debug('This operation will process %d video_transcription job', max_no_jobs)

        for index, video in enumerate(videos['Contents']):
            try:
                if index >= max_no_jobs:
                    break
                video_key = video['Key']

                if video_key == prefix:
                    continue

                transcription_destination_key = self.get_transcription_destination_key(prefix, video_key)
                if self.s3_exist_object(transcription_destination_key):
                    self.logger.debug('The samples_generation %s has been already transcribed, '
                                      'this video_transcription will be skipped',
                                      video_key)
                    continue

                job_name = self.generate_transcription_job_name(video_key)
                media_format = io_utils.get_file_extension(video_key, exclude_dot=True)
                media_url = self.get_media_file_uri(video_key)
                self.start_audio_transcription(job_name, media_format, media_url)

                transcript_file_uri = self.wait_for_job_completion(job_name)

                transcription_source_key = self.get_transcription_source_key(transcript_file_uri)
                self.s3_move_object(transcription_source_key, transcription_destination_key)
                self.s3_delete_object(transcription_source_key)
                self.transcribe.delete_transcription_job(TranscriptionJobName=job_name)

                self.logger.debug('Audio video_transcription for %s is available at %s', job_name, transcription_source_key)
            except Exception as e:
                self.logger.error(e)

    def start_audio_transcription(self, job_name, media_format, media_url):
        response = self.transcribe.start_transcription_job(
            TranscriptionJobName=job_name,
            LanguageCode=self.preferred_language_code,
            MediaFormat=media_format,
            Media={
                'MediaFileUri': media_url
            },
            OutputBucketName=self.bucketName,
            Settings={
                'VocabularyName': self.preferred_vocabulary_name,
                'ShowSpeakerLabels': True,
                'MaxSpeakerLabels': self.preferred_max_speakers
            }
        )

        self.logger.debug('A new audio video_transcription job was started: \tJob Name: %s \tMedia Format: %s \tMedia '
                          'File Url: %s', job_name, media_format, media_url)
        return response

    def wait_for_job_completion(self, job_name):
        while True:
            resp = self.transcribe.get_transcription_job(TranscriptionJobName=job_name)
            status = resp['TranscriptionJob']['TranscriptionJobStatus']
            if status in ['COMPLETED', 'FAILED']:
                self.logger.debug('The video_transcription job %s has completed with status of: %s', job_name, status)
                if status == 'FAILED':
                    failed_reason = resp['TranscriptionJob']['TranscriptionJob']
                    raise RuntimeError(failed_reason)
                break
            self.logger.debug('The video_transcription job %s is not completed yet', job_name)
            time.sleep(5)

        return resp['TranscriptionJob']['Transcript']['TranscriptFileUri']

    def get_media_file_uri(self, video_key):
        return 'https://{}.s3.amazonaws.com/{}'.format(self.bucketName, video_key)

    def get_transcription_source_key(self, transcript_file_uri):
        return transcript_file_uri.replace('/'.join(['https://s3.us-east-1.amazonaws.com', self.bucketName + '/']), '')

    def get_transcription_destination_key(self, prefix, video_key):
        return io_utils.change_extension(
            video_key.replace(prefix, self.transcription_base), '.json')

    @staticmethod
    def generate_transcription_job_name(video_key: str):
        str_timestamp = str(datetime.datetime.now().timestamp()).replace('.', '-')
        key_no_extension = io_utils.get_filename_without_extension(video_key)

        return '_'.join([str_timestamp, key_no_extension])
