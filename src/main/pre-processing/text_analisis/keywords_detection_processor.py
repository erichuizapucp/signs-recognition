import logging
import boto3
import time
import os

from datetime import datetime
from processor import Processor


class KeywordsDetectionProcessor(Processor):
    __preferred_language_code = 'es'

    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger(__name__)
        self._comprehend = boto3.client('comprehend')
        self._data_access_role_arn = os.getenv('DATA_ACCESS_ROLE_ARN', None)

    def process(self, data):
        input_location = data['inputLocation']
        output_location = data['outputLocation']

        job_name = self.__get_job_name()
        resp = self.__start_keywords_detection_job(job_name, input_location, output_location)
        job_id = resp['JobId']

        output_file_uri = self.__wait_for_job_completion(job_id)
        self.logger.debug('The detected keyword phrases are available at %s', output_file_uri)

    def __start_keywords_detection_job(self, job_name, input_location, output_location):
        response = self._comprehend.start_key_phrases_detection_job(
            JobName=job_name,
            LanguageCode=self.__preferred_language_code,
            InputDataConfig={
                'S3Uri': input_location,
                'InputFormat': 'ONE_DOC_PER_FILE'
            },
            OutputDataConfig={
                'S3Uri': output_location
            },
            DataAccessRoleArn=self._data_access_role_arn
        )

        self.logger.debug('A new keywords detection job was started: \tJob Name: %s \tInput Location: %s \tOutput '
                          'Location: %s', job_name, input_location, output_location)

        return response

    def __wait_for_job_completion(self, job_name):
        while True:
            resp = self._comprehend.describe_key_phrases_detection_job(JobId=job_name)
            status = resp['KeyPhrasesDetectionJobProperties']['JobStatus']
            if status in ['COMPLETED', 'FAILED']:
                self.logger.debug('The keywords detection job %s has completed with status of: %s', job_name, status)
                if status == 'FAILED':
                    failed_reason = resp['KeyPhrasesDetectionJobProperties']['Message']
                    raise RuntimeError(failed_reason)
                break

            self.logger.debug('The keyword detection job %s is not completed yet', job_name)
            time.sleep(5)

        return resp['KeyPhrasesDetectionJobProperties']['OutputDataConfig']['S3Uri']

    @staticmethod
    def __get_job_name():
        str_timestamp = str(datetime.now().timestamp()).replace('.', '-')
        return '_'.join([str_timestamp, 'keywords-detection-job'])
