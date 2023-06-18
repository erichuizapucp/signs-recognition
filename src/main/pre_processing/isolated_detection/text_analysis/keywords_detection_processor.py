import os
import time
import boto3
import logging


from datetime import datetime
from pre_processing.common.processor import Processor


class KeywordsDetectionProcessor(Processor):
    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger(__name__)
        self.comprehend = boto3.client('comprehend')
        self.data_access_role_arn = os.getenv('DATA_ACCESS_ROLE_ARN', None)
        self.preferred_language_code = 'es'

        self.sleep_time_in_ms = 5

    def process(self, data):
        input_location = data['inputLocation']
        output_location = data['outputLocation']

        job_name = self.get_job_name()
        resp = self.start_keywords_detection_job(job_name, input_location, output_location)
        job_id = resp['JobId']

        output_file_uri = self.wait_for_job_completion(job_id)
        self.logger.debug('The detected keyword phrases are available at %s', output_file_uri)

    def start_keywords_detection_job(self, job_name, input_location, output_location):
        response = self.comprehend.start_key_phrases_detection_job(
            JobName=job_name,
            LanguageCode=self.preferred_language_code,
            InputDataConfig={
                'S3Uri': input_location,
                'InputFormat': 'ONE_DOC_PER_FILE'
            },
            OutputDataConfig={
                'S3Uri': output_location
            },
            DataAccessRoleArn=self.data_access_role_arn
        )

        self.logger.debug('A new keywords detection job was started: \tJob Name: %s \tInput Location: %s \tOutput '
                          'Location: %s', job_name, input_location, output_location)

        return response

    def wait_for_job_completion(self, job_name):
        while True:
            resp = self.comprehend.describe_key_phrases_detection_job(JobId=job_name)
            status = resp['KeyPhrasesDetectionJobProperties']['JobStatus']
            if status in ['COMPLETED', 'FAILED']:
                self.logger.debug('The keywords detection job %s has completed with status of: %s', job_name, status)
                if status == 'FAILED':
                    failed_reason = resp['KeyPhrasesDetectionJobProperties']['Message']
                    raise RuntimeError(failed_reason)
                break

            self.logger.debug('The keyword detection job %s is not completed yet', job_name)
            time.sleep(self.sleep_time_in_ms)

        return resp['KeyPhrasesDetectionJobProperties']['OutputDataConfig']['S3Uri']

    @staticmethod
    def get_job_name():
        str_timestamp = str(datetime.now().timestamp()).replace('.', '-')
        return '_'.join([str_timestamp, 'keywords-detection-job'])
