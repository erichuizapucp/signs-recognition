import logging
import boto3
from processor import Processor


class KeywordsDetectionProcessor(Processor):
    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger(__name__)
        self._comprehend = boto3.client('comprehend')

    def process(self, data):
        input_location = data['inputLocation']
        output_location = data['outputLocation']

        self._comprehend.start_key_phrases_detection_job(
            InputDataConfig={
                'S3Uri': input_location
            },
            OutputDataConfig={
                'S3Uri': output_location
            }
        )
