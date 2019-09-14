import logging

from processor import Processor


class SamplesGenerationProcessor(Processor):
    __transcription_files_prefix = "transcriptions/"
    __nouns_file_name = "detected-nouns.csv"
    __numbers_file_name = "detected-numbers.csv"

    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger(__name__)

    def process(self, data):
        print()
