import logging
from processor import Processor


class KeywordsDetectionProcessor(Processor):
    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger(__name__)

    def process(self, data):
        print()
