import os
from pre_processing.common.processor import Processor


class Downloader(Processor):
    def __init__(self):
        super().__init__()

    def process(self, data):
        super().process(data)

        # download the samples_generation clip from aws s3 and save to temp file
        self._video_file_path = self.generate_video_file_path(self.video_key)

        if not os.path.exists(self._video_file_path):
            os.makedirs(os.path.dirname(self._video_file_path), exist_ok=True)

            self.s3.download_file(self.bucketName, self.video_key, self._video_file_path)
            print('the samples_generation clip is available at: {}'.format(self._video_file_path))
        else:
            print('the samples_generation is already available at: {}'.format(self._video_file_path))
