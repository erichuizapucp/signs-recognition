import os
import csv
import shutil
import logging

from pre_processing.common import io_utils
from rgb_samples_handler import RGBSamplesHandler
from pre_processing.common.processor import Processor


class RGBSamplesProcessor(Processor):
    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger(__name__)
        self.handler = RGBSamplesHandler()

    def process(self, data):
        metadata_file_path = os.path.join(self.work_dir, data['metadata'])
        dataset_path = data['dataset_path']

        local_video_prefix = data['local_video_prefix']
        s3_video_prefix = data['s3_video_prefix']

        if not os.path.exists(metadata_file_path):
            self.logger.debug('the %s metadata file does not exist, the process cannot continue', metadata_file_path)
            return

        reset_dataset = bool(data['reset_dataset'])
        if reset_dataset and os.path.exists(os.path.join(self.work_dir, dataset_path)):
            shutil.rmtree(os.path.join(self.work_dir, dataset_path))

        delay_factor = float(data['translation_delay_factor'])
        end_time_delay = float(data['end_time_bound_in_seconds'])
        metadata_fields = ['token', 'video_name', 'start_time', 'end_time']

        with open(metadata_file_path) as metadata_file:
            reader = csv.DictReader(metadata_file, fieldnames=metadata_fields)
            next(reader, None)  # skip the csv header row
            for metadata in reader:
                self.handle_sample_metadata(metadata, dataset_path, local_video_prefix, s3_video_prefix, delay_factor,
                                            end_time_delay)

        logging.debug('process is completed')

    def handle_sample_metadata(self, metadata, dataset_path, local_video_prefix, s3_video_prefix, delay_factor: float,
                               end_time_delay):
        token = metadata['token']
        video_key = os.path.join(s3_video_prefix, metadata['video_name'])
        video_local_path = os.path.join(self.work_dir, local_video_prefix, metadata['video_name'])

        original_start_time = float(metadata['start_time'])
        original_end_time = float(metadata['end_time'])

        start_time = self.get_start_time(original_start_time, delay_factor)
        end_time = original_end_time + end_time_delay

        io_utils.check_path_file(video_local_path)
        if not os.path.exists(video_local_path):
            self.download_file(video_key, video_local_path)

        self.logger.debug('Starting a RGB sample generation with the following parameters => token: %s, '
                          'video_path: %s, start_time: %s, end_time: %s', token, video_local_path, start_time, end_time)

        self.handler.handle_sample(VideoPath=video_local_path, StartTime=start_time, EndTime=end_time,
                                   TokenName=token, DataSetPath=dataset_path)

    @staticmethod
    def get_start_time(start_time: float, delay_factor: float):
        temp_start_time = start_time - delay_factor
        return start_time - (start_time / 2) if temp_start_time < 0 else temp_start_time
