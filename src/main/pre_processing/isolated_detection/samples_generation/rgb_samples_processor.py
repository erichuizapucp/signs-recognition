import os
import shutil
import logging
import csv

from pre_processing.common.processor import Processor
from pre_processing.common import io_utils
from rgb_samples_handler import RGBSamplesHandler


class RGBSamplesProcessor(Processor):
    __videos_dataset_rgb_path = 'dataset/rgb'

    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger(__name__)
        
    def process(self, data):
        metadata_file_path = os.path.join(self._work_dir, data['metadata'])
        if not os.path.exists(metadata_file_path):
            self.logger.debug('the %s metadata file does not exist, the process cannot continue', metadata_file_path)
            return

        reset_dataset = bool(data['reset_dataset'])
        if reset_dataset and os.path.exists(os.path.join(self._work_dir, self.__videos_dataset_rgb_path)):
            shutil.rmtree(os.path.join(self._work_dir, self.__videos_dataset_rgb_path))

        delay_factor = float(data['translation_delay_factor'])
        metadata_fields = ['token', 'video_path', 'start_time', 'end_time']

        with open(metadata_file_path) as metadata_file:
            reader = csv.DictReader(metadata_file, fieldnames=metadata_fields)
            next(reader, None)  # skip the csv header row
            for metadata in reader:
                self.__handle_sample_metadata(metadata, delay_factor)

        logging.debug('process is completed')

    def __handle_sample_metadata(self, metadata, delay_factor: float):
        token = metadata['token']
        video_key = metadata['video_path']
        video_local_path = os.path.join(self._work_dir, metadata['video_path'])

        original_start_time = float(metadata['start_time'])
        original_end_time = float(metadata['end_time'])

        start_time = self.__get_start_time(original_start_time, delay_factor)
        end_time = original_end_time + 0.5

        io_utils.check_path_file(video_local_path)
        if not os.path.exists(video_local_path):
            self.download_file(video_key, video_local_path)

        self.logger.debug('Starting a RGB sample generation with the following parameters => token: %s, '
                          'video_path: %s, start_time: %s, end_time: %s', token, video_local_path, start_time, end_time)

        sample_folder_path = self.__get_sample_folder_path(token)
        io_utils.check_path_dir(sample_folder_path)

        handler = RGBSamplesHandler()
        handler.handle_sample(VideoPath=video_local_path, StartTime=start_time, EndTime=end_time,
                              FolderPath=sample_folder_path)

    def __get_sample_folder_path(self, token):
        token_folder_path = os.path.join(self._work_dir, self.__videos_dataset_rgb_path, token)
        io_utils.check_path_dir(token_folder_path)

        num_dirs = len([folder for folder in os.listdir(token_folder_path) if os.path.isdir(
            os.path.join(token_folder_path, folder))])

        return os.path.join(token_folder_path, str(num_dirs + 1))

    @staticmethod
    def __get_start_time(start_time: float, delay_factor: float):
        temp_start_time = start_time - delay_factor
        return start_time - (start_time / 2) if temp_start_time < 0 else temp_start_time
