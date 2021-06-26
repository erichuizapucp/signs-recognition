import os
import shutil
import logging
import glob

from pre_processing.common.processor import Processor
from pre_processing.common import io_utils
from opticalflow_samples_handler import OpticalflowSamplesHandler


class OpticalflowSamplesProcessor(Processor):
    __opticalflow_dataset_relative_path = 'dataset/opticalflow'
    # we need the rgb relative path because opticalflow samples are calculated from rgb samples
    __rgb_dataset_relative_path = 'dataset/rgb'

    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger(__name__)

    def process(self, data):
        rgb_dataset_path = os.path.join(self._work_dir, self.__rgb_dataset_relative_path)
        if not os.path.exists(rgb_dataset_path):
            self.logger.debug('The RGB dataset does not exist at %s, the process cannot continue.', rgb_dataset_path)
            return

        opticalflow_dataset_path = os.path.join(self._work_dir, self.__opticalflow_dataset_relative_path)
        reset_dataset = bool(data['reset_dataset'])
        if reset_dataset and os.path.exists(opticalflow_dataset_path):
            shutil.rmtree(opticalflow_dataset_path)
            self.logger.debug('optical flow dataset at %s was reset', opticalflow_dataset_path)

        handler = OpticalflowSamplesHandler()
        for rgb_token_folder_path in sorted(glob.glob(rgb_dataset_path + '/*/')):
            for rgb_sample_folder_path in sorted(glob.glob(rgb_token_folder_path + '/*/')):
                opticalflow_sample_path = self.__get_opticalflow_sample_path(rgb_sample_folder_path)
                io_utils.check_path_file(opticalflow_sample_path)

                handler.handle_sample(RGBSampleFolderPath=rgb_sample_folder_path, OFSamplePath=opticalflow_sample_path)

        self.logger.debug('The optical flow samples generation is completed.')

    @staticmethod
    def __get_opticalflow_sample_path(rgb_sample_folder_path):
        temp_path: str = rgb_sample_folder_path.replace('rgb', 'opticalflow')
        temp_path = os.path.dirname(temp_path)

        last_slash_index = temp_path.rfind(os.path.sep)
        sample_number = temp_path[last_slash_index + 1:]

        temp_path = temp_path[:last_slash_index + 1]
        token_name = os.path.basename(os.path.dirname(temp_path))
        return os.path.join(temp_path, token_name + '-' + sample_number + '.jpg')
