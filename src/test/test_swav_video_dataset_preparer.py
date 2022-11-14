import os
import unittest

from learning.dataset.prepare.swav.swav_video_dataset_preparer import SwAVDatasetPreparer
from learning.common.model_utility import ModelUtility
from mocks import video_processing_mocks as mocks


class TestSwAVDatasetPreparer(unittest.TestCase):
    def setUp(self):
        os.environ['WORK_DIR'] = '../../'

        self.samples_extractor = SwAVDatasetPreparer(train_dataset_path='',
                                                     test_dataset_path='',
                                                     person_detection_model=mocks.person_detection_model_mock)

    def test_data_generator2_all_valid(self):
        pass

    def test_data_generator2_all_invalid(self):
        pass

    def test_data_generator2_some_invalid(self):
        pass


if __name__ == "__main__":
    unittest.main()
