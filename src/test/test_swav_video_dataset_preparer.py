import os
import unittest
import tensorflow as tf

from learning.dataset.prepare.swav.swav_video_dataset_preparer import SwAVDatasetPreparer
from unittest.mock import patch, call
from learning.common.model_utility import ModelUtility
from mocks import video_processing_mocks as mocks


class TestSwAVDatasetPreparer(unittest.TestCase):
    def setUp(self):
        os.environ['WORK_DIR'] = '../../'

        self.mock_sample = mocks.get_sample_mock('fixtures/extracted_person_sample/')
        self.dataset_preparer = SwAVDatasetPreparer(train_dataset_path='fixtures',
                                                    test_dataset_path='fixtures',
                                                    person_detection_model=mocks.person_detection_model_mock,
                                                    crop_sizes=[224, 96],
                                                    num_crops=[2, 3],
                                                    min_scale=[0.14, 0.05],
                                                    max_scale=[1., 0.14])

    def test_data_generator2_all_valid(self):
        data_gen = self.dataset_preparer.data_generator2([], [], [])

    def test_data_generator2_all_invalid(self):
        pass

    def test_data_generator2_some_invalid(self):
        pass

    @patch('learning.dataset.prepare.swav.multi_crop.tie_together')
    def test_prepare_sample3(self, mock_tie_together):
        mock_tie_together.side_effect = mocks.mock_multicrop_tie_together
        sample = self.dataset_preparer.prepare_sample3(self.mock_sample)

        calls = [call(self.mock_sample, 0.14, 1., 224), call(self.mock_sample, 0.14, 1., 224),
                 call(self.mock_sample, 0.05, 0.14, 96), call(self.mock_sample, 0.05, 0.14, 96),
                 call(self.mock_sample, 0.05, 0.14, 96)]

        self.assertIsNotNone(sample)
        mock_tie_together.assert_has_calls(calls, any_order=False)
        self.assertEqual(len(sample), 5)
        self.assertEqual(sample[0].shape, (224, 224, 3))
        self.assertEqual(sample[1].shape, (224, 224, 3))
        self.assertEqual(sample[2].shape, (96, 96, 3))
        self.assertEqual(sample[3].shape, (96, 96, 3))
        self.assertEqual(sample[4].shape, (96, 96, 3))


if __name__ == "__main__":
    unittest.main()
