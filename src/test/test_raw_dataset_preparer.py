import os
import unittest
import tensorflow as tf

from unittest.mock import patch
from learning.dataset.prepare.raw_dataset_preparer import RawDatasetPreparer


class TestRawDatasetPreparer(unittest.TestCase):
    def setUp(self):
        self.dataset_path = 'fixtures'
        self.dataset_preparer = RawDatasetPreparer(train_dataset_path=self.dataset_path,
                                                   test_dataset_path=self.dataset_path)

    @patch.object(RawDatasetPreparer, 'get_raw_file_types', return_value=['*.mp4', '*.avi'])
    def test_get_raw_file_list2(self, mock_get_raw_file_types):
        video_path_list, chunk_start_list, chunk_end_list = self.dataset_preparer.get_raw_file_list2('fixtures')

        self.assertIsNotNone(video_path_list)
        self.assertIsNotNone(chunk_start_list)
        self.assertIsNotNone(chunk_end_list)

        self.assertGreater(len(video_path_list), 0)
        self.assertGreater(len(chunk_start_list), 0)
        self.assertGreater(len(chunk_end_list), 0)

    def test_prepare_dataset(self):
        pass
