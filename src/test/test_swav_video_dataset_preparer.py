import os
import unittest
import tensorflow as tf

from learning.dataset.prepare.swav.swav_video_dataset_preparer import SwAVDatasetPreparer
from pre_processing.isolated_detection.samples_generation.rgb_person_sample_extractor import RGBPersonSamplesExtractor
from unittest.mock import patch, call
from learning.common.model_utility import ModelUtility
from mocks import video_processing_mocks as mocks


class TestSwAVDatasetPreparer(unittest.TestCase):
    def setUp(self):
        os.environ['WORK_DIR'] = '../../'

        self.dataset_preparer = SwAVDatasetPreparer(train_dataset_path='fixtures',
                                                    test_dataset_path='fixtures',
                                                    person_detection_model=mocks.person_detection_model_mock,
                                                    crop_sizes=[224, 96],
                                                    num_crops=[2, 3],
                                                    min_scale=[0.14, 0.05],
                                                    max_scale=[1., 0.14],
                                                    sample_duration_range=[0.5, 1])
        self.model_utility = ModelUtility()
        self.person_detection_model_name = 'centernet_resnet50_v1_fpn_512x512_coco17_tpu-8'
        self.person_detection_checkout_prefix = 'ckpt-0'

    @patch.object(RGBPersonSamplesExtractor, 'extract_sample')
    @patch.object(SwAVDatasetPreparer, 'get_sample_duration', return_value=1.0)
    def test_data_generator2_all_valid(self, get_sample_duration_mock, extract_sample_mock):
        extract_sample_mock.return_value = (True, mocks.get_sample_mock('fixtures/extracted_person_sample/'))

        video_path_list = mocks.video_path_list
        chunk_start_list = mocks.chunk_start_list
        chunk_end_list = mocks.chunk_end_list

        samples_generator = self.dataset_preparer.data_generator2(video_path_list, chunk_start_list, chunk_end_list)

        for generated_sample in samples_generator:
            self.assertIsNotNone(generated_sample)

        assert extract_sample_mock.call_count == 136

    @patch.object(RGBPersonSamplesExtractor, 'extract_sample', return_value=(False, None))
    @patch.object(SwAVDatasetPreparer, 'get_sample_duration', return_value=1.0)
    def test_data_generator2_all_invalid(self, get_sample_duration_mock, extract_sample_mock):
        video_path_list = mocks.video_path_list
        chunk_start_list = mocks.chunk_start_list
        chunk_end_list = mocks.chunk_end_list

        samples_generator = self.dataset_preparer.data_generator2(video_path_list, chunk_start_list, chunk_end_list)

        assert len(list(samples_generator)) == 0

        assert extract_sample_mock.call_count == 136

    @patch.object(RGBPersonSamplesExtractor, 'extract_sample')
    @patch.object(SwAVDatasetPreparer, 'get_sample_duration', return_value=1.0)
    def test_data_generator2_some_invalid(self, get_sample_duration_mock, extract_sample_mock):
        sample_mock = mocks.get_sample_mock('fixtures/extracted_person_sample/')
        extract_sample_mock.side_effect = lambda video_path, start_time, end_time: \
            (True, sample_mock) if video_path == 'fixtures/consultant-12-sesion-01-part-02-section-02.mp4' else (False, None)

        video_path_list = mocks.video_path_list
        chunk_start_list = mocks.chunk_start_list
        chunk_end_list = mocks.chunk_end_list

        samples_generator = self.dataset_preparer.data_generator2(video_path_list, chunk_start_list, chunk_end_list)

        count = 0
        for generated_sample in samples_generator:
            self.assertIsNotNone(generated_sample)
            generated_sample.mark_used()
            count = count + 1

        assert extract_sample_mock.call_count == 136
        assert count == 125

    @patch('learning.dataset.prepare.swav.multi_crop.tie_together')
    def test_prepare_sample3(self, mock_tie_together):
        mock_tie_together.side_effect = mocks.mock_multicrop_tie_together
        self.mock_sample = mocks.get_sample_mock('fixtures/extracted_person_sample/')
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

    @patch.object(SwAVDatasetPreparer, 'get_sample_duration', return_value=1.0)
    def test_integration_data_generator2(self, get_sample_duration_mock):
        detection_model = self.model_utility.get_object_detection_model(self.person_detection_model_name,
                                                                        self.person_detection_checkout_prefix)
        self.dataset_preparer.detection_model = detection_model
        self.dataset_preparer.extractor.detection_model = detection_model

        video_path_list = mocks.video_path_list
        chunk_start_list = mocks.chunk_start_list
        chunk_end_list = mocks.chunk_end_list

        samples_generator = self.dataset_preparer.data_generator2(video_path_list, chunk_start_list, chunk_end_list)

        count = 0
        for generated_sample in samples_generator:
            self.assertIsNotNone(generated_sample)
            count = count + 1

        assert count == 136

    @patch.object(SwAVDatasetPreparer, 'get_sample_duration', return_value=1.0)
    def test_integration_prepare_dataset(self, get_sample_duration_mock):
        detection_model = self.model_utility.get_object_detection_model(self.person_detection_model_name,
                                                                        self.person_detection_checkout_prefix)
        self.dataset_preparer.detection_model = detection_model
        self.dataset_preparer.extractor.detection_model = detection_model

        dataset: tf.data.Dataset = self.dataset_preparer.prepare_dataset("fixtures", 2)
        count = 0
        for sample in dataset:
            self.assertIsNotNone(sample)
            count = count + 1

        assert count == 68


if __name__ == "__main__":
    unittest.main()
