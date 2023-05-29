import os
import shutil
import unittest

import tensorflow as tf

from learning.common import dataset_type
from learning.common import features
from learning.dataset.tfrecord.tf_record_utility import TFRecordUtility
from mocks import video_processing_mocks as mocks


class TestTFRecordUtility(unittest.TestCase):
    def setUp(self):
        os.environ['WORK_DIR'] = '../../'

        self.mock_high_res_1_sample = tf.image.convert_image_dtype(
            mocks.get_sample_mock('fixtures/high_res_multicrop_sample/').stack(), tf.float32)
        self.mock_high_res_2_sample = tf.image.convert_image_dtype(
            mocks.get_sample_mock('fixtures/high_res_multicrop_sample/').stack(), tf.float32)
        self.mock_low_res_1_sample = tf.image.convert_image_dtype(
            mocks.get_sample_mock('fixtures/low_res_multicrop_sample/').stack(), tf.float32)
        self.mock_low_res_2_sample = tf.image.convert_image_dtype(
            mocks.get_sample_mock('fixtures/low_res_multicrop_sample/').stack(), tf.float32)
        self.mock_los_res_3_sample = tf.image.convert_image_dtype(
            mocks.get_sample_mock('fixtures/low_res_multicrop_sample/').stack(), tf.float32)

        self.tf_record_utility = TFRecordUtility()

    def test_serialize_swav_sample(self):
        mock_high_res_1 = tf.expand_dims(self.mock_high_res_1_sample, axis=0)
        mock_high_res_2 = tf.expand_dims(self.mock_high_res_2_sample, axis=0)
        mock_low_res_1 = tf.expand_dims(self.mock_low_res_1_sample, axis=0)
        mock_low_res_2 = tf.expand_dims(self.mock_low_res_2_sample, axis=0)
        mock_los_res_3 = tf.expand_dims(self.mock_los_res_3_sample, axis=0)

        multicrop_crop_seqs = (mock_high_res_1, mock_high_res_2, mock_low_res_1, mock_low_res_2, mock_los_res_3)
        serialized_sample = self.tf_record_utility.serialize_swav_sample(multicrop_crop_seqs)
        parsed_sample = tf.io.parse_single_example(serialized_sample, self.tf_record_utility.swav_feature_description)

        assert len(serialized_sample) == 23040351
        tf.debugging.assert_type(serialized_sample, tf_type=tf.string)
        tf.debugging.assert_type(parsed_sample[features.HIGH_RES_FRAMES_SEQ_1], tf_type=tf.string)
        tf.debugging.assert_type(parsed_sample[features.HIGH_RES_FRAMES_SEQ_2], tf_type=tf.string)
        tf.debugging.assert_type(parsed_sample[features.LOW_RES_FRAMES_SEQ_1], tf_type=tf.string)
        tf.debugging.assert_type(parsed_sample[features.LOW_RES_FRAMES_SEQ_2], tf_type=tf.string)
        tf.debugging.assert_type(parsed_sample[features.LOW_RES_FRAMES_SEQ_3], tf_type=tf.string)
        tf.debugging.assert_type(parsed_sample[features.NO_FRAMES], tf_type=tf.int64)

    def test_parse_swav_dict_sample(self):
        serialized_sample = tf.io.read_file('fixtures/serialized_sample/consultant_02_sample.dat')
        parsed_sample = self.tf_record_utility.parse_swav_dict_sample(serialized_sample)

        self.assert_parsed_swav_sample(parsed_sample)

    def test_swav_serialize_dataset_single_file(self):
        shutil.rmtree('temp')

        dataset = tf.data.Dataset.from_tensors((self.mock_high_res_1_sample,
                                                self.mock_high_res_2_sample,
                                                self.mock_low_res_1_sample,
                                                self.mock_low_res_2_sample,
                                                self.mock_los_res_3_sample))
        dataset = dataset.padded_batch(1, drop_remainder=True)

        self.tf_record_utility.serialize_dataset(dataset_type.SWAV,
                                                 dataset, 'temp',
                                                 dataset_type.SWAV,
                                                 100,
                                                 self.tf_record_utility.serialize_swav_sample)

        list_temp_files = os.listdir('temp')
        assert len(list_temp_files) == 1
        assert list_temp_files[0] == 'swav_0.tfrecord'

    def test_swav_serialize_dataset_multiple_file(self):
        if os.path.exists('temp'):
            shutil.rmtree('temp')

        dataset = tf.data.Dataset.from_tensors((self.mock_high_res_1_sample,
                                                self.mock_high_res_2_sample,
                                                self.mock_low_res_1_sample,
                                                self.mock_low_res_2_sample,
                                                self.mock_los_res_3_sample)).repeat(3)
        dataset = dataset.padded_batch(1, drop_remainder=True)

        self.tf_record_utility.serialize_dataset(dataset_type.SWAV,
                                                 dataset, 'temp',
                                                 dataset_type.SWAV,
                                                 10,
                                                 self.tf_record_utility.serialize_swav_sample)
        list_temp_files = os.listdir('temp')

        assert len(list_temp_files) == 2
        assert 'swav_0.tfrecord' in list_temp_files
        assert 'swav_1.tfrecord' in list_temp_files

    def test_swav_deserialize_dataset(self):
        deserialized_dataset = self.tf_record_utility.deserialize_dataset('fixtures/serialized_dataset',
                                                                          self.tf_record_utility.parse_swav_dict_sample,
                                                                          batch_size=2)

        assert deserialized_dataset.element_spec == (tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
                                                     tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
                                                     tf.TensorSpec(shape=(None, 96, 96, 3), dtype=tf.float32),
                                                     tf.TensorSpec(shape=(None, 96, 96, 3), dtype=tf.float32),
                                                     tf.TensorSpec(shape=(None, 96, 96, 3), dtype=tf.float32))

        for sample in deserialized_dataset:
            self.assert_parsed_swav_sample(sample)

    def assert_parsed_swav_sample(self, sample):
        self.assertEqual(tf.math.reduce_all(tf.equal(self.mock_high_res_1_sample, sample[0])), tf.constant(True))
        self.assertEqual(tf.math.reduce_all(tf.equal(self.mock_high_res_2_sample, sample[1])), tf.constant(True))
        self.assertEqual(tf.math.reduce_all(tf.equal(self.mock_low_res_1_sample, sample[2])), tf.constant(True))
        self.assertEqual(tf.math.reduce_all(tf.equal(self.mock_low_res_2_sample, sample[3])), tf.constant(True))
        self.assertEqual(tf.math.reduce_all(tf.equal(self.mock_los_res_3_sample, sample[4])), tf.constant(True))
