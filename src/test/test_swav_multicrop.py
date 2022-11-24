import os
import unittest
import random
from unittest.mock import patch
import tensorflow as tf

from mocks import video_processing_mocks as mocks
from learning.dataset.prepare.swav import multi_crop


class TestSwAVMultiCrop(unittest.TestCase):
    def setUp(self):
        os.environ['WORK_DIR'] = '../../'

        self.mock_sample = mocks.get_sample_mock('fixtures/extracted_person_sample/')

        self.min_scale_high_res = 0.14
        self.max_scale_high_res = 1.
        self.crop_size_high_res = 224

        self.min_scale_low_res = 0.05
        self.max_scale_low_res = 0.14
        self.crop_size_low_res = 96

    def test_random_resize_crop_high_res(self):
        _, random_resized_crop = self.get_random_resized_crop(self.min_scale_high_res, self.max_scale_high_res,
                                                              self.crop_size_high_res)

        self.assertIsNotNone(random_resized_crop)
        self.assertEqual(random_resized_crop.shape, [224, 224, 3])

    def test_random_resize_crop_low_res(self):
        _, random_resized_crop = self.get_random_resized_crop(self.min_scale_low_res, self.max_scale_low_res,
                                                              self.crop_size_low_res)

        self.assertIsNotNone(random_resized_crop)
        self.assertEqual(random_resized_crop.shape, [96, 96, 3])

    def test_gaussian_blur_high_res(self):
        index, random_resized_crop = self.get_random_resized_crop(self.min_scale_high_res, self.max_scale_high_res,
                                                                  self.crop_size_high_res)

        gaussian_blurred_crop = multi_crop.gaussian_blur(random_resized_crop, index)

        self.assertIsNotNone(gaussian_blurred_crop)
        self.assertEqual(gaussian_blurred_crop.shape, [224, 224, 3])

    def test_gaussian_blur_low_res(self):
        index, random_resized_crop = self.get_random_resized_crop(self.min_scale_low_res, self.max_scale_low_res,
                                                                  self.crop_size_low_res)

        gaussian_blurred_crop = multi_crop.gaussian_blur(random_resized_crop, index)

        self.assertIsNotNone(gaussian_blurred_crop)
        self.assertEqual(gaussian_blurred_crop.shape, [96, 96, 3])

    def test_color_jitter_high_res(self):
        index, random_resized_crop = self.get_random_resized_crop(self.min_scale_high_res, self.max_scale_high_res,
                                                                  self.crop_size_high_res)

        color_jitter_crop = multi_crop.color_jitter(random_resized_crop, index)

        self.assertIsNotNone(color_jitter_crop)
        self.assertEqual(color_jitter_crop.shape, [224, 224, 3])

    def test_color_jitter_low_res(self):
        index, random_resized_crop = self.get_random_resized_crop(self.min_scale_low_res, self.max_scale_low_res,
                                                                  self.crop_size_low_res)

        color_jitter_crop = multi_crop.color_jitter(random_resized_crop, index)

        self.assertIsNotNone(color_jitter_crop)
        self.assertEqual(color_jitter_crop.shape, [96, 96, 3])

    def test_color_drop_high_res(self):
        index, random_resized_crop = self.get_random_resized_crop(self.min_scale_high_res, self.max_scale_high_res,
                                                                  self.crop_size_high_res)

        color_drop_crop = multi_crop.color_drop(random_resized_crop, index)

        self.assertIsNotNone(color_drop_crop)
        self.assertEqual(color_drop_crop.shape, [224, 224, 3])

    def test_color_drop_low_res(self):
        index, random_resized_crop = self.get_random_resized_crop(self.min_scale_low_res, self.max_scale_low_res,
                                                                  self.crop_size_low_res)

        color_drop_crop = multi_crop.color_drop(random_resized_crop, index)

        self.assertIsNotNone(color_drop_crop)
        self.assertEqual(color_drop_crop.shape, [96, 96, 3])

    @patch('learning.dataset.prepare.swav.multi_crop.flip_left_right')
    @patch('learning.dataset.prepare.swav.multi_crop.gaussian_blur')
    @patch('learning.dataset.prepare.swav.multi_crop.color_jitter')
    @patch('learning.dataset.prepare.swav.multi_crop.color_drop')
    def test_custom_augment_call_all(self, mock_color_drop, mock_color_jitter, mock_gaussian_blur, mock_flip_left_right):
        mock_color_drop.return_value = mocks.high_res_multicrop_frame_mock
        mock_color_jitter.return_value = mocks.high_res_multicrop_frame_mock
        mock_gaussian_blur.return_value = mocks.high_res_multicrop_frame_mock
        mock_flip_left_right.return_value = mocks.high_res_multicrop_frame_mock

        index, random_resized_crop = self.get_random_resized_crop(self.min_scale_high_res, self.max_scale_high_res,
                                                                  self.crop_size_high_res)

        cropped_frame = multi_crop.custom_augment(random_resized_crop, index, prob=(1.0, 1.0, 1.0, 1.0))

        self.assertIsNotNone(cropped_frame)
        self.assertEqual(cropped_frame.shape, [224, 224, 3])

        mock_color_drop.assert_called_once()
        mock_color_jitter.assert_called_once()
        mock_gaussian_blur.assert_called_once()
        mock_flip_left_right.assert_called_once()

    @patch('learning.dataset.prepare.swav.multi_crop.flip_left_right')
    @patch('learning.dataset.prepare.swav.multi_crop.gaussian_blur')
    @patch('learning.dataset.prepare.swav.multi_crop.color_jitter')
    @patch('learning.dataset.prepare.swav.multi_crop.color_drop')
    def test_custom_augment_not_call_any(self, mock_color_drop, mock_color_jitter, mock_gaussian_blur,
                                         mock_flip_left_right):
        mock_color_drop.return_value = mocks.low_res_multicrop_frame_mock
        mock_color_jitter.return_value = mocks.low_res_multicrop_frame_mock
        mock_gaussian_blur.return_value = mocks.low_res_multicrop_frame_mock
        mock_flip_left_right.return_value = mocks.low_res_multicrop_frame_mock

        index, random_resized_crop = self.get_random_resized_crop(self.min_scale_low_res, self.max_scale_low_res,
                                                                  self.crop_size_low_res)

        cropped_frame = multi_crop.custom_augment(random_resized_crop, index, prob=(0.0, 0.0, 0.0, 0.0))

        self.assertIsNotNone(cropped_frame)
        self.assertEqual(cropped_frame.shape, [96, 96, 3])

        mock_color_drop.assert_not_called()
        mock_color_jitter.assert_not_called()
        mock_gaussian_blur.assert_not_called()
        mock_flip_left_right.assert_not_called()

    def test_tie_together_high_res(self):
        multi_cropped_sample = multi_crop.tie_together(self.mock_sample, self.min_scale_high_res,
                                                       self.max_scale_high_res, self.crop_size_high_res)

        assert multi_cropped_sample.size() > 0
        assert multi_cropped_sample.size() == 60
        assert multi_cropped_sample.element_shape == (224, 224, 3)

        # frames = multi_cropped_sample.stack()
        # for index, frame in enumerate(frames):
        #     file_name = 'fixtures/high_res_multicrop_sample/frame_{}.jpg'.format(index)
        #     tf.io.write_file(file_name, tf.image.encode_jpeg(tf.image.convert_image_dtype(frame, tf.uint8)))

    def test_tie_together_low_res(self):
        multi_cropped_sample = multi_crop.tie_together(self.mock_sample, self.min_scale_low_res,
                                                       self.max_scale_low_res, self.crop_size_low_res)

        assert multi_cropped_sample.size() > 0
        assert multi_cropped_sample.size() == 60
        assert multi_cropped_sample.element_shape == (96, 96, 3)

        # frames = multi_cropped_sample.stack()
        # for index, frame in enumerate(frames):
        #     file_name = 'fixtures/low_res_multicrop_sample/frame_{}.jpg'.format(index)
        #     tf.io.write_file(file_name, tf.image.encode_jpeg(tf.image.convert_image_dtype(frame, tf.uint8)))

    def get_random_resized_crop(self, min_scale, max_scale, crop_size):
        index = random.randint(0, 59)
        frame_to_crop = tf.image.convert_image_dtype(self.mock_sample.read(index), tf.float32)

        random_resized_crop = multi_crop.random_resize_crop(frame_to_crop, index, min_scale, max_scale, crop_size)
        return index, random_resized_crop


if __name__ == "__main__":
    unittest.main()
