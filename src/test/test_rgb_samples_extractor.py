import os
import unittest
import numpy as np
import tensorflow as tf
from unittest.mock import patch
from pre_processing.isolated_detection.samples_generation.rgb_samples_extractor import RGBSamplesExtractor
from learning.common.model_utility import ModelUtility
from mocks import video_processing_mocks as mocks


class TestRGBSamplesExtractor(unittest.TestCase):
    def setUp(self):
        self.samples_extractor = RGBSamplesExtractor()
        self.samples_extractor_person_detect = RGBSamplesExtractor(detect_person=True,
                                                                   detection_model=mocks.person_detection_model_mock)
        self.model_utility = ModelUtility()
        self.video_path = '/Users/erichuiza/Documents/engineering-doctorate/dev/signs-recognition/src/test/fixtures' \
                          '/consultant-02-session-01-part-01-00.mp4'

    @patch.object(RGBSamplesExtractor, 'get_detection_bounding_box', return_value=(True, mocks.bounding_box_mock))
    @patch.object(RGBSamplesExtractor, 'extract_person_from_frame', return_value=mocks.person_video_frame_mock)
    def test_extract_sample_with_person_detect(self, mock_extract_person_from_frame, mock_get_detection_bounding_box):
        # with patch('cv2.VideoCapture', side_effect=mocks.mock_video_capture):
        extracted_frames: tf.TensorArray = self.samples_extractor_person_detect.extract_sample(tf.constant(self.video_path),
                                                                                               tf.constant(0.00),
                                                                                               tf.constant(2.00))
        extracted_frames: tf.TensorArray = self.samples_extractor_person_detect.extract_sample(tf.constant(self.video_path),
                                                                                               tf.constant(0.00),
                                                                                               tf.constant(5.00))
        extracted_frames: tf.TensorArray = self.samples_extractor_person_detect.extract_sample(tf.constant(self.video_path),
                                                                                               tf.constant(0.00),
                                                                                               tf.constant(6.00))
        extracted_frames: tf.TensorArray = self.samples_extractor_person_detect.extract_sample(tf.constant(self.video_path),
                                                                                               tf.constant(0.00),
                                                                                               tf.constant(4.00))
        extracted_frames: tf.TensorArray = self.samples_extractor_person_detect.extract_sample(tf.constant(self.video_path),
                                                                               tf.constant(0.00),
                                                                               tf.constant(3.00))
        extracted_frames: tf.TensorArray = self.samples_extractor_person_detect.extract_sample(
            tf.constant(self.video_path),
            tf.constant(0.00),
            tf.constant(5.00))
        extracted_frames: tf.TensorArray = self.samples_extractor_person_detect.extract_sample(
            tf.constant(self.video_path),
            tf.constant(0.00),
            tf.constant(2.00))
        extracted_frames: tf.TensorArray = self.samples_extractor_person_detect.extract_sample(
            tf.constant(self.video_path),
            tf.constant(0.00),
            tf.constant(7.00))
        extracted_frames: tf.TensorArray = self.samples_extractor_person_detect.extract_sample(
            tf.constant(self.video_path),
            tf.constant(0.00),
            tf.constant(4.00))


        # assert extracted_frames.size() > 0
        # assert extracted_frames.size() == 90
        # extracted_frames.
        # assert np.shape(extracted_frames) == (180, 672, 448, 3)

        # mock_get_detection_bounding_box.assert_called_once()
        # self.assertEqual(mock_extract_person_from_frame.call_count, 180)

    @patch.object(RGBSamplesExtractor, 'get_detection_bounding_box', return_value=(True, mocks.bounding_box_mock))
    @patch.object(RGBSamplesExtractor, 'extract_person_from_frame', return_value=mocks.video_frame_mock)
    def test_extract_sample_without_person_detect(self, mock_extract_person_from_frame,
                                                  mock_get_detection_bounding_box):
        with patch('cv2.VideoCapture', side_effect=mocks.mock_video_capture):
            extracted, extracted_frames = self.samples_extractor.extract_sample(tf.constant('test_video_path'),
                                                                                tf.constant(0.00),
                                                                                tf.constant(3.00))
            assert extracted
            self.assertIsNotNone(extracted_frames)
            assert len(extracted_frames) == 180
            assert np.shape(extracted_frames) == (180, 1280, 720, 3)

        mock_get_detection_bounding_box.assert_not_called()
        mock_extract_person_from_frame.assert_not_called()

    @patch.object(RGBSamplesExtractor, 'get_detection_bounding_box', return_value=(False, None))
    @patch.object(RGBSamplesExtractor, 'extract_person_from_frame', return_value=mocks.person_video_frame_mock)
    def test_extract_sample_with_person_detect_no_bounding_box(self,
                                                               mock_extract_person_from_frame,
                                                               mock_get_detection_bounding_box):
        with patch('cv2.VideoCapture', side_effect=mocks.mock_video_capture):
            extracted, extracted_frames = self.samples_extractor_person_detect.extract_sample(tf.constant('video_path'),
                                                                                              tf.constant(0.00),
                                                                                              tf.constant(3.00))
            assert not extracted
            self.assertIsNone(extracted_frames)

        mock_get_detection_bounding_box.assert_called_once()
        mock_extract_person_from_frame.assert_not_called()

    @patch.object(RGBSamplesExtractor, 'get_detection_bounding_box', return_value=(False, None))
    @patch.object(RGBSamplesExtractor, 'extract_person_from_frame', return_value=mocks.person_video_frame_mock)
    def test_extract_sample_with_corrupted_sample(self, mock_extract_person_from_frame,
                                                  mock_get_detection_bounding_box):
        with patch('cv2.VideoCapture', side_effect=mocks.mock_video_capture_no_read):
            extracted, extracted_frames = self.samples_extractor.extract_sample(tf.constant('test_video_path'),
                                                                                tf.constant(0.00),
                                                                                tf.constant(3.00))
            assert not extracted
            self.assertIsNone(extracted_frames)

        mock_get_detection_bounding_box.assert_not_called()
        mock_extract_person_from_frame.assert_not_called()

    @patch.object(RGBSamplesExtractor, 'detect_person_in_frame', return_value=mocks.valid_person_detection_mock)
    def test_bounding_box_with_valid_person_detection(self, mock_detect_person):
        frame = mocks.video_frame_mock
        is_bounding_box_detected, bounding_box = self.samples_extractor_person_detect.get_detection_bounding_box(frame)

        assert is_bounding_box_detected
        tf.assert_equal(bounding_box, [0.08196905, 0.33152586, 0.9854214, 0.6759717])
        mock_detect_person.assert_called_once()

    @patch.object(RGBSamplesExtractor, 'detect_person_in_frame', return_value=mocks.weak_person_detection_mock)
    def test_bounding_box_with_weak_person_detection(self, mock_detect_person):
        frame = mocks.video_frame_mock
        is_bounding_box_detected, bounding_box = self.samples_extractor_person_detect.get_detection_bounding_box(frame)

        assert not is_bounding_box_detected
        tf.assert_equal(bounding_box, tf.zeros(4))
        mock_detect_person.assassert_called_once()

    @patch.object(RGBSamplesExtractor, 'detect_person_in_frame', return_value=mocks.invalid_person_detection_mock)
    def test_bounding_box_with_invalid_person_detection(self, mock_detect_person):
        frame = mocks.video_frame_mock
        is_bounding_box_detected, bounding_box = self.samples_extractor_person_detect.get_detection_bounding_box(frame)

        assert not is_bounding_box_detected
        tf.assert_equal(bounding_box, tf.zeros(4))
        mock_detect_person.assassert_called_once()

    # def test_extract_person_from_frame(self):
    #     frame = mocks.video_frame_mock
    #     bounding_box = mocks.bounding_box_mock
    #
    #     cropped_img = self.samples_extractor.extract_person_from_frame(frame, bounding_box)


if __name__ == "__main__":
    unittest.main()
