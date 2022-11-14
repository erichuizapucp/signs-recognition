import os
import unittest
import tensorflow as tf
from unittest.mock import patch
from pre_processing.isolated_detection.samples_generation.rgb_person_sample_extractor import RGBPersonSamplesExtractor
from learning.common.model_utility import ModelUtility
from mocks import video_processing_mocks as mocks


class TestRGBPersonSamplesExtractor(unittest.TestCase):
    def setUp(self):
        os.environ['WORK_DIR'] = '../../'

        self.samples_extractor = RGBPersonSamplesExtractor(detection_model=mocks.person_detection_model_mock)
        self.model_utility = ModelUtility()
        self.video_path = 'fixtures/consultant-02-session-01-part-01-00.mp4'
        self.person_detection_model_name = 'centernet_resnet50_v1_fpn_512x512_coco17_tpu-8'
        self.person_detection_checkout_prefix = 'ckpt-0'

    def test_extract_sample_with_person_detect(self):
        detection_model = self.model_utility.get_object_detection_model(self.person_detection_model_name,
                                                                        self.person_detection_checkout_prefix)
        self.samples_extractor.detection_model = detection_model
        success, extracted_frames = self.samples_extractor.extract_sample(self.video_path, 0.00, 2.00)

        assert success
        assert extracted_frames.size() > 0
        assert extracted_frames.size() == 60
        assert extracted_frames.element_shape == (672, 448, 3)

        # frames = extracted_frames.stack()
        # for index, frame in enumerate(frames):
        #     file_name = 'fixtures/extracted_person_sample/frame_{}.jpg'.format(index)
        #     tf.io.write_file(file_name, tf.image.encode_jpeg(frame))

    @patch.object(RGBPersonSamplesExtractor, 'get_detection_bounding_box', return_value=(False, None))
    @patch.object(RGBPersonSamplesExtractor, 'extract_person_from_frame', return_value=mocks.person_video_frame_mock)
    @patch('cv2.VideoCapture', mocks.mock_video_capture)
    @patch('cv2.cvtColor', mocks.mock_cv2_cvt_color)
    def test_extract_sample_with_person_detect_no_bounding_box(self,
                                                               mock_extract_person_from_frame,
                                                               mock_get_detection_bounding_box):
        success, extracted_frames = self.samples_extractor.extract_sample(self.video_path, 0.00, 3.00)
        assert not success
        self.assertIsNone(extracted_frames)

        mock_get_detection_bounding_box.assert_called_once()
        mock_extract_person_from_frame.assert_not_called()

    @patch.object(RGBPersonSamplesExtractor, 'get_detection_bounding_box', return_value=(False, None))
    @patch.object(RGBPersonSamplesExtractor, 'extract_person_from_frame', return_value=mocks.person_video_frame_mock)
    @patch('cv2.VideoCapture', mocks.mock_video_capture_no_read)
    @patch('cv2.cvtColor', mocks.mock_cv2_cvt_color)
    def test_extract_sample_with_corrupted_sample(self,
                                                  mock_extract_person_from_frame,
                                                  mock_get_detection_bounding_box):
        extracted, extracted_frames = self.samples_extractor.extract_sample(self.video_path, 0.00, 3.00)
        assert not extracted
        self.assertIsNone(extracted_frames)

        mock_get_detection_bounding_box.assert_not_called()
        mock_extract_person_from_frame.assert_not_called()

    @patch.object(RGBPersonSamplesExtractor, 'detect_person_in_frame', return_value=mocks.valid_person_detection_mock)
    def test_bounding_box_with_valid_person_detection(self, mock_detect_person):
        frame = mocks.video_frame_mock
        is_bounding_box_detected, bounding_box = self.samples_extractor.get_detection_bounding_box(frame)

        assert is_bounding_box_detected
        tf.assert_equal(bounding_box, [0.08196905, 0.33152586, 0.9854214, 0.6759717])
        mock_detect_person.assert_called_once()

    @patch.object(RGBPersonSamplesExtractor, 'detect_person_in_frame', return_value=mocks.weak_person_detection_mock)
    def test_bounding_box_with_weak_person_detection(self, mock_detect_person):
        frame = mocks.video_frame_mock
        is_bounding_box_detected, bounding_box = self.samples_extractor.get_detection_bounding_box(frame)

        assert not is_bounding_box_detected
        tf.assert_equal(bounding_box, tf.zeros(4))
        mock_detect_person.assassert_called_once()

    @patch.object(RGBPersonSamplesExtractor, 'detect_person_in_frame', return_value=mocks.invalid_person_detection_mock)
    def test_bounding_box_with_invalid_person_detection(self, mock_detect_person):
        frame = mocks.video_frame_mock
        is_bounding_box_detected, bounding_box = self.samples_extractor.get_detection_bounding_box(frame)

        assert not is_bounding_box_detected
        tf.assert_equal(bounding_box, tf.zeros(4))
        mock_detect_person.assassert_called_once()


if __name__ == "__main__":
    unittest.main()
