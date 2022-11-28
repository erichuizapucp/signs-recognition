import numpy as np
import tensorflow as tf
import os

from unittest.mock import Mock

person_detection_model_mock = Mock(name='person_detection_model_mock')
video_frame_mock = np.random.randint(low=0, high=255, size=(1280, 720, 3))
person_video_frame_mock = tf.random.uniform(shape=[672, 448, 3], maxval=255, dtype=tf.int32)
bounding_box_mock = tf.constant([0.08196905, 0.33152586, 0.9854214, 0.6759717])

valid_person_detection_mock = (tf.constant([[[0.08196905, 0.33152586, 0.9854214, 0.6759717]]]), tf.constant([[0]]), tf.constant([[0.9]]))
invalid_person_detection_mock = (tf.constant([[[0.08196905, 0.33152586, 0.9854214, 0.6759717]]]), tf.constant([[2]]), tf.constant([[0.9]]))
weak_person_detection_mock = (tf.zeros(4), tf.constant(([[0]])), tf.constant([[0.4]]))


high_res_multicrop_frame_mock = tf.random.uniform(shape=[224, 224, 3], maxval=1, dtype=tf.float32)
low_res_multicrop_frame_mock = tf.random.uniform(shape=[96, 96, 3], maxval=1, dtype=tf.float32)


video_path_list = ['fixtures/consultant-12-sesion-01-part-02-section-02.mp4',
                   'fixtures/consultant-12-sesion-01-part-02-section-02.mp4',
                   'fixtures/consultant-12-sesion-01-part-02-section-02.mp4',
                   'fixtures/consultant-12-sesion-01-part-02-section-02.mp4',
                   'fixtures/consultant-12-sesion-01-part-02-section-02.mp4',
                   'fixtures/consultant-12-sesion-01-part-02-section-02.mp4',
                   'fixtures/consultant-12-sesion-01-part-02-section-02.mp4',
                   'fixtures/consultant-12-sesion-01-part-02-section-02.mp4',
                   'fixtures/consultant-12-sesion-01-part-02-section-02.mp4',
                   'fixtures/consultant-02-session-01-part-01-00.mp4',
                   'fixtures/consultant-12-sesion-01-part-02-section-02.mp4',
                   'fixtures/consultant-24-sesion-01-part-03.mp4',
                   'fixtures/consultant-12-sesion-01-part-02-section-02.mp4',
                   'fixtures/consultant-12-sesion-01-part-02-section-02.mp4']
chunk_start_list = [30.0, 80.0, 100.0, 110.0, 0.0, 10.0, 20.0, 60.0, 50.0, 0.0, 90.0, 0.0, 40.0, 70.0]
chunk_end_list = [40.0, 90.0, 110.0, 125.64, 10.0, 20.0, 30.0, 70.0, 60.0, 6.006, 100.0, 5.005, 50.0, 80.0]


def mock_cv2_cvt_color(frame, param):
    return video_frame_mock


def mock_video_capture(video_path):
    video_capture_mock = Mock(name=video_path)
    video_capture_mock.get.return_value = 30
    video_capture_mock.read.return_value = (True, video_frame_mock)

    return video_capture_mock


def mock_video_capture_no_read(video_path):
    video_capture_mock = Mock(name=video_path)
    video_capture_mock.get.return_value = 60
    video_capture_mock.read.return_value = (False, None)

    return video_capture_mock


def get_sample_mock(sample_path):
    mock_sample = tf.TensorArray(dtype=tf.uint8, size=0, dynamic_size=True, clear_after_read=False)

    index = 0
    for frame_file_name in sorted(os.listdir(sample_path)):
        if frame_file_name.endswith('.jpg'):
            frame_file = tf.io.read_file(os.path.join(sample_path, frame_file_name))
            mock_sample = mock_sample.write(index, tf.image.decode_jpeg(frame_file, 3))
            index = index + 1

    return mock_sample


def extract_sample_some_invalid(video_path, start_time, end_time):
    return (True, get_sample_mock('fixtures/extracted_person_sample/')) \
        if video_path == 'fixtures/consultant-12-sesion-01-part-02-section-02.mp4' else (False, None)


def mock_multicrop_tie_together(video_fragment, min_scale, max_scale, crop_size):
    return high_res_multicrop_frame_mock if crop_size == 224 else low_res_multicrop_frame_mock
