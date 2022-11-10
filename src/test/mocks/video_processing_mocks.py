import tensorflow as tf
from unittest.mock import Mock

person_detection_model_mock = Mock(name='person_detection_model_mock')
video_frame_mock = tf.random.uniform(shape=[1280, 720, 3], maxval=255, dtype=tf.int32)  # np.random.randint(low=0, high=255, size=(1280, 720, 3))
person_video_frame_mock = tf.random.uniform(shape=[672, 448, 3], maxval=255, dtype=tf.int32)
bounding_box_mock = tf.constant([0.08196905, 0.33152586, 0.9854214, 0.6759717])

valid_person_detection_mock = (tf.constant([[[0.08196905, 0.33152586, 0.9854214, 0.6759717]]]), tf.constant([[0]]), tf.constant([[0.9]]))
invalid_person_detection_mock = (tf.constant([[[0.08196905, 0.33152586, 0.9854214, 0.6759717]]]), tf.constant([[2]]), tf.constant([[0.9]]))
weak_person_detection_mock = (tf.zeros(4), tf.constant(([[0]])), tf.constant([[0.4]]))


def mock_video_capture(video_path):
    video_capture_mock = Mock(name=video_path)
    video_capture_mock.get.return_value = 60
    video_capture_mock.read.return_value = (True, video_frame_mock)

    return video_capture_mock


def mock_video_capture_no_read(video_path):
    video_capture_mock = Mock(name=video_path)
    video_capture_mock.get.return_value = 60
    video_capture_mock.read.return_value = (False, None)

    return video_capture_mock
