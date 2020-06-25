import os
import logging
import tensorflow as tf

from learning.dataset.tfrecord.tf_record_dataset_reader import TFRecordDatasetReader
from learning.common.common_config import IMAGENET_CONFIG, FRAMES_SEQ_CONFIG, DATASET_BATCH_SIZE


class BaseDatasetPreparer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        self.working_dir = os.getenv('WORK_DIR', './')
        self.dataset_dir = 'serialized-dataset'

        self.imagenet_img_width = IMAGENET_CONFIG['img_width']
        self.imagenet_img_height = IMAGENET_CONFIG['img_height']
        self.imagenet_rgb_no_channels = IMAGENET_CONFIG['rgb_no_channels']

        self.frames_seq_img_width = FRAMES_SEQ_CONFIG['img_width']
        self.frames_seq_img_height = FRAMES_SEQ_CONFIG['img_height']
        self.frames_seq_no_channels = FRAMES_SEQ_CONFIG['rgb_no_channels']

        self.rgb_feature_dim = self.frames_seq_img_width * self.frames_seq_img_height * self.frames_seq_no_channels

        self.dataset_batch_size = DATASET_BATCH_SIZE

    def prepare_train_dataset(self):
        return self._prepare(self._get_train_dataset_path)

    def prepare_test_dataset(self):
        return self._prepare(self._get_test_dataset_path)

    def _prepare(self, get_dataset_path_func):
        dataset_path = get_dataset_path_func()
        dataset_reader = TFRecordDatasetReader(self._get_dataset_type(), dataset_path)
        dataset = dataset_reader.read()
        return dataset

    def _get_train_dataset_path(self):
        return os.path.join(self.working_dir, self.dataset_dir, self._get_dataset_type())

    # TODO: for now the test dataset path is the same as the train dataset path
    def _get_test_dataset_path(self):
        return self._get_train_dataset_path()

    def _transform_image(self, img, resize_shape=None):
        if resize_shape is None:
            resize_shape = [self.imagenet_img_width, self.imagenet_img_height]

        # convert to integers
        img = tf.image.decode_jpeg(img, channels=self.imagenet_rgb_no_channels)
        # convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, resize_shape)

    def _transform_decoded_image(self, img, resize_shape=None):
        if resize_shape is None:
            resize_shape = [self.imagenet_img_width, self.imagenet_img_height]

        # convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, resize_shape)

    def _py_transform_frame_seq(self, sample):
        transformed_images = []
        for image in sample:  # iterate over each frame in the sample
            # decode, scale and resize frame
            transformed_img = self._transform_image(image, [self.frames_seq_img_width, self.frames_seq_img_height])

            # reshape a frame from 2D to 1D
            transformed_img = tf.reshape(transformed_img, [self.rgb_feature_dim])

            transformed_images.extend([transformed_img])
        return tf.stack(transformed_images)  # shape (no_frames, flattened_dim)

    def _py_transform_decoded_frame_seq(self, sample):
        transformed_images = []
        for image in sample:  # iterate over each frame in the sample
            # decode, scale and resize frame
            transformed_img = self._transform_decoded_image(image,
                                                            [self.frames_seq_img_width, self.frames_seq_img_height])

            # reshape a frame from 2D to 1D
            transformed_img = tf.reshape(transformed_img, [self.rgb_feature_dim])

            transformed_images.extend([transformed_img])
        return tf.stack(transformed_images)  # shape (no_frames, flattened_dim)

    def _prepare_sample(self, feature, label):
        raise NotImplementedError('_prepare_sample method not implemented.')

    def _prepare_sample2(self, feature1, feature2, label):
        raise NotImplementedError('_prepare_sample method not implemented.')

    def _get_dataset_type(self):
        raise NotImplementedError('_get_dataset_type method not implemented.')

    def transform_feature_for_predict(self):
        raise NotImplementedError('transform_feature_for_prediction method not implemented.')
