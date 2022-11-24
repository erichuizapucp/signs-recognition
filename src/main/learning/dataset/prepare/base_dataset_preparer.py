import logging
import tensorflow as tf

from learning.common.common_config import IMAGENET_CONFIG, FRAMES_SEQ_CONFIG


class BaseDatasetPreparer:
    def __init__(self, train_dataset_path, test_dataset_path):
        self.logger = logging.getLogger(__name__)

        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path

        self.imagenet_img_width = IMAGENET_CONFIG['img_width']
        self.imagenet_img_height = IMAGENET_CONFIG['img_height']
        self.imagenet_rgb_no_channels = IMAGENET_CONFIG['rgb_no_channels']

        self.frames_seq_img_width = FRAMES_SEQ_CONFIG['img_width']
        self.frames_seq_img_height = FRAMES_SEQ_CONFIG['img_height']
        self.frames_seq_no_channels = FRAMES_SEQ_CONFIG['rgb_no_channels']

        self.rgb_feature_dim = self.frames_seq_img_width * self.frames_seq_img_height * self.frames_seq_no_channels

    def prepare_train_dataset(self, batch_size):
        return self.prepare_dataset(self.train_dataset_path, batch_size)

    def prepare_test_dataset(self, batch_size):
        return self.prepare_dataset(self.test_dataset_path, batch_size)

    def prepare(self, batch_size):
        return self.prepare_train_dataset(batch_size), self.prepare_test_dataset(batch_size)

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

    @staticmethod
    def _transform(input_data, trans_ops):
        transformed_input = input_data
        for op in trans_ops:
            transformed_input = op(transformed_input)
        return transformed_input

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

    def prepare_dataset(self, dataset_path, batch_size):
        raise NotImplementedError('_prepare method not implemented.')

    def prepare_sample(self, feature, label):
        raise NotImplementedError('_prepare_sample method not implemented.')

    def prepare_sample2(self, feature1, feature2, label):
        raise NotImplementedError('_prepare_sample method not implemented.')

    def prepare_sample3(self, feature):
        raise NotImplementedError('_prepare_sample3 method not implemented.')
