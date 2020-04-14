import logging
import tensorflow as tf

from learning.dataset.prepare.base_dataset_preparer import BaseDatasetPreparer
from learning.common.dataset_type import RGB


class RGBDatasetPreparer(BaseDatasetPreparer):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        self.feature_dim = self.imagenet_img_width * self.imagenet_img_height * self.rgb_no_channels

    def prepare_train_dataset(self):
        dataset = super().prepare_train_dataset()
        # apply image transformation and data augmentation
        dataset = dataset.map(
            lambda frames_seq, height, width, depth, label: self._prepare_sample(frames_seq, label),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.padded_batch(self.dataset_batch_size, padded_shapes=([None, None], [None]))
        return dataset

    # TODO: implement prepare_test_dataset
    def prepare_test_dataset(self):
        return self.prepare_train_dataset()

    def _prepare_sample(self, sample, label):
        # RNN feature
        transformed_frames_seq = tf.py_function(self.py_transform_frame_seq, [sample], tf.float32)
        # RNN cells require a 3D input (batch, steps, feature)
        return transformed_frames_seq, label

    def py_transform_frame_seq(self, sample):
        transformed_images = []
        for image in sample:
            transformed_img = tf.reshape(self._transform_image(image), [self.feature_dim])
            transformed_images.extend([transformed_img])
        return tf.stack(transformed_images)

    def _get_dataset_type(self):
        return RGB

