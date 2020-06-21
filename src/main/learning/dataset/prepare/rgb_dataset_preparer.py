import logging
import tensorflow as tf

from learning.dataset.prepare.base_dataset_preparer import BaseDatasetPreparer
from learning.common.dataset_type import RGB


class RGBDatasetPreparer(BaseDatasetPreparer):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

    def prepare_train_dataset(self):
        dataset = super().prepare_train_dataset()
        # apply image transformation and data augmentation
        dataset = dataset.map(self._prepare_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.padded_batch(self.dataset_batch_size, padded_shapes=([None, None], [None]))
        return dataset

    # TODO: implement prepare_test_dataset
    def prepare_test_dataset(self):
        return self.prepare_train_dataset()

    def _prepare_sample(self, feature, label):
        # RNN feature
        transformed_frames_seq = tf.py_function(self.py_transform_frame_seq, [feature], tf.float32)
        # RNN cells require a 3D input (batch, steps, feature)
        return transformed_frames_seq, label

    def _prepare_sample2(self, feature1, feature2, label):
        raise NotImplementedError('This method is not supported for the RGB dataset')

    def _get_dataset_type(self):
        return RGB
