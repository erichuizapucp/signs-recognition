import logging
import tensorflow as tf

from learning.dataset.prepare.base_dataset_preparer import BaseDatasetPreparer
from learning.common.dataset_type import COMBINED


class CombinedDatasetPreparer(BaseDatasetPreparer):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

    def prepare_train_dataset(self):
        dataset = super().prepare_train_dataset()
        # apply transformations
        dataset = dataset.map(self._prepare_sample2, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.padded_batch(self.dataset_batch_size,
                                       padded_shapes=([None, None, None], [None, None], [None]))
        return dataset

    # TODO: implement prepare_test_dataset
    def prepare_test_dataset(self):
        return self.prepare_train_dataset()

    def _prepare_sample(self, feature, sample):
        raise NotImplementedError('This method is not supported for the Combined dataset')

    def _prepare_sample2(self, feature1, feature2, label):
        # feature1: opticalflow image
        # feature2: rgb frames set

        transformed_img = self._transform_image(feature1)
        transformed_frames_seq = tf.py_function(self.py_transform_frame_seq, [feature2], tf.float32)

        return transformed_img, transformed_frames_seq, label

    def _get_dataset_type(self):
        return COMBINED
