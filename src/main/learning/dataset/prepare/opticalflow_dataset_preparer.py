import logging
import tensorflow as tf

from learning.dataset.prepare.base_dataset_preparer import BaseDatasetPreparer
from learning.common.dataset_type import OPTICAL_FLOW


class OpticalflowDatasetPreparer(BaseDatasetPreparer):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

    def prepare_train_dataset(self):
        dataset = super().prepare_train_dataset()

        # apply image transformation and data augmentation
        dataset = dataset.map(lambda img, height, width, depth, label: self._prepare_sample(img, label),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(self.dataset_batch_size)
        return dataset

    # TODO: implement prepare_test_dataset
    def prepare_test_dataset(self):
        return self.prepare_train_dataset()

    def _prepare_sample(self, sample, label):
        transformed_img = self._transform_image(sample)
        return transformed_img, label

    def _get_dataset_type(self):
        return OPTICAL_FLOW
