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
        dataset = dataset.map(self._prepare_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(self.dataset_batch_size)
        return dataset

    # TODO: implement prepare_test_dataset
    def prepare_test_dataset(self):
        return self.prepare_train_dataset()

    def transform_feature_for_predict(self, **kwargs):
        raise NotImplementedError('This method is not supported for the Opticalflow dataset')

    def _prepare_sample(self, opticalflow_sample, label):
        transformed_img = self._transform_image(opticalflow_sample)
        return transformed_img, label

    def _get_dataset_type(self):
        return OPTICAL_FLOW

    def _prepare_sample2(self, feature1, feature2, label):
        raise NotImplementedError('This method is not supported for the Opticalflow dataset')
