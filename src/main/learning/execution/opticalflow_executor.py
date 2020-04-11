import logging
import tensorflow as tf

from tensorflow.keras.models import Model
from learning.execution.base_model_executor import BaseModelExecutor
from learning.common import dataset_type, model_type


class OpticalflowExecutor(BaseModelExecutor):
    def __init__(self, model: Model):
        super().__init__(model)
        self.logger = logging.getLogger(__name__)

    def _get_train_dataset(self):
        dataset = super()._get_train_dataset()

        # apply image transformation and data augmentation
        dataset = dataset.map(lambda img, height, width, depth, label: self._prepare_sample(img, label),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(self.dataset_batch_size)
        return dataset

    def _prepare_sample(self, sample, label):
        transformed_img = self._transform_image(sample)
        return transformed_img, label

    def _get_dataset_type(self):
        return dataset_type.OPTICAL_FLOW

    def _get_model_type(self):
        return model_type.OPTICAL_FLOW
