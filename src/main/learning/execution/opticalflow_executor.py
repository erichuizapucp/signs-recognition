import logging
import tensorflow as tf

from tensorflow.keras.models import Model
from learning.execution.model_executor import ModelExecutor
from learning.common.dataset_type import OPTICAL_FLOW


class OpticalflowExecutor(ModelExecutor):
    def __init__(self, model: Model, working_dir):
        super().__init__(model, working_dir)
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
        return OPTICAL_FLOW

    def _get_pre_trained_model_filename(self):
        return 'opticalflow.h5'

    def _get_training_history_filename(self):
        return 'opticalflow_history.npy'
