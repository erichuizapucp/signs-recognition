import logging
import tensorflow as tf

from learning.execution.base_model_executor import BaseModelExecutor
from learning.common import model_type


class NSDMExecutor(BaseModelExecutor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(__name__)

    def _get_train_dataset(self, dataset):
        dataset = dataset.map(self.__map_combined_sample,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset

    def _get_test_dataset(self, dataset):
        dataset = dataset.map(self.__map_combined_sample,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset

    def _get_model_type(self):
        return model_type.NSDM

    def get_callback(self, checkpoint_storage_path, model):
        pass

    @staticmethod
    def __map_combined_sample(opticalflow_feature, rgb_feature, label):
        # opticalflow and rgb features should be sent in a tuple so that Keras can send them to the opticalflow and rgb
        # models respectively.
        return (opticalflow_feature, rgb_feature), label
