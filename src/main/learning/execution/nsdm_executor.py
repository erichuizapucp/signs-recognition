import logging
import tensorflow as tf

from tensorflow.keras.models import Model
from learning.execution.base_model_executor import BaseModelExecutor
from learning.common import model_type
from learning.dataset.prepare.combined_dataset_preparer import CombinedDatasetPreparer


class NSDMExecutor(BaseModelExecutor):
    def __init__(self, model: Model):
        super().__init__(model)
        self.logger = logging.getLogger(__name__)

        self.dataset_preparer = CombinedDatasetPreparer()

    def _get_train_dataset(self):
        dataset = self.dataset_preparer.prepare_train_dataset()
        dataset = dataset.map(self.__map_combined_sample,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return dataset

    # TODO: implement _get_test_dataset
    def _get_test_dataset(self):
        return self._get_train_dataset()

    def _get_model_type(self):
        return model_type.NSDM

    @staticmethod
    def __map_combined_sample(opticalflow_feature, rgb_feature, label):
        # opticalflow and rgb features should be sent in a tuple so that Keras can send them to the opticalflow and rgb
        # models respectively.
        return (opticalflow_feature, rgb_feature), label
