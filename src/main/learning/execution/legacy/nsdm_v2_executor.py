import logging
import tensorflow as tf

from learning.execution.legacy.nsdm_executor import NSDMExecutor
from learning.common import model_type


class NSDMExecutorV2(NSDMExecutor):
    def __init__(self, model: tf.keras.models.Model, train_dataset_path=None, test_dataset_path=None):
        super().__init__(model, train_dataset_path, test_dataset_path)
        self.logger = logging.getLogger(__name__)

    def _get_model_type(self):
        return model_type.NSDMV2