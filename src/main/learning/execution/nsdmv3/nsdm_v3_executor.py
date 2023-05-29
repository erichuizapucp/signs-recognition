import logging
import tensorflow as tf

from learning.common import model_type
from learning.execution.base_model_executor import BaseModelExecutor


class NSDMExecutorV3(BaseModelExecutor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(__name__)

    def get_model_type(self):
        return model_type.NSDMV3

    @staticmethod
    def get_metrics():
        return [tf.keras.metrics.AUC(curve='PR')]
