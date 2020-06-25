import logging

from tensorflow.keras.models import Model
from learning.execution.nsdm_executor import NSDMExecutor
from learning.common import model_type


class NSDMExecutorV2(NSDMExecutor):
    def __init__(self, model: Model):
        super().__init__(model)
        self.logger = logging.getLogger(__name__)

    def _get_model_type(self):
        return model_type.NSDMV2
