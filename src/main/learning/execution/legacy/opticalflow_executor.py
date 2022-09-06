import logging

from learning.execution.base_model_executor import BaseModelExecutor
from learning.common import model_type


class OpticalflowExecutor(BaseModelExecutor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(__name__)

    def _get_model_type(self):
        return model_type.OPTICAL_FLOW

    def get_callback(self, checkpoint_storage_path, model):
        pass
