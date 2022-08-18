import logging

from learning.execution.base_model_executor import BaseModelExecutor
from learning.common import model_type


class OpticalflowExecutor(BaseModelExecutor):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

    def _get_model_type(self):
        return model_type.OPTICAL_FLOW
