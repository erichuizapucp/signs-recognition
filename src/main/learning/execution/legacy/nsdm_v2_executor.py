import logging

from learning.execution.legacy.nsdm_executor import NSDMExecutor
from learning.common import model_type


class NSDMExecutorV2(NSDMExecutor):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

    def _get_model_type(self):
        return model_type.NSDMV2
