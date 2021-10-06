import logging

from tensorflow.keras.models import Model
from learning.execution.base_model_executor import BaseModelExecutor
from learning.common import model_type
from learning.dataset.prepare.legacy.opticalflow_dataset_preparer import OpticalflowDatasetPreparer


class OpticalflowExecutor(BaseModelExecutor):
    def __init__(self, model: Model, train_dataset_path=None, test_dataset_path=None):
        super().__init__(model)
        self.logger = logging.getLogger(__name__)

        self.dataset_preparer = OpticalflowDatasetPreparer(train_dataset_path, test_dataset_path)

    def _get_model_type(self):
        return model_type.OPTICAL_FLOW
