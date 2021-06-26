import logging

from tensorflow.keras.models import Model
from base_model_executor import BaseModelExecutor
from learning.common import model_type
from legacy.rgb_dataset_preparer import RGBDatasetPreparer


class RGBExecutor(BaseModelExecutor):
    def __init__(self, model: Model):
        super().__init__(model)
        self.logger = logging.getLogger(__name__)

        self.dataset_preparer = RGBDatasetPreparer()

    def _get_train_dataset(self):
        dataset = self.dataset_preparer.prepare_train_dataset()
        return dataset

    # TODO: implement _get_test_dataset
    def _get_test_dataset(self):
        return self._get_train_dataset()

    def _get_model_type(self):
        return model_type.RGB
