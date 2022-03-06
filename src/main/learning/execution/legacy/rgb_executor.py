import logging
import tensorflow as tf

from learning.execution.base_model_executor import BaseModelExecutor
from learning.common import model_type
from learning.dataset.prepare.legacy.rgb_dataset_preparer import RGBDatasetPreparer


class RGBExecutor(BaseModelExecutor):
    def __init__(self, model: tf.keras.models.Model, train_dataset_path=None, test_dataset_path=None):
        super().__init__(model)
        self.logger = logging.getLogger(__name__)

        self.dataset_preparer = RGBDatasetPreparer(train_dataset_path, test_dataset_path)

    def _get_model_type(self):
        return model_type.RGB
