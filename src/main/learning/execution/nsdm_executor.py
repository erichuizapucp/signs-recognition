import logging

from tensorflow.keras.models import Model
from learning.execution.model_executor import ModelExecutor


class NsdmExecutor(ModelExecutor):
    def __init__(self, model: Model, working_dir):
        super().__init__(model, working_dir)
        self.logger = logging.getLogger(__name__)

    def _get_train_dataset(self):
        pass

    def _get_test_dataset(self):
        pass

    def _get_optimizer(self):
        pass

    def _get_loss(self):
        pass

    def _get_metrics(self):
        pass

    def _get_model_serialization_path(self):
        pass

    def _get_model_history_serialization_path(self):
        pass

    def _get_dataset_path(self):
        pass
