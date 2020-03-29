import tensorflow as tf
import logging

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Recall, AUC, Precision
from learning.execution.model_executor import ModelExecutor


class RGBExecutor(ModelExecutor):
    def __init__(self, model: Model, working_dir):
        super().__init__(model, working_dir)
        self.logger = logging.getLogger(__name__)

        self.learning_rate = 0.001
        self.pre_trained_model_file = 'rgb.h5'
        self.training_history_file = 'rgb_history.npy'

    def _get_train_dataset(self):
        pass

    def _get_test_dataset(self):
        pass

    def _get_optimizer(self):
        return Adam(learning_rate=self.learning_rate)

    def _get_loss(self):
        return SparseCategoricalCrossentropy()

    def _get_metrics(self):
        return [Recall(), AUC(curve='PR'), Precision()]

    def _get_model_serialization_path(self):
        pass

    def _get_model_history_serialization_path(self):
        pass

    def _get_dataset_path(self):
        pass
