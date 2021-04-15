import numpy as np
import os
import logging

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Precision, Recall, AUC
from learning.common.model_utility import ModelUtility


class BaseModelExecutor:
    def __init__(self, model: Model):
        self.logger = logging.getLogger(__name__)

        self.model = model
        self.working_dir = os.getenv('WORK_DIR', '../../../../')

        self.learning_rate = 0.001

        self.model_utility = ModelUtility()
        self.dataset_preparer = None

    def configure(self):
        optimizer = self._get_optimizer()
        loss = self._get_loss()
        metrics = self._get_metrics()

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train_model(self, no_epochs, no_steps_per_epoch=None):
        dataset = self._get_train_dataset()
        history = self.model.fit(dataset, epochs=no_epochs, steps_per_epoch=no_steps_per_epoch, verbose=2)

        # save training history in fs for further review
        history_serialization_path = self._get_model_history_serialization_path()
        np.save(history_serialization_path, history.history)

        # save trained model and weights to the file system for future use
        model_serialization_path = self._get_model_serialization_path()
        self.model.save(model_serialization_path)

    def evaluate_model(self):
        dataset = self._get_test_dataset()
        return self.model.evaluate(dataset)

    def predict_with_model(self):
        dataset = self._get_test_dataset()
        return self.model.predict(dataset)

    def _get_optimizer(self):
        return Adam(learning_rate=self.learning_rate)

    @staticmethod
    def _get_loss():
        return CategoricalCrossentropy()

    @staticmethod
    def _get_metrics():
        return [Recall(), AUC(curve='PR'), Precision()]

    def _get_model_serialization_path(self):
        return self.model_utility.get_model_serialization_path(self._get_model_type())

    def _get_model_history_serialization_path(self):
        return self.model_utility.get_model_history_serialization_path(self._get_model_type())

    def _get_train_dataset(self):
        raise NotImplementedError('_get_train_dataset method not implemented.')

    def _get_test_dataset(self):
        raise NotImplementedError('_get_test_dataset method not implemented.')

    def _get_model_type(self):
        raise NotImplementedError('_get_model_type method not implemented.')
