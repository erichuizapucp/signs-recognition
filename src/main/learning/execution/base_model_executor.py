import os
import logging
import tensorflow as tf
import numpy as np

from learning.common.model_utility import ModelUtility
from learning.dataset.prepare.base_dataset_preparer import BaseDatasetPreparer


class BaseModelExecutor:
    def __init__(self,
                 model1: tf.keras.models.Model,
                 model2: tf.keras.models.Model = None,
                 model3: tf.keras.models.Model = None):
        self.logger = logging.getLogger(__name__)
        self.training_logger = logging.getLogger('training')

        self.model1 = model1
        self.model2 = model2
        self.model3 = model3

        self.working_dir = os.getenv('WORK_DIR', 'src/')

        self.learning_rate = 0.001

        self.model_utility = ModelUtility()
        self.dataset_preparer: BaseDatasetPreparer = ...

        self.callback1: tf.keras.callbacks.Callback = ...
        self.callback2: tf.keras.callbacks.Callback = ...
        self.callback3: tf.keras.callbacks.Callback = ...

    def configure(self):
        optimizer = self._get_optimizer()
        loss = self._get_loss()
        metrics = self._get_metrics()

        self.model1.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train_model(self, batch_size, no_epochs, no_steps_per_epoch=None):
        dataset = self._get_train_dataset(batch_size)
        history = self.model1.fit(dataset, epochs=no_epochs, steps_per_epoch=no_steps_per_epoch, verbose=2)

        # save training history in fs for further review
        history_serialization_path = self._get_model_history_serialization_path()
        np.save(history_serialization_path, history.history)

        # save trained model and weights to the file system for future use
        model_serialization_path = self._get_model_serialization_path()
        self.model1.save(model_serialization_path)

    def evaluate_model(self, batch_size):
        dataset = self._get_test_dataset(batch_size)
        return self.model1.evaluate(dataset)

    def predict_with_model(self, batch_size):
        dataset = self._get_test_dataset(batch_size)
        return self.model1.predict(dataset)

    def _get_optimizer(self):
        return tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    @staticmethod
    def _get_loss():
        return tf.keras.losses.CategoricalCrossentropy()

    @staticmethod
    def _get_metrics():
        return [tf.keras.metrics.Recall(), tf.keras.metrics.AUC(curve='PR'), tf.keras.metrics.Precision()]

    def _get_model_serialization_path(self):
        return self.model_utility.get_model_serialization_path(self._get_model_type())

    def _get_model_history_serialization_path(self):
        return self.model_utility.get_model_history_serialization_path(self._get_model_type())

    def _get_train_dataset(self, batch_size):
        dataset = self.dataset_preparer.prepare_train_dataset(batch_size)
        return dataset

    def _get_test_dataset(self, batch_size):
        dataset = self.dataset_preparer.prepare_test_dataset(batch_size)
        return dataset

    def _get_model_type(self):
        raise NotImplementedError('_get_model_type method not implemented.')
