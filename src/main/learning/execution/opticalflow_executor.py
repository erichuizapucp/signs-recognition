import os
import datetime

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Precision, Recall, AUC

from model_executor import ModelExecutor
from learning.common.dataset_type import OPTICAL_FLOW


class OpticalflowExecutor(ModelExecutor):
    def __init__(self):
        super().__init__(OPTICAL_FLOW)
        self.learning_rate = 0.001
        self.pre_trained_model_file = 'opticalflow.h5'

    def _get_optimizer(self):
        return Adam(learning_rate=self.learning_rate)

    def _get_loss(self):
        return SparseCategoricalCrossentropy()

    def _get_metrics(self):
        return [Recall(), AUC(curve='PR'), Precision()]

    def _get_saved_model_path(self):
        return self._build_saved_model_path(self.pre_trained_model_file)
