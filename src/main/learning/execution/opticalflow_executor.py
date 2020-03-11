from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from model_executor import ModelExecutor
from learning.common.dataset_type import OPTICAL_FLOW


class OpticalflowExecutor(ModelExecutor):
    def __init__(self):
        super().__init__(OPTICAL_FLOW)

        self.learning_rate = 0.001

    def _get_optimizer(self):
        return Adam(learning_rate=self.learning_rate)

    def _get_loss(self):
        return SparseCategoricalCrossentropy()

    def _get_metrics(self):
        raise NotImplementedError('get_metrics method not implemented.')

    def _get_saved_model_path(self):
        raise NotImplementedError('get_saved_model_path method not implemented.')
