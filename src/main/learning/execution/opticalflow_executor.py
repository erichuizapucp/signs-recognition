import os
import logging

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.models import Model

from learning.execution.model_executor import ModelExecutor
from learning.dataset.tf_record_dataset_reader import TFRecordDatasetReader
from learning.common.dataset_type import OPTICAL_FLOW


class OpticalflowExecutor(ModelExecutor):
    def __init__(self, model: Model):
        super().__init__(model)
        self.logger = logging.getLogger(__name__)

        self.learning_rate = 0.001
        self.pre_trained_model_file = 'opticalflow.h5'
        self.training_history_file = 'opticalflow_history.npy'

    def _get_train_dataset(self):
        dataset_path = self._get_dataset_path()
        dataset_reader = TFRecordDatasetReader(OPTICAL_FLOW, dataset_path)
        dataset = dataset_reader.read()

        # map all dataset images transformations
        dataset = dataset.map(self._prepare_single_image)
        return dataset

    def _get_test_dataset(self):
        return os.path.join(self.working_dir, self.dataset_dir, 'opticalflow')

    def _get_dataset_path(self):
        return os.path.join(self.working_dir, self.dataset_dir, 'opticalflow')

    def _get_optimizer(self):
        return Adam(learning_rate=self.learning_rate)

    def _get_loss(self):
        return SparseCategoricalCrossentropy()

    def _get_metrics(self):
        return [Recall(), AUC(curve='PR'), Precision()]

    def _get_model_serialization_path(self):
        return self._build_serialization_path(self.pre_trained_models_dir, self.pre_trained_model_file)

    def _get_model_history_serialization_path(self):
        return self._build_serialization_path(self.pre_trained_models_dir, self.training_history_file)
