import tensorflow as tf
import pathlib
import os
from tensorflow.keras import layers
from tensorflow.keras.models import Model


class BaseNetwork:
    def __init__(self, dataset_path, **kwargs):
        self._dataset_path = dataset_path
        self._batch_size = kwargs['BatchSize']
        self._no_classes = kwargs['NoClasses']
        self._epochs = kwargs['NoEpochs']

        self.data_folder = pathlib.Path(self._dataset_path)

    def get_class_names(self):
        pass

    def load_dataset(self):
        pass

    def get_model(self):
        pass

    def train_network(self):
        pass

    def evaluate_network(self):
        pass

    def predict(self):
        pass
