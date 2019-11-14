import tensorflow as tf
import pathlib
import os
from tensorflow.keras import layers
from tensorflow.keras.models import Model


class BaseModel(tf.keras.Model):
    def __init__(self, dataset_path, **kwargs):
        super(BaseModel, self).__init__()

        self._dataset_path = dataset_path
        self._batch_size = kwargs['BatchSize']
        self._no_classes = kwargs['NoClasses']
        self._epochs = kwargs['NoEpochs']
        self._model = None

        self.data_folder = pathlib.Path(self._dataset_path)

    def call(self, inputs, training=None, mask=None):
        pass

    def configure(self):
        pass

    def get_class_names(self):
        pass

    def load_dataset(self):
        pass

    def configure_model(self):
        pass

    def train_network(self):
        self.configure_model()

    def evaluate_network(self):
        pass

    def predict(self):
        pass
