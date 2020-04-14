import logging
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from learning.execution.base_model_executor import BaseModelExecutor
from learning.common import model_type
from learning.dataset.prepare.opticalflow_dataset_preparer import OpticalflowDatasetPreparer
from learning.dataset.prepare.rgb_dataset_preparer import RGBDatasetPreparer


class NSDMExecutor(BaseModelExecutor):
    def __init__(self, model: Model):
        super().__init__(model)
        self.logger = logging.getLogger(__name__)

    # def train_model(self, no_epochs, no_steps_per_epoch=None):
    #     dataset = self._get_train_dataset()
    #     history = self.model.fit(dataset, epochs=no_epochs, steps_per_epoch=no_steps_per_epoch, verbose=2)
    #
    #     # save training history in fs for further review
    #     history_serialization_path = self._get_model_history_serialization_path()
    #     np.save(history_serialization_path, history.history)
    #
    #     # save trained model and weights to the file system for future use
    #     model_serialization_path = self._get_model_serialization_path()
    #     self.model.save(model_serialization_path)

    def _get_train_dataset(self):
        # return both datasets (opticalflow and RGB)
        opticalflow_ds_prep = OpticalflowDatasetPreparer()
        rgb_ds_prep = RGBDatasetPreparer()

        opticalflow_ds = opticalflow_ds_prep.prepare_train_dataset()
        rgb_ds = rgb_ds_prep.prepare_train_dataset()

        no_calls = tf.data.experimental.AUTOTUNE
        # split opticalflow features and labels
        opticalflow_features = opticalflow_ds.map(lambda sample, label: sample, num_parallel_calls=no_calls)
        opticalflow_labels = opticalflow_ds.map(lambda sample, label: label, num_parallel_calls=no_calls)

        for label in opticalflow_labels:
            print(label.numpy())

        # split rgb features and labels
        rgb_features = rgb_ds.map(lambda sample, label: sample, num_parallel_calls=no_calls)
        rgb_labels = rgb_ds.map(lambda sample, label: label, num_parallel_calls=no_calls)

        for label in rgb_labels:
            print(label.numpy())

        # zip opticalflow and rgb features as tuples
        dataset_features = tf.data.Dataset.zip((opticalflow_features, rgb_features))
        # average opticalflow and rgb labels
        # dataset_labels = tf.data.Dataset.zip((opticalflow_labels, rgb_labels))

        # we assume that the labels for opticalflow are the same as RGB so we send just one
        dataset = tf.data.Dataset.zip((dataset_features, opticalflow_labels))

        return dataset

    def __average_target_labels(self, opticalflow_labels: tf.data.Dataset, rgb_labels: tf.data.Dataset):
        opticalflow_labels.interleave(lambda x: rgb_labels.map(lambda y: self.__average_target_labels_fn(x, y)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return None

    def __average_target_labels_fn(self, x, y):
        pass

    # TODO: implement _get_test_dataset
    def _get_test_dataset(self):
        return self._get_train_dataset()

    def _get_model_type(self):
        return model_type.NSDM
