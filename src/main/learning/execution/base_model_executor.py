import tensorflow as tf
import numpy as np
import os
import logging

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Precision, Recall, AUC
from learning.dataset.tf_record_dataset_reader import TFRecordDatasetReader
from learning.common.imagenet_config import IMAGENET_CONFIG
from learning.common.model_utility import ModelUtility


class BaseModelExecutor:
    def __init__(self, model: Model):
        self.logger = logging.getLogger(__name__)

        self.model = model

        self.working_dir = os.getenv('WORK_DIR', './')
        # self.pre_trained_models_dir = 'pre-trained-models'
        self.dataset_dir = 'serialized-dataset'

        self.imagenet_img_width = IMAGENET_CONFIG['imagenet_img_width']
        self.imagenet_img_height = IMAGENET_CONFIG['imagenet_img_height']
        self.rgb_no_channels = IMAGENET_CONFIG['rgb_no_channels']

        self.learning_rate = 0.001
        self.dataset_batch_size = 64

        self.model_utility = ModelUtility()

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

    def _get_train_dataset(self):
        dataset_path = self._get_dataset_path()
        dataset_reader = TFRecordDatasetReader(self._get_dataset_type(), dataset_path)
        dataset = dataset_reader.read()
        return dataset

    def _get_test_dataset(self):
        return os.path.join(self.working_dir, self.dataset_dir, self._get_dataset_type())

    def _get_dataset_path(self):
        return os.path.join(self.working_dir, self.dataset_dir, self._get_dataset_type())

    def _get_optimizer(self):
        return Adam(learning_rate=self.learning_rate)

    def _get_loss(self):
        return CategoricalCrossentropy()

    def _get_metrics(self):
        return [Recall(), AUC(curve='PR'), Precision()]

    def _get_model_serialization_path(self):
        return self.model_utility.get_model_serialization_path(self._get_model_type())

    def _get_model_history_serialization_path(self):
        return self.model_utility.get_model_history_serialization_path(self._get_model_type())

    def _transform_image(self, img):
        # convert to integers
        img = tf.image.decode_jpeg(img, channels=self.rgb_no_channels)
        # convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, [self.imagenet_img_width, self.imagenet_img_height])

    def _prepare_sample(self, sample, label):
        raise NotImplementedError('_prepare_sample method not implemented.')

    def _get_dataset_type(self):
        raise NotImplementedError('_get_dataset_type method not implemented.')

    def _get_model_type(self):
        raise NotImplementedError('_get_model_type method not implemented.')
