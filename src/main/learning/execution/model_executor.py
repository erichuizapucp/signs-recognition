import tensorflow as tf
import numpy as np
import os
import logging

from tensorflow.keras.models import Model


class ModelExecutor:
    def __init__(self, model: Model):
        self.logger = logging.getLogger(__name__)

        self.model = model

        self.working_dir = os.getenv('WORK_DIR', './')
        self.pre_trained_models_dir = 'pre-trained-models'
        self.dataset_dir = 'serialized-dataset'

        self.imagenet_img_width = 224
        self.imagenet_img_height = 224
        self.rgb_no_channels = 3

    def configure_model(self):
        optimizer = self._get_optimizer()
        loss = self._get_loss()
        metrics = self._get_metrics()

        compiled_model = self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return compiled_model

    def train_model(self, no_epochs, no_steps_per_epoch):
        dataset = self._get_train_dataset()
        history = self.model.fit(dataset, epochs=no_epochs, steps_per_epoch=no_steps_per_epoch)

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
        raise NotImplementedError('_get_train_dataset method not implemented')

    def _get_test_dataset(self):
        raise NotImplementedError('_get_test_dataset method not implemented')

    def _get_optimizer(self):
        raise NotImplementedError('_get_optimizer method not implemented.')

    def _get_loss(self):
        raise NotImplementedError('_get_loss method not implemented.')

    def _get_metrics(self):
        raise NotImplementedError('_get_metrics method not implemented.')

    def _get_model_serialization_path(self):
        raise NotImplementedError('_get_saved_model_path method not implemented.')

    def _get_model_history_serialization_path(self):
        raise NotImplementedError('_get_model_history_serialization_path method not implemented.')

    def _get_dataset_path(self):
        raise NotImplementedError('_get_dataset_path method not implemented.')

    def _build_serialization_path(self, dir_name, file_name):
        dir_path = os.path.join(self.working_dir, dir_name)
        file_index = len(os.listdir(dir_path))
        return os.path.join(self.working_dir, dir_name, '{}-{}'.format(file_index, file_name))

    def _transform_image(self, img):
        # convert to integers
        img = tf.image.decode_jpeg(img, channels=self.rgb_no_channels)
        # convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, [self.imagenet_img_width, self.imagenet_img_height])

    def _prepare_single_image(self, sample):
        transformed_img = self._transform_image(sample[0])
        label = sample[4]
        return transformed_img, label
