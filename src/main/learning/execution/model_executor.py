import tensorflow as tf
import os

from tensorflow.keras.models import Model


class ModelExecutor:
    def __init__(self, model: Model, model_type):
        self.model = model
        self.model_type = model_type

        self.working_dir = os.getenv('WORK_DIR', './')
        self.pre_trained_models_dir = 'pre-trained-models'

    def configure_model(self):
        optimizer = self._get_optimizer()
        loss = self._get_loss()
        metrics = self._get_metrics()

        compiled_model = self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return compiled_model

    def train_model(self, dataset: tf.data.Dataset, no_epochs, no_steps_per_epoch):
        history = self.model.fit(dataset, epochs=no_epochs, steps_per_epoch=no_steps_per_epoch)

        # save trained model and weights to the file system for future use
        saved_model_path = self._get_saved_model_path()
        self.model.save(saved_model_path)

    def evaluate_model(self, dataset: tf.data.Dataset):
        return self.model.evaluate(dataset)

    def predict_with_model(self, dataset: tf.data.Dataset):
        return self.model.predict(dataset)

    def _get_optimizer(self):
        raise NotImplementedError('get_optimizer method not implemented.')

    def _get_loss(self):
        raise NotImplementedError('get_loss method not implemented.')

    def _get_metrics(self):
        raise NotImplementedError('get_metrics method not implemented.')

    def _get_saved_model_path(self):
        raise NotImplementedError('get_saved_model_path method not implemented.')

    def _build_saved_model_path(self, file_name):
        dir_path = os.path.join(self.working_dir, self.pre_trained_models_dir)
        file_index = len(os.listdir(dir_path))
        return os.path.join(self.working_dir, self.pre_trained_models_dir, '{}-{}'.format(file_index, file_name))
