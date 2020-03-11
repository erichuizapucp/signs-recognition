import tensorflow as tf
import logging
import os

from abc import abstractmethod
from tensorflow.keras.models import Model


class BaseModel:
    SAVED_MODEL_FOLDER_NAME = 'saved-model'
    MODEL_NAME = 'base'
    TRAIN_DATASET_FOLDER_NAME = 'train'
    TEST_DATASET_FOLDER_NAME = 'test'

    def __init__(self, working_folder, dataset_root_path):
        self.logger = logging.getLogger(__name__)

        self.working_folder = working_folder
        self.dataset_root_path = dataset_root_path
        self.classes = self.get_labels()

    def train(self, **kwargs):
        no_epochs = kwargs['NoEpochs']
        no_steps_per_epoch = kwargs['NoStepsPerEpoch']
        batch_size = kwargs['BatchSize']
        shuffle_buffer_size = kwargs['ShuffleBufferSize']
        img_width = kwargs['ImageWidth']
        img_height = kwargs['ImageHeight']
        no_channels = kwargs['NoChannels']
        learning_rate = kwargs['LearningRate']

        train_dataset_path = self.get_dataset_path(self.TRAIN_DATASET_FOLDER_NAME)
        dataset = self.get_dataset(train_dataset_path,
                                   BatchSize=batch_size, ShuffleBufferSize=shuffle_buffer_size,
                                   ImageWidth=img_width, ImageHeight=img_height, NoChannels=no_channels)

        model = self.get_model(LearningRate=learning_rate, ImageWidth=img_width, ImageHeight=img_height,
                               NoChannels=no_channels)
        history = model.fit(dataset, epochs=no_epochs, steps_per_epoch=no_steps_per_epoch)

        # save trained model and weights to the file system for future use
        self.save_model(model)

    def evaluate(self):
        self.logger.debug('Opticalflow model evaluation started')

        test_dataset_path = self.get_dataset_path(self.TEST_DATASET_FOLDER_NAME)
        dataset = self.get_dataset(test_dataset_path)
        model = self.get_saved_model()

        model.evaluate(dataset)

    def predict(self):
        self.logger.debug('Opticalflow model predict started')

        test_dataset_path = self.get_dataset_path(self.TEST_DATASET_FOLDER_NAME)
        dataset = self.get_dataset(test_dataset_path)
        model = self.get_saved_model()

        model.predict(dataset)

    @abstractmethod
    def get_dataset(self, dataset_path, **kwargs) -> tf.data.Dataset:
        pass

    @abstractmethod
    def get_model(self, **kwargs) -> Model:
        pass

    def get_labels(self, train=True) -> list:
        dataset_folder_name = self.TRAIN_DATASET_FOLDER_NAME if train else self.TEST_DATASET_FOLDER_NAME
        path = self.get_dataset_path(dataset_folder_name)
        return [label for label in os.listdir(path) if os.path.isdir(os.path.join(path, label))]

    def get_saved_model(self) -> Model:
        saved_model_path = os.path.join(self.working_folder, self.SAVED_MODEL_FOLDER_NAME, self.MODEL_NAME)
        model = tf.saved_model.load(saved_model_path)
        return model

    def save_model(self, model: Model):
        saved_model_path = os.path.join(self.working_folder, self.SAVED_MODEL_FOLDER_NAME, self.MODEL_NAME)
        tf.saved_model.save(model, saved_model_path)
        self.logger.debug('model saved to %s', saved_model_path)

    def get_dataset_path(self, operation_folder_name):
        dataset_path = os.path.join(self.dataset_root_path, operation_folder_name, self.MODEL_NAME)

        if not os.path.exists(dataset_path):
            self.logger.error('The provided dataset path %s does not exist.', dataset_path)
            raise FileNotFoundError()

        return dataset_path
