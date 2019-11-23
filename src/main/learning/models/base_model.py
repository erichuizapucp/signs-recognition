import logging
import os
import tensorflow as tf

from tensorflow.keras.models import Model


class BaseModel:
    DEFAULT_NO_CHANNELS = 3  # Color images
    DEFAULT_IMG_WIDTH = 224  # Imagenet default image width
    DEFAULT_IMG_HEIGHT = 224  # Imagenet default image height

    SAVED_MODEL_FOLDER_NAME = 'saved-models/'

    def __init__(self, working_folder, dataset_path, **kwargs):
        self.logger = logging.getLogger(__name__)

        self.working_folder = working_folder
        self.dataset_path = dataset_path

        self.learning_rate = kwargs['LearningRate']
        self.no_epochs = kwargs['NoEpochs']
        self.batch_size = kwargs['BatchSize']

        self.no_channels = kwargs.get('NoChannels') or self.DEFAULT_NO_CHANNELS
        self.img_width = kwargs.get('ImageWidth') or self.DEFAULT_IMG_WIDTH
        self.img_height = kwargs.get('ImageHeight') or self.DEFAULT_IMG_HEIGHT

        if not os.path.exists(self.dataset_path):
            self.logger.error('The provided dataset path %s does not exist.', self.dataset_path)
            raise FileNotFoundError()

        self.classes = self.__get_labels()
        self.model: Model = Model()

    def get_model(self) -> Model:
        pass

    def train(self):
        pass

    def evaluate(self):
        pass

    def predict(self):
        pass

    def get_dataset(self) -> tf.data.Dataset:
        pass

    def __get_labels(self) -> list:
        path = self.dataset_path
        return [label for label in os.listdir(path) if os.path.isdir(os.path.join(path, label))]

    def __save_model(self):
        saved_model_path = os.path.join(self.working_folder, self.SAVED_MODEL_FOLDER_NAME)
        tf.saved_model.save(self.model, saved_model_path)
        self.logger.debug('model saved to %s', saved_model_path)
