import logging
import os
import pathlib
import tensorflow as tf

from tensorflow.keras.models import Model


class BaseModel:
    DEFAULT_NO_CLASSES = 10
    DEFAULT_LEARNING_RATE = 0.001
    DEFAULT_NO_EPOCHS = 50

    SAVED_MODEL_FOLDER_NAME = 'saved-models/'

    def __init__(self, working_folder, **kwargs):
        self.working_folder = working_folder

        self.learning_rate = kwargs['LearningRate'] or self.DEFAULT_LEARNING_RATE
        self.no_classes = kwargs['NoClasses'] or self.DEFAULT_NO_CLASSES
        self.no_epochs = kwargs['NoEpochs'] or self.DEFAULT_NO_EPOCHS

        self.model: Model = Model()  # using a default model if not assigned
        self.model_history = None
        self.pre_trained_model: Model

        self.logger = logging.getLogger(__name__)

    def get_model(self):
        pass

    def show_sample(self):
        pass

    def train(self):
        pass

    def evaluate(self):
        pass

    def predict(self):
        pass

    def get_class_names(self):
        pass

    def load_dataset(self):
        pass

    def __save_model(self):
        saved_model_path = os.path.join(self.working_folder, self.SAVED_MODEL_FOLDER_NAME)
        tf.saved_model.save(self.model, saved_model_path)
        self.logger.debug('model saved to %s', saved_model_path)
