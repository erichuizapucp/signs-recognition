import logging
import tensorflow as tf
import os

from models.base_model import BaseModel
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import Input
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Recall, AUC, Precision


class OpticalFlowModel(BaseModel):
    DEFAULT_NO_CHANNELS = 3  # Color images
    DEFAULT_IMG_WIDTH = 224  # Imagenet default image width
    DEFAULT_IMG_HEIGHT = 224  # Imagenet default image height

    SAVED_MODEL_FOLDER_NAME = 'saved-models/opticalflow/'

    def __init__(self, working_folder, **kwargs):
        super(OpticalFlowModel, self).__init__(working_folder, **kwargs)

        self.no_channels = kwargs['NoChannels'] or self.DEFAULT_NO_CHANNELS
        self.img_width = kwargs['ImageWidth'] or self.DEFAULT_IMG_WIDTH
        self.img_height = kwargs['ImageHeight'] or self.DEFAULT_IMG_HEIGHT

        # use a ResNet152 pre trained model as a base model, we are not including the top layer for maximizing the
        # learned features
        self.pre_trained_model: Model = ResNet152V2(include_top=False, weights='imagenet')

        self.logger = logging.getLogger(__name__)

    def load_dataset(self):
        print()

    def get_model(self):
        # Freeze pre-trained base model weights
        self.pre_trained_model.trainable = False

        # Opticalflow model definition
        inputs = Input(shape=(self.img_width, self.img_height, self.no_channels), name='inputs')
        x = self.pre_trained_model.trainable(inputs)
        x = GlobalAveragePooling2D(name='global avg pooling')(x)
        outputs = Dense(self.no_classes, activation=sigmoid(), name='classifier')(x)

        # Opticalflow model assembling
        self.model = Model(inputs=inputs, outputs=outputs, name='opticalflow model')
        self.logger.debug('Opticalflow model summary: \n %s', self.model.summary())

        # Opticalflow model compilation
        optimizer = Adam(self.learning_rate)
        self.model.compile(optimizer=optimizer, loss=CategoricalCrossentropy(), metrics=[Recall(), AUC(), Precision()])

    def train(self):
        self.logger.debug('Opticalflow model train started with the following parameters')
        self.model_history = self.model.fit()

        # save trained model and weights to the file system for future use
        self.__save_model()

    def evaluate(self):
        self.logger.debug('Opticalflow model evaluation started')
        self.model.evaluate()

    def predict(self):
        self.logger.debug('Opticalflow model predict started')
        self.model.predict()

