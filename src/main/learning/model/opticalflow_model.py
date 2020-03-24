import logging
import tensorflow as tf
import os

from model.base_model import BaseModel

from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import Input


class OpticalFlowModel(BaseModel):
    MODEL_NAME = 'opticalflow'

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

    def get_model(self, **kwargs) -> Model:
        img_width = kwargs['ImageWidth']
        img_height = kwargs['ImageHeight']
        no_channels = kwargs['NoChannels']

        no_classes = len(self.classes)

        # use a ResNet152 pre trained model as a base model, we are not including the top layer for maximizing the
        # learned features
        backbone_model: Model = ResNet152V2(include_top=False, weights='imagenet')
        # Freeze pre-trained base model weights
        backbone_model.trainable = False

        # Opticalflow model definition
        inputs = Input(shape=(img_width, img_height, no_channels), name='inputs')
        x = backbone_model(inputs)
        x = GlobalAveragePooling2D(name='OpticalflowGlobalAvgPooling')(x)
        outputs = Dense(no_classes, activation='softmax', name='OpticalflowClassifier')(x)

        # Opticalflow model assembling
        model = Model(inputs=inputs, outputs=outputs, name='OpticalflowModel')

        return model

    def get_image_label(self, file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        return parts[-2] == self.classes
