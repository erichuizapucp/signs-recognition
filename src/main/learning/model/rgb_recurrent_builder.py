import logging
import tensorflow as tf

from model.base_model_builder import BaseModelBuilder
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Recall, AUC, Precision


class RGBRecurrentModelBuilder(BaseModelBuilder):
    MODEL_NAME = 'rgb'

    def __init__(self):
        super(RGBRecurrentModelBuilder, self).__init__()
        self.logger = logging.getLogger(__name__)

    def get_dataset(self, dataset_path, **kwargs) -> tf.data.Dataset:
        pass

    def build(self, **kwargs) -> Model:
        img_width = kwargs['ImageWidth']
        img_height = kwargs['ImageHeight']
        no_channels = kwargs['NoChannels']
        learning_rate = kwargs['LearningRate']

        model = Model()

        optimizer = Adam(learning_rate)
        model.compile(optimizer=optimizer, loss=CategoricalCrossentropy(),
                      metrics=[Recall(), AUC(curve='PR'), Precision()])

        return model
