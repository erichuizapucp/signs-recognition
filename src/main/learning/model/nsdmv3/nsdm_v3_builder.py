import logging
import tensorflow as tf

from learning.model import models
from learning.common.model_type import NSDMV3
from learning.model.legacy.signs_detection_base_model import SignsDetectionBaseModelBuilder


class NSDMV3ModelBuilder(SignsDetectionBaseModelBuilder):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

    def build(self, **kwargs) -> tf.keras.models.Model:
        return self.build_model(**kwargs)

    def build2(self, **kwargs) -> tf.keras.models.Model:
        return self.build_model(load_weights=False, **kwargs)

    def get_model_type(self):
        return NSDMV3

    def build_model(self, **kwargs):
        lstm_cells = kwargs['lstm_cells']
        load_weights = kwargs['load_weights']
        swav_features_weights_path = kwargs['swav_features_weights_path']

        no_dense_layer1_neurons = kwargs['no_dense_layer1_neurons']  # 256
        no_dense_layer2_neurons = kwargs['no_dense_layer2_neurons']  # 128
        no_dense_layer3_neurons = kwargs['no_dense_layer3_neurons']  # 64

        swav_features_model = models.get_swav_features_model(lstm_cells=lstm_cells)
        if load_weights:
            swav_features_model.load_weights(swav_features_weights_path)
            self.logger.debug('weights loaded from: ' + swav_features_weights_path)
        swav_features_model.trainable = True

        model = tf.keras.Sequential()
        model.add(swav_features_model)
        model.add(tf.keras.layers.Dense(no_dense_layer1_neurons, activation='relu'))
        model.add(tf.keras.layers.Dense(no_dense_layer2_neurons, activation='relu'))
        model.add(tf.keras.layers.Dense(no_dense_layer3_neurons, activation='relu'))
        model.add(tf.keras.layers.Dense(self.no_classes, activation='softmax'))

        return model

    def build3(self, **kwargs) -> tf.keras.models.Model:
        raise NotImplementedError('build3 method not implemented.')
