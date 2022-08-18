import logging
import tensorflow as tf

from learning.model.legacy.signs_detection_base_model import SignsDetectionBaseModelBuilder
from learning.common.model_type import RGB


class RGBRecurrentModelBuilder(SignsDetectionBaseModelBuilder):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # feature dim calculated from 224x224x3
        self.feature_dim = self.frames_seq_img_width * self.frames_seq_img_height * self.frames_seq_no_channels
        self.no_lstm_units = 64
        self.no_dense_neurons_1 = 64

    def build(self, **kwargs) -> tf.keras.models.Model:
        # (no_steps, 224x224x3), no_steps is None because it will be determined at runtime by Keras
        input_shape = (None, self.feature_dim)
        inputs = tf.keras.Input(shape=input_shape, name='rgb_inputs')
        x = tf.keras.layers.Masking(name='masking')(inputs)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.no_lstm_units), name='rgb_bidirectional')(x)
        x = tf.keras.layers.Dense(self.no_dense_neurons_1, activation='relu', name='rgb_dense_layer1')(x)
        output = tf.keras.layers.Dense(self.no_classes, activation='softmax', name='rgb_classifier')(x)

        return tf.keras.models.Model(inputs=inputs, outputs=output, name='rgb_model')

    def get_model_type(self):
        return RGB

    def build2(self, **kwargs) -> tf.keras.models.Model:
        raise NotImplementedError('build2 method not implemented.')

    def build3(self, **kwargs) -> tf.keras.models.Model:
        raise NotImplementedError('build3 method not implemented.')
