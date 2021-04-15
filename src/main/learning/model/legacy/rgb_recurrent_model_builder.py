import logging

from legacy.base_model_builder import BaseModelBuilder
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Masking, Bidirectional, Dense, LSTM
from learning.common.model_type import RGB


class RGBRecurrentModelBuilder(BaseModelBuilder):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # feature dim calculated from 224x224x3
        self.feature_dim = self.frames_seq_img_width * self.frames_seq_img_height * self.frames_seq_no_channels
        self.no_lstm_units = 64
        self.no_dense_neurons_1 = 64

    def build(self, **kwargs) -> Model:
        # (no_steps, 224x224x3), no_steps is None because it will be determined at runtime by Keras
        input_shape = (None, self.feature_dim)
        inputs = Input(shape=input_shape, name='rgb_inputs')
        x = Masking(name='masking')(inputs)
        x = Bidirectional(LSTM(self.no_lstm_units), name='rgb_bidirectional')(x)
        x = Dense(self.no_dense_neurons_1, activation='relu', name='rgb_dense_layer1')(x)
        output = Dense(self.no_classes, activation='softmax', name='rgb_classifier')(x)

        return Model(inputs=inputs, outputs=output, name='rgb_model')

    def get_model_type(self):
        return RGB
