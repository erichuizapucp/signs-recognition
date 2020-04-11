import logging

from model.base_model_builder import BaseModelBuilder
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Masking, Bidirectional, Dense, LSTM, Flatten


class RGBRecurrentModelBuilder(BaseModelBuilder):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # feature dim calculated from 224x224x3
        self.feature_dim = self.imagenet_img_width * self.imagenet_img_height * self.rgb_no_channels
        self.no_lstm_units = 128
        self.no_dense_neurons_1 = 64

    def build(self) -> Model:
        input_shape = (None, self.feature_dim)
        inputs = Input(shape=input_shape, name='inputs')
        x = Masking(name='masking')(inputs)
        x = Bidirectional(LSTM(self.no_lstm_units))(x)
        x = Dense(self.no_dense_neurons_1, activation='relu', name='rgb_dense_layer1')(x)
        output = Dense(self.no_classes, activation='softmax', name='rgb_classifier')(x)

        return Model(inputs=inputs, outputs=output)
