import tensorflow as tf

from learning.model.base_model_builder import BaseModelBuilder
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Activation, Masking, \
    Bidirectional, LSTM, TimeDistributed
from learning.common.model_utility import SWAV


class SwAVModelBuilder(BaseModelBuilder):
    def __init__(self):
        super().__init__()

        self.no_projection_1_neurons = 1024
        self.no_projection_2_neurons = 96
        self.prototype_vector_dim = 15
        self.num_channels = 3
        self.lstm_cells = 2048
        self.embedding_size = 4096

    # Features detection (Embeddings) / CNN-LSTM model
    def build(self, **kwargs) -> Model:
        backbone_model: Model = ResNet50(include_top=False, weights=None, input_shape=(None, None, self.num_channels))
        backbone_model.trainable = True

        features_input_shape = (None, None, self.num_channels)
        features_inputs = Input(shape=features_input_shape)
        features_x = backbone_model(features_inputs, training=True)
        # TODO: evaluate if using Flatten instead of GlobalAveragePooling2D
        features_x = GlobalAveragePooling2D()(features_x)
        features_model = Model(features_inputs, features_x)

        seq_input_shape = (None, None, None, self.num_channels)
        seq_inputs = Input(shape=seq_input_shape)
        seq_x = TimeDistributed(features_model)(seq_inputs)
        seq_x = Masking()(seq_x)
        seq_x = Bidirectional(LSTM(self.lstm_cells))(seq_x)

        model = Model(seq_inputs, seq_x)

        return model

    # Prototype vector and projections
    def build2(self, **kwargs) -> Model:
        input_shape = (self.embedding_size,)

        inputs = Input(shape=input_shape)
        projection_1 = Dense(self.no_projection_1_neurons)(inputs)
        projection_1 = BatchNormalization()(projection_1)
        projection_1 = Activation('relu')(projection_1)

        projection_2 = Dense(self.no_projection_2_neurons)(projection_1)
        # L2 Norm (Euclidean Distance) -> Dot Product (Projection x Projection Transpose)
        projection_2_normalize = tf.math.l2_normalize(projection_2, axis=1, name='projection')

        prototype = Dense(self.prototype_vector_dim, use_bias=False, name='prototype')(projection_2_normalize)
        model = Model(inputs=inputs, outputs=[projection_2_normalize, prototype])

        return model

    def get_model_type(self):
        return SWAV

    def build3(self, **kwargs) -> Model:
        raise NotImplementedError('build3 method not implemented.')
