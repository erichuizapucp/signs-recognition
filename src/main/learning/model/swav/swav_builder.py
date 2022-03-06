import tensorflow as tf

from learning.model.base_model_builder import BaseModelBuilder
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
    def build(self, **kwargs) -> tf.keras.models.Model:
        backbone_model: tf.keras.models.Model = \
            tf.keras.applications.resnet50.ResNet50(include_top=False, weights=None,
                                                    input_shape=(None, None, self.num_channels))
        backbone_model.trainable = True

        features_input_shape = (None, None, self.num_channels)
        features_inputs = tf.keras.Input(shape=features_input_shape)
        features_x = backbone_model(features_inputs, training=True)
        # TODO: evaluate if using Flatten instead of GlobalAveragePooling2D
        features_x = tf.keras.layers.GlobalAveragePooling2D()(features_x)
        features_model = tf.keras.models.Model(features_inputs, features_x)

        seq_input_shape = (None, None, None, self.num_channels)
        seq_inputs = tf.keras.Input(shape=seq_input_shape)
        seq_x = tf.keras.layers.TimeDistributed(features_model)(seq_inputs)
        seq_x = tf.keras.layers.Masking()(seq_x)
        seq_x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_cells))(seq_x)

        model = tf.keras.models.Model(seq_inputs, seq_x)

        return model

    # Prototype vector and projections
    def build2(self, **kwargs) -> tf.keras.models.Model:
        input_shape = (self.embedding_size,)

        inputs = tf.keras.Input(shape=input_shape)
        projection_1 = tf.keras.layers.Dense(self.no_projection_1_neurons)(inputs)
        projection_1 = tf.keras.layers.BatchNormalization()(projection_1)
        projection_1 = tf.keras.layers.Activation('relu')(projection_1)

        projection_2 = tf.keras.layers.Dense(self.no_projection_2_neurons)(projection_1)
        # L2 Norm (Euclidean Distance) -> Dot Product (Projection x Projection Transpose)
        projection_2_normalize = tf.math.l2_normalize(projection_2, axis=1, name='projection')

        prototype = tf.keras.layers.Dense(self.prototype_vector_dim, use_bias=False,
                                          name='prototype')(projection_2_normalize)
        model = tf.keras.models.Model(inputs=inputs, outputs=[projection_2_normalize, prototype])

        return model

    def get_model_type(self):
        return SWAV

    def build3(self, **kwargs) -> tf.keras.models.Model:
        raise NotImplementedError('build3 method not implemented.')
