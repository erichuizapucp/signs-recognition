import tensorflow as tf

from learning.model.base_model_builder import BaseModelBuilder
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Activation
from learning.common.model_utility import SWAV


class SwAVModelBuilder(BaseModelBuilder):
    def __init__(self):
        super().__init__()

        self.no_projection_1_neurons = 1024
        self.no_projection_2_neurons = 96
        self.prototype_vector_dim = 15

    # Features detection (Embeddings)
    def build(self, **kwargs) -> Model:
        input_shape = (None, None, 3)

        backbone_model: Model = ResNet50(include_top=False, weights=None, input_shape=(None, None, 3))
        backbone_model.trainable = True

        inputs = Input(shape=input_shape)
        x = backbone_model(inputs, training=True)


        # # (no_steps, 224x224x3), no_steps is None because it will be determined at runtime by Keras
        # input_shape = (None, self.feature_dim)
        # inputs = Input(shape=input_shape, name='rgb_inputs')
        # x = Masking(name='masking')(inputs)
        # x = Bidirectional(LSTM(self.no_lstm_units), name='rgb_bidirectional')(x)
        # x = Dense(self.no_dense_neurons_1, activation='relu', name='rgb_dense_layer1')(x)
        # output = Dense(self.no_classes, activation='softmax', name='rgb_classifier')(x)
        #
        # return Model(inputs=inputs, outputs=output, name='rgb_model')

        x = GlobalAveragePooling2D()(x)

        model = Model(inputs, x)
        return model

    # Prototype vector and projections
    def build2(self, **kwargs) -> Model:
        input_shape = (2048,)

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
