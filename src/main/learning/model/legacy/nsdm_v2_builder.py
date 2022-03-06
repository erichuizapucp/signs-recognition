import logging
import tensorflow as tf

from learning.model.legacy.nsdm_builder import NSDMModelBuilder
from learning.common.model_type import NSDMV2


class NSDMV2ModelBuilder(NSDMModelBuilder):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

    def build(self, **kwargs) -> tf.keras.models.Model:
        opticalflow_model: tf.keras.models.Model = kwargs['OpticalflowModel']
        rgb_model: tf.keras.models.Model = kwargs['RGBModel']

        opticalflow_model.trainable = False
        rgb_model.trainable = False

        opticalflow_input = tf.keras.Input(shape=self.opticalflow_input_shape, name='opticalflow_inputs')
        rgb_input = tf.keras.Input(shape=self.rgb_input_shape, name='rgb_inputs')

        opticalflow_model = tf.keras.models.Sequential(opticalflow_model.layers[:-1])
        rgb_model = tf.keras.models.Sequential(rgb_model.layers[:-1])

        opticalflow_target = opticalflow_model(opticalflow_input)
        rgb_target = rgb_model(rgb_input)

        concate = tf.keras.layers.Concatenate()([opticalflow_target, rgb_target])
        dense_output = tf.keras.layers.Dense(64, activation='relu', name='dense_layer1')(concate)
        outputs = tf.keras.layers.Dense(self.no_classes, activation='softmax', name='nsdm_output')(dense_output)

        # ensemble model
        nsdm_model = tf.keras.models.Model(inputs=[opticalflow_input, rgb_input], outputs=outputs, name='nsdm_model')

        return nsdm_model

    def get_model_type(self):
        return NSDMV2

    def build2(self, **kwargs) -> tf.keras.models.Model:
        raise NotImplementedError('build2 method not implemented.')

    def build3(self, **kwargs) -> tf.keras.models.Model:
        raise NotImplementedError('build3 method not implemented.')
