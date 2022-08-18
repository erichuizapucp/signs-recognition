import logging
import tensorflow as tf

from learning.model.legacy.signs_detection_base_model import SignsDetectionBaseModelBuilder
from learning.common.model_type import NSDM


class NSDMModelBuilder(SignsDetectionBaseModelBuilder):
    def __init__(self):
        super().__init__()

        # input shape 224x224x3
        self.opticalflow_input_shape = (self.imagenet_img_width, self.imagenet_img_height, self.imagenet_rgb_no_channels)
        # feature dim calculated from 224 * 224 * 3
        self.rgb_input_shape = (None,
                                self.frames_seq_img_width * self.frames_seq_img_height * self.frames_seq_no_channels)

        self.logger = logging.getLogger(__name__)

    def build(self, **kwargs) -> tf.keras.models.Model:
        opticalflow_model = kwargs['OpticalflowModel']
        rgb_model = kwargs['RGBModel']

        opticalflow_input = tf.keras.Input(shape=self.opticalflow_input_shape, name='opticalflow_inputs')
        rgb_input = tf.keras.Input(shape=self.rgb_input_shape, name='rgb_inputs')

        opticalflow_target = opticalflow_model(opticalflow_input)
        rgb_target = rgb_model(rgb_input)

        # average opticalflow and rgb models outputs
        outputs = tf.keras.layers.Average(name='nsdm_output')([opticalflow_target, rgb_target])
        # ensemble model
        nsdm_model = tf.keras.models.Model(inputs=[opticalflow_input, rgb_input], outputs=outputs, name='nsdm_model')

        return nsdm_model

    def get_model_type(self):
        return NSDM

    def build2(self, **kwargs) -> tf.keras.models.Model:
        raise NotImplementedError('build2 method not implemented.')

    def build3(self, **kwargs) -> tf.keras.models.Model:
        raise NotImplementedError('build3 method not implemented.')
