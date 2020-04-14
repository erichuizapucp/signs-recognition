import logging

from tensorflow.keras.models import Model
from learning.model.base_model_builder import BaseModelBuilder
from tensorflow.keras import Input
from tensorflow.keras.layers import Average
from learning.common.model_type import NSDM


class NSDMModelBuilder(BaseModelBuilder):
    def __init__(self, opticalflow_model: Model, rgb_model: Model):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.opticalflow_model = opticalflow_model
        self.rgb_model = rgb_model

    def build(self) -> Model:
        # input shape 224x224x3
        opticalflow_input_shape = (self.imagenet_img_width, self.imagenet_img_height, self.rgb_no_channels)
        # feature dim calculated from 224 * 224 * 3
        rgb_input_shape = (None, self.imagenet_img_width * self.imagenet_img_height * self.rgb_no_channels)

        opticalflow_input = Input(shape=opticalflow_input_shape, name='opticalflow_inputs')
        rgb_input = Input(shape=rgb_input_shape, name='rgb_inputs')

        opticalflow_target = self.opticalflow_model(opticalflow_input)
        rgb_target = self.rgb_model(rgb_input)

        # average opticalflow and rgb models outputs
        outputs = Average(name='nsdm_output')([opticalflow_target, rgb_target])
        # ensemble model
        nsdm_model = Model(inputs=[opticalflow_input, rgb_input], outputs=outputs, name='nsdm_model')

        return nsdm_model

    def get_model_type(self):
        return NSDM
