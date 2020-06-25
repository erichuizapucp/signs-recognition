import logging

from tensorflow.keras.models import Model, Sequential
from learning.model.nsdm_builder import NSDMModelBuilder
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Concatenate
from learning.common.model_type import NSDMV2


class NSDMV2ModelBuilder(NSDMModelBuilder):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

    def build(self, **kwargs) -> Model:
        opticalflow_model: Model = kwargs['OpticalflowModel']
        rgb_model: Model = kwargs['RGBModel']

        opticalflow_model.trainable = False
        rgb_model.trainable = False

        opticalflow_input = Input(shape=self.opticalflow_input_shape, name='opticalflow_inputs')
        rgb_input = Input(shape=self.rgb_input_shape, name='rgb_inputs')

        opticalflow_model = Sequential(opticalflow_model.layers[:-1])
        rgb_model = Sequential(rgb_model.layers[:-1])

        opticalflow_target = opticalflow_model(opticalflow_input)
        rgb_target = rgb_model(rgb_input)

        concate = Concatenate()([opticalflow_target, rgb_target])
        dense_output = Dense(64, activation='relu', name='dense_layer1')(concate)
        outputs = Dense(self.no_classes, activation='softmax', name='nsdm_output')(dense_output)

        # ensemble model
        nsdm_model = Model(inputs=[opticalflow_input, rgb_input], outputs=outputs, name='nsdm_model')

        return nsdm_model

    def get_model_type(self):
        return NSDMV2
