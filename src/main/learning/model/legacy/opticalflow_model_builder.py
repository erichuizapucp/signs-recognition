import logging
import tensorflow as tf

from learning.model.legacy.signs_detection_base_model import SignsDetectionBaseModelBuilder
from learning.common.model_utility import OPTICAL_FLOW


class OpticalFlowModelBuilder(SignsDetectionBaseModelBuilder):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

    def build(self, **kwargs) -> tf.keras.models.Model:
        # input shape 224x224x3
        input_shape = (self.imagenet_img_width, self.imagenet_img_height, self.imagenet_rgb_no_channels)

        # use a ResNet152 pre trained model as a base model, we are not including the top layer for maximizing the
        # learned features
        backbone_model: tf.keras.models.Model = tf.keras.applicatinos.resnet_v2.ResNet152V2(include_top=False,
                                                                                            weights='imagenet')
        # Freeze pre-trained base model weights
        backbone_model.trainable = False

        # Opticalflow model definition
        inputs = tf.keras.Input(shape=input_shape, name='opticalflow_inputs')
        x = backbone_model(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D(name='opticalflow_global_avg_pooling')(x)
        outputs = tf.keras.layers.Dense(self.no_classes, activation='softmax', name='opticalflow_classifier')(x)

        # Opticalflow model assembling
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name='opticalflow_model')

        return model

    def get_model_type(self):
        return OPTICAL_FLOW

    def build2(self, **kwargs) -> tf.keras.models.Model:
        raise NotImplementedError('build2 method not implemented.')

    def build3(self, **kwargs) -> tf.keras.models.Model:
        raise NotImplementedError('build3 method not implemented.')
