import tensorflow as tf

from learning.model.base_model_builder import BaseModelBuilder
from learning.common.labels import SIGNS_CLASSES
from learning.common.common_config import IMAGENET_CONFIG, FRAMES_SEQ_CONFIG


class SignsDetectionBaseModelBuilder(BaseModelBuilder):
    def __init__(self):
        super().__init__()

        self.classes = SIGNS_CLASSES
        self.no_classes = len(self.classes)

        self.imagenet_img_width = IMAGENET_CONFIG['img_width']
        self.imagenet_img_height = IMAGENET_CONFIG['img_height']
        self.imagenet_rgb_no_channels = IMAGENET_CONFIG['rgb_no_channels']

        self.frames_seq_img_width = FRAMES_SEQ_CONFIG['img_width']
        self.frames_seq_img_height = FRAMES_SEQ_CONFIG['img_height']
        self.frames_seq_no_channels = FRAMES_SEQ_CONFIG['rgb_no_channels']

    def build(self, **kwargs) -> tf.keras.models.Model:
        raise NotImplementedError('build1 method not implemented.')

    def build2(self, **kwargs) -> tf.keras.models.Model:
        raise NotImplementedError('build2 method not implemented.')

    def build3(self, **kwargs) -> tf.keras.models.Model:
        raise NotImplementedError('build3 method not implemented.')

    def get_model_type(self):
        raise NotImplementedError('get_model_type method not implemented.')
