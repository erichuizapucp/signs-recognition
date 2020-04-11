import logging

from abc import abstractmethod
from tensorflow.keras.models import Model
from learning.common.labels import SIGNS_CLASSES
from learning.common.imagenet_config import IMAGENET_CONFIG


class BaseModelBuilder:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.classes = SIGNS_CLASSES
        self.no_classes = len(self.classes)

        self.imagenet_img_width = IMAGENET_CONFIG['imagenet_img_width']
        self.imagenet_img_height = IMAGENET_CONFIG['imagenet_img_height']
        self.rgb_no_channels = IMAGENET_CONFIG['rgb_no_channels']

    @abstractmethod
    def build(self) -> Model:
        raise NotImplementedError('get_model method not implemented.')
