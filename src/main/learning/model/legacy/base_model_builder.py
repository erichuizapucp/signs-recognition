import logging

from abc import abstractmethod
from tensorflow.keras.models import Model, load_model
from learning.common.labels import SIGNS_CLASSES
from learning.common.common_config import IMAGENET_CONFIG, FRAMES_SEQ_CONFIG
from learning.common.model_utility import ModelUtility


class BaseModelBuilder:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.classes = SIGNS_CLASSES
        self.no_classes = len(self.classes)

        self.imagenet_img_width = IMAGENET_CONFIG['img_width']
        self.imagenet_img_height = IMAGENET_CONFIG['img_height']
        self.imagenet_rgb_no_channels = IMAGENET_CONFIG['rgb_no_channels']

        self.frames_seq_img_width = FRAMES_SEQ_CONFIG['img_width']
        self.frames_seq_img_height = FRAMES_SEQ_CONFIG['img_height']
        self.frames_seq_no_channels = FRAMES_SEQ_CONFIG['rgb_no_channels']

        self.model_utility = ModelUtility()

    @abstractmethod
    def build(self, **kwargs) -> Model:
        raise NotImplementedError('build method not implemented.')

    def get_model_type(self):
        raise NotImplementedError('get_model_type method not implemented.')

    def load_saved_model(self) -> Model:
        return load_model(self.model_utility.get_model_serialization_path(self.get_model_type()))
