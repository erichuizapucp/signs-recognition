import logging
import tensorflow as tf

from abc import abstractmethod
from learning.common.model_utility import ModelUtility


class BaseModelBuilder:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model_utility = ModelUtility()

    @abstractmethod
    def build(self, **kwargs) -> tf.keras.models.Model:
        raise NotImplementedError('build method not implemented.')

    @abstractmethod
    def build2(self, **kwargs) -> tf.keras.models.Model:
        raise NotImplementedError('build2 method not implemented.')

    @abstractmethod
    def build3(self, **kwargs) -> tf.keras.models.Model:
        raise NotImplementedError('build3 method not implemented.')

    @abstractmethod
    def get_model_type(self):
        raise NotImplementedError('get_model_type method not implemented.')

    def load_saved_model(self) -> tf.keras.models.Model:
        return tf.keras.models.load_model(self.model_utility.get_model_serialization_path(self.get_model_type()))
