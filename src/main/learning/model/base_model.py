import logging

from abc import abstractmethod
from tensorflow.keras.models import Model
from learning.common.labels import SIGNS_CLASSES


class BaseModel:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.classes = SIGNS_CLASSES

    @abstractmethod
    def get_model(self, **kwargs) -> Model:
        raise NotImplementedError('get_model method not implemented.')
