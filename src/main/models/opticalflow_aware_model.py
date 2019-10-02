import tensorflow as tf
from base_model import BaseModel


class OpticalFlowModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
