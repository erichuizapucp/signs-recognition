import tensorflow as tf
from base_model import BaseModel


class RGBAwareModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
