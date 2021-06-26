from tensorflow.python.keras import Model
from learning.model.base_model_builder import BaseModelBuilder


class SwAVModelBuilder(BaseModelBuilder):
    def __init__(self):
        super().__init__()

    def build(self, **kwargs) -> Model:
        pass

    def build2(self, **kwargs) -> Model:
        pass

    def build3(self, **kwargs) -> Model:
        pass

    def get_model_type(self):
        pass
