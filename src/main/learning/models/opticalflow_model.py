import pathlib
import tensorflow as tf
from models.base_model import BaseModel


class OpticalFlowModel(BaseModel):
    def __init__(self, dataset_name, **kwargs):
        super().__init__(dataset_name, **kwargs),

    def configure(self):
        print()

    def get_class_names(self):
        pass
        # self.data_folder.glob("*").

    def configure_model(self):
        self._model = tf.keras.Model()
        print()

    def train_network(self):
        print('OpticalFlow network train')
