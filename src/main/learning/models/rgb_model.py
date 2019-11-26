import logging
import tensorflow as tf

from models.base_model import BaseModel
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Recall, AUC, Precision


class RGBModel(BaseModel):
    SAVED_MODEL_FOLDER_NAME = 'saved-models/rgb/'

    def __init__(self, working_folder, dataset_root_path):
        super(RGBModel, self).__init__(working_folder, dataset_root_path)
        self.logger = logging.getLogger(__name__)

    def get_dataset(self, dataset_path, **kwargs) -> tf.data.Dataset:
        pass

    def get_model(self, **kwargs) -> Model:
        img_width = kwargs['ImageWidth']
        img_height = kwargs['ImageHeight']
        no_channels = kwargs['NoChannels']
        learning_rate = kwargs['LearningRate']

        model = Model()

        optimizer = Adam(learning_rate)
        model.compile(optimizer=optimizer, loss=CategoricalCrossentropy(),
                      metrics=[Recall(), AUC(curve='PR'), Precision()])

        return model
