import logging
import os
import tensorflow as tf

from base_model import BaseModel
from opticalflow_model import OpticalFlowModel
from rgb_model import RGBModel


class CombinedModel(BaseModel):
    OPTICALFLOW_MODEL_FOLDER_NAME = 'saved-models/opticalflow/'
    RGB_MODEL_FOLDER_NAME = 'saved-models/rgb/'

    def __init__(self, working_folder, dataset_path, **kwargs):
        super(CombinedModel, self).__init__(working_folder, dataset_path, **kwargs)

        self.opticalflow_model_path = os.path.join(self.working_folder, OpticalFlowModel.SAVED_MODEL_FOLDER_NAME)
        self.rgb_model_path = os.path.join(self.working_folder, RGBModel.SAVED_MODEL_FOLDER_NAME)

        self.logger = logging.getLogger(__name__)

    def get_dataset(self):
        pass

    def get_model(self):
        if not os.path.exists(self.opticalflow_model_path):
            self.logger.error('the pre-trained opticalflow model does not exist at %s', self.opticalflow_model_path)

        # load a pre-trained opticalflow model
        opticalflow_model = tf.saved_model.load(self.opticalflow_model_path)

        if not os.path.exists(self.rgb_model_path):
            self.logger.error('the pre-trained rgb recurrent model does not exist at %s', self.rgb_model_path)

        # load a pre-trained rgb recurrent model
        rgb_model = tf.saved_model.load(self.rgb_model_path)

    def train(self):
        self.logger.debug('Opticalflow model train started with the following parameters')
        self.model_history = self.model.fit()

    def evaluate(self):
        self.logger.debug('Opticalflow model evaluation started')
        self.model.evaluate()

    def predict(self):
        self.logger.debug('Opticalflow model predict started')
        self.model.predict()
