import logging

from models.base_model import BaseModel


class RGBModel(BaseModel):
    SAVED_MODEL_FOLDER_NAME = 'saved-models/rgb/'

    def __init__(self, working_folder, dataset_path, **kwargs):
        super().__init__(working_folder, dataset_path, **kwargs)
        self.logger = logging.getLogger(__name__)

    def train(self):
        self.logger.debug('Opticalflow model train started with the following parameters')
        self.model_history = self.model.fit()

        # save trained model and weights to the file system for future use
        self.__save_model()

    def evaluate(self):
        self.logger.debug('Opticalflow model evaluation started')
        self.model.evaluate()

    def predict(self):
        self.logger.debug('Opticalflow model predict started')
        self.model.predict()
