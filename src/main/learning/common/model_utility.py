import os
from learning.common.model_type import *


class ModelUtility:
    def __init__(self):
        self.working_dir = os.getenv('WORK_DIR', './')
        self.pre_trained_models_dir = 'pre-trained-models'

        self.model_file_name = {
            OPTICAL_FLOW: 'opticalflow.h5',
            RGB: 'rgb.h5',
            NSDM: 'nsdm.h5',
        }

        self.model_history_file_name = {
            OPTICAL_FLOW: 'opticalflow_history.npy',
            RGB: 'rgb_history.npy',
            NSDM: 'nsdm.npy',
        }

    def get_model_serialization_path(self, model_type):
        return self.build_serialization_path(self.pre_trained_models_dir,
                                             self.__get_pre_trained_model_filename(model_type), model_type)

    def get_model_history_serialization_path(self, model_type):
        return self.build_serialization_path(self.pre_trained_models_dir,
                                             self.__get_training_history_filename(model_type), model_type)

    def build_serialization_path(self, dir_name, file_name, model_type):
        dir_path = os.path.join(self.working_dir, dir_name, model_type)
        os.makedirs(dir_path, exist_ok=True)
        return os.path.join(dir_path, file_name)

    def __get_pre_trained_model_filename(self, model_type):
        return self.model_file_name[model_type]

    def __get_training_history_filename(self, model_type):
        return self.model_history_file_name[model_type]
