import os
import tensorflow as tf

from learning.common.model_type import *
from object_detection.utils import config_util
from object_detection.builders import model_builder


class ModelUtility:
    def __init__(self):
        self.working_dir = os.getenv('WORK_DIR', './')
        self.pre_trained_models_dir = 'pre-trained-models'

        self.model_file_name = {
            OPTICAL_FLOW: 'opticalflow.h5',
            RGB: 'rgb.h5',
            NSDM: 'nsdm.h5',
            NSDMV2: 'nsdmv2.h5',
        }

        self.model_history_file_name = {
            OPTICAL_FLOW: 'opticalflow_history.npy',
            RGB: 'rgb_history.npy',
            NSDM: 'nsdm.npy',
            NSDMV2: 'nsdmv2.h5',
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

    def get_object_detection_model(self, model_name, checkpoint_prefix):
        pipeline_config_path = os.path.join(self.working_dir, 'models', model_name, 'pipeline.config')
        configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
        model_config = configs['model']

        detection_model = model_builder.build(model_config=model_config, is_training=False)

        # Restore checkpoint
        checkpoint_path = os.path.join(self.working_dir, 'models', model_name, 'checkpoint')
        checkpoint = tf.compat.v2.train.Checkpoint(model=detection_model)
        checkpoint.restore(os.path.join(checkpoint_path, checkpoint_prefix)).expect_partial()

        return detection_model
