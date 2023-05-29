import os
import tensorflow as tf

from object_detection.utils import config_util
from object_detection.builders import model_builder


class ObjectDetectionUtility:
    def __init__(self):
        self.working_dir = os.getenv('WORK_DIR', './')

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
