import tensorflow as tf
import logging

from tensorflow.keras.models import Model
from learning.execution.model_executor import ModelExecutor
from learning.common.dataset_type import RGB


class RGBExecutor(ModelExecutor):
    def __init__(self, model: Model, working_dir):
        super().__init__(model, working_dir)
        self.logger = logging.getLogger(__name__)

        self.pre_trained_model_file = 'rgb.h5'
        self.training_history_file = 'rgb_history.npy'
        self.feature_dim = self.imagenet_img_width * self.imagenet_img_height * self.rgb_no_channels

    def _get_train_dataset(self):
        dataset = super()._get_train_dataset()
        # apply image transformation and data augmentation
        dataset = dataset.map(
            lambda frames_seq, height, width, depth, label: self._prepare_sample(frames_seq, label),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.padded_batch(self.dataset_batch_size, padded_shapes=([None, None], [None]))
        return dataset

    def _prepare_sample(self, sample, label):
        # RNN feature
        transformed_frames_seq = tf.py_function(self.py_transform_frame_seq, [sample], tf.float32)
        # RNN cells require a 3D input (batch, steps, feature)
        return transformed_frames_seq, label

    def py_transform_frame_seq(self, sample):
        transformed_images = []
        for image in sample:
            transformed_img = tf.reshape(self._transform_image(image), [self.feature_dim])
            transformed_images.extend([transformed_img])
        return tf.stack(transformed_images)

    def _get_dataset_type(self):
        return RGB

    def _get_pre_trained_model_filename(self):
        return 'rgb.h5'

    def _get_training_history_filename(self):
        return 'rgb_history.npy'
