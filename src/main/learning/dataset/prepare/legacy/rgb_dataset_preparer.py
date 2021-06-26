import logging
import tensorflow as tf

from learning.dataset.prepare.serialized_dataset_preparer import SerializedDatasetPreparer
from learning.common.dataset_type import RGB


class RGBDatasetPreparer(SerializedDatasetPreparer):
    def __init__(self, train_dataset_path, test_dataset_path):
        super().__init__(train_dataset_path, test_dataset_path)
        self.logger = logging.getLogger(__name__)

    def prepare_train_dataset(self):
        dataset = super().prepare_train_dataset()
        # apply image transformation and data augmentation
        dataset = dataset.map(self._prepare_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.padded_batch(self.dataset_batch_size, padded_shapes=([None, None], [None]))
        return dataset

    # TODO: implement prepare_test_dataset
    def prepare_test_dataset(self):
        return self.prepare_train_dataset()

    def _prepare_sample(self, rgb_sample, label):
        # RNN feature
        transformed_frames_seq = tf.py_function(self._py_transform_frame_seq, [rgb_sample], tf.float32)
        # RNN cells require a 3D input (batch, steps, feature)
        return transformed_frames_seq, label

    def _prepare_sample2(self, feature1, feature2, label):
        raise NotImplementedError('This method is not supported for the RGB dataset')

    def _prepare_sample3(self, feature):
        raise NotImplementedError('This method is not supported for the RGB dataset')

    def _get_dataset_type(self):
        return RGB

    def transform_feature_for_predict(self, **kwargs):
        raise NotImplementedError('This method is not supported for the RGB dataset')
