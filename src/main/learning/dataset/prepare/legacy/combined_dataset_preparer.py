import logging
import tensorflow as tf

from learning.dataset.prepare.serialized_dataset_preparer import SerializedDatasetPreparer
from learning.common.dataset_type import COMBINED


class CombinedDatasetPreparer(SerializedDatasetPreparer):
    def __init__(self, train_dataset_path, test_dataset_path):
        super().__init__(train_dataset_path, test_dataset_path)
        self.logger = logging.getLogger(__name__)

    def prepare_train_dataset(self, batch_size):
        # don't batch the resulting dataset
        dataset = super().prepare_train_dataset(batch_size=None)

        # apply transformations
        dataset = dataset.map(self._prepare_sample2, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.padded_batch(batch_size,
                                       padded_shapes=([None, None, None], [None, None], [None]))
        return dataset

    # TODO: implement prepare_test_dataset
    def prepare_test_dataset(self, batch_size):
        return self.prepare_train_dataset(batch_size)

    def _prepare_sample(self, feature, sample):
        raise NotImplementedError('This method is not supported for the Combined dataset')

    def _prepare_sample2(self, opticalflow_sample, rgb_sample, label):
        transformed_img = self._transform_image(opticalflow_sample)
        transformed_frames_seq = tf.py_function(self._py_transform_frame_seq, [rgb_sample], tf.float32)

        return transformed_img, transformed_frames_seq, label

    def _prepare_sample3(self, feature):
        raise NotImplementedError('This method is not supported for the Combined dataset')

    def _get_dataset_type(self):
        return COMBINED

    def transform_feature_for_predict(self, **kwargs):
        opticalflow_feature = kwargs['OpticalflowFeature']
        rgb_feature = kwargs['RGBFeature']

        # add one dimension that is required by models e.g. 224x224x3 to 1x224x224x3
        transformed_img = tf.expand_dims(self._transform_decoded_image(opticalflow_feature), axis=0)

        # add one dimension that is required by models e.g. 1x14x150528
        transformed_frames_seq = tf.expand_dims(self._py_transform_decoded_frame_seq(rgb_feature), axis=0)

        return transformed_img, transformed_frames_seq
