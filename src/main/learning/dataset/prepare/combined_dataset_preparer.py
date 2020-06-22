import logging
import tensorflow as tf

from learning.dataset.prepare.base_dataset_preparer import BaseDatasetPreparer
from learning.common.dataset_type import COMBINED


class CombinedDatasetPreparer(BaseDatasetPreparer):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

    def prepare_train_dataset(self):
        dataset = super().prepare_train_dataset()
        # apply transformations
        dataset = dataset.map(self._prepare_sample2, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.padded_batch(self.dataset_batch_size,
                                       padded_shapes=([None, None, None], [None, None], [None]))
        return dataset

    # TODO: implement prepare_test_dataset
    def prepare_test_dataset(self):
        return self.prepare_train_dataset()

    def transform_feature_for_predict(self, **kwargs):
        opticalflow_feature = kwargs['OpticalflowFeature']
        rgb_feature = kwargs['RGBFeature']

        # add one dimension that is required by models e.g. 224x224x3 to 1x224x224x3
        transformed_img = tf.expand_dims(self._transform_decoded_image(opticalflow_feature), axis=0)

        # add one dimension that is required by models e.g. 1x14x150528
        transformed_frames_seq = tf.expand_dims(self._py_transform_decoded_frame_seq(rgb_feature), axis=0)

        return transformed_img, transformed_frames_seq

    def _prepare_sample2(self, opticalflow_feature, rgb_feature, label):
        transformed_img = self._transform_image(opticalflow_feature)
        transformed_frames_seq = tf.py_function(self._py_transform_frame_seq, [rgb_feature], tf.float32)

        return transformed_img, transformed_frames_seq, label

    def _get_dataset_type(self):
        return COMBINED

    def _prepare_sample(self, feature, sample):
        raise NotImplementedError('This method is not supported for the Combined dataset')
