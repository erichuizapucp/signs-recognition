import logging
import tensorflow as tf

from learning.dataset.prepare.serialized_dataset_preparer import SerializedDatasetPreparer
from learning.dataset.tfrecord.tf_record_dataset_reader import TFRecordDatasetReader
from learning.common.dataset_type import SWAV


class SerializedSwAVDatasetPreparer(SerializedDatasetPreparer):
    def __init__(self, train_dataset_path, test_dataset_path):
        super().__init__(train_dataset_path, test_dataset_path)
        self.logger = logging.getLogger(__name__)

    def prepare_dataset(self, dataset_path, batch_size):
        dataset_reader = TFRecordDatasetReader(self.get_dataset_type(), dataset_path, batch_size)
        dataset = dataset_reader.read()
        dataset = dataset.padded_batch(batch_size, drop_remainder=True)

        return dataset

    def prepare_sample3(self, feature):
        pass

    def get_dataset_type(self):
        return SWAV

    def transform_feature_for_predict(self):
        pass


