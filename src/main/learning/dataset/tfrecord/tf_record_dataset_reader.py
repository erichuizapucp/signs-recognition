import logging

from learning.dataset.tfrecord.tf_record_utility import TFRecordUtility
from learning.common.dataset_type import COMBINED, OPTICAL_FLOW, RGB, SWAV


class TFRecordDatasetReader:
    def __init__(self, dataset_type, dataset_path, batch_size):
        self.logger = logging.getLogger(__name__)

        self.dataset_type = dataset_type
        self.dataset_path = dataset_path
        self.batch_size = batch_size

        self.tf_record_util = TFRecordUtility()

    def read(self):
        read_tfrecord_operations = {
            COMBINED:
                lambda: self.tf_record_util.deserialize_dataset(self.dataset_path,
                                                                self.tf_record_util.parse_combined_dict_sample,
                                                                self.batch_size),
            OPTICAL_FLOW:
                lambda: self.tf_record_util.deserialize_dataset(self.dataset_path,
                                                                self.tf_record_util.parse_opticalflow_dict_sample,
                                                                self.batch_size),
            RGB:
                lambda: self.tf_record_util.deserialize_dataset(self.dataset_path,
                                                                self.tf_record_util.parse_rgb_dict_sample,
                                                                self.batch_size),
            SWAV:
                lambda: self.tf_record_util.deserialize_dataset(self.dataset_path,
                                                                self.tf_record_util.parse_swav_dict_sample,
                                                                self.batch_size),
        }

        if self.dataset_type in read_tfrecord_operations:
            self.logger.debug('%s dataset read was selected', self.dataset_type)
            dataset = read_tfrecord_operations[self.dataset_type]()
        else:
            raise ValueError('Unrecognized operation "{}"'.format(self.dataset_type))

        return dataset
