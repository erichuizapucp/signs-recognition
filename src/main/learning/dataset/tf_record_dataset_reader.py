import logging

from learning.dataset.tf_record_utility import TFRecordUtility
from learning.common.dataset_type import OPTICAL_FLOW, RGB


class TFRecordDatasetReader:
    def __init__(self, dataset_type, dataset_path):
        self.logger = logging.getLogger(__name__)

        self.dataset_type = dataset_type
        self.dataset_path = dataset_path

        self.tf_record_util = TFRecordUtility()

    def read(self):
        read_tfrecord_operations = {
            OPTICAL_FLOW:
                lambda: self.tf_record_util.deserialize_dataset(self.dataset_path,
                                                                self.tf_record_util.parse_opticalflow_dict_sample),
            RGB:
                lambda: self.tf_record_util.deserialize_dataset(self.dataset_path,
                                                                self.tf_record_util.parse_rgb_dict_sample),
        }

        if self.dataset_type in read_tfrecord_operations:
            self.logger.debug('An %s dataset read was selected', self.dataset_type)
            dataset = read_tfrecord_operations[self.dataset_type]()
        else:
            raise ValueError('Unrecognized operation "{}"'.format(self.dataset_type))

        return dataset
