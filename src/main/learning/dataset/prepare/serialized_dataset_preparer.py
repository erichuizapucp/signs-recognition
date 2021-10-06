from learning.dataset.prepare.base_dataset_preparer import BaseDatasetPreparer
from learning.dataset.tfrecord.tf_record_dataset_reader import TFRecordDatasetReader


class SerializedDatasetPreparer(BaseDatasetPreparer):
    def __init__(self, train_dataset_path, test_dataset_path):
        super().__init__(train_dataset_path, test_dataset_path)

    def _prepare(self, dataset_path, batch_size):
        dataset_reader = TFRecordDatasetReader(self._get_dataset_type(), dataset_path)
        dataset = dataset_reader.read()
        if batch_size:
            dataset = dataset.batch(batch_size)

        return dataset

    def _prepare_sample(self, feature, label):
        pass

    def _prepare_sample2(self, feature1, feature2, label):
        pass

    def _prepare_sample3(self, feature):
        raise NotImplementedError('_prepare_sample3 method not implemented.')

    def _get_dataset_type(self):
        raise NotImplementedError('_get_dataset_type method not implemented.')

    def transform_feature_for_predict(self):
        raise NotImplementedError('transform_feature_for_prediction method not implemented.')


