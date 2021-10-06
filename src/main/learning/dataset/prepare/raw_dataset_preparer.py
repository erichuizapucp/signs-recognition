import tensorflow as tf

from learning.dataset.prepare.base_dataset_preparer import BaseDatasetPreparer
from pathlib import Path


class RawDatasetPreparer(BaseDatasetPreparer):
    def __init__(self, train_dataset_path, test_dataset_path):
        super().__init__(train_dataset_path, test_dataset_path)

    def _prepare(self, dataset_path, batch_size):
        raw_file_list = self._get_raw_file_list(dataset_path)
        gen_out_signature = tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.int32)
        dataset: tf.data.Dataset = tf.data.Dataset.from_generator(self._data_generator,
                                                                  args=[raw_file_list],
                                                                  output_signature=gen_out_signature)
        dataset = dataset.map(self._prepare_sample3, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        # dataset = dataset.batch(batch_size)

        return dataset

    def _get_raw_file_list(self, dataset_path):
        file_types = self._get_raw_file_types()
        raw_file_list = []

        for file_type in file_types:
            raw_file_list.extend([str(file_path) for file_path in Path(dataset_path).rglob(file_type)])

        return raw_file_list

    def _prepare_sample(self, feature, label):
        raise NotImplementedError('_prepare_sample method not implemented.')

    def _prepare_sample2(self, feature1, feature2, label):
        raise NotImplementedError('_prepare_sample2 method not implemented.')

    def _prepare_sample3(self, feature):
        raise NotImplementedError('_prepare_sample3 method not implemented.')

    def _get_raw_file_types(self):
        raise NotImplementedError('_get_raw_file_types method not implemented.')

    def _data_generator(self, list_video_path):
        raise NotImplementedError('_data_generator method not implemented.')
