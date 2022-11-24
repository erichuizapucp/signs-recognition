import os
import tensorflow as tf

from learning.dataset.prepare.base_dataset_preparer import BaseDatasetPreparer
from learning.common import video_utility

from pathlib import Path
from random import shuffle


class RawDatasetPreparer(BaseDatasetPreparer):
    def __init__(self, train_dataset_path, test_dataset_path):
        super().__init__(train_dataset_path, test_dataset_path)

        self.chunks_duration_in_sec = 10

    def prepare_dataset(self, dataset_path, batch_size):
        video_path_list, chunk_start_list, chunk_end_list = self.get_raw_file_list2(dataset_path)
        gen_out_signature = tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.int32)
        dataset: tf.data.Dataset = tf.data.Dataset.from_generator(self.data_generator2,
                                                                  args=[video_path_list, chunk_start_list, chunk_end_list],
                                                                  output_signature=gen_out_signature)
        dataset = dataset.map(self.prepare_sample3, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        dataset = dataset.padded_batch(batch_size, drop_remainder=True)

        return dataset

    def get_raw_file_list(self, dataset_path):
        file_types = self.get_raw_file_types()
        raw_file_list = []

        for file_type in file_types:
            raw_file_list.extend([str(file_path) for file_path in Path(dataset_path).rglob(file_type)])

        shuffle(raw_file_list)

        return raw_file_list

    def get_raw_file_list2(self, dataset_path):
        file_types = self.get_raw_file_types()
        raw_file_list = []

        for file_type in file_types:
            raw_file_list.extend([str(file_path) for file_path in Path(dataset_path).rglob(file_type)])

        chunks_list = []
        for raw_file_path in raw_file_list:
            video_duration = video_utility.get_video_duration(raw_file_path)
            num_chunks = int(video_duration / self.chunks_duration_in_sec) \
                if video_duration / self.chunks_duration_in_sec > 1.0 else 1
            chunk_duration = \
                video_duration if video_duration < self.chunks_duration_in_sec else self.chunks_duration_in_sec

            for chunk_index in range(num_chunks):
                chunk_start = float(chunk_index * chunk_duration)
                if chunk_index == num_chunks - 1:
                    chunk_end = float(video_duration)
                else:
                    chunk_end = float(chunk_start + chunk_duration)

                chunks_list.extend([(raw_file_path, chunk_start, chunk_end)])

        shuffle(chunks_list)

        video_path_list = []
        chunk_start_list = []
        chunk_end_list = []
        for video_path, chunk_start, chunk_end in chunks_list:
            video_path_list.extend([video_path])
            chunk_start_list.extend([chunk_start])
            chunk_end_list.extend([chunk_end])

        return video_path_list, chunk_start_list, chunk_end_list

    def prepare_sample(self, feature, label):
        raise NotImplementedError('_prepare_sample method not implemented.')

    def prepare_sample2(self, feature1, feature2, label):
        raise NotImplementedError('_prepare_sample2 method not implemented.')

    def prepare_sample3(self, feature):
        raise NotImplementedError('_prepare_sample3 method not implemented.')

    def get_raw_file_types(self):
        raise NotImplementedError('_get_raw_file_types method not implemented.')

    def data_generator(self, list_video_path):
        raise NotImplementedError('_data_generator method not implemented.')

    def data_generator2(self, video_path_list, chunk_start_list, chunk_end_list):
        raise NotImplementedError('_data_generator method not implemented.')
