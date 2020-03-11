import tensorflow as tf
import os
import logging

from learning.dataset.tf_record_utility import TFRecordUtility
from learning.common.dataset_type import OPTICAL_FLOW, RGB
from learning.common.labels import SIGN_CLASSES
from pathlib import Path


class TFRecordDatasetCreator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.tf_record_util = TFRecordUtility()

    def get_raw_dataset(self, raw_dataset_path, dataset_type, shuffle_buffer_size):
        build_dataset_operations = {
            OPTICAL_FLOW: lambda: self.__build_raw_dataset(raw_dataset_path, shuffle_buffer_size,
                                                           self.__get_raw_opticalflow_list,
                                                           self.__tf_get_opticalflow_sample),

            RGB: lambda: self.__build_raw_dataset(raw_dataset_path, shuffle_buffer_size,
                                                  self.__get_raw_rgb_list,
                                                  self.__tf_get_rgb_sample),
        }

        if dataset_type in build_dataset_operations:
            self.logger.debug('An %s dataset generation was selected', dataset_type)
            dataset = build_dataset_operations[dataset_type]()
        else:
            raise ValueError('Unrecognized operation "{}"'.format(dataset_type))

        return dataset

    def create_dataset(self, raw_dataset, dataset_type, output_dir_path, output_prefix, output_max_size):
        create_tfrecord_operations = {
            OPTICAL_FLOW:
                lambda: self.tf_record_util.serialize_dataset(raw_dataset, output_dir_path, output_prefix,
                                                              output_max_size,
                                                              self.tf_record_util.serialize_opticalflow_sample),
            RGB:
                lambda: self.tf_record_util.serialize_dataset(raw_dataset, output_dir_path, output_prefix,
                                                              output_max_size,
                                                              self.tf_record_util.serialize_rgb_sample),
        }

        if dataset_type in create_tfrecord_operations:
            self.logger.debug('An %s TFRecord dataset generation started', dataset_type)
            create_tfrecord_operations[dataset_type]()
        else:
            raise ValueError('Unrecognized operation "{}"'.format(dataset_type))

    def __py_get_opticalflow_sample(self, file_path):
        img_raw = tf.io.read_file(file_path)
        label = self.__get_label(file_path)
        return img_raw, label

    def __py_get_rgb_sample(self, folder_path):
        pattern = tf.strings.join([folder_path, tf.constant('*.jpg')], separator='/')
        sample_files_paths = tf.io.matching_files(pattern)

        rgb_frames = []
        for sample_file_path in sample_files_paths:
            raw_frame = tf.io.read_file(sample_file_path)
            rgb_frames.extend([raw_frame])

        label = self.__get_label(folder_path)
        return rgb_frames, label

    @staticmethod
    def __build_raw_dataset(dataset_path, shuffle_buffer_size, raw_sample_list_func, map_func):
        raw_samples_list = raw_sample_list_func(dataset_path)
        dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(raw_samples_list)
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.map(map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset

    @staticmethod
    def __get_raw_opticalflow_list(dataset_path):
        return [str(file_path) for file_path in Path(dataset_path).rglob('*.jpg')]

    @staticmethod
    def __get_raw_rgb_list(dataset_path):
        return [str(dir_path) for dir_path in Path(dataset_path).rglob('*/*') if dir_path.is_dir()]

    @staticmethod
    def __get_label(file_path):
        sign_name = tf.strings.split(file_path, os.path.sep)[-2]
        decoded_sign_name = sign_name.numpy().decode('UTF-8')
        return SIGN_CLASSES.index(decoded_sign_name)

    def __tf_get_opticalflow_sample(self, file_path):
        return tf.py_function(self.__py_get_opticalflow_sample, [file_path], (tf.string, tf.int64))

    def __tf_get_rgb_sample(self, sample_folder_path):
        return tf.py_function(self.__py_get_rgb_sample, [sample_folder_path], (tf.string, tf.int64))
