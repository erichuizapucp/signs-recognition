import os
import logging
import tensorflow as tf

from pathlib import Path
from learning.common.labels import SIGNS_CLASSES
from learning.common.common_config import IMAGENET_CONFIG
from learning.dataset.tfrecord.tf_record_utility import TFRecordUtility
from learning.common.object_detection_utility import ObjectDetectionUtility
from learning.common.dataset_type import COMBINED, OPTICAL_FLOW, RGB, SWAV, NSDMV3
from learning.dataset.prepare.swav.swav_video_dataset_preparer import SwAVDatasetPreparer


class TFRecordDatasetCreator:
    def __init__(self, dataset_type, raw_files_path):
        self.logger = logging.getLogger(__name__)

        self.dataset_type = dataset_type
        self.raw_files_path = raw_files_path

        self.tf_record_util = TFRecordUtility()

        self.imagenet_img_width = IMAGENET_CONFIG['img_width']
        self.imagenet_img_height = IMAGENET_CONFIG['img_height']
        self.imagenet_rgb_no_channels = IMAGENET_CONFIG['rgb_no_channels']

    def create(self, output_dir_path, output_prefix, output_max_size, **kwargs):
        raw_dataset = self.get_raw_dataset(self.raw_files_path, self.dataset_type, **kwargs)

        create_tfrecord_operations = {
            COMBINED: lambda: self.tf_record_util.serialize_dataset(COMBINED,
                                                                    raw_dataset,
                                                                    output_dir_path,
                                                                    output_prefix,
                                                                    output_max_size,
                                                                    self.tf_record_util.serialize_combined_sample),
            OPTICAL_FLOW: lambda: self.tf_record_util.serialize_dataset(OPTICAL_FLOW,
                                                                        raw_dataset,
                                                                        output_dir_path,
                                                                        output_prefix,
                                                                        output_max_size,
                                                                        self.tf_record_util.serialize_opticalflow_sample),
            RGB: lambda: self.tf_record_util.serialize_dataset(RGB,
                                                               raw_dataset,
                                                               output_dir_path,
                                                               output_prefix,
                                                               output_max_size,
                                                               self.tf_record_util.serialize_rgb_sample),
            NSDMV3: lambda: self.tf_record_util.serialize_dataset(NSDMV3,
                                                                  raw_dataset,
                                                                  output_dir_path,
                                                                  output_prefix,
                                                                  output_max_size,
                                                                  self.tf_record_util.serialize_nsdmv3_sample),
            SWAV: lambda: self.tf_record_util.serialize_dataset(SWAV,
                                                                raw_dataset,
                                                                output_dir_path,
                                                                output_prefix,
                                                                output_max_size,
                                                                self.tf_record_util.serialize_swav_sample),
        }

        if self.dataset_type in create_tfrecord_operations:
            self.logger.debug('An %s TFRecord dataset generation started', self.dataset_type)
            create_tfrecord_operations[self.dataset_type]()
        else:
            raise ValueError('Unrecognized operation "{}"'.format(self.dataset_type))

    def get_raw_dataset(self, raw_dataset_path, dataset_type, **kwargs):
        build_dataset_operations = {
            COMBINED: lambda: self.build_raw_dataset(raw_dataset_path,
                                                     kwargs['shuffle_buffer_size'],
                                                     self.get_raw_combined_list,
                                                     self.tf_get_combined_sample),
            OPTICAL_FLOW: lambda: self.build_raw_dataset(raw_dataset_path,
                                                         kwargs['shuffle_buffer_size'],
                                                         self.get_raw_opticalflow_list,
                                                         self.tf_get_opticalflow_sample),
            RGB: lambda: self.build_raw_dataset(raw_dataset_path,
                                                kwargs['shuffle_buffer_size'],
                                                self.get_raw_rgb_list,
                                                self.get_rgb_sample),
            NSDMV3: lambda: self.build_raw_dataset(raw_dataset_path,
                                                   kwargs['shuffle_buffer_size'],
                                                   self.get_raw_rgb_list,
                                                   self.get_rgb_sample),
            SWAV: lambda: self.get_swav_raw_dataset(
                                        raw_dataset_path,
                                        batch_size=kwargs['batch_size'],
                                        person_detection_model_name=kwargs['person_detection_model_name'],
                                        person_detection_checkout_prefix=kwargs['person_detection_checkout_prefix'],
                                        crop_sizes=kwargs['crop_sizes'],
                                        num_crops=kwargs['num_crops'],
                                        min_scale=kwargs['min_scale'],
                                        max_scale=kwargs['max_scale'],
                                        sample_duration_range=kwargs['sample_duration_range'],
                                        fixed_sample_duration=kwargs['fixed_sample_duration']),
        }

        if dataset_type in build_dataset_operations:
            self.logger.debug('An %s dataset generation was selected', dataset_type)
            dataset = build_dataset_operations[dataset_type]()
        else:
            raise ValueError('Unrecognized operation "{}"'.format(dataset_type))

        return dataset

    @staticmethod
    def get_swav_raw_dataset(train_dataset_path, batch_size, **kwargs):
        person_detection_model_name = kwargs['person_detection_model_name']
        person_detection_checkout_prefix = kwargs['person_detection_checkout_prefix']

        object_detection_utility = ObjectDetectionUtility()
        person_detection_model = object_detection_utility.get_object_detection_model(person_detection_model_name,
                                                                                     person_detection_checkout_prefix)
        data_preparer = SwAVDatasetPreparer(train_dataset_path,
                                            test_dataset_path=None,
                                            person_detection_model=person_detection_model,
                                            crop_sizes=kwargs['crop_sizes'],
                                            num_crops=kwargs['num_crops'],
                                            min_scale=kwargs['min_scale'],
                                            max_scale=kwargs['max_scale'],
                                            sample_duration_range=kwargs['sample_duration_range'],
                                            fixed_sample_duration=kwargs['fixed_sample_duration'])

        return data_preparer.prepare_train_dataset(batch_size)

    def py_get_opticalflow_sample(self, file_path):
        img_raw = tf.io.read_file(file_path)
        label = self.get_label(file_path)
        return img_raw, label

    def get_rgb_sample(self, folder_path):
        pattern = tf.strings.join([folder_path, tf.constant('*.jpg')], separator='/')
        sample_files_paths = tf.io.matching_files(pattern)

        rgb_frames = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        frame_index = 0

        for sample_file_path in sample_files_paths:
            frame = tf.io.read_file(sample_file_path)
            frame = tf.io.decode_jpeg(frame, channels=self.imagenet_rgb_no_channels)
            frame = tf.image.convert_image_dtype(frame, tf.float32)
            frame = tf.image.resize(frame, [self.imagenet_img_width, self.imagenet_img_height])
            rgb_frames = rgb_frames.write(frame_index, frame)
            frame_index += 1

        _, label_sparse = self.get_label(folder_path)
        return rgb_frames.stack(), label_sparse

    def py_get_combined_sample(self, sample_path):
        opticalflow_sample_path = sample_path[0]
        rgb_sample_path = sample_path[1]

        opticalflow_sample, label1 = self.py_get_opticalflow_sample(opticalflow_sample_path)
        rgb_sample, label2 = self.get_rgb_sample(rgb_sample_path)

        if tf.reduce_all(tf.not_equal(label1, label2)).numpy():
            raise ValueError('labels for Opticalflow and RGB samples do not match')

        return opticalflow_sample, rgb_sample, label1

    @staticmethod
    def build_raw_dataset(dataset_path, shuffle_buffer_size, raw_sample_list_func, map_func):
        raw_samples_list = raw_sample_list_func(dataset_path)
        dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(raw_samples_list)
        dataset = dataset.map(map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        return dataset

    @staticmethod
    def get_raw_opticalflow_list(dataset_path):
        # we return a sorted list to ensure a 1:1 mapping with opticalflow when using a combined dataset
        return sorted([str(file_path) for file_path in Path(dataset_path).rglob('*.jpg')])

    @staticmethod
    def get_raw_rgb_list(dataset_path):
        # we return a sorted list to ensure a 1:1 mapping with opticalflow when using a combined dataset
        return sorted([str(dir_path) for dir_path in Path(dataset_path).rglob('*/*') if dir_path.is_dir()])

    def get_raw_combined_list(self, dataset_path):
        if type(dataset_path) != list:
            raise ValueError('dataset_path is not a list.')

        if len(dataset_path) < 2:
            raise ValueError('passed dataset paths do not have the right amount (2)')

        # opticalflow and rgb dataset paths are passed as comma separated values.
        opticalflow_dataset_path = dataset_path[0]
        rgb_dataset_path = dataset_path[1]

        # we combine both sample path lists
        raw_samples_list = list(zip(self.get_raw_opticalflow_list(opticalflow_dataset_path),
                                    self.get_raw_rgb_list(rgb_dataset_path)))
        return raw_samples_list

    @staticmethod
    def get_label(file_path):
        sign_name = tf.strings.split(file_path, os.path.sep)[-2]

        class_index = tf.argmax(
            tf.cast(tf.map_fn(lambda x: tf.strings.regex_full_match(sign_name, x),
                              SIGNS_CLASSES, dtype=tf.bool), tf.int32))

        sparse_label = tf.zeros(len(SIGNS_CLASSES))
        indices = [[0, class_index]]
        values = [1.]
        shape = [1, len(SIGNS_CLASSES)]
        delta = tf.SparseTensor(indices, values, shape)

        return sign_name, tf.add(sparse_label, tf.sparse.to_dense(delta))

    def tf_get_opticalflow_sample(self, file_path):
        return tf.py_function(self.py_get_opticalflow_sample, [file_path], (tf.string, tf.float32))

    # def tf_get_rgb_sample(self, sample_folder_path):
    #     return tf.py_function(self.py_get_rgb_sample, [sample_folder_path], (tf.string, tf.float32))

    def tf_get_combined_sample(self, sample_path):
        return tf.py_function(self.py_get_combined_sample, [sample_path], (tf.string, tf.string, tf.float32))
