import os
import tensorflow as tf
import logging

from learning.common import features
from learning.common.dataset_type import OPTICAL_FLOW, RGB, SWAV


class TFRecordUtility:
    def __init__(self):
        self.compression_type = 'ZLIB'
        self.max_file_length_size = 1048576  # 100MB
        self.logger = logging.getLogger(__name__)

    def serialize_opticalflow_sample(self, image_raw, label):
        feature = {
            features.IMAGE_RAW: self.__bytes_feature(image_raw),
            features.LABEL: self.__array_float_feature(label),
        }

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()  # TFRecord requires scalar strings

    def serialize_rgb_sample(self, rgb_frames, label):
        feature = {
            features.FRAMES_SEQ: self.__array_bytes_feature(rgb_frames),
            features.LABEL: self.__array_float_feature(label),
        }

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()  # TFRecord requires scalar strings

    def serialize_combined_sample(self, image_raw, rgb_frames, label):
        feature = {
            features.IMAGE_RAW: self.__bytes_feature(image_raw),
            features.FRAMES_SEQ: self.__array_bytes_feature(rgb_frames),
            features.LABEL: self.__array_float_feature(label),
        }

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()  # TFRecord requires scalar strings

    def serialize_swav_sample(self, multicrop_crop_seqs):
        high_res_frames_seq_1 = multicrop_crop_seqs[0]  # 224x224
        high_res_frames_seq_2 = multicrop_crop_seqs[1]  # 224x224
        low_res_frames_seq_1 = multicrop_crop_seqs[2]   # 96x96
        low_res_frames_seq_2 = multicrop_crop_seqs[3]   # 96x96
        low_res_frames_seq_3 = multicrop_crop_seqs[4]   # 96x96

        feature = {
            features.HIGH_RES_FRAMES_SEQ_1: self.__array_bytes_feature(high_res_frames_seq_1),
            features.HIGH_RES_FRAMES_SEQ_2: self.__array_bytes_feature(high_res_frames_seq_2),
            features.LOW_RES_FRAMES_SEQ_1: self.__array_bytes_feature(low_res_frames_seq_1),
            features.LOW_RES_FRAMES_SEQ_2: self.__array_bytes_feature(low_res_frames_seq_2),
            features.LOW_RES_FRAMES_SEQ_3: self.__array_bytes_feature(low_res_frames_seq_3),
        }

        example_proto = tf.train.Example(features=tf.train.Feature(feature=feature))
        return example_proto.SerializeToString()

    @staticmethod
    def parse_opticalflow_dict_sample(sample):
        feature_description = {
            features.IMAGE_RAW: tf.io.FixedLenFeature([], tf.string, default_value=''),
            features.LABEL: tf.io.VarLenFeature(tf.float32),
        }
        feature = tf.io.parse_single_example(sample, feature_description)

        image_raw = feature[features.IMAGE_RAW]
        label = tf.sparse.to_dense(feature[features.LABEL])

        return image_raw, label

    @staticmethod
    def parse_rgb_dict_sample(sample):
        feature_description = {
            features.FRAMES_SEQ: tf.io.VarLenFeature(tf.string),
            features.LABEL: tf.io.VarLenFeature(tf.float32),
        }
        feature = tf.io.parse_single_example(sample, feature_description)

        dense_frames_seq = tf.sparse.to_dense(feature[features.FRAMES_SEQ])
        label = tf.sparse.to_dense(feature[features.LABEL])

        return dense_frames_seq, label

    @staticmethod
    def parse_combined_dict_sample(sample):
        feature_description = {
            features.IMAGE_RAW: tf.io.FixedLenFeature([], tf.string, default_value=''),
            features.FRAMES_SEQ: tf.io.VarLenFeature(tf.string),
            features.LABEL: tf.io.VarLenFeature(tf.float32),
        }

        feature = tf.io.parse_single_example(sample, feature_description)

        image_raw = feature[features.IMAGE_RAW]
        dense_frames_seq = tf.sparse.to_dense(feature[features.FRAMES_SEQ])
        label = tf.sparse.to_dense(feature[features.LABEL])

        return image_raw, dense_frames_seq, label

    @staticmethod
    def parse_swav_dict_sample(sample):
        feature_description = {
            features.HIGH_RES_FRAMES_SEQ_1: tf.io.VarLenFeature(tf.string),
            features.HIGH_RES_FRAMES_SEQ_2: tf.io.VarLenFeature(tf.string),
            features.LOW_RES_FRAMES_SEQ_1: tf.io.VarLenFeature(tf.string),
            features.LOW_RES_FRAMES_SEQ_2: tf.io.VarLenFeature(tf.string),
            features.LOW_RES_FRAMES_SEQ_3: tf.io.VarLenFeature(tf.string),
        }
        feature = tf.io.parse_single_example(sample, feature_description)

        dense_high_res_seq_1 = tf.sparse.to_dense(feature[features.HIGH_RES_FRAMES_SEQ_1])  # 224x224
        dense_high_res_seq_2 = tf.sparse.to_dense(feature[features.HIGH_RES_FRAMES_SEQ_2])  # 224x224
        dense_low_res_seq_1 = tf.sparse.to_dense(feature[features.LOW_RES_FRAMES_SEQ_1])  # 96x96
        dense_low_res_seq_2 = tf.sparse.to_dense(feature[features.LOW_RES_FRAMES_SEQ_2])  # 96x96
        dense_low_res_seq_3 = tf.sparse.to_dense(feature[features.LOW_RES_FRAMES_SEQ_3])  # 96x96

        return dense_high_res_seq_1, dense_high_res_seq_2, dense_low_res_seq_1, dense_low_res_seq_2, dense_low_res_seq_3

    def serialize_dataset(self,
                          dataset_type,
                          dataset: tf.data.Dataset,
                          output_dir_path,
                          output_prefix,
                          max_size_per_file: float,
                          sample_serialization_func):
        file_index = 0
        file_size = 0
        output_file_path = self.__handle_split_file_name(output_dir_path, output_prefix, file_index)

        writer = tf.io.TFRecordWriter(output_file_path, options=self.compression_type)
        index = 1

        for sample in dataset:
            if dataset_type == SWAV:
                feature = sample[0]
                example = sample_serialization_func(feature)
                writer.write(example)
            elif dataset_type in (OPTICAL_FLOW, RGB):
                feature = sample[0]
                label = sample[1]

                example = sample_serialization_func(feature, label)
                writer.write(example)
            else:
                feature1 = sample[0]
                feature2 = sample[1]
                label = sample[2]

                example = sample_serialization_func(feature1, feature2, label)
                writer.write(example)

            self.logger.debug('Sample %s serialization process completed.', str(index))

            file_size = file_size + (len(example) / self.max_file_length_size)
            if file_size > max_size_per_file:
                self.logger.debug('TFRecord file %s exceed maximum allowed file size %sMB, a new file will be created',
                                  output_file_path, str(max_size_per_file))
                writer.close()

                file_index = file_index + 1
                output_file_path = self.__handle_split_file_name(output_dir_path, output_prefix, file_index)
                writer = tf.io.TFRecordWriter(output_file_path, options=self.compression_type)

                self.logger.debug('TFRecord file %s created.', output_file_path)
                file_size = 0

            index = index + 1

        writer.close()

    def deserialize_dataset(self, tf_records_folder_path, parse_dict_sample_func):
        file_pattern = os.path.join(tf_records_folder_path, '*.tfrecord')
        files_dataset = tf.data.Dataset.list_files(file_pattern)

        dataset = tf.data.TFRecordDataset(files_dataset, compression_type=self.compression_type)
        dataset = dataset.map(lambda sample: parse_dict_sample_func(sample))
        return dataset

    @staticmethod
    def __handle_split_file_name(output_dir_path, output_prefix, file_index):
        output_file_path = os.path.join(output_dir_path, '{}_{}.tfrecord'.format(output_prefix, file_index))
        if os.path.exists(output_file_path):
            os.remove(output_file_path)

        os.makedirs(output_dir_path, exist_ok=True)
        return output_file_path

    @staticmethod
    def __bytes_feature(value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def __array_bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value.numpy()))

    @staticmethod
    def __array_int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value.numpy()))

    @staticmethod
    def __float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def __array_float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value.numpy()))

    @staticmethod
    def __int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
