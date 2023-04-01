import os
import tensorflow as tf
import logging

from learning.common import features
from learning.common.dataset_type import OPTICAL_FLOW, RGB, SWAV


class TFRecordUtility:
    def __init__(self):
        self.tf_record_opts = tf.io.TFRecordOptions(compression_type='ZLIB', compression_level=9)
        self.compression_type = 'ZLIB'
        self.max_file_length_size = 1048576  # 100MB
        self.logger = logging.getLogger(__name__)

        self.swav_feature_description = {
            features.HIGH_RES_FRAMES_SEQ_1: tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            features.HIGH_RES_FRAMES_SEQ_2: tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            features.LOW_RES_FRAMES_SEQ_1: tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            features.LOW_RES_FRAMES_SEQ_2: tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            features.LOW_RES_FRAMES_SEQ_3: tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            features.NO_FRAMES: tf.io.FixedLenFeature([], dtype=tf.int64),
        }

    def serialize_opticalflow_sample(self, image_raw, label):
        feature = {
            features.IMAGE_RAW: self.bytes_feature(image_raw),
            features.LABEL: self.array_float_feature(label),
        }

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()  # TFRecord requires scalar strings

    def serialize_rgb_sample(self, rgb_frames, label):
        feature = {
            features.FRAMES_SEQ: self.array_bytes_feature(rgb_frames),
            features.LABEL: self.array_float_feature(label),
        }

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()  # TFRecord requires scalar strings

    def serialize_combined_sample(self, image_raw, rgb_frames, label):
        feature = {
            features.IMAGE_RAW: self.bytes_feature(image_raw),
            features.FRAMES_SEQ: self.array_bytes_feature(rgb_frames),
            features.LABEL: self.array_float_feature(label),
        }

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()  # TFRecord requires scalar strings

    def serialize_swav_sample(self, multicrop_crop_seqs):
        high_res_frames_seq_1 = tf.image.convert_image_dtype(tf.squeeze(multicrop_crop_seqs[0], axis=0), tf.uint8)  # 224x224
        high_res_frames_seq_2 = tf.image.convert_image_dtype(tf.squeeze(multicrop_crop_seqs[1], axis=0), tf.uint8)  # 224x224
        low_res_frames_seq_1 = tf.image.convert_image_dtype(tf.squeeze(multicrop_crop_seqs[2], axis=0), tf.uint8)   # 96x96
        low_res_frames_seq_2 = tf.image.convert_image_dtype(tf.squeeze(multicrop_crop_seqs[3], axis=0), tf.uint8)   # 96x96
        low_res_frames_seq_3 = tf.image.convert_image_dtype(tf.squeeze(multicrop_crop_seqs[4], axis=0), tf.uint8)  # 96x96

        feature = {
            features.HIGH_RES_FRAMES_SEQ_1: self.bytes_feature(tf.io.serialize_tensor(high_res_frames_seq_1)),
            features.HIGH_RES_FRAMES_SEQ_2: self.bytes_feature(tf.io.serialize_tensor(high_res_frames_seq_2)),
            features.LOW_RES_FRAMES_SEQ_1: self.bytes_feature(tf.io.serialize_tensor(low_res_frames_seq_1)),
            features.LOW_RES_FRAMES_SEQ_2: self.bytes_feature(tf.io.serialize_tensor(low_res_frames_seq_2)),
            features.LOW_RES_FRAMES_SEQ_3: self.bytes_feature(tf.io.serialize_tensor(low_res_frames_seq_3)),
            features.NO_FRAMES: self.int64_feature(high_res_frames_seq_1.shape[0])
        }

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
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

    @tf.function
    def parse_swav_dict_sample(self, sample):
        feature = tf.io.parse_single_example(sample, self.swav_feature_description)

        high_res_seq_1 = tf.io.parse_tensor(feature[features.HIGH_RES_FRAMES_SEQ_1], out_type=tf.uint8)  # 224x224
        high_res_seq_2 = tf.io.parse_tensor(feature[features.HIGH_RES_FRAMES_SEQ_2], out_type=tf.uint8)  # 224x224
        low_res_seq_1 = tf.io.parse_tensor(feature[features.LOW_RES_FRAMES_SEQ_1], out_type=tf.uint8)  # 96x96
        low_res_seq_2 = tf.io.parse_tensor(feature[features.LOW_RES_FRAMES_SEQ_2], out_type=tf.uint8)  # 96x96
        low_res_seq_3 = tf.io.parse_tensor(feature[features.LOW_RES_FRAMES_SEQ_3], out_type=tf.uint8)  # 96x96

        high_res_seq_1 = tf.ensure_shape(high_res_seq_1, [None, 224, 224, 3])
        high_res_seq_2 = tf.ensure_shape(high_res_seq_2, [None, 224, 224, 3])
        low_res_seq_1 = tf.ensure_shape(low_res_seq_1, [None, 96, 96, 3])
        low_res_seq_2 = tf.ensure_shape(low_res_seq_2, [None, 96, 96, 3])
        low_res_seq_3 = tf.ensure_shape(low_res_seq_3, [None, 96, 96, 3])

        high_res_seq_1 = tf.image.convert_image_dtype(high_res_seq_1, tf.float32)
        high_res_seq_2 = tf.image.convert_image_dtype(high_res_seq_2, tf.float32)
        low_res_seq_1 = tf.image.convert_image_dtype(low_res_seq_1, tf.float32)
        low_res_seq_2 = tf.image.convert_image_dtype(low_res_seq_2, tf.float32)
        low_res_seq_3 = tf.image.convert_image_dtype(low_res_seq_3, tf.float32)

        return high_res_seq_1, high_res_seq_2, low_res_seq_1, low_res_seq_2, low_res_seq_3

    def serialize_dataset(self,
                          dataset_type,
                          dataset,
                          output_dir_path,
                          output_prefix,
                          max_size_per_file: float,
                          sample_serialization_func):
        file_index = 0
        output_file_path = self.handle_split_file_name(output_dir_path, output_prefix, file_index)

        index = 1

        for sample in dataset:
            if not os.path.exists(output_file_path):
                writer = tf.io.TFRecordWriter(output_file_path, options=self.tf_record_opts)
                self.logger.debug('TFRecord file %s created.', output_file_path)

            if dataset_type == SWAV:
                example = sample_serialization_func(sample)
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

            file_size = os.stat(output_file_path).st_size >> 20
            if file_size > max_size_per_file:
                self.logger.debug('TFRecord file %s exceed maximum allowed file size %sMB, a new file will be created',
                                  output_file_path, str(max_size_per_file))
                writer.close()

                file_index = file_index + 1
                output_file_path = self.handle_split_file_name(output_dir_path, output_prefix, file_index)

            index = index + 1

        writer.close()

    def deserialize_dataset(self, tf_records_folder_path, parse_dict_sample_func, batch_size):
        file_pattern = os.path.join(tf_records_folder_path, '*.tfrecord')
        files_dataset = tf.data.Dataset.list_files(file_pattern)

        dataset = tf.data.TFRecordDataset(files_dataset, compression_type=self.compression_type)
        dataset = dataset.map(lambda sample: parse_dict_sample_func(sample), num_parallel_calls=tf.data.AUTOTUNE,
                              deterministic=False)

        return dataset

    @staticmethod
    def handle_split_file_name(output_dir_path, output_prefix, file_index):
        output_file_path = os.path.join(output_dir_path, '{}_{}.tfrecord'.format(output_prefix, file_index))
        if os.path.exists(output_file_path):
            os.remove(output_file_path)

        os.makedirs(output_dir_path, exist_ok=True)
        return output_file_path

    @staticmethod
    def bytes_feature(value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def array_bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value.numpy()))

    @staticmethod
    def array_int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value.numpy()))

    @staticmethod
    def float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def array_float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=tf.reshape(value, -1)))

    @staticmethod
    def int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
