import os
import tensorflow as tf
import numpy as np

from learning.common import features

COMPRESSION_TYPE = 'ZLIB'
MAX_FILE_LENGTH_SIZE = 1048576


def bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def array_int64_feature(value):
    tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_opticalflow_sample(image_raw, label):
    image_shape = tf.image.decode_jpeg(image_raw).shape

    feature = {
        features.IMAGE_RAW: bytes_feature(image_raw),
        features.HEIGHT: int64_feature(image_shape[0]),
        features.WIDTH: int64_feature(image_shape[1]),
        features.DEPTH: int64_feature(image_shape[2]),
        features.LABEL: bytes_feature(label),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()  # TFRecord requires scalar strings


def serialize_rgb_sample(raw_rgb_sample, label):
    rgb_sample_features = pack_rgb_sample_features(raw_rgb_sample)
    feature = {
        features.RGB_FEATURES: array_int64_feature(rgb_sample_features),
        features.LABEL: bytes_feature(label),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()  # TFRecord requires scalar strings


def pack_rgb_sample_features(tf_raw_rgb_sample):
    raw_rgb_sample = tf_raw_rgb_sample.numpy()

    all_frames_features = []
    for raw_rgb_sample_frame in raw_rgb_sample:
        decoded_image = tf.image.decode_jpeg(raw_rgb_sample_frame)
        image_shape = decoded_image.shape

        image_height = image_shape[0]
        image_width = image_shape[1]
        image_depth = image_shape[2]

        frame_features = [decoded_image, image_height, image_width, image_depth]
        all_frames_features.extend([frame_features])

    return all_frames_features


def parse_opticalflow_dict_sample(sample):
    feature_description = {
        features.IMAGE_RAW: tf.io.FixedLenFeature([], tf.string, default_value=''),
        features.HEIGHT: tf.io.FixedLenFeature([], tf.int64, default_value=0),
        features.WIDTH: tf.io.FixedLenFeature([], tf.int64, default_value=0),
        features.DEPTH: tf.io.FixedLenFeature([], tf.int64, default_value=0),
        features.LABEL: tf.io.FixedLenFeature([], tf.string, default_value=''),
    }

    feature = tf.io.parse_single_example(sample, feature_description)
    return feature[features.IMAGE_RAW], feature[features.HEIGHT], feature[features.WIDTH], feature[features.DEPTH], \
        feature[features.LABEL]


def tf_parse_opticalflow_dict_sample(sample):
    parse_func = parse_opticalflow_dict_sample
    return_type = (tf.string, tf.int64, tf.int64, tf.int64, tf.string)
    tf_image_raw, tf_height, tf_width, tf_depth, tf_label = tf.py_function(parse_func, [sample], return_type)

    return tf_image_raw, tf_height, tf_width, tf_depth, tf_label


def parse_rgb_dict_sample(sample):
    feature_description = {
        features.RGB_FEATURES: tf.io.VarLenFeature(tf.string),
        features.LABEL: tf.io.FixedLenFeature([], tf.string, default_value=''),
    }

    feature = tf.io.parse_single_example(sample, feature_description)
    return feature[features.RGB_FEATURES], feature[features.LABEL]


def tf_parse_rgb_dict_sample(sample):
    parse_func = parse_rgb_dict_sample
    return_type = (tf.string, tf.string)
    tf_rgb_features, tf_label = tf.py_function(parse_func, [sample], return_type)
    return tf_rgb_features, tf_label


def serialize_dataset(dataset: tf.data.Dataset, output_dir_path, output_prefix, max_size_per_file: float,
                      sample_serialization_func):
    file_index = 0
    file_size = 0
    output_file_path = __handle_split_file_name(output_dir_path, output_prefix, file_index)

    writer = tf.io.TFRecordWriter(output_file_path, options=COMPRESSION_TYPE)
    for raw_sample, label in dataset:
        example = sample_serialization_func(raw_sample, label)
        writer.write(example)

        file_size = file_size + (len(example) / MAX_FILE_LENGTH_SIZE)
        if file_size > max_size_per_file:
            writer.close()

            file_index = file_index + 1
            output_file_path = __handle_split_file_name(output_dir_path, output_prefix, file_index)
            writer = tf.io.TFRecordWriter(output_file_path, options=COMPRESSION_TYPE)

            file_size = 0

    writer.close()


def deserialize_dataset(tf_records_folder_path):
    file_pattern = os.path.join(tf_records_folder_path, '*.tfrecord')
    files_dataset = tf.data.Dataset.list_files(file_pattern)

    dataset = tf.data.TFRecordDataset(files_dataset, compression_type=COMPRESSION_TYPE)
    dataset = dataset.map(lambda sample: tf_parse_opticalflow_dict_sample(sample))
    return dataset


def __handle_split_file_name(output_dir_path, output_prefix, file_index):
    output_file_path = os.path.join(output_dir_path, '{}_{}.tfrecord'.format(output_prefix, file_index))
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    os.makedirs(output_dir_path, exist_ok=True)
    return output_file_path
