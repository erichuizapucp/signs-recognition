import os
import tensorflow as tf
import logging

from learning.common import features

COMPRESSION_TYPE = 'ZLIB'
MAX_FILE_LENGTH_SIZE = 1048576

logger = logging.getLogger(__name__)


def bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def array_bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value.numpy()))


def array_int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value.numpy()))


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


def serialize_rgb_sample(rgb_frames, label):
    frames_shape = tf.image.decode_jpeg(rgb_frames[0]).shape

    feature = {
        features.FRAMES_SEQ: array_bytes_feature(rgb_frames),
        features.HEIGHT: int64_feature(frames_shape[0]),
        features.WIDTH: int64_feature(frames_shape[1]),
        features.DEPTH: int64_feature(frames_shape[2]),
        features.LABEL: bytes_feature(label),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()  # TFRecord requires scalar strings


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
        features.FRAMES_SEQ: tf.io.VarLenFeature(tf.string),
        features.HEIGHT: tf.io.FixedLenFeature([], tf.int64, default_value=0),
        features.WIDTH: tf.io.FixedLenFeature([], tf.int64, default_value=0),
        features.DEPTH: tf.io.FixedLenFeature([], tf.int64, default_value=0),
        features.LABEL: tf.io.FixedLenFeature([], tf.string, default_value=''),
    }

    feature = tf.io.parse_single_example(sample, feature_description)

    dense_frames_seq = tf.sparse.to_dense(feature[features.FRAMES_SEQ])
    height = feature[features.HEIGHT]
    width = feature[features.WIDTH]
    depth = feature[features.DEPTH]
    label = feature[features.LABEL]

    return dense_frames_seq, height, width, depth, label


def tf_parse_rgb_dict_sample(sample):
    parse_func = parse_rgb_dict_sample
    return_type = (tf.string, tf.int64, tf.int64, tf.int64, tf.string)
    tf_frames_seq, tf_height, tf_width, tf_depth, tf_label = tf.py_function(parse_func, [sample], return_type)
    return tf_frames_seq, tf_height, tf_width, tf_depth, tf_label


def serialize_dataset(dataset: tf.data.Dataset, output_dir_path, output_prefix, max_size_per_file: float,
                      sample_serialization_func):
    file_index = 0
    file_size = 0
    output_file_path = __handle_split_file_name(output_dir_path, output_prefix, file_index)

    writer = tf.io.TFRecordWriter(output_file_path, options=COMPRESSION_TYPE)
    index = 1
    for raw_sample, label in dataset:
        example = sample_serialization_func(raw_sample, label)
        writer.write(example)

        logger.debug('Sample %s serialization process completed.', str(index))

        file_size = file_size + (len(example) / MAX_FILE_LENGTH_SIZE)
        if file_size > max_size_per_file:
            logger.debug('TFRecord file %s exceed maximum allowed file size %sMB, a new file will be created',
                         output_file_path,
                         str(max_size_per_file))

            writer.close()

            file_index = file_index + 1
            output_file_path = __handle_split_file_name(output_dir_path, output_prefix, file_index)
            writer = tf.io.TFRecordWriter(output_file_path, options=COMPRESSION_TYPE)

            logger.debug('TFRecord file %s created.', output_file_path)

            file_size = 0

        index = index + 1

    writer.close()


def deserialize_dataset(tf_records_folder_path, parse_dict_sample_func):
    file_pattern = os.path.join(tf_records_folder_path, '*.tfrecord')
    files_dataset = tf.data.Dataset.list_files(file_pattern)

    dataset = tf.data.TFRecordDataset(files_dataset, compression_type=COMPRESSION_TYPE)
    dataset = dataset.map(lambda sample: parse_dict_sample_func(sample))
    return dataset


def __handle_split_file_name(output_dir_path, output_prefix, file_index):
    output_file_path = os.path.join(output_dir_path, '{}_{}.tfrecord'.format(output_prefix, file_index))
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    os.makedirs(output_dir_path, exist_ok=True)
    return output_file_path
