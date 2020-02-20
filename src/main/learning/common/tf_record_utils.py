import tensorflow as tf
import os

from constants import *


def bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_sample(image_raw, label):
    image_shape = tf.image.decode_jpeg(image_raw).shape

    feature = {
        HEIGHT_FEATURE_NAME: int64_feature(image_shape[0]),
        WIDTH_FEATURE_NAME: int64_feature(image_shape[1]),
        DEPTH_FEATURE_NAME: int64_feature(image_shape[2]),
        IMAGE_RAW_FEATURE_NAME: bytes_feature(image_raw),
        TARGET_FEATURE_NAME: int64_feature(label),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_sample(image_raw, label):
    tf_sample = tf.py_function(serialize_sample, (image_raw, label), tf.string)
    return tf.reshape(tf_sample, ())


def parse_dict_sample(sample):
    feature_description = {
        HEIGHT_FEATURE_NAME: tf.io.FixedLenFeature([], tf.int64, default_value=0),
        WIDTH_FEATURE_NAME: tf.io.FixedLenFeature([], tf.int64, default_value=0),
        DEPTH_FEATURE_NAME: tf.io.FixedLenFeature([], tf.int64, default_value=0),
        IMAGE_RAW_FEATURE_NAME: tf.io.FixedLenFeature([], tf.string, default_value=''),
        TARGET_FEATURE_NAME: tf.io.FixedLenFeature([], tf.int64, default_value=0),
    }

    feature = tf.io.parse_single_example(sample, feature_description)
    return feature[HEIGHT_FEATURE_NAME], feature[WIDTH_FEATURE_NAME], feature[DEPTH_FEATURE_NAME], \
        feature[IMAGE_RAW_FEATURE_NAME], feature[TARGET_FEATURE_NAME]


def tf_parse_dict_sample(sample, is_encoded=False):
    parse_func = parse_dict_sample
    return_type = (tf.int64, tf.int64) if is_encoded else (tf.string, tf.int64)
    tf_height, tf_width, tf_depth, tf_image_raw, tf_label = tf.py_function(parse_func, [sample], return_type)

    return tf_height, tf_width, tf_depth, tf_image_raw, tf_label


def serialize_dataset(dataset: tf.data.Dataset, output_file_path):
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    # serialize dataset to a .tfrecord file for further usage
    serialized_dataset = dataset.map(lambda image_raw, label: tf_serialize_sample(image_raw, label))
    writer = tf.io.TFRecordWriter.TFRecordWriter(output_file_path)
    writer.write(serialized_dataset)


def deserialize_dataset(tf_record_file_path):
    dataset = tf.data.TFRecordDataset(tf_record_file_path).map(lambda sample: tf_parse_dict_sample(sample))
    return dataset
