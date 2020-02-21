import os
import tensorflow as tf

from learning.common import features


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
        features.IMAGE_RAW: bytes_feature(image_raw),
        features.HEIGHT: int64_feature(image_shape[0]),
        features.WIDTH: int64_feature(image_shape[1]),
        features.DEPTH: int64_feature(image_shape[2]),
        features.LABEL: int64_feature(label),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()  # TFRecord requires scalar strings


def tf_serialize_sample(image_raw, label):
    tf_sample = tf.py_function(serialize_sample, (image_raw, label), tf.string)
    return tf.reshape(tf_sample, ())  # TFRecord requires scalar strings


def parse_dict_sample(sample):
    feature_description = {
        features.IMAGE_RAW: tf.io.FixedLenFeature([], tf.string, default_value=''),
        features.HEIGHT: tf.io.FixedLenFeature([], tf.int64, default_value=0),
        features.WIDTH: tf.io.FixedLenFeature([], tf.int64, default_value=0),
        features.DEPTH: tf.io.FixedLenFeature([], tf.int64, default_value=0),
        features.LABEL: tf.io.FixedLenFeature([], tf.int64, default_value=0),  # Label should be an integer
    }

    feature = tf.io.parse_single_example(sample, feature_description)
    return feature[features.IMAGE_RAW], feature[features.HEIGHT], feature[features.WIDTH], feature[features.DEPTH], \
        feature[features.LABEL]


def tf_parse_dict_sample(sample):
    parse_func = parse_dict_sample
    return_type = (tf.string, tf.int64, tf.int64, tf.int64, tf.int64)
    tf_height, tf_width, tf_depth, tf_image_raw, tf_label = tf.py_function(parse_func, [sample], return_type)

    return tf_height, tf_width, tf_depth, tf_image_raw, tf_label


def serialize_dataset(dataset: tf.data.Dataset, output_prefix, max_size_per_file):
    # serialize dataset to a .tfrecord file for further usage
    serialized_dataset = dataset.map(lambda image_raw, label: tf_serialize_sample(image_raw, label))

    file_index = 1
    output_file_path = '{}_{}.tfrecord'.format(output_prefix, file_index)
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    while not os.stat(output_file_path).st_size > max_size_per_file * 1048576:
        pass

    writer = tf.io.TFRecordWriter.TFRecordWriter(output_file_path)
    writer.write(serialized_dataset)


def deserialize_dataset(tf_record_file_path):
    dataset = tf.data.TFRecordDataset(tf_record_file_path).map(lambda sample: tf_parse_dict_sample(sample))
    return dataset
