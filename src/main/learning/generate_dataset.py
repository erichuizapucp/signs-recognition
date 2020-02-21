import tensorflow as tf
import os

from argparse import ArgumentParser
from logger_config import setup_logging
from learning.common import tf_record_utils

DEFAULT_SHUFFLE_BUFFER_SIZE = 50000
DEFAULT_RGB_DIR_NAME = 'rgb'
DEFAULT_OPTICALFLOW_DIR_NAME = 'opticalflow'
DEFAULT_RGB_TF_RECORD_PREFIX = ''
DEFAULT_OPTICALFLOW_TF_RECORD_PREFIX = ''
DEFAULT_BATCH_SIZE = 5


def get_cmd_args():
    parser = ArgumentParser()
    parser.add_argument('-dp', '--dataset_path', help='Dataset path', required=True)
    parser.add_argument('-dt', '--dataset_type', help='Dataset type(opticalflow, rgb)', required=True)
    parser.add_argument('-od', '--output_dir_path', help='Output dir', required=True)
    parser.add_argument('-of', '--output_prefix', help='Output prefix', required=True)
    parser.add_argument('-fs', '--output_max_size', help='Max size per file in MB', default=100)
    parser.add_argument('-bf', '--shuffle_buffer_size', help='Dataset shuffle buffer size',
                        default=DEFAULT_SHUFFLE_BUFFER_SIZE)
    parser.add_argument('-bs', '--batch_size', help='Batch size',
                        default=DEFAULT_BATCH_SIZE)

    return parser.parse_args()


def main():
    args = get_cmd_args()

    dataset_path = args.dataset_path
    dataset_type = args.dataset_type
    output_dir_path = args.output_dir_path
    output_prefix = args.output_prefix
    output_max_size = args.output_max_size

    shuffle_buffer_size = args.shuffle_buffer_size
    batch_size = args.batch_size

    raw_dataset = get_raw_dataset(dataset_path, dataset_type, shuffle_buffer_size, batch_size)
    tf_record_utils.serialize_dataset(raw_dataset, '')


def get_raw_dataset(dataset_path, dataset_type, shuffle_buffer_size, batch_size):
    dataset: tf.data.Dataset = tf.data.Dataset.list_files(dataset_path + '/*/*')
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.map(tf_get_opticalflow_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset


def get_opticalflow_sample(file_path):
    with open(file_path, 'rb') as img_file:
        img_raw = img_file.read()
    label = tf.strings.split(file_path, os.path.sep)[-2]
    return img_raw, label


def tf_get_opticalflow_sample(file_path):
    return tf.py_function(get_opticalflow_sample, [file_path])


if __name__ == '__main__':
    main()