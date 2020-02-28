import tensorflow as tf
import os
import logging

from argparse import ArgumentParser
from logger_config import setup_logging
from learning.common import tf_record_utils

DEFAULT_SHUFFLE_BUFFER_SIZE = 50000
DEFAULT_RGB_TF_RECORD_PREFIX = 'rgb'
DEFAULT_OPTICALFLOW_TF_RECORD_PREFIX = 'opticalflow'


def get_cmd_args():
    parser = ArgumentParser()
    parser.add_argument('-dp', '--dataset_path', help='Dataset path', required=True)
    parser.add_argument('-dt', '--dataset_type', help='Dataset type(opticalflow, rgb)', required=True)
    parser.add_argument('-od', '--output_dir_path', help='Output dir', required=True)
    parser.add_argument('-of', '--output_prefix', help='Output prefix', required=True)
    parser.add_argument('-fs', '--output_max_size', help='Max size per file in MB', default=100)
    parser.add_argument('-bf', '--shuffle_buffer_size', help='Dataset shuffle buffer size',
                        default=DEFAULT_SHUFFLE_BUFFER_SIZE)

    return parser.parse_args()


def main():
    working_folder = os.getenv('WORK_DIR', './')
    setup_logging(working_folder, 'dataset-logging.yaml')
    logger = logging.getLogger(__name__)

    logger.info('Dataset generation process started')

    args = get_cmd_args()

    dataset_path = args.dataset_path
    dataset_type = args.dataset_type
    output_dir_path = args.output_dir_path
    output_prefix = args.output_prefix
    output_max_size = args.output_max_size
    shuffle_buffer_size = args.shuffle_buffer_size

    logger.debug('Source dataset path: %s', dataset_path)
    logger.debug('Source dataset type: %s', dataset_type)
    logger.debug('Output dir path: %s', output_dir_path)
    logger.debug('Output prefix: %s', output_prefix)
    logger.debug('Max size per output file: %s', output_max_size)
    logger.debug('Shuffle buffer size: %s', shuffle_buffer_size)

    raw_dataset = get_raw_dataset(dataset_path, dataset_type, shuffle_buffer_size)
    tf_record_utils.serialize_dataset(raw_dataset, output_dir_path, output_prefix, output_max_size)

    logger.info('Dataset generation process completed')


def get_raw_dataset(dataset_path, dataset_type, shuffle_buffer_size):
    dataset: tf.data.Dataset = tf.data.Dataset.list_files(dataset_path + '/*/*')
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.map(tf_get_opticalflow_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset


def get_opticalflow_sample(file_path):
    img_raw = tf.io.read_file(file_path)
    label = tf.strings.split(file_path, os.path.sep)[-2]
    return img_raw, label


def tf_get_opticalflow_sample(file_path):
    return tf.py_function(get_opticalflow_sample, [file_path], (tf.string, tf.string))


def get_rgb_sample(tf_sample_folder_path):
    sample_folder_path = tf_sample_folder_path.numpy()
    pass


def tf_get_rgb_sample(sample_folder_path):
    return tf.py_function(get_rgb_sample, [sample_folder_path], (tf.string, tf.string))


if __name__ == '__main__':
    main()
