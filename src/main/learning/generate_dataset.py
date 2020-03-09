import tensorflow as tf
import os
import logging
from pathlib import Path

from argparse import ArgumentParser
from logger_config import setup_logging
from learning.common.tf_record_utils import serialize_dataset, serialize_opticalflow_sample, serialize_rgb_sample

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
    args = get_cmd_args()

    dataset_path = args.dataset_path
    dataset_type = args.dataset_type
    output_dir_path = args.output_dir_path
    output_prefix = args.output_prefix
    output_max_size: float = float(args.output_max_size)
    shuffle_buffer_size = args.shuffle_buffer_size

    logger.debug('Source dataset path: %s', dataset_path)
    logger.debug('Source dataset type: %s', dataset_type)
    logger.debug('Output dir path: %s', output_dir_path)
    logger.debug('Output prefix: %s', output_prefix)
    logger.debug('Max size per output file: %s', output_max_size)
    logger.debug('Shuffle buffer size: %s', shuffle_buffer_size)

    # obtain dataset from image samples at the file system
    raw_dataset = get_raw_dataset(dataset_path, dataset_type, shuffle_buffer_size)

    # serialize samples into the TFRecord format for better I/O
    create_tfrecord_dataset(raw_dataset, dataset_type, output_dir_path, output_prefix, output_max_size)

    logger.info('Dataset generation process completed')


def get_raw_dataset(dataset_path, dataset_type, shuffle_buffer_size):
    build_dataset_operations = {
        'opticalflow': lambda: build_raw_dataset(dataset_path, shuffle_buffer_size, get_raw_opticalflow_list,
                                                 tf_get_opticalflow_sample),
        'rgb': lambda: build_raw_dataset(dataset_path, shuffle_buffer_size, get_raw_rgb_list, tf_get_rgb_sample),
    }

    if dataset_type in build_dataset_operations:
        logger.debug('An %s dataset generation was selected', dataset_type)
        dataset = build_dataset_operations[dataset_type]()
    else:
        raise ValueError('Unrecognized operation "{}"'.format(dataset_type))

    return dataset


def create_tfrecord_dataset(raw_dataset, dataset_type, output_dir_path, output_prefix, output_max_size):
    create_tfrecord_operations = {
        'opticalflow': lambda: serialize_dataset(raw_dataset,
                                                 output_dir_path,
                                                 output_prefix,
                                                 output_max_size,
                                                 serialize_opticalflow_sample),
        'rgb': lambda: serialize_dataset(raw_dataset,
                                         output_dir_path,
                                         output_prefix,
                                         output_max_size,
                                         serialize_rgb_sample),
    }

    if dataset_type in create_tfrecord_operations:
        logger.debug('An %s TFRecord dataset generation started', dataset_type)
        create_tfrecord_operations[dataset_type]()
    else:
        raise ValueError('Unrecognized operation "{}"'.format(dataset_type))


def build_raw_dataset(dataset_path, shuffle_buffer_size, raw_sample_list_func, map_func):
    raw_samples_list = raw_sample_list_func(dataset_path)
    dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(raw_samples_list)
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.map(map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


def get_raw_opticalflow_list(dataset_path):
    return [str(file_path) for file_path in Path(dataset_path).rglob('*.jpg')]


def get_raw_rgb_list(dataset_path):
    return [str(dir_path) for dir_path in Path(dataset_path).rglob('*/*') if dir_path.is_dir()]


def py_get_opticalflow_sample(file_path):
    img_raw = tf.io.read_file(file_path)
    label = tf.strings.split(file_path, os.path.sep)[-2]
    return img_raw, label


def tf_get_opticalflow_sample(file_path):
    return tf.py_function(py_get_opticalflow_sample, [file_path], (tf.string, tf.string))


def py_get_rgb_sample(folder_path):
    pattern = tf.strings.join([folder_path, tf.constant('*.jpg')], separator='/')
    sample_files_paths = tf.io.matching_files(pattern)

    rgb_frames = []
    for sample_file_path in sample_files_paths:
        raw_frame = tf.io.read_file(sample_file_path)
        rgb_frames.extend([raw_frame])

    label = tf.strings.split(folder_path, os.path.sep)[-2]
    return rgb_frames, label


def tf_get_rgb_sample(sample_folder_path):
    return tf.py_function(py_get_rgb_sample, [sample_folder_path], (tf.string, tf.string))


if __name__ == '__main__':
    working_folder = os.getenv('WORK_DIR', './')
    setup_logging(working_folder, 'dataset-logging.yaml')
    logger = logging.getLogger(__name__)

    logger.info('Dataset generation process started')

    main()
