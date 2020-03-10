import os
import logging

from argparse import ArgumentParser
from logger_config import setup_logging
from learning.dataset.tf_record_dataset_creator import TFRecordDatasetCreator

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

    # obtain an instance of a dataset creator
    dataset_creator = TFRecordDatasetCreator()

    # obtain dataset from image samples at the file system
    raw_dataset = dataset_creator.get_raw_dataset(dataset_path, dataset_type, shuffle_buffer_size)

    # serialize samples into the TFRecord format for better I/O
    dataset_creator.create_tfrecord_dataset(raw_dataset, dataset_type, output_dir_path, output_prefix, output_max_size)

    logger.info('Dataset generation process completed')


if __name__ == '__main__':
    working_folder = os.getenv('WORK_DIR', './')
    setup_logging(working_folder, 'dataset-logging.yaml')
    logger = logging.getLogger(__name__)

    logger.info('Dataset generation process started')

    main()
