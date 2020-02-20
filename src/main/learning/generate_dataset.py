import tensorflow as tf
import os

from argparse import ArgumentParser
from logger_config import setup_logging

DEFAULT_SHUFFLE_BUFFER_SIZE = 50000
DEFAULT_RGB_DATASET_PATH = 'dataset/rgb'
DEFAULT_OPTICALFLOW_DATASET_PATH = 'dataset/opticalflow'
DEFAULT_RGB_TF_RECORD_PREFIX = ''
DEFAULT_OPTICALFLOW_TF_RECORD_PREFIX = ''


def get_cmd_args():
    parser = ArgumentParser()
    parser.add_argument('-rf', '--rgb_path', help='RGB Dataset path',
                        default=DEFAULT_RGB_DATASET_PATH)
    parser.add_argument('-of', '--opticalflow_path', help='Opticalflow Dataset path',
                        default=DEFAULT_OPTICALFLOW_DATASET_PATH)
    parser.add_argument('-bf', '--shuffle_buffer_size', help='Dataset shuffle buffer size',
                        default=DEFAULT_SHUFFLE_BUFFER_SIZE)
    parser.add_argument('-rp', '--rgb_output_prefix', help='RGB output file prefix',
                        default=DEFAULT_RGB_TF_RECORD_PREFIX)
    parser.add_argument('-op', '--opticalflow_output_prefix', help='Opticalflow output file prefix',
                        default=DEFAULT_OPTICALFLOW_TF_RECORD_PREFIX)

    return parser.parse_args()


def main():
    working_folder = os.getenv('WORK_DIR', './')

    args = get_cmd_args()

    rgb_path = args.rgb_path
    opticalflow_path = args.opticalflow_path
    rgb_output_prefix = args.rgb_output_prefix
    opticalflow_output_prefix = args.opticalflow_output_prefix
    shuffle_buffer_size: int = int(args.shuffle_buffer_size)

    # dataset = tf.data.Dataset.list_files(dataset_path + '/*/*').shuffle(buffer_size=shuffle_buffer_size)\
    #         .map(lambda x: self.process_image_path(x, no_channels, img_width, img_height),
    #              num_parallel_calls=tf.data.experimental.AUTOTUNE)\
    #         .batch(batch_size)


if __name__ == '__main__':
    main()
