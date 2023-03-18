import os
import logging

from argparse import ArgumentParser
from logger_config import setup_logging
from learning.common.dataset_type import SWAV, COMBINED
from learning.dataset.tfrecord.tf_record_dataset_creator import TFRecordDatasetCreator

DEFAULT_SHUFFLE_BUFFER_SIZE = 50000
DEFAULT_RGB_TF_RECORD_PREFIX = 'rgb'
DEFAULT_OPTICALFLOW_TF_RECORD_PREFIX = 'opticalflow'


def get_cmd_args():
    parser = ArgumentParser()
    parser.add_argument('-dp', '--dataset_path',
                        help='Dataset path (enter a comma separated opticalflow and rgb data sets path if using the '
                             'combined option)',
                        required=True)
    parser.add_argument('-dt', '--dataset_type', help='Dataset type (combined, opticalflow, rgb)', required=True,
                        default='combined')
    parser.add_argument('--batch_size', type=int, help='Dataset batch size', default=3)
    parser.add_argument('-od', '--output_dir_path', help='Output dir', required=True)
    parser.add_argument('-of', '--output_prefix', help='Output prefix (eg. opticalflow, rgb or combined)',
                        required=True)
    parser.add_argument('-fs', '--output_max_size', help='Max size per file in MB', default=100)
    parser.add_argument('-bf', '--shuffle_buffer_size', help='Dataset shuffle buffer size',
                        default=DEFAULT_SHUFFLE_BUFFER_SIZE)

    # SwAV specific arguments
    parser.add_argument('--num_crops', type=int, help='SwAV number of multi-crops', nargs='+', default=[2, 3])
    parser.add_argument('--crops_for_assign', type=int, help='SwAV crops for assign', nargs='+', default=[0, 1])
    parser.add_argument('--crop_sizes_list', type=int, help='SwAV crop sizes list', nargs='+',
                        default=[224, 224, 96, 96, 96])
    parser.add_argument('--crop_sizes', type=int, help='SwAV crop sizes', nargs='+', default=[224, 96])
    parser.add_argument('--min_scale', type=float, help='SwAV Multi-crop min scale', nargs='+', default=[0.14, 0.05])
    parser.add_argument('--max_scale', type=float, help='SwAV Multi-crop max scale', nargs='+', default=[1., 0.14])
    parser.add_argument('--sample_duration_range', type=float, help='', nargs='+', default=[0.3, 0.5])

    # Person detection arguments
    parser.add_argument('--person_detection_model_name', help='Person Detection Model Name',
                        default='centernet_resnet50_v1_fpn_512x512_coco17_tpu-8')
    parser.add_argument('--person_detection_checkout_prefix', help='Person Detection Checkout Prefix', default='ckpt-0')

    return parser.parse_args()


def main():
    args = get_cmd_args()

    dataset_type = args.dataset_type
    dataset_path = args.dataset_path.split(',') if dataset_type == COMBINED else args.dataset_path
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
    dataset_creator = TFRecordDatasetCreator(dataset_type, dataset_path)
    # serialize samples into the TFRecord format for better I/O

    if dataset_type == SWAV:
        dataset_creator.create(output_dir_path,
                               output_prefix,
                               output_max_size,
                               batch_size=args.batch_size,
                               person_detection_model_name=args.person_detection_model_name,
                               person_detection_checkout_prefix=args.person_detection_checkout_prefix,
                               crop_sizes=args.crop_sizes,
                               num_crops=args.num_crops,
                               min_scale=args.min_scale,
                               max_scale=args.max_scale,
                               sample_duration_range=args.sample_duration_range)
    else:
        dataset_creator.create(output_dir_path,
                               output_prefix,
                               output_max_size, shuffle_buffer_size=args.shuffle_buffer_size)

    logger.info('Dataset generation process completed')


if __name__ == '__main__':
    working_folder = os.getenv('WORK_DIR', './')
    setup_logging(working_folder, 'dataset-logging.yaml')
    logger = logging.getLogger(__name__)

    logger.info('Dataset generation process started')

    main()
