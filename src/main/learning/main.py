import logging
import os
import tensorflow as tf

from learning.models.opticalflow_model import OpticalFlowModel
from learning.models.rgb_recurrent_model import RGBRecurrentModel
from learning.models.novel_signs_detection_model import NovelSignsDetectionModel
from argparse import ArgumentParser

from logger_config import setup_logging

DEFAULT_LEARNING_RATE = 0.001
DEFAULT_NO_EPOCHS = 5
DEFAULT_NO_STEPS_EPOCHS = 4
DEFAULT_BATCH_SIZE = 1
DEFAULT_NO_CHANNELS = 3  # Color images
DEFAULT_IMG_WIDTH = 224  # Imagenet default image width
DEFAULT_IMG_HEIGHT = 224  # Imagenet default image height
DEFAULT_SHUFFLE_BUFFER_SIZE = 100  # Data shuffling buffer size


def get_cmd_args():
    parser = ArgumentParser()

    parser.add_argument('-m', '--model', help='Model Name', required=True)
    parser.add_argument('-o', '--operation', help='Operation', required=True)
    parser.add_argument('-ds', '--dataset_path', help='Dataset Path', required=True)
    parser.add_argument('-bs', '--batch_size', help='Batch Size', default=DEFAULT_BATCH_SIZE)
    parser.add_argument('-ne', '--no_epochs', help='Number of epochs', default=DEFAULT_NO_EPOCHS)
    parser.add_argument('-lr', '--learning_rate', help='Leaning Rate', default=DEFAULT_LEARNING_RATE)
    parser.add_argument('-ns', '--no_steps', help='Number of steps per epoch', default=DEFAULT_NO_STEPS_EPOCHS)
    parser.add_argument('-nc', '--no_channels', help='Number of channels', default=DEFAULT_NO_CHANNELS)
    parser.add_argument('-iw', '--img_width', help='Image width', default=DEFAULT_IMG_WIDTH)
    parser.add_argument('-ih', '--img_height', help='Image height', default=DEFAULT_IMG_HEIGHT)
    parser.add_argument('-bf', '--shuffle_buffer_size', help='Dataset shuffle buffer size',
                        default=DEFAULT_SHUFFLE_BUFFER_SIZE)
    return parser.parse_args()


def get_train_params(args):
    train_kwargs = {
        'BatchSize': args.batch_size,
        'NoEpochs': args.no_epochs,
        'LearningRate': args.learning_rate,
        'ShuffleBufferSize': args.shuffle_buffer_size,
        'ImageWidth': args.img_width,
        'ImageHeight': args.img_height,
        'NoChannels': args.no_channels,
        'NoStepsPerEpoch': args.no_steps,
    }
    return train_kwargs


def get_model(working_folder, dataset_root_path, model_name):
    models = {
        'opticalflow': lambda: OpticalFlowModel(working_folder, dataset_root_path),
        'rgb': lambda: RGBRecurrentModel(working_folder, dataset_root_path),
        'nsdm': lambda: NovelSignsDetectionModel(working_folder, dataset_root_path)
    }
    model = models[model_name]()
    return model


def main():
    working_folder = os.getenv('WORK_DIR', './')

    # configure logging
    setup_logging(working_folder, 'learning-logging.yaml')
    logger = logging.getLogger(__name__)

    args = get_cmd_args()
    dataset_root_path = args.dataset_path

    logger.debug('learning operation started with the following parameters: %s', args)

    train_kwargs = get_train_params(args)
    model = get_model(working_folder, dataset_root_path, args.model)

    operations = {
        'train': lambda: model.train(**train_kwargs),
        'evaluate': lambda: model.evaluate(),
        'predict': lambda: model.predict(),
    }
    operations[args.operation]()

    logger.debug('learning operation is completed')


if __name__ == '__main__':
    main()
