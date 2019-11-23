import logging
import os

from learning.models.opticalflow_model import OpticalFlowModel
from learning.models.rgb_model import RGBModel
from argparse import ArgumentParser

from logger_config import setup_logging

DEFAULT_NO_CLASSES = 10
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_NO_EPOCHS = 50
DEFAULT_BATCH_SIZE = 10

optical_flow_network_name = 'opticalflow'
rgb_network_name = 'rgb'


def get_opticalflow_model():
    nn = OpticalFlowModel(working_folder, dataset_path, **kwargs)
    return nn


def get_rgb_model():
    nn = RGBModel(working_folder, dataset_path, **kwargs)
    return nn


def get_cmd_args():
    parser = ArgumentParser()

    parser.add_argument('-n', '--model', help='Model Name', required=True)
    parser.add_argument('-o', '--operation', help='Operation', default='train')
    parser.add_argument('-ds', '--dataset_path', help='Dataset Path', required=True)
    parser.add_argument('-bs', '--batch_size', help='Batch Size', default=DEFAULT_BATCH_SIZE)
    parser.add_argument('-ne', '--no_epochs', help='Number of epochs', default=DEFAULT_NO_EPOCHS)
    parser.add_argument('-lr', '--learning_rate', help='Leaning Rate', default=DEFAULT_LEARNING_RATE)

    return parser.parse_args()


if __name__ == '__main__':
    working_folder = os.getenv('WORK_DIR', './')

    # configure logging
    setup_logging(working_folder, 'learning-logging.yaml')
    logger = logging.getLogger(__name__)

    args = get_cmd_args()

    dataset_path = args.dataset_path
    kwargs = {
        'BatchSize': args.batch_size,
        'NoEpochs': args.no_epochs,
        'LearningRate': args.learning_rate
    }

    models = {
        'opticalflow': get_opticalflow_model,
        'rgb': get_rgb_model,
    }
    model = models[args.model]()

    operations = {
        'train': model.train,
        'evaluate': model.evaluate,
        'predict': model.predict,
    }
    operations[args.operation]()
