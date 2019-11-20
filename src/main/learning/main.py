import logging
import os

from learning.models.opticalflow_model import OpticalFlowModel
from learning.models.rgb_model import RGBModel
from argparse import ArgumentParser

from logger_config import setup_logging

optical_flow_network_name = 'opticalflow'
rgb_network_name = 'rgb'


def get_opticalflow_model():
    dataset_path = os.path.join(base_dataset_path, 'opticalflow')
    nn = OpticalFlowModel(dataset_path, **kwargs)
    nn.configure()
    return nn


def get_rgb_model():
    dataset_path = os.path.join(base_dataset_path, 'rgb')
    nn = RGBModel(dataset_path, **kwargs)
    nn.configure()
    return nn


def get_cmd_args():
    parser = ArgumentParser()

    parser.add_argument('-n', '--model', help='Model Name', required=True)
    parser.add_argument('-o', '--operation', help='Operation', default='train')

    parser.add_argument('-bs', '--batch_size', help='Batch Size', default=10)
    parser.add_argument('-nc', '--no_classes', help='Number of classes', default=10)
    parser.add_argument('-ne', '--no_epochs', help='Number of epochs', default=20)

    return parser.parse_args()


if __name__ == '__main__':
    working_folder = os.getenv('WORK_DIR', './')

    # configure logging
    setup_logging(working_folder, 'learning-logging.yaml')
    logger = logging.getLogger(__name__)

    base_dataset_path = os.path.join(working_folder, 'dataset')
    args = get_cmd_args()

    kwargs = {
        'BatchSize': args.batch_size,
        'NoClasses': args.no_classes,
        'NoEpochs': args.no_epochs,
    }

    models = {
        'opticalflow': get_opticalflow_model,
        'rgb': get_rgb_model,
    }
    model = models[args.model]()

    operations = {
        'train': model.train_network,
        'evaluate': model.evaluate_network,
        'predict': model.predict,
    }
    operations[args.operation]()
