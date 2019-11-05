import os

from models.networks.opticalflow_network import OpticalFlowNetwork
from models.networks.rgb_network import RGBNetwork
from argparse import ArgumentParser

optical_flow_network_name = 'opticalflow'
rgb_network_name = 'rgb'


def get_opticalflow_network(x):
    dataset_path = os.path.join(base_dataset_path, 'opticalflow')
    nn = OpticalFlowNetwork(dataset_path, **x)
    nn.load_dataset()
    return nn


def get_rgb_network(x):
    dataset_path = os.path.join(base_dataset_path, 'rgb')
    nn = RGBNetwork(dataset_path, **x)
    nn.load_dataset()
    return nn


if __name__ == '__main__':
    working_folder = os.getenv('WORK_DIR', './')
    base_dataset_path = os.path.join(working_folder, 'dataset')

    parser = ArgumentParser()

    parser.add_argument('-n', '--network', help='Network Name', required=True)
    parser.add_argument('-o', '--operation', help='Operation', default='train')

    parser.add_argument('-bs', '--batch_size', help='Batch Size', default=10)
    parser.add_argument('-nc', '--no_classes', help='Number of classes', default=10)
    parser.add_argument('-ne', '--no_epochs', help='Number of epochs', default=20)

    args = parser.parse_args()

    kwargs = {
        'BatchSize': args.batch_size,
        'NoClasses': args.no_classes,
        'NoEpochs': args.no_epochs,
    }

    networks = {
        'opticalflow': get_opticalflow_network,
        'rgb': get_rgb_network,
    }
    network = networks[args.network](kwargs)

    operations = {
        'train': network.train_network(),
        'evaluate': network.evaluate_network(),
        'predict': network.predict(),
    }
