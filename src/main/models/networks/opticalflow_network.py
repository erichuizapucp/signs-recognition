import pathlib
from networks.base_network import BaseNetwork


class OpticalFlowNetwork(BaseNetwork):
    def __init__(self, dataset_name, **kwargs):
        super().__init__(dataset_name, **kwargs),

    def get_class_names(self):
        pass
        # self.data_folder.glob("*").

    def train_network(self):
        print('OpticalFlow network train')
