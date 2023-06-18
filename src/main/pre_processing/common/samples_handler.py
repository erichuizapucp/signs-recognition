import os
import abc


class SamplesHandler:
    def __init__(self):
        self.work_dir = os.getenv('WORK_DIR', './')

    @abc.abstractmethod
    def handle_sample(self, **kwargs):
        pass
