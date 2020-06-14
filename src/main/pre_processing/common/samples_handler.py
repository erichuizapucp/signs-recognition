import abc


class SamplesHandler:
    @abc.abstractmethod
    def handle_sample(self, **kwargs):
        pass
