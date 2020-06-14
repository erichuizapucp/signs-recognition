import abc


class SamplesExtractor:
    @abc.abstractmethod
    def extract_sample(self, **kwargs):
        pass
