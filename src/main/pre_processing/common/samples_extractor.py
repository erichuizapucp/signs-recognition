import abc


class SamplesExtractor:
    @abc.abstractmethod
    def extract_sample(self, **kwargs):
        raise NotImplementedError('extract_sample is not implemented.')
