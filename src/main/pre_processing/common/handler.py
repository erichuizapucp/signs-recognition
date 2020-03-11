import abc


class Handler:
    @abc.abstractmethod
    def handle(self, **kwargs):
        pass
