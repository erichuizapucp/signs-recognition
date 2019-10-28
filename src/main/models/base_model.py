from keras.models import Sequential


class BaseModel:
    __default_batch_size = 10
    __default_no_classes = 5
    __default_epochs = 20

    def __init__(self, **kwargs):
        self._batch_size = kwargs['BatchSize'] or self.__default_batch_size
        self._no_classes = kwargs['NoClasses'] or self.__default_no_classes
        self._epochs = kwargs['Epochs'] or self.__default_epochs

        model = Sequential()
        model.compile(optimizer='')

    def get_optimizer(self):
        pass

    def get_loss_function(self):
        pass



    def train(self):
        pass

    def test(self):
        pass
