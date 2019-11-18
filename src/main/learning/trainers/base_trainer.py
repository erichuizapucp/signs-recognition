from tensorflow.keras.models import Model


class BaseTrainer:
    def __init__(self, model: Model, **kwargs):
        self._model = model
        self._epochs = kwargs['NoEpochs']
        self._learning_rate = kwargs['LearningRate']

    def train(self):
        pass
