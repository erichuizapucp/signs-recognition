from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model


class BaseModelConfigurer:
    def __init__(self, model: Model):
        self.model = model

    def configure(self):
        self.model.compile(optimizer=SGD)

    def get_optimizer(self):
        return SGD()

    def get_loss_function(self):
        pass

    def get_metrics(self):
        pass
