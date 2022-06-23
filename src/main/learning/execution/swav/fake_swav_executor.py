import numpy as np
import tensorflow as tf
import random

from learning.execution.base_model_executor import BaseModelExecutor
from learning.execution.swav.swav_callback import SwAVCallback
from learning.common.model_utility import SWAV


class FakeSwAVExecutor(BaseModelExecutor):
    def __init__(self):
        super().__init__()

        # 224x224, 224x224, 96x96, 96x96, 96x96
        self.num_crops = [2, 3]

        # swapped assigment is intended only for 224x224 crops
        self.crops_for_assign = [0, 1]
        self.crop_sizes = [224, 224, 96, 96, 96]
        self.temperature = 0.1

        # To be assigned on the train_model method
        self.optimizer = None
        self.feature_backbone_model = None
        self.prototype_projection_model = None

        self.loss = self._get_loss()

        self.training_logger.info('SwAV initialized with: num_crops: %s, crops_for_assign: %s, '
                                  'temperature: %s', self.num_crops, self.crops_for_assign, self.temperature)

    def train_model(self, models, dataset, no_epochs, no_steps_per_epoch=None, **kwargs):
        def train(train_step_fn, optimizer, callback):
            try:
                self.feature_backbone_model = models[0]
                self.prototype_projection_model = models[1]
                self.optimizer = optimizer

                step_wise_loss = []
                epoch_wise_loss = []

                self.training_logger.info('SwAV Training started with: no_epochs: %s', no_epochs)

                callback.on_train_begin()
                for epoch in range(no_epochs):
                    callback.on_epoch_begin(epoch=epoch)

                    for i, inputs in enumerate(reversed(range(100))):
                        loss = train_step_fn(inputs)
                        step_wise_loss.append(loss)

                        self.training_logger.debug('training step: {}, step_loss: {:.3f};'.format(i + 1, loss))

                    epoch_loss = np.mean(step_wise_loss)
                    epoch_wise_loss.append(epoch_loss)

                    self.training_logger.info('epoch: {}, epoch_loss: {:.3f};'.format(epoch + 1, epoch_loss))

                    callback.on_epoch_end(epoch, logs={'loss': epoch_loss})

                self.training_logger.info('SwAV Training is completed.')
            except Exception as e:
                self.training_logger.error(e)
                raise RuntimeError(e)

        return train

    def train_step(self, global_batch_size, per_replica_batch_size):
        @tf.function
        def step(input_value):
            return float(input_value / 25)

        return step

    def _get_model_type(self):
        return SWAV

    @staticmethod
    def _get_loss():
        return tf.keras.losses.CategoricalCrossentropy(axis=1, reduction=tf.keras.losses.Reduction.NONE)

    def get_callback(self, checkpoint_storage_path, model):
        callback = SwAVCallback(model[0], model[1], checkpoint_storage_path)
        return callback

    def configure(self, models):
        print('SwAV does not require a configure method.')
